"""Generate raw AgentBench candidate prediction lines.

This script creates ``results/agentbench/${dataset}_${tag}_predictions.jsonl``.
It generates candidate attempts only; official environment benchmarks still need
``score``/``is_correct`` from their harness or an evaluator command at selection
time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acdan.agentbench.adapters import AgentBenchTask, read_tasks


def _prompt(task: AgentBenchTask) -> str:
    if task.dataset == "swe_bench_verified":
        return (
            "You are solving a SWE-Bench Verified issue. Produce a concise patch plan "
            "and, if possible, a unified diff.\n\n"
            f"Repository task:\n{task.instruction}\n\n"
            "Final answer should include the patch or the exact files/functions to change."
        )
    if task.dataset == "webvoyager":
        return (
            "You are a web-navigation agent. Describe the browsing steps needed and "
            "finish with the final answer.\n\n"
            f"Task:\n{task.instruction}\n\nFinal answer:"
        )
    if task.dataset == "tau2_bench":
        return (
            "You are a stateful tool-using customer-support agent. Propose a complete "
            "trajectory and final resolution.\n\n"
            f"Task:\n{task.instruction}\n\nFinal answer:"
        )
    if task.dataset == "mcp_bench":
        return (
            "You are an MCP tool-use agent. Propose the tool trajectory and final "
            "task result.\n\n"
            f"Task:\n{task.instruction}\n\nFinal answer:"
        )
    return (
        "Answer the task. Reason briefly if needed, then finish with a final answer.\n\n"
        f"Task:\n{task.instruction}\n\nFinal answer:"
    )


def _mock_predictions(tasks: Iterable[AgentBenchTask], k: int) -> Iterable[dict]:
    for task in tasks:
        for i in range(k):
            answer = str(task.gold) if task.gold is not None and i == 0 else f"mock_attempt_{i}"
            row = {
                "task_id": task.task_id,
                "candidate_id": str(i),
                "final_answer": answer,
                "trajectory": [{"role": "assistant", "content": answer}],
                "generator": "mock",
            }
            if task.evaluator == "semantic_qa" and task.gold is not None:
                row["is_correct"] = i == 0
            yield row


def _vllm_predictions(
    tasks: list[AgentBenchTask],
    model: str,
    k: int,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> Iterable[dict]:
    from vllm import LLM, SamplingParams  # lazy GPU import

    llm = LLM(model=model, seed=seed)
    prompts = [_prompt(task) for task in tasks]
    sp = SamplingParams(n=k, temperature=temperature, max_tokens=max_tokens)
    outs = llm.generate(prompts, sp)
    for task, out in zip(tasks, outs):
        for i, sample in enumerate(out.outputs):
            text = sample.text.strip()
            yield {
                "task_id": task.task_id,
                "candidate_id": str(i),
                "final_answer": text,
                "trajectory": [
                    {"role": "user", "content": task.instruction},
                    {"role": "assistant", "content": text},
                ],
                "generator": "vllm",
                "model": model,
            }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate AgentBench prediction JSONL.")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["mock", "vllm"], default="vllm")
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    tasks = read_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    if args.backend == "vllm" and not args.model:
        raise ValueError("--backend vllm requires --model")

    rows = (
        _mock_predictions(tasks, args.k)
        if args.backend == "mock"
        else _vllm_predictions(tasks, args.model, args.k, args.temperature, args.max_tokens, args.seed)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} predictions -> {out}")


if __name__ == "__main__":
    main()
