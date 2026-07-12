"""Evaluate ACDAN self-choice on AgentBench candidate trajectories.

Input candidate JSONL schema:

{
  "task": { ... AgentBenchTask fields ... },
  "candidates": [
    {"candidate_id": "0", "final_answer": "...", "trajectory": [...],
     "patch": "...", "is_correct": true}
  ]
}

For official environment benchmarks, candidates should either include
``is_correct``/``score`` from the official harness, or pass an evaluator command:

  --evaluator-command external_swe_bench="python eval_swe.py --in {input} --out {output}"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acdan.agent import ACDANAgent
from acdan.agentbench.adapters import AgentBenchTask
from acdan.agentbench.evaluators import Candidate, CandidateEvaluator, EXTERNAL_EVALUATORS
from acdan.agentbench.metrics import summarize_selection
from acdan.baselines import (
    adaptive_self_consistency,
    best_of_n_prm,
    cot_greedy,
    self_consistency,
)
from acdan.config import ACDANConfig, AblationFlags
from acdan.datasets.base import RawTask
from acdan.latent_reasoning import LatentReasoner
from acdan.run_experiment import _build_core, _build_encoder, _build_prm, _to_task
from acdan.verification import SelfVerifier


def _parse_external(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--evaluator-command must be name=command")
        key, cmd = value.split("=", 1)
        out[key.strip()] = cmd.strip()
    return out


def _task_from_dict(d: dict[str, Any]) -> AgentBenchTask:
    return AgentBenchTask(
        task_id=str(d["task_id"]),
        dataset=str(d["dataset"]),
        domain=str(d.get("domain", "unknown")),
        instruction=str(d["instruction"]),
        evaluator=str(d.get("evaluator", "external")),
        gold=d.get("gold"),
        metadata=dict(d.get("metadata", {}) or {}),
    )


def _load_candidate_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _candidate_is_scored(candidate: Candidate) -> bool:
    return "score" in candidate.raw or "is_correct" in candidate.raw


def _has_external_command(evaluator: str, external_commands: dict[str, str]) -> bool:
    env_key = "ACDAN_" + evaluator.upper() + "_CMD"
    return bool(external_commands.get(evaluator) or os.environ.get(env_key))


def _preflight_evaluation(
    rows: list[dict[str, Any]],
    external_commands: dict[str, str],
    allow_unevaluated: bool,
) -> None:
    for line_no, row in enumerate(rows, start=1):
        task = _task_from_dict(row["task"])
        if task.evaluator not in EXTERNAL_EVALUATORS:
            continue
        candidates = [Candidate.from_obj(c, i) for i, c in enumerate(row["candidates"])]
        if not candidates or all(_candidate_is_scored(c) for c in candidates):
            continue
        if allow_unevaluated or _has_external_command(task.evaluator, external_commands):
            continue
        env_key = "ACDAN_" + task.evaluator.upper() + "_CMD"
        raise RuntimeError(
            f"{task.dataset} row {line_no} uses evaluator '{task.evaluator}', but at least "
            "one candidate has no score/is_correct. Provide official scores, pass "
            f"--evaluator-command {task.evaluator}=<cmd>, set {env_key}, or add "
            "--allow-unevaluated for a smoke run only. Unscored external-evaluator "
            "runs are not reportable metrics."
        )


def _text_preview(text: str, limit: int, label: str) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    marker = f"\n\n[truncated middle of {label} for selector scoring]\n\n"
    available = max(2, limit - len(marker))
    head = available // 2
    tail = available - head
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def _raw_task(
    task: AgentBenchTask,
    candidates: list[Candidate],
    candidate_preview_chars: int = 2048,
    task_preview_chars: int = 4096,
) -> RawTask:
    labels = tuple(c.candidate_id for c in candidates)
    templates = {
        c.candidate_id: _text_preview(c.display_text(), candidate_preview_chars, "candidate")
        for c in candidates
    }
    prompt = _text_preview(task.instruction, task_preview_chars, "task")
    return RawTask(
        task_id=task.task_id,
        prompt=prompt,
        vocab=labels,
        horizon=1,
        gold=None,
        family=f"agentbench:{task.dataset}",
        difficulty=float(task.metadata.get("difficulty", 0.7) or 0.7),
        action_templates=templates,
        metadata={
            "agentbench_task": task.to_json(),
            "candidate_payloads": [c.raw for c in candidates],
        },
    )


def _select(
    method: str,
    core,
    prm,
    reasoner,
    verifier,
    config,
    task,
    latent,
    n: int,
    asc_threshold: float,
    seed: int,
):
    if method == "acdan":
        agent = ACDANAgent(config, core, prm, reasoner, verifier, outcome_checker=lambda _t, _a: True)
        result = agent.run_task(task)
        return (
            [step.action_id for step in result.steps],
            dataclasses.asdict(result.metrics),
            float(result.verification.confidence),
            bool(result.verification.abstained),
        )
    if method == "cot":
        br = cot_greedy(core, task, latent)
    elif method == "sc":
        br = self_consistency(core, task, latent, n=n, seed=seed)
    elif method == "asc":
        br = adaptive_self_consistency(core, task, latent, n=n, threshold=asc_threshold, seed=seed)
    elif method == "bon":
        br = best_of_n_prm(core, prm, task, latent, n=n, seed=seed)
    else:
        raise KeyError(method)
    return list(br.actions), dict(br.cost), float(br.confidence), bool(br.abstained)


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_rows = _load_candidate_rows(args.candidates_path, args.limit)
    external_commands = _parse_external(args.evaluator_command)
    if not args.selection_only:
        _preflight_evaluation(candidate_rows, external_commands, args.allow_unevaluated)

    config = ACDANConfig(name=args.method, seed=args.seed)
    if args.no_latent:
        config = dataclasses.replace(
            config,
            ablation=dataclasses.replace(
                config.ablation,
                latent_reasoning=False,
                in_place_ttt=False,
            ),
        )
    elif args.no_ttt:
        config = dataclasses.replace(
            config,
            ablation=dataclasses.replace(config.ablation, in_place_ttt=False),
        )

    core = _build_core(args.policy, args.policy_model, args)
    encoder = _build_encoder(args, core)
    prm = _build_prm(args.prm, args.prm_model, core, args.seed)
    reasoner = LatentReasoner(config.latent, feature_dim=encoder.dim, seed=args.seed)
    if hasattr(prm, "set_latent_quality_fn"):
        prm.set_latent_quality_fn(reasoner.quality)
    verifier = SelfVerifier(config.verification, probe=None, independent=None)
    evaluator = CandidateEvaluator(
        external_commands=external_commands,
        allow_unevaluated=args.allow_unevaluated,
    )

    rows = []
    t0 = time.perf_counter()
    for idx, row in enumerate(candidate_rows, start=1):
        task_spec = _task_from_dict(row["task"])
        candidates = [Candidate.from_obj(c, i) for i, c in enumerate(row["candidates"])]
        if not candidates:
            continue
        raw = _raw_task(
            task_spec,
            candidates,
            args.candidate_preview_chars,
            args.task_preview_chars,
        )
        task = _to_task(raw, encoder.encode(raw.prompt))
        ab = config.ablation
        trace = (
            reasoner.reason(task.prompt_features, use_ttt=ab.in_place_ttt)
            if ab.latent_reasoning
            else reasoner.passthrough(task.prompt_features)
        )
        task_started = time.perf_counter()
        prompt_tokens_before = int(getattr(core, "n_prompt_tokens", 0))
        actions, cost, confidence, abstained = _select(
            args.method, core, prm, reasoner, verifier, config, task,
            trace.final_state, args.n, args.asc_threshold, args.seed,
        )
        selection_wall_s = time.perf_counter() - task_started
        selection_prompt_tokens = int(getattr(core, "n_prompt_tokens", 0)) - prompt_tokens_before
        selected = int(actions[0]) if actions else 0
        selected = max(0, min(selected, len(candidates) - 1))
        result_row = {
            "task_id": task_spec.task_id,
            "dataset": task_spec.dataset,
            "domain": task_spec.domain,
            "selected": selected,
            "selected_candidate_id": candidates[selected].candidate_id,
            "confidence": confidence,
            "raw_confidence": confidence,
            "abstained": abstained,
            "n_candidates": len(candidates),
            "selection_prompt_tokens": selection_prompt_tokens,
            "selection_wall_s": selection_wall_s,
            "cost": cost,
        }
        if not args.selection_only:
            evals = [evaluator.evaluate(task_spec, c) for c in candidates]
            selected_eval = evals[selected]
            result_row.update({
                "selected_score": float(selected_eval["score"]),
                "selected_correct": bool(selected_eval["correct"]),
                "oracle_score": max(float(e["score"]) for e in evals),
                "pass_at_k": any(bool(e["correct"]) for e in evals),
            })
        rows.append(result_row)
        if args.monitor and not args.selection_only and (idx == 1 or idx % max(1, args.progress_every) == 0):
            acc = np.mean([r["selected_correct"] for r in rows])
            print(f"[agentbench] {idx} tasks selected_acc={acc:.3f}")
        elif args.monitor and (idx == 1 or idx % max(1, args.progress_every) == 0):
            print(f"[agentbench] {idx} tasks selected (blind mode)")

    n = len(rows)
    summary: dict[str, Any] = {
        "method": args.method,
        "policy": args.policy,
        "policy_model": args.policy_model,
        "prm": args.prm,
        "encoder": args.encoder,
        "ablation": dataclasses.asdict(config.ablation),
        "n_tasks": n,
        "selection_only": bool(args.selection_only),
        "mean_candidates": float(np.mean([r["n_candidates"] for r in rows])) if n else 0.0,
        "mean_selection_prompt_tokens": float(np.mean([r["selection_prompt_tokens"] for r in rows])) if n else 0.0,
        "total_selection_prompt_tokens": int(sum(r["selection_prompt_tokens"] for r in rows)),
        "mean_selection_latency_s": float(np.mean([r["selection_wall_s"] for r in rows])) if n else 0.0,
        "mean_samples": float(np.mean([float(r["cost"].get("samples", 1.0)) for r in rows])) if n else 0.0,
        "mean_verified_candidates": float(np.mean([
            float(r["cost"].get("verified_candidates", 0.0)) for r in rows
        ])) if n else 0.0,
        "wall_s": time.perf_counter() - t0,
    }
    if not args.selection_only:
        summary.update(summarize_selection(rows))
    out = {"summary": summary, "per_task": rows if (args.save_per_task or args.selection_only) else []}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run ACDAN self-choice on AgentBench candidates.")
    p.add_argument("--candidates-path", required=True)
    p.add_argument("--method", default="acdan", choices=["acdan", "cot", "sc", "asc", "bon"])
    p.add_argument("--policy", default="mock", choices=["mock", "vllm"])
    p.add_argument("--policy-model", default=None)
    p.add_argument("--vllm-max-model-len", type=int, default=4096)
    p.add_argument("--vllm-dtype", default="bfloat16")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--vllm-hidden-state-layer-ids", type=int, nargs="*", default=None)
    p.add_argument("--vllm-hidden-state-storage-path", default=None)
    p.add_argument("--prm", default="mock", choices=["mock", "llm"])
    p.add_argument("--prm-model", default=None)
    p.add_argument("--encoder", default="hash", choices=["hash", "st", "hf", "vllm_hidden"])
    p.add_argument("--encoder-model", default=None)
    p.add_argument("--encoder-mode", default="last_hidden", choices=["last_hidden", "input_emb"])
    p.add_argument("--encoder-pooling", default="last", choices=["last", "mean"])
    p.add_argument("--encoder-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--encoder-device", default=None)
    p.add_argument("--encoder-max-length", type=int, default=2048)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--n", type=int, default=8)
    p.add_argument(
        "--candidate-preview-chars",
        type=int,
        default=2048,
        help=(
            "Maximum characters from each candidate trajectory shown to the selector. "
            "Use 0 to score full candidates."
        ),
    )
    p.add_argument(
        "--task-preview-chars",
        type=int,
        default=4096,
        help="Maximum task-instruction characters shown to the selector. Use 0 for full tasks.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-latent", action="store_true")
    p.add_argument("--no-ttt", action="store_true")
    p.add_argument("--asc-threshold", type=float, default=0.70)
    p.add_argument("--evaluator-command", action="append", default=[])
    p.add_argument("--allow-unevaluated", action="store_true")
    p.add_argument(
        "--selection-only",
        action="store_true",
        help="Blindly select and save candidate IDs without loading or invoking official scores.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--save-per-task", action="store_true")
    p.add_argument("--monitor", action="store_true")
    p.add_argument("--progress-every", type=int, default=25)
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        run(build_parser().parse_args(argv))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
