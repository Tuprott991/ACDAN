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
from acdan.agentbench.evaluators import Candidate, CandidateEvaluator
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


def _raw_task(task: AgentBenchTask, candidates: list[Candidate]) -> RawTask:
    labels = tuple(c.candidate_id for c in candidates)
    templates = {c.candidate_id: c.display_text() for c in candidates}
    return RawTask(
        task_id=task.task_id,
        prompt=task.instruction,
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


def _select(method: str, core, prm, reasoner, verifier, config, task, latent, n: int, asc_threshold: float):
    if method == "acdan":
        agent = ACDANAgent(config, core, prm, reasoner, verifier, outcome_checker=lambda _t, _a: True)
        result = agent.run_task(task)
        return [step.action_id for step in result.steps], dataclasses.asdict(result.metrics)
    if method == "cot":
        br = cot_greedy(core, task, latent)
    elif method == "sc":
        br = self_consistency(core, task, latent, n=n)
    elif method == "asc":
        br = adaptive_self_consistency(core, task, latent, n=n, threshold=asc_threshold)
    elif method == "bon":
        br = best_of_n_prm(core, prm, task, latent, n=n)
    else:
        raise KeyError(method)
    return list(br.actions), dict(br.cost)


def run(args: argparse.Namespace) -> dict[str, Any]:
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
        external_commands=_parse_external(args.evaluator_command),
        allow_unevaluated=args.allow_unevaluated,
    )

    rows = []
    t0 = time.perf_counter()
    for idx, row in enumerate(_load_candidate_rows(args.candidates_path, args.limit), start=1):
        task_spec = _task_from_dict(row["task"])
        candidates = [Candidate.from_obj(c, i) for i, c in enumerate(row["candidates"])]
        if not candidates:
            continue
        raw = _raw_task(task_spec, candidates)
        task = _to_task(raw, encoder.encode(raw.prompt))
        ab = config.ablation
        trace = (
            reasoner.reason(task.prompt_features, use_ttt=ab.in_place_ttt)
            if ab.latent_reasoning
            else reasoner.passthrough(task.prompt_features)
        )
        actions, cost = _select(
            args.method, core, prm, reasoner, verifier, config, task,
            trace.final_state, args.n, args.asc_threshold,
        )
        selected = int(actions[0]) if actions else 0
        selected = max(0, min(selected, len(candidates) - 1))
        evals = [evaluator.evaluate(task_spec, c) for c in candidates]
        selected_eval = evals[selected]
        oracle = max(float(e["score"]) for e in evals)
        pass_at_k = any(bool(e["correct"]) for e in evals)
        rows.append({
            "task_id": task_spec.task_id,
            "dataset": task_spec.dataset,
            "domain": task_spec.domain,
            "selected": selected,
            "selected_candidate_id": candidates[selected].candidate_id,
            "selected_score": float(selected_eval["score"]),
            "selected_correct": bool(selected_eval["correct"]),
            "oracle_score": oracle,
            "pass_at_k": bool(pass_at_k),
            "n_candidates": len(candidates),
            "cost": cost,
        })
        if args.monitor and (idx == 1 or idx % max(1, args.progress_every) == 0):
            acc = np.mean([r["selected_correct"] for r in rows])
            print(f"[agentbench] {idx} tasks selected_acc={acc:.3f}")

    n = len(rows)
    summary = {
        "method": args.method,
        "policy": args.policy,
        "policy_model": args.policy_model,
        "prm": args.prm,
        "encoder": args.encoder,
        "ablation": dataclasses.asdict(config.ablation),
        "n_tasks": n,
        "selected_accuracy": float(np.mean([r["selected_correct"] for r in rows])) if n else 0.0,
        "selected_score": float(np.mean([r["selected_score"] for r in rows])) if n else 0.0,
        "oracle_score": float(np.mean([r["oracle_score"] for r in rows])) if n else 0.0,
        "pass_at_k": float(np.mean([r["pass_at_k"] for r in rows])) if n else 0.0,
        "verification_gap": (
            float(np.mean([r["pass_at_k"] for r in rows]))
            - float(np.mean([r["selected_correct"] for r in rows]))
            if n else 0.0
        ),
        "mean_candidates": float(np.mean([r["n_candidates"] for r in rows])) if n else 0.0,
        "wall_s": time.perf_counter() - t0,
    }
    out = {"summary": summary, "per_task": rows if args.save_per_task else []}
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-latent", action="store_true")
    p.add_argument("--no-ttt", action="store_true")
    p.add_argument("--asc-threshold", type=float, default=0.70)
    p.add_argument("--evaluator-command", action="append", default=[])
    p.add_argument("--allow-unevaluated", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--save-per-task", action="store_true")
    p.add_argument("--monitor", action="store_true")
    p.add_argument("--progress-every", type=int, default=25)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
