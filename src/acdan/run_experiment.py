"""Unified real-experiment runner (offline-safe).

One entrypoint drives every method × model × benchmark combination used in the
paper. Heavy backends (vLLM/PRM/Claude) load only when selected; with the
defaults (`--policy mock --prm mock --dataset synthetic`) it runs on CPU with no
extra dependencies, so the experiment scripts are testable before the VM.

    python -m acdan.run_experiment --method acdan --dataset synthetic --limit 24 \
        --out results/smoke.json

See ``experiments/PLAN.md`` for the full command matrix and duration estimates.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from acdan.agent import ACDANAgent
from acdan.baselines import (
    adaptive_self_consistency,
    best_of_n_prm,
    cot_greedy,
    self_consistency,
)
from acdan.config import ACDANConfig, AblationFlags
from acdan.datasets.base import RawTask, build_dataset, build_outcome_checker
from acdan.inertia import InertialSensor
from acdan.latent_reasoning import LatentReasoner
from acdan.metrics import expected_calibration_error
from acdan.types import Task
from acdan.verification import SelfVerifier


# Friendly ablation names -> AblationFlags field names.
_ABLATION_ALIAS = {
    "no_dto": "dto",
    "no_graph": "dependency_graph",
    "no_inertia": "inertial_sensing",
    "no_verification": "verification",
    "no_latent": "latent_reasoning",
    "no_ttt": "in_place_ttt",
    "no_confidence_margin": "confidence_margin",
}


# --------------------------------------------------------------------- build

def _build_core(name: str, model: Optional[str], seed: int):
    if name == "mock":
        from acdan.registry import build_core_model
        return build_core_model("mock", seed=seed)
    if name == "vllm":
        from acdan.backends.vllm_core import VLLMCoreModel
        return VLLMCoreModel(model_name=model, seed=seed)
    raise KeyError(f"unknown policy '{name}'")


def _build_prm(name: str, model: Optional[str], core, seed: int):
    if name == "mock":
        from acdan.registry import build_prm
        return build_prm("mock", seed=seed)
    if name == "llm":
        from acdan.backends.prm_adapter import LLMAsProcessReward
        # reuse the policy LLM when it is the vLLM core; else load `model`.
        return LLMAsProcessReward(core=core if hasattr(core, "llm") else None,
                                  model_name=model if not hasattr(core, "llm") else None)
    raise KeyError(f"unknown prm '{name}'")


def _build_encoder(name: str):
    from acdan.backends.encoder import build_encoder
    return build_encoder("hash" if name == "hash" else "st")


def _to_task(raw: RawTask, features: np.ndarray) -> Task:
    """Convert a RawTask to a frozen Task; derive optimal_plan for mock signal."""
    optimal: Optional[Tuple[int, ...]] = None
    vocab = raw.vocab
    if isinstance(raw.gold, str) and raw.gold in vocab:
        optimal = tuple([vocab.index(raw.gold)] * raw.horizon)
    elif isinstance(raw.gold, (list, tuple)):
        idx = [vocab.index(g) for g in raw.gold if g in vocab]
        if idx:
            optimal = tuple(idx)
    metadata = dict(raw.metadata)
    metadata.update({
        "prompt": raw.prompt,
        "gold": raw.gold,
        "family": raw.family,
        "action_templates": raw.action_templates,
    })
    return Task(
        task_id=raw.task_id, prompt_features=np.asarray(features, dtype=np.float64),
        vocab=tuple(vocab), horizon=raw.horizon, difficulty=raw.difficulty,
        optimal_plan=optimal,
        metadata=metadata,
    )


def _fit_inertia(tasks: List[Task], config: ACDANConfig) -> Dict[str, InertialSensor]:
    """Fit per-family transition models from gold/optimal sequences (tool tasks)."""
    by_family: Dict[str, List[Task]] = {}
    for t in tasks:
        by_family.setdefault(str(t.metadata.get("family", "")), []).append(t)
    sensors: Dict[str, InertialSensor] = {}
    for fam, fts in by_family.items():
        vsizes = {t.vocab_size for t in fts}
        if len(vsizes) != 1:
            continue  # shared vocab required for index-based transitions
        traces = [list(t.optimal_plan) for t in fts if t.optimal_plan]
        if not traces:
            continue
        sensor = InertialSensor(config.inertia, fts[0].vocab_size).fit(traces)
        sensors[fam] = sensor
    return sensors


def _dataset_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {}
    if args.dataset in {"gsm8k", "math"}:
        evidence = args.math_evidence
        kwargs.update({
            "include_candidate_counts": evidence in {"prompt", "all"},
            "include_candidate_reasoning": not args.no_candidate_reasoning,
            "use_prm_count_bonus": evidence in {"prm", "all"},
        })
    return kwargs


# --------------------------------------------------------------------- run

def run(args: argparse.Namespace) -> dict:
    config = ACDANConfig(name=args.method, seed=args.seed)
    if args.disable:
        flags = dataclasses.asdict(AblationFlags())
        for k in args.disable.split(","):
            k = k.strip()
            if not k:
                continue
            field = _ABLATION_ALIAS.get(k, k)
            if field not in flags:
                raise KeyError(
                    f"unknown ablation '{k}'. Use a flag field {list(flags)} "
                    f"or an alias {list(_ABLATION_ALIAS)}."
                )
            flags[field] = False
        config = dataclasses.replace(config, ablation=AblationFlags(**flags))

    encoder = _build_encoder(args.encoder)
    core = _build_core(args.policy, args.policy_model, args.seed)
    prm = _build_prm(args.prm, args.prm_model, core, args.seed)
    reasoner = LatentReasoner(config.latent, feature_dim=encoder.dim, seed=args.seed)

    if args.dataset in {"gsm8k", "math"}:
        dto_sc = args.math_count_weight if args.math_evidence in {"dto", "all"} else 0.0
        config = dataclasses.replace(
            config,
            dto=dataclasses.replace(config.dto, self_consistency_weight=dto_sc),
        )

    dataset = build_dataset(
        args.dataset, path=args.data_path, limit=args.limit, **_dataset_kwargs(args)
    )
    checker = build_outcome_checker(args.dataset)
    tasks = [_to_task(r, encoder.encode(r.prompt)) for r in dataset.tasks()]

    sensors = {}
    if args.method == "acdan" and args.fit_inertia:
        if args.inertia_fit_path:
            fit_dataset = build_dataset(
                args.dataset,
                path=args.inertia_fit_path,
                limit=args.inertia_fit_limit,
                **_dataset_kwargs(args),
            )
            fit_tasks = [_to_task(r, encoder.encode(r.prompt)) for r in fit_dataset.tasks()]
            sensors = _fit_inertia(fit_tasks, config)
        elif args.dataset == "synthetic":
            sensors = _fit_inertia(tasks, config)
        else:
            print(
                "warning: --fit-inertia ignored without --inertia-fit-path; "
                "not fitting inertia on evaluation tasks."
            )

    # Independent verifier backend
    independent = None
    if args.verifier == "claude":
        from acdan.backends.claude import ClaudeIndependentVerifier
        independent = ClaudeIndependentVerifier(model=args.judge_model)
    verifier = SelfVerifier(config.verification, probe=None, independent=independent)

    t0 = time.perf_counter()
    rows: List[dict] = []
    base_tokens = getattr(core, "n_prompt_tokens", 0) + getattr(prm, "n_prompt_tokens", 0)
    for task in tasks:
        ts = time.perf_counter()
        if args.method == "acdan":
            agent = ACDANAgent(config, core, prm, reasoner, verifier, sensors,
                               outcome_checker=checker)
            res = agent.run_task(task)
            correct = res.metrics.correct
            conf = res.verification.confidence
            abst = res.metrics.abstained
            tok_surrogate = res.metrics.token_cost
            cost_info = {
                "policy_passes": 1,
                "prm_passes": 1,
                "samples": 1,
                "verified_candidates": task.horizon * task.vocab_size,
            }
        else:
            ab = config.ablation
            trace = (reasoner.reason(task.prompt_features, use_ttt=ab.in_place_ttt)
                     if ab.latent_reasoning else reasoner.passthrough(task.prompt_features))
            latent = trace.final_state
            if args.method == "cot":
                br = cot_greedy(core, task, latent)
            elif args.method == "sc":
                br = self_consistency(core, task, latent, n=args.n, seed=args.seed)
            elif args.method == "asc":
                br = adaptive_self_consistency(
                    core,
                    task,
                    latent,
                    n=args.n,
                    threshold=args.asc_threshold,
                    min_samples=args.asc_min_samples,
                    seed=args.seed,
                )
            elif args.method == "bon":
                br = best_of_n_prm(core, prm, task, latent, n=args.n, seed=args.seed)
            else:
                raise KeyError(f"unknown method '{args.method}'")
            correct = bool(checker(task, br.actions))
            conf, abst = 1.0, False
            cost_info = dict(br.cost)
            tok_surrogate = float(task.horizon) * float(cost_info.get("samples", 1.0))
        cur_tokens = getattr(core, "n_prompt_tokens", 0) + getattr(prm, "n_prompt_tokens", 0)
        rows.append({
            "task_id": task.task_id, "family": task.metadata.get("family"),
            "correct": bool(correct), "confidence": float(conf), "abstained": bool(abst),
            "latency_s": time.perf_counter() - ts,
            "real_prompt_tokens": cur_tokens - base_tokens,
            "token_surrogate": tok_surrogate,
            "policy_passes": float(cost_info.get("policy_passes", 0.0)),
            "prm_passes": float(cost_info.get("prm_passes", 0.0)),
            "samples": float(cost_info.get("samples", 1.0)),
            "verified_candidates": float(cost_info.get("verified_candidates", 0.0)),
        })
        base_tokens = cur_tokens

    wall = time.perf_counter() - t0
    n = len(rows)
    correct = np.array([r["correct"] for r in rows], dtype=bool)
    answered = ~np.array([r["abstained"] for r in rows], dtype=bool)
    answer_tasks = [t for t in tasks if t.horizon == 1 and t.vocab_size > 0]
    oracle_acc = None
    first_acc = None
    last_acc = None
    highest_count_acc = None
    if args.dataset in {"gsm8k", "math", "jsonl"} and answer_tasks:
        oracle_acc = float(np.mean([
            any(bool(checker(t, [i])) for i in range(t.vocab_size))
            for t in answer_tasks
        ]))
        first_acc = float(np.mean([bool(checker(t, [0])) for t in answer_tasks]))
        last_acc = float(np.mean([
            bool(checker(t, [t.vocab_size - 1])) for t in answer_tasks
        ]))
        count_hits = []
        for t in answer_tasks:
            counts = t.metadata.get("candidate_counts", {}) or {}
            if counts:
                best = max(
                    range(t.vocab_size),
                    key=lambda i: (float(counts.get(str(t.vocab[i]), 1.0)), -i),
                )
            else:
                best = 0
            count_hits.append(bool(checker(t, [best])))
        highest_count_acc = float(np.mean(count_hits))
    summary = {
        "method": args.method, "dataset": args.dataset, "policy": args.policy,
        "policy_model": args.policy_model, "prm": args.prm, "seed": args.seed,
        "math_evidence": args.math_evidence if args.dataset in {"gsm8k", "math"} else None,
        "dto_self_consistency_weight": config.dto.self_consistency_weight,
        "inertia_fit_path": args.inertia_fit_path,
        "n_tasks": n,
        "accuracy": float(correct.mean()) if n else 0.0,
        "coverage": float(answered.mean()) if n else 0.0,
        "selective_accuracy": float(correct[answered].mean()) if answered.any() else 0.0,
        "ece": expected_calibration_error([r["confidence"] for r in rows],
                                          [r["correct"] for r in rows]) if n else 0.0,
        "mean_latency_s": float(np.mean([r["latency_s"] for r in rows])) if n else 0.0,
        "wall_s": wall,
        "mean_real_prompt_tokens": float(np.mean([r["real_prompt_tokens"] for r in rows])) if n else 0.0,
        "total_real_prompt_tokens": int(sum(r["real_prompt_tokens"] for r in rows)),
        "mean_token_surrogate": float(np.mean([r["token_surrogate"] for r in rows])) if n else 0.0,
        "mean_samples": float(np.mean([r["samples"] for r in rows])) if n else 0.0,
        "mean_verified_candidates": float(np.mean([r["verified_candidates"] for r in rows])) if n else 0.0,
        "mean_prm_passes": float(np.mean([r["prm_passes"] for r in rows])) if n else 0.0,
    }
    if answer_tasks:
        summary["mean_vocab_size"] = float(np.mean([t.vocab_size for t in answer_tasks]))
    if oracle_acc is not None:
        summary["oracle_candidate_accuracy"] = oracle_acc
        summary["always_first_accuracy"] = first_acc
        summary["always_last_accuracy"] = last_acc
        summary["highest_count_accuracy"] = highest_count_acc
    out = {"summary": summary, "per_task": rows if args.save_per_task else []}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="acdan-run", description="ACDAN real experiment runner.")
    p.add_argument("--method", default="acdan",
                   choices=["acdan", "cot", "sc", "asc", "bon"], help="ACDAN or a baseline.")
    p.add_argument("--disable", default="", help="Comma list of ablation flags to disable.")
    p.add_argument("--dataset", default="synthetic",
                   choices=["synthetic", "jsonl", "gsm8k", "math", "bfcl", "toolbench", "gaia"])
    p.add_argument("--data-path", default=None, help="Local JSONL path for real datasets.")
    p.add_argument("--limit", type=int, default=None, help="Max tasks.")
    p.add_argument("--policy", default="mock", choices=["mock", "vllm"])
    p.add_argument("--policy-model", default=None, help="HF id for vLLM policy.")
    p.add_argument("--prm", default="mock", choices=["mock", "llm"])
    p.add_argument("--prm-model", default=None, help="HF id for a dedicated PRM.")
    p.add_argument("--encoder", default="hash", choices=["hash", "st"])
    p.add_argument("--verifier", default="none", choices=["none", "claude"])
    p.add_argument("--judge-model", default="claude-opus-4-8")
    p.add_argument("--n", type=int, default=8, help="N for sc/bon.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--math-evidence",
        default="none",
        choices=["none", "prompt", "prm", "dto", "all"],
        help="How self-consistency candidate counts may influence math runs.",
    )
    p.add_argument("--math-count-weight", type=float, default=0.35)
    p.add_argument("--asc-threshold", type=float, default=0.70)
    p.add_argument("--asc-min-samples", type=int, default=2)
    p.add_argument("--no-candidate-reasoning", action="store_true")
    p.add_argument("--fit-inertia", action="store_true", help="Fit inertia from a separate split.")
    p.add_argument("--inertia-fit-path", default=None, help="Train/dev JSONL for fitting inertia.")
    p.add_argument("--inertia-fit-limit", type=int, default=None)
    p.add_argument("--save-per-task", action="store_true")
    p.add_argument("--out", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
