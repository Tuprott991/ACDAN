"""Command-line interface for ACDAN offline experiments.

Commands:
  acdan demo        Run one task and print the full module-by-module trace.
  acdan eval        Evaluate a config over the synthetic suite; print/save summary.
  acdan ablation    Run full ACDAN + per-module ablations; print a comparison.
  acdan config-dump Write the default config to a YAML file.

Everything runs offline. Use --seed / --n for reproducible variations.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import List, Optional

from acdan.config import ACDANConfig, AblationFlags, baseline_cot_config
from acdan.evaluate import build_agent, run_evaluation
from acdan.metrics import format_summary
from acdan.tasks.synthetic import make_suite


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------

def _cmd_demo(args: argparse.Namespace) -> int:
    config = _load_config(args.config, args.seed)
    agent = build_agent(config, n_per_family=args.n)
    tasks = make_suite(n_per_family=args.n, seed=config.seed)
    task = tasks[args.task_index % len(tasks)]
    result = agent.run_task(task)

    print(f"=== ACDAN demo | config={config.name} | {config.ablation.describe()} ===")
    print(f"task         : {task.task_id}  (family={task.metadata.get('family')}, "
          f"difficulty={task.difficulty:.2f})")
    print(f"vocab        : {list(task.vocab)}")
    if task.optimal_plan is not None:
        print(f"optimal plan : {[task.vocab[a] for a in task.optimal_plan]}")
    print(f"DTO steps    : {result.plan.dto_steps}")
    if result.plan.objective_trace:
        first, last = result.plan.objective_trace[0], result.plan.objective_trace[-1]
        print(f"DTO objective: {first:.4f} -> {last:.4f} (lower is better)")
    print("\nstep | action      | from_inertia | dead | prm   | nig    | conf")
    print("-" * 64)
    for s in result.steps:
        print(f"{s.index:>4} | {s.action_name:<11} | {str(s.from_inertia):<12} | "
              f"{str(s.is_dead_step):<4} | {s.prm_score:.3f} | {s.nig:+.3f} | {s.confidence:.3f}")

    v = result.verification
    m = result.metrics
    print("\n--- verification ---")
    print(f"confidence={v.confidence:.3f}  margin={v.margin:.3f}  "
          f"independent={v.independent_agreement:.3f}  verified={v.verified}  abstained={v.abstained}")
    print("\n--- metrics ---")
    print(f"correct={m.correct}  token_cost={m.token_cost:.2f}  llm_calls={m.llm_calls}  "
          f"inertia_saved={m.inertia_saved_calls}  dead_pruned={m.dead_steps_pruned}  "
          f"dep_entropy={m.dependency_entropy:.3f}")
    return 0


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------

def _cmd_eval(args: argparse.Namespace) -> int:
    config = _load_config(args.config, args.seed)
    summary = run_evaluation(config, n_per_family=args.n)
    print(format_summary(summary))
    if args.out:
        _save_json(args.out, summary.to_dict())
        print(f"\n[saved] {args.out}")
    return 0


# --------------------------------------------------------------------------
# ablation
# --------------------------------------------------------------------------

def _ablation_configs(base: ACDANConfig) -> List[ACDANConfig]:
    """Full ACDAN, baseline CoT, and one config per disabled module."""
    configs: List[ACDANConfig] = []

    full = dataclasses.replace(base, name="acdan_full", ablation=AblationFlags())
    configs.append(full)

    module_flags = [
        ("no_latent", {"latent_reasoning": False, "in_place_ttt": False}),
        ("no_ttt", {"in_place_ttt": False}),
        ("no_dto", {"dto": False}),
        ("no_graph", {"dependency_graph": False}),
        ("no_inertia", {"inertial_sensing": False}),
        ("no_verification", {"verification": False}),
        ("no_confidence_margin", {"confidence_margin": False}),
    ]
    for name, overrides in module_flags:
        flags = AblationFlags(**{**dataclasses.asdict(AblationFlags()), **overrides})
        configs.append(dataclasses.replace(base, name=name, ablation=flags))

    configs.append(baseline_cot_config(seed=base.seed))
    return configs


def _cmd_ablation(args: argparse.Namespace) -> int:
    base = _load_config(args.config, args.seed)
    configs = _ablation_configs(base)

    rows = []
    for cfg in configs:
        s = run_evaluation(cfg, n_per_family=args.n)
        rows.append(s)

    header = (f"{'config':<22} {'acc':>6} {'sel_acc':>8} {'cov':>6} "
              f"{'tok_cost':>9} {'llm':>6} {'inertia':>8} {'dead':>6} {'ece':>6} {'dep_H':>6}")
    print(header)
    print("-" * len(header))
    for s in rows:
        print(f"{s.config_name:<22} {s.accuracy:>6.3f} {s.selective_accuracy:>8.3f} "
              f"{s.coverage:>6.3f} {s.mean_token_cost:>9.3f} {s.mean_llm_calls:>6.2f} "
              f"{s.mean_inertia_saved:>8.2f} {s.mean_dead_steps_pruned:>6.2f} "
              f"{s.ece:>6.3f} {s.mean_dependency_entropy:>6.3f}")

    if args.out:
        _save_json(args.out, {"ablation": [s.to_dict() for s in rows]})
        print(f"\n[saved] {args.out}")
    return 0


# --------------------------------------------------------------------------
# config-dump
# --------------------------------------------------------------------------

def _cmd_config_dump(args: argparse.Namespace) -> int:
    config = ACDANConfig(name="default", seed=args.seed)
    if args.out:
        config.save_yaml(args.out)
        print(f"[saved] {args.out}")
    else:
        print(json.dumps(config.to_dict(), indent=2))
    return 0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _load_config(path: Optional[str], seed: Optional[int]) -> ACDANConfig:
    config = ACDANConfig.from_yaml(path) if path else ACDANConfig()
    if seed is not None:
        config = dataclasses.replace(config, seed=seed)
    return config


def _save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="acdan", description="ACDAN offline experiments.")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=str, default=None, help="YAML config path.")
    common.add_argument("--seed", type=int, default=None, help="Override config seed.")
    common.add_argument("--n", type=int, default=8, help="Tasks per family.")

    d = sub.add_parser("demo", parents=[common], help="Run one task with full trace.")
    d.add_argument("--task-index", type=int, default=0, help="Which task to run.")
    d.set_defaults(func=_cmd_demo)

    e = sub.add_parser("eval", parents=[common], help="Evaluate a config over the suite.")
    e.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    e.set_defaults(func=_cmd_eval)

    a = sub.add_parser("ablation", parents=[common], help="Run ablation comparison.")
    a.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    a.set_defaults(func=_cmd_ablation)

    c = sub.add_parser("config-dump", help="Dump the default config.")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", type=str, default=None, help="Optional YAML output path.")
    c.set_defaults(func=_cmd_config_dump)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
