"""PS-GRPO training runner (offline, numpy).

Trains the ACDAN policy with Process-Supervised GRPO + RLCM on the learnable
synthetic suite and reports the eval-accuracy learning curve. Fully offline and
deterministic; this is the training-time counterpart to ``run_experiment.py``.

    python -m acdan.train --iters 60 --out results/psgrpo.json
    python -m acdan.train --no-process            # ablate process supervision
    python -m acdan.train --no-confidence-margin  # ablate RLCM margin

See docs/math.md for the objective and experiments/PLAN.md for the training study.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

from acdan.config import LatentConfig
from acdan.latent_reasoning import LatentReasoner
from acdan.registry import build_prm
from acdan.training.psgrpo import PSGRPOConfig, PSGRPOTrainer
from acdan.training.tasks import FEATURE_DIM, make_learnable_suite


def run(args: argparse.Namespace) -> dict:
    train_tasks = make_learnable_suite(n_per_family=args.n_per_family, seed=args.seed,
                                       feature_dim=FEATURE_DIM)
    eval_tasks = make_learnable_suite(n_per_family=args.eval_n, seed=args.seed + 1,
                                      feature_dim=FEATURE_DIM)

    cfg = PSGRPOConfig(
        iters=args.iters, group_size=args.group_size, lr=args.lr, seed=args.seed,
        use_process=not args.no_process,
        use_confidence_margin=not args.no_confidence_margin,
        use_group_baseline=not args.no_baseline,
        use_kl=not args.no_kl,
    )
    prm = build_prm("mock", seed=args.seed)
    reasoner = LatentReasoner(LatentConfig(), feature_dim=FEATURE_DIM, seed=args.seed)
    trainer = PSGRPOTrainer(cfg, prm, reasoner)

    hist = trainer.train(train_tasks, eval_tasks)
    first, last = hist.iters[0], hist.iters[-1]
    summary = {
        "iters": cfg.iters, "group_size": cfg.group_size, "seed": cfg.seed,
        "ablation": {"process": cfg.use_process, "confidence_margin": cfg.use_confidence_margin,
                     "group_baseline": cfg.use_group_baseline, "kl": cfg.use_kl},
        "eval_acc_start": first["eval_acc"], "eval_acc_final": last["eval_acc"],
        "train_reward_start": first["train_reward"], "train_reward_final": last["train_reward"],
    }
    out = {"summary": summary, "curve": hist.iters}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)

    print(f"PS-GRPO  ablation={summary['ablation']}")
    print(f"  eval acc : {first['eval_acc']:.3f} -> {last['eval_acc']:.3f}")
    print(f"  train rwd: {first['train_reward']:.3f} -> {last['train_reward']:.3f}")
    if args.print_curve:
        for r in hist.iters:
            print(f"  it={r['iter']:>3}  reward={r['train_reward']:.3f}  eval_acc={r['eval_acc']:.3f}")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="acdan-train", description="PS-GRPO training (offline).")
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--n-per-family", type=int, default=16)
    p.add_argument("--eval-n", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-process", action="store_true", help="Ablate process supervision.")
    p.add_argument("--no-confidence-margin", action="store_true", help="Ablate RLCM margin.")
    p.add_argument("--no-baseline", action="store_true", help="Ablate GRPO group baseline.")
    p.add_argument("--no-kl", action="store_true", help="Ablate KL penalty.")
    p.add_argument("--print-curve", action="store_true")
    p.add_argument("--out", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
