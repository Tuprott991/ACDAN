"""Run the full ACDAN ablation grid over several seeds and print mean +/- std.

This reproduces the kind of ablation table you would put in a paper / rebuttal.
Run:  python examples/run_ablation.py
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Dict, List

import numpy as np

from acdan.config import ACDANConfig, AblationFlags, baseline_cot_config
from acdan.evaluate import run_evaluation

SEEDS = [0, 1, 2]
N_PER_FAMILY = 8


def configs_for_seed(seed: int) -> List[ACDANConfig]:
    base = ACDANConfig(seed=seed)
    out = [dataclasses.replace(base, name="acdan_full", ablation=AblationFlags())]
    overrides = {
        "no_latent": {"latent_reasoning": False, "in_place_ttt": False},
        "no_dto": {"dto": False},
        "no_graph": {"dependency_graph": False},
        "no_inertia": {"inertial_sensing": False},
        "no_verification": {"verification": False},
    }
    for name, ov in overrides.items():
        flags = AblationFlags(**{**dataclasses.asdict(AblationFlags()), **ov})
        out.append(dataclasses.replace(base, name=name, ablation=flags))
    out.append(baseline_cot_config(seed=seed))
    return out


def main() -> None:
    acc: Dict[str, List[float]] = defaultdict(list)
    tok: Dict[str, List[float]] = defaultdict(list)
    ece: Dict[str, List[float]] = defaultdict(list)

    order: List[str] = []
    for seed in SEEDS:
        for cfg in configs_for_seed(seed):
            s = run_evaluation(cfg, n_per_family=N_PER_FAMILY)
            if cfg.name not in order:
                order.append(cfg.name)
            acc[cfg.name].append(s.accuracy)
            tok[cfg.name].append(s.mean_token_cost)
            ece[cfg.name].append(s.ece)

    print(f"Ablation over seeds={SEEDS}, n_per_family={N_PER_FAMILY}\n")
    print(f"{'config':<18} {'accuracy':>16} {'token_cost':>16} {'ECE':>16}")
    print("-" * 68)
    for name in order:
        a = f"{np.mean(acc[name]):.3f}+/-{np.std(acc[name]):.3f}"
        t = f"{np.mean(tok[name]):.3f}+/-{np.std(tok[name]):.3f}"
        e = f"{np.mean(ece[name]):.3f}+/-{np.std(ece[name]):.3f}"
        print(f"{name:<18} {a:>16} {t:>16} {e:>16}")


if __name__ == "__main__":
    main()
