"""Learnable synthetic tasks for PS-GRPO.

Unlike the test-time synthetic suite (where features are random noise so DTO can
be studied in isolation), these tasks make the prompt features **informative**:
each task's features are a fixed linear projection of its optimal-plan one-hots
plus noise. The projection is shared within a family, so a parametric policy can
*learn* the features → plan mapping — giving PS-GRPO a real training signal and a
measurable learning curve.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from acdan.tasks.synthetic import _FAMILIES, _motif_chain, _sample_plan
from acdan.types import Task, stable_seed

FEATURE_DIM = 32


def _projection(family: str, feature_dim: int, hv: int) -> np.ndarray:
    """Fixed, family-shared projection matrix (D x H*V)."""
    rng = np.random.default_rng(stable_seed(family, "psgrpo-proj", feature_dim, hv))
    return rng.normal(0.0, 1.0, size=(feature_dim, hv))


def make_learnable_task(task_id: str, family: str, seed: int,
                        feature_dim: int = FEATURE_DIM, noise: float = 0.5) -> Task:
    if family not in _FAMILIES:
        raise KeyError(f"unknown family '{family}'")
    vocab, horizon = _FAMILIES[family]
    V = len(vocab)
    rng = np.random.default_rng(seed)
    T = _motif_chain(V, seed=stable_seed(family, "motif"))
    start = int(rng.integers(0, V))
    optimal = tuple(_sample_plan(T, horizon, start, rng))

    onehot = np.zeros(horizon * V)
    for h, a in enumerate(optimal):
        onehot[h * V + a] = 1.0
    M = _projection(family, feature_dim, horizon * V)
    feats = M @ onehot + noise * rng.normal(0.0, 1.0, size=feature_dim)
    feats /= (np.linalg.norm(feats) + 1e-9)

    difficulty = float(np.clip(rng.uniform(0.25, 0.85), 0.0, 1.0))
    return Task(task_id=task_id, prompt_features=feats, vocab=vocab, horizon=horizon,
                difficulty=difficulty, optimal_plan=optimal,
                metadata={"family": family, "seed": seed})


def make_learnable_suite(n_per_family: int = 16, seed: int = 0,
                         feature_dim: int = FEATURE_DIM,
                         families: Sequence[str] = ("math", "code", "tool")) -> List[Task]:
    tasks: List[Task] = []
    counter = 0
    for fam in families:
        for k in range(n_per_family):
            tasks.append(make_learnable_task(f"{fam}-{k:03d}", fam,
                                             seed=seed * 10000 + counter,
                                             feature_dim=feature_dim))
            counter += 1
    return tasks
