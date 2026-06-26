"""Synthetic, fully-offline agentic tasks.

Each task is a short *tool-planning* problem: pick the horizon-length sequence of
actions (tool calls) that matches a hidden optimal plan. We build three task
families mirroring the proposal's evaluation groups:

  * ``math``  — reasoning/logic flavour (cf. GSM8K / MATH)
  * ``code``  — programming / tool-use flavour (cf. LiveCodeBench / ToolBench)
  * ``tool``  — repetitive workflow flavour (cf. agentic office tasks)

Crucially, optimal plans within a family are generated from a near-deterministic
*motif chain*, so the resulting execution traces contain strong, recurring
transitions. This is what gives Tool-Usage-Inertia something real to exploit, and
lets the inertia ablation show a measurable token-cost difference.

Nothing here downloads anything; everything is seeded numpy.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from acdan.types import ActionId, Task, stable_seed

FEATURE_DIM = 24

# Per-family configuration: (vocab names, horizon).
_FAMILIES: Dict[str, Tuple[Tuple[str, ...], int]] = {
    "math": (("read", "decompose", "compute", "substitute", "simplify", "check"), 5),
    "code": (("spec", "scaffold", "implement", "run_tests", "debug", "refactor", "submit"), 6),
    "tool": (("open", "search", "extract", "transform", "write", "verify"), 5),
}


def _motif_chain(vocab_size: int, seed: int) -> np.ndarray:
    """A near-deterministic transition matrix defining a family's canonical flow.

    Each state has one dominant successor (probability ~0.8), creating the strong
    recurring transitions that inertial sensing can learn.
    """
    rng = np.random.default_rng(seed)
    T = np.full((vocab_size, vocab_size), 0.2 / (vocab_size - 1))
    for s in range(vocab_size):
        nxt = (s + 1) % vocab_size  # canonical "next stage" motif
        T[s] = 0.2 / (vocab_size - 1)
        T[s, nxt] = 0.8
    return T


def _sample_plan(T: np.ndarray, horizon: int, start: int, rng: np.random.Generator) -> List[ActionId]:
    plan = [start]
    s = start
    for _ in range(horizon - 1):
        s = int(rng.choice(T.shape[0], p=T[s]))
        plan.append(s)
    return plan


def make_task(task_id: str, family: str, seed: int) -> Task:
    """Create one synthetic task from a given family."""
    if family not in _FAMILIES:
        raise KeyError(f"unknown family '{family}'. Available: {list(_FAMILIES)}")
    vocab, horizon = _FAMILIES[family]
    V = len(vocab)
    rng = np.random.default_rng(seed)
    T = _motif_chain(V, seed=stable_seed(family, "motif"))
    start = int(rng.integers(0, V))
    optimal = tuple(_sample_plan(T, horizon, start, rng))
    features = rng.normal(0.0, 1.0, size=FEATURE_DIM)
    difficulty = float(np.clip(rng.uniform(0.25, 0.85), 0.0, 1.0))
    return Task(
        task_id=task_id,
        prompt_features=features,
        vocab=vocab,
        horizon=horizon,
        difficulty=difficulty,
        optimal_plan=optimal,
        metadata={"family": family, "seed": seed},
    )


def make_suite(n_per_family: int = 8, seed: int = 0, families: Sequence[str] = ("math", "code", "tool")) -> List[Task]:
    """Build a reproducible evaluation suite across families."""
    tasks: List[Task] = []
    counter = 0
    for fam in families:
        for k in range(n_per_family):
            tasks.append(make_task(f"{fam}-{k:03d}", fam, seed=seed * 1000 + counter))
            counter += 1
    return tasks


def make_traces(tasks: Sequence[Task], n_per_task: int = 4, seed: int = 0, noise: float = 0.1) -> List[List[ActionId]]:
    """Generate historical execution traces for fitting the inertial sensor.

    Traces are noisy realisations of each task's optimal plan: with probability
    ``noise`` a step is replaced by a random action. This yields strong-but-
    imperfect transition statistics, exactly like real execution logs.
    """
    rng = np.random.default_rng(seed)
    traces: List[List[ActionId]] = []
    for t in tasks:
        if t.optimal_plan is None:
            continue
        V = t.vocab_size
        for _ in range(n_per_task):
            tr: List[ActionId] = []
            for a in t.optimal_plan:
                if rng.random() < noise:
                    tr.append(int(rng.integers(0, V)))
                else:
                    tr.append(int(a))
            traces.append(tr)
    return traces
