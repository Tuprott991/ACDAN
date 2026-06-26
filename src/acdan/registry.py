"""Pluggable backends: core model, PRM, and dataset adapters.

This module is the single seam between ACDAN's architecture and the outside
world. The offline demo wires up *mock* numpy backends here; swapping in real
LLMs / PRMs / datasets later means implementing the same tiny interfaces and
registering them — **no change to the core architecture** (agent, DTO, graph,
inertia, verification all program against the interfaces, not the backends).

Interfaces:
  * ``CoreModel.prior_logits(task, latent) -> (horizon, vocab)`` — the base
    action distribution the DTO loop refines. Real backend: an LLM's action-head
    logits. Mock backend: a seeded, deliberately-imperfect prior.
  * ``ProcessRewardModel`` — see ``acdan.rewards``.
  * ``DatasetAdapter.tasks() -> Iterable[Task]`` — yields tasks. Real backend:
    GAIA / GSM8K / LiveCodeBench loaders (NOT shipped, NOT downloaded). Mock
    backend: ``acdan.tasks.synthetic``.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Protocol

import numpy as np

from acdan.rewards import MockProcessReward, ProcessRewardModel
from acdan.types import Task, stable_seed


class CoreModel(Protocol):
    """The base agent policy that emits prior action logits."""

    def prior_logits(self, task: Task, latent: np.ndarray) -> np.ndarray:
        ...


class DatasetAdapter(Protocol):
    """Yields tasks for evaluation."""

    def tasks(self) -> Iterable[Task]:
        ...


class MockCoreModel:
    """Offline core model producing a weak, imperfect prior over actions.

    The prior places a small, latent-modulated bump on the (known) optimal
    action at each step plus noise, so that:
      * greedy decoding alone is only partially correct (room to improve),
      * a better latent state sharpens the prior, and
      * the DTO loop + PRM can measurably push the plan toward the optimum.

    A real LLM core model would replace this with genuine action-head logits;
    nothing else in the pipeline changes.
    """

    def __init__(self, signal: float = 0.6, noise: float = 1.0, seed: int = 0):
        self.signal = signal
        self.noise = noise
        self._seed = seed

    def prior_logits(self, task: Task, latent: np.ndarray) -> np.ndarray:
        H, V = task.horizon, task.vocab_size
        rng = np.random.default_rng(stable_seed(task.task_id, "core"))
        logits = self.noise * rng.normal(0.0, 1.0, size=(H, V))
        # Latent quality in [0, 1] sharpens the correct-action bump.
        lq = float(np.tanh(np.linalg.norm(latent) / (np.sqrt(latent.size) + 1e-9)))
        bump = self.signal * (0.5 + 0.5 * lq)
        if task.optimal_plan is not None:
            for h in range(H):
                a = task.optimal_plan[h % len(task.optimal_plan)]
                logits[h, a] += bump
        return logits


# --------------------------------------------------------------------------
# Simple registries so configs can name a backend by string.
# --------------------------------------------------------------------------

_CORE_MODELS: Dict[str, Callable[..., CoreModel]] = {
    "mock": MockCoreModel,
}
_PRMS: Dict[str, Callable[..., ProcessRewardModel]] = {
    "mock": MockProcessReward,
}


def register_core_model(name: str, factory: Callable[..., CoreModel]) -> None:
    _CORE_MODELS[name] = factory


def register_prm(name: str, factory: Callable[..., ProcessRewardModel]) -> None:
    _PRMS[name] = factory


def build_core_model(name: str = "mock", **kwargs) -> CoreModel:
    if name not in _CORE_MODELS:
        raise KeyError(f"unknown core model '{name}'. Available: {list(_CORE_MODELS)}")
    return _CORE_MODELS[name](**kwargs)


def build_prm(name: str = "mock", **kwargs) -> ProcessRewardModel:
    if name not in _PRMS:
        raise KeyError(f"unknown PRM '{name}'. Available: {list(_PRMS)}")
    return _PRMS[name](**kwargs)
