"""How to plug a REAL backend (PRM / core model / dataset) without touching the
core architecture.

This example implements a custom ``ProcessRewardModel`` and a custom
``CoreModel`` and registers them. In a real project these would wrap an actual
LLM (for prior logits) and a real PRM (e.g. TIM-PRM / Athena-PRM). Here we use
trivial deterministic stand-ins so the example still runs fully offline.

Run:  python examples/custom_backend.py
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from acdan.config import ACDANConfig
from acdan.latent_reasoning import LatentReasoner
from acdan.registry import register_core_model, register_prm, build_core_model, build_prm
from acdan.tasks.synthetic import FEATURE_DIM, make_suite
from acdan.types import Task, stable_seed


class MyPRM:
    """A custom PRM. Only these four methods are required by the DTO loop."""

    def step_reward_matrix(self, task: Task, latent: np.ndarray) -> np.ndarray:
        # Real impl: query your PRM for per-step per-action scores.
        rng = np.random.default_rng(stable_seed(task.task_id, 'myprm'))
        return rng.normal(0.0, 1.0, size=(task.horizon, task.vocab_size))

    def _target(self, task: Task, latent: np.ndarray) -> np.ndarray:
        R = self.step_reward_matrix(task, latent)
        z = R - R.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def score_probs(self, task: Task, latent: np.ndarray, probs: np.ndarray) -> float:
        return float(np.sum(probs * self._target(task, latent)) / probs.shape[0])

    def grad_wrt_probs(self, task: Task, latent: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return self._target(task, latent) / probs.shape[0]

    def score_actions(self, task: Task, latent: np.ndarray, actions: Sequence[int]) -> List[float]:
        t = self._target(task, latent)
        return [float(t[min(h, t.shape[0] - 1), int(a)]) for h, a in enumerate(actions)]


class MyCoreModel:
    """A custom core model returning prior action logits."""

    def prior_logits(self, task: Task, latent: np.ndarray) -> np.ndarray:
        # Real impl: return your LLM's action-head logits.
        rng = np.random.default_rng(stable_seed(task.task_id, 'mycore'))
        return rng.normal(0.0, 1.0, size=(task.horizon, task.vocab_size))


def main() -> None:
    register_prm("my_prm", MyPRM)
    register_core_model("my_core", MyCoreModel)

    config = ACDANConfig(name="custom", seed=0)
    prm = build_prm("my_prm")
    core = build_core_model("my_core")
    reasoner = LatentReasoner(config.latent, FEATURE_DIM, seed=config.seed)

    task = make_suite(n_per_family=2, seed=0)[0]
    latent = reasoner.reason(task.prompt_features).final_state

    # The DTO loop is backend-agnostic: it only sees the interfaces.
    from acdan.dto import DifferentiableTextOptimizer
    dto = DifferentiableTextOptimizer(config.dto, prm)
    plan = dto.optimize(task, latent, core.prior_logits(task, latent))

    print("custom backend plan:", [task.vocab[a] for a in plan.actions])
    print("objective:", round(plan.objective_trace[0], 4), "->",
          round(plan.objective_trace[-1], 4))


if __name__ == "__main__":
    main()
