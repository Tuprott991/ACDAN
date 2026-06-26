"""Aggregation of per-task rollouts into a reproducible evaluation summary."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from acdan.types import EvalSummary, RolloutMetrics
from acdan.verification import expected_calibration_error


def summarize(rollouts: Sequence[RolloutMetrics], config_name: str = "default") -> EvalSummary:
    """Aggregate per-task metrics into an :class:`EvalSummary`.

    Notable derived metrics:
      * ``coverage`` — fraction of tasks the agent did *not* abstain on.
      * ``selective_accuracy`` — accuracy among non-abstained answers (the metric
        that matters for risk-controlled deployment).
      * ``ece`` — calibration of the confidence estimates vs. correctness.
    """
    n = len(rollouts)
    if n == 0:
        return EvalSummary(
            n_tasks=0, accuracy=0.0, coverage=0.0, selective_accuracy=0.0,
            mean_token_cost=0.0, mean_llm_calls=0.0, mean_inertia_saved=0.0,
            mean_dead_steps_pruned=0.0, mean_prm=0.0, ece=0.0,
            mean_dependency_entropy=0.0, config_name=config_name, per_task=[],
        )

    correct = np.array([r.correct for r in rollouts], dtype=bool)
    abstained = np.array([r.abstained for r in rollouts], dtype=bool)
    answered = ~abstained

    accuracy = float(correct.mean())
    coverage = float(answered.mean())
    if answered.any():
        selective_accuracy = float(correct[answered].mean())
    else:
        selective_accuracy = 0.0

    ece = expected_calibration_error(
        [r.confidence for r in rollouts],
        [bool(r.correct) for r in rollouts],
    )

    return EvalSummary(
        n_tasks=n,
        accuracy=accuracy,
        coverage=coverage,
        selective_accuracy=selective_accuracy,
        mean_token_cost=float(np.mean([r.token_cost for r in rollouts])),
        mean_llm_calls=float(np.mean([r.llm_calls for r in rollouts])),
        mean_inertia_saved=float(np.mean([r.inertia_saved_calls for r in rollouts])),
        mean_dead_steps_pruned=float(np.mean([r.dead_steps_pruned for r in rollouts])),
        mean_prm=float(np.mean([r.mean_prm for r in rollouts])),
        ece=ece,
        mean_dependency_entropy=float(np.mean([r.dependency_entropy for r in rollouts])),
        config_name=config_name,
        per_task=list(rollouts),
    )


def format_summary(summary: EvalSummary) -> str:
    """Human-readable one-block report (used by the CLI)."""
    s = summary
    lines = [
        f"=== ACDAN eval summary: {s.config_name} ===",
        f"tasks                : {s.n_tasks}",
        f"accuracy             : {s.accuracy:.3f}",
        f"coverage             : {s.coverage:.3f}",
        f"selective accuracy   : {s.selective_accuracy:.3f}",
        f"mean token cost      : {s.mean_token_cost:.3f}",
        f"mean LLM plan calls  : {s.mean_llm_calls:.3f}",
        f"mean inertia saved   : {s.mean_inertia_saved:.3f}",
        f"mean dead-steps pruned: {s.mean_dead_steps_pruned:.3f}",
        f"mean PRM             : {s.mean_prm:.3f}",
        f"ECE (calibration)    : {s.ece:.3f}",
        f"mean dep. entropy    : {s.mean_dependency_entropy:.3f}",
    ]
    return "\n".join(lines)
