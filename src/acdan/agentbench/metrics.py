"""Selection, calibration, and cost metrics for frozen AgentBench candidates."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from acdan.verification import expected_calibration_error


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    if not confidences:
        return 0.0
    p = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(correct, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def binary_nll(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    if not confidences:
        return 0.0
    p = np.clip(np.asarray(confidences, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    y = np.asarray(correct, dtype=np.float64)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def aurc(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the empirical risk-coverage curve; lower is better."""
    if not confidences:
        return 0.0
    order = np.argsort(-np.asarray(confidences, dtype=np.float64), kind="stable")
    errors = 1.0 - np.asarray(correct, dtype=np.float64)[order]
    risks = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(np.mean(risks))


def summarize_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {
            "n_tasks": 0,
            "selected_accuracy": 0.0,
            "selected_score": 0.0,
            "pass_at_k": 0.0,
            "oracle_score": 0.0,
            "verification_gap": 0.0,
            "recovery_rate": 0.0,
            "oracle_regret": 0.0,
            "ece": 0.0,
            "brier": 0.0,
            "nll": 0.0,
            "aurc": 0.0,
        }
    correct = [bool(row["selected_correct"]) for row in rows]
    confidence = [float(row.get("confidence", 0.5)) for row in rows]
    selected_accuracy = float(np.mean(correct))
    pass_at_k = float(np.mean([bool(row["pass_at_k"]) for row in rows]))
    oracle_score = float(np.mean([float(row["oracle_score"]) for row in rows]))
    selected_score = float(np.mean([float(row["selected_score"]) for row in rows]))
    answered = [row for row in rows if not bool(row.get("abstained", False))]
    selective_accuracy = (
        float(np.mean([bool(row["selected_correct"]) for row in answered]))
        if answered else 0.0
    )
    return {
        "n_tasks": n,
        "selected_accuracy": selected_accuracy,
        "selected_score": selected_score,
        "pass_at_k": pass_at_k,
        "oracle_score": oracle_score,
        "verification_gap": pass_at_k - selected_accuracy,
        "recovery_rate": selected_accuracy / pass_at_k if pass_at_k > 0 else 0.0,
        "oracle_regret": oracle_score - selected_score,
        "ece": expected_calibration_error(confidence, correct),
        "brier": brier_score(confidence, correct),
        "nll": binary_nll(confidence, correct),
        "aurc": aurc(confidence, correct),
        "coverage": len(answered) / n,
        "selective_accuracy": selective_accuracy,
        "mean_confidence": float(np.mean(confidence)),
    }


def paired_bootstrap_delta(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    field: str = "selected_correct",
    samples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    a = {str(row["task_id"]): float(row[field]) for row in rows_a}
    b = {str(row["task_id"]): float(row[field]) for row in rows_b}
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("paired bootstrap requires overlapping task IDs")
    delta = np.asarray([a[task_id] - b[task_id] for task_id in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        draws[i] = float(np.mean(rng.choice(delta, size=len(delta), replace=True)))
    return {
        "delta": float(np.mean(delta)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "n_pairs": float(len(ids)),
    }
