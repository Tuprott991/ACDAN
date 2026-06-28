"""Comparison baselines that share ACDAN's core model + PRM (for fair studies).

All baselines operate on the same (H, V) prior the policy produces, so the only
thing that differs from ACDAN is the *decision rule* — discrete argmax / sampling
vs. ACDAN's continuous DTO refinement. This isolates the contribution of DTO at a
matched model + PRM budget (the key comparison in the paper).

Each returns a :class:`BaselineResult` with executed actions and cost counters
(policy passes, PRM passes, samples) for real efficiency reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from acdan.registry import CoreModel
from acdan.rewards import ProcessRewardModel
from acdan.types import Task, softmax


@dataclass
class BaselineResult:
    actions: List[int]
    cost: Dict[str, float] = field(default_factory=dict)


def cot_greedy(core: CoreModel, task: Task, latent: np.ndarray) -> BaselineResult:
    """Chain-of-thought analogue: one pass, greedy argmax of the prior."""
    prior = core.prior_logits(task, latent)
    actions = [int(np.argmax(prior[h])) for h in range(prior.shape[0])]
    return BaselineResult(actions, {"policy_passes": 1, "prm_passes": 0, "samples": 1})


def self_consistency(core: CoreModel, task: Task, latent: np.ndarray,
                     n: int = 8, seed: int = 0) -> BaselineResult:
    """Sample N plans from the prior; majority-vote per step (no PRM)."""
    prior = core.prior_logits(task, latent)
    probs = softmax(prior, axis=1)
    rng = np.random.default_rng(seed)
    H, V = prior.shape
    votes = np.zeros((H, V))
    for _ in range(n):
        for h in range(H):
            a = int(rng.choice(V, p=probs[h]))
            votes[h, a] += 1
    actions = [int(np.argmax(votes[h])) for h in range(H)]
    return BaselineResult(actions, {"policy_passes": 1, "prm_passes": 0, "samples": n})


def _candidate_sample_trace(task: Task) -> List[int]:
    """Return a deterministic candidate sample trace for grouped math files."""

    raw_trace = task.metadata.get("candidate_sample_answers", []) or []
    if raw_trace:
        trace = []
        for answer in raw_trace:
            if str(answer) in task.vocab:
                trace.append(task.vocab.index(str(answer)))
        if trace:
            return trace

    counts = task.metadata.get("candidate_counts", {}) or {}
    first = task.metadata.get("candidate_first_indices", {}) or {}
    if counts:
        ordered = sorted(
            range(task.vocab_size),
            key=lambda i: int(first.get(str(task.vocab[i]), i)),
        )
        trace = []
        for i in ordered:
            trace.append(i)
        for i in ordered:
            trace.extend([i] * max(0, int(counts.get(str(task.vocab[i]), 1)) - 1))
        if trace:
            return trace

    return list(range(task.vocab_size))


def adaptive_self_consistency(
    core: CoreModel,
    task: Task,
    latent: np.ndarray,
    n: int = 8,
    threshold: float = 0.70,
    min_samples: int = 2,
    seed: int = 0,
) -> BaselineResult:
    """Early-stopping self-consistency over candidates.

    For math candidate files this replays the stored sample-answer trace, or a
    reconstruction from counts/first indices.  For generic tasks it samples from
    the model prior and stops once the current plurality has enough support.
    """

    H, V = task.horizon, task.vocab_size
    votes = np.zeros((H, V), dtype=np.float64)
    used = 0

    if H == 1 and task.metadata.get("candidate_counts"):
        trace = _candidate_sample_trace(task)[:max(1, n)]
        for used, action in enumerate(trace, start=1):
            votes[0, int(action)] += 1
            if used >= min_samples and votes[0].max() / used >= threshold:
                break
    else:
        prior = core.prior_logits(task, latent)
        probs = softmax(prior, axis=1)
        rng = np.random.default_rng(seed)
        for used in range(1, n + 1):
            for h in range(H):
                a = int(rng.choice(V, p=probs[h]))
                votes[h, a] += 1
            confident = [
                votes[h].max() / max(1.0, votes[h].sum()) >= threshold
                for h in range(H)
            ]
            if used >= min_samples and all(confident):
                break

    actions = [int(np.argmax(votes[h])) if votes[h].sum() else 0 for h in range(H)]
    return BaselineResult(
        actions,
        {
            "policy_passes": 1,
            "prm_passes": 0,
            "samples": float(used or 1),
            "stop_threshold": float(threshold),
        },
    )


def best_of_n_prm(core: CoreModel, prm: ProcessRewardModel, task: Task,
                  latent: np.ndarray, n: int = 8, seed: int = 0) -> BaselineResult:
    """Sample N plans from the prior; keep the one with the highest PRM score.

    This is the strongest fair baseline for DTO: same prior, same PRM, discrete
    search instead of continuous optimisation.
    """
    prior = core.prior_logits(task, latent)
    probs = softmax(prior, axis=1)
    R = prm.step_reward_matrix(task, latent)  # one PRM pass, reused for rerank
    rng = np.random.default_rng(seed)
    H, V = prior.shape
    best_actions, best_score = None, -np.inf
    for _ in range(n):
        plan = [int(rng.choice(V, p=probs[h])) for h in range(H)]
        score = float(sum(R[h, plan[h]] for h in range(H)))
        if score > best_score:
            best_score, best_actions = score, plan
    return BaselineResult(best_actions or [0] * H,
                          {
                              "policy_passes": 1,
                              "prm_passes": 1,
                              "samples": n,
                              "verified_candidates": H * V,
                          })
