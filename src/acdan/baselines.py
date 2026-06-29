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


# ---------------------------------------------------------------------------
# Additional related-work baselines (Tree-of-Thoughts, RAP, Self-Refine, s1).
# These share ACDAN's prior + PRM so the only thing that varies is the search /
# refinement rule. They are framework-agnostic: when H==1 (math candidate
# selection) they reduce to principled selectors over the K answer candidates;
# when H>1 (tool plans) they search the action-step lattice.
# ---------------------------------------------------------------------------


def tree_of_thoughts(
    core: CoreModel,
    prm: ProcessRewardModel,
    task: Task,
    latent: np.ndarray,
    n_per_step: int = 4,
    keep_top_b: int = 2,
    seed: int = 0,
) -> BaselineResult:
    """Tree-of-Thoughts (Yao et al., 2023): stepwise BFS with value-based pruning.

    At each step we expand each surviving partial by sampling ``n_per_step``
    distinct successor actions from the prior, score each extension with the
    PRM reward field, and keep the top ``keep_top_b`` partials by cumulative
    PRM score. Shares ACDAN's prior + PRM for a fair comparison.
    """
    prior = core.prior_logits(task, latent)
    R = prm.step_reward_matrix(task, latent)
    H, V = prior.shape
    probs = softmax(prior, axis=1)
    rng = np.random.default_rng(seed)

    frontier: List[tuple[List[int], float]] = [([], 0.0)]
    total_samples = 0
    for h in range(H):
        nxt: List[tuple[List[int], float]] = []
        for actions, cum in frontier:
            seen: set[int] = set()
            for _ in range(n_per_step):
                a = int(rng.choice(V, p=probs[h]))
                seen.add(a)
            total_samples += len(seen)
            for a in seen:
                nxt.append((actions + [a], cum + float(R[h, a])))
        nxt.sort(key=lambda item: item[1], reverse=True)
        frontier = nxt[: max(1, keep_top_b)] or [([0] * (h + 1), 0.0)]

    best = frontier[0][0] if frontier else [0] * H
    return BaselineResult(
        best,
        {
            "policy_passes": 1,
            "prm_passes": 1,
            "samples": float(total_samples),
            "verified_candidates": H * V,
            "keep_top_b": float(keep_top_b),
        },
    )


def reasoning_as_planning(
    core: CoreModel,
    prm: ProcessRewardModel,
    task: Task,
    latent: np.ndarray,
    n_rollouts: int = 8,
    c_puct: float = 1.4,
    seed: int = 0,
) -> BaselineResult:
    """RAP (Hao et al., 2023): MCTS with the LLM as world model + reward.

    We approximate the world-model rollout by treating the policy prior as
    the action policy and the PRM step reward as the value estimate. PUCT
    selection drives exploration; after ``n_rollouts`` simulations we read off
    the best plan by cumulative simulated reward.
    """
    prior = core.prior_logits(task, latent)
    R = prm.step_reward_matrix(task, latent)
    H, V = prior.shape
    probs = softmax(prior, axis=1)

    counts: Dict[tuple, np.ndarray] = {}
    values: Dict[tuple, np.ndarray] = {}

    def _puct(state: tuple) -> np.ndarray:
        n = counts.setdefault(state, np.zeros(V, dtype=np.float64))
        w = values.setdefault(state, np.zeros(V, dtype=np.float64))
        total = float(n.sum()) + 1.0
        q = w / np.maximum(n, 1.0)
        u = c_puct * probs[len(state)] * np.sqrt(total) / (1.0 + n)
        return q + u

    best_actions: List[int] = [int(np.argmax(R[h])) for h in range(H)]
    best_cum = -np.inf
    for _ in range(n_rollouts):
        state: tuple = ()
        actions: List[int] = []
        cum = 0.0
        for h in range(H):
            a = int(np.argmax(_puct(state)))
            actions.append(a)
            cum += float(R[h, a])
            state = state + (a,)
        # backup
        s: tuple = ()
        for a in actions:
            counts[s][a] += 1.0
            values[s][a] += cum
            s = s + (a,)
        if cum > best_cum:
            best_cum, best_actions = cum, actions

    return BaselineResult(
        best_actions,
        {
            "policy_passes": 1,
            "prm_passes": 1,
            "samples": float(n_rollouts),
            "verified_candidates": H * V,
            "c_puct": float(c_puct),
        },
    )


def self_refine(
    core: CoreModel,
    prm: ProcessRewardModel,
    task: Task,
    latent: np.ndarray,
    max_iters: int = 4,
    feedback_weight: float = 1.0,
    seed: int = 0,
) -> BaselineResult:
    """Self-Refine (Madaan et al., 2023): propose -> critique -> refine loop.

    Approximates the critic with the shared PRM: at each iteration we replace
    the current action by the argmax of ``feedback_weight * PRM + log prior``
    when that argmax differs from the current pick. Stops on the first
    no-change iteration (convergence).
    """
    prior = core.prior_logits(task, latent)
    R = prm.step_reward_matrix(task, latent)
    H, V = prior.shape
    log_p = np.log(softmax(prior, axis=1) + 1e-9)

    actions: List[int] = [int(np.argmax(prior[h])) for h in range(H)]
    iters_used = 0
    for _ in range(max_iters):
        iters_used += 1
        changed = False
        for h in range(H):
            blended = feedback_weight * R[h] + log_p[h]
            new = int(np.argmax(blended))
            if new != actions[h]:
                actions[h] = new
                changed = True
        if not changed:
            break

    return BaselineResult(
        actions,
        {
            "policy_passes": 1,
            "prm_passes": 1,
            "samples": float(iters_used),
            "verified_candidates": H * V,
            "max_iters": float(max_iters),
        },
    )


def s1_budget_forcing(
    core: CoreModel,
    prm: ProcessRewardModel,
    task: Task,
    latent: np.ndarray,
    max_budget: int = 16,
    min_budget: int = 2,
    plurality_threshold: float = 0.70,
    reward_temp: float = 1.0,
    seed: int = 0,
) -> BaselineResult:
    """s1 budget-forcing (Muennighoff et al., 2025): controlled thinking length.

    Maps the "Wait/Final-Answer" mechanism onto our candidate framework:
    PRM-weighted sampling is drawn from ``prior * exp(reward_temp * R)`` (the
    "thought continuation" distribution) and we keep drawing until either
    every step reaches plurality ``plurality_threshold`` (s1 "Final Answer"
    trigger) or the budget hits ``max_budget`` (s1 "Wait" cap).
    """
    prior = core.prior_logits(task, latent)
    R = prm.step_reward_matrix(task, latent)
    H, V = prior.shape
    probs = softmax(prior, axis=1)
    rng = np.random.default_rng(seed)

    weights = probs * np.exp(reward_temp * R)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-9)

    votes = np.zeros((H, V), dtype=np.float64)
    used = 0
    for n in range(1, max(min_budget, max_budget) + 1):
        used = n
        for h in range(H):
            a = int(rng.choice(V, p=weights[h]))
            votes[h, a] += 1.0
        if used < min_budget:
            continue
        plurality_ok = True
        for h in range(H):
            total = float(votes[h].sum())
            if total <= 0 or (votes[h].max() / total) < plurality_threshold:
                plurality_ok = False
                break
        if plurality_ok:
            break

    actions = [int(np.argmax(votes[h])) if votes[h].sum() else 0 for h in range(H)]
    return BaselineResult(
        actions,
        {
            "policy_passes": 1,
            "prm_passes": 1,
            "samples": float(used),
            "verified_candidates": H * V,
            "plurality_threshold": float(plurality_threshold),
            "max_budget": float(max_budget),
        },
    )
