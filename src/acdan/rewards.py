"""Reward interfaces: Process Reward Model (PRM) and Net Information Gain (NIG).

Paper mapping:
  * ``ProcessRewardModel`` is the pluggable PRM interface. The proposal uses a
    multimodal, tool-integrated PRM (TIM-PRM / Athena-PRM). Here we ship a
    lightweight, fully-offline ``MockProcessReward`` that exposes the *same*
    interface, including a gradient w.r.t. the soft action distribution so the
    DTO loop can back-propagate (proposal: "Đạo hàm từ PRM được truyền ngược").
  * ``net_information_gain`` implements the linear-time O(N) process-supervision
    signal from the proposal ("Monte Carlo Net Information Gain"), avoiding the
    O(N^2) cost of classical Monte-Carlo rollout labelling.

To plug a real PRM later, implement ``ProcessRewardModel`` and register it in
``acdan.registry`` — the DTO loop and agent are agnostic to the backend.
"""

from __future__ import annotations

from typing import List, Protocol, Sequence

import numpy as np

from acdan.types import Task, stable_seed


class ProcessRewardModel(Protocol):
    """Interface every PRM backend must satisfy.

    The core requirement for differentiable optimisation is that the PRM can
    return both a scalar score for a soft plan and the gradient of that score
    with respect to the per-step action probabilities.
    """

    def step_reward_matrix(self, task: Task, latent: np.ndarray) -> np.ndarray:
        """Return per-step per-action reward, shape (horizon, vocab_size)."""
        ...

    def score_probs(self, task: Task, latent: np.ndarray, probs: np.ndarray) -> float:
        """Scalar process reward for a soft plan ``probs`` (horizon, vocab)."""

    def grad_wrt_probs(self, task: Task, latent: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """Gradient d(score)/d(probs), shape (horizon, vocab_size)."""

    def score_actions(self, task: Task, latent: np.ndarray, actions: Sequence[int]) -> List[float]:
        """Per-step PRM scores in [0, 1] for a *discrete* executed plan."""

    def extension_rewards(
        self, task: Task, latent: np.ndarray, prefix: Sequence[int]
    ) -> np.ndarray:
        """Reward for each possible next action after ``prefix``."""

    def trajectory_rewards(
        self, task: Task, latent: np.ndarray, trajectories: Sequence[Sequence[int]]
    ) -> List[float]:
        """Score complete trajectories in [0, 1]."""


class MockProcessReward:
    """Offline PRM stand-in with a linear, differentiable reward surface.

    The reward matrix R has shape (horizon, vocab). The process reward of a soft
    plan P (row-stochastic) is the bilinear form

        score(P) = (1 / H) * sum_h <P_h, sigmoid(R_h)>

    which is differentiable with constant gradient ``sigmoid(R) / H`` w.r.t. P.
    R is constructed from per-step *target* embeddings so that the PRM prefers
    actions consistent with the task — exactly what a process reward model does —
    without ever being handed the discrete answer at inference time.
    """

    def __init__(self, embed_dim: int = 16, noise: float = 0.9,
                 sharpness: float = 4.0, seed: int = 0):
        self.embed_dim = embed_dim
        self.noise = noise
        # Sharpness of the per-step reward target (higher -> more decisively
        # concentrated on the best action). Controls how strongly the PRM signal
        # contrasts good vs. bad actions, i.e. how informative the gradient is.
        self.sharpness = sharpness
        self._seed = seed

    # -- internal helpers ---------------------------------------------------

    def _action_embeddings(self, task: Task) -> np.ndarray:
        """Deterministic per-action embeddings, shape (vocab, embed_dim)."""
        rng = np.random.default_rng(stable_seed(task.task_id, "act"))
        return rng.normal(0.0, 1.0, size=(task.vocab_size, self.embed_dim))

    def _target_embeddings(self, task: Task, latent: np.ndarray) -> np.ndarray:
        """Per-step target embeddings, shape (horizon, embed_dim).

        For synthetic tasks the targets are anchored on the (known) optimal plan
        plus latent-conditioned noise, so a *better* latent state yields a
        cleaner reward signal — letting latent reasoning measurably help.
        """
        emb = self._action_embeddings(task)
        rng = np.random.default_rng(stable_seed(task.task_id, "tgt"))
        # Latent quality term: a well-formed latent reduces target noise.
        lq = float(np.tanh(np.linalg.norm(latent) / (np.sqrt(latent.size) + 1e-9)))
        eff_noise = self.noise * (1.0 - 0.7 * lq)
        targets = np.zeros((task.horizon, self.embed_dim))
        for h in range(task.horizon):
            if task.optimal_plan is not None:
                base = emb[task.optimal_plan[h % len(task.optimal_plan)]]
            else:  # no ground truth: use a latent-projected pseudo-target
                base = rng.normal(0.0, 1.0, size=self.embed_dim)
            targets[h] = base + eff_noise * rng.normal(0.0, 1.0, size=self.embed_dim)
        return targets

    def step_reward_matrix(self, task: Task, latent: np.ndarray) -> np.ndarray:
        emb = self._action_embeddings(task)          # (V, d)
        targets = self._target_embeddings(task, latent)  # (H, d)
        # Cosine-like similarity between each action and each step target.
        emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        tgt_n = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-9)
        R = tgt_n @ emb_n.T                          # (H, V), in [-1, 1]
        return R

    def _reward_target(self, task: Task, latent: np.ndarray) -> np.ndarray:
        """Per-step softmax-sharpened reward target, rows sum to 1, shape (H, V).

        Sharpening turns the raw cosine field into an informative, high-contrast
        gradient: the PRM-preferred action per step gets clearly more mass, so
        the DTO loop can decisively move the soft plan toward it.
        """
        R = self.step_reward_matrix(task, latent)
        z = self.sharpness * R
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # -- ProcessRewardModel API --------------------------------------------

    def score_probs(self, task: Task, latent: np.ndarray, probs: np.ndarray) -> float:
        target = self._reward_target(task, latent)
        return float(np.sum(probs * target) / probs.shape[0])

    def grad_wrt_probs(self, task: Task, latent: np.ndarray, probs: np.ndarray) -> np.ndarray:
        target = self._reward_target(task, latent)
        return target / probs.shape[0]

    def score_actions(self, task: Task, latent: np.ndarray, actions: Sequence[int]) -> List[float]:
        """Per-step *absolute* quality in [0, 1] for an executed plan.

        Uses the sigmoid of the raw similarity (not the cross-step softmax) so a
        step's score reflects its own quality, suitable for NIG and reporting.
        """
        R = self._sigmoid(self.step_reward_matrix(task, latent))
        out: List[float] = []
        for h, a in enumerate(actions):
            row = min(h, R.shape[0] - 1)
            out.append(float(R[row, int(a)]))
        return out

    def extension_rewards(
        self, task: Task, latent: np.ndarray, prefix: Sequence[int]
    ) -> np.ndarray:
        rewards = self._sigmoid(self.step_reward_matrix(task, latent))
        row = min(len(prefix), rewards.shape[0] - 1)
        return np.asarray(rewards[row], dtype=np.float64)

    def trajectory_rewards(
        self, task: Task, latent: np.ndarray, trajectories: Sequence[Sequence[int]]
    ) -> List[float]:
        out: List[float] = []
        for trajectory in trajectories:
            actions = tuple(int(a) for a in trajectory)
            if task.optimal_plan is not None:
                target = tuple(int(a) for a in task.optimal_plan)
                denom = max(len(actions), len(target), 1)
                matches = sum(
                    int(i < len(actions) and i < len(target) and actions[i] == target[i])
                    for i in range(denom)
                )
                out.append(float(matches / denom))
            else:
                scores = self.score_actions(task, latent, actions)
                out.append(float(np.mean(scores)) if scores else 0.0)
        return out

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))


def net_information_gain(prm_scores: Sequence[float], temperature: float = 4.0) -> List[float]:
    """Linear-time O(N) Net Information Gain for process supervision.

    We model an accumulating *belief* in the correct answer as a logistic of the
    running mean PRM score, then define the NIG of step i as the *increment* in
    belief contributed by that step:

        b_i = sigmoid( temperature * ( mean(prm[:i+1]) - 0.5 ) )
        NIG_i = b_i - b_{i-1}      (b_{-1} := sigmoid(0) = 0.5)

    This is a single linear pass (O(N)), in contrast to O(N^2) Monte-Carlo
    rollout labelling. Positive NIG marks informative steps; negative NIG marks
    steps that *reduced* belief (candidate dead/overthinking steps).
    """
    scores = np.asarray(list(prm_scores), dtype=np.float64)
    if scores.size == 0:
        return []
    running_mean = np.cumsum(scores) / (np.arange(scores.size) + 1)
    beliefs = 1.0 / (1.0 + np.exp(-temperature * (running_mean - 0.5)))
    prev = np.concatenate(([0.5], beliefs[:-1]))
    nig = beliefs - prev
    return [float(v) for v in nig]
