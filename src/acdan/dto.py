"""Differentiable Textual Optimization (DTO).

Paper mapping (proposal sections "Tối ưu hóa Văn bản Vi phân" and "Tối ưu hóa
bậc một qua DTO"):

Instead of discrete search (Best-of-N, MCTS), ACDAN keeps a *soft logit matrix*
L over the planned action sequence and performs first-order continuous
optimisation at test time:

    minimise   J(L) = - w_ll * loglik(L)                      (core-model prior)
                       - alpha * R_prm(softmax(L))            (PRM process reward)
                       + beta  * length_penalty(softmax(L))   (anti-overthinking)
                       - gamma * vN_entropy(A_dep(softmax(L))) (path diversity)

    update     L <- L - eta * dJ/dL        (T iterations)
    decode     a*_h = argmax_v softmax(L/temp)_{h,v}

The von Neumann-entropy term couples DTO to the dependency layer of the agentic
graph (``acdan.graph``): it rewards diverse data-dependency structure and guards
against latent state-space collapse (proposal: "entropy von Neumann ... ngăn
chặn hiện tượng sụp đổ không gian trạng thái ẩn").

All gradients are analytic numpy — no autograd framework required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from acdan.config import DTOConfig
from acdan.datasets.base import MATH_DATASETS
from acdan.rewards import ProcessRewardModel
from acdan.types import Plan, Task, softmax

# A hook that maps a soft plan (probs) -> (entropy_value, d entropy / d probs).
# Supplied by the agent so DTO stays decoupled from the graph implementation.
EntropyHook = Callable[[np.ndarray], "tuple[float, np.ndarray]"]


@dataclass
class _Objective:
    value: float
    reward: float
    likelihood: float
    length_pen: float
    entropy: float
    self_consistency: float = 0.0


def _softmax_jacobian_vjp(probs_row: np.ndarray, grad_row: np.ndarray) -> np.ndarray:
    """Vector-Jacobian product through a single softmax row.

    For p = softmax(z), the VJP of upstream gradient g is:
        dz = p * (g - <p, g>)
    """
    dot = float(np.dot(probs_row, grad_row))
    return probs_row * (grad_row - dot)


class DifferentiableTextOptimizer:
    """First-order optimiser over a soft action-logit matrix."""

    def __init__(
        self,
        config: DTOConfig,
        prm: ProcessRewardModel,
        entropy_hook: Optional[EntropyHook] = None,
    ):
        self.cfg = config
        self.prm = prm
        self.entropy_hook = entropy_hook

    # ------------------------------------------------------------- objective

    def _length_penalty_and_grad(self, probs: np.ndarray):
        """Penalise *peaky repetition*: reward per-step confidence but penalise
        repeating the same action across consecutive steps (anti-overthinking).

        pen = mean_h sum_v p_h[v] * p_{h-1}[v]   (consecutive-step overlap)
        """
        H = probs.shape[0]
        if H < 2:
            return 0.0, np.zeros_like(probs)
        overlap = np.sum(probs[1:] * probs[:-1]) / (H - 1)
        grad = np.zeros_like(probs)
        grad[1:] += probs[:-1] / (H - 1)
        grad[:-1] += probs[1:] / (H - 1)
        return float(overlap), grad

    def _likelihood_and_grad(self, logits: np.ndarray, probs: np.ndarray, prior: np.ndarray):
        """Log-likelihood of the soft plan under a fixed core-model ``prior``.

        loglik = mean_h sum_v p_h[v] * log_prior_h[v]
        d loglik / d logits handled by caller via softmax VJP on ``log_prior``.
        """
        log_prior = np.log(prior + 1e-9)
        ll = float(np.sum(probs * log_prior) / probs.shape[0])
        grad_probs = log_prior / probs.shape[0]
        return ll, grad_probs

    def _self_consistency_prior_and_grad(self, task: Task, probs: np.ndarray):
        """Optional explicit count prior for math candidate selection."""
        weight = float(getattr(self.cfg, "self_consistency_weight", 0.0))
        family = str(task.metadata.get("family", ""))
        if weight == 0.0 or family not in MATH_DATASETS:
            return 0.0, np.zeros_like(probs)

        counts = task.metadata.get("candidate_counts", {}) or {}
        vals = np.asarray(
            [float(counts.get(str(name), 1.0)) for name in task.vocab],
            dtype=np.float64,
        )
        if vals.size == 0 or float(vals.max()) <= float(vals.min()):
            return 0.0, np.zeros_like(probs)

        z = np.log1p(vals)
        z = (z - float(z.mean())) / (float(z.std()) + 1e-9)
        z = np.clip(z, -2.0, 2.0)
        tiled = np.tile(z, (probs.shape[0], 1))
        prior = float(np.sum(probs * tiled) / probs.shape[0])
        return prior, tiled / probs.shape[0]

    def _evaluate(
        self,
        task: Task,
        latent: np.ndarray,
        logits: np.ndarray,
        prior: np.ndarray,
        coeffs: "tuple[float, float, float, float]",
    ):
        """Compute objective value and gradient w.r.t. logits."""
        w_ll, alpha, beta, gamma = coeffs
        probs = softmax(logits / max(self.cfg.temperature, 1e-6), axis=1)

        # --- reward term (linear in probs) ---
        reward = self.prm.score_probs(task, latent, probs)
        g_reward = self.prm.grad_wrt_probs(task, latent, probs)

        # --- likelihood prior term ---
        ll, g_ll = self._likelihood_and_grad(logits, probs, prior)

        # --- length / repetition penalty ---
        length_pen, g_len = self._length_penalty_and_grad(probs)

        # --- von Neumann entropy term (diversity) ---
        if self.entropy_hook is not None:
            entropy, g_ent = self.entropy_hook(probs)
        else:
            entropy, g_ent = 0.0, np.zeros_like(probs)

        # --- optional explicit self-consistency prior for math counts ---
        sc_prior, g_sc = self._self_consistency_prior_and_grad(task, probs)
        delta_sc = float(getattr(self.cfg, "self_consistency_weight", 0.0))

        # J = -w_ll*ll - alpha*reward + beta*len - gamma*entropy - delta_sc*SC
        value = (
            -w_ll * ll - alpha * reward + beta * length_pen
            - gamma * entropy - delta_sc * sc_prior
        )
        grad_probs = (
            -w_ll * g_ll - alpha * g_reward + beta * g_len
            - gamma * g_ent - delta_sc * g_sc
        )

        # Back-prop grad_probs through per-row softmax to logit space.
        scale = 1.0 / max(self.cfg.temperature, 1e-6)
        grad_logits = np.empty_like(logits)
        for h in range(logits.shape[0]):
            grad_logits[h] = _softmax_jacobian_vjp(probs[h], grad_probs[h]) * scale

        obj = _Objective(value=value, reward=reward, likelihood=ll,
                         length_pen=length_pen, entropy=entropy,
                         self_consistency=sc_prior)
        return obj, grad_logits

    # ----------------------------------------------------------------- solve

    def optimize(
        self,
        task: Task,
        latent: np.ndarray,
        prior_logits: np.ndarray,
        difficulty_scale: float = 1.0,
    ) -> Plan:
        """Run T first-order updates on the logit matrix and decode a plan.

        Args:
            prior_logits: initial logits from the core model, shape (H, V).
            difficulty_scale: multiplies reward/entropy weights so that harder
                tasks (proposal: "hệ số điều phối thích ứng tùy thuộc độ khó")
                spend more optimisation pressure on the PRM and diversity terms.
        """
        logits = np.array(prior_logits, dtype=np.float64, copy=True)
        prior = softmax(prior_logits, axis=1)  # fixed core-model distribution

        coeffs = (
            self.cfg.likelihood_weight,
            self.cfg.alpha_reward * difficulty_scale,
            self.cfg.beta_length,
            self.cfg.gamma_entropy * difficulty_scale,
        )

        trace: List[float] = []
        for _ in range(self.cfg.iters):
            obj, grad = self._evaluate(task, latent, logits, prior, coeffs)
            trace.append(obj.value)
            logits = logits - self.cfg.lr * grad

        # Final decode.
        probs = softmax(logits / max(self.cfg.temperature, 1e-6), axis=1)
        actions = [int(np.argmax(probs[h])) for h in range(probs.shape[0])]
        return Plan(actions=actions, logits=logits, dto_steps=self.cfg.iters,
                    objective_trace=trace)

    @staticmethod
    def greedy_decode(prior_logits: np.ndarray) -> Plan:
        """Ablation path (dto=False): decode the core-model logits greedily."""
        actions = [int(np.argmax(prior_logits[h])) for h in range(prior_logits.shape[0])]
        return Plan(actions=actions, logits=np.array(prior_logits, copy=True),
                    dto_steps=0, objective_trace=[])
