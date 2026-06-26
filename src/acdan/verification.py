"""Self-verification & confidence calibration.

Paper mapping (proposal section "Tự Kiểm chứng Đa phương thức và Hiệu chuẩn Lòng
tin"):

  * ``ConfidenceProbe`` — an ultra-light 2-layer MLP attached to intermediate
    latent states that predicts P(final answer correct | reasoning prefix).
    This is the RLCM confidence probe.
  * ``margin_score`` — margin-based confidence (top-1 minus top-2 of the answer
    distribution), used to coordinate adaptive search / risk control.
  * ``IndependentVerifier`` — the "Independent Question Asking" interface: query
    evidence from an external tool / sandbox *without* conditioning on the
    current reasoning context, removing confirmation bias. A mock noisy-oracle
    backend ships for offline use; a real TIM-PRM tool can be plugged in.
  * ``expected_calibration_error`` — standard ECE for reporting calibration.

``SelfVerifier`` combines the probe, the margin, and the independent verdict into
a single accept / abstain decision.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

import numpy as np

from acdan.config import VerificationConfig
from acdan.types import Plan, Task, VerificationOutcome, stable_seed


# --------------------------------------------------------------------------
# Confidence probe (RLCM)
# --------------------------------------------------------------------------

class ConfidenceProbe:
    """Two-layer MLP probe: hidden features -> P(correct).

    Trained by logistic regression (binary cross-entropy) via plain gradient
    descent so it has no framework dependency. In the offline demo it is fit on a
    synthetic calibration set assembled by the evaluation harness.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        s1 = 1.0 / np.sqrt(input_dim)
        s2 = 1.0 / np.sqrt(hidden_dim)
        self.W1 = rng.normal(0.0, s1, size=(hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0.0, s2, size=hidden_dim)
        self.b2 = 0.0
        self._fitted = False

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _forward(self, X: np.ndarray):
        Z1 = X @ self.W1.T + self.b1
        H = np.tanh(Z1)
        logit = H @ self.W2 + self.b2
        p = self._sigmoid(logit)
        return p, H

    def predict(self, x: np.ndarray) -> float:
        p, _ = self._forward(np.asarray(x, dtype=np.float64).reshape(1, -1))
        return float(p[0])

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.1) -> "ConfidenceProbe":
        """Fit the probe with full-batch gradient descent on BCE loss."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = X.shape[0]
        if n == 0:
            return self
        for _ in range(epochs):
            p, H = self._forward(X)
            err = (p - y) / n                  # dL/dlogit for BCE
            # output layer
            gW2 = H.T @ err
            gb2 = float(np.sum(err))
            # hidden layer
            dH = np.outer(err, self.W2) * (1.0 - H ** 2)
            gW1 = dH.T @ X
            gb1 = dH.sum(axis=0)
            self.W2 -= lr * gW2
            self.b2 -= lr * gb2
            self.W1 -= lr * gW1
            self.b1 -= lr * gb1
        self._fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# --------------------------------------------------------------------------
# Margin-based confidence
# --------------------------------------------------------------------------

def margin_score(plan: Plan, floor: float = 0.05) -> float:
    """Mean top1-minus-top2 margin of the per-step action distributions.

    A confident, well-separated plan has a large margin; a hedging plan has a
    small one. Floored to avoid degenerate zeros.
    """
    probs = plan.probs
    if probs.shape[0] == 0:
        return floor
    margins = []
    for row in probs:
        top2 = np.sort(row)[-2:]
        m = float(top2[-1] - top2[0]) if row.shape[0] >= 2 else float(top2[-1])
        margins.append(m)
    return max(float(np.mean(margins)), floor)


# --------------------------------------------------------------------------
# Independent verification (Independent Question Asking)
# --------------------------------------------------------------------------

class IndependentVerifier(Protocol):
    """Query evidence independently of the reasoning context.

    Returns an agreement score in [0, 1]: how strongly independent evidence
    supports the agent's plan being correct.
    """

    def agreement(self, task: Task, plan: Plan) -> float:
        ...


class MockIndependentVerifier:
    """Offline noisy-oracle verifier.

    It compares the executed plan against the task's known optimal plan (the
    "external evidence") with calibrated noise, *without* seeing the agent's
    internal reasoning — modelling a tool/sandbox check. For tasks without ground
    truth it falls back to the plan's own self-consistency (low information),
    which is honestly a weak signal and is documented as such.
    """

    def __init__(self, noise: float = 0.15, seed: int = 0):
        self.noise = noise
        self._seed = seed

    def agreement(self, task: Task, plan: Plan) -> float:
        if task.optimal_plan is None:
            # No external ground truth: report low-confidence neutral evidence.
            return 0.5
        opt = task.optimal_plan
        n = min(len(plan.actions), len(opt))
        if n == 0:
            return 0.5
        matches = sum(1 for i in range(n) if plan.actions[i] == opt[i % len(opt)])
        frac = matches / n
        rng = np.random.default_rng(stable_seed(task.task_id, "verify"))
        noisy = frac + rng.normal(0.0, self.noise)
        return float(np.clip(noisy, 0.0, 1.0))


# --------------------------------------------------------------------------
# Calibration metric
# --------------------------------------------------------------------------

def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Standard Expected Calibration Error (ECE) with equal-width bins."""
    conf = np.asarray(list(confidences), dtype=np.float64)
    acc = np.asarray([1.0 if c else 0.0 for c in correct], dtype=np.float64)
    if conf.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = conf.size
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_conf = conf[mask].mean()
        bin_acc = acc[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


# --------------------------------------------------------------------------
# Combined self-verifier
# --------------------------------------------------------------------------

class SelfVerifier:
    """Combine probe confidence, margin, and independent evidence into a verdict."""

    def __init__(
        self,
        config: VerificationConfig,
        probe: Optional[ConfidenceProbe],
        independent: Optional[IndependentVerifier],
    ):
        self.cfg = config
        self.probe = probe
        self.independent = independent

    def feature_vector(self, latent: np.ndarray, plan: Plan, mean_prm: float) -> np.ndarray:
        """Probe input: [latent summary stats, margin, mean PRM]."""
        latent = np.asarray(latent, dtype=np.float64).reshape(-1)
        summary = np.array([
            latent.mean(),
            latent.std(),
            float(np.linalg.norm(latent) / (np.sqrt(latent.size) + 1e-9)),
        ])
        return np.concatenate([summary, [margin_score(plan, self.cfg.margin_floor), mean_prm]])

    def verify(
        self,
        task: Task,
        latent: np.ndarray,
        plan: Plan,
        mean_prm: float,
        use_calibration: bool = True,
    ) -> VerificationOutcome:
        """Produce the calibrated verification outcome for a plan."""
        margin = margin_score(plan, self.cfg.margin_floor)

        # Base confidence: probe if available & fitted, else margin-derived.
        if self.probe is not None and self.probe.is_fitted:
            raw_conf = self.probe.predict(self.feature_vector(latent, plan, mean_prm))
        else:
            raw_conf = float(np.clip(0.5 + 0.5 * (margin - 0.1), 0.0, 1.0))

        # Independent Question Asking.
        if self.independent is not None:
            agree = self.independent.agreement(task, plan)
        else:
            agree = raw_conf  # no independent evidence -> fall back to self

        # RLCM confidence-margin blend.
        if use_calibration:
            w = self.cfg.independent_weight
            confidence = (1.0 - w) * raw_conf + w * agree
        else:
            confidence = raw_conf

        verified = confidence >= self.cfg.accept_threshold
        abstained = confidence < self.cfg.abstain_threshold
        return VerificationOutcome(
            confidence=float(confidence),
            margin=float(margin),
            independent_agreement=float(agree),
            verified=bool(verified),
            abstained=bool(abstained),
        )
