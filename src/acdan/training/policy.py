"""Parametric (H, V) softmax policy with analytic gradients.

logits = (W @ x + b).reshape(H, V);  pi(a|x) = softmax(logits, axis=1)

This is the trainable policy PS-GRPO optimises. It is deliberately small and
linear so the whole trainer stays numpy-only and finite-difference-testable. The
same advantages drive a real LLM action head on the VM (see package docstring).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from acdan.types import softmax


class PolicyHead:
    def __init__(self, feature_dim: int, horizon: int, vocab_size: int,
                 seed: int = 0, init_scale: float = 0.01):
        self.D, self.H, self.V = feature_dim, horizon, vocab_size
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, init_scale, size=(horizon * vocab_size, feature_dim))
        self.b = np.zeros(horizon * vocab_size)

    # --------------------------------------------------------------- forward

    def logits(self, x: np.ndarray) -> np.ndarray:
        return (self.W @ x + self.b).reshape(self.H, self.V)

    def probs(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.logits(x), axis=1)

    def sample(self, x: np.ndarray, rng: np.random.Generator) -> Tuple[List[int], np.ndarray]:
        p = self.probs(x)
        actions = [int(rng.choice(self.V, p=p[h])) for h in range(self.H)]
        return actions, p

    def greedy(self, x: np.ndarray) -> List[int]:
        p = self.probs(x)
        return [int(np.argmax(p[h])) for h in range(self.H)]

    # ---------------------------------------------------------------- params

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.W.copy(), self.b.copy()

    def set_params(self, params: Tuple[np.ndarray, np.ndarray]) -> None:
        self.W = np.array(params[0], copy=True)
        self.b = np.array(params[1], copy=True)

    # --------------------------------------------------------- grad helpers

    def logprobs_for(self, x: np.ndarray, actions: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (per-step logprob of chosen actions, probs matrix)."""
        p = self.probs(x)
        lp = np.array([np.log(p[h, actions[h]] + 1e-12) for h in range(self.H)])
        return lp, p

    def apply_logit_grad(self, x: np.ndarray, dlogits: np.ndarray,
                         lr: float) -> None:
        """Gradient *ascent* step: params += lr * d(objective)/d(params).

        ``dlogits`` is d(objective)/d(logits), shape (H, V). Since
        logits = W @ x + b, the parameter gradients are outer(dlogits_flat, x)
        and dlogits_flat.
        """
        g = dlogits.reshape(-1)
        self.W += lr * np.outer(g, x)
        self.b += lr * g
