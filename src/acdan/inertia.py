"""Tool Usage Inertia / Inertial Sensing.

Paper mapping (proposal section "Phán đoán Quán tính", AutoTool):

ACDAN builds a probabilistic tool-transition graph from historical execution
traces. Before each action, the inertial sensor checks whether the current
tool-transition lies in a *high-inertia* region. If so, the agent fires the tool
and fills parameters directly — *skipping the expensive LLM planning call*. This
is the mechanism behind the proposal's >50% inference-token reduction on
repetitive workflows.

The transition model is a simple first-order Markov chain over action ids with
Laplace smoothing, fit from observed traces. It is intentionally trivial to
audit and fully offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from acdan.config import InertiaConfig

# A trace is a sequence of executed action ids.
Trace = Sequence[int]


@dataclass
class InertiaDecision:
    """Outcome of an inertial-sensing query for one step."""

    use_inertia: bool          # True -> skip LLM planning, reuse the transition
    predicted_action: Optional[int]
    inertia_score: float       # max transition probability from current state
    observations: int          # how many transitions observed from this state


class InertialSensor:
    """First-order Markov tool-transition model with inertia gating."""

    def __init__(self, config: InertiaConfig, vocab_size: int):
        self.cfg = config
        self.vocab_size = vocab_size
        # counts[s, s'] = observed transitions s -> s'
        self.counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        # start[s] = observed first-actions (for the very first step)
        self.start = np.zeros(vocab_size, dtype=np.float64)

    # ------------------------------------------------------------- fitting

    def fit(self, traces: Sequence[Trace]) -> "InertialSensor":
        """Accumulate transition statistics from historical traces."""
        for tr in traces:
            tr = [int(a) for a in tr if 0 <= int(a) < self.vocab_size]
            if not tr:
                continue
            self.start[tr[0]] += 1.0
            for a, b in zip(tr[:-1], tr[1:]):
                self.counts[a, b] += 1.0
        return self

    def observe(self, prev_action: int, action: int) -> None:
        """Online update: record a single observed transition."""
        if 0 <= prev_action < self.vocab_size and 0 <= action < self.vocab_size:
            self.counts[prev_action, action] += 1.0

    # ----------------------------------------------------------- inference

    def _transition_probs(self, state: int) -> Tuple[np.ndarray, int]:
        """Smoothed P(next | state) and the raw observation count."""
        row = self.counts[state]
        n_obs = int(row.sum())
        smoothed = row + self.cfg.smoothing
        probs = smoothed / smoothed.sum()
        return probs, n_obs

    def query(self, prev_action: Optional[int]) -> InertiaDecision:
        """Decide whether to act by inertia given the previous action.

        Inertia fires only when (a) we have seen enough transitions from this
        state (``min_observations``) and (b) the most likely next action exceeds
        the ``threshold``. Otherwise the agent must plan (an "LLM call").
        """
        if prev_action is None:
            # First step: use start distribution, but never skip planning on the
            # very first action (no transition context yet).
            return InertiaDecision(False, None, 0.0, int(self.start.sum()))

        probs, n_obs = self._transition_probs(int(prev_action))
        best = int(np.argmax(probs))
        score = float(probs[best])
        fire = (n_obs >= self.cfg.min_observations) and (score >= self.cfg.threshold)
        return InertiaDecision(
            use_inertia=fire,
            predicted_action=best if fire else None,
            inertia_score=score,
            observations=n_obs,
        )

    def transition_matrix(self) -> np.ndarray:
        """Full smoothed transition matrix (for inspection / plots)."""
        smoothed = self.counts + self.cfg.smoothing
        return smoothed / smoothed.sum(axis=1, keepdims=True)
