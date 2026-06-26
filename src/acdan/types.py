"""Shared data structures used across ACDAN modules.

These are intentionally plain ``dataclasses`` (no framework objects) so that the
data flowing between modules is transparent and easy to serialise for rebuttal
artefacts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# An outcome-based correctness function for real tasks (overrides optimal-plan
# matching): ``checker(task, executed_action_ids) -> bool``.
OutcomeChecker = Callable[["Task", List[int]], bool]


def stable_seed(*parts: object) -> int:
    """Deterministic 32-bit seed from arbitrary parts.

    Uses BLAKE2b rather than the built-in ``hash()``, which is salted per process
    (``PYTHONHASHSEED``) and would make results non-reproducible *across runs*.
    This is the seed source for every per-task RNG, guaranteeing identical numbers
    in every process — a hard requirement for reproducible research artefacts.
    """
    key = "␟".join(repr(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2 ** 32)

# A "token" in ACDAN's action space is an *action / tool id*, not a sub-word.
# We keep the vocabulary abstract: tasks define their own action vocabulary.
ActionId = int


@dataclass(frozen=True)
class Task:
    """A single synthetic (or real, via adapter) agentic task.

    Attributes:
        task_id: Stable identifier for reproducibility.
        prompt_features: Dense feature vector describing the task input. In the
            offline demo this is a synthetic embedding; with a real backend it is
            produced by an encoder adapter (see ``acdan.registry``).
        vocab: Ordered list of human-readable action / tool names.
        optimal_plan: Ground-truth optimal action sequence (known for synthetic
            tasks; ``None`` for real tasks). Used only for evaluation, never by
            the agent at inference time.
        horizon: Planning horizon (number of action steps).
        difficulty: Scalar in [0, 1]; drives ACDAN's adaptive coefficients.
        metadata: Free-form extra info (task family, seed, etc.).
    """

    task_id: str
    prompt_features: np.ndarray
    vocab: Tuple[str, ...]
    horizon: int
    difficulty: float = 0.5
    optimal_plan: Optional[Tuple[ActionId, ...]] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


@dataclass
class StepRecord:
    """One executed step of an agent rollout (a node in the execution layer)."""

    index: int
    action_id: ActionId
    action_name: str
    # Per-step process reward (PRM) score in [0, 1].
    prm_score: float = 0.0
    # Net Information Gain assigned to this step (see rewards.NetInformationGain).
    nig: float = 0.0
    # Confidence predicted by the probe from the latent prefix at this step.
    confidence: float = 0.0
    # Whether this step's action was emitted by inertial sensing (no LLM plan).
    from_inertia: bool = False
    # Whether the graph pruner flagged this as a dead step (redundant).
    is_dead_step: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class Plan:
    """A decoded action plan plus the soft logits it came from."""

    actions: List[ActionId]
    logits: np.ndarray  # shape (horizon, vocab_size)
    # Number of DTO gradient steps actually performed (0 if DTO disabled).
    dto_steps: int = 0
    # Objective value trajectory across DTO iterations (for plots / rebuttal).
    objective_trace: List[float] = field(default_factory=list)

    @property
    def probs(self) -> np.ndarray:
        """Row-wise softmax of the (possibly refined) logits."""
        z = self.logits - self.logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


@dataclass
class VerificationOutcome:
    """Result of the self-verification / calibration stage."""

    confidence: float            # calibrated answer confidence in [0, 1]
    margin: float                # margin-based confidence score
    independent_agreement: float # agreement with the independent verifier in [0, 1]
    verified: bool               # final accept/reject decision
    abstained: bool = False      # True if confidence below abstention threshold


@dataclass
class RolloutMetrics:
    """Per-task metrics produced by a single agent rollout."""

    task_id: str
    correct: bool
    # Surrogate inference cost (counts "LLM planning calls"); lower is better.
    llm_calls: int
    # LLM planning calls saved by inertial sensing.
    inertia_saved_calls: int
    # Number of action steps in the executed plan.
    steps: int
    # Steps pruned as dead by the dependency-graph analysis.
    dead_steps_pruned: int
    # Token-cost surrogate (proportional to executed, non-pruned steps).
    token_cost: float
    # Mean process reward over executed steps.
    mean_prm: float
    # Final calibrated confidence and whether the agent abstained.
    confidence: float
    abstained: bool
    # von Neumann entropy of the dependency layer (path diversity).
    dependency_entropy: float
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class EvalSummary:
    """Aggregate metrics over a suite of tasks."""

    n_tasks: int
    accuracy: float
    coverage: float                 # fraction of non-abstained answers
    selective_accuracy: float       # accuracy among non-abstained answers
    mean_token_cost: float
    mean_llm_calls: float
    mean_inertia_saved: float
    mean_dead_steps_pruned: float
    mean_prm: float
    ece: float                      # expected calibration error
    mean_dependency_entropy: float
    config_name: str = "default"
    per_task: List[RolloutMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """JSON-serialisable summary (excludes per-task detail by default)."""
        return {
            "config_name": self.config_name,
            "n_tasks": self.n_tasks,
            "accuracy": self.accuracy,
            "coverage": self.coverage,
            "selective_accuracy": self.selective_accuracy,
            "mean_token_cost": self.mean_token_cost,
            "mean_llm_calls": self.mean_llm_calls,
            "mean_inertia_saved": self.mean_inertia_saved,
            "mean_dead_steps_pruned": self.mean_dead_steps_pruned,
            "mean_prm": self.mean_prm,
            "ece": self.ece,
            "mean_dependency_entropy": self.mean_dependency_entropy,
        }


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def one_hot(indices: Sequence[int], depth: int) -> np.ndarray:
    """One-hot encode a sequence of indices to shape (len(indices), depth)."""
    out = np.zeros((len(indices), depth), dtype=np.float64)
    for i, idx in enumerate(indices):
        out[i, int(idx)] = 1.0
    return out
