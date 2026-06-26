"""Configuration and ablation flags for ACDAN.

Every architectural module can be independently switched on/off through
``AblationFlags`` so that ablation studies (a reviewer favourite) are a one-line
config change. Hyper-parameters live in ``ACDANConfig`` and map directly to the
symbols in ``docs/math.md``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:  # PyYAML is a declared dependency but we degrade gracefully for import.
    import yaml
except Exception:  # pragma: no cover - only hit if PyYAML missing
    yaml = None  # type: ignore


@dataclass
class AblationFlags:
    """Toggle each ACDAN module independently.

    Setting a flag to ``False`` replaces the module with a faithful baseline:
      * latent_reasoning=False -> use raw task features (no recurrent depth).
      * in_place_ttt=False     -> skip test-time weight adaptation.
      * dto=False              -> greedy single-path decode (no logit descent).
      * dependency_graph=False -> no graph analysis / no dead-step pruning.
      * inertial_sensing=False -> always "call the LLM" to plan (no tool reuse).
      * verification=False     -> accept the answer without checking.
      * confidence_margin=False-> uncalibrated raw confidence.
    """

    latent_reasoning: bool = True
    in_place_ttt: bool = True
    dto: bool = True
    dependency_graph: bool = True
    inertial_sensing: bool = True
    verification: bool = True
    confidence_margin: bool = True

    def describe(self) -> str:
        on = [k for k, v in dataclasses.asdict(self).items() if v]
        off = [k for k, v in dataclasses.asdict(self).items() if not v]
        return f"enabled={on} disabled={off}"


@dataclass
class LatentConfig:
    """Latent-space reasoning + in-place TTT (paper section: Lập luận ẩn)."""

    latent_dim: int = 32
    recurrent_depth: int = 4          # number of unrolled recurrent-block steps
    ttt_steps: int = 3                # in-place test-time training steps
    ttt_lr: float = 0.05              # TTT learning rate


@dataclass
class DTOConfig:
    """Differentiable Textual Optimization (paper section: DTO)."""

    iters: int = 40                   # number of first-order logit updates (T)
    lr: float = 0.8                   # test-time learning rate (eta)
    # Adaptive objective coefficients. Effective values are scaled by task
    # difficulty inside the agent (see math.md, "hệ số điều phối thích ứng").
    alpha_reward: float = 2.0         # weight on PRM process reward
    beta_length: float = 0.02         # length / repetition penalty
    gamma_entropy: float = 0.01       # von Neumann entropy bonus (diversity)
    likelihood_weight: float = 0.10   # weight on core-model log-likelihood prior
    temperature: float = 1.0          # softmax temperature for decoding


@dataclass
class InertiaConfig:
    """Tool Usage Inertia / Inertial Sensing (paper section: Quán tính)."""

    threshold: float = 0.65           # high-inertia activation threshold
    smoothing: float = 0.5            # Laplace smoothing for transition counts
    min_observations: int = 3         # min transitions before trusting inertia


@dataclass
class VerificationConfig:
    """Self-verification & confidence calibration (RLCM / TIM-PRM)."""

    accept_threshold: float = 0.5     # min calibrated confidence to accept
    abstain_threshold: float = 0.35   # below this the agent abstains
    independent_weight: float = 0.5   # blend of independent verifier vs. probe
    margin_floor: float = 0.05        # numerical floor for the margin score


@dataclass
class ACDANConfig:
    """Top-level ACDAN configuration."""

    name: str = "default"
    seed: int = 0
    # A plan is scored "correct" if at least this fraction of its positions match
    # the task's optimal plan (synthetic-task evaluation only).
    success_threshold: float = 0.6
    ablation: AblationFlags = field(default_factory=AblationFlags)
    latent: LatentConfig = field(default_factory=LatentConfig)
    dto: DTOConfig = field(default_factory=DTOConfig)
    inertia: InertiaConfig = field(default_factory=InertiaConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)

    # ---- (de)serialisation -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ACDANConfig":
        """Build a config from a (possibly partial) nested dict."""
        data = dict(data or {})
        sub = {
            "ablation": (AblationFlags, data.pop("ablation", None)),
            "latent": (LatentConfig, data.pop("latent", None)),
            "dto": (DTOConfig, data.pop("dto", None)),
            "inertia": (InertiaConfig, data.pop("inertia", None)),
            "verification": (VerificationConfig, data.pop("verification", None)),
        }
        kwargs: Dict[str, Any] = {}
        for key, (klass, raw) in sub.items():
            kwargs[key] = klass(**raw) if isinstance(raw, dict) else klass()
        kwargs.update(data)  # remaining scalars: name, seed, ...
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "ACDANConfig":
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load YAML configs.")
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def save_yaml(self, path: str) -> None:
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to save YAML configs.")
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def baseline_cot_config(seed: int = 0) -> ACDANConfig:
    """A 'Chain-of-Thought-like' baseline: every ACDAN module disabled.

    Corresponds to the leftmost column of the comparison table in the proposal
    (Lập luận Chuỗi Tuần tự / CoT).
    """
    return ACDANConfig(
        name="baseline_cot",
        seed=seed,
        ablation=AblationFlags(
            latent_reasoning=False,
            in_place_ttt=False,
            dto=False,
            dependency_graph=False,
            inertial_sensing=False,
            verification=False,
            confidence_margin=False,
        ),
    )
