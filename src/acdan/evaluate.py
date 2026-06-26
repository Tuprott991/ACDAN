"""Evaluation harness: build an ACDAN agent and run a reproducible task suite.

This module is the offline experiment driver. It:
  1. builds the mock core model / PRM / latent reasoner from a config,
  2. fits a per-family inertial sensor from synthetic execution traces,
  3. fits the RLCM confidence probe on a held-out calibration split,
  4. runs the evaluation suite and aggregates an :class:`EvalSummary`.

Determinism: everything is seeded from ``config.seed`` so two runs with the same
config produce identical numbers — a hard requirement for rebuttal tables.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from acdan.agent import ACDANAgent
from acdan.config import ACDANConfig
from acdan.inertia import InertialSensor
from acdan.latent_reasoning import LatentReasoner
from acdan.metrics import summarize
from acdan.registry import build_core_model, build_prm
from acdan.tasks.synthetic import FEATURE_DIM, make_suite, make_traces
from acdan.types import EvalSummary, Task
from acdan.verification import ConfidenceProbe, MockIndependentVerifier, SelfVerifier

_PROBE_INPUT_DIM = 5  # [latent mean, std, norm, plan margin, mean prm]


def _build_inertia_sensors(
    config: ACDANConfig, tasks: Sequence[Task], traces_per_task: int = 4
) -> Dict[str, InertialSensor]:
    """Fit one first-order transition model per task family."""
    by_family: Dict[str, List[Task]] = {}
    for t in tasks:
        by_family.setdefault(str(t.metadata.get("family", "")), []).append(t)
    sensors: Dict[str, InertialSensor] = {}
    for family, fam_tasks in by_family.items():
        vocab_size = fam_tasks[0].vocab_size
        sensor = InertialSensor(config.inertia, vocab_size)
        traces = make_traces(fam_tasks, n_per_task=traces_per_task, seed=config.seed)
        sensor.fit(traces)
        sensors[family] = sensor
    return sensors


def _bare_agent(
    config: ACDANConfig,
    inertia_sensors: Optional[Dict[str, InertialSensor]],
    probe: Optional[ConfidenceProbe],
) -> ACDANAgent:
    core = build_core_model("mock", seed=config.seed)
    prm = build_prm("mock", seed=config.seed)
    reasoner = LatentReasoner(config.latent, feature_dim=FEATURE_DIM, seed=config.seed)
    verifier = SelfVerifier(
        config.verification,
        probe=probe,
        independent=MockIndependentVerifier(seed=config.seed),
    )
    return ACDANAgent(config, core, prm, reasoner, verifier, inertia_sensors)


def _fit_probe(config: ACDANConfig, calib_tasks: Sequence[Task],
               inertia_sensors: Dict[str, InertialSensor]) -> ConfidenceProbe:
    """Fit the RLCM confidence probe on a calibration split.

    We run a probe-less agent over the calibration tasks to collect
    (probe_features, correctness) pairs, then fit the MLP. This mirrors training
    a confidence head on rollouts where ground-truth correctness is known.
    """
    agent = _bare_agent(config, inertia_sensors, probe=None)
    X: List[np.ndarray] = []
    y: List[float] = []
    for t in calib_tasks:
        res = agent.run_task(t)
        feats = agent.verifier.feature_vector(res.latent, res.plan, res.metrics.mean_prm)
        X.append(feats)
        y.append(1.0 if res.metrics.correct else 0.0)
    probe = ConfidenceProbe(input_dim=_PROBE_INPUT_DIM, seed=config.seed)
    if len(set(y)) >= 2:  # need both classes to fit a useful probe
        probe.fit(np.array(X), np.array(y))
    return probe


def build_agent(config: ACDANConfig, n_per_family: int = 8) -> ACDANAgent:
    """Build a fully-wired ACDAN agent (inertia fit + probe fit) for a config."""
    eval_tasks = make_suite(n_per_family=n_per_family, seed=config.seed)
    calib_tasks = make_suite(n_per_family=n_per_family, seed=config.seed + 7)
    sensors = _build_inertia_sensors(config, eval_tasks)
    probe: Optional[ConfidenceProbe] = None
    if config.ablation.verification:
        probe = _fit_probe(config, calib_tasks, sensors)
    return _bare_agent(config, sensors, probe)


def run_evaluation(config: ACDANConfig, n_per_family: int = 8) -> EvalSummary:
    """Run the full offline evaluation for one config and summarise it."""
    eval_tasks = make_suite(n_per_family=n_per_family, seed=config.seed)
    agent = build_agent(config, n_per_family=n_per_family)
    rollouts = [agent.run_task(t).metrics for t in eval_tasks]
    return summarize(rollouts, config_name=config.name)
