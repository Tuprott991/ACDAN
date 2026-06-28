"""ACDAN agent: orchestration of all modules into one rollout.

This is the integration point. A single ``run_task`` call walks the full pipeline
from the proposal:

    features
      -> latent reasoning (+ in-place TTT)            [latent_reasoning.py]
      -> core-model prior logits                      [registry.py]
      -> inertial sensing (skip planning where safe)  [inertia.py]
      -> Differentiable Textual Optimization          [dto.py]
      -> execute, score with PRM + Net-Info-Gain      [rewards.py]
      -> two-layer graph + dead-step pruning          [graph.py]
      -> self-verification & confidence calibration   [verification.py]
      -> RolloutMetrics                               [types.py]

Every stage is gated by ``config.ablation`` so ablations are pure config changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from acdan.config import ACDANConfig
from acdan.dto import DifferentiableTextOptimizer
from acdan.graph import AgenticComputationGraph, make_entropy_hook
from acdan.inertia import InertialSensor
from acdan.latent_reasoning import LatentReasoner
from acdan.registry import CoreModel
from acdan.rewards import ProcessRewardModel, net_information_gain
from acdan.types import (
    OutcomeChecker,
    Plan,
    RolloutMetrics,
    StepRecord,
    Task,
    VerificationOutcome,
)
from acdan.verification import SelfVerifier

# Inference-cost surrogates (relative "token" units per executed step).
_PLAN_STEP_COST = 1.0      # a step that required an LLM planning call
_INERTIA_STEP_COST = 0.2   # a step served by inertial sensing (no LLM call)


@dataclass
class AgentResult:
    """Rich result of one rollout (metrics + intermediate artefacts)."""

    metrics: RolloutMetrics
    plan: Plan
    steps: List[StepRecord]
    verification: VerificationOutcome
    latent: np.ndarray
    graph: Optional[AgenticComputationGraph]


class ACDANAgent:
    """The ACDAN agent. Construct via :meth:`__init__` or the evaluate harness."""

    def __init__(
        self,
        config: ACDANConfig,
        core_model: CoreModel,
        prm: ProcessRewardModel,
        latent_reasoner: LatentReasoner,
        verifier: SelfVerifier,
        inertia_sensors: Optional[Dict[str, InertialSensor]] = None,
        outcome_checker: Optional["OutcomeChecker"] = None,
    ):
        self.cfg = config
        self.core = core_model
        self.prm = prm
        self.reasoner = latent_reasoner
        self.verifier = verifier
        self.inertia_sensors = inertia_sensors or {}
        # Optional outcome-based correctness for *real* tasks. When provided, it
        # overrides the synthetic optimal-plan match. Signature:
        #     checker(task, actions) -> bool
        # (e.g. exact-match answer, unit-test pass, tool-call success).
        self.outcome_checker = outcome_checker

    # ------------------------------------------------------------- stages

    def _latent(self, task: Task) -> np.ndarray:
        ab = self.cfg.ablation
        if ab.latent_reasoning:
            trace = self.reasoner.reason(task.prompt_features, use_ttt=ab.in_place_ttt)
        else:
            trace = self.reasoner.passthrough(task.prompt_features)
        return trace.final_state

    def _inertia_plan(self, task: Task, draft: List[int]) -> Dict[int, int]:
        """Return {step_index: action} for steps served by inertial sensing."""
        if not self.cfg.ablation.inertial_sensing:
            return {}
        sensor = self.inertia_sensors.get(str(task.metadata.get("family", "")))
        if sensor is None:
            return {}
        fixed: Dict[int, int] = {}
        for h in range(task.horizon):
            prev = draft[h - 1] if h > 0 else None
            decision = sensor.query(prev)
            if decision.use_inertia and decision.predicted_action is not None:
                fixed[h] = int(decision.predicted_action)
        return fixed

    def _plan(self, task: Task, latent: np.ndarray) -> Plan:
        prior = self.core.prior_logits(task, latent)
        if not self.cfg.ablation.dto:
            return DifferentiableTextOptimizer.greedy_decode(prior)
        entropy_hook = (
            make_entropy_hook(task) if self.cfg.ablation.dependency_graph else None
        )
        dto = DifferentiableTextOptimizer(self.cfg.dto, self.prm, entropy_hook)
        # Adaptive coefficients: harder tasks get more reward/diversity pressure.
        difficulty_scale = 0.5 + task.difficulty
        return dto.optimize(task, latent, prior, difficulty_scale=difficulty_scale)

    # ------------------------------------------------------------- rollout

    def run_task(self, task: Task) -> AgentResult:
        ab = self.cfg.ablation
        latent = self._latent(task)

        # 1) Plan (DTO or greedy) and 2) overlay inertial-sensing decisions.
        plan = self._plan(task, latent)
        draft = list(plan.actions)
        fixed = self._inertia_plan(task, draft)

        actions: List[int] = []
        from_inertia: List[bool] = []
        for h in range(task.horizon):
            if h in fixed:
                actions.append(fixed[h])
                from_inertia.append(True)
            else:
                actions.append(draft[h] if h < len(draft) else 0)
                from_inertia.append(False)

        # 3) Score with PRM and assign Net Information Gain.
        prm_scores = self.prm.score_actions(task, latent, actions)
        nig = net_information_gain(prm_scores)

        # 4) Build the two-layer graph and prune dead steps (DECS).
        graph: Optional[AgenticComputationGraph] = None
        dead_idx: List[int] = []
        dep_entropy = 0.0
        if ab.dependency_graph:
            graph = AgenticComputationGraph(task, actions, prm_scores, nig)
            dep_entropy = graph.von_neumann_entropy()
            _, dead_idx = graph.prune()

        kept = [h for h in range(task.horizon) if h not in set(dead_idx)]

        # 5) Inference-cost surrogate over *kept* steps.
        token_cost = sum(
            _INERTIA_STEP_COST if from_inertia[h] else _PLAN_STEP_COST for h in kept
        )
        llm_calls = sum(1 for h in kept if not from_inertia[h])
        inertia_saved = sum(1 for h in range(task.horizon) if from_inertia[h])

        # 6) Self-verification & confidence calibration.
        mean_prm = float(np.mean(prm_scores)) if prm_scores else 0.0
        if ab.verification:
            executed_plan = Plan(
                actions=actions,
                logits=plan.logits,
                dto_steps=plan.dto_steps,
                objective_trace=plan.objective_trace,
            )
            verification = self.verifier.verify(
                task, latent, executed_plan, mean_prm, use_calibration=ab.confidence_margin
            )
        else:
            verification = VerificationOutcome(
                confidence=mean_prm, margin=0.0, independent_agreement=mean_prm,
                verified=True, abstained=False,
            )

        # 7) Correctness vs. the known optimal plan (synthetic eval only).
        correct = self._is_correct(task, actions)

        steps = [
            StepRecord(
                index=h,
                action_id=actions[h],
                action_name=task.vocab[actions[h]],
                prm_score=prm_scores[h] if h < len(prm_scores) else 0.0,
                nig=nig[h] if h < len(nig) else 0.0,
                confidence=verification.confidence,
                from_inertia=from_inertia[h],
                is_dead_step=h in set(dead_idx),
            )
            for h in range(task.horizon)
        ]

        metrics = RolloutMetrics(
            task_id=task.task_id,
            correct=correct,
            llm_calls=llm_calls,
            inertia_saved_calls=inertia_saved,
            steps=task.horizon,
            dead_steps_pruned=len(dead_idx),
            token_cost=token_cost,
            mean_prm=mean_prm,
            confidence=verification.confidence,
            abstained=verification.abstained,
            dependency_entropy=dep_entropy,
            metadata={"family": task.metadata.get("family", "?")},
        )
        return AgentResult(
            metrics=metrics, plan=plan, steps=steps,
            verification=verification, latent=latent, graph=graph,
        )

    # ------------------------------------------------------------- helpers

    def _is_correct(self, task: Task, actions: List[int]) -> bool:
        if self.outcome_checker is not None:
            return bool(self.outcome_checker(task, actions))
        if task.optimal_plan is None:
            return False
        opt = task.optimal_plan
        n = min(len(actions), len(opt))
        if n == 0:
            return False
        matches = sum(1 for i in range(n) if actions[i] == opt[i % len(opt)])
        return (matches / n) >= self.cfg.success_threshold
