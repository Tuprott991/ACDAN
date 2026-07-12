"""Autoregressive lattice DTO: gradients, credit assignment, and integration."""

from __future__ import annotations

import numpy as np

from acdan.config import DTOConfig, SequenceDTOConfig
from acdan.dto import DifferentiableTextOptimizer
from acdan.registry import MockCoreModel
from acdan.rewards import MockProcessReward
from acdan.run_experiment import build_parser, run
from acdan.sequence_dto import AutoregressiveLatticeOptimizer
from acdan.types import Task


class PrefixCore:
    def __init__(self, target: tuple[int, ...]):
        self.target = target
        self.calls = 0

    def prior_logits(self, task, latent):
        logits = np.zeros((task.horizon, task.vocab_size), dtype=np.float64)
        logits[:, 0] = 2.0
        return logits

    def conditional_prior_logits(self, task, latent, prefix, include_stop=False):
        self.calls += 1
        row = np.full(task.vocab_size, -1.0, dtype=np.float64)
        depth = len(prefix)
        if depth < len(self.target):
            row[self.target[depth]] = 2.0
        stop = 3.0 if depth >= len(self.target) else -4.0
        return np.concatenate([row, [stop]]) if include_stop else row


class PrefixReward:
    def __init__(self, target: tuple[int, ...]):
        self.target = target
        self.extension_calls = 0
        self.trajectory_calls = 0

    def extension_rewards(self, task, latent, prefix):
        self.extension_calls += 1
        rewards = np.full(task.vocab_size, 0.1, dtype=np.float64)
        if len(prefix) < len(self.target):
            rewards[self.target[len(prefix)]] = 0.9
        return rewards

    def trajectory_rewards(self, task, latent, trajectories):
        self.trajectory_calls += 1
        return [1.0 if tuple(path) == self.target else 0.0 for path in trajectories]

    def score_actions(self, task, latent, actions):
        return [1.0 if i < len(self.target) and action == self.target[i] else 0.0
                for i, action in enumerate(actions)]


def _task(target=(0, 1), horizon=2):
    return Task(
        task_id="sequence-test",
        prompt_features=np.zeros(4),
        vocab=("tool_a", "tool_b", "tool_c"),
        horizon=horizon,
        optimal_plan=tuple(target),
        metadata={"family": "bfcl", "prompt": "perform all requested operations"},
    )


def _config(**kwargs):
    values = dict(
        enabled=True,
        max_steps=3,
        min_steps=1,
        beam_width=4,
        samples=6,
        iters=40,
        lr=0.5,
        kl_weight=0.05,
        step_reward_weight=1.0,
        trajectory_reward_weight=3.0,
        self_consistency_weight=0.0,
        length_cost=0.0,
    )
    values.update(kwargs)
    return SequenceDTOConfig(**values)


def test_lattice_gradient_matches_finite_difference():
    task = _task()
    optimizer = AutoregressiveLatticeOptimizer(
        _config(samples=0, iters=1), PrefixCore((0, 1)), PrefixReward((0, 1))
    )
    lattice = optimizer.build_lattice(task, np.zeros(4))
    objective, gradients = optimizer._evaluate_lattice(lattice)
    root = lattice.nodes[()]
    eps = 1e-6
    root.offsets[0] += eps
    perturbed, _ = optimizer._evaluate_lattice(lattice)
    root.offsets[0] -= eps
    finite_difference = (perturbed - objective) / eps
    assert np.isclose(finite_difference, gradients[()][0], rtol=1e-3, atol=1e-4)


def test_sequence_dto_uses_prefix_and_trajectory_credit():
    task = _task(target=(0, 1), horizon=2)
    core = PrefixCore((0, 1))
    reward = PrefixReward((0, 1))
    plan = AutoregressiveLatticeOptimizer(_config(), core, reward).optimize(
        task, np.zeros(4)
    )
    assert plan.actions == [0, 1]
    assert plan.metadata["planner"] == "autoregressive_lattice_dto"
    assert plan.metadata["expanded_nodes"] > 1
    assert plan.objective_trace[-1] >= plan.objective_trace[0]


def test_sequence_dto_allows_legitimate_repeated_tool_calls():
    task = _task(target=(0, 0), horizon=2)
    plan = AutoregressiveLatticeOptimizer(
        _config(), PrefixCore((0, 0)), PrefixReward((0, 0))
    ).optimize(task, np.zeros(4))
    assert plan.actions == [0, 0]


def test_sequence_dto_stop_is_not_bound_to_task_horizon():
    task = _task(target=(0, 1), horizon=1)
    plan = AutoregressiveLatticeOptimizer(
        _config(max_steps=3), PrefixCore((0, 1)), PrefixReward((0, 1))
    ).optimize(task, np.zeros(4))
    assert plan.actions == [0, 1]
    assert len(plan.actions) != task.horizon


def test_every_root_tool_has_a_complete_candidate():
    task = _task()
    optimizer = AutoregressiveLatticeOptimizer(
        _config(samples=0), PrefixCore((0, 1)), PrefixReward((0, 1))
    )
    lattice = optimizer.build_lattice(task, np.zeros(4))
    roots = {path[0] for path in lattice.candidates}
    assert roots == set(range(task.vocab_size))


def test_optimization_iterations_do_not_add_backend_calls():
    task = _task()
    core = PrefixCore((0, 1))
    reward = PrefixReward((0, 1))
    plan = AutoregressiveLatticeOptimizer(
        _config(iters=60), core, reward
    ).optimize(task, np.zeros(4))
    assert core.calls == plan.metadata["policy_score_batches"]
    assert reward.extension_calls + 1 == plan.metadata["prm_score_batches"]
    assert reward.trajectory_calls == 1


def test_one_step_sequence_mode_preserves_dense_dto_choice():
    task = _task(target=(1,), horizon=1)
    core = MockCoreModel(seed=0)
    prm = MockProcessReward(seed=0)
    latent = np.ones(8)
    independent = DifferentiableTextOptimizer(DTOConfig(), prm).optimize(
        task, latent, core.prior_logits(task, latent)
    )
    sequence = AutoregressiveLatticeOptimizer(
        _config(max_steps=1), core, prm
    ).optimize(task, latent)
    assert sequence.actions == independent.actions


def test_no_dto_sequence_ablation_follows_frozen_prior():
    task = _task(target=(1,), horizon=1)
    core = PrefixCore((1,))
    core.prior_logits = lambda task, latent: np.asarray([[3.0, 0.0, 0.0]])
    core.conditional_prior_logits = lambda task, latent, prefix, include_stop=False: (
        np.asarray([3.0, 0.0, 0.0, -4.0])
        if include_stop else np.asarray([3.0, 0.0, 0.0])
    )
    optimizer = AutoregressiveLatticeOptimizer(
        _config(max_steps=1), core, PrefixReward((1,))
    )
    no_dto = optimizer.optimize(task, np.zeros(4), use_dto=False)
    assert no_dto.actions == [0]


def test_runner_autoregressive_mode_executes_variable_plan(tmp_path):
    path = tmp_path / "bfcl.jsonl"
    path.write_text(
        '{"task_id":"t1","prompt":"do A then B","tools":["a","b"],'
        '"gold":["a","b"],"horizon":2}\n',
        encoding="utf-8",
    )
    args = build_parser().parse_args([
        "--method", "acdan",
        "--dataset", "bfcl",
        "--data-path", str(path),
        "--policy", "mock",
        "--prm", "mock",
        "--dto-mode", "autoregressive",
        "--sequence-max-steps", "3",
        "--sequence-iters", "30",
        "--no-latent",
        "--disable", "no_graph,no_inertia,no_verification",
        "--save-per-task",
    ])
    result = run(args)
    assert result["summary"]["dto_mode"] == "autoregressive"
    assert result["summary"]["mean_expanded_nodes"] > 0
    assert result["summary"]["mean_trajectory_candidates"] > 0
