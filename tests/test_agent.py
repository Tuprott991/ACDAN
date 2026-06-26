"""End-to-end agent / evaluation behaviour and ablation effects."""

import dataclasses

from acdan.config import ACDANConfig, AblationFlags, baseline_cot_config
from acdan.evaluate import build_agent, run_evaluation
from acdan.tasks.synthetic import make_suite


def test_determinism_same_seed_same_result():
    cfg = ACDANConfig(seed=0)
    a = run_evaluation(cfg, n_per_family=4)
    b = run_evaluation(cfg, n_per_family=4)
    assert a.to_dict() == b.to_dict()


def test_full_acdan_beats_baseline_accuracy():
    full = run_evaluation(ACDANConfig(name="full", seed=0), n_per_family=8)
    base = run_evaluation(baseline_cot_config(seed=0), n_per_family=8)
    assert full.accuracy > base.accuracy


def test_full_acdan_cheaper_than_baseline():
    full = run_evaluation(ACDANConfig(name="full", seed=0), n_per_family=8)
    base = run_evaluation(baseline_cot_config(seed=0), n_per_family=8)
    assert full.mean_token_cost < base.mean_token_cost


def test_disabling_dto_hurts_accuracy():
    full = run_evaluation(ACDANConfig(name="full", seed=0), n_per_family=8)
    flags = AblationFlags(dto=False)
    no_dto = run_evaluation(dataclasses.replace(ACDANConfig(seed=0), name="no_dto", ablation=flags),
                            n_per_family=8)
    assert no_dto.accuracy < full.accuracy


def test_disabling_inertia_raises_token_cost():
    full = run_evaluation(ACDANConfig(name="full", seed=0), n_per_family=8)
    flags = AblationFlags(inertial_sensing=False)
    no_in = run_evaluation(dataclasses.replace(ACDANConfig(seed=0), name="no_inertia", ablation=flags),
                           n_per_family=8)
    assert no_in.mean_token_cost > full.mean_token_cost


def test_disabling_verification_worsens_calibration():
    full = run_evaluation(ACDANConfig(name="full", seed=0), n_per_family=8)
    flags = AblationFlags(verification=False)
    no_v = run_evaluation(dataclasses.replace(ACDANConfig(seed=0), name="no_verification", ablation=flags),
                          n_per_family=8)
    assert no_v.ece >= full.ece


def test_rollout_artefacts_present():
    agent = build_agent(ACDANConfig(seed=0), n_per_family=2)
    task = make_suite(n_per_family=2, seed=0)[0]
    res = agent.run_task(task)
    assert len(res.steps) == task.horizon
    assert res.plan.logits.shape == (task.horizon, task.vocab_size)
    assert 0.0 <= res.verification.confidence <= 1.0
