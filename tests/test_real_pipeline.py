"""The real-experiment pipeline must run offline with mock backends (no GPU)."""

import numpy as np

from acdan.agent import ACDANAgent
from acdan.baselines import (
    adaptive_self_consistency,
    best_of_n_prm,
    cot_greedy,
    reasoning_as_planning,
    s1_budget_forcing,
    self_consistency,
    self_refine,
    tree_of_thoughts,
)
from acdan.config import ACDANConfig
from acdan.datasets.base import (
    SyntheticRawDataset,
    outcome_exact,
    outcome_tool_sequence,
    build_dataset,
    build_outcome_checker,
)
from acdan.latent_reasoning import LatentReasoner
from acdan.registry import build_core_model, build_prm
from acdan.run_experiment import _to_task, build_parser, run
from acdan.verification import SelfVerifier


def _task(raw):
    from acdan.backends.encoder import HashingEncoder
    enc = HashingEncoder(dim=32)
    return _to_task(raw, enc.encode(raw.prompt))


def test_outcome_exact_and_tool_checkers():
    raw = next(iter(SyntheticRawDataset(n=1, k=4, seed=0).tasks()))
    task = _task(raw)
    gold_idx = task.vocab.index(raw.gold)
    assert outcome_exact(task, [gold_idx]) is True
    assert outcome_exact(task, [(gold_idx + 1) % 4]) is False


def test_outcome_exact_does_not_last_number_match_malformed_answers():
    from acdan.datasets.base import RawTask
    raw = RawTask("m", "solve", ("wrong trail 42", "42"), horizon=1, gold="42", family="math")
    task = _task(raw)
    assert outcome_exact(task, [1]) is True
    assert outcome_exact(task, [0]) is False


def test_outcome_exact_handles_boxed_decimal_equivalence():
    from acdan.datasets.base import RawTask
    raw = RawTask("m", "solve", (r"\boxed{2.00}",), horizon=1, gold="2", family="math")
    task = _task(raw)
    assert outcome_exact(task, [0]) is True


def test_outcome_exact_handles_repeated_answer_marker_noise():
    from acdan.datasets.base import RawTask
    raw = RawTask(
        "m",
        "solve",
        ("The answer is 2.00. >>>> The answer is 2.00.",),
        horizon=1,
        gold="2",
        family="math",
    )
    task = _task(raw)
    assert outcome_exact(task, [0]) is True


def test_outcome_tool_sequence():
    from acdan.datasets.base import RawTask
    raw = RawTask("t", "do it", ("a", "b", "c"), horizon=2, gold=["a", "b"], family="bfcl")
    task = _task(raw)
    assert outcome_tool_sequence(task, [0, 1]) is True
    assert outcome_tool_sequence(task, [1, 0]) is False


def test_agent_uses_outcome_checker():
    raw = next(iter(SyntheticRawDataset(n=1, k=4, seed=1).tasks()))
    task = _task(raw)
    cfg = ACDANConfig(seed=0)
    core = build_core_model("mock", seed=0)
    prm = build_prm("mock", seed=0)
    rea = LatentReasoner(cfg.latent, feature_dim=task.prompt_features.size, seed=0)
    ver = SelfVerifier(cfg.verification, probe=None, independent=None)
    agent = ACDANAgent(cfg, core, prm, rea, ver, outcome_checker=outcome_exact)
    res = agent.run_task(task)
    # correctness now comes from the checker, not optimal_plan internals
    assert res.metrics.correct == outcome_exact(task, [s.action_id for s in res.steps])


def test_baselines_run_offline():
    raw = next(iter(SyntheticRawDataset(n=1, k=4, seed=2).tasks()))
    task = _task(raw)
    cfg = ACDANConfig(seed=0)
    core = build_core_model("mock", seed=0)
    prm = build_prm("mock", seed=0)
    rea = LatentReasoner(cfg.latent, feature_dim=task.prompt_features.size, seed=0)
    latent = rea.reason(task.prompt_features).final_state
    for br in (cot_greedy(core, task, latent),
               self_consistency(core, task, latent, n=4),
               best_of_n_prm(core, prm, task, latent, n=4)):
        assert len(br.actions) == task.horizon
        assert "policy_passes" in br.cost


def test_adaptive_self_consistency_early_stops_grouped_candidates():
    from acdan.datasets.base import RawTask
    raw = RawTask(
        "m",
        "solve",
        ("right", "wrong"),
        horizon=1,
        gold="right",
        family="gsm8k",
        metadata={
            "candidate_counts": {"right": 6, "wrong": 2},
            "candidate_first_indices": {"right": 0, "wrong": 1},
        },
    )
    task = _task(raw)
    cfg = ACDANConfig(seed=0)
    core = build_core_model("mock", seed=0)
    latent = np.zeros(task.prompt_features.size)
    br = adaptive_self_consistency(core, task, latent, n=8, threshold=0.70, min_samples=2)
    assert br.actions == [0]
    assert br.cost["samples"] < 8


def test_build_dataset_synthetic_and_checker():
    ds = build_dataset("synthetic", limit=5)
    assert len(list(ds.tasks())) == 5
    assert build_outcome_checker("synthetic") is outcome_exact


def test_runner_end_to_end_mock(tmp_path):
    out = tmp_path / "r.json"
    args = build_parser().parse_args(
        ["--method", "acdan", "--dataset", "synthetic", "--limit", "12",
         "--out", str(out)])
    res = run(args)
    s = res["summary"]
    assert s["n_tasks"] == 12
    assert 0.0 <= s["accuracy"] <= 1.0
    assert out.exists()


def test_runner_baseline_mock():
    args = build_parser().parse_args(
        ["--method", "bon", "--dataset", "synthetic", "--limit", "8", "--n", "4"])
    res = run(args)
    assert res["summary"]["n_tasks"] == 8


def test_runner_adaptive_sc_mock():
    args = build_parser().parse_args(
        ["--method", "asc", "--dataset", "synthetic", "--limit", "8", "--n", "4"])
    res = run(args)
    assert res["summary"]["n_tasks"] == 8
    assert "mean_samples" in res["summary"]


def test_runner_ablation_alias():
    # friendly ablation name maps to the flag field and runs
    args = build_parser().parse_args(
        ["--method", "acdan", "--disable", "no_dto", "--dataset", "synthetic", "--limit", "8"])
    res = run(args)
    assert res["summary"]["n_tasks"] == 8


def test_new_baselines_run_offline():
    """ToT / RAP / Self-Refine / s1 produce well-formed plans on mock backends."""
    raw = next(iter(SyntheticRawDataset(n=1, k=4, seed=3).tasks()))
    task = _task(raw)
    cfg = ACDANConfig(seed=0)
    core = build_core_model("mock", seed=0)
    prm = build_prm("mock", seed=0)
    rea = LatentReasoner(cfg.latent, feature_dim=task.prompt_features.size, seed=0)
    latent = rea.reason(task.prompt_features).final_state
    results = {
        "tot": tree_of_thoughts(core, prm, task, latent, n_per_step=3, keep_top_b=2),
        "rap": reasoning_as_planning(core, prm, task, latent, n_rollouts=6),
        "refine": self_refine(core, prm, task, latent, max_iters=4),
        "s1": s1_budget_forcing(core, prm, task, latent, max_budget=6, min_budget=2),
    }
    for name, br in results.items():
        assert len(br.actions) == task.horizon, name
        assert all(0 <= a < task.vocab_size for a in br.actions), name
        assert br.cost["samples"] >= 1, name
        assert br.cost["verified_candidates"] == task.horizon * task.vocab_size, name


def test_runner_new_baselines_mock():
    for method in ("tot", "rap", "refine", "s1"):
        args = build_parser().parse_args(
            ["--method", method, "--dataset", "synthetic", "--limit", "6", "--n", "4"])
        res = run(args)
        assert res["summary"]["n_tasks"] == 6
        assert 0.0 <= res["summary"]["accuracy"] <= 1.0
