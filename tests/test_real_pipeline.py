"""The real-experiment pipeline must run offline with mock backends (no GPU)."""

import numpy as np

from acdan.agent import ACDANAgent
from acdan.baselines import best_of_n_prm, cot_greedy, self_consistency
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


def test_runner_ablation_alias():
    # friendly ablation name maps to the flag field and runs
    args = build_parser().parse_args(
        ["--method", "acdan", "--disable", "no_dto", "--dataset", "synthetic", "--limit", "8"])
    res = run(args)
    assert res["summary"]["n_tasks"] == 8
