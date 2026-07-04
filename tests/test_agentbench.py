import importlib.util
import json
import sys

from acdan.agentbench.adapters import AgentBenchTask, prepare_dataset, read_tasks
from acdan.agentbench.evaluators import Candidate, CandidateEvaluator


def _selection_module():
    spec = importlib.util.spec_from_file_location(
        "run_agentbench_selection", "experiments/run_agentbench_selection.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _merge_module():
    spec = importlib.util.spec_from_file_location(
        "build_agentbench_candidates", "experiments/build_agentbench_candidates.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prediction_module():
    spec = importlib.util.spec_from_file_location(
        "gen_agentbench_predictions", "experiments/gen_agentbench_predictions.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validator_module():
    spec = importlib.util.spec_from_file_location(
        "validate_agentbench", "experiments/validate_agentbench.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_agentbench_generic_mathhay_adapter(tmp_path):
    source = tmp_path / "raw_mathhay"
    source.mkdir()
    (source / "tasks.jsonl").write_text(
        '{"id":"m1","question":"2+2?","answer":"4"}\n'
        '{"id":"m2","question":"3+3?","answer":"6"}\n',
        encoding="utf-8",
    )

    out = prepare_dataset(
        "mathhay",
        tmp_path / "agentbench",
        source_path=source,
        limit=1,
        overwrite=True,
    )
    tasks = read_tasks(out)

    assert len(tasks) == 1
    assert tasks[0].dataset == "mathhay"
    assert tasks[0].evaluator == "semantic_qa"


def test_candidate_evaluator_uses_precomputed_scores():
    task = AgentBenchTask(
        task_id="swe-1",
        dataset="swe_bench_verified",
        domain="coding",
        instruction="fix bug",
        evaluator="external_swe_bench",
    )
    ev = CandidateEvaluator()

    assert ev.evaluate(task, Candidate.from_obj({"is_correct": True}, 0))["correct"] is True
    assert ev.evaluate(task, Candidate.from_obj({"score": 0.25}, 1))["score"] == 0.25


def test_agentbench_selection_runner_mock(tmp_path):
    mod = _selection_module()
    path = tmp_path / "candidates.jsonl"
    row = {
        "task": {
            "task_id": "b1",
            "dataset": "browsecomp",
            "domain": "search",
            "instruction": "Capital of France?",
            "evaluator": "semantic_qa",
            "gold": "Paris",
        },
        "candidates": [
            {"candidate_id": "a", "final_answer": "Lyon", "is_correct": False},
            {"candidate_id": "b", "final_answer": "Paris", "is_correct": True},
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    args = mod.build_parser().parse_args([
        "--candidates-path", str(path),
        "--method", "bon",
        "--policy", "mock",
        "--prm", "mock",
        "--no-latent",
        "--limit", "1",
    ])
    result = mod.run(args)

    assert result["summary"]["n_tasks"] == 1
    assert result["summary"]["pass_at_k"] == 1.0
    assert result["summary"]["oracle_score"] == 1.0
    assert "verification_gap" in result["summary"]


def test_agentbench_raw_task_truncates_long_candidates():
    mod = _selection_module()
    task = AgentBenchTask(
        task_id="b1",
        dataset="browsecomp",
        domain="search",
        instruction="Find the answer from web evidence.",
        evaluator="external_browsecomp",
    )
    cand = Candidate.from_obj(
        {
            "candidate_id": "0",
            "final_answer": "A" * 120,
            "trajectory": [{"role": "assistant", "content": "B" * 120}],
        },
        0,
    )

    raw = mod._raw_task(task, [cand], candidate_preview_chars=64)

    text = raw.action_templates["0"]
    assert len(text) < len(cand.display_text())
    assert "[truncated candidate for selector scoring]" in text


def test_agentbench_validator_can_allow_external_unscored_candidates(tmp_path):
    mod = _validator_module()
    path = tmp_path / "candidates.jsonl"
    row = {
        "task": {
            "task_id": "b1",
            "dataset": "browsecomp",
            "domain": "search",
            "instruction": "Encrypted BrowseComp task requiring official scoring.",
            "evaluator": "external_browsecomp",
        },
        "candidates": [{"candidate_id": "0", "final_answer": "raw prediction"}],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    strict_errors = mod.validate_candidates(path, min_candidates=1)
    allowed_errors = mod.validate_candidates(
        path,
        min_candidates=1,
        allow_external_unscored=True,
    )

    assert strict_errors
    assert allowed_errors == []


def test_agentbench_selection_cli_fails_cleanly_on_unscored_external(tmp_path, capsys):
    mod = _selection_module()
    path = tmp_path / "candidates.jsonl"
    row = {
        "task": {
            "task_id": "b1",
            "dataset": "browsecomp",
            "domain": "search",
            "instruction": "Encrypted BrowseComp task requiring official scoring.",
            "evaluator": "external_browsecomp",
        },
        "candidates": [{"candidate_id": "0", "final_answer": "raw prediction"}],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    rc = mod.main([
        "--candidates-path", str(path),
        "--method", "cot",
        "--policy", "mock",
        "--prm", "mock",
        "--no-latent",
    ])
    captured = capsys.readouterr()

    assert rc == 2
    assert "ERROR:" in captured.err
    assert "--allow-unevaluated" in captured.err
    assert "Traceback" not in captured.err


def test_agentbench_selection_allows_explicit_unscored_smoke(tmp_path):
    mod = _selection_module()
    path = tmp_path / "candidates.jsonl"
    row = {
        "task": {
            "task_id": "b1",
            "dataset": "browsecomp",
            "domain": "search",
            "instruction": "Encrypted BrowseComp task requiring official scoring.",
            "evaluator": "external_browsecomp",
        },
        "candidates": [{"candidate_id": "0", "final_answer": "raw prediction"}],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    args = mod.build_parser().parse_args([
        "--candidates-path", str(path),
        "--method", "cot",
        "--policy", "mock",
        "--prm", "mock",
        "--no-latent",
        "--allow-unevaluated",
    ])
    result = mod.run(args)

    assert result["summary"]["n_tasks"] == 1
    assert result["summary"]["selected_accuracy"] == 0.0
    assert result["per_task"] == []


def test_build_agentbench_candidates_groups_prediction_lines(tmp_path):
    mod = _merge_module()
    tasks = tmp_path / "tasks.jsonl"
    preds = tmp_path / "predictions.jsonl"
    out = tmp_path / "candidates.jsonl"
    task = AgentBenchTask(
        task_id="b1",
        dataset="browsecomp",
        domain="search",
        instruction="Capital?",
        evaluator="semantic_qa",
        gold="Paris",
    )
    tasks.write_text(json.dumps(task.to_json()) + "\n", encoding="utf-8")
    preds.write_text(
        '{"task_id":"b1","final_answer":"Lyon","is_correct":false}\n'
        '{"task_id":"b1","final_answer":"Paris","is_correct":true}\n',
        encoding="utf-8",
    )

    mod.main([
        "--tasks", str(tasks),
        "--predictions", str(preds),
        "--out", str(out),
        "--min-candidates", "2",
    ])
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])

    assert row["task"]["task_id"] == "b1"
    assert len(row["candidates"]) == 2
    assert row["candidates"][1]["final_answer"] == "Paris"


def test_gen_agentbench_predictions_mock_writes_prediction_lines(tmp_path):
    mod = _prediction_module()
    tasks = tmp_path / "tasks.jsonl"
    out = tmp_path / "predictions.jsonl"
    task = AgentBenchTask(
        task_id="m1",
        dataset="mathhay",
        domain="reason",
        instruction="2+2?",
        evaluator="semantic_qa",
        gold="4",
    )
    tasks.write_text(json.dumps(task.to_json()) + "\n", encoding="utf-8")

    mod.main([
        "--tasks", str(tasks),
        "--out", str(out),
        "--backend", "mock",
        "--k", "3",
    ])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 3
    assert rows[0]["task_id"] == "m1"
    assert rows[0]["is_correct"] is True
    assert rows[1]["is_correct"] is False
