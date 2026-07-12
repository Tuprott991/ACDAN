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


def _official_import_module():
    spec = importlib.util.spec_from_file_location(
        "import_general_agentbench", "experiments/import_general_agentbench.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _official_evaluation_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_agentbench_selection", "experiments/evaluate_agentbench_selection.py"
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
    assert "[truncated middle of candidate for selector scoring]" in text


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


def test_explicit_general_agent_adapter_reads_only_known_task_file(tmp_path):
    checkout = tmp_path / "General-AgentBench"
    data = checkout / "general_agent" / "data"
    data.mkdir(parents=True)
    (data / "search_benchmark.json").write_text(json.dumps([
        {
            "benchmark": "search",
            "id": 1,
            "question": "Find the requested fact using web evidence.",
            "dataset": "browsecomp",
            "domain": "browsecomp",
            "type": "browsecomp",
            "golden_answer": "hidden from selector",
        }
    ]), encoding="utf-8")
    # This unrelated file must never be discovered as a task.
    (checkout / "workflow.json").write_text('{"name":"Deploy Documentation"}', encoding="utf-8")

    out = prepare_dataset(
        "browsecomp",
        tmp_path / "manifests",
        source_path=checkout,
        source_revision="abc123",
        limit=1,
    )
    tasks = read_tasks(out)

    assert [task.task_id for task in tasks] == ["browsecomp_1"]
    assert tasks[0].gold is None
    assert tasks[0].metadata["source_revision"] == "abc123"
    assert len(tasks[0].metadata["source_sha256"]) == 64


def test_import_native_passes_keeps_scores_separate(tmp_path):
    mod = _official_import_module()
    tasks_path = tmp_path / "browsecomp_tasks.jsonl"
    task = AgentBenchTask(
        task_id="browsecomp_1",
        dataset="browsecomp",
        domain="search",
        instruction="Find a fact.",
        evaluator="external_browsecomp",
    )
    tasks_path.write_text(json.dumps(task.to_json()) + "\n", encoding="utf-8")
    pass_dir = tmp_path / "native" / "pass_1"
    (pass_dir / "traces").mkdir(parents=True)
    (pass_dir / "evaluations").mkdir()
    (pass_dir / "traces" / "browsecomp_1.json").write_text(json.dumps({
        "benchmark": "search",
        "task_id": "browsecomp_1",
        "predicted_answer": "Paris",
        "trace": {
            "messages": [{"role": "assistant", "content": "Paris"}],
            "total_prompt_tokens": 10,
            "total_output_tokens": 2,
            "total_tokens": 12,
            "duration": 0.5,
        },
    }), encoding="utf-8")
    (pass_dir / "evaluations" / "browsecomp_1.json").write_text(json.dumps({
        "task_id": "browsecomp_1",
        "score": 1.0,
        "is_correct": True,
    }), encoding="utf-8")

    candidates, trajectories, scores = mod.import_passes(
        source=tmp_path / "native",
        tasks_path=tasks_path,
        dataset="browsecomp",
        k=1,
        evaluator_version="fixture-v1",
        strict=True,
    )

    candidate = candidates[0]["candidates"][0]
    assert candidate["candidate_id"] == "pass_1"
    assert "score" not in candidate and "correct" not in candidate
    assert trajectories[0]["cost"]["total_tokens"] == 12
    assert scores[0]["correct"] is True


def test_native_score_supports_tau_reward_and_swe_resolved():
    mod = _official_import_module()
    tau = mod._score(
        "tau2_bench", "tau2:airline:0", "pass_1",
        {"reward_info": {"reward": 1.0}}, "fixture-v1",
    )
    swe = mod._score(
        "swe_bench_verified", "django__django-1", "pass_1",
        {"report": {"resolved": True}}, "fixture-v1",
    )

    assert tau.score == 1.0 and tau.correct is True
    assert swe.score == 1.0 and swe.correct is True


def test_blind_selection_can_be_joined_with_official_scores(tmp_path):
    selection_mod = _selection_module()
    evaluation_mod = _official_evaluation_module()
    candidates_path = tmp_path / "candidates.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    selection_path = tmp_path / "selection.json"
    row = {
        "task": {
            "task_id": "b1",
            "dataset": "browsecomp",
            "domain": "search",
            "instruction": "Capital of France?",
            "evaluator": "external_browsecomp",
        },
        "candidates": [
            {"candidate_id": "pass_1", "final_answer": "Paris", "cost": {"total_tokens": 20}},
            {"candidate_id": "pass_2", "final_answer": "Lyon", "cost": {"total_tokens": 22}},
        ],
    }
    candidates_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    scores_path.write_text(
        json.dumps({
            "task_id": "b1", "candidate_id": "pass_1", "dataset": "browsecomp",
            "score": 1.0, "correct": True, "evaluator": "external_browsecomp",
            "evaluator_version": "fixture-v1",
        }) + "\n" +
        json.dumps({
            "task_id": "b1", "candidate_id": "pass_2", "dataset": "browsecomp",
            "score": 0.0, "correct": False, "evaluator": "external_browsecomp",
            "evaluator_version": "fixture-v1",
        }) + "\n",
        encoding="utf-8",
    )
    args = selection_mod.build_parser().parse_args([
        "--candidates-path", str(candidates_path),
        "--method", "cot", "--policy", "mock", "--prm", "mock",
        "--no-latent", "--selection-only", "--out", str(selection_path),
    ])
    blind = selection_mod.run(args)

    assert blind["summary"]["selection_only"] is True
    assert "selected_correct" not in blind["per_task"][0]
    evaluated = evaluation_mod.evaluate(
        selection_path, candidates_path, scores_path, None, None
    )
    assert evaluated["summary"]["selection_only"] is False
    assert "ece" in evaluated["summary"]
    assert evaluated["summary"]["pass_at_k"] == 1.0
