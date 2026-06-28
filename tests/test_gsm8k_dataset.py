import json

from acdan.backends.encoder import HashingEncoder
from acdan.datasets.gsm8k import GSM8KDataset
from acdan.run_experiment import _to_task, build_parser, run


def test_gsm8k_dataset_preserves_candidate_evidence(tmp_path):
    path = tmp_path / "gsm.jsonl"
    row = {
        "question": "What is 2+2?",
        "answer": "4",
        "candidates": ["3", "4"],
        "candidate_solutions": {"4": "2+2=4. The answer is 4."},
        "candidate_counts": {"3": 1, "4": 3},
        "candidate_first_indices": {"3": 0, "4": 1},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    raw = next(GSM8KDataset(str(path), include_candidate_counts=True).tasks())
    assert raw.vocab == ("3", "4")
    assert raw.metadata["candidate_counts"]["4"] == 3
    assert "Candidate reasoning" in raw.action_templates["4"]

    task = _to_task(raw, HashingEncoder(dim=8).encode(raw.prompt))
    assert task.metadata["candidate_solutions"]["4"].startswith("2+2=4")
    assert task.metadata["candidate_counts"]["4"] == 3


def test_gsm8k_dataset_hides_candidate_counts_by_default(tmp_path):
    path = tmp_path / "gsm.jsonl"
    row = {
        "question": "What is 2+2?",
        "answer": "4",
        "candidates": ["3", "4"],
        "candidate_counts": {"3": 1, "4": 3},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    raw = next(GSM8KDataset(str(path)).tasks())
    assert "Self-consistency count" not in raw.action_templates["4"]
    assert raw.metadata["use_prm_count_bonus"] is False


def test_math_dataset_alias_runs_with_candidate_file(tmp_path):
    path = tmp_path / "math500.jsonl"
    row = {
        "question": "What is 2+2?",
        "answer": "4",
        "candidates": ["3", "4"],
        "candidate_counts": {"3": 1, "4": 3},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    args = build_parser().parse_args([
        "--method", "acdan",
        "--dataset", "math500",
        "--data-path", str(path),
        "--policy", "mock",
        "--prm", "mock",
        "--math-evidence", "all",
        "--limit", "1",
    ])
    res = run(args)

    assert res["summary"]["dataset"] == "math500"
    assert res["summary"]["math_evidence"] == "all"
    assert res["summary"]["n_tasks"] == 1
    assert res["summary"]["oracle_candidate_accuracy"] == 1.0
