import json

from acdan.backends.encoder import HashingEncoder
from acdan.datasets.gsm8k import GSM8KDataset
from acdan.run_experiment import _to_task


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

    raw = next(GSM8KDataset(str(path)).tasks())
    assert raw.vocab == ("3", "4")
    assert raw.metadata["candidate_counts"]["4"] == 3
    assert "Candidate reasoning" in raw.action_templates["4"]

    task = _to_task(raw, HashingEncoder(dim=8).encode(raw.prompt))
    assert task.metadata["candidate_solutions"]["4"].startswith("2+2=4")
    assert task.metadata["candidate_counts"]["4"] == 3
