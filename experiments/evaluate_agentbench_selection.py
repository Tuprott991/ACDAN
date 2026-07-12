"""Join blind selector outputs with immutable official scores and report metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acdan.agentbench.artifacts import file_sha256, read_jsonl, read_scores
from acdan.agentbench.calibration import PlattCalibrator
from acdan.agentbench.metrics import summarize_selection


def _candidate_index(path: Path) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for line_no, row in enumerate(read_jsonl(path), start=1):
        task = row.get("task", {}) or {}
        task_id = str(task.get("task_id", ""))
        candidates = row.get("candidates")
        if not task_id or not isinstance(candidates, list):
            raise ValueError(f"{path}:{line_no}: invalid candidate row")
        if task_id in index:
            raise ValueError(f"{path}:{line_no}: duplicate task {task_id}")
        index[task_id] = [dict(candidate) for candidate in candidates]
    return index


def evaluate(
    selection_path: Path,
    candidates_path: Path,
    scores_path: Path,
    calibrator_path: Path | None,
    abstain_threshold: float | None,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_rows = selection.get("per_task", [])
    if not isinstance(selected_rows, list) or not selected_rows:
        raise ValueError("selection result must contain non-empty per_task rows")
    candidates = _candidate_index(candidates_path)
    scores = read_scores(scores_path)
    calibrator = PlattCalibrator.load(calibrator_path) if calibrator_path else None

    evaluated: list[dict[str, Any]] = []
    generation_tokens: list[float] = []
    generation_tool_calls: list[float] = []
    evaluator_versions: set[str] = set()
    for selected in selected_rows:
        task_id = str(selected["task_id"])
        candidate_id = str(selected["selected_candidate_id"])
        task_candidates = candidates.get(task_id)
        if task_candidates is None:
            raise ValueError(f"selected task {task_id} is absent from candidate file")
        candidate_ids = [str(candidate.get("candidate_id", i)) for i, candidate in enumerate(task_candidates)]
        if candidate_id not in candidate_ids:
            raise ValueError(f"selected candidate {task_id}/{candidate_id} is absent")
        task_scores = []
        for current_id in candidate_ids:
            key = (task_id, current_id)
            if key not in scores:
                raise ValueError(f"missing official score for {task_id}/{current_id}")
            task_scores.append(scores[key])
            evaluator_versions.add(scores[key].evaluator_version)
        chosen = scores[(task_id, candidate_id)]
        raw_confidence = float(selected.get("raw_confidence", selected.get("confidence", 0.5)))
        confidence = calibrator.predict(raw_confidence) if calibrator else raw_confidence
        abstained = bool(selected.get("abstained", False))
        if abstain_threshold is not None:
            abstained = confidence < abstain_threshold
        row = dict(selected)
        row.update({
            "raw_confidence": raw_confidence,
            "confidence": confidence,
            "abstained": abstained,
            "selected_score": float(chosen.score),
            "selected_correct": bool(chosen.correct),
            "oracle_score": max(float(score.score) for score in task_scores),
            "pass_at_k": any(bool(score.correct) for score in task_scores),
            "official_evaluator": chosen.evaluator,
        })
        evaluated.append(row)
        for candidate in task_candidates:
            cost = candidate.get("cost", {}) or {}
            generation_tokens.append(float(cost.get("total_tokens", 0.0) or 0.0))
            generation_tool_calls.append(float(cost.get("tool_calls", 0.0) or 0.0))

    summary = dict(selection.get("summary", {}) or {})
    summary.update(summarize_selection(evaluated))
    summary.update({
        "selection_only": False,
        "candidate_file_sha256": file_sha256(candidates_path),
        "score_file_sha256": file_sha256(scores_path),
        "selection_file_sha256": file_sha256(selection_path),
        "calibrator": str(calibrator_path) if calibrator_path else None,
        "evaluator_versions": sorted(version for version in evaluator_versions if version),
        "mean_generation_tokens_per_candidate": float(np.mean(generation_tokens)) if generation_tokens else 0.0,
        "mean_generation_tool_calls_per_candidate": float(np.mean(generation_tool_calls)) if generation_tool_calls else 0.0,
    })
    return {"summary": summary, "per_task": evaluated}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--calibrator", default=None)
    parser.add_argument("--abstain-threshold", type=float, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        Path(args.selection),
        Path(args.candidates),
        Path(args.scores),
        Path(args.calibrator) if args.calibrator else None,
        args.abstain_threshold,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
