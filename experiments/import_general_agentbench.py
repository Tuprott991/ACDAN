"""Import native General AgentBench passes into blind ACDAN candidate artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acdan.agentbench.adapters import read_tasks
from acdan.agentbench.artifacts import (
    OfficialScore,
    TrajectoryArtifact,
    TrajectoryCost,
    write_jsonl,
)


EVALUATOR = {
    "browsecomp": "external_browsecomp",
    "mathhay": "external_mathhay",
    "swe_bench_verified": "external_swe_bench",
    "terminal_bench": "external_terminal_bench",
    "tau2_bench": "external_tau2",
    "mcp_bench": "external_mcp_bench",
}


def _read_objects(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not directory.is_dir():
        return []
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            rows.append((path, value))
    return rows


def _canonical_id(dataset: str, value: dict[str, Any]) -> str:
    raw = str(value.get("task_id", value.get("id", "")))
    if not raw:
        raise ValueError("native artifact has no task_id")
    if dataset == "tau2_bench":
        domain = str(value.get("domain", "unknown"))
        return f"tau2:{domain}:{raw}"
    return raw


def _messages(trace: dict[str, Any]) -> list[dict[str, Any]]:
    nested = trace.get("trace", {}) or {}
    messages = nested.get("messages", trace.get("messages", [])) or []
    return [dict(message) for message in messages if isinstance(message, dict)]


def _tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in messages:
        for call in message.get("tool_calls", []) or []:
            if isinstance(call, dict):
                calls.append(dict(call))
    return calls


def _final_answer(trace: dict[str, Any], messages: list[dict[str, Any]]) -> str:
    for key in ("predicted_answer", "final_response", "raw_response", "answer"):
        value = trace.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    nested = trace.get("trace", {}) or {}
    for key in ("final_response", "predicted_answer", "raw_response"):
        value = nested.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    for message in reversed(messages):
        if message.get("role") == "assistant" and str(message.get("content", "")).strip():
            return str(message["content"]).strip()
    return ""


def _patch(trace: dict[str, Any], evaluation: dict[str, Any] | None) -> str:
    sources = [trace, trace.get("trace", {}) or {}, evaluation or {}]
    for source in sources:
        for key in ("patch", "model_patch", "git_patch", "diff"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return ""


def _cost(trace: dict[str, Any], messages: list[dict[str, Any]]) -> TrajectoryCost:
    nested = trace.get("trace", {}) or {}
    input_tokens = int(nested.get("total_prompt_tokens", trace.get("total_input_tokens", 0)) or 0)
    output_tokens = int(nested.get("total_output_tokens", trace.get("total_output_tokens", 0)) or 0)
    total_tokens = int(nested.get("total_tokens", trace.get("total_tokens", input_tokens + output_tokens)) or 0)
    tool_calls = len(_tool_calls(messages))
    rounds = nested.get("rounds", []) or []
    return TrajectoryCost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or input_tokens + output_tokens,
        model_calls=len(rounds) or sum(message.get("role") == "assistant" for message in messages),
        tool_calls=tool_calls,
        wall_s=float(nested.get("duration", trace.get("duration", 0.0)) or 0.0),
        monetary_cost=float(nested.get("cost", trace.get("cost", 0.0)) or 0.0),
    )


def _score(
    dataset: str,
    task_id: str,
    candidate_id: str,
    evaluation: dict[str, Any],
    evaluator_version: str,
) -> OfficialScore:
    score_value = evaluation.get("reward", evaluation.get("score"))
    correct_value = evaluation.get("is_correct", evaluation.get("success", evaluation.get("correct")))
    if score_value is None and isinstance(evaluation.get("reward_info"), dict):
        score_value = evaluation["reward_info"].get("reward")
    if correct_value is None and isinstance(evaluation.get("report"), dict):
        correct_value = evaluation["report"].get("resolved")
    if correct_value is None and "test_passed" in evaluation:
        correct_value = evaluation.get("test_passed")
    if score_value is None and correct_value is None:
        raise ValueError(f"official evaluation for {task_id}/{candidate_id} has no score or success")
    score = float(score_value if score_value is not None else (1.0 if correct_value else 0.0))
    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"official score for {task_id}/{candidate_id} is not normalized: {score}. "
            "Import the native pass summary reward instead of a raw sub-metric."
        )
    correct = bool(correct_value if correct_value is not None else score >= 0.8)
    omit = {
        "messages", "rounds", "rounds_detail", "test_output", "final_response",
        "raw_response", "native_evaluation",
    }
    compact_raw = {key: value for key, value in evaluation.items() if key not in omit}
    return OfficialScore(
        task_id=task_id,
        candidate_id=candidate_id,
        dataset=dataset,
        score=score,
        correct=correct,
        evaluator=EVALUATOR[dataset],
        evaluator_version=evaluator_version,
        raw=compact_raw,
    )


def import_passes(
    *,
    source: Path,
    tasks_path: Path,
    dataset: str,
    k: int,
    evaluator_version: str,
    strict: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = {task.task_id: task for task in read_tasks(tasks_path)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectories: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []

    for pass_index in range(1, k + 1):
        pass_dir = source / f"pass_{pass_index}"
        evaluations: dict[str, dict[str, Any]] = {}
        for eval_path, value in _read_objects(pass_dir / "evaluations"):
            try:
                eval_task_id = _canonical_id(dataset, value)
            except ValueError:
                if dataset == "browsecomp" and eval_path.stem.startswith("result_"):
                    eval_task_id = "browsecomp_" + eval_path.stem.removeprefix("result_")
                else:
                    raise
            evaluations[eval_task_id] = value
        summary_path = pass_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for value in summary.get("results", []) or []:
                if not isinstance(value, dict):
                    continue
                if dataset == "tau2_bench" and not value.get("domain"):
                    raw_id = str(value.get("task_id", value.get("id", "")))
                    matches = [key for key in evaluations if key.endswith(f":{raw_id}")]
                    if len(matches) != 1:
                        raise ValueError(
                            f"cannot uniquely map Tau summary task {raw_id}; native summary must include domain"
                        )
                    summary_task_id = matches[0]
                else:
                    summary_task_id = _canonical_id(dataset, value)
                native = evaluations.get(summary_task_id, {})
                evaluations[summary_task_id] = {**native, **value, "native_evaluation": native}
        traces = _read_objects(pass_dir / "traces")
        if strict and not traces:
            raise RuntimeError(f"missing native traces under {pass_dir / 'traces'}")
        for trace_path, trace in traces:
            task_id = _canonical_id(dataset, trace)
            if task_id not in tasks:
                if strict:
                    raise RuntimeError(f"trace task {task_id} is absent from {tasks_path}")
                continue
            candidate_id = f"pass_{pass_index}"
            evaluation = evaluations.get(task_id)
            messages = _messages(trace)
            artifact = TrajectoryArtifact(
                task_id=task_id,
                candidate_id=candidate_id,
                dataset=dataset,
                final_answer=_final_answer(trace, messages),
                trajectory=messages,
                patch=_patch(trace, evaluation),
                tool_calls=_tool_calls(messages),
                artifact_paths={
                    "trace": str(trace_path),
                    "evaluation": str(pass_dir / "evaluations") if evaluation else "",
                },
                cost=_cost(trace, messages),
                generator={
                    "protocol": "general-agentbench-parallel-scaling",
                    "pass": pass_index,
                    "model": trace.get("model_name", ""),
                },
            )
            candidate = artifact.to_json()
            candidate.pop("task_id", None)
            candidate.pop("dataset", None)
            grouped[task_id].append(candidate)
            trajectories.append(artifact.to_json())
            if evaluation is not None:
                scores.append(_score(
                    dataset, task_id, candidate_id, evaluation, evaluator_version
                ).to_json())
            elif strict:
                raise RuntimeError(f"missing official evaluation for {task_id}/{candidate_id}")

    candidate_rows: list[dict[str, Any]] = []
    for task_id, task in tasks.items():
        candidates = grouped.get(task_id, [])
        if strict and len(candidates) != k:
            raise RuntimeError(f"{task_id}: expected {k} candidates, found {len(candidates)}")
        if candidates:
            candidate_rows.append({"task": task.to_json(), "candidates": candidates})
    return candidate_rows, trajectories, scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Native output containing pass_1..pass_K.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(EVALUATOR))
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--candidates-out", required=True)
    parser.add_argument("--trajectories-out", required=True)
    parser.add_argument("--scores-out", required=True)
    parser.add_argument("--evaluator-version", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    candidate_rows, trajectories, scores = import_passes(
        source=Path(args.source),
        tasks_path=Path(args.tasks),
        dataset=args.dataset,
        k=args.k,
        evaluator_version=args.evaluator_version,
        strict=not args.allow_incomplete,
    )
    n_candidates = write_jsonl(args.candidates_out, candidate_rows)
    n_trajectories = write_jsonl(args.trajectories_out, trajectories)
    n_scores = write_jsonl(args.scores_out, scores)
    print(
        f"imported {n_candidates} tasks, {n_trajectories} trajectories, "
        f"{n_scores} official scores"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
