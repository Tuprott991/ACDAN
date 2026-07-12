"""Validate AgentBench task and candidate files before expensive VM runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_EVALUATORS = {
    "browsecomp": "external_browsecomp",
    "webvoyager": "external_webvoyager",
    "swe_bench_verified": "external_swe_bench",
    "terminal_bench": "external_terminal_bench",
    "mathhay": "external_mathhay",
    "tau2_bench": "external_tau2",
    "mcp_bench": "external_mcp_bench",
}


EXPECTED_TASKS = {
    "browsecomp": 124,
    "webvoyager": 65,
    "swe_bench_verified": 50,
    "terminal_bench": 80,
    "mathhay": 75,
    "tau2_bench": 50,
    "mcp_bench": 52,
}

EXTERNAL_EVALS = {
    "external_browsecomp",
    "external_webvoyager",
    "external_swe_bench",
    "external_terminal_bench",
    "external_mathhay",
    "external_tau2",
    "external_mcp_bench",
}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _looks_encrypted(text: str) -> bool:
    if len(text) < 80:
        return False
    alphabet = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    return sum(ch in alphabet for ch in text) / max(len(text), 1) > 0.92


def validate_tasks(path: Path, require_provenance: bool = False) -> list[str]:
    errors: list[str] = []
    rows = _read_jsonl(path)
    dataset = path.name.removesuffix("_tasks.jsonl")
    expected = EXPECTED_TASKS.get(dataset)
    if expected is not None and len(rows) != expected:
        errors.append(f"{path}: expected {expected} rows, found {len(rows)}")
    seen = set()
    for i, row in enumerate(rows):
        prefix = f"{path}:{i + 1}"
        for key in ("task_id", "dataset", "domain", "instruction", "evaluator"):
            if key not in row:
                errors.append(f"{prefix}: missing {key}")
        task_id = row.get("task_id")
        if task_id in seen:
            errors.append(f"{prefix}: duplicate task_id {task_id}")
        seen.add(task_id)
        instruction = str(row.get("instruction", "")).strip()
        evaluator = str(row.get("evaluator", ""))
        if str(row.get("dataset", "")) != dataset:
            errors.append(f"{prefix}: dataset field does not match filename dataset {dataset}")
        if len(instruction) < 12:
            errors.append(f"{prefix}: suspiciously short instruction {instruction!r}")
        if evaluator == "semantic_qa" and not str(row.get("gold", "")).strip():
            errors.append(f"{prefix}: semantic_qa requires non-empty gold")
        if dataset == "browsecomp" and evaluator != "external_browsecomp":
            errors.append(f"{prefix}: BrowseComp should use external_browsecomp, not {evaluator}")
        if (
            dataset == "browsecomp"
            and row.get("gold") is not None
            and not (row.get("metadata") or {}).get("encrypted")
        ):
            errors.append(f"{prefix}: embedded BrowseComp gold must be marked encrypted=true")
        if dataset != "browsecomp" and _looks_encrypted(instruction):
            errors.append(f"{prefix}: instruction looks encrypted/base64")
        expected_evaluator = EXPECTED_EVALUATORS.get(dataset)
        if require_provenance and expected_evaluator and evaluator != expected_evaluator:
            errors.append(f"{prefix}: expected official evaluator {expected_evaluator}, found {evaluator}")
        if require_provenance:
            metadata = row.get("metadata") or {}
            for key in ("original_task_id", "source_revision", "source_sha256"):
                if not str(metadata.get(key, "")).strip():
                    errors.append(f"{prefix}: reportable manifest requires metadata.{key}")
            if str(metadata.get("source_revision", "")) == "unversioned":
                errors.append(f"{prefix}: source_revision must be a pinned commit")
    return errors


def validate_candidates(
    path: Path,
    min_candidates: int,
    allow_external_unscored: bool = False,
    require_blind: bool = False,
) -> list[str]:
    errors: list[str] = []
    rows = _read_jsonl(path)
    for i, row in enumerate(rows):
        prefix = f"{path}:{i + 1}"
        task = row.get("task")
        candidates = row.get("candidates")
        if not isinstance(task, dict):
            errors.append(f"{prefix}: missing task object")
            continue
        if not isinstance(candidates, list):
            errors.append(f"{prefix}: missing candidates list")
            continue
        if len(candidates) < min_candidates:
            errors.append(f"{prefix}: expected at least {min_candidates} candidates, found {len(candidates)}")
        evaluator = str(task.get("evaluator", ""))
        candidate_ids = [str(cand.get("candidate_id", j)) for j, cand in enumerate(candidates)]
        if len(set(candidate_ids)) != len(candidate_ids):
            errors.append(f"{prefix}: duplicate candidate_id values")
        if require_blind:
            for j, cand in enumerate(candidates):
                if "score" in cand or "is_correct" in cand or "correct" in cand:
                    errors.append(f"{prefix}: blind candidate {j} contains an outcome field")
                    break
        if evaluator in EXTERNAL_EVALS and not allow_external_unscored:
            for j, cand in enumerate(candidates):
                if "score" not in cand and "is_correct" not in cand:
                    errors.append(
                        f"{prefix}: candidate {j} for {evaluator} needs score/is_correct "
                        "or run selection with --evaluator-command"
                    )
                    break
    return errors


def validate_scores(path: Path) -> tuple[list[str], set[tuple[str, str]]]:
    errors: list[str] = []
    pairs: set[tuple[str, str]] = set()
    for i, row in enumerate(_read_jsonl(path)):
        prefix = f"{path}:{i + 1}"
        for key in ("task_id", "candidate_id", "dataset", "score", "correct", "evaluator"):
            if key not in row:
                errors.append(f"{prefix}: missing {key}")
        pair = (str(row.get("task_id", "")), str(row.get("candidate_id", "")))
        if pair in pairs:
            errors.append(f"{prefix}: duplicate task/candidate score {pair}")
        pairs.add(pair)
        try:
            score = float(row.get("score", 0.0))
            if not 0.0 <= score <= 1.0:
                errors.append(f"{prefix}: normalized score must be in [0,1], found {score}")
        except (TypeError, ValueError):
            errors.append(f"{prefix}: score is not numeric")
        if not str(row.get("evaluator_version", "")).strip():
            errors.append(f"{prefix}: missing evaluator_version")
    return errors, pairs


def candidate_pairs(path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for row in _read_jsonl(path):
        task_id = str((row.get("task") or {}).get("task_id", ""))
        for index, candidate in enumerate(row.get("candidates", []) or []):
            pairs.add((task_id, str(candidate.get("candidate_id", index))))
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate AgentBench manifests/candidate files.")
    ap.add_argument("--tasks-dir", default=None)
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--candidates", nargs="*", default=[])
    ap.add_argument("--scores", nargs="*", default=[])
    ap.add_argument("--min-candidates", type=int, default=1)
    ap.add_argument(
        "--allow-external-unscored",
        action="store_true",
        help=(
            "Allow external-evaluator candidates without score/is_correct. "
            "Use only when selection will receive --evaluator-command, or for smoke runs."
        ),
    )
    ap.add_argument("--require-provenance", action="store_true")
    ap.add_argument("--require-blind", action="store_true")
    args = ap.parse_args(argv)

    task_paths = [Path(p) for p in args.tasks]
    if args.tasks_dir:
        task_paths.extend(sorted(Path(args.tasks_dir).glob("*_tasks.jsonl")))
    errors: list[str] = []
    for path in task_paths:
        errors.extend(validate_tasks(path, args.require_provenance))
    for path in [Path(p) for p in args.candidates]:
        errors.extend(
            validate_candidates(
                path,
                args.min_candidates,
                args.allow_external_unscored or bool(args.scores),
                args.require_blind,
            )
        )

    score_pairs: set[tuple[str, str]] = set()
    for path in [Path(p) for p in args.scores]:
        score_errors, current_pairs = validate_scores(path)
        errors.extend(score_errors)
        overlap = score_pairs & current_pairs
        if overlap:
            errors.append(f"{path}: duplicates {len(overlap)} score pairs from another score file")
        score_pairs.update(current_pairs)
    if args.scores and args.candidates:
        expected_pairs: set[tuple[str, str]] = set()
        for path in [Path(p) for p in args.candidates]:
            expected_pairs.update(candidate_pairs(path))
        missing = expected_pairs - score_pairs
        extra = score_pairs - expected_pairs
        if missing:
            errors.append(f"official score files are missing {len(missing)} candidate outcomes")
        if extra:
            errors.append(f"official score files contain {len(extra)} unknown candidate outcomes")

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1
    print(
        f"validated {len(task_paths)} task file(s), {len(args.candidates)} candidate file(s), "
        f"{len(args.scores)} score file(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
