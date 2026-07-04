"""Validate AgentBench task and candidate files before expensive VM runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def validate_tasks(path: Path) -> list[str]:
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
        if len(instruction) < 12:
            errors.append(f"{prefix}: suspiciously short instruction {instruction!r}")
        if evaluator == "semantic_qa" and not str(row.get("gold", "")).strip():
            errors.append(f"{prefix}: semantic_qa requires non-empty gold")
        if dataset == "browsecomp" and evaluator != "external_browsecomp":
            errors.append(f"{prefix}: BrowseComp should use external_browsecomp, not {evaluator}")
        if dataset == "browsecomp" and not (row.get("metadata") or {}).get("encrypted"):
            errors.append(f"{prefix}: BrowseComp metadata should mark encrypted=true")
        if dataset != "browsecomp" and _looks_encrypted(instruction):
            errors.append(f"{prefix}: instruction looks encrypted/base64")
    return errors


def validate_candidates(path: Path, min_candidates: int) -> list[str]:
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
        if evaluator in EXTERNAL_EVALS:
            for j, cand in enumerate(candidates):
                if "score" not in cand and "is_correct" not in cand:
                    errors.append(
                        f"{prefix}: candidate {j} for {evaluator} needs score/is_correct "
                        "or run selection with --evaluator-command"
                    )
                    break
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate AgentBench manifests/candidate files.")
    ap.add_argument("--tasks-dir", default=None)
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--candidates", nargs="*", default=[])
    ap.add_argument("--min-candidates", type=int, default=1)
    args = ap.parse_args(argv)

    task_paths = [Path(p) for p in args.tasks]
    if args.tasks_dir:
        task_paths.extend(sorted(Path(args.tasks_dir).glob("*_tasks.jsonl")))
    errors: list[str] = []
    for path in task_paths:
        errors.extend(validate_tasks(path))
    for path in [Path(p) for p in args.candidates]:
        errors.extend(validate_candidates(path, args.min_candidates))

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1
    print(f"validated {len(task_paths)} task file(s), {len(args.candidates)} candidate file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
