"""Canonical, score-blind artifacts for AgentBench trajectory selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TrajectoryCost:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_calls: int = 0
    verifier_calls: int = 0
    tool_calls: int = 0
    wall_s: float = 0.0
    monetary_cost: float = 0.0


@dataclass
class TrajectoryArtifact:
    task_id: str
    candidate_id: str
    dataset: str
    final_answer: str = ""
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    patch: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    cost: TrajectoryCost = field(default_factory=TrajectoryCost)
    generator: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OfficialScore:
    task_id: str
    candidate_id: str
    dataset: str
    score: float
    correct: bool
    evaluator: str
    evaluator_version: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_scores(path: str | Path) -> dict[tuple[str, str], OfficialScore]:
    scores: dict[tuple[str, str], OfficialScore] = {}
    for row in read_jsonl(path):
        value = OfficialScore(
            task_id=str(row["task_id"]),
            candidate_id=str(row["candidate_id"]),
            dataset=str(row["dataset"]),
            score=float(row["score"]),
            correct=bool(row["correct"]),
            evaluator=str(row["evaluator"]),
            evaluator_version=str(row.get("evaluator_version", "")),
            raw=dict(row.get("raw", {}) or {}),
        )
        key = (value.task_id, value.candidate_id)
        if key in scores:
            raise ValueError(f"duplicate official score for task/candidate {key}")
        scores[key] = value
    return scores
