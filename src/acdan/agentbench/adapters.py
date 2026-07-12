"""Explicit task adapters for the pinned General AgentBench protocol.

The official repositories contain many JSON/YAML configuration files that are
not tasks. This module deliberately opens only known task files and never scans
an archive recursively.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence


AGENTBENCH_DATASETS = (
    "browsecomp",
    "webvoyager",
    "swe_bench_verified",
    "terminal_bench",
    "mathhay",
    "tau2_bench",
    "mcp_bench",
)

PAPER_SAMPLE_SIZES = {
    "browsecomp": 124,
    "webvoyager": 65,
    "swe_bench_verified": 50,
    "terminal_bench": 80,
    "mathhay": 75,
    "tau2_bench": 50,
    "mcp_bench": 52,
}

DOMAINS = {
    "browsecomp": "search",
    "webvoyager": "search",
    "swe_bench_verified": "coding",
    "terminal_bench": "coding",
    "mathhay": "reason",
    "tau2_bench": "tool-calling",
    "mcp_bench": "tool-calling",
}

HF_DATASETS = {
    "browsecomp": ("smolagents/browse_comp", "test"),
    "webvoyager": ("btrabucco/web-voyager", "test"),
    "swe_bench_verified": ("SWE-bench/SWE-bench_Verified", "test"),
    "tau2_bench": ("Genteki/tau2-bench", "train"),
}

GENERAL_AGENT_TASK_FILES = {
    "browsecomp": "general_agent/data/search_benchmark.json",
    "swe_bench_verified": "general_agent/data/swebench_test_50.json",
    "terminal_bench": "general_agent/data/terminalbench_benchmark.json",
    "mathhay": "general_agent/data/mathhay_benchmark.json",
    "tau2_bench": "general_agent/data/tau2bench_benchmark.json",
    "mcp_bench": "general_agent/data/mcpbench_benchmark.json",
}

EXTERNAL_EVALUATOR = {
    "browsecomp": "external_browsecomp",
    "webvoyager": "external_webvoyager",
    "swe_bench_verified": "external_swe_bench",
    "terminal_bench": "external_terminal_bench",
    "mathhay": "external_mathhay",
    "tau2_bench": "external_tau2",
    "mcp_bench": "external_mcp_bench",
}


@dataclass
class AgentBenchTask:
    task_id: str
    dataset: str
    domain: str
    instruction: str
    evaluator: str
    gold: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def write_tasks(path: str | Path, tasks: Iterable[AgentBenchTask], overwrite: bool = True) -> int:
    path = Path(path)
    if path.exists() and not overwrite:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(task.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_tasks(path: str | Path) -> list[AgentBenchTask]:
    tasks: list[AgentBenchTask] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            tasks.append(AgentBenchTask(
                task_id=str(value["task_id"]),
                dataset=str(value["dataset"]),
                domain=str(value.get("domain", DOMAINS.get(str(value["dataset"]), "unknown"))),
                instruction=str(value["instruction"]),
                evaluator=str(value.get("evaluator", "external")),
                gold=value.get("gold"),
                metadata=dict(value.get("metadata", {}) or {}),
            ))
    return tasks


def _sample(rows: Sequence[Any], limit: int | None, seed: int) -> list[Any]:
    rows = list(rows)
    if limit is None or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[i] for i in indices]


def _load_hf_dataset(name: str, split: str, token: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Install dataset dependencies first: .venv/bin/pip install -r requirements-data.txt"
        ) from exc
    return load_dataset(name, split=split, token=token)


@lru_cache(maxsize=32)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exact_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(values, dict):
            values = values.get("tasks", values.get("data", [values]))
    else:
        raise ValueError(f"official task source must be .json or .jsonl: {path}")
    if not isinstance(values, list) or not all(isinstance(row, dict) for row in values):
        raise ValueError(f"expected a list of task objects in {path}")
    return list(values)


def _resolve_source(dataset: str, source_path: str | Path) -> Path:
    source = Path(source_path)
    if source.is_file():
        return source
    candidates: list[Path] = []
    if dataset in GENERAL_AGENT_TASK_FILES:
        candidates.extend([
            source / GENERAL_AGENT_TASK_FILES[dataset],
            source / Path(GENERAL_AGENT_TASK_FILES[dataset]).name,
        ])
    if dataset == "webvoyager":
        candidates.extend([
            source / "data" / "WebVoyager_data.jsonl",
            source / "WebVoyager_data.jsonl",
        ])
    # Explicit legacy fixture name; unlike the old adapter this never scans recursively.
    candidates.extend([source / f"{dataset}_tasks.jsonl", source / "tasks.jsonl"])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"no official task file found for {dataset}; checked: {checked}")


def _provenance(path: Path, revision: str | None, original_id: str) -> dict[str, Any]:
    return {
        "source_file": str(path),
        "source_sha256": _sha256(path),
        "source_revision": revision or "unversioned",
        "original_task_id": original_id,
    }


def _from_general_agent(
    dataset: str,
    row: dict[str, Any],
    index: int,
    source: Path,
    revision: str | None,
) -> AgentBenchTask:
    task = dict(row.get("task", {}) or {})
    evaluator = EXTERNAL_EVALUATOR[dataset]
    metadata = _provenance(source, revision, "")
    metadata.update({
        "source_file": GENERAL_AGENT_TASK_FILES[dataset],
        "upstream_benchmark": row.get("benchmark"),
        "upstream_dataset": row.get("dataset"),
        "upstream_domain": row.get("domain"),
    })

    if dataset == "browsecomp":
        original = str(row["id"])
        task_id = f"browsecomp_{original}"
        instruction = str(row["question"])
    elif dataset == "mcp_bench":
        original = str(task.get("id", index))
        task_id = f"mcpbench_{index}"
        instruction = str(task.get("fuzzy_description") or task.get("task_description") or "")
        metadata.update({
            "servers": task.get("servers", []),
            "combination_type": task.get("combination_type"),
        })
    elif dataset == "tau2_bench":
        original = str(task.get("id", index))
        upstream_domain = str(row.get("domain", row.get("dataset", "unknown")))
        task_id = f"tau2:{upstream_domain}:{original}"
        description = task.get("description", {}) or {}
        scenario = task.get("user_scenario", {}) or {}
        instructions = scenario.get("instructions", {}) or {}
        instruction = str(
            description.get("purpose")
            or instructions.get("reason_for_call")
            or task.get("instruction")
            or original
        )
        metadata.update({"tau2_domain": upstream_domain, "upstream_task": task})
    else:
        original = str(task.get("id", task.get("instance_id", index)))
        task_id = original
        instruction = str(task.get("question") or task.get("instruction") or task.get("task_description") or "")
        if dataset == "swe_bench_verified":
            metadata.update({"repo": task.get("repo"), "runtime": task.get("runtime", {})})
        elif dataset == "terminal_bench":
            metadata.update({"runtime": task.get("runtime", {}), "category": task.get("category")})
        elif dataset == "mathhay":
            metadata.update({"task_type": task.get("task_type"), "document_ids": task.get("document_ids", [])})

    metadata["original_task_id"] = original
    if not instruction.strip():
        raise ValueError(f"{source}: task {original} has no instruction")
    return AgentBenchTask(
        task_id=task_id,
        dataset=dataset,
        domain=DOMAINS[dataset],
        instruction=instruction,
        evaluator=evaluator,
        gold=None,
        metadata=metadata,
    )


def _task_from_hf(dataset: str, row: dict[str, Any], index: int) -> AgentBenchTask:
    if dataset == "browsecomp":
        return AgentBenchTask(
            task_id=f"browsecomp_{index + 1}",
            dataset=dataset,
            domain=DOMAINS[dataset],
            instruction=str(row["problem"]),
            evaluator=EXTERNAL_EVALUATOR[dataset],
            metadata={"encrypted": True, "source": "smolagents/browse_comp", "original_task_id": index},
        )
    if dataset == "webvoyager":
        task = str(row.get("task", row.get("text", row.get("instruction", ""))))
        start_url = str(row.get("domain", row.get("website", ""))).strip()
        task_id = str(row.get("identifier", row.get("id", f"webvoyager-{index:05d}")))
        return AgentBenchTask(
            task_id=task_id,
            dataset=dataset,
            domain=DOMAINS[dataset],
            instruction=f"{task}\nStart URL: {start_url}" if start_url else task,
            evaluator=EXTERNAL_EVALUATOR[dataset],
            metadata={"source": "btrabucco/web-voyager", "start_url": start_url, "original_task_id": task_id},
        )
    if dataset == "swe_bench_verified":
        task_id = str(row.get("instance_id", f"swe-{index:05d}"))
        return AgentBenchTask(
            task_id=task_id,
            dataset=dataset,
            domain=DOMAINS[dataset],
            instruction=str(row.get("problem_statement", "")),
            evaluator=EXTERNAL_EVALUATOR[dataset],
            metadata={
                "original_task_id": task_id,
                **{key: row.get(key) for key in (
                    "repo", "base_commit", "test_patch", "hints_text", "version",
                    "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit", "difficulty",
                )},
            },
        )
    if dataset == "tau2_bench":
        metadata = dict(row.get("metadata", {}) or {})
        args = dict(row.get("args", {}) or {})
        original = str(row.get("id", metadata.get("task_id", args.get("task_id", index))))
        domain = str(metadata.get("domain", args.get("domain", "unknown")))
        description = str(metadata.get("description") or f"Tau2 {domain} task {original}")
        return AgentBenchTask(
            task_id=f"tau2:{domain}:{original}",
            dataset=dataset,
            domain=DOMAINS[dataset],
            instruction=description,
            evaluator=EXTERNAL_EVALUATOR[dataset],
            metadata={"env": row.get("env", {}), "args": args, "metadata": metadata, "original_task_id": original},
        )
    raise KeyError(dataset)


def _task_from_explicit_file(
    dataset: str,
    row: dict[str, Any],
    index: int,
    source: Path,
    revision: str | None,
) -> AgentBenchTask:
    if "benchmark" in row and dataset != "webvoyager":
        return _from_general_agent(dataset, row, index, source, revision)
    if dataset == "webvoyager":
        task = str(row.get("ques", row.get("task", row.get("instruction", row.get("text", "")))))
        if not task.strip():
            task = str(row.get("text", ""))
        original = str(row.get("id", row.get("identifier", f"webvoyager-{index:05d}")))
        start_url = str(row.get("web", row.get("domain", row.get("start_url", ""))))
        return AgentBenchTask(
            task_id=original,
            dataset=dataset,
            domain=DOMAINS[dataset],
            instruction=f"{task}\nStart URL: {start_url}" if start_url else task,
            evaluator=EXTERNAL_EVALUATOR[dataset],
            metadata={
                **_provenance(source, revision, original),
                "source_file": "data/WebVoyager_data.jsonl",
                "start_url": start_url,
            },
        )
    # Minimal explicit fixture compatibility, useful for local adapter tests.
    original = str(row.get("task_id", row.get("id", row.get("instance_id", index))))
    instruction = str(row.get("question", row.get("instruction", row.get("problem", ""))))
    if not instruction:
        raise ValueError(f"{source}: row {index + 1} does not match the {dataset} task schema")
    evaluator = "semantic_qa" if dataset == "mathhay" and row.get("answer") is not None else EXTERNAL_EVALUATOR[dataset]
    return AgentBenchTask(
        task_id=original,
        dataset=dataset,
        domain=DOMAINS[dataset],
        instruction=instruction,
        evaluator=evaluator,
        gold=row.get("answer") if evaluator == "semantic_qa" else None,
        metadata=_provenance(source, revision, original),
    )


def prepare_dataset(
    dataset: str,
    out_dir: str | Path,
    *,
    source_path: str | Path | None = None,
    limit: int | None = None,
    seed: int = 0,
    token: str | None = None,
    overwrite: bool = True,
    source_revision: str | None = None,
) -> Path:
    if dataset not in AGENTBENCH_DATASETS:
        raise KeyError(f"unknown AgentBench dataset '{dataset}'")
    out_dir = Path(out_dir)
    limit = PAPER_SAMPLE_SIZES.get(dataset) if limit is None else limit
    if source_path is not None:
        source = _resolve_source(dataset, source_path)
        rows = _read_exact_rows(source)
        selected = rows if limit is None or limit >= len(rows) else rows[:limit]
        tasks = [
            _task_from_explicit_file(dataset, row, index, source, source_revision)
            for index, row in enumerate(selected)
        ]
    elif dataset in HF_DATASETS:
        name, split = HF_DATASETS[dataset]
        rows = list(_load_hf_dataset(name, split, token))
        tasks = [_task_from_hf(dataset, row, index) for index, row in enumerate(_sample(rows, limit, seed))]
    else:
        default_source = out_dir.parent / "external" / "General-AgentBench"
        source = _resolve_source(dataset, default_source)
        rows = _read_exact_rows(source)
        selected = rows if limit is None or limit >= len(rows) else rows[:limit]
        tasks = [
            _task_from_explicit_file(dataset, row, index, source, source_revision)
            for index, row in enumerate(selected)
        ]
    path = out_dir / f"{dataset}_tasks.jsonl"
    write_tasks(path, tasks, overwrite=overwrite)
    return path


def prepare_many(datasets: Sequence[str], out_dir: str | Path, **kwargs: Any) -> dict[str, str]:
    return {dataset: str(prepare_dataset(dataset, out_dir, **kwargs)) for dataset in datasets}
