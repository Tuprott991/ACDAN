"""Task adapters for General AgentBench-style evaluation.

These adapters prepare *tasks*, not candidate answers. The paper-style protocol
requires an agent to generate trajectories in the appropriate environment, then
the selection/evaluation layer can measure pass@K and self-choice.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


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
    "webvoyager": ("agentorg/webvoyager", "test"),
    "swe_bench_verified": ("SWE-bench/SWE-bench_Verified", "test"),
    "tau2_bench": ("Genteki/tau2-bench", "train"),
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
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tasks.append(AgentBenchTask(
                task_id=str(d["task_id"]),
                dataset=str(d["dataset"]),
                domain=str(d.get("domain", DOMAINS.get(str(d["dataset"]), "unknown"))),
                instruction=str(d["instruction"]),
                evaluator=str(d.get("evaluator", "external")),
                gold=d.get("gold"),
                metadata=dict(d.get("metadata", {}) or {}),
            ))
    return tasks


def _sample(rows: Sequence[Any], limit: int | None, seed: int) -> list[Any]:
    rows = list(rows)
    if limit is None or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(rows)), limit))
    return [rows[i] for i in idxs]


def _load_hf_dataset(name: str, split: str, token: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Install dataset dependencies first: .venv/bin/pip install -r requirements-data.txt"
        ) from exc
    return load_dataset(name, split=split, token=token)


def _task_from_hf(dataset: str, row: dict[str, Any], i: int) -> AgentBenchTask:
    if dataset == "browsecomp":
        return AgentBenchTask(
            task_id=f"browsecomp-{i:05d}",
            dataset=dataset,
            domain=DOMAINS[dataset],
            instruction=str(row["problem"]),
            evaluator="semantic_qa",
            gold=str(row["answer"]),
            metadata={
                "problem_topic": row.get("problem_topic", ""),
                "source": "smolagents/browse_comp",
            },
        )
    if dataset == "webvoyager":
        text = str(row.get("text", row.get("instruction", row)))
        return AgentBenchTask(
            task_id=f"webvoyager-{i:05d}",
            dataset=dataset,
            domain=DOMAINS[dataset],
            instruction=text,
            evaluator="external_webvoyager",
            metadata={"source": "agentorg/webvoyager", "raw": row},
        )
    if dataset == "swe_bench_verified":
        iid = str(row.get("instance_id", f"swe-{i:05d}"))
        return AgentBenchTask(
            task_id=iid,
            dataset=dataset,
            domain=DOMAINS[dataset],
            instruction=str(row.get("problem_statement", "")),
            evaluator="external_swe_bench",
            gold=str(row.get("patch", "")),
            metadata={k: row.get(k) for k in (
                "repo", "base_commit", "test_patch", "hints_text", "version",
                "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit", "difficulty",
            )},
        )
    if dataset == "tau2_bench":
        metadata = dict(row.get("metadata", {}) or {})
        args = dict(row.get("args", {}) or {})
        tid = str(row.get("id", metadata.get("task_id", args.get("task_id", f"tau2-{i:05d}"))))
        return AgentBenchTask(
            task_id=tid,
            dataset=dataset,
            domain=DOMAINS[dataset],
            instruction=str(row.get("scenario", metadata.get("description", ""))),
            evaluator="external_tau2",
            metadata={"env": row.get("env", {}), "args": args, "metadata": metadata},
        )
    raise KeyError(dataset)


def _iter_jsonish(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_file():
        files = [path]
    else:
        files = []
        for pattern in ("*.jsonl", "*.json", "*.yaml", "*.yml"):
            files.extend(path.rglob(pattern))
    for file in sorted(files):
        try:
            if file.suffix == ".jsonl":
                for line in file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        value = json.loads(line)
                        if isinstance(value, dict):
                            yield value
            elif file.suffix == ".json":
                value = json.loads(file.read_text(encoding="utf-8"))
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item
                elif isinstance(value, dict):
                    yield value
            elif file.suffix in {".yaml", ".yml"}:
                value = yaml.safe_load(file.read_text(encoding="utf-8"))
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item
                elif isinstance(value, dict):
                    yield value
        except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError):
            continue


def _first_text(row: dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value
    return json.dumps(row, ensure_ascii=False)[:4000]


def _task_from_generic(dataset: str, row: dict[str, Any], i: int) -> AgentBenchTask:
    tid = str(row.get("task_id", row.get("id", row.get("instance_id", f"{dataset}-{i:05d}"))))
    if dataset == "mathhay":
        gold = row.get("answer", row.get("gold", row.get("final_answer")))
        return AgentBenchTask(
            task_id=tid,
            dataset=dataset,
            domain=DOMAINS[dataset],
            instruction=_first_text(row, ("question", "problem", "prompt", "instruction")),
            evaluator="semantic_qa",
            gold=gold,
            metadata={"raw": row},
        )
    evaluator = {
        "terminal_bench": "external_terminal_bench",
        "mcp_bench": "external_mcp_bench",
    }[dataset]
    return AgentBenchTask(
        task_id=tid,
        dataset=dataset,
        domain=DOMAINS[dataset],
        instruction=_first_text(row, ("instruction", "task", "prompt", "description", "question")),
        evaluator=evaluator,
        gold=row.get("answer", row.get("gold")),
        metadata={"raw": row},
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
) -> Path:
    if dataset not in AGENTBENCH_DATASETS:
        raise KeyError(f"unknown AgentBench dataset '{dataset}'")
    out_dir = Path(out_dir)
    limit = PAPER_SAMPLE_SIZES.get(dataset) if limit is None else limit
    if dataset in HF_DATASETS:
        name, split = HF_DATASETS[dataset]
        rows = list(_load_hf_dataset(name, split, token))
        tasks = [_task_from_hf(dataset, row, i) for i, row in enumerate(_sample(rows, limit, seed))]
    else:
        if source_path is None:
            source_path = out_dir.parent / "raw" / "agentic_benchmarks" / dataset
        rows = list(_iter_jsonish(Path(source_path)))
        if not rows:
            raise RuntimeError(
                f"no parseable JSON/YAML tasks found for {dataset} under {source_path}"
            )
        tasks = [
            _task_from_generic(dataset, row, i)
            for i, row in enumerate(_sample(rows, limit, seed))
        ]
    path = out_dir / f"{dataset}_tasks.jsonl"
    write_tasks(path, tasks, overwrite=overwrite)
    return path


def prepare_many(
    datasets: Sequence[str],
    out_dir: str | Path,
    **kwargs: Any,
) -> dict[str, str]:
    return {dataset: str(prepare_dataset(dataset, out_dir, **kwargs)) for dataset in datasets}
