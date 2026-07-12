"""BFCL / ToolBench as tool-sequence selection (the cleanest fit for ACDAN).

Expected JSONL line::

    {"prompt": "user request ...",
     "tools": ["get_weather","search_web","send_email", ...],   # candidate set V
     "gold":  ["get_weather"],                                   # reference call(s)
     "horizon": 1,
     "action_templates": {"get_weather": "call get_weather(...)"}}  # optional

``vocab`` = the available tools; ``gold`` = the reference tool-name sequence;
checker is ``outcome_tool_sequence``. ``horizon`` defaults to len(gold).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Optional

from acdan.datasets.base import RawTask


def stratified_bfcl_split(
    rows: list[dict[str, Any]], dev_fraction: float = 0.2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stable, category-stratified, non-overlapping BFCL dev/test split."""
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1")
    ids = [str(row.get("task_id", "")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("BFCL task_id values must be globally unique before splitting")
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(str(row.get("category", "unknown")), []).append(row)
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for category, category_rows in sorted(by_category.items()):
        ordered = sorted(
            category_rows,
            key=lambda row: hashlib.blake2b(
                f"{category}:{row['task_id']}:acdan-bfcl-v1".encode("utf-8"),
                digest_size=16,
            ).digest(),
        )
        n_dev = min(len(ordered) - 1, max(1, round(len(ordered) * dev_fraction)))
        dev.extend(ordered[:n_dev])
        test.extend(ordered[n_dev:])
    def output_order(row: dict[str, Any]) -> bytes:
        return hashlib.blake2b(
            f"{row['task_id']}:acdan-bfcl-output-v1".encode("utf-8"),
            digest_size=16,
        ).digest()

    dev.sort(key=output_order)
    test.sort(key=output_order)
    return dev, test


class BFCLDataset:
    def __init__(self, path: Optional[str], limit: Optional[int] = None):
        if not path:
            raise ValueError("BFCL/ToolBench adapter needs --data-path to a JSONL.")
        self.path = path
        self.limit = limit

    def tasks(self) -> Iterable[RawTask]:
        n = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                tools = [str(t) for t in d["tools"]]
                gold = d["gold"]
                gold = [gold] if isinstance(gold, str) else [str(g) for g in gold]
                gold_calls = d.get("gold_calls", [])
                yield RawTask(
                    task_id=d.get("task_id", f"bfcl-{i:05d}"),
                    prompt=d["prompt"],
                    vocab=tuple(tools),
                    horizon=int(d.get("horizon", len(gold))),
                    gold=gold,
                    family="bfcl",
                    difficulty=float(d.get("difficulty", 0.5)),
                    action_templates=d.get("action_templates", {}) or {},
                    metadata={
                        "n_tools": len(tools),
                        "gold_calls": gold_calls,
                        "source": d.get("source"),
                        "category": d.get("category"),
                    },
                )
                n += 1
                if self.limit and n >= self.limit:
                    return
