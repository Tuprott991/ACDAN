"""GAIA / AgentBench as tool-sequence tasks judged by an LLM (Claude).

GAIA outcomes are open-ended, so correctness uses the Claude judge checker
(``build_outcome_checker('gaia')``) rather than string match. Run this family
last — it is the hardest to wire and the most expensive to judge.

Expected JSONL line::

    {"prompt": "...", "tools": ["browse","search","python", ...],
     "gold": "the reference final answer", "horizon": 3}
"""

from __future__ import annotations

import json
from typing import Iterable, Optional

from acdan.datasets.base import RawTask


class GAIADataset:
    def __init__(self, path: Optional[str], limit: Optional[int] = None):
        if not path:
            raise ValueError("GAIA adapter needs --data-path to a JSONL.")
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
                tools = [str(t) for t in d.get("tools", ["search", "browse", "python", "answer"])]
                yield RawTask(
                    task_id=d.get("task_id", f"gaia-{i:05d}"),
                    prompt=d["prompt"],
                    vocab=tuple(tools),
                    horizon=int(d.get("horizon", 3)),
                    gold=str(d.get("gold", "")),
                    family="gaia",
                    difficulty=float(d.get("difficulty", 0.7)),
                    metadata={"gold": str(d.get("gold", ""))},
                )
                n += 1
                if self.limit and n >= self.limit:
                    return
