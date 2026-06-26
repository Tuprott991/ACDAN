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

import json
from typing import Iterable, Optional

from acdan.datasets.base import RawTask


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
                yield RawTask(
                    task_id=d.get("task_id", f"bfcl-{i:05d}"),
                    prompt=d["prompt"],
                    vocab=tuple(tools),
                    horizon=int(d.get("horizon", len(gold))),
                    gold=gold,
                    family="bfcl",
                    difficulty=float(d.get("difficulty", 0.5)),
                    action_templates=d.get("action_templates", {}) or {},
                    metadata={"n_tools": len(tools)},
                )
                n += 1
                if self.limit and n >= self.limit:
                    return
