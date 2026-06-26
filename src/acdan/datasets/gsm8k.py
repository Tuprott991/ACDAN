"""GSM8K / MATH as answer-selection over pre-generated candidates.

In the tool/operator-selection DTO framing, math is run as a (H=1, V=K) choice:
sample K candidate solutions with the policy first (offline preprocessing — see
``experiments/PLAN.md`` Step 1), extract each candidate's final answer, then let
DTO + PRM pick the best. Gold is the reference answer; checker is ``outcome_exact``.

Expected JSONL line::

    {"question": "...", "candidates": ["ans1","ans2",...], "answer": "42"}

(`candidates` are the extracted final answers of the K sampled solutions.)
"""

from __future__ import annotations

import json
from typing import Iterable, Optional

from acdan.datasets.base import RawTask


class GSM8KDataset:
    def __init__(self, path: Optional[str], limit: Optional[int] = None):
        if not path:
            raise ValueError(
                "GSM8K adapter needs --data-path to a candidates JSONL "
                "(see experiments/PLAN.md Step 1 to generate it)."
            )
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
                cands = [str(c) for c in d["candidates"]]
                if not cands:
                    continue
                yield RawTask(
                    task_id=d.get("task_id", f"gsm8k-{i:05d}"),
                    prompt=d["question"],
                    vocab=tuple(cands),
                    horizon=1,
                    gold=str(d["answer"]),
                    family="gsm8k",
                    difficulty=float(d.get("difficulty", 0.5)),
                    metadata={"n_candidates": len(cands)},
                )
                n += 1
                if self.limit and n >= self.limit:
                    return
