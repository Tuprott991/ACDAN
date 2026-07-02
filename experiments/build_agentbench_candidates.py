"""Group AgentBench task manifests and prediction lines into selector input.

Prediction JSONL accepts either one candidate per line:

{"task_id": "...", "candidate_id": "0", "final_answer": "...", "is_correct": true}

or grouped rows:

{"task_id": "...", "candidates": [{...}, {...}]}
"""

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


def _read_predictions(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_id = str(row["task_id"])
            if "candidates" in row:
                for i, cand in enumerate(row["candidates"]):
                    cand = dict(cand)
                    cand.setdefault("candidate_id", str(i))
                    grouped[task_id].append(cand)
            else:
                cand = dict(row)
                cand.pop("task_id", None)
                cand.setdefault("candidate_id", str(len(grouped[task_id])))
                cand.setdefault("source_line", line_idx)
                grouped[task_id].append(cand)
    return grouped


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build AgentBench candidate JSONL for ACDAN selection.")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-candidates", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    tasks = read_tasks(args.tasks)
    preds = _read_predictions(args.predictions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    skipped = 0
    with out.open("w", encoding="utf-8") as fh:
        for task in tasks:
            candidates = preds.get(task.task_id, [])
            if len(candidates) < args.min_candidates:
                skipped += 1
                continue
            fh.write(json.dumps({
                "task": task.to_json(),
                "candidates": candidates,
            }, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"wrote {n} candidate tasks -> {out} (skipped {skipped})")


if __name__ == "__main__":
    main()
