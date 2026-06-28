"""Rewrite grouped math candidate files with a neutral candidate order.

This is useful for legacy candidate files generated before `--order sample`
became the default.  It preserves candidates, solutions, counts, and first
indices, then rewrites the candidate list in first-sample or shuffled order.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _candidate_key(row: dict, answer: str, fallback: int) -> int:
    first = row.get("candidate_first_indices", {}) or {}
    try:
        return int(first.get(str(answer), fallback))
    except (TypeError, ValueError):
        return fallback


def reorder_row(row: dict, order: str, seed: int, row_idx: int) -> dict:
    out = dict(row)
    candidates = [str(c) for c in row.get("candidates", [])]
    indexed = list(enumerate(candidates))
    if order == "sample":
        indexed.sort(key=lambda item: _candidate_key(row, item[1], item[0]))
    elif order == "shuffle":
        indexed.sort(key=lambda item: _candidate_key(row, item[1], item[0]))
        random.Random(seed + row_idx).shuffle(indexed)
    elif order == "plurality":
        counts = row.get("candidate_counts", {}) or {}
        indexed.sort(
            key=lambda item: (
                -float(counts.get(str(item[1]), 1.0)),
                _candidate_key(row, item[1], item[0]),
            )
        )
    else:  # pragma: no cover - argparse constrains it
        raise ValueError(order)
    out["candidates"] = [answer for _, answer in indexed]
    out["candidate_order"] = order
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Reorder grouped candidate JSONL.")
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--order", choices=["sample", "shuffle", "plurality"], default="sample")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    args = parser.parse_args()

    src = Path(args.inp)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open("r", encoding="utf-8") as fh, dst.open("w", encoding="utf-8") as out:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            row = reorder_row(json.loads(line), args.order, args.shuffle_seed, i)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
