"""Create a deterministic native General AgentBench task subset for smoke/calibration runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _row_id(row: dict, index: int) -> str:
    task = row.get("task", {}) or {}
    return str(task.get("id", task.get("instance_id", row.get("id", index))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default=None, help="Optional text file with one original task ID per line.")
    args = parser.parse_args(argv)
    if args.limit is None and args.ids is None:
        parser.error("provide --limit or --ids")
    rows = json.loads(Path(args.source).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("native source must be a JSON list")
    if args.ids:
        wanted = {
            line.strip() for line in Path(args.ids).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        selected = [row for index, row in enumerate(rows) if _row_id(row, index) in wanted]
        found = {_row_id(row, index) for index, row in enumerate(selected)}
        missing = wanted - found
        if missing:
            raise ValueError(f"native source is missing requested IDs: {', '.join(sorted(missing))}")
    else:
        selected = rows[: max(0, args.limit)]
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(selected)} native tasks -> {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
