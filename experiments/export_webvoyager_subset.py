"""Export the exact ACDAN WebVoyager task IDs in the native runner format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acdan.agentbench.adapters import read_tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--native-source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    wanted = [task.task_id for task in read_tasks(args.manifest)]
    native_rows = {}
    for line in Path(args.native_source).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            native_rows[str(row["id"])] = row
    missing = [task_id for task_id in wanted if task_id not in native_rows]
    if missing:
        raise ValueError(f"native WebVoyager source is missing {len(missing)} manifest task IDs")
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for task_id in wanted:
            fh.write(json.dumps(native_rows[task_id], ensure_ascii=False) + "\n")
    print(f"wrote {len(wanted)} native WebVoyager tasks -> {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
