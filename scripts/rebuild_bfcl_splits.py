"""Rebuild non-overlapping BFCL dev/test files from checked-in bfcl_full.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from acdan.datasets.bfcl import stratified_bfcl_split


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", default="data/bfcl_full.jsonl")
    parser.add_argument("--dev", default="data/bfcl_dev.jsonl")
    parser.add_argument("--test", default="data/bfcl_test.jsonl")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    args = parser.parse_args()
    rows = _read_jsonl(Path(args.full))
    dev, test = stratified_bfcl_split(rows, dev_fraction=args.dev_fraction)
    _write_jsonl(Path(args.dev), dev)
    _write_jsonl(Path(args.test), test)
    print(f"BFCL split: full={len(rows)} dev={len(dev)} test={len(test)} overlap=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
