"""Fit a Platt calibrator from a held-out, officially evaluated selector run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acdan.agentbench.calibration import PlattCalibrator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-result", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args(argv)
    result = json.loads(Path(args.calibration_result).read_text(encoding="utf-8"))
    rows = result.get("per_task", [])
    if not rows:
        raise ValueError("calibration result has no per_task rows")
    confidence = [float(row.get("raw_confidence", row.get("confidence", 0.5))) for row in rows]
    correct = [bool(row["selected_correct"]) for row in rows]
    calibrator = PlattCalibrator().fit(confidence, correct, epochs=args.epochs, lr=args.lr)
    calibrator.save(args.out)
    print(json.dumps({
        "out": args.out,
        "n_calibration_tasks": len(rows),
        "slope": calibrator.slope,
        "intercept": calibrator.intercept,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
