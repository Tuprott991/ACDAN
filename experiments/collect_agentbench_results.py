"""Collect reportable AgentBench results and paired bootstrap comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acdan.agentbench.metrics import paired_bootstrap_delta


FIELDS = (
    "dataset", "method", "n_tasks", "selected_accuracy", "selected_score",
    "pass_at_k", "oracle_score", "verification_gap", "recovery_rate",
    "oracle_regret", "ece", "brier", "nll", "aurc", "coverage",
    "selective_accuracy", "mean_selection_prompt_tokens",
    "mean_selection_latency_s", "mean_samples", "mean_verified_candidates",
    "mean_generation_tokens_per_candidate",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--comparisons-out", required=True)
    parser.add_argument("--reference", default="bon")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    records = []
    full = {}
    hashes_by_dataset: dict[str, set[str]] = {}
    for raw_path in args.results:
        path = Path(raw_path)
        value = json.loads(path.read_text(encoding="utf-8"))
        summary = dict(value.get("summary", {}) or {})
        dataset = str(value.get("per_task", [{}])[0].get("dataset", summary.get("dataset", "")))
        method = str(summary.get("method", path.stem.rsplit("_", 1)[-1]))
        if not dataset:
            raise ValueError(f"cannot determine dataset for {path}")
        record = {field: summary.get(field, "") for field in FIELDS}
        record.update({"dataset": dataset, "method": method})
        records.append(record)
        full[(dataset, method)] = value
        hashes_by_dataset.setdefault(dataset, set()).add(str(summary.get("candidate_file_sha256", "")))
    mismatched = [dataset for dataset, hashes in hashes_by_dataset.items() if len(hashes - {""}) > 1]
    if mismatched:
        raise ValueError(f"methods did not share one candidate pool for: {', '.join(mismatched)}")

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: (str(row["dataset"]), str(row["method"]))))

    comparisons = []
    for (dataset, method), value in sorted(full.items()):
        if method == args.reference or (dataset, args.reference) not in full:
            continue
        delta = paired_bootstrap_delta(
            value["per_task"],
            full[(dataset, args.reference)]["per_task"],
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
        comparisons.append({
            "dataset": dataset,
            "method": method,
            "reference": args.reference,
            **delta,
        })
    comparisons_path = Path(args.comparisons_out)
    comparisons_path.parent.mkdir(parents=True, exist_ok=True)
    comparisons_path.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} rows -> {csv_path}")
    print(f"wrote {len(comparisons)} paired comparisons -> {comparisons_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
