"""Prepare score-blind task manifests from explicit official task files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acdan.agentbench.adapters import AGENTBENCH_DATASETS, PAPER_SAMPLE_SIZES, prepare_dataset


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Prepare AgentBench task JSONL files.")
    ap.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset keys or 'all'.",
    )
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "agentbench"))
    ap.add_argument("--source-path", default=None, help="Local source for one archive-backed dataset.")
    ap.add_argument(
        "--source-revision",
        default=None,
        help="Pinned source git revision recorded in task provenance.",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--no-overwrite", action="store_true")
    args = ap.parse_args(argv)

    datasets = list(AGENTBENCH_DATASETS) if args.datasets == "all" else [
        item.strip() for item in args.datasets.split(",") if item.strip()
    ]
    out_dir = Path(args.out_dir)
    for dataset in datasets:
        limit = args.limit if args.limit is not None else PAPER_SAMPLE_SIZES.get(dataset)
        path = prepare_dataset(
            dataset,
            out_dir,
            source_path=args.source_path,
            limit=limit,
            seed=args.seed,
            token=args.hf_token,
            overwrite=not args.no_overwrite,
            source_revision=args.source_revision,
        )
        print(f"{dataset}: wrote {path}")


if __name__ == "__main__":
    main()
