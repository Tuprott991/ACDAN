"""Materialize pinned WebVoyager multimodal-judge verdicts as official scores."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from acdan.agentbench.artifacts import OfficialScore, write_jsonl


def _load_evaluator(checkout: Path):
    path = checkout / "evaluation" / "auto_eval.py"
    if not path.is_file():
        raise RuntimeError(f"missing pinned WebVoyager evaluator: {path}")
    spec = importlib.util.spec_from_file_location("pinned_webvoyager_auto_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", default=str(ROOT / "data" / "external" / "WebVoyager"))
    parser.add_argument("--source", required=True, help="Directory containing pass_1..pass_K.")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-attached-imgs", type=int, default=15)
    parser.add_argument("--evaluator-version", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing judge API key environment variable {args.api_key_env}")
    module = _load_evaluator(Path(args.checkout))
    client = module.OpenAI(api_key=api_key)
    existing = {}
    output = Path(args.out)
    if args.resume and output.is_file():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[(str(row["task_id"]), str(row["candidate_id"]))] = row
    scores = dict(existing)
    for pass_index in range(1, args.k + 1):
        pass_dir = Path(args.source) / f"pass_{pass_index}"
        for task_dir in sorted(pass_dir.glob("task*")):
            task_id = task_dir.name.removeprefix("task")
            candidate_id = f"pass_{pass_index}"
            key = (task_id, candidate_id)
            if key in scores:
                continue
            verdict = module.auto_eval_by_gpt4v(
                str(task_dir), client, args.judge_model, args.max_attached_imgs
            )
            if verdict is None:
                raise RuntimeError(f"WebVoyager judge returned no verdict for {task_id}/{candidate_id}")
            scores[key] = OfficialScore(
                task_id=task_id,
                candidate_id=candidate_id,
                dataset="webvoyager",
                score=float(verdict),
                correct=bool(verdict),
                evaluator="external_webvoyager",
                evaluator_version=args.evaluator_version,
                raw={"native_verdict": int(verdict), "judge_model": args.judge_model},
            ).to_json()
            write_jsonl(output, [scores[item] for item in sorted(scores)])
    write_jsonl(output, [scores[item] for item in sorted(scores)])
    print(f"wrote {len(scores)} WebVoyager official scores -> {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
