"""Import K native WebVoyager runs into blind ACDAN candidate artifacts."""

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
from acdan.agentbench.artifacts import TrajectoryArtifact, TrajectoryCost, write_jsonl


def _answer(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    str(item.get("text", "")) if isinstance(item, dict) else str(item)
                    for item in content
                )
            if str(content).strip():
                return str(content).strip()
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Directory containing pass_1..pass_K outputs.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--candidates-out", required=True)
    parser.add_argument("--trajectories-out", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)

    tasks = {task.task_id: task for task in read_tasks(args.tasks)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectories = []
    for pass_index in range(1, args.k + 1):
        pass_dir = Path(args.source) / f"pass_{pass_index}"
        for task_dir in sorted(pass_dir.glob("task*")):
            interaction = task_dir / "interact_messages.json"
            if not interaction.is_file():
                continue
            task_id = task_dir.name.removeprefix("task")
            if task_id not in tasks:
                if args.allow_incomplete:
                    continue
                raise RuntimeError(f"WebVoyager output task {task_id} is absent from {args.tasks}")
            messages = json.loads(interaction.read_text(encoding="utf-8"))
            screenshots = sorted(task_dir.glob("screenshot*.png"))
            artifact = TrajectoryArtifact(
                task_id=task_id,
                candidate_id=f"pass_{pass_index}",
                dataset="webvoyager",
                final_answer=_answer(messages),
                trajectory=[dict(message) for message in messages if isinstance(message, dict)],
                artifact_paths={
                    "interaction": str(interaction),
                    "last_screenshot": str(screenshots[-1]) if screenshots else "",
                },
                cost=TrajectoryCost(model_calls=sum(
                    isinstance(message, dict) and message.get("role") == "assistant"
                    for message in messages
                )),
                generator={"protocol": "webvoyager-native", "pass": pass_index},
            )
            row = artifact.to_json()
            trajectories.append(row)
            candidate = dict(row)
            candidate.pop("task_id")
            candidate.pop("dataset")
            grouped[task_id].append(candidate)

    candidates = []
    for task_id, task in tasks.items():
        values = grouped.get(task_id, [])
        if not args.allow_incomplete and len(values) != args.k:
            raise RuntimeError(f"{task_id}: expected {args.k} WebVoyager candidates, found {len(values)}")
        if values:
            candidates.append({"task": task.to_json(), "candidates": values})
    write_jsonl(args.candidates_out, candidates)
    write_jsonl(args.trajectories_out, trajectories)
    print(f"imported {len(candidates)} WebVoyager tasks and {len(trajectories)} trajectories")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
