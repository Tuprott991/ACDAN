"""Run pinned General AgentBench parallel scaling through its native harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs" / "agentbench.lock.json"


def _head(path: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, text=True,
        stdout=subprocess.PIPE,
    )
    return proc.stdout.strip()


def build_command(args: argparse.Namespace, lock: dict) -> tuple[list[str], Path]:
    if args.dataset == "webvoyager":
        raise RuntimeError(
            "WebVoyager is not supported by General AgentBench's unified runner. "
            "Use its pinned native checkout and evaluation/run_eval.sh as documented in PLAN.md."
        )
    dataset_spec = lock["datasets"][args.dataset]
    checkout = Path(args.checkout)
    expected = lock["general_agentbench"]["commit"]
    if not checkout.exists():
        raise RuntimeError(f"missing official checkout: {checkout}; run scripts/setup_agentbench_official.py")
    current = _head(checkout)
    if current != expected:
        raise RuntimeError(f"General-AgentBench commit mismatch: {current}, expected {expected}")
    agent_dir = checkout / "general_agent"
    script = agent_dir / "scripts" / "run_parallel_scaling.sh"
    command = [
        "bash", str(script),
        "--agent-dir", str(agent_dir),
        "--model", args.model,
        "--model-name", args.model_name,
        "--benchmark", dataset_spec["upstream_benchmark"],
        "--num-passes", str(args.k),
        "--base-seed", str(args.base_seed),
        "--temperature", str(args.temperature),
        "--mode", args.mode,
        "--distraction", args.distraction,
        "--compress-tools", "yes" if args.compress_tools else "no",
        "--tool-desc-max-len", str(args.tool_desc_max_len),
        "--output-dir", str(Path(args.output_dir).resolve()),
        "--env-file", str(Path(args.env_file).resolve()),
    ]
    if args.max_tokens:
        command.extend(["--max-tokens", str(args.max_tokens)])
    if args.task_timeout:
        command.extend(["--task-timeout", str(args.task_timeout)])
    if args.max_steps:
        command.extend(["--max-steps", str(args.max_steps)])
    if args.task_file:
        command.extend(["--task-file", str(Path(args.task_file).resolve())])
    command.append("--no-resume" if args.no_resume else "--resume")
    if not args.execute:
        command.append("--dry-run")
    return command, agent_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=[
        "browsecomp", "mathhay", "swe_bench_verified", "terminal_bench",
        "tau2_bench", "mcp_bench", "webvoyager",
    ])
    parser.add_argument("--model", required=True, help="LiteLLM model route used by General AgentBench.")
    parser.add_argument("--model-name", required=True, help="Filesystem-safe stable model tag.")
    parser.add_argument("--checkout", default=str(ROOT / "data" / "external" / "General-AgentBench"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--mode", choices=["sequential", "tmux"], default="sequential")
    parser.add_argument("--distraction", default="all")
    parser.add_argument("--tool-desc-max-len", type=int, default=75)
    parser.add_argument("--compress-tools", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--task-timeout", type=int, default=None)
    parser.add_argument("--task-file", default=None, help="Optional native General AgentBench task subset file.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Execute; without this flag, upstream runs in dry-run mode.")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be >= 1")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    command, cwd = build_command(args, lock)
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
