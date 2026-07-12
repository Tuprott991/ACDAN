"""Clone and verify the pinned official AgentBench execution repositories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "configs" / "agentbench.lock.json"


def _run(command: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    print("+ " + " ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def _head(path: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return proc.stdout.strip()


def _clone_pinned(
    *,
    name: str,
    url: str,
    commit: str,
    destination: Path,
    sparse_general_agentbench: bool,
    dry_run: bool,
) -> None:
    if destination.exists():
        if not (destination / ".git").exists():
            raise RuntimeError(f"{destination} exists but is not a git checkout")
        current = _head(destination)
        if current != commit:
            raise RuntimeError(
                f"{name} is at {current}, expected pinned commit {commit}. "
                "Move the checkout aside and rerun setup."
            )
        print(f"verified {name}: {commit}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(destination)],
        dry_run=dry_run,
    )
    if dry_run:
        return
    if sparse_general_agentbench:
        _run(["git", "sparse-checkout", "init", "--no-cone"], cwd=destination)
        sparse_file = destination / ".git" / "info" / "sparse-checkout"
        sparse_file.write_text(
            "/*\n!/*/\n/README.md\n/LICENSE\n/general_agent/\n"
            "!/general_agent/traces/\n!/general_agent/results/\n/benchmarks/\n",
            encoding="utf-8",
        )
    _run(["git", "checkout", "--detach", commit], cwd=destination)
    current = _head(destination)
    if current != commit:
        raise RuntimeError(f"failed to pin {name}: got {current}, expected {commit}")
    print(f"installed {name}: {destination} @ {commit}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--external-dir", default=str(ROOT / "data" / "external"))
    parser.add_argument(
        "--repos",
        default="general_agentbench,webvoyager",
        help="Comma-separated: general_agentbench,webvoyager",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
    selected = {item.strip() for item in args.repos.split(",") if item.strip()}
    unknown = selected - {"general_agentbench", "webvoyager"}
    if unknown:
        raise ValueError(f"unknown official repository key(s): {', '.join(sorted(unknown))}")
    external = Path(args.external_dir)
    if "general_agentbench" in selected:
        spec = lock["general_agentbench"]
        _clone_pinned(
            name="General-AgentBench",
            url=spec["url"],
            commit=spec["commit"],
            destination=external / "General-AgentBench",
            sparse_general_agentbench=True,
            dry_run=args.dry_run,
        )
    if "webvoyager" in selected:
        spec = lock["webvoyager"]
        _clone_pinned(
            name="WebVoyager",
            url=spec["url"],
            commit=spec["commit"],
            destination=external / "WebVoyager",
            sparse_general_agentbench=False,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
