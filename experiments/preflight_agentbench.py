"""Fail-fast VM preflight for pinned AgentBench execution dependencies."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs" / "agentbench.lock.json"


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _command_ok(command: list[str]) -> bool:
    try:
        return subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _configured(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().strip("\"'").lower()
    return not any(marker in normalized for marker in ("your_", "replace", "example", "changeme"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="browsecomp,mathhay,swe_bench_verified,terminal_bench,tau2_bench,mcp_bench",
    )
    parser.add_argument("--general-agentbench", default=str(ROOT / "data" / "external" / "General-AgentBench"))
    parser.add_argument("--webvoyager", default=str(ROOT / "data" / "external" / "WebVoyager"))
    parser.add_argument("--tasks-dir", default=str(ROOT / "data" / "agentbench"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    general_root = Path(args.general_agentbench)
    if any(dataset != "webvoyager" for dataset in datasets):
        head = _git_head(general_root)
        expected = lock["general_agentbench"]["commit"]
        check("general_agentbench_commit", head == expected, f"found={head} expected={expected}")
        check(
            "general_agentbench_runner",
            (general_root / lock["general_agentbench"]["runner"]).is_file(),
            lock["general_agentbench"]["runner"],
        )
        check("python_litellm", importlib.util.find_spec("litellm") is not None, "pip install -e general_agent")
    if "webvoyager" in datasets:
        web_root = Path(args.webvoyager)
        head = _git_head(web_root)
        expected = lock["webvoyager"]["commit"]
        check("webvoyager_commit", head == expected, f"found={head} expected={expected}")
        check("python_selenium", importlib.util.find_spec("selenium") is not None, "WebVoyager requirements")
        chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
        check("chrome", chrome is not None, str(chrome))
    if {"swe_bench_verified", "terminal_bench"} & set(datasets):
        check("docker", _command_ok(["docker", "info"]), "docker info")
    if "tau2_bench" in datasets:
        check("python_tau2", importlib.util.find_spec("tau2") is not None, "pip install -e benchmarks/tau2-bench")
    if "mcp_bench" in datasets:
        mcp_root = general_root / "benchmarks" / "mcp-bench" / "mcp_servers"
        check("mcp_install_script", (mcp_root / "install.sh").is_file(), str(mcp_root / "install.sh"))
        key_file = mcp_root / "api_key"
        required = ("NPS_API_KEY", "NASA_API_KEY", "HF_TOKEN", "GOOGLE_MAPS_API_KEY", "NCI_API_KEY")
        key_text = key_file.read_text(encoding="utf-8") if key_file.is_file() else ""
        missing = []
        for key in required:
            values = [
                line.split("=", 1)[1].strip()
                for line in key_text.splitlines()
                if "=" in line and line.split("=", 1)[0].removeprefix("export ").strip() == key
            ]
            if not values or not _configured(values[-1]):
                missing.append(key)
        check("mcp_api_keys", not missing, "missing=" + ",".join(missing) if missing else str(key_file))
    if "browsecomp" in datasets:
        check("serper_key", _configured(os.environ.get("SERPER_API_KEY")), "SERPER_API_KEY")
    for dataset in datasets:
        task_path = Path(args.tasks_dir) / f"{dataset}_tasks.jsonl"
        check(f"tasks_{dataset}", task_path.is_file() and task_path.stat().st_size > 0, str(task_path))

    failed = [item for item in checks if not item["ok"]]
    print(json.dumps({"ok": not failed, "checks": checks}, indent=2))
    if failed and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
