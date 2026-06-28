"""Download and prepare benchmark data for ACDAN experiments.

This script keeps the core package offline-friendly while giving experiment VMs a
single reproducible entrypoint for data setup.  It reads `.env` via `load_env()`
so private/gated Hugging Face access can use `HF_TOKEN` without exporting it in
the shell.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from acdan.datasets.math_answer import extract_final_answer


ROOT = Path(__file__).resolve().parents[1]


def load_env(path: str | Path = ROOT / ".env") -> dict[str, str]:
    """Load KEY=VALUE pairs into os.environ without overriding existing values."""

    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
        loaded[key] = os.environ.get(key, value)
    return loaded


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised on bare envs
        raise RuntimeError(
            "Install optional data dependencies first: "
            ".venv/bin/pip install -r requirements-data.txt"
        ) from exc
    return load_dataset


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]], overwrite: bool) -> int:
    if path.exists() and not overwrite:
        print(f"skip existing {path}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows -> {path}")
    return n


def _gsm_answer(answer: str) -> str:
    text = str(answer)
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()
    return extract_final_answer(text)


def prepare_math(out_dir: Path, overwrite: bool, token: str | None) -> dict[str, int]:
    load_dataset = _require_datasets()
    counts: dict[str, int] = {}

    gsm_train = load_dataset("openai/gsm8k", "main", split="train", token=token)
    gsm_test = load_dataset("openai/gsm8k", "main", split="test", token=token)
    counts["gsm8k_train"] = _write_jsonl(
        out_dir / "gsm8k_train.jsonl",
        (
            {
                "task_id": f"gsm8k-train-{i:05d}",
                "question": r["question"],
                "answer": _gsm_answer(r["answer"]),
                "source": "openai/gsm8k",
                "split": "train",
            }
            for i, r in enumerate(gsm_train)
        ),
        overwrite,
    )
    counts["gsm8k_test"] = _write_jsonl(
        out_dir / "gsm8k_test.jsonl",
        (
            {
                "task_id": f"gsm8k-test-{i:05d}",
                "question": r["question"],
                "answer": _gsm_answer(r["answer"]),
                "source": "openai/gsm8k",
                "split": "test",
            }
            for i, r in enumerate(gsm_test)
        ),
        overwrite,
    )

    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test", token=token)
    counts["math500"] = _write_jsonl(
        out_dir / "math500.jsonl",
        (
            {
                "task_id": r.get("unique_id", f"math500-{i:05d}"),
                "question": r["problem"],
                "answer": r["answer"],
                "solution": r.get("solution", ""),
                "subject": r.get("subject", ""),
                "level": r.get("level", None),
                "source": "HuggingFaceH4/MATH-500",
                "split": "test",
            }
            for i, r in enumerate(math500)
        ),
        overwrite,
    )

    aime = load_dataset("MathArena/aime_2025", split="train", token=token)
    counts["aime2025"] = _write_jsonl(
        out_dir / "aime2025.jsonl",
        (
            {
                "task_id": f"aime2025-{int(r['problem_idx']):02d}",
                "question": r["problem"],
                "answer": str(r["answer"]),
                "problem_type": r.get("problem_type", []),
                "source": "MathArena/aime_2025",
                "split": "train",
            }
            for r in aime
        ),
        overwrite,
    )

    omni = load_dataset("KbsdJames/Omni-MATH", split="test", token=token)
    counts["omni_math"] = _write_jsonl(
        out_dir / "omni_math.jsonl",
        (
            {
                "task_id": f"omni-math-{i:05d}",
                "question": r["problem"],
                "answer": r["answer"],
                "solution": r.get("solution", ""),
                "domain": r.get("domain", []),
                "difficulty": r.get("difficulty", None),
                "source": r.get("source", "KbsdJames/Omni-MATH"),
                "split": "test",
            }
            for i, r in enumerate(omni)
        ),
        overwrite,
    )
    return counts


def _load_jsonish(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _prompt_from_bfcl_question(question: Any) -> str:
    messages: list[str] = []
    if isinstance(question, str):
        return question
    for convo in question if isinstance(question, list) else [question]:
        for msg in convo if isinstance(convo, list) else [convo]:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                messages.append(f"{role}: {content}")
    return "\n".join(messages)


def _tool_template(fn: dict[str, Any]) -> str:
    name = str(fn.get("name", ""))
    desc = str(fn.get("description", "")).strip()
    params = fn.get("parameters", {}) or {}
    props = params.get("properties", {}) if isinstance(params, dict) else {}
    required = set(params.get("required", [])) if isinstance(params, dict) else set()
    pieces = [f"{name} - {desc}" if desc else name]
    if isinstance(props, dict) and props:
        arg_bits = []
        for arg, spec in props.items():
            typ = spec.get("type", "any") if isinstance(spec, dict) else "any"
            mark = "" if arg in required else "?"
            arg_bits.append(f"{arg}{mark}: {typ}")
        pieces.append("(" + ", ".join(arg_bits) + ")")
    return " ".join(pieces)


def _bfcl_gold_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for item in row.get("ground_truth", []) or []:
        if not isinstance(item, dict):
            continue
        for name, args in item.items():
            calls.append({"name": str(name), "arguments": args})
    return calls


def prepare_bfcl(out_dir: Path, overwrite: bool, token: str | None) -> dict[str, int]:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    raw_dir = out_dir / "raw" / "bfcl"
    snapshot_download(repo, repo_type="dataset", local_dir=raw_dir, token=token)

    api = HfApi(token=token)
    files = api.list_repo_files(repo, repo_type="dataset")
    answer_files = sorted(
        f for f in files
        if f.startswith("possible_answer/BFCL_v3_") and f.endswith(".json")
    )

    rows = []
    for answer_file in answer_files:
        base = answer_file.split("/", 1)[1]
        if base not in files:
            continue
        task_path = Path(hf_hub_download(repo, base, repo_type="dataset", token=token))
        answer_path = Path(hf_hub_download(repo, answer_file, repo_type="dataset", token=token))
        tasks = {str(r.get("id")): r for r in _load_jsonish(task_path)}
        answers = {str(r.get("id")): r for r in _load_jsonish(answer_path)}
        category = base.removesuffix(".json")
        for task_id, task in tasks.items():
            answer = answers.get(task_id)
            if not answer:
                continue
            functions = task.get("function", []) or []
            tools = [str(fn.get("name", "")) for fn in functions if fn.get("name")]
            if not tools:
                continue
            gold_calls = _bfcl_gold_calls(answer)
            gold = [call["name"] for call in gold_calls]
            if not gold:
                continue
            rows.append({
                "task_id": task_id,
                "prompt": (
                    _prompt_from_bfcl_question(task.get("question", ""))
                    + "\n\nAvailable tools:\n"
                    + "\n".join(f"- {_tool_template(fn)}" for fn in functions)
                    + "\n\nSelect the next tool call name."
                ),
                "tools": tools,
                "gold": gold,
                "gold_calls": gold_calls,
                "horizon": len(gold),
                "action_templates": {
                    str(fn.get("name", "")): f"call {_tool_template(fn)}"
                    for fn in functions if fn.get("name")
                },
                "difficulty": 0.5,
                "source": repo,
                "category": category,
            })

    count = _write_jsonl(out_dir / "bfcl_full.jsonl", rows, overwrite)
    return {"bfcl_full": count}


def snapshot_agentic_raw(out_dir: Path, token: str | None) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    raw = out_dir / "raw"
    paths = {}
    for name, repo in {
        "tau2_bench_data": "HuggingFaceH4/tau2-bench-data",
        "tau2_bench_hud": "Genteki/tau2-bench",
    }.items():
        dest = raw / name
        snapshot_download(repo, repo_type="dataset", local_dir=dest, token=token)
        paths[name] = str(dest)
        print(f"snapshot -> {dest}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare ACDAN benchmark datasets.")
    parser.add_argument("--out-dir", default=str(ROOT / "data"))
    parser.add_argument("--suite", choices=["math", "bfcl", "agentic_raw", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    args = parser.parse_args()

    load_env(args.env_file)
    token = os.environ.get("HF_TOKEN")
    out_dir = Path(args.out_dir)
    manifest: dict[str, Any] = {"out_dir": str(out_dir), "suite": args.suite}

    if args.suite in {"math", "all"}:
        manifest["math"] = prepare_math(out_dir, args.overwrite, token)
    if args.suite in {"bfcl", "all"}:
        manifest["bfcl"] = prepare_bfcl(out_dir, args.overwrite, token)
    if args.suite in {"agentic_raw", "all"}:
        manifest["agentic_raw"] = snapshot_agentic_raw(out_dir, token)

    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
