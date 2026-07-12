"""Candidate trajectory evaluators for AgentBench-style self-choice."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acdan.agentbench.adapters import AgentBenchTask
from acdan.datasets.math_answer import answers_equivalent


EXTERNAL_EVALUATORS = {
    "external_browsecomp",
    "external_webvoyager",
    "external_swe_bench",
    "external_terminal_bench",
    "external_mathhay",
    "external_tau2",
    "external_mcp_bench",
}


@dataclass
class Candidate:
    candidate_id: str
    final_answer: str = ""
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    patch: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: Any, index: int) -> "Candidate":
        if isinstance(obj, str):
            return cls(candidate_id=f"cand_{index}", final_answer=obj, raw={"text": obj})
        if not isinstance(obj, dict):
            text = str(obj)
            return cls(candidate_id=f"cand_{index}", final_answer=text, raw={"value": obj})
        return cls(
            candidate_id=str(obj.get("candidate_id", obj.get("id", f"cand_{index}"))),
            final_answer=str(obj.get("final_answer", obj.get("answer", obj.get("text", "")))),
            trajectory=list(obj.get("trajectory", []) or []),
            patch=str(obj.get("patch", "")),
            tool_calls=list(obj.get("tool_calls", []) or []),
            raw=dict(obj),
        )

    def display_text(self) -> str:
        pieces = []
        if self.final_answer:
            pieces.append(f"Final answer:\n{self.final_answer}")
        if self.patch:
            pieces.append(f"Patch:\n{self.patch}")
        if self.tool_calls:
            pieces.append("Tool calls:\n" + json.dumps(self.tool_calls, ensure_ascii=False))
        if self.trajectory:
            pieces.append("Trajectory:\n" + json.dumps(self.trajectory, ensure_ascii=False))
        if not pieces:
            pieces.append(json.dumps(self.raw, ensure_ascii=False))
        return "\n\n".join(pieces)


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


class CandidateEvaluator:
    """Evaluate selected candidates.

    For official environment benchmarks, pass ``external_commands`` mapping an
    evaluator name to a command template. The template may contain
    ``{input}`` and ``{output}`` placeholders. The command should write a JSON
    object containing ``score`` or ``correct`` to ``{output}``.
    """

    def __init__(
        self,
        external_commands: dict[str, str] | None = None,
        allow_unevaluated: bool = False,
    ):
        self.external_commands = dict(external_commands or {})
        self.allow_unevaluated = allow_unevaluated
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}

    def evaluate(self, task: AgentBenchTask, candidate: Candidate) -> dict[str, Any]:
        key = (task.task_id, candidate.candidate_id)
        if key in self._cache:
            return self._cache[key]
        result = self._evaluate_uncached(task, candidate)
        self._cache[key] = result
        return result

    def _evaluate_uncached(self, task: AgentBenchTask, candidate: Candidate) -> dict[str, Any]:
        raw = candidate.raw
        if "score" in raw:
            score = float(raw["score"])
            return {
                "score": score,
                "correct": bool(score > 0),
                "method": "candidate_score",
            }
        if "is_correct" in raw:
            correct = bool(raw["is_correct"])
            return {
                "score": 1.0 if correct else 0.0,
                "correct": correct,
                "method": "candidate_is_correct",
            }
        if task.evaluator == "semantic_qa":
            return self._semantic_qa(task, candidate)
        if task.evaluator in EXTERNAL_EVALUATORS:
            command = self.external_commands.get(task.evaluator)
            env_key = "ACDAN_" + task.evaluator.upper() + "_CMD"
            command = command or os.environ.get(env_key)
            if command:
                return self._external(task, candidate, command)
            if self.allow_unevaluated:
                return {"score": 0.0, "correct": False, "method": "unevaluated"}
            raise RuntimeError(
                f"{task.dataset} requires official evaluator '{task.evaluator}'. "
                f"Pass --evaluator-command {task.evaluator}=<cmd> or set {env_key}. "
                "Alternatively provide candidate fields score/is_correct from an official run."
            )
        raise KeyError(f"unknown evaluator '{task.evaluator}'")

    def _semantic_qa(self, task: AgentBenchTask, candidate: Candidate) -> dict[str, Any]:
        gold = "" if task.gold is None else str(task.gold)
        pred = candidate.final_answer
        correct = bool(gold) and (_norm(pred) == _norm(gold) or answers_equivalent(pred, gold))
        return {"score": 1.0 if correct else 0.0, "correct": correct, "method": "semantic_qa_exact"}

    def _external(self, task: AgentBenchTask, candidate: Candidate, command: str) -> dict[str, Any]:
        payload = {
            "task": task.to_json(),
            "candidate": {
                "candidate_id": candidate.candidate_id,
                "final_answer": candidate.final_answer,
                "patch": candidate.patch,
                "tool_calls": candidate.tool_calls,
                "trajectory": candidate.trajectory,
                "raw": candidate.raw,
            },
        }
        with tempfile.TemporaryDirectory(prefix="acdan_agentbench_eval_") as tmp:
            inp = Path(tmp) / "input.json"
            out = Path(tmp) / "output.json"
            inp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            formatted = command.format(input=str(inp), output=str(out))
            proc = subprocess.run(
                formatted,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"external evaluator failed ({proc.returncode}): {formatted}\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            if out.exists():
                value = json.loads(out.read_text(encoding="utf-8"))
            else:
                stdout = proc.stdout.strip()
                value = json.loads(stdout) if stdout else {}
            score = float(value.get("score", 1.0 if value.get("correct") else 0.0))
            return {
                "score": score,
                "correct": bool(value.get("correct", score > 0)),
                "method": task.evaluator,
                "raw": value,
            }
