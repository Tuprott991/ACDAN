"""Claude API backends for the roles that do NOT need logits.

DTO cannot run on a closed API (no token logits, no gradients), so Claude is used
only as:
  * ``ClaudeIndependentVerifier`` — the proposal's *Independent Question Asking*:
    query an independent judgment of a plan without the agent's reasoning context.
  * ``ClaudeJudge`` — an LLM-as-judge outcome checker for open-ended tasks (GAIA).

Both lazily import the ``anthropic`` SDK and default to ``claude-opus-4-8``. Use
the Batches API for large offline judging (50% cheaper) — see PLAN.md.
"""

from __future__ import annotations

import json
from typing import List

import numpy as np

from acdan.types import Plan, Task

_DEFAULT_MODEL = "claude-opus-4-8"


class ClaudeIndependentVerifier:
    """Independent verifier via Claude structured output (agreement in [0,1])."""

    def __init__(self, model: str = _DEFAULT_MODEL):
        import anthropic  # lazy

        self._client = anthropic.Anthropic()
        self.model = model
        self.calls = 0

    def agreement(self, task: Task, plan: Plan) -> float:
        prompt = str(task.metadata.get("prompt", ""))
        actions = [task.vocab[a] for a in plan.actions]
        schema = {
            "type": "object",
            "properties": {"supported": {"type": "boolean"},
                           "confidence": {"type": "number"}},
            "required": ["supported", "confidence"],
            "additionalProperties": False,
        }
        msg = (
            "You are an independent verifier. Without assuming the agent's "
            "reasoning is correct, judge whether the proposed action sequence "
            "solves the task.\n\n"
            f"TASK:\n{prompt}\n\nPROPOSED ACTIONS:\n{actions}\n\n"
            "Return whether independent evidence supports this plan and your "
            "confidence in [0,1]."
        )
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": msg}],
        )
        self.calls += 1
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        try:
            data = json.loads(text)
        except Exception:
            return 0.5
        conf = float(data.get("confidence", 0.5))
        return conf if data.get("supported", False) else 1.0 - conf


class ClaudeJudge:
    """LLM-as-judge outcome checker for open-ended tasks (e.g. GAIA)."""

    def __init__(self, model: str = _DEFAULT_MODEL):
        import anthropic  # lazy

        self._client = anthropic.Anthropic()
        self.model = model
        self.calls = 0

    def is_correct(self, task: Task, predicted: str) -> bool:
        gold = str(task.metadata.get("gold", ""))
        schema = {
            "type": "object",
            "properties": {"correct": {"type": "boolean"}},
            "required": ["correct"], "additionalProperties": False,
        }
        msg = (
            "Judge whether the predicted answer matches the reference answer for "
            f"the task.\n\nTASK:\n{task.metadata.get('prompt','')}\n\n"
            f"REFERENCE:\n{gold}\n\nPREDICTED:\n{predicted}\n\n"
            "Return correct=true only if the prediction is essentially correct."
        )
        resp = self._client.messages.create(
            model=self.model, max_tokens=64,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": msg}],
        )
        self.calls += 1
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        try:
            return bool(json.loads(text).get("correct", False))
        except Exception:
            return False
