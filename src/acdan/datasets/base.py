"""Raw tasks, JSONL loading, and outcome checkers.

A ``RawTask`` is backend-agnostic (text + candidate actions + gold). The
experiment harness encodes it into a frozen :class:`acdan.types.Task` (filling
``prompt_features`` via the chosen encoder and stashing ``prompt``/``gold`` in
``metadata``), then attaches one ``OutcomeChecker``.

Two task shapes cover all three benchmark families with the same (H, V) machinery:
  * **answer-selection** (H=1): ``vocab`` = K candidate answers for this task,
    ``gold`` = the correct answer string. Use for GSM8K / MATH (candidates are
    pre-generated samples) and as multiple-choice. Checker: ``outcome_exact``.
  * **tool/operator sequence** (H>1): ``vocab`` = the tool/operator set,
    ``gold`` = the reference action-name sequence. Use for BFCL / ToolBench.
    Checker: ``outcome_tool_sequence``.

Open-ended tasks (GAIA) use the Claude judge checker (see ``gaia.py``).

Expected JSONL line schema (one task per line)::

    {"prompt": "...", "vocab": ["a","b",...], "horizon": 1,
     "gold": "b"  | ["tool1","tool2"], "difficulty": 0.5}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from acdan.datasets.math_answer import answers_equivalent
from acdan.types import Task


MATH_DATASETS = frozenset({"gsm8k", "math", "math500", "aime2025", "omni_math"})
ROADMAP_ANSWER_DATASETS = frozenset({"browsecomp_proxy", "mathhay_proxy"})
ANSWER_SELECTION_DATASETS = frozenset({"synthetic", "jsonl"}) | MATH_DATASETS | ROADMAP_ANSWER_DATASETS


@dataclass
class RawTask:
    task_id: str
    prompt: str
    vocab: Tuple[str, ...]
    horizon: int
    gold: Any                       # str (answer) or list[str] (tool sequence)
    family: str = "generic"
    difficulty: float = 0.5
    action_templates: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Dataset adapters (DatasetAdapter Protocol: .tasks() -> Iterable[RawTask])
# --------------------------------------------------------------------------

class JSONLRawDataset:
    """Generic JSONL adapter (schema in module docstring)."""

    def __init__(self, path: str, family: str = "generic", limit: Optional[int] = None):
        self.path = path
        self.family = family
        self.limit = limit

    def tasks(self) -> Iterable[RawTask]:
        n = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                yield RawTask(
                    task_id=d.get("task_id", f"{self.family}-{i:05d}"),
                    prompt=d["prompt"],
                    vocab=tuple(d["vocab"]),
                    horizon=int(d.get("horizon", 1)),
                    gold=d["gold"],
                    family=self.family,
                    difficulty=float(d.get("difficulty", 0.5)),
                    action_templates=d.get("action_templates", {}) or {},
                    metadata={k: v for k, v in d.items()
                              if k not in {"prompt", "vocab", "horizon", "gold"}},
                )
                n += 1
                if self.limit and n >= self.limit:
                    return


class SyntheticRawDataset:
    """Offline fallback: answer-selection tasks with a known gold option."""

    def __init__(self, n: int = 24, k: int = 4, seed: int = 0):
        self.n, self.k, self.seed = n, k, seed

    def tasks(self) -> Iterable[RawTask]:
        import numpy as np
        rng = np.random.default_rng(self.seed)
        for i in range(self.n):
            opts = [f"option_{j}" for j in range(self.k)]
            gold_idx = int(rng.integers(0, self.k))
            yield RawTask(
                task_id=f"syn-{i:04d}",
                prompt=f"Synthetic question {i}: choose the correct option.",
                vocab=tuple(opts),
                horizon=1,
                gold=opts[gold_idx],
                family="synthetic",
                difficulty=float(rng.uniform(0.3, 0.8)),
                metadata={"gold_idx": gold_idx},
            )


def build_dataset(kind: str, path: Optional[str] = None, limit: Optional[int] = None,
                  **kwargs):
    """Factory for dataset adapters by name."""
    if kind == "synthetic":
        return SyntheticRawDataset(n=limit or 24, **kwargs)
    if kind == "jsonl":
        if not path:
            raise ValueError("jsonl dataset requires --data-path")
        return JSONLRawDataset(path, family="jsonl", limit=limit)
    if kind in MATH_DATASETS:
        from acdan.datasets.gsm8k import GSM8KDataset
        return GSM8KDataset(path, family=kind, limit=limit, **kwargs)
    if kind in ROADMAP_ANSWER_DATASETS:
        from acdan.datasets.gsm8k import GSM8KDataset
        return GSM8KDataset(path, family=kind, limit=limit, **kwargs)
    if kind in {"bfcl", "toolbench"}:
        from acdan.datasets.bfcl import BFCLDataset
        return BFCLDataset(path, limit=limit)
    if kind == "gaia":
        from acdan.datasets.gaia import GAIADataset
        return GAIADataset(path, limit=limit)
    raise KeyError(f"unknown dataset '{kind}'")


# --------------------------------------------------------------------------
# Outcome checkers: checker(task, executed_action_ids) -> bool
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def outcome_exact(task: Task, actions: Sequence[int]) -> bool:
    """Answer-selection: the chosen option (step 0) equals the gold answer.

    Compares normalised extracted answers and numeric values only when the
    extracted answers themselves are numeric.
    """
    if not actions:
        return False
    pred = task.vocab[int(actions[0])]
    gold = str(task.metadata.get("gold", ""))
    if _norm(pred) == _norm(gold):
        return True
    return answers_equivalent(pred, gold)


def outcome_tool_sequence(task: Task, actions: Sequence[int]) -> bool:
    """Tool/operator: executed action names exactly match the gold sequence."""
    gold = task.metadata.get("gold", [])
    if isinstance(gold, str):
        gold = [gold]
    pred = [task.vocab[int(a)] for a in actions]
    return (
        len(pred) == len(gold)
        and len(pred) > 0
        and all(_norm(pred[i]) == _norm(gold[i]) for i in range(len(gold)))
    )


def build_outcome_checker(kind: str):
    """Return the default checker for a dataset family (Claude judge for gaia)."""
    if kind in ANSWER_SELECTION_DATASETS:
        return outcome_exact
    if kind in {"bfcl", "toolbench"}:
        return outcome_tool_sequence
    if kind == "gaia":
        from acdan.backends.claude import ClaudeJudge
        judge = ClaudeJudge()

        def _judge(task: Task, actions: Sequence[int]) -> bool:
            pred = " ".join(task.vocab[int(a)] for a in actions)
            return judge.is_correct(task, pred)

        return _judge
    raise KeyError(f"no outcome checker for '{kind}'")
