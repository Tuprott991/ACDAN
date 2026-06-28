"""GSM8K / MATH as answer-selection over pre-generated candidates.

In the tool/operator-selection DTO framing, math is run as a (H=1, V=K) choice:
sample K candidate solutions with the policy first (offline preprocessing — see
``experiments/PLAN.md`` Step 1), extract each candidate's final answer, then let
DTO + PRM pick the best. Gold is the reference answer; checker is ``outcome_exact``.

Expected JSONL line::

    {"question": "...", "candidates": ["ans1","ans2",...], "answer": "42"}

(`candidates` are the extracted final answers of the K sampled solutions.)
Newer candidate files may also include ``candidate_solutions``,
``candidate_counts``, and ``candidate_first_indices``; the adapter forwards
those as metadata/action templates for answer verification.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from acdan.datasets.base import RawTask


class GSM8KDataset:
    def __init__(
        self,
        path: Optional[str],
        family: str = "gsm8k",
        limit: Optional[int] = None,
        include_candidate_counts: bool = False,
        include_candidate_reasoning: bool = True,
        use_prm_count_bonus: bool = False,
    ):
        if not path:
            raise ValueError(
                "Math answer-selection adapter needs --data-path to a candidates JSONL "
                "(see experiments/PLAN.md Step 1 to generate it)."
            )
        self.path = path
        self.family = family
        self.limit = limit
        self.include_candidate_counts = include_candidate_counts
        self.include_candidate_reasoning = include_candidate_reasoning
        self.use_prm_count_bonus = use_prm_count_bonus

    def _candidate_answer(self, raw: Any) -> str:
        if isinstance(raw, dict):
            return str(raw.get("answer", raw.get("final_answer", raw.get("text", ""))))
        return str(raw)

    def _candidate_metadata(self, d: dict, candidates: list[str]) -> tuple[dict, dict, dict]:
        solutions = {str(k): str(v) for k, v in (d.get("candidate_solutions") or {}).items()}
        counts = {str(k): int(v) for k, v in (d.get("candidate_counts") or {}).items()}
        first_indices = {
            str(k): int(v) for k, v in (d.get("candidate_first_indices") or {}).items()
        }

        for i, raw in enumerate(d.get("candidates", [])):
            if not isinstance(raw, dict):
                continue
            answer = self._candidate_answer(raw)
            if "solution" in raw:
                solutions.setdefault(answer, str(raw["solution"]))
            if "count" in raw:
                counts.setdefault(answer, int(raw["count"]))
            if "first_index" in raw:
                first_indices.setdefault(answer, int(raw["first_index"]))

        for i, answer in enumerate(candidates):
            counts.setdefault(answer, 1)
            first_indices.setdefault(answer, i)
        return solutions, counts, first_indices

    def _action_templates(self, candidates: list[str], solutions: dict, counts: dict) -> dict:
        templates = {}
        for answer in candidates:
            parts = [f"Final answer: {answer}"]
            if self.include_candidate_counts and answer in counts:
                parts.append(f"Self-consistency count: {counts[answer]}")
            solution = str(solutions.get(answer, "")).strip()
            if self.include_candidate_reasoning and solution:
                parts.append(f"Candidate reasoning:\n{solution}")
            templates[answer] = "\n".join(parts)
        return templates

    def tasks(self) -> Iterable[RawTask]:
        n = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                cands = [self._candidate_answer(c) for c in d["candidates"]]
                if not cands:
                    continue
                solutions, counts, first_indices = self._candidate_metadata(d, cands)
                yield RawTask(
                    task_id=d.get("task_id", f"{self.family}-{i:05d}"),
                    prompt=d["question"],
                    vocab=tuple(cands),
                    horizon=1,
                    gold=str(d["answer"]),
                    family=self.family,
                    difficulty=float(d.get("difficulty", 0.5)),
                    action_templates=self._action_templates(cands, solutions, counts),
                    metadata={
                        "n_candidates": len(cands),
                        "candidate_solutions": solutions,
                        "candidate_counts": counts,
                        "candidate_first_indices": first_indices,
                        "candidate_sample_answers": [
                            str(x) for x in d.get("candidate_sample_answers", [])
                        ],
                        "candidate_order": d.get("candidate_order", "unknown"),
                        "include_candidate_counts": self.include_candidate_counts,
                        "include_candidate_reasoning": self.include_candidate_reasoning,
                        "use_prm_count_bonus": self.use_prm_count_bonus,
                    },
                )
                n += 1
                if self.limit and n >= self.limit:
                    return
