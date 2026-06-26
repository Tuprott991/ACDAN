"""LLM-as-PRM: a process reward model backed by an open-weight LLM.

For each step ``h`` and candidate action ``v`` we ask the model whether the step
is correct/useful and read P(yes) from the first generated token's logprobs. The
resulting (H, V) reward field is consumed by DTO exactly like the mock PRM: the
per-step softmax-sharpened target is the gradient (``grad_wrt_probs``), so no
autograd is required.

Reuse the policy's vLLM handle (`core.llm`) to avoid loading a second model, or
pass ``model_name`` to load a dedicated PRM (e.g. a Math-PRM). Lazy imports; GPU
only.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence

import numpy as np

from acdan.types import Task


class LLMAsProcessReward:
    """Process reward via P(yes) on a verification prompt."""

    def __init__(
        self,
        core=None,                     # reuse a VLLMCoreModel's llm/tokenizer
        model_name: Optional[str] = None,
        sharpness: float = 4.0,
        yes_tokens: Sequence[str] = (" yes", "yes", "Yes", " Yes"),
        no_tokens: Sequence[str] = (" no", "no", "No", " No"),
    ):
        if core is not None:
            self.llm, self.tok = core.llm, core.tok
        elif model_name is not None:
            from vllm import LLM  # lazy
            self.llm = LLM(model=model_name)
            self.tok = self.llm.get_tokenizer()
        else:
            raise ValueError("provide either `core` or `model_name`")
        self.sharpness = sharpness
        self._yes = self._ids(yes_tokens)
        self._no = self._ids(no_tokens)
        self._reward_cache: dict[tuple[str, str, int, int], np.ndarray] = {}
        self._target_cache: dict[tuple[str, str, int, int], np.ndarray] = {}
        self.n_prompt_tokens = 0

    def _ids(self, toks: Sequence[str]) -> List[int]:
        out = []
        for t in toks:
            ids = self.tok(t, add_special_tokens=False).input_ids
            if ids:
                out.append(ids[-1])
        return out

    def _p_yes(self, prompts: List[str]) -> List[float]:
        from vllm import SamplingParams  # lazy

        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
        outs = self.llm.generate(prompts, sp)
        ps: List[float] = []
        for p, out in zip(prompts, outs):
            self.n_prompt_tokens += len(self.tok(p).input_ids)
            lp = out.outputs[0].logprobs[0]  # {token_id: Logprob}
            ly = max((lp[t].logprob for t in self._yes if t in lp), default=-1e9)
            ln = max((lp[t].logprob for t in self._no if t in lp), default=-1e9)
            m = max(ly, ln)
            ey, en = np.exp(ly - m), np.exp(ln - m)
            ps.append(float(ey / (ey + en + 1e-9)))
        return ps

    # ------------------------------------------------------ reward field

    def _cache_key(self, task: Task, latent: np.ndarray) -> tuple[str, str, int, int]:
        """Stable key for LLM reward calls that do not depend on soft probs."""
        arr = np.ascontiguousarray(latent, dtype=np.float64)
        latent_digest = hashlib.blake2b(arr.view(np.uint8), digest_size=12).hexdigest()
        templates = task.metadata.get("action_templates", {}) or {}
        task_payload = repr((task.task_id, task.vocab, templates)).encode("utf-8")
        task_digest = hashlib.blake2b(task_payload, digest_size=12).hexdigest()
        return (task_digest, latent_digest, task.horizon, task.vocab_size)

    def _reward_prompt(self, task: Task, h: int, v: int) -> str:
        prompt = str(task.metadata.get("prompt", ""))
        family = str(task.metadata.get("family", ""))
        action = str(task.vocab[v])
        if family in {"gsm8k", "math"}:
            solutions = task.metadata.get("candidate_solutions", {}) or {}
            counts = task.metadata.get("candidate_counts", {}) or {}
            solution = str(solutions.get(action, "")).strip()
            count = counts.get(action)
            count_line = f"\nCandidate sample count: {count}" if count is not None else ""
            solution_block = (
                f"\nCandidate reasoning:\n{solution}\n" if solution else "\n"
            )
            return (
                "You are verifying a math answer. Recompute the problem and "
                "judge only whether the proposed final answer is correct.\n\n"
                f"Problem:\n{prompt}\n\n"
                f"Proposed final answer: {action}"
                f"{count_line}"
                f"{solution_block}"
                "Is the proposed final answer correct? Answer yes or no.\nAnswer:"
            )
        return (
            f"{prompt}\nProposed step {h + 1}: {action}\n"
            f"Is this step correct and useful? Answer yes or no.\nAnswer:"
        )

    def _self_consistency_bonus(self, task: Task) -> np.ndarray:
        family = str(task.metadata.get("family", ""))
        if family not in {"gsm8k", "math"}:
            return np.zeros((task.horizon, task.vocab_size), dtype=np.float64)

        counts = task.metadata.get("candidate_counts", {}) or {}
        vals = np.asarray(
            [float(counts.get(str(name), 1.0)) for name in task.vocab],
            dtype=np.float64,
        )
        if vals.size == 0 or float(vals.max()) <= float(vals.min()):
            return np.zeros((task.horizon, task.vocab_size), dtype=np.float64)
        z = np.log1p(vals)
        z = (z - float(z.mean())) / (float(z.std()) + 1e-9)
        z = np.clip(z, -2.0, 2.0)
        return 0.35 * np.tile(z, (task.horizon, 1))

    def step_reward_matrix(self, task: Task, latent: np.ndarray) -> np.ndarray:
        key = self._cache_key(task, latent)
        cached = self._reward_cache.get(key)
        if cached is not None:
            return cached.copy()

        H, V = task.horizon, task.vocab_size
        prompts: List[str] = []
        for h in range(H):
            for v in range(V):
                prompts.append(self._reward_prompt(task, h, v))
        p = np.asarray(self._p_yes(prompts), dtype=np.float64).reshape(H, V)
        # map [0,1] P(yes) to a centered reward field and add cheap evidence.
        R = 2.0 * p - 1.0 + self._self_consistency_bonus(task)
        self._reward_cache[key] = R.copy()
        return R

    def _reward_target(self, task: Task, latent: np.ndarray) -> np.ndarray:
        key = self._cache_key(task, latent)
        cached = self._target_cache.get(key)
        if cached is not None:
            return cached.copy()

        R = self.step_reward_matrix(task, latent)
        z = self.sharpness * R
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        target = e / e.sum(axis=1, keepdims=True)
        self._target_cache[key] = target.copy()
        return target

    # -------------------------------------------------- ProcessRewardModel

    def score_probs(self, task: Task, latent: np.ndarray, probs: np.ndarray) -> float:
        return float(np.sum(probs * self._reward_target(task, latent)) / probs.shape[0])

    def grad_wrt_probs(self, task: Task, latent: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return self._reward_target(task, latent) / probs.shape[0]

    def score_actions(self, task: Task, latent: np.ndarray, actions: Sequence[int]) -> List[float]:
        R = self.step_reward_matrix(task, latent)
        sig = 1.0 / (1.0 + np.exp(-R))
        return [float(sig[min(h, R.shape[0] - 1), int(a)]) for h, a in enumerate(actions)]
