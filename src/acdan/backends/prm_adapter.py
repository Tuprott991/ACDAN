"""LLM-as-PRM: a process reward model backed by an open-weight LLM.

For each step ``h`` and candidate action ``v`` we ask the model whether the step
is correct/useful and read P(yes) from the first generated token's logprobs. The
resulting (H, V) reward field is consumed by DTO exactly like the mock PRM: the
per-step softmax target is sharpened according to latent quality and used as the
gradient (``grad_wrt_probs``), so no autograd is required.

Reuse the policy's vLLM handle (`core.llm`) to avoid loading a second model, or
pass ``model_name`` to load a dedicated PRM (e.g. a Math-PRM). Lazy imports; GPU
only.
"""

from __future__ import annotations

import hashlib
from typing import Callable, List, Optional, Sequence

import numpy as np

from acdan.datasets.base import MATH_DATASETS
from acdan.types import Task


class LLMAsProcessReward:
    """Process reward via P(yes) on a verification prompt."""

    def __init__(
        self,
        core=None,                     # reuse a VLLMCoreModel's llm/tokenizer
        model_name: Optional[str] = None,
        sharpness: float = 4.0,
        latent_quality_gain: float = 0.5,
        latent_quality_fn: Optional[Callable[[np.ndarray], float]] = None,
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
        self.latent_quality_gain = latent_quality_gain
        self._latent_quality_fn = latent_quality_fn
        self._yes = self._ids(yes_tokens)
        self._no = self._ids(no_tokens)
        self._reward_cache: dict[tuple[str, str, int, int], np.ndarray] = {}
        self._target_cache: dict[tuple[str, str, int, int], np.ndarray] = {}
        self._extension_cache: dict[tuple[str, str, tuple[int, ...]], np.ndarray] = {}
        self._trajectory_cache: dict[tuple[str, str, tuple[int, ...]], float] = {}
        self.n_prompt_tokens = 0

    def set_latent_quality_fn(self, fn: Callable[[np.ndarray], float]) -> None:
        self._latent_quality_fn = fn
        self._target_cache.clear()

    def latent_quality(self, latent: np.ndarray) -> float:
        fn = getattr(self, "_latent_quality_fn", None)
        if fn is not None:
            q = float(fn(latent))
        else:
            h = np.asarray(latent, dtype=np.float64).reshape(-1)
            q = float(np.tanh(np.linalg.norm(h) / (np.sqrt(h.size) + 1e-9)))
        return float(np.clip(q, 0.0, 1.0))

    def effective_sharpness(self, latent: np.ndarray) -> float:
        gain = float(getattr(self, "latent_quality_gain", 0.5))
        q = self.latent_quality(latent)
        scale = 1.0 + gain * (2.0 * q - 1.0)
        return float(self.sharpness * max(scale, 0.1))

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
        if family in MATH_DATASETS:
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
        if family not in MATH_DATASETS:
            return np.zeros((task.horizon, task.vocab_size), dtype=np.float64)
        if not bool(task.metadata.get("use_prm_count_bonus", False)):
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
        z = self.effective_sharpness(latent) * R
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

    # ------------------------------------------ autoregressive sequence rewards

    def _latent_digest(self, latent: np.ndarray) -> str:
        arr = np.ascontiguousarray(latent, dtype=np.float64)
        return hashlib.blake2b(arr.view(np.uint8), digest_size=12).hexdigest()

    @staticmethod
    def _format_prefix(task: Task, prefix: Sequence[int]) -> str:
        if not prefix:
            return "(none)"
        return "\n".join(
            f"{i + 1}. {task.vocab[int(action)]}"
            for i, action in enumerate(prefix)
        )

    def extension_rewards(
        self, task: Task, latent: np.ndarray, prefix: Sequence[int]
    ) -> np.ndarray:
        """P(useful) for every extension, conditioned on the selected prefix."""
        return self.extension_rewards_batch(task, latent, [prefix])[0]

    def extension_rewards_batch(
        self,
        task: Task,
        latent: np.ndarray,
        prefixes: Sequence[Sequence[int]],
    ) -> np.ndarray:
        """Score all active prefixes in one PRM generation batch."""
        prefix_keys = [tuple(int(action) for action in prefix) for prefix in prefixes]
        digest = self._latent_digest(latent)
        missing = [
            prefix for prefix in prefix_keys
            if (task.task_id, digest, prefix) not in self._extension_cache
        ]
        prompt = str(task.metadata.get("prompt", ""))
        if missing:
            prompts: List[str] = []
            for prefix in missing:
                selected = self._format_prefix(task, prefix)
                prompts.extend([
                    (
                        "Judge the next tool call in the context of the complete request and "
                        "the calls already selected. Reward coverage of an unresolved subtask; "
                        "reject wrong ordering and redundant duplicate work.\n\n"
                        f"Request and tools:\n{prompt}\n\n"
                        f"Selected calls:\n{selected}\n\n"
                        f"Proposed next call: {name}\n"
                        "Is this the correct next call? Answer yes or no.\nAnswer:"
                    )
                    for name in task.vocab
                ])
            matrix = np.asarray(self._p_yes(prompts), dtype=np.float64).reshape(
                len(missing), task.vocab_size
            )
            for prefix, row in zip(missing, matrix):
                self._extension_cache[(task.task_id, digest, prefix)] = row.copy()
        return np.stack([
            self._extension_cache[(task.task_id, digest, prefix)]
            for prefix in prefix_keys
        ])

    def trajectory_rewards(
        self,
        task: Task,
        latent: np.ndarray,
        trajectories: Sequence[Sequence[int]],
    ) -> List[float]:
        """P(complete and correct) for complete tool-name trajectories."""
        digest = self._latent_digest(latent)
        prompt = str(task.metadata.get("prompt", ""))
        scores: List[float | None] = []
        missing_prompts: List[str] = []
        missing_positions: List[int] = []
        keys: List[tuple[str, str, tuple[int, ...]]] = []
        for trajectory in trajectories:
            path = tuple(int(action) for action in trajectory)
            key = (task.task_id, digest, path)
            keys.append(key)
            cached = self._trajectory_cache.get(key)
            scores.append(cached)
            if cached is None:
                missing_positions.append(len(scores) - 1)
                missing_prompts.append(
                    "Judge this complete tool-call plan. It must cover every user request "
                    "exactly once, preserve required order, and contain no missing or "
                    "redundant calls.\n\n"
                    f"Request and tools:\n{prompt}\n\n"
                    f"Proposed complete plan:\n{self._format_prefix(task, path)}\n\n"
                    "Does this plan fully and correctly satisfy the request? "
                    "Answer yes or no.\nAnswer:"
                )
        if missing_prompts:
            values = self._p_yes(missing_prompts)
            for position, value in zip(missing_positions, values):
                score = float(value)
                scores[position] = score
                self._trajectory_cache[keys[position]] = score
        return [float(score) for score in scores]
