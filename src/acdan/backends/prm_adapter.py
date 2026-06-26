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

    def step_reward_matrix(self, task: Task, latent: np.ndarray) -> np.ndarray:
        H, V = task.horizon, task.vocab_size
        prompt = str(task.metadata.get("prompt", ""))
        prompts: List[str] = []
        for h in range(H):
            for v in range(V):
                prompts.append(
                    f"{prompt}\nProposed step {h + 1}: {task.vocab[v]}\n"
                    f"Is this step correct and useful? Answer yes or no.\nAnswer:"
                )
        p = np.asarray(self._p_yes(prompts), dtype=np.float64).reshape(H, V)
        # map [0,1] P(yes) to a centered reward field in [-1, 1]
        return 2.0 * p - 1.0

    def _reward_target(self, task: Task, latent: np.ndarray) -> np.ndarray:
        R = self.step_reward_matrix(task, latent)
        z = self.sharpness * R
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # -------------------------------------------------- ProcessRewardModel

    def score_probs(self, task: Task, latent: np.ndarray, probs: np.ndarray) -> float:
        return float(np.sum(probs * self._reward_target(task, latent)) / probs.shape[0])

    def grad_wrt_probs(self, task: Task, latent: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return self._reward_target(task, latent) / probs.shape[0]

    def score_actions(self, task: Task, latent: np.ndarray, actions: Sequence[int]) -> List[float]:
        R = self.step_reward_matrix(task, latent)
        sig = 1.0 / (1.0 + np.exp(-R))
        return [float(sig[min(h, R.shape[0] - 1), int(a)]) for h, a in enumerate(actions)]
