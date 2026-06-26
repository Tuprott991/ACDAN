"""vLLM-backed CoreModel: builds the (H, V) prior by *scoring* candidate actions.

For the tool/operator-selection DTO framing, the prior logit of action ``v`` at
step ``h`` is the model's length-normalised log-probability of that action's
template, conditioned on the task prompt and a step marker. All H*V candidate
scorings for a task are issued in a single batched ``llm.generate`` call.

Requires vLLM + a local GPU. Imported lazily; nothing here runs offline. The
exact vLLM logprob plumbing should be smoke-tested on the VM (see
``experiments/PLAN.md`` → Step 0).
"""

from __future__ import annotations

from typing import List

import numpy as np

from acdan.types import Task


class VLLMCoreModel:
    """Open-weight policy that emits prior action logits via candidate scoring."""

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.90,
        seed: int = 0,
    ):
        from vllm import LLM  # lazy

        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
        )
        self.tok = self.llm.get_tokenizer()
        # Telemetry for real cost reporting.
        self.n_score_batches = 0
        self.n_prompt_tokens = 0

    # ------------------------------------------------------------- scoring

    def _score_continuations(self, contexts: List[str], conts: List[str]) -> List[float]:
        """Mean-token logprob of each ``cont`` given its ``context``."""
        from vllm import SamplingParams  # lazy

        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
        fulls = [c + k for c, k in zip(contexts, conts)]
        outs = self.llm.generate(fulls, sp)
        scores: List[float] = []
        for ctx, cont, out in zip(contexts, conts, outs):
            n_ctx = len(self.tok(ctx).input_ids)
            ids = self.tok(ctx + cont).input_ids
            self.n_prompt_tokens += len(ids)
            pls = out.prompt_logprobs  # aligned to prompt tokens; pls[0] is None
            total, cnt = 0.0, 0
            for i in range(n_ctx, len(ids)):
                lp = pls[i] if i < len(pls) else None
                if not lp:
                    continue
                tid = ids[i]
                if tid in lp:
                    total += lp[tid].logprob
                    cnt += 1
            scores.append(total / max(cnt, 1))
        return scores

    # ----------------------------------------------------------- CoreModel

    def prior_logits(self, task: Task, latent: np.ndarray) -> np.ndarray:
        H, V = task.horizon, task.vocab_size
        prompt = str(task.metadata.get("prompt", ""))
        templates = task.metadata.get("action_templates", {}) or {}
        family = str(task.metadata.get("family", ""))
        contexts: List[str] = []
        conts: List[str] = []
        for h in range(H):
            if family in {"gsm8k", "math"}:
                ctx = (
                    "Solve the math problem and evaluate the proposed answer.\n\n"
                    f"Problem:\n{prompt}\n\n"
                    "Plausible correct candidate:\n"
                )
            else:
                ctx = f"{prompt}\nStep {h + 1}: "
            for v in range(V):
                name = task.vocab[v]
                contexts.append(ctx)
                if family in {"gsm8k", "math"}:
                    conts.append(str(templates.get(name, f"Final answer: {name}")))
                else:
                    conts.append(str(templates.get(name, name)))
        scores = self._score_continuations(contexts, conts)
        self.n_score_batches += 1
        return np.asarray(scores, dtype=np.float64).reshape(H, V)
