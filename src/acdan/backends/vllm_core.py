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

import os
import tempfile
from typing import List, Optional

import numpy as np

from acdan.datasets.base import ANSWER_SELECTION_DATASETS
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
        extract_hidden_states: bool = False,
        hidden_state_layer_ids: Optional[List[int]] = None,
        hidden_state_storage_path: Optional[str] = None,
    ):
        from vllm import LLM  # lazy

        self.model_name = model_name
        self.extract_hidden_states = bool(extract_hidden_states)
        self.hidden_state_layer_ids = list(hidden_state_layer_ids or [])
        self.hidden_state_storage_path = hidden_state_storage_path
        llm_kwargs = {
            "model": model_name,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "seed": seed,
            "enable_prefix_caching": True,
        }
        if self.extract_hidden_states:
            from vllm.config.kv_transfer import KVTransferConfig  # lazy

            storage = hidden_state_storage_path or self._default_hidden_state_storage()
            os.makedirs(storage, exist_ok=True)
            self.hidden_state_storage_path = storage
            layer_ids = self.hidden_state_layer_ids or self._default_hidden_layer_ids(model_name)
            self.hidden_state_layer_ids = layer_ids
            llm_kwargs.update({
                "enable_chunked_prefill": False,
                "speculative_config": {
                    "method": "extract_hidden_states",
                    "num_speculative_tokens": 1,
                    "draft_model_config": {
                        "hf_config": {
                            "eagle_aux_hidden_state_layer_ids": layer_ids,
                        },
                    },
                },
                "kv_transfer_config": KVTransferConfig(
                    kv_connector="ExampleHiddenStatesConnector",
                    kv_role="kv_producer",
                    kv_connector_extra_config={
                        "shared_storage_path": storage,
                    },
                ),
            })
        self.llm = LLM(**llm_kwargs)
        self.tok = self.llm.get_tokenizer()
        self.hidden_size = self._hidden_size(model_name)
        # Telemetry for real cost reporting.
        self.n_score_batches = 0
        self.n_prompt_tokens = 0
        self._hidden_cache: dict[str, np.ndarray] = {}
        self._conditional_cache: dict[tuple[str, tuple[int, ...], bool], np.ndarray] = {}

    @staticmethod
    def _default_hidden_state_storage() -> str:
        base = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        return os.path.join(base, "acdan_vllm_hidden_states")

    @staticmethod
    def _default_hidden_layer_ids(model_name: str) -> List[int]:
        try:
            from transformers import AutoConfig  # lazy

            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            n_layers = getattr(cfg, "num_hidden_layers", None)
            if n_layers is not None:
                return [int(n_layers)]
        except Exception:
            pass
        # Qwen2.5-7B/Llama-3.1-8B both have 28/32 layers; final-ish fallback.
        return [32]

    @staticmethod
    def _hidden_size(model_name: str) -> int:
        try:
            from transformers import AutoConfig  # lazy

            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            hidden = getattr(cfg, "hidden_size", None)
            if hidden is not None:
                return int(hidden)
            # Check nested config (e.g., Gemma-4's text_config)
            if hasattr(cfg, "text_config"):
                hidden = getattr(cfg.text_config, "hidden_size", None)
                if hidden is not None:
                    return int(hidden)
        except Exception:
            pass
        return 4096

    # ------------------------------------------------------------- hidden states

    def prompt_hidden_features(
        self,
        prompt: str,
        pooling: str = "last",
        normalize: bool = True,
    ) -> np.ndarray:
        """Extract pooled prompt hidden states from the same vLLM engine.

        Requires constructing this backend with ``extract_hidden_states=True``.
        The returned vector is pooled over prompt tokens and the last extracted
        layer. Files produced by the vLLM connector are removed after loading.
        """
        if not self.extract_hidden_states:
            raise RuntimeError(
                "vLLM hidden-state encoder requested, but the core was not "
                "created with extract_hidden_states=True."
            )
        key = f"{pooling}:{normalize}:{prompt or ''}"
        cached = self._hidden_cache.get(key)
        if cached is not None:
            return cached.copy()

        from vllm import SamplingParams  # lazy
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            example_hidden_states_connector,
        )

        # The connector saves prompt-token hidden states by default. We generate
        # one token because vLLM generate requires a decode step, but the saved
        # tensor excludes output tokens unless include_output_tokens=True.
        outs = self.llm.generate([prompt or ""], SamplingParams(max_tokens=1))
        if not outs or not getattr(outs[0], "kv_transfer_params", None):
            raise RuntimeError("vLLM did not return hidden_states_path.")
        path = outs[0].kv_transfer_params["hidden_states_path"]
        obj = example_hidden_states_connector.load_hidden_states(path)
        try:
            os.remove(path)
        except OSError:
            pass
        hidden = obj["hidden_states"]
        if hasattr(hidden, "detach"):
            hidden = hidden.detach().float().cpu().numpy()
        hidden = np.asarray(hidden, dtype=np.float64)
        if hidden.ndim != 3:
            raise RuntimeError(f"unexpected hidden_states shape: {hidden.shape}")
        layer_states = hidden[:, -1, :]
        if pooling == "mean":
            vec = layer_states.mean(axis=0)
        elif pooling == "last":
            vec = layer_states[-1]
        else:
            raise ValueError("pooling must be 'last' or 'mean'")
        if normalize:
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
        self._hidden_cache[key] = vec.copy()
        return vec

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

    def _tool_prefix_context(self, task: Task, prefix: tuple[int, ...]) -> str:
        prompt = str(task.metadata.get("prompt", ""))
        if prefix:
            selected = "\n".join(
                f"{i + 1}. {task.vocab[action]}" for i, action in enumerate(prefix)
            )
        else:
            selected = "(none)"
        return (
            f"{prompt}\n\nSelected tool calls so far:\n{selected}\n\n"
            "Choose the next tool call. If the request is fully covered, choose <STOP>.\n"
            "Next call: "
        )

    def conditional_prior_logits(
        self,
        task: Task,
        latent: np.ndarray,
        prefix: tuple[int, ...],
        include_stop: bool = False,
    ) -> np.ndarray:
        """Score all next tools under the actual selected prefix."""
        return self.conditional_prior_logits_batch(
            task, latent, [tuple(prefix)], include_stop=include_stop
        )[0]

    def conditional_prior_logits_batch(
        self,
        task: Task,
        latent: np.ndarray,
        prefixes,
        include_stop: bool = False,
    ) -> np.ndarray:
        """Batch all active prefixes at one lattice depth in one vLLM call."""
        prefixes = [tuple(int(action) for action in prefix) for prefix in prefixes]
        missing = [
            prefix for prefix in prefixes
            if (task.task_id, prefix, bool(include_stop)) not in self._conditional_cache
        ]
        templates = task.metadata.get("action_templates", {}) or {}
        if missing:
            contexts: List[str] = []
            continuations: List[str] = []
            action_text = [str(templates.get(name, name)) for name in task.vocab]
            if include_stop:
                action_text.append("<STOP>")
            for prefix in missing:
                context = self._tool_prefix_context(task, prefix)
                contexts.extend([context] * len(action_text))
                continuations.extend(action_text)
            scores = self._score_continuations(contexts, continuations)
            width = len(action_text)
            matrix = np.asarray(scores, dtype=np.float64).reshape(len(missing), width)
            self.n_score_batches += 1
            for prefix, row in zip(missing, matrix):
                self._conditional_cache[
                    (task.task_id, prefix, bool(include_stop))
                ] = row.copy()
        return np.stack([
            self._conditional_cache[(task.task_id, prefix, bool(include_stop))]
            for prefix in prefixes
        ])

    def prior_logits(self, task: Task, latent: np.ndarray) -> np.ndarray:
        H, V = task.horizon, task.vocab_size
        prompt = str(task.metadata.get("prompt", ""))
        templates = task.metadata.get("action_templates", {}) or {}
        family = str(task.metadata.get("family", ""))
        contexts: List[str] = []
        conts: List[str] = []
        for h in range(H):
            if family in ANSWER_SELECTION_DATASETS:
                ctx = (
                    "Evaluate whether the proposed answer is correct for the task.\n\n"
                    f"Task:\n{prompt}\n\n"
                    "Plausible correct candidate:\n"
                )
            else:
                ctx = f"{prompt}\nStep {h + 1}: "
            for v in range(V):
                name = task.vocab[v]
                contexts.append(ctx)
                if family in ANSWER_SELECTION_DATASETS:
                    conts.append(str(templates.get(name, f"Final answer: {name}")))
                else:
                    conts.append(str(templates.get(name, name)))
        scores = self._score_continuations(contexts, conts)
        self.n_score_batches += 1
        return np.asarray(scores, dtype=np.float64).reshape(H, V)
