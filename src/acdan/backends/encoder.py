"""Latent feature encoders: turn task prompt text into a feature vector.

The default ``HashingEncoder`` has **no dependencies** (deterministic hashed
bag-of-tokens), so the real pipeline runs offline. On the VM you can switch to a
sentence-transformer for genuine semantic features without changing anything
else — the ``LatentReasoner`` is constructed with ``feature_dim = encoder.dim``.

For stronger real-model experiments, ``HFLLMHiddenStateEncoder`` extracts prompt
features from the same causal LLM family used as policy: either pooled input
embeddings or final-layer hidden states immediately before the LM head. This is
still controller-side feature extraction; it does not update the base LLM.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEncoder:
    """Dependency-free deterministic text encoder (hashed token features)."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for tok in _TOKEN.findall((text or "").lower()):
            h = int(hashlib.blake2b(tok.encode(), digest_size=4).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec


class SentenceTransformerEncoder:
    """Semantic encoder backed by sentence-transformers (lazy import)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # lazy

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> np.ndarray:
        v = self._model.encode([text or ""], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float64)


class HFLLMHiddenStateEncoder:
    """Causal-LLM prompt encoder backed by Hugging Face Transformers.

    ``mode="last_hidden"`` returns hidden states after the final transformer
    block and before the LM head, pooled over prompt tokens. ``mode="input_emb"``
    is cheaper and returns pooled token embeddings before any decode/transformer
    computation. Both paths are deterministic and read-only.
    """

    def __init__(
        self,
        model_name: str,
        mode: str = "last_hidden",
        pooling: str = "last",
        dtype: str = "auto",
        device: Optional[str] = None,
        max_length: int = 2048,
        trust_remote_code: bool = False,
        normalize: bool = True,
    ):
        if mode not in {"last_hidden", "input_emb"}:
            raise ValueError("mode must be 'last_hidden' or 'input_emb'")
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")

        import torch  # lazy
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

        self.model_name = model_name
        self.mode = mode
        self.pooling = pooling
        self.max_length = int(max_length)
        self.normalize = bool(normalize)
        self._cache: Dict[str, np.ndarray] = {}

        torch_dtype = "auto"
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32

        self.tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)

    def _pool(self, states, attention_mask):
        import torch

        mask = attention_mask.to(states.device).bool()
        if self.pooling == "last":
            last_idx = mask.long().sum(dim=1).clamp_min(1) - 1
            return states[torch.arange(states.shape[0], device=states.device), last_idx]
        weights = mask.unsqueeze(-1).to(states.dtype)
        return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def encode(self, text: str) -> np.ndarray:
        key = text or ""
        cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()

        import torch

        batch = self.tok(
            key,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            if self.mode == "input_emb":
                states = self.model.get_input_embeddings()(batch["input_ids"])
            else:
                out = self.model(
                    **batch,
                    output_hidden_states=True,
                    use_cache=False,
                )
                states = out.hidden_states[-1]
            pooled = self._pool(states, batch["attention_mask"]).detach().float().cpu().numpy()[0]
        vec = np.asarray(pooled, dtype=np.float64)
        if self.normalize:
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
        self._cache[key] = vec.copy()
        return vec


def build_encoder(kind: str = "hash", **kwargs):
    """Factory: ``hash`` (offline), ``st`` or ``hf`` LLM hidden-state features."""
    if kind == "hash":
        return HashingEncoder(**kwargs)
    if kind == "st":
        return SentenceTransformerEncoder(**kwargs)
    if kind == "hf":
        return HFLLMHiddenStateEncoder(**kwargs)
    raise KeyError(f"unknown encoder '{kind}' (use 'hash' or 'st')")
