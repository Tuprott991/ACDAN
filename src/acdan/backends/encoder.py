"""Latent feature encoders: turn task prompt text into a feature vector.

The default ``HashingEncoder`` has **no dependencies** (deterministic hashed
bag-of-tokens), so the real pipeline runs offline. On the VM you can switch to a
sentence-transformer for genuine semantic features without changing anything
else — the ``LatentReasoner`` is constructed with ``feature_dim = encoder.dim``.
"""

from __future__ import annotations

import hashlib
import re
from typing import List

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


def build_encoder(kind: str = "hash", **kwargs):
    """Factory: ``hash`` (default, offline) or ``st`` (sentence-transformers)."""
    if kind == "hash":
        return HashingEncoder(**kwargs)
    if kind == "st":
        return SentenceTransformerEncoder(**kwargs)
    raise KeyError(f"unknown encoder '{kind}' (use 'hash' or 'st')")
