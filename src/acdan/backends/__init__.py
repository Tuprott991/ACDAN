"""Real (GPU/API) backends for ACDAN.

Everything here imports its heavy dependency (vLLM, torch, transformers,
sentence-transformers, anthropic) **lazily, inside functions/constructors**, so
that ``import acdan`` and the offline test suite never require them. Install the
extras only on the experiment VM:

    pip install -e ".[gpu]"      # vllm + transformers + torch + sentence-transformers
    pip install anthropic        # only for the Claude independent-verifier/judge

Each adapter implements the same tiny Protocol the core programs against
(``CoreModel`` / ``ProcessRewardModel`` / ``IndependentVerifier``), so swapping a
mock for a real backend is a one-line change in ``run_experiment.py`` — the
agent, DTO, graph, inertia, and verification code are untouched.
"""

from acdan.backends.encoder import HashingEncoder, build_encoder

__all__ = ["HashingEncoder", "build_encoder"]
