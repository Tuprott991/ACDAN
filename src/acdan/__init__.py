"""ACDAN: Adaptive Calibrated Differentiable Agentic Networks.

A reproducible, dependency-light, *offline* reference implementation of the
core ideas from the AAAI-2027 proposal:

    "Adaptive Calibrated Differentiable Agentic Networks: A New Direction for
     Universal Test-Time Agentic Systems."

The package deliberately uses lightweight, well-documented numpy implementations
and *mock / pluggable* interfaces (rewards, core model, datasets) so that every
experiment runs locally without downloading any weights or datasets. Real
backends can be plugged in later through ``acdan.registry`` without changing the
core architecture.

See ``docs/module_to_paper_mapping.md`` for how each symbol maps to the paper.
"""

from __future__ import annotations

__version__ = "0.1.0"

from acdan.config import ACDANConfig, AblationFlags
from acdan.agent import ACDANAgent, AgentResult

__all__ = [
    "__version__",
    "ACDANConfig",
    "AblationFlags",
    "ACDANAgent",
    "AgentResult",
]
