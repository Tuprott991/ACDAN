"""PS-GRPO post-training for ACDAN (offline, numpy, analytic gradients).

This package implements the proposal's *Process-Supervised Group Relative Policy
Optimization with Confidence Margin* (PS-GRPO + RLCM) as a complete, dependency-
light reference trainer over a parametric ``(H, V)`` action policy. It is the
training-time counterpart to the test-time DTO loop and is fully self-contained:
no torch, no LLM, deterministic, finite-difference-tested.

The advantage computation (group-relative + process supervision + drop-moment +
confidence margin) is backend-agnostic — to train a real LLM policy on the VM,
replace :class:`PolicyHead` with an LLM action head / LoRA and feed the same
advantages into your autograd optimizer; everything else carries over.
"""

from acdan.training.policy import PolicyHead
from acdan.training.psgrpo import PSGRPOConfig, PSGRPOTrainer, TrainHistory
from acdan.training.tasks import make_learnable_suite, make_learnable_task

__all__ = [
    "PolicyHead",
    "PSGRPOConfig",
    "PSGRPOTrainer",
    "TrainHistory",
    "make_learnable_suite",
    "make_learnable_task",
]
