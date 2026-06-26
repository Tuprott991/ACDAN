"""LLM-as-PRM adapter behavior that should not require a live vLLM backend."""

import numpy as np

from acdan.backends.prm_adapter import LLMAsProcessReward
from acdan.types import Task, softmax


class CountingLLMPRM(LLMAsProcessReward):
    def __init__(self):
        self.sharpness = 4.0
        self._yes = []
        self._no = []
        self._reward_cache = {}
        self._target_cache = {}
        self.n_prompt_tokens = 0
        self.calls = 0

    def _p_yes(self, prompts):
        self.calls += 1
        return [0.75 for _ in prompts]


def _task():
    return Task(
        task_id="cache-test",
        prompt_features=np.zeros(4),
        vocab=("a", "b", "c"),
        horizon=1,
        metadata={"prompt": "choose"},
    )


def test_llm_prm_caches_reward_target_across_dto_style_calls():
    prm = CountingLLMPRM()
    task = _task()
    latent = np.ones(4)
    probs = softmax(np.zeros((task.horizon, task.vocab_size)), axis=1)

    for _ in range(40):
        prm.score_probs(task, latent, probs)
        prm.grad_wrt_probs(task, latent, probs)

    assert prm.calls == 1


def test_llm_prm_math_bonus_uses_candidate_counts():
    prm = CountingLLMPRM()
    task = Task(
        task_id="math-cache-test",
        prompt_features=np.zeros(4),
        vocab=("wrong", "right"),
        horizon=1,
        metadata={
            "prompt": "What is 2+2?",
            "family": "gsm8k",
            "candidate_counts": {"wrong": 1, "right": 4},
        },
    )
    R = prm.step_reward_matrix(task, np.ones(4))
    assert R[0, 1] > R[0, 0]
