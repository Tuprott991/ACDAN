"""Two-layer agentic computation graph: entropy, dead steps, entropy-hook grad."""

import numpy as np

from acdan.graph import AgenticComputationGraph, make_entropy_hook
from acdan.tasks.synthetic import make_task
from acdan.types import softmax


def _task():
    return make_task("tool-000", "tool", seed=0)


def test_execution_layer_is_a_chain():
    t = _task()
    g = AgenticComputationGraph(t, actions=[0, 1, 2, 3, 4])
    assert g.ex_edges == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_von_neumann_entropy_nonnegative():
    t = _task()
    g = AgenticComputationGraph(t, actions=[0, 1, 2, 3, 4])
    h = g.von_neumann_entropy()
    assert h >= -1e-9
    # entropy of an n-node density matrix is at most log(n)
    assert h <= np.log(5) + 1e-6


def test_dead_step_detection_and_pruning():
    t = _task()
    actions = [0, 1, 2, 3, 4]
    # Force a negative-NIG, disconnected middle step to be flagged dead.
    nig = [1.0, -0.5, 1.0, 1.0, 1.0]
    g = AgenticComputationGraph(t, actions=actions, nig=nig, sim_threshold=2.0)
    # sim_threshold=2.0 => no ED edges => every non-final step is disconnected.
    dead = g.dead_steps()
    assert 1 in dead          # the negative-NIG step is dead
    assert (len(actions) - 1) not in dead  # final step never pruned
    pruned, dead_idx = g.prune()
    assert len(pruned) == len(actions) - len(dead_idx)


def test_entropy_hook_gradient_matches_finite_difference():
    t = _task()
    hook = make_entropy_hook(t)
    probs = softmax(np.random.default_rng(0).normal(size=(t.horizon, t.vocab_size)), axis=1)
    value, grad = hook(probs)
    assert value >= 0.0
    eps = 1e-6
    for (i, j) in [(0, 0), (2, 1)]:
        pert = probs.copy()
        pert[i, j] += eps
        v2, _ = hook(pert)
        fd = (v2 - value) / eps
        assert np.isclose(fd, grad[i, j], rtol=1e-3, atol=1e-4)


def test_no_dead_steps_when_all_helpful():
    t = _task()
    g = AgenticComputationGraph(t, actions=[0, 1, 2, 3, 4], nig=[1.0] * 5)
    assert g.dead_steps() == []
