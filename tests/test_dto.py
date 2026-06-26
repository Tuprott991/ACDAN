"""Differentiable Textual Optimization: convergence and gradient correctness."""

import numpy as np

from acdan.config import DTOConfig
from acdan.dto import DifferentiableTextOptimizer
from acdan.graph import make_entropy_hook
from acdan.rewards import MockProcessReward
from acdan.tasks.synthetic import make_task


def _setup():
    t = make_task("code-000", "code", seed=0)
    prm = MockProcessReward(seed=0)
    latent = 3.0 * np.ones(8)
    prior = np.zeros((t.horizon, t.vocab_size))
    return t, prm, latent, prior


def test_objective_decreases_monotone_ish():
    t, prm, latent, prior = _setup()
    dto = DifferentiableTextOptimizer(DTOConfig(iters=40, lr=0.8), prm,
                                      make_entropy_hook(t))
    plan = dto.optimize(t, latent, prior)
    trace = plan.objective_trace
    assert len(trace) == 40
    # Final objective should be well below the initial one.
    assert trace[-1] < trace[0]


def test_dto_improves_prm_score_over_greedy():
    t, prm, latent, prior = _setup()
    greedy = DifferentiableTextOptimizer.greedy_decode(prior)
    dto = DifferentiableTextOptimizer(DTOConfig(), prm, make_entropy_hook(t))
    refined = dto.optimize(t, latent, prior)
    s_greedy = prm.score_probs(t, latent, greedy.probs)
    s_refined = prm.score_probs(t, latent, refined.probs)
    assert s_refined >= s_greedy


def test_gradient_matches_finite_difference():
    t, prm, latent, prior = _setup()
    dto = DifferentiableTextOptimizer(DTOConfig(), prm, make_entropy_hook(t))
    coeffs = (0.1, 2.0, 0.02, 0.01)
    logits = np.random.default_rng(0).normal(size=(t.horizon, t.vocab_size))
    obj, grad = dto._evaluate(t, latent, logits, prior=prior * 0 + 1.0 / t.vocab_size, coeffs=coeffs)
    # Finite-difference check on a few entries.
    eps = 1e-6
    for (i, j) in [(0, 0), (1, 2), (t.horizon - 1, t.vocab_size - 1)]:
        pert = logits.copy()
        pert[i, j] += eps
        o2, _ = dto._evaluate(t, latent, pert, prior=prior * 0 + 1.0 / t.vocab_size, coeffs=coeffs)
        fd = (o2.value - obj.value) / eps
        assert np.isclose(fd, grad[i, j], rtol=1e-3, atol=1e-4)


def test_greedy_decode_no_steps():
    _, _, _, prior = _setup()
    plan = DifferentiableTextOptimizer.greedy_decode(prior + np.eye(prior.shape[0], prior.shape[1]))
    assert plan.dto_steps == 0
    assert len(plan.actions) == prior.shape[0]
