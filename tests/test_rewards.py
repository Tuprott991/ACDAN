"""PRM reward surface and Net Information Gain."""

import numpy as np

from acdan.rewards import MockProcessReward, net_information_gain
from acdan.tasks.synthetic import make_task
from acdan.types import softmax


def _task():
    return make_task("math-000", "math", seed=0)


def test_reward_target_is_row_stochastic():
    prm = MockProcessReward(seed=0)
    t = _task()
    latent = np.ones(8)
    target = prm._reward_target(t, latent)
    assert target.shape == (t.horizon, t.vocab_size)
    np.testing.assert_allclose(target.sum(axis=1), 1.0, atol=1e-9)


def test_grad_matches_score_linearity():
    # score is linear in probs with gradient == target, so a finite step along
    # the gradient must increase the score by grad . delta.
    prm = MockProcessReward(seed=1)
    t = _task()
    latent = np.ones(8)
    probs = softmax(np.zeros((t.horizon, t.vocab_size)), axis=1)
    g = prm.grad_wrt_probs(t, latent, probs)
    delta = 1e-4 * g
    s0 = prm.score_probs(t, latent, probs)
    s1 = prm.score_probs(t, latent, probs + delta)
    assert np.isclose(s1 - s0, float(np.sum(g * delta)), rtol=1e-5, atol=1e-9)


def test_better_latent_lowers_noise_signal():
    # A higher-norm (better) latent should, on average, make the PRM argmax agree
    # more with the optimal plan.
    prm = MockProcessReward(seed=0)
    t = _task()
    good = 3.0 * np.ones(8)   # high norm -> high quality -> low noise
    poor = 0.1 * np.ones(8)
    Rg = prm.step_reward_matrix(t, good)
    Rp = prm.step_reward_matrix(t, poor)
    opt = t.optimal_plan
    agree_g = np.mean([int(np.argmax(Rg[h])) == opt[h] for h in range(t.horizon)])
    agree_p = np.mean([int(np.argmax(Rp[h])) == opt[h] for h in range(t.horizon)])
    assert agree_g >= agree_p


def test_nig_linear_and_bounded():
    nig = net_information_gain([0.9, 0.8, 0.1, 0.95])
    assert len(nig) == 4
    # belief is bounded in (0,1), so each increment is in (-1, 1)
    assert all(-1.0 < v < 1.0 for v in nig)


def test_nig_empty():
    assert net_information_gain([]) == []
