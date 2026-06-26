"""Latent reasoning + in-place TTT."""

import numpy as np

from acdan.config import LatentConfig
from acdan.latent_reasoning import LatentReasoner
from acdan.tasks.synthetic import FEATURE_DIM


def _reasoner(seed=0):
    return LatentReasoner(LatentConfig(), feature_dim=FEATURE_DIM, seed=seed)


def test_ttt_reduces_reconstruction_loss():
    r = _reasoner()
    x = np.random.default_rng(0).normal(size=FEATURE_DIM)
    trace = r.reason(x, use_ttt=True)
    assert len(trace.ttt_losses) == r.cfg.ttt_steps
    assert trace.ttt_losses[-1] <= trace.ttt_losses[0] + 1e-9


def test_deep_path_has_higher_quality_than_shallow():
    r = _reasoner()
    rng = np.random.default_rng(1)
    deep_q, shallow_q = [], []
    for _ in range(20):
        x = rng.normal(size=FEATURE_DIM)
        deep_q.append(r.quality(r.reason(x).final_state))
        shallow_q.append(r.quality(r.passthrough(x).final_state))
    assert np.mean(deep_q) > np.mean(shallow_q)


def test_output_shape_and_determinism():
    r = _reasoner()
    x = np.ones(FEATURE_DIM)
    a = r.reason(x).final_state
    b = r.reason(x).final_state
    assert a.shape == (r.dim,)
    np.testing.assert_allclose(a, b)


def test_feature_dim_mismatch_raises():
    r = _reasoner()
    try:
        r.reason(np.ones(FEATURE_DIM + 1))
        assert False, "expected ValueError"
    except ValueError:
        pass
