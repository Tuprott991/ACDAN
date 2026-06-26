"""Self-verification, confidence probe, margin, and calibration."""

import numpy as np

from acdan.config import VerificationConfig
from acdan.verification import (
    ConfidenceProbe,
    MockIndependentVerifier,
    SelfVerifier,
    expected_calibration_error,
    margin_score,
)
from acdan.tasks.synthetic import make_task
from acdan.types import Plan


def test_probe_fits_separable_data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.5, size=(50, 4)), rng.normal(2, 0.5, size=(50, 4))])
    y = np.array([0] * 50 + [1] * 50, dtype=float)
    probe = ConfidenceProbe(input_dim=4, seed=0).fit(X, y, epochs=400, lr=0.2)
    preds = np.array([probe.predict(x) for x in X])
    acc = np.mean((preds > 0.5) == (y > 0.5))
    assert acc > 0.9
    assert probe.is_fitted


def test_margin_in_unit_range():
    logits = np.array([[5.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    plan = Plan(actions=[0, 0], logits=logits)
    m = margin_score(plan)
    assert 0.0 < m <= 1.0


def test_ece_perfect_calibration_is_low():
    # Confidences equal to accuracy in each regime -> ECE near 0.
    conf = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    correct = [True, True, True, False, False, False, False, False, False, False]
    # 3/3 correct at 0.9 conf; 0/7 correct at 0.1 conf -> well calibrated-ish
    assert expected_calibration_error(conf, correct) < 0.2


def test_independent_verifier_rewards_matching_plan():
    t = make_task("math-000", "math", seed=0)
    iv = MockIndependentVerifier(noise=0.0, seed=0)
    good = Plan(actions=list(t.optimal_plan), logits=np.zeros((t.horizon, t.vocab_size)))
    bad = Plan(actions=[(a + 1) % t.vocab_size for a in t.optimal_plan],
               logits=np.zeros((t.horizon, t.vocab_size)))
    assert iv.agreement(t, good) > iv.agreement(t, bad)


def test_abstention_below_threshold():
    cfg = VerificationConfig(accept_threshold=0.6, abstain_threshold=0.5)
    t = make_task("math-000", "math", seed=0)
    verifier = SelfVerifier(cfg, probe=None, independent=None)
    # A flat plan -> tiny margin -> low confidence -> abstain.
    plan = Plan(actions=[0] * t.horizon, logits=np.zeros((t.horizon, t.vocab_size)))
    out = verifier.verify(t, np.zeros(8), plan, mean_prm=0.0, use_calibration=False)
    assert 0.0 <= out.confidence <= 1.0
    assert out.abstained == (out.confidence < cfg.abstain_threshold)
