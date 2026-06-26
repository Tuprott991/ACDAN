"""Tool Usage Inertia / Inertial Sensing."""

from acdan.config import InertiaConfig
from acdan.inertia import InertialSensor


def test_high_inertia_fires_after_enough_observations():
    sensor = InertialSensor(InertiaConfig(threshold=0.6, min_observations=3), vocab_size=4)
    # Strong, repeated transition 0 -> 1.
    sensor.fit([[0, 1], [0, 1], [0, 1], [0, 1]])
    decision = sensor.query(prev_action=0)
    assert decision.use_inertia
    assert decision.predicted_action == 1
    assert decision.inertia_score > 0.6


def test_respects_min_observations():
    sensor = InertialSensor(InertiaConfig(threshold=0.5, min_observations=10), vocab_size=4)
    sensor.fit([[0, 1], [0, 1]])  # only 2 observations from state 0
    decision = sensor.query(prev_action=0)
    assert not decision.use_inertia


def test_first_step_never_uses_inertia():
    sensor = InertialSensor(InertiaConfig(), vocab_size=4)
    sensor.fit([[0, 1, 2]])
    assert not sensor.query(prev_action=None).use_inertia


def test_low_inertia_does_not_fire():
    sensor = InertialSensor(InertiaConfig(threshold=0.9, min_observations=1), vocab_size=4)
    # Uniform-ish transitions -> max prob well below threshold.
    sensor.fit([[0, 1], [0, 2], [0, 3], [0, 1], [0, 2], [0, 3]])
    assert not sensor.query(prev_action=0).use_inertia


def test_transition_matrix_rows_sum_to_one():
    sensor = InertialSensor(InertiaConfig(), vocab_size=5)
    sensor.fit([[0, 1, 2, 3, 4]])
    T = sensor.transition_matrix()
    assert T.shape == (5, 5)
    assert abs(T.sum(axis=1).mean() - 1.0) < 1e-9
