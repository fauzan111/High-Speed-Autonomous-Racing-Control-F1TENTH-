import numpy as np
import pytest

from racer.estimation.ekf import ExtendedKalmanFilter
from racer.estimation.grip import (
    GripSchedule,
    dry_to_wet,
    friction_utilization,
    mu_lower_bound,
)


def test_friction_circle():
    assert friction_utilization(0.0, 9.81, mu=1.0) == pytest.approx(1.0)
    assert friction_utilization(3.0, 4.0, mu=1.0, g=10.0) == pytest.approx(0.5)
    assert mu_lower_bound(3.0, 4.0, g=10.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        friction_utilization(1.0, 1.0, mu=0.0)


def test_dry_to_wet_schedule_wraps_around_the_lap():
    sched = dry_to_wet(lap_length=100.0, wet_start=0.4, wet_end=0.7, mu_dry=1.0, mu_wet=0.6)
    assert sched.mu_at(10) == 1.0
    assert sched.mu_at(50) == 0.6
    assert sched.mu_at(80) == 1.0
    assert sched.mu_at(150) == 0.6  # second lap


def test_schedule_validation():
    with pytest.raises(ValueError):
        GripSchedule(segments=((5.0, 1.0),), lap_length=100.0)
    with pytest.raises(ValueError):
        GripSchedule(segments=((0.0, 1.0), (0.0, 0.5)), lap_length=100.0)


def test_ekf_converges_on_a_constant():
    rng = np.random.default_rng(0)
    true_mu = 0.6
    ekf = ExtendedKalmanFilter(x0=[1.0], P0=[[1.0]])
    for _ in range(300):
        ekf.predict(lambda x: x, F=[[1.0]], Q=[[1e-6]])
        ekf.update([true_mu + rng.normal(0, 0.05)], lambda x: x, H=[[1.0]], R=[[0.05**2]])
    assert ekf.x[0] == pytest.approx(true_mu, abs=0.02)
    assert ekf.P[0, 0] < 1e-3


@pytest.mark.parametrize("lap_length", [0.0, -1.0, float('nan'), float('inf')])
def test_schedule_rejects_invalid_length(lap_length):
    with pytest.raises(ValueError):
        GripSchedule(((0.0, 1.0),), lap_length)


def test_schedule_rejects_nonfinite_segments_and_progress():
    with pytest.raises(ValueError):
        GripSchedule(((0.0, float('nan')),), 100.0)
    with pytest.raises(ValueError):
        dry_to_wet(100.0).mu_at(float('inf'))


def test_grip_estimator_learns_from_noisy_yaw_response():
    from racer.config import VehicleParams
    from racer.estimation.online import GripEstimator, predict_single_track
    params = VehicleParams()
    estimator = GripEstimator()
    state = np.array([0.0, 0.0, 0.6])
    rng = np.random.default_rng(2)
    for i in range(1000):
        steering = 0.07 + 0.03 * np.sin(i * 0.03)
        state = predict_single_track(state, 6.0, steering, 0.0, 0.01, params)
        result = estimator.update(6.0, steering, state[0] + rng.normal(0, 0.01), 0.01)
    assert result.excited
    assert result.mu_hat == pytest.approx(0.6, abs=0.06)
    assert result.sigma < estimator.initial_sigma
    assert result.mu_used < result.mu_hat
    assert np.linalg.eigvalsh(estimator.filter.P).min() >= -1e-10


def test_grip_estimator_retains_uncertainty_without_excitation():
    from racer.estimation.online import GripEstimator
    estimator = GripEstimator()
    for _ in range(100):
        result = estimator.update(5.0, 0.0, 0.0, 0.01)
    assert not result.excited
    assert result.mu_hat == estimator.initial_mu
    assert result.sigma >= estimator.initial_sigma
    estimator.reset()
    for _ in range(100):
        result = estimator.update(0.0, 0.0, 0.0, 0.01)
    assert not result.excited
    assert result.mu_hat == estimator.initial_mu
    assert result.sigma >= estimator.initial_sigma


@pytest.mark.parametrize('field', ['speed', 'steering', 'yaw_rate', 'dt', 'acceleration'])
def test_grip_estimator_rejects_nonfinite_input(field):
    from racer.estimation.online import GripEstimator
    measurements = dict(speed=5.0, steering=0.1, yaw_rate=1.0, dt=0.01, acceleration=0.0)
    measurements[field] = float('nan')
    with pytest.raises(ValueError):
        GripEstimator().update(**measurements)


def test_grip_estimator_tracks_dry_wet_transition():
    from racer.config import VehicleParams
    from racer.estimation.online import GripEstimator, predict_single_track
    params = VehicleParams()
    estimator = GripEstimator()
    state = np.array([0.0, 0.0, 1.0489])
    for i in range(1200):
        state[2] = 1.0489 if i < 500 else 0.6
        steering = 0.07 + 0.03 * np.sin(i * 0.03)
        state = predict_single_track(state, 6.0, steering, 0.0, 0.01, params)
        result = estimator.update(6.0, steering, state[0], 0.01)
    assert result.mu_hat == pytest.approx(0.6, abs=0.06)


def test_low_speed_reset_discards_stale_dynamic_covariance():
    from racer.estimation.online import GripEstimator
    estimator = GripEstimator()
    estimator.update(5.0, 0.05, 0.5, 0.01)
    estimator.filter.P = np.array([[0.1, 0.01, 0.01],
                                   [0.01, 0.1, 0.01],
                                   [0.01, 0.01, 0.1]])
    estimator.update(0.0, 0.0, 0.0, 0.01)
    assert estimator.filter.P == pytest.approx(np.diag([
        estimator.yaw_noise**2, 0.05**2, 0.1 + 0.0025 * 0.01]))


def test_synthetic_validation_is_deterministic_and_replay_consistent(tmp_path):
    import csv
    from scripts.replay_grip import replay
    from scripts.validate_grip import validate
    first_dir = tmp_path / 'first'
    second_dir = tmp_path / 'second'
    first = validate(first_dir, seed=7)
    second = validate(second_dir, seed=7)
    assert first == second
    assert first['validation_kind'] == 'matched_model_synthetic'
    assert first['passed']
    assert (first_dir / 'samples.csv').read_bytes() == (second_dir / 'samples.csv').read_bytes()
    replayed = first_dir / 'replayed.csv'
    replay(first_dir / 'samples.csv', replayed)
    with (first_dir / 'samples.csv').open() as source, replayed.open() as target:
        original = list(csv.DictReader(source))[1:]
        repeated = list(csv.DictReader(target))
    assert len(original) == len(repeated)
    for a, b in zip(original, repeated):
        assert float(a['mu_hat']) == pytest.approx(float(b['mu_hat']), abs=1e-9)
        assert float(a['sigma']) == pytest.approx(float(b['sigma']), abs=1e-9)
