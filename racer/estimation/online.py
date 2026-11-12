"""Yaw-rate based grip EKF for the F1TENTH single-track model.

State is [yaw rate, body slip angle, friction coefficient]. Steering must be the
actual measured angle, not its command. Only yaw rate is observed; friction is
inferred through the dynamic response. This identifies the simulator's friction
parameter, which scales linear cornering stiffness even below saturation. Real
tyre peak friction is not generally identifiable from gentle driving this way.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..config import VehicleParams
from .ekf import ExtendedKalmanFilter


def single_track_rates(state, speed, steering, acceleration, params):
    """Return [yaw acceleration, slip-angle rate, 0] (F1TENTH ST equations).

    Reference: github.com/f1tenth/f1tenth_gym/blob/v1.0.0/
    f1tenth_gym/envs/dynamic_models.py, vehicle_dynamics_st dynamic branch.

    Valid in the dynamic regime only; caller handles low speed separately.
    """
    r, beta, mu = state
    p = params
    v = max(float(speed), 0.5)
    wheelbase = p.lf + p.lr
    front = 9.81 * p.lr - acceleration * p.h
    rear = 9.81 * p.lf + acceleration * p.h
    yaw_dot = (mu * p.m / (p.I * wheelbase) * (
        p.lf * p.C_Sf * front * steering
        + (p.lr * p.C_Sr * rear - p.lf * p.C_Sf * front) * beta
        - (p.lf**2 * p.C_Sf * front + p.lr**2 * p.C_Sr * rear) * r / v))
    beta_dot = (mu / (v * wheelbase) * (
        p.C_Sf * front * steering
        - (p.C_Sr * rear + p.C_Sf * front) * beta
        + (p.C_Sr * rear * p.lr - p.C_Sf * front * p.lf) * r / v) - r)
    return np.array([yaw_dot, beta_dot, 0.0])


def predict_single_track(state, speed, steering, acceleration, dt, params):
    """RK4 transition with small substeps for stiff lateral dynamics."""
    result = np.array(state, dtype=float, copy=True)
    # Timestamp subtraction can move an exact 10 ms interval a few ulps above
    # its boundary; avoid changing the substep count during CSV replay.
    count = max(1, math.ceil(dt / 0.005 - 1e-12))
    step = dt / count
    for _ in range(count):
        def f(x):
            return single_track_rates(x, speed, steering, acceleration, params)
        k1 = f(result)
        k2 = f(result + step * k1 / 2)
        k3 = f(result + step * k2 / 2)
        k4 = f(result + step * k3)
        result += step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return result


@dataclass(frozen=True)
class GripEstimate:
    mu_hat: float
    sigma: float
    mu_used: float
    excited: bool


class GripEstimator:
    """Augmented EKF using speed, measured steering and yaw rate.

    Inputs describe the interval ending at the current yaw-rate sample. At low
    speed or low lateral loading, mu is frozen and its variance grows. The
    uncertainty margin is a tuning heuristic, not a probabilistic safety claim.
    """

    def __init__(self, params=None, initial_mu=1.0489, initial_sigma=0.25,
                 min_mu=0.1, max_mu=1.5, margin=1.0, yaw_noise=0.02,
                 min_speed=1.0, excitation_g=0.15):
        values = [initial_mu, initial_sigma, min_mu, max_mu, margin, yaw_noise,
                  min_speed, excitation_g]
        if not np.all(np.isfinite(values)):
            raise ValueError("estimator parameters must be finite")
        if not 0 < min_mu <= initial_mu <= max_mu or initial_sigma <= 0:
            raise ValueError("invalid initial friction, uncertainty or bounds")
        if margin < 0 or yaw_noise <= 0 or min_speed < 0.5 or excitation_g <= 0:
            raise ValueError("invalid noise, margin or excitation threshold")
        self.params = params or VehicleParams()
        self.initial_mu, self.initial_sigma = initial_mu, initial_sigma
        self.min_mu, self.max_mu, self.margin = min_mu, max_mu, margin
        self.yaw_noise, self.min_speed, self.excitation_g = yaw_noise, min_speed, excitation_g
        self.reset()

    def reset(self):
        self.filter = ExtendedKalmanFilter(
            [0.0, 0.0, self.initial_mu],
            np.diag([self.yaw_noise**2, 0.05**2, self.initial_sigma**2]))
        self._initialized = False

    def _estimate(self, excited):
        mu = float(self.filter.x[2])
        sigma = float(np.sqrt(self.filter.P[2, 2]))
        return GripEstimate(mu, sigma, max(self.min_mu, mu - self.margin * sigma), excited)

    def update(self, speed, steering, yaw_rate, dt, acceleration=0.0):
        if not np.all(np.isfinite([speed, steering, yaw_rate, dt, acceleration])):
            raise ValueError("measurements must be finite")
        if not 0 < dt <= 0.1:
            raise ValueError("dt must be in (0, 0.1] seconds")
        if not self._initialized:
            self.filter.x[0] = yaw_rate
            self._initialized = True
            return self._estimate(False)
        excited = bool(speed >= self.min_speed and abs(speed * yaw_rate) >= self.excitation_g * 9.81)
        old_mu = self.filter.x[2]
        old_var = self.filter.P[2, 2]
        if speed < self.min_speed:
            self.filter.x[0] = yaw_rate
            self.filter.x[1] = math.atan(self.params.lr / self.params.wheelbase * math.tan(steering))
            # These two states were replaced directly above, so discard stale
            # dynamic covariances before returning to the high-speed model.
            self.filter.P[:2, :] = 0.0
            self.filter.P[:, :2] = 0.0
            self.filter.P[0, 0] = self.yaw_noise**2
            self.filter.P[1, 1] = 0.05**2
            self.filter.P[2, 2] += 0.0025 * dt
            return self._estimate(False)

        def transition(x):
            return predict_single_track(x, speed, steering, acceleration, dt, self.params)
        x = self.filter.x.copy()
        jacobian = np.empty((3, 3))
        for j in range(3):
            dx = np.zeros(3)
            dx[j] = 1e-5
            jacobian[:, j] = (transition(x + dx) - transition(x - dx)) / 2e-5
        if not excited:
            self.filter.P[:2, 2] = self.filter.P[2, :2] = 0.0
            jacobian[:2, 2] = 0.0
        self.filter.predict(transition, jacobian, np.diag([0.04, 0.0025, 0.0025]) * dt)
        self.filter.update([yaw_rate], lambda x: x[:1], [[1.0, 0.0, 0.0]], [[self.yaw_noise**2]])
        self.filter.x[2] = np.clip(self.filter.x[2], self.min_mu, self.max_mu)
        if not excited:
            self.filter.x[2] = old_mu
            self.filter.P[2, 2] = old_var + 0.0025 * dt
        self.filter.P[2, 2] = max(self.filter.P[2, 2], 0.02**2)
        self.filter.P = (self.filter.P + self.filter.P.T) / 2
        return self._estimate(excited)
