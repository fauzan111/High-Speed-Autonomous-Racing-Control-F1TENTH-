"""
Pure Pursuit — geometric path tracker (Stage 1 baseline).

Steering is chosen so the rear axle follows a circular arc to a lookahead point
on the racing line:

    delta = atan2( 2 * L * sin(alpha), Ld )

where L is the wheelbase, Ld the lookahead distance, and alpha the angle to the
lookahead point in the vehicle frame. Target speed is read straight from the
optimized velocity profile of the racing line (scaled by `speed_gain`).

Fast to reason about and hard to make unstable, so it's the baseline the MPC
must beat on lap time.
"""
from __future__ import annotations

import numpy as np

from .. import config
from ..raceline import Raceline
from .base import CarState, Controller


class PurePursuit(Controller):
    name = "pure_pursuit"

    def __init__(
        self,
        raceline: Raceline,
        params: config.VehicleParams | None = None,
        k_lookahead: float = 0.13,   # lookahead grows with speed [s]
        min_lookahead: float = 0.5,  # [m]
        max_lookahead: float = 1.4,  # [m]
        speed_gain: float = 0.8,     # scale the raceline speed profile
        speed_preview: float = 2.5,  # brake for the slowest point within [m]
    ):
        self.rl = raceline
        self.p = params or config.VehicleParams()
        self.k = k_lookahead
        self.min_ld = min_lookahead
        self.max_ld = max_lookahead
        self.speed_gain = speed_gain
        self.speed_preview = speed_preview

    def __call__(self, state: CarState) -> tuple[float, float]:
        # Speed-scheduled lookahead: look further ahead when going fast.
        ld = np.clip(self.k * state.v + self.min_ld, self.min_ld, self.max_ld)
        px, py, _, _ = self.rl.lookahead_point(state.x, state.y, ld)
        # Brake for the slowest point in the preview window, not the far
        # lookahead point (which would be the faster corner exit).
        preview = self.speed_preview + 0.35 * state.v  # longer window when fast
        target_v = self.rl.min_speed_ahead(state.x, state.y, preview)

        # Vector to the lookahead point, expressed in the vehicle frame.
        dx, dy = px - state.x, py - state.y
        alpha = np.arctan2(dy, dx) - state.yaw
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # wrap to [-pi, pi]

        steering = np.arctan2(2.0 * self.p.wheelbase * np.sin(alpha), ld)
        steering = float(np.clip(steering, self.p.s_min, self.p.s_max))

        speed = float(np.clip(target_v * self.speed_gain, self.p.v_min, self.p.v_max))
        return steering, speed
