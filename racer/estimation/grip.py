"""Grip (tyre-road friction) utilities.

The MPC currently assumes a fixed friction coefficient (config.VehicleParams.mu). On a track
that changes from dry to wet, a fixed-grip controller either slides off (mu over-estimated)
or drives slower than it could (mu under-estimated). These helpers support estimating grip
online and simulating friction that varies along the lap.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

G = 9.81


def friction_utilization(ax: float, ay: float, mu: float, g: float = G) -> float:
    """Fraction of available grip in use (friction circle). Above 1.0 the tyres slide."""
    if not all(math.isfinite(v) for v in (ax, ay, mu, g)) or mu <= 0 or g <= 0:
        raise ValueError("accelerations must be finite; mu and g must be positive and finite")
    return math.hypot(ax, ay) / (mu * g)


def mu_lower_bound(ax: float, ay: float, g: float = G) -> float:
    """If the car achieved (ax, ay) without sliding, the road offers at least this much grip.

    Note: grip is only *observable* near the limit. Driving gently tells you mu >= small
    value, not what mu actually is. The estimator has to account for this.
    """
    if not all(math.isfinite(v) for v in (ax, ay, g)) or g <= 0:
        raise ValueError("accelerations must be finite and g positive and finite")
    return math.hypot(ax, ay) / g


@dataclass(frozen=True)
class GripSchedule:
    """Piecewise-constant friction along the lap: segments of (s_start [m], mu)."""

    segments: tuple[tuple[float, float], ...]
    lap_length: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lap_length) or self.lap_length <= 0:
            raise ValueError("lap_length must be positive and finite")
        if any(not math.isfinite(s) or not math.isfinite(mu) for s, mu in self.segments):
            raise ValueError("segments must be finite")
        if not self.segments or self.segments[0][0] != 0.0:
            raise ValueError("first segment must start at s = 0")
        starts = [s for s, _ in self.segments]
        if starts != sorted(starts) or len(set(starts)) != len(starts):
            raise ValueError("segment starts must be strictly increasing")
        if starts[-1] >= self.lap_length:
            raise ValueError("segments must start before the end of the lap")
        if any(mu <= 0 for _, mu in self.segments):
            raise ValueError("mu must be positive")

    def mu_at(self, s: float) -> float:
        if not math.isfinite(s):
            raise ValueError("progress must be finite")
        s = s % self.lap_length
        starts = [x for x, _ in self.segments]
        return self.segments[bisect.bisect_right(starts, s) - 1][1]


def dry_to_wet(lap_length: float, wet_start: float = 0.4, wet_end: float = 0.7,
               mu_dry: float = 1.0489, mu_wet: float = 0.6) -> GripSchedule:
    """A wet patch between fractions wet_start and wet_end of the lap (default mu from config)."""
    if not 0.0 < wet_start < wet_end < 1.0:
        raise ValueError("need 0 < wet_start < wet_end < 1")
    return GripSchedule(
        segments=((0.0, mu_dry), (wet_start * lap_length, mu_wet), (wet_end * lap_length, mu_dry)),
        lap_length=lap_length,
    )
