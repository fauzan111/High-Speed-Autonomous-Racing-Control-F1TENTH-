"""
Racing-line representation.

Loads the pre-optimized minimum-time trajectory shipped with each F1TENTH map
(columns: s, x, y, psi, kappa, vx, ax) and exposes the queries a tracking
controller needs:

  * nearest_index(x, y)         -> closest waypoint (progress localization)
  * lookahead_point(x, y, Ld)   -> point Ld metres ahead (pure pursuit)
  * sample_ahead(x, y, ds, N)   -> N reference states ahead (MPC horizon)

The trajectory is treated as a closed loop; arc length wraps at the total
track length so a car can lap indefinitely.
"""
from __future__ import annotations

import numpy as np

from . import config


class Raceline:
    def __init__(self, s, x, y, psi, kappa, vx, ax):
        self.s = np.asarray(s, dtype=float)
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.psi = np.asarray(psi, dtype=float)      # heading [rad]
        self.kappa = np.asarray(kappa, dtype=float)   # curvature [1/m]
        self.vx = np.asarray(vx, dtype=float)         # target speed [m/s]
        self.ax = np.asarray(ax, dtype=float)         # target accel [m/s^2]
        self._pts = np.column_stack((self.x, self.y))
        # Closed-loop length = last arc-length + gap back to the first point.
        seg_close = np.hypot(self.x[0] - self.x[-1], self.y[0] - self.y[-1])
        self.length = float(self.s[-1] + seg_close)

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_csv(cls, path) -> "Raceline":
        data = np.loadtxt(str(path), delimiter=";", comments="#")
        s, x, y, psi, kappa, vx, ax = data.T
        return cls(s, x, y, psi, kappa, vx, ax)

    @classmethod
    def load(cls, map_name: str = config.DEFAULT_MAP) -> "Raceline":
        return cls.from_csv(config.raceline_csv(map_name))

    def __len__(self) -> int:
        return len(self.s)

    # -- queries --------------------------------------------------------------
    def nearest_index(self, x: float, y: float) -> int:
        """Index of the closest waypoint to (x, y)."""
        d2 = (self.x - x) ** 2 + (self.y - y) ** 2
        return int(np.argmin(d2))

    def progress(self, x: float, y: float) -> float:
        """Arc length s at the nearest waypoint."""
        return float(self.s[self.nearest_index(x, y)])

    def lookahead_point(self, x: float, y: float, lookahead: float):
        """
        Return (px, py, target_vx, idx) for the waypoint ~`lookahead` metres
        ahead of the car along the racing line (wrapping around the lap).
        """
        i0 = self.nearest_index(x, y)
        target_s = (self.s[i0] + lookahead) % self.length
        idx = int(np.searchsorted(self.s, target_s) % len(self.s))
        return self.x[idx], self.y[idx], self.vx[idx], idx

    def min_speed_ahead(self, x: float, y: float, preview: float) -> float:
        """
        Minimum target speed over the next `preview` metres of racing line.

        Commanding this (instead of the speed at a single far point) forces the
        car to brake *before* the slowest part of an upcoming corner, which a
        rate-limited actuator cannot do if it only reacts on corner entry.
        """
        i0 = self.nearest_index(x, y)
        s0 = self.s[i0]
        n = len(self.s)
        span = max(1, int(preview / max(np.median(np.diff(self.s)), 1e-6)))
        idxs = (i0 + np.arange(0, span + 1)) % n
        return float(self.vx[idxs].min())

    def sample_ahead(self, x: float, y: float, ds: float, n: int):
        """
        Sample `n` reference states spaced `ds` metres ahead of the car,
        interpolated along arc length. Returns arrays (xr, yr, psir, vxr),
        each length n. Used to build an MPC reference over the horizon.
        """
        s0 = self.progress(x, y)
        s_query = (s0 + ds * np.arange(1, n + 1)) % self.length
        xr = np.interp(s_query, self.s, self.x, period=self.length)
        yr = np.interp(s_query, self.s, self.y, period=self.length)
        vxr = np.interp(s_query, self.s, self.vx, period=self.length)
        # Heading must be interpolated as an unwrapped angle to avoid the
        # -pi/+pi discontinuity blowing up near wrap points.
        psi_unwrapped = np.unwrap(self.psi)
        psir = np.interp(s_query, self.s, psi_unwrapped, period=self.length)
        psir = np.arctan2(np.sin(psir), np.cos(psir))
        return xr, yr, psir, vxr

    def start_pose(self) -> np.ndarray:
        """[x, y, theta] at the first waypoint — a clean grid start."""
        return np.array([self.x[0], self.y[0], self.psi[0]])
