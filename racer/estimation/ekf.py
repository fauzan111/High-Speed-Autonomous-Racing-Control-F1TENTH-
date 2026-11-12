"""A small, generic Extended Kalman Filter (numpy only)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, x0, P0) -> None:
        self.x = np.asarray(x0, dtype=float).reshape(-1)
        self.P = np.asarray(P0, dtype=float)
        n = self.x.size
        if self.P.shape != (n, n):
            raise ValueError("P0 must be n x n")

    def predict(self, f: Callable[[np.ndarray], np.ndarray], F, Q) -> None:
        """x <- f(x);  P <- F P F^T + Q   (F is the Jacobian of f at the current x)."""
        F = np.asarray(F, dtype=float)
        self.x = np.asarray(f(self.x), dtype=float).reshape(-1)
        self.P = F @ self.P @ F.T + np.asarray(Q, dtype=float)

    def update(self, z, h: Callable[[np.ndarray], np.ndarray], H, R) -> np.ndarray:
        """Measurement update (Joseph form, numerically stable). Returns the innovation."""
        z = np.asarray(z, dtype=float).reshape(-1)
        H = np.asarray(H, dtype=float)
        R = np.asarray(R, dtype=float)
        y = z - np.asarray(h(self.x), dtype=float).reshape(-1)
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ y
        I_KH = np.eye(self.x.size) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return y
