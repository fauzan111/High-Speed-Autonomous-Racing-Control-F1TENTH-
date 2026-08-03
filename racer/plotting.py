"""Plotting helpers: trajectory overlays and speed traces."""
from __future__ import annotations

import numpy as np

from . import config


def _load_track_boundaries(map_name: str):
    """Left/right track edges from the centerline file (x, y, w_right, w_left)."""
    data = np.loadtxt(str(config.centerline_csv(map_name)), delimiter=",", comments="#")
    cx, cy, wr, wl = data.T
    # Normal direction along the centerline.
    dx = np.gradient(cx)
    dy = np.gradient(cy)
    norm = np.hypot(dx, dy) + 1e-9
    nx, ny = -dy / norm, dx / norm  # left normal
    left = np.column_stack((cx + nx * wl, cy + ny * wl))
    right = np.column_stack((cx - nx * wr, cy - ny * wr))
    return left, right


def plot_trajectories(results, raceline, map_name=config.DEFAULT_MAP,
                      out_path=None, title=None):
    """Overlay one or more RunResults on the racing line and track edges."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8))
    try:
        left, right = _load_track_boundaries(map_name)
        ax.plot(left[:, 0], left[:, 1], color="0.6", lw=0.8)
        ax.plot(right[:, 0], right[:, 1], color="0.6", lw=0.8, label="track edge")
    except Exception:
        pass
    ax.plot(raceline.x, raceline.y, "--", color="0.35", lw=1.0, label="racing line")

    colors = ["#e6194B", "#3cb44b", "#4363d8", "#f58231"]
    for r, c in zip(results, colors):
        sc = ax.scatter(r.xs, r.ys, c=r.vs, cmap="viridis", s=4, zorder=3)
        if r.crashed and len(r.xs):
            ax.scatter([r.xs[-1]], [r.ys[-1]], marker="x", s=120,
                       color=c, zorder=5, label=f"{r.controller} crash")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("speed [m/s]")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title or f"{map_name} — trajectory")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130)
    return fig


def plot_speed(results, out_path=None, title=None):
    """Speed vs distance for each result."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    for r in results:
        ax.plot(r.ts, r.vs, lw=1.2, label=r.controller)
    ax.set_xlabel("time [s]"); ax.set_ylabel("speed [m/s]")
    ax.grid(alpha=0.3); ax.legend()
    ax.set_title(title or "Speed profile")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130)
    return fig
