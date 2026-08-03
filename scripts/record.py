"""
Record a polished bird's-eye MP4 of a controller lapping the track: the car
traces the racing line, its trail is colored by speed, and a HUD shows live
speed + lap time. Built for sharing (LinkedIn / demos).

Usage:
    python -m scripts.record --controller mpc
    python -m scripts.record --controller mpc --seconds 30 --fps 30
"""
from __future__ import annotations

import argparse

import cv2
import gymnasium as gym
import numpy as np

import f1tenth_gym  # noqa: F401  (registers env)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from racer import config, plotting
from racer.controllers.base import CarState
from racer.raceline import Raceline


def build_controller(name, rl):
    if name == "pure_pursuit":
        from racer.controllers.pure_pursuit import PurePursuit
        return PurePursuit(rl)
    from racer.controllers.mpc import MPCController
    return MPCController(rl)


def simulate(controller, rl, map_name, laps, max_time):
    """Run the controller and log the trajectory for ~`laps` laps."""
    env = gym.make("f1tenth_gym:f1tenth-v0",
                   config={"num_agents": 1, "map": map_name}, render_mode=None)
    obs, _ = env.reset(options={"poses": rl.start_pose().reshape(1, 3)})
    controller.reset()
    dt = config.SIM_TIMESTEP
    xs, ys, vs, ts = [], [], [], []
    half = rl.length / 2.0
    prev_s, wraps = 0.0, 0
    wrap_idx = []                       # step index at each finish-line crossing
    for step in range(int(max_time / dt)):
        st = CarState(float(obs["poses_x"][0]), float(obs["poses_y"][0]),
                      float(obs["poses_theta"][0]), float(obs["linear_vels_x"][0]),
                      float(obs["ang_vels_z"][0]))
        steer, spd = controller(st)
        obs, *_ , = env.step(np.array([[steer, spd]], dtype=np.float32))
        xs.append(st.x); ys.append(st.y); vs.append(st.v); ts.append(step * dt)
        s = rl.s[rl.nearest_index(st.x, st.y)]
        if s < prev_s - half:
            wraps += 1
            wrap_idx.append(len(xs) - 1)
            if wraps >= 2:              # standing lap + one full flying lap
                break
        prev_s = s
        if obs["collisions"][0]:
            break
    env.close()
    return (np.array(xs), np.array(ys), np.array(vs), np.array(ts), wrap_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="mpc", choices=["mpc", "pure_pursuit"])
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--seconds", type=float, default=30.0, help="target video length")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--laps", type=float, default=1.15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rl = Raceline.load(args.map)
    ctrl = build_controller(args.controller, rl)
    print(f"simulating {ctrl.name} ...")
    xs, ys, vs, ts, wrap_idx = simulate(ctrl, rl, args.map, args.laps, max_time=200.0)
    print(f"  logged {len(xs)} steps ({ts[-1]:.1f}s sim), {len(wrap_idx)} crossings")

    # Show exactly one clean flying lap: from the 1st finish-line crossing (car
    # already at speed) to the 2nd. Falls back to the whole log if incomplete.
    if len(wrap_idx) >= 2:
        start, end = wrap_idx[0], wrap_idx[1]
    else:
        start, end = 0, len(xs) - 1
    lap_t0 = ts[start]

    # subsample so the movie is ~`seconds` long at `fps`
    n_frames = int(args.seconds * args.fps)
    idx = np.linspace(start, end, min(n_frames, end - start)).astype(int)

    try:
        left, right = plotting._load_track_boundaries(args.map)
    except Exception:
        left = right = None

    out = args.out or str(config.RESULTS_DIR / f"{args.controller}_lap.mp4")
    vmin, vmax = float(vs.min()), float(vs.max())

    # figure -> RGB frame
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    fig.patch.set_facecolor("#0d1117")
    writer = None
    print(f"rendering {len(idx)} frames -> {out}")
    for fi, i in enumerate(idx):
        ax.clear()
        ax.set_facecolor("#0d1117")
        if left is not None:
            ax.plot(left[:, 0], left[:, 1], color="#3a3f4b", lw=1.2)
            ax.plot(right[:, 0], right[:, 1], color="#3a3f4b", lw=1.2)
        ax.plot(rl.x, rl.y, "--", color="#5a6270", lw=0.8)
        # speed-colored trail built up over the current lap
        ax.scatter(xs[start:i + 1], ys[start:i + 1], c=vs[start:i + 1],
                   cmap="viridis", s=6, vmin=vmin, vmax=vmax, zorder=3)
        # the car
        ax.scatter([xs[i]], [ys[i]], c="white", edgecolors="black",
                   s=90, zorder=5)
        ax.set_aspect("equal"); ax.axis("off")

        lap_t = max(ts[i] - lap_t0, 0.0)
        bg = dict(facecolor="#0d1117", alpha=0.65, edgecolor="none", pad=3)
        ax.text(0.02, 0.98, f"{ctrl.name.replace('_', ' ').upper()}  ·  F1TENTH",
                transform=ax.transAxes, color="white", fontsize=13,
                fontweight="bold", va="top", bbox=bg, zorder=10)
        ax.text(0.02, 0.935, "tracking the optimal racing line at the grip limit",
                transform=ax.transAxes, color="#9aa4b2", fontsize=9, va="top",
                bbox=bg, zorder=10)
        ax.text(0.02, 0.065, f"SPEED  {vs[i]:4.1f} m/s",
                transform=ax.transAxes, color="#39d353", fontsize=15,
                fontweight="bold", va="bottom", family="monospace", bbox=bg, zorder=10)
        ax.text(0.02, 0.02, f"LAP    {lap_t:4.1f} s",
                transform=ax.transAxes, color="white", fontsize=15,
                fontweight="bold", va="bottom", family="monospace", bbox=bg, zorder=10)

        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (w, h))
        writer.write(frame)
        if fi % 100 == 0:
            print(f"  frame {fi}/{len(idx)}")
    writer.release()
    plt.close(fig)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
