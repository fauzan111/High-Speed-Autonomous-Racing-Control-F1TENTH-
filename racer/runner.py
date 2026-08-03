"""
Episode runner: drives a Controller around a map in the F1TENTH gym, logs the
trajectory, and measures a clean *flying* lap time.

Lap timing note: the gym starts the car on the finish line, so the first partial
lap is a rolling start (warmup). We time from the first finish-line crossing
(lap_counts 0 -> 1) to the next (1 -> 2): one full flying lap.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

import f1tenth_gym  # noqa: F401  (registers the env)

from . import config
from .controllers.base import CarState, Controller
from .raceline import Raceline


@dataclass
class RunResult:
    controller: str
    map_name: str
    completed: bool
    crashed: bool
    lap_time: float | None            # flying-lap time [s], None if not completed
    sim_time: float                   # total simulated time [s]
    max_speed: float
    mean_speed: float
    mean_cross_track: float           # mean lateral error vs raceline [m]
    max_cross_track: float
    solve_ms_mean: float              # mean controller compute time [ms]
    # Per-step logs (world frame)
    xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    ys: np.ndarray = field(default_factory=lambda: np.empty(0))
    vs: np.ndarray = field(default_factory=lambda: np.empty(0))
    ts: np.ndarray = field(default_factory=lambda: np.empty(0))

    def summary(self) -> str:
        lt = f"{self.lap_time:.3f}s" if self.lap_time is not None else "DNF"
        status = "CRASH" if self.crashed else ("OK" if self.completed else "timeout")
        return (
            f"[{self.controller:>12}] lap={lt:>8}  vmax={self.max_speed:5.2f}  "
            f"vmean={self.mean_speed:5.2f}  cte(mean/max)={self.mean_cross_track:.3f}/"
            f"{self.max_cross_track:.3f}m  solve={self.solve_ms_mean:5.2f}ms  [{status}]"
        )


def run_episode(
    controller: Controller,
    map_name: str = config.DEFAULT_MAP,
    render: bool = False,
    max_sim_time: float = 120.0,
    start_offset: float = 0.0,
) -> RunResult:
    """Run one episode (until 2 finish-line crossings, crash, or timeout)."""
    raceline = Raceline.load(map_name)
    render_mode = "human" if render else None
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={"num_agents": 1, "map": map_name},
        render_mode=render_mode,
    )

    start = raceline.start_pose().reshape(1, 3)
    obs, info = env.reset(options={"poses": start})
    controller.reset()

    dt = config.SIM_TIMESTEP
    max_steps = int(max_sim_time / dt)

    xs, ys, vs, ts, ctes, solve_ms = [], [], [], [], [], []
    # Lap timing is done on our own arc-length progress (the gym's finish-line
    # toggle is unreliable when the car starts exactly on the line). The car
    # starts at s~0, so the 1st progress wrap ends the standing-start lap and
    # the 2nd wrap ends one clean *flying* lap.
    half_len = raceline.length / 2.0
    prev_s = 0.0
    wrap_times: list[float] = []
    lap_start_t = None
    lap_time = None
    crashed = False
    completed = False

    for step in range(max_steps):
        st = CarState(
            x=float(obs["poses_x"][0]),
            y=float(obs["poses_y"][0]),
            yaw=float(obs["poses_theta"][0]),
            v=float(obs["linear_vels_x"][0]),
            yaw_rate=float(obs["ang_vels_z"][0]),
        )

        t0 = time.perf_counter()
        steering, speed = controller(st)
        solve_ms.append((time.perf_counter() - t0) * 1e3)

        action = np.array([[steering, speed]], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if render:
            env.render()

        # logging
        sim_t = step * dt
        xs.append(st.x)
        ys.append(st.y)
        vs.append(st.v)
        ts.append(sim_t)
        # cross-track error vs nearest raceline point
        idx = raceline.nearest_index(st.x, st.y)
        ctes.append(np.hypot(st.x - raceline.x[idx], st.y - raceline.y[idx]))

        # lap timing via arc-length progress wrap (s jumps from ~length to ~0)
        s = raceline.s[idx]
        if s < prev_s - half_len:
            wrap_times.append(sim_t)
            if len(wrap_times) == 1:
                lap_start_t = sim_t                  # end of standing-start lap
            elif len(wrap_times) == 2:
                lap_time = sim_t - lap_start_t       # one clean flying lap
                completed = True
        prev_s = s

        if obs["collisions"][0]:
            crashed = True
            break
        if completed:
            break

    env.close()

    vs_arr = np.asarray(vs)
    ctes_arr = np.asarray(ctes)
    return RunResult(
        controller=controller.name,
        map_name=map_name,
        completed=completed,
        crashed=crashed,
        lap_time=lap_time,
        sim_time=ts[-1] if ts else 0.0,
        max_speed=float(vs_arr.max()) if vs else 0.0,
        mean_speed=float(vs_arr.mean()) if vs else 0.0,
        mean_cross_track=float(ctes_arr.mean()) if ctes else 0.0,
        max_cross_track=float(ctes_arr.max()) if ctes else 0.0,
        solve_ms_mean=float(np.mean(solve_ms)) if solve_ms else 0.0,
        xs=np.asarray(xs),
        ys=np.asarray(ys),
        vs=vs_arr,
        ts=np.asarray(ts),
    )
