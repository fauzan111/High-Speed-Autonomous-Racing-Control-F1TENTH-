"""
Benchmark: run each controller on a map, print a comparison table, and save
trajectory + speed plots. Also reports the theoretical raceline lap time
(perfect tracking) as the ceiling.

Usage:
    python -m scripts.benchmark [--map Spielberg]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path to allow running as a script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from racer import config
from racer.controllers.mpc import MPCController
from racer.controllers.pure_pursuit import PurePursuit
from racer.raceline import Raceline
from racer.runner import run_episode


def ideal_lap_time(rl: Raceline) -> float:
    ds = np.diff(np.append(rl.s, rl.length))
    return float(np.sum(ds / rl.vx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--speed-cap", type=float, default=0.7)
    ap.add_argument("--require-completion", action="store_true")
    ap.add_argument("--max-time", type=float, default=200.0)
    args = ap.parse_args()

    rl = Raceline.load(args.map)
    ideal = ideal_lap_time(rl)

    controllers = [
        PurePursuit(rl),
        MPCController(rl, speed_cap=args.speed_cap),
    ]

    results = []
    for c in controllers:
        print(f"running {c.name} ...")
        results.append(run_episode(c, map_name=args.map, max_sim_time=args.max_time))

    print("\n" + "=" * 78)
    print(f"BENCHMARK — {args.map}")
    print("=" * 78)
    print(f"{'controller':>14} {'lap [s]':>9} {'vmax':>6} {'vmean':>6} "
          f"{'cte_mean':>9} {'cte_max':>8} {'solve_ms':>9}  status")
    print("-" * 78)
    print(f"{'ideal (raceline)':>14} {ideal:>9.2f} {rl.vx.max():>6.2f} "
          f"{rl.vx.mean():>6.2f} {'—':>9} {'—':>8} {'—':>9}  perfect tracking")
    for r in results:
        lap = f"{r.lap_time:.2f}" if r.lap_time is not None else "DNF"
        status = "CRASH" if r.crashed else ("OK" if r.completed else "timeout")
        print(f"{r.controller:>14} {lap:>9} {r.max_speed:>6.2f} {r.mean_speed:>6.2f} "
              f"{r.mean_cross_track:>9.3f} {r.max_cross_track:>8.3f} "
              f"{r.solve_ms_mean:>9.2f}  {status}")

    (config.RESULTS_DIR / "baseline_summary.json").write_text(json.dumps([
        dict(controller=r.controller, completed=r.completed, crashed=r.crashed,
             lap_time=r.lap_time, max_cross_track=r.max_cross_track,
             mean_cross_track=r.mean_cross_track,
             control_ms_per_physics_step=r.solve_ms_mean) for r in results
    ], indent=2))

    # gap vs baseline
    laps = {r.controller: r.lap_time for r in results if r.lap_time}
    if "pure_pursuit" in laps and "mpc" in laps:
        gain = laps["pure_pursuit"] - laps["mpc"]
        pct = 100 * gain / laps["pure_pursuit"]
        print("-" * 78)
        print(f"MPC is {gain:.2f}s ({pct:.1f}%) faster than pure pursuit; "
              f"{laps['mpc'] - ideal:.2f}s off the ideal raceline.")
    print("=" * 78)

    # plots
    from racer import plotting
    completed = [r for r in results]
    plotting.plot_trajectories(
        completed, rl, map_name=args.map,
        out_path=str(config.RESULTS_DIR / f"{args.map}_trajectories.png"),
        title=f"{args.map} — controller trajectories (colored by speed)")
    plotting.plot_speed(
        completed,
        out_path=str(config.RESULTS_DIR / f"{args.map}_speed.png"),
        title=f"{args.map} — speed profiles")
    print(f"plots saved to {config.RESULTS_DIR}")
    if args.require_completion and any(not r.completed or r.crashed for r in results):
        raise SystemExit("Baseline controller failed to complete")


if __name__ == "__main__":
    main()
