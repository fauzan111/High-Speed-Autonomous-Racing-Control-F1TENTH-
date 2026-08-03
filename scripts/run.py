"""
Run a single controller for one episode and print the result.

Usage:
    python -m scripts.run --controller pure_pursuit [--render] [--map Spielberg]
    python -m scripts.run --controller mpc --render
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow running as a script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from racer import config
from racer.raceline import Raceline
from racer.runner import run_episode


def build_controller(name: str, raceline: Raceline):
    if name == "pure_pursuit":
        from racer.controllers.pure_pursuit import PurePursuit
        return PurePursuit(raceline)
    if name == "mpc":
        from racer.controllers.mpc import MPCController
        return MPCController(raceline)
    raise SystemExit(f"unknown controller: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="pure_pursuit",
                    choices=["pure_pursuit", "mpc"])
    ap.add_argument("--map", default=config.DEFAULT_MAP)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--max-time", type=float, default=120.0)
    args = ap.parse_args()

    raceline = Raceline.load(args.map)
    controller = build_controller(args.controller, raceline)
    result = run_episode(controller, map_name=args.map,
                         render=args.render, max_sim_time=args.max_time)
    print(result.summary())


if __name__ == "__main__":
    main()
