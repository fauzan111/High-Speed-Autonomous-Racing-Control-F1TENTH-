"""Compare fixed and adaptive MPC. Saves summaries and measured-motion CSV logs.

python -m scripts.grip_experiment --surface wet --out results/grip_wet
Requires the simulator and map assets; setup is documented in README.md.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess

from racer import config
from racer.controllers.mpc import MPCController
from racer.estimation.grip import GripSchedule, dry_to_wet
from racer.estimation.online import GripEstimator
from racer.raceline import Raceline
from racer.runner import run_episode


def revision(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default=config.DEFAULT_MAP)
    parser.add_argument("--surface", choices=["dry", "wet"], default="wet")
    parser.add_argument("--wet-mu", type=float, default=0.6)
    parser.add_argument("--max-time", type=float, default=200.0)
    parser.add_argument("--require-completion", nargs="*", choices=["fixed_dry", "fixed_wet", "adaptive"], default=[])
    parser.add_argument("--excitation-g", type=float, default=0.05,
                        help="simulator linear-parameter excitation gate; not real peak-grip observability")
    parser.add_argument("--speed-cap", type=float, default=0.7)
    parser.add_argument("--slack-weight", type=float, default=1000.0)
    parser.add_argument("--out", type=Path, default=Path("results/grip_experiment"))
    args = parser.parse_args()
    rl = Raceline.load(args.map)
    dry_mu = config.VehicleParams().mu
    if not 0.1 <= args.wet_mu <= dry_mu:
        parser.error("wet-mu must be between 0.1 and nominal dry friction")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty; choose a new directory to preserve prior runs")
    args.out.mkdir(parents=True, exist_ok=True)
    schedule = (dry_to_wet(rl.length, mu_wet=args.wet_mu) if args.surface == "wet"
                else GripSchedule(((0.0, dry_mu),), rl.length))
    manifest = dict(python=platform.python_version(), platform=platform.platform(),
                    project_revision=revision(config.REPO_ROOT),
                    simulator_revision=revision(config.GYM_ROOT),
                    configuration={**vars(args), "out": str(args.out)},
                    observation="noise-free simulator motion; measured state steering",
                    interpretation="single deterministic demonstration, not a robustness study",
                    dependencies={})
    for name in ["numpy", "casadi", "gymnasium", "f1tenth_gym"]:
        try:
            manifest["dependencies"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            manifest["dependencies"][name] = "unknown"
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summaries = []
    for name in ["fixed_dry", "fixed_wet", "adaptive"]:
        controller = MPCController(rl, speed_cap=args.speed_cap, w_slack=args.slack_weight)
        # Apply the same profile adaptation algorithm to every comparison arm.
        controller.set_grip(args.wet_mu if name == "fixed_wet" else dry_mu, sigma=0.0)
        estimator = GripEstimator(excitation_g=args.excitation_g) if name == "adaptive" else None
        result = run_episode(controller, map_name=args.map, max_sim_time=args.max_time,
                             grip_schedule=schedule, grip_estimator=estimator)
        print(name, result.summary(), flush=True)
        summary = {k: v for k, v in asdict(result).items()
                   if k not in {"xs", "ys", "vs", "ts", "grip_log"}}
        summary.update(variant=name, solve_calls=controller.solve_calls,
                       solve_fails=controller.solve_fails)
        if name == "adaptive" and result.grip_log:
            errors = [r["mu_hat"] - r["true_mu"] for r in result.grip_log]
            summary["mu_rmse"] = math.sqrt(sum(e * e for e in errors) / len(errors))
            summary["estimator_diagnostics"] = dict(
                final_mu_hat=result.grip_log[-1]["mu_hat"],
                final_sigma=result.grip_log[-1]["sigma"],
                final_mu_used=result.grip_log[-1]["mu_used"],
                min_mu_hat=min(r["mu_hat"] for r in result.grip_log),
                max_sigma=max(r["sigma"] for r in result.grip_log),
                excited_steps=sum(bool(r["excited"]) for r in result.grip_log),
                final_progress=result.grip_log[-1]["progress"],
                final_lap=result.grip_log[-1]["lap"],
            )
            for label, subset in [
                ("wet", [r for r in result.grip_log if r["true_mu"] < dry_mu]),
                ("first_lap_wet", [r for r in result.grip_log if r["true_mu"] < dry_mu and r["lap"] == 0]),
            ]:
                summary[f"{label}_samples"] = len(subset)
                summary[f"{label}_mu_rmse"] = (math.sqrt(sum(
                    (r["mu_hat"] - r["true_mu"]) ** 2 for r in subset) / len(subset))
                    if subset else None)
        summary["control_ms_per_physics_step"] = summary.pop("solve_ms_mean")
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        with (args.out / f"{name}.csv").open("w", newline="") as target:
            if result.grip_log:
                writer = csv.DictWriter(target, fieldnames=list(result.grip_log[0]))
                writer.writeheader()
                writer.writerows(result.grip_log)
        (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))

    from scripts.plot_grip import plot_experiment
    print(f"Comparison plot: {plot_experiment(args.out)}")
    failed = [r["variant"] for r in summaries
              if r["variant"] in args.require_completion and (not r["completed"] or r["crashed"])]
    if failed:
        raise SystemExit("Required controllers did not finish: " + ", ".join(failed))


if __name__ == "__main__":
    main()
