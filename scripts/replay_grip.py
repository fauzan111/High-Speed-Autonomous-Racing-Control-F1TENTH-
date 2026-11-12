"""Replay measured-motion CSV through the grip estimator, without the simulator.

Required columns: time [s], speed [m/s], steering [rad, actual measured angle],
yaw_rate [rad/s]. Optional: acceleration [m/s²], true_mu (evaluation only).
Run: python -m scripts.replay_grip measurements.csv --out results/grip_replay.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from racer.estimation.online import GripEstimator


def replay(input_path: Path, output_path: Path) -> dict:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    with input_path.open(newline="") as source:
        reader = csv.DictReader(source)
        required = {"time", "speed", "steering", "yaw_rate"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"required columns: {', '.join(sorted(required))}")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("need at least two measured samples")
    # Validate the complete recording before writing output.
    samples = []
    previous = None
    for row in rows:
        values = {key: float(row[key]) for key in required}
        if row.get("acceleration"):
            values["acceleration"] = float(row["acceleration"])
        if row.get("true_mu"):
            values["true_mu"] = float(row["true_mu"])
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("recording contains nonfinite values")
        if previous is not None and values["time"] <= previous:
            raise ValueError("timestamps must be strictly increasing")
        previous = values["time"]
        samples.append(values)
    estimator = GripEstimator()
    first = samples[0]
    estimator.update(speed=first["speed"], steering=first["steering"],
                     yaw_rate=first["yaw_rate"], dt=samples[1]["time"] - first["time"],
                     acceleration=first.get("acceleration", 0.0))
    output = []
    errors = []
    for previous, sample in zip(samples, samples[1:]):
        dt = sample["time"] - previous["time"]
        acceleration = sample.get("acceleration", (sample["speed"] - previous["speed"]) / dt)
        estimate = estimator.update(
            speed=(previous["speed"] + sample["speed"]) / 2, steering=(previous["steering"] + sample["steering"]) / 2,
            yaw_rate=sample["yaw_rate"], dt=dt, acceleration=acceleration,
        )
        result = dict(sample, acceleration=acceleration, mu_hat=estimate.mu_hat,
                      sigma=estimate.sigma, mu_used=estimate.mu_used,
                      excited=estimate.excited)
        output.append(result)
        if "true_mu" in sample:
            errors.append(estimate.mu_hat - sample["true_mu"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in output)))
    with output_path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    return {"samples": len(output), "evaluated_samples": len(errors),
            "mu_rmse": math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None,
            "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/grip_replay.csv"))
    args = parser.parse_args()
    print(json.dumps(replay(args.input, args.out), indent=2))


if __name__ == "__main__":
    main()
