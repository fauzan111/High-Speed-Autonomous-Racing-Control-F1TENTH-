"""Deterministic matched-model grip EKF check; no simulator or plotting required.

Run: python -m scripts.validate_grip --out-dir results/synthetic_grip --seed 7

Writes replay-compatible samples.csv and summary.json. The plant is the same
linear single-track model as the estimator, integrated separately with small
Euler steps. This checks numerical behavior, noise handling, and parameter
tracking; it does NOT validate F1TENTH integration, real tyres, or lap safety.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from racer.config import VehicleParams
from racer.estimation.online import GripEstimator, single_track_rates


LIMITATION = (
    "Matched-model synthetic validation only; not F1TENTH integration or real-world "
    "peak-friction validation. Friction scales linear cornering stiffness in this "
    "model, with no nonlinear tyre saturation. Speed and steering are noise-free; "
    "only yaw-rate measurements contain noise. Reported EKF uncertainty is not a "
    "calibrated safety guarantee."
)


def _steering(time: float) -> float:
    # Start straight to verify that absent excitation does not create certainty.
    ramp = min(max(time - 2.0, 0.0), 1.0)
    return ramp * (0.07 + 0.025 * math.sin(2.0 * time))


def _true_mu(time: float) -> float:
    return 0.6 if 8.0 <= time < 16.0 else 1.0489


def _segment_metrics(rows: list[dict], start: float, end: float) -> dict:
    selected = [row for row in rows if start <= row["time"] < end]
    errors = np.array([row["estimate_error"] for row in selected])
    return {
        "start_s": start,
        "end_s": end,
        "samples": len(selected),
        "mu_rmse": float(np.sqrt(np.mean(errors**2))),
        "mu_bias": float(np.mean(errors)),
        "mean_sigma": float(np.mean([row["sigma"] for row in selected])),
    }


def _settling_time(rows: list[dict], start: float, end: float) -> float | None:
    """First time absolute error remains <=0.08 for at least 0.5 s."""
    candidate = None
    for row in rows:
        time = row["time"]
        if not start <= time < end:
            continue
        if abs(row["estimate_error"]) <= 0.08:
            if candidate is None:
                candidate = time
            if time - candidate >= 0.5 - 1e-9:
                return round(candidate - start, 6)
        else:
            candidate = None
    return None


def validate(out_dir: Path, seed: int = 7) -> dict:
    """Run a fixed 24 s dry/wet/dry experiment; return its JSON-ready summary."""
    rng = np.random.default_rng(seed)
    params = VehicleParams()
    estimator = GripEstimator(yaw_noise=0.02)
    dt = 0.01
    speed = 6.0
    state = np.array([0.0, 0.0, _true_mu(0.0)])
    rows = []
    finite_and_psd = True
    for index in range(2401):
        time = round(index * dt, 8)
        if index:
            # Independent integrator: 20 forward Euler steps rather than the
            # estimator's RK4 predictor. The underlying plant model is shared.
            for substep in range(20):
                at = time - dt + substep * dt / 20
                state[2] = _true_mu(at)
                state += dt / 20 * single_track_rates(
                    state, speed, _steering(at), 0.0, params)
        truth = _true_mu(time)
        measurement = float(state[0] + rng.normal(0.0, 0.02))
        measured_steering = _steering(time)
        interval_steering = (measured_steering + _steering(max(time - dt, 0.0))) / 2
        estimate = estimator.update(
            speed=speed, steering=interval_steering, yaw_rate=measurement,
            dt=dt, acceleration=0.0)
        covariance = estimator.filter.P
        finite_and_psd = bool(
            finite_and_psd and np.all(np.isfinite(estimator.filter.x))
            and np.all(np.isfinite(covariance))
            and np.linalg.eigvalsh(covariance).min() >= -1e-10)
        rows.append({
            "time": time, "speed": speed, "steering": measured_steering,
            "yaw_rate": measurement, "acceleration": 0.0,
            "true_mu": truth, "yaw_rate_true": float(state[0]),
            "mu_hat": estimate.mu_hat, "sigma": estimate.sigma,
            "mu_used": estimate.mu_used, "excited": estimate.excited,
            "estimate_error": estimate.mu_hat - truth,
        })
    segments = {
        "dry_before": _segment_metrics(rows, 4.0, 8.0),
        "wet": _segment_metrics(rows, 10.0, 16.0),
        "dry_after": _segment_metrics(rows, 18.0, 24.0),
    }
    straight = [row for row in rows if row["time"] <= 2.0]
    checks = {
        "finite_state_and_positive_semidefinite_covariance": finite_and_psd,
        "no_false_updates_on_straight": all(not row["excited"] for row in straight),
        "uncertainty_preserved_on_straight": all(
            row["sigma"] >= estimator.initial_sigma for row in straight),
        "steady_segment_rmse_below_0_08": all(
            segment["mu_rmse"] < 0.08 for segment in segments.values()),
        "final_mu_error_below_0_08": abs(rows[-1]["estimate_error"]) < 0.08,
    }
    errors = np.array([row["estimate_error"] for row in rows])
    summary = {
        "validation_kind": "matched_model_synthetic",
        "limitation": LIMITATION,
        "seed": seed, "duration_s": 24.0, "dt_s": dt, "samples": len(rows),
        "yaw_noise_std_rad_s": 0.02,
        "plant_integrator": "forward Euler with 0.0005 s substeps",
        "estimator_integrator": "RK4 with <=0.005 s substeps",
        "schedule": [{"start_s": 0.0, "mu": 1.0489},
                     {"start_s": 8.0, "mu": 0.6},
                     {"start_s": 16.0, "mu": 1.0489}],
        "overall_mu_rmse": float(np.sqrt(np.mean(errors**2))),
        "steady_segments": segments,
        "wet_settling_time_s": _settling_time(rows, 8.0, 16.0),
        "dry_recovery_settling_time_s": _settling_time(rows, 16.0, 24.0),
        "settling_definition": "first |mu error| <=0.08 sustained for 0.5 s",
        "final_mu_hat": rows[-1]["mu_hat"],
        "final_sigma": rows[-1]["sigma"],
        "checks": checks, "passed": all(checks.values()),
        "sample_timing": (
            "CSV steering is measured at each timestamp; interval mean steering "
            "drives the EKF. Yaw rate is measured at the interval end. true_mu is "
            "ground truth at the timestamp and is used for evaluation only."
        ),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "samples.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/synthetic_grip"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    summary = validate(args.out_dir, args.seed)
    print(json.dumps(summary, indent=2, allow_nan=False))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
