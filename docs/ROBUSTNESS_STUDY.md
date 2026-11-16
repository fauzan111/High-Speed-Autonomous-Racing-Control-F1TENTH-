# Seeded sensor-noise study

This experiment checks how the fixed-dry, fixed-wet and adaptive controllers
behave when their controller/estimator inputs are noisy. It is separate from the
published noise-free validation. No results from this study should be reported
as measured sensor performance: the noise values below are controlled
simulation perturbations, not a characterization of a physical F1TENTH car.

## Paired experiment setup

For each seed, all three controller variants receive the same seeded sequence
of independent, zero-mean Gaussian samples for position, heading, speed,
yaw-rate and measured steering. The controllers change the trajectory, so the
later observations naturally diverge; the shared seed pairs the noise draws by
time step rather than forcing trajectories to match.

Sensor noise affects the state passed to the controller and the signals passed
to the grip estimator. Simulator truth still drives the friction schedule,
lap timing and scoring. Adaptive CSV logs retain both the noisy estimator input
columns and corresponding true values. Acceleration is derived from successive
noisy speed readings, as it would be from a speed sensor.

The initial pilot uses these standard deviations:

| Measurement | Standard deviation |
| --- | ---: |
| Position, each axis | 0.01 m |
| Heading | 0.005 rad |
| Speed | 0.03 m/s |
| Yaw rate | 0.02 rad/s |
| Measured steering | 0.002 rad |

Three seeds (11, 23 and 47) are run on dry, mild-wet (mu=0.8) and wet (mu=0.6)
conditions. This is an initial sensitivity check, not enough trials for a
general statistical claim. Report every seed and completion/crash outcome,
then summarize lap time, true cross-track error, and adaptive wet-segment grip
RMSE across seeds. Keep the original noise-free results as their own baseline.

## Run locally

After following [environment setup](ENVIRONMENT.md) and preparing the map, run
one paired trial per command. Choose a fresh output directory each time:

```bash
while IFS=' ' read -r surface wet_mu; do
  for seed in 11 23 47; do
    .venv/bin/python -m scripts.grip_experiment \
      --surface "$surface" --wet-mu "$wet_mu" --seed "$seed" \
      --position-noise-m 0.01 --heading-noise-rad 0.005 \
      --speed-noise-mps 0.03 --yaw-rate-noise-rad-s 0.02 \
      --steering-noise-rad 0.002 \
      --out "results/robustness/${surface}-${wet_mu}/seed_${seed}"
  done
done <<'CONDITIONS'
dry 0.6
wet 0.8
wet 0.6
CONDITIONS
```

The `Grip sensor-noise robustness` workflow runs this nine-trial matrix on
GitHub Actions and uploads one artifact per condition and seed. A push to the
study branch starts the pilot automatically; after merge, it can also be started
from the Actions tab on `main`. Artifacts contain each trial's `manifest.json`,
`summary.json`, CSV logs and comparison plot.

## Interpretation

The noise model is independent white Gaussian noise. It does not model sensor
bias, temporal correlation, latency, dropouts, map localization error, or
vehicle-model mismatch. The F1TENTH simulator still uses a linear tyre response
without nonlinear saturation. A successful seeded run therefore demonstrates
repeatability under this declared perturbation only; it does not establish
real-car robustness or a safety advantage.
