# Grip-estimation validation evidence

## Scope

This is an extension of the existing F1TENTH racing repository, not a separate
project. The experiments identify a simulator friction parameter and compare
controllers under a spatial dry/wet/dry schedule. They do not establish real
peak-tyre-friction estimation or calibrated safety guarantees.

## Initial integration and regression diagnosis

[Initial automated run](https://github.com/fauzan111/High-Speed-Autonomous-Racing-Control-F1TENTH-/actions/runs/43191109875): 38 tests passed, including real IPOPT solves. Map bootstrap and simulator execution succeeded. Pure Pursuit reproduced 57.85 s; the MPC with its newly activated friction-slack penalty crashed. All initial comparison arms also crashed before the wet patch. Successful command execution did not mean successful control.

[Dry-only calibration run](https://github.com/fauzan111/High-Speed-Autonomous-Racing-Control-F1TENTH-/actions/runs/43199874043) isolated the issue:

| Speed cap | Slack weight | Profile adaptation | Completed | Lap time |
|---|---:|---|---|---:|
| 0.97 | 0 (diagnostic) | Off | Yes | 47.22 s |
| 0.97 | 0 (diagnostic) | On | Yes | 47.46 s |
| 0.97 | 1, 10, 100, 1000 | Either | No | DNF |
| 0.80 | 1000 | Either | No | DNF |

Weight zero reproduces the historical controller but disables grip enforcement;
it is a diagnostic, not an accepted adaptive-control configuration. The failed
runs remain available in workflow artifacts.

The original kinematic lateral-acceleration constraint treats all steering as
instantaneous yaw. The simulator needs additional steering to compensate linear
tyre slip. This mismatch explains why merely activating that constraint can
suppress necessary corrective steering. The next validation uses an explicit
steady-state single-track yaw/slip approximation, with a positive slack penalty.

## Synthetic estimator check

The committed `results/synthetic_grip/summary.json` is reproducible with
`python -m scripts.validate_grip --out-dir results/synthetic_grip --seed 7`.
It reports steady wet RMSE 0.0175 and wet settling time 1.91 s under a matched
linear model with noisy yaw-rate measurements. These numbers are separate from
simulator lap results. See the summary for thresholds, noise, integrators and limitations.

## Acceptance gates

Full solver tests must execute without dependency skips. The corrected dry
baseline and all three dry comparison arms must complete. Fixed-wet and adaptive
MPC must complete wet/mild-wet trials; fixed-dry is allowed to fail on wet ground.
All arms use the same model and tuning except their friction information.
No adaptive superiority claim is made without completed-run evidence. Runs are
deterministic demonstrations with noise-free simulated motion, not a robustness study.


## Corrected-model dry calibration

[Corrected-model speed sweep](https://github.com/fauzan111/High-Speed-Autonomous-Racing-Control-F1TENTH-/actions/runs/43201373737), with positive slack weight 1000:

| Speed cap | Default profile | Grip-adapted profile |
|---|---|---|
| 0.97 | Crash | Crash |
| 0.90 | Crash | Crash |
| 0.80 | Crash | Crash |
| 0.70 | 65.16 s, complete | 65.16 s, complete |

At 0.70, maximum tracking error was 0.1059 m and there were no solver failures.
The new controller therefore uses a conservative 0.70 speed cap pending further
model/performance improvements. It is slower than the historical free-slack MPC
and the tuned Pure Pursuit baseline. No speed advantage is claimed. Full adaptive
comparisons use this same setting across all friction-information variants.
