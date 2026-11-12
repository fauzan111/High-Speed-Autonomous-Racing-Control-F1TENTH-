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

## Reproduce the historical controller (diagnostic)

After preparing the map, this explicitly disables the formerly unregistered
slack penalty and uses the original kinematic model. It reproduces the historical
benchmark, but does not validate grip enforcement:

```bash
python - <<'PY'
from racer.controllers.mpc import MPCController
from racer.raceline import Raceline
from racer.runner import run_episode
rl = Raceline.load()
controller = MPCController(rl, speed_cap=0.97, w_slack=0,
                           understeer_compensation=False)
print(run_episode(controller, max_sim_time=200).summary())
PY
```


## Final wet comparisons

[Validation run](https://github.com/fauzan111/High-Speed-Autonomous-Racing-Control-F1TENTH-/actions/runs/43207169981), source commit `3e3d02c9acd2c56d67623bae489f34d370dd4a4d`:

| Surface | Fixed dry | Fixed wet | Adaptive | Adaptive wet-portion mu RMSE |
|---|---:|---:|---:|---:|
| Dry | 65.16 s | 86.58 s | 69.59 s | Not applicable |
| Wet patch, mu=0.8 | 65.21 s | 74.76 s | 73.05 s | 0.0652 |
| Wet patch, mu=0.6 | 65.22 s | 86.61 s | 77.66 s | 0.1131 |

All listed controllers completed without crashes. In the stronger-wet case,
adaptive maximum tracking error was 0.1027 m, first-lap wet-portion mu RMSE was
0.1141, overall mu RMSE was 0.0768, and there were no solver failures. The minimum
estimated mu was 0.6011. Wet-portion errors include transitions; they are not
steady-state errors. Full diagnostics and traces are in each scenario artifact.

Adaptive MPC was 8.95 s (10.3%) faster than fixed-wet on mu=0.6, and 1.71 s (2.3%)
faster on mu=0.8. Fixed-dry completed both scenarios faster than adaptive. These
results support improvement relative to a uniformly conservative speed policy;
they do not demonstrate that adaptation was necessary to avoid a crash, or that
adaptive MPC outperforms an optimistic controller in this linear-tyre simulator.

### Excitation-policy ablation

With the globally scaled profile and the class default 0.15g excitation gate,
the strong-wet adaptive run stayed on track but timed out at 200 s, averaging
2.24 m/s. The experiment harness now exposes `--excitation-g` and defaults to
0.05g for the simulator's linear-parameter identification task. The prior run
is retained at [run 43205731364](https://github.com/fauzan111/High-Speed-Autonomous-Racing-Control-F1TENTH-/actions/runs/43205731364).

This lower threshold is an explicit simulator-specific tuning choice, not proof
of real peak-grip observability or calibrated confidence. The class default and
seed-7 synthetic benchmark remain at 0.15g. Gentle noisy driving can still yield
weak information. The controller retains a two-sigma heuristic margin and a
0.70 speed cap; all comparison arms keep nominal model dynamics fixed.

For simulator replay, match the recorded harness configuration:

```bash
python -m scripts.replay_grip results/grip_wet/adaptive.csv --excitation-g 0.05 --out results/grip_replay.csv
```

Replay matches mu_hat and sigma; its `mu_used` is the estimator helper's default
one-sigma diagnostic. Online control applies the separate two-sigma policy,
clipped to nominal dry friction. This diagnostic should not be substituted for
the logged controller budget when comparing commands.

### Evidence delivery

Download `simulator-evidence-wet`, `simulator-evidence-wet_mild`, and
`simulator-evidence-dry` from the validation run. Each experiment directory
contains `manifest.json`, `summary.json`, three measured-motion CSVs and
`comparison.png`. The dry artifact additionally contains the corrected baseline
summary and plots. `synthetic-grip-evidence` contains the separate matched-model
check. Original README images/video describe historical performance.


## Final status

All 50 tests and every completion gate passed in run 43207169981. The dry
baseline reproduced Pure Pursuit at 57.85 s and the new constrained MPC at
65.16 s. Dry adaptive mu RMSE was 0.01035. All nine comparison runs completed;
there were no adaptive solver failures in any reported final trial.

PR #1 was merged into main on 2026-11-12. This document records validation of
source commit 3e3d02c9acd2c56d67623bae489f34d370dd4a4d, included in merge commit
3e7db4dd59054f998b7387aa83f04f4cd8389488. Documentation changes do not change the
validated controller implementation.
