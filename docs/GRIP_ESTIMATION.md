# Online grip estimation for the MPC

Branch: `feature/grip-estimation`. Upgrade of the existing racer stack.

## The problem

The MPC plans with a fixed friction coefficient (`config.VehicleParams.mu = 1.0489`). Real
grip changes: a wet patch, cold tyres, oil, worn tyres. With a fixed `mu`:

- **mu over-estimated** (track got wet): the plan asks for more grip than exists and the car slides off.
- **mu under-estimated** (tuned conservatively): the car is safe but slow everywhere.

## The goal

Estimate grip online from signals the car already has (speed, yaw rate, steering, commanded
acceleration) and feed it to the MPC, which then adapts its speed profile.

**Headline experiment:** a lap where the track turns wet between 40% and 70% of the lap.

| Controller | Expected outcome |
|---|---|
| MPC, fixed dry mu | slides off (DNF) or large tracking error in the wet patch |
| MPC, fixed wet mu | finishes, but slow everywhere |
| **MPC + online grip estimate** | finishes, slows only in the wet patch, close to the best possible lap |

## Method

1. **Friction schedule in the sim** (`racer/estimation/grip.py: GripSchedule`): make the
   gym's friction vary with arc length. *Week 1 investigation:* how f1tenth_gym v1.0.0 exposes
   vehicle parameters at runtime (update params per step versus a small patch of the dynamics).
2. **Baseline estimator: EKF** (`racer/estimation/ekf.py`): augment the state with mu and
   compare measured yaw rate and lateral acceleration with the single-track model's prediction.
3. **Learned estimator:** a small physics-informed network (single-track model residual)
   versus the EKF: accuracy, convergence time, compute.
4. **Closing the loop:** the MPC reads the estimated mu each solve, with a safety margin
   (`mu_used = mu_hat - k * sigma`).

## Honest limitation to address

Grip is only **observable near the limit**: driving gently only tells you `mu >= |a|/g`
(`mu_lower_bound`). The estimator must keep its uncertainty high until the tyres are loaded,
and the controller should stay conservative while uncertainty is high. That is the
interesting engineering problem, and a good interview topic.

## Roadmap (about 2 weeks)

### Week 1
- [ ] Set up f1tenth_gym v1.0.0 and reproduce the README benchmark (MPC 47.22 s, pure pursuit 57.85 s)
- [ ] Friction varying along the lap in the sim; reproduce the DNF of the fixed-dry MPC on the wet patch
- [ ] EKF grip estimator running offline on logged laps; estimation error vs ground truth

### Week 2
- [ ] Estimator in the loop: MPC reads mu_hat with an uncertainty margin
- [ ] Learned estimator vs EKF comparison
- [ ] Results table (3 controllers x dry / wet-patch track), plots, README section, video

## Relevant employers

Pirelli (Cyber Tyre), Brembo (Sensify), Ferrari, Dallara, Marelli, PoliMOVE.

## Implementation status (2026-11-11)

Implementation, synthetic evidence and full simulator validation are available.
[Run 43207169981](https://github.com/fauzan111/High-Speed-Autonomous-Racing-Control-F1TENTH-/actions/runs/43207169981)
passed 50 tests and all dry/wet completion gates. See [the results](VALIDATION_RESULTS.md).

- `racer/estimation/online.py`: dynamic single-track augmented EKF using measured steering and yaw response, with speed/acceleration inputs, excitation gating and uncertainty.
- `MPCController.set_grip`: runtime friction parameter, conservative margin, curvature-based speed cap, and cyclic braking/acceleration propagation. This retains separate lateral/longitudinal limits; it is not a full combined-force constraint.
- `scripts/replay_grip.py`: offline measured-motion CSV replay; optional `true_mu` is used only for scoring.
- `scripts/grip_experiment.py`: dry/wet comparison of fixed-dry, fixed-wet and adaptive MPC, with CSV logs, JSON summaries and environment manifest.

After completing [environment setup](ENVIRONMENT.md):

```bash
.venv/bin/python -m scripts.prepare_map --map Spielberg
.venv/bin/python -m scripts.check_environment
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.benchmark
.venv/bin/python -m scripts.grip_experiment --surface dry --out results/grip_dry_01
.venv/bin/python -m scripts.grip_experiment --surface wet --out results/grip_wet_01
.venv/bin/python -m scripts.replay_grip measurements.csv --out results/replay.csv
```

**Model limitation:** source inspection of F1TENTH Gym v1.0.0 shows that its dynamic single-track model scales linear cornering response with `mu`; it does not implement nonlinear tyre saturation. Consequently, this EKF identifies a simulator model parameter. Its matched-model synthetic convergence tests do not establish real-world peak-friction observability. Low-excitation gating is a deliberate conservative heuristic. Establishing a realistic friction-limit experiment may require a nonlinear tyre model; first measure what the existing model actually demonstrates. The expected DNF/adaptive advantage in the original proposal remains a hypothesis.

The experiment uses noise-free simulator motion and actual steering from the simulator vehicle state (the default observation omits it). True friction only enters simulator parameter updates and logs. Every comparison arm uses the same speed-profile adaptation logic; fixed-wet has prior knowledge of the chosen minimum friction and is a conservative reference. An adaptive estimate assumes current friction throughout its preview and receives no future wet-patch locations. Report first-lap and flying-lap behavior separately: repeated laps also test estimator memory.

See [the implementation plan](IMPLEMENTATION_PLAN.md) for milestone acceptance gates. The historical 47.22 s MPC lap was reproduced with the original kinematic model
and disabled slack penalty as a diagnostic. The new constrained configuration
uses the lower 0.70 speed cap; its results are reported separately.


The automated `Grip estimation validation` workflow installs the pinned simulator,
runs actual IPOPT regression tests, and saves dry/wet/mild-wet comparisons. Download
its `synthetic-grip-evidence` and `simulator-evidence` artifacts to inspect results.
The checked-in synthetic report is not a substitute for these integration runs.
`control_ms_per_physics_step` includes estimator and controller overhead, amortized
across held commands; it is not a mean NLP solve time.


### Controller model correction

The MPC's default model now accounts for the zero-longitudinal-acceleration
steady-state response of the simulator's linear single-track tyres:

```
L_eff = wheelbase + v² / (mu*g) * (1/C_Sf - 1/C_Sr)
curvature = steering / L_eff
r = v * curvature
beta = (lr - v²/(mu*g*C_Sr)) * curvature
xdot = v*cos(yaw + beta), ydot = v*sin(yaw + beta)
```

Below 0.5 m/s it uses the original kinematic branch. The reference heading is
converted from path tangent to body yaw using the same slip approximation. The
lateral soft constraint uses `v*r`. By default, the nominal `model_mu` in the
prediction equations stays fixed across controllers; conservative `mu_used`
updates the acceleration budget and speed caps. Experimental `adapt_dynamics=True`
uses the un-margined `mu_hat` for the physical model. An uncertainty margin must
never be interpreted as a point estimate of tyre cornering stiffness.
This removes the original assumption that all measured steering immediately
creates kinematic yaw, which conflicts with the simulator's tyre slip. Transient
slip, acceleration load transfer and combined-force saturation are still omitted.
`understeer_compensation=False` exists for historical diagnostics only.


### Reference speed policy

The same policy is used in every comparison arm: start with the dry racing-line
speeds multiplied by the selected speed cap, scale the entire profile by
`sqrt(mu_used / nominal_mu)`, cap curved sections using the lateral budget, then
propagate braking/acceleration limits around the loop. Scaling even motor-limited
straights is intentionally conservative. This is not a minimum-time solution;
any adaptive advantage over fixed-wet partly reflects avoiding that uniform
conservative slowdown. Current nominal dynamics remain fixed across all arms.


### Final experiment settings

The simulator harness defaults to `--excitation-g 0.05`, recorded in its manifest.
The estimator class and synthetic benchmark retain 0.15g. This threshold is
specific to linear-model parameter identification; weak noisy motion may still
produce poor estimates and misleadingly small covariance. Runtime diagnostics
report point estimates, uncertainty, excitation counts and wet-segment errors.
See VALIDATION_RESULTS.md for the ablation and matching replay command.
