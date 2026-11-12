# Grip estimation implementation plan

## Objective
Demonstrate whether online grip estimation improves racing control when friction changes. Compare fixed-dry, fixed-wet, and adaptive MPC under the same conditions. A crash or lack of improvement is a result to investigate, not a reason to alter reported outcomes.

## Milestones and acceptance gates

1. **Reproducible baseline**
   - Install an isolated Python environment and a pinned F1TENTH simulator revision.
   - Run the existing unit tests and repair the MPC objective registration.
   - Re-run dry-track Pure Pursuit and MPC; retain machine-readable results, dependency versions, and simulator revision.
   - Gate: both controllers complete or any regression is explained. Existing README times remain historical until reproduced.
2. **Controlled friction experiment**
   - Verify runtime simulator friction updates against its source and dynamics.
   - Apply a spatial dry/wet/dry schedule, initially wet at 40–70% of the lap.
   - Log timestamps, progress, motion, actual steering, commands, true friction, estimates, uncertainty, and solver failures.
   - Gate: applied changes reach the simulator vehicle model; true friction is used only by simulation and evaluation.
3. **Offline estimator**
   - Use a dynamic single-track model and measured vehicle motion, never true friction observations.
   - Check convergence under excitation, lack of false confidence under gentle driving, finite inputs, and bounded estimates.
   - Gate: report error and delay through transitions, including model mismatch limitations.
4. **Adaptive control**
   - Parameterize MPC grip without rebuilding the solver. Use a bounded uncertainty margin.
   - Adjust reference speeds and braking feasibility as well as lateral constraints.
   - Gate: compare fixed-dry, fixed-wet, and adaptive controllers on identical tracks and initial conditions; report completion and tracking error before speed improvements.
5. **Evidence and presentation**
   - Repeat trials with explicit seeds/noise settings; include dry-track regression and more than one wet severity.
   - Export raw logs, summaries, friction/uncertainty/speed/error plots, environment manifest, and reproduction commands.
   - Update README and video only with verified results.

## Initial work allocation

- Primary agent: plan, runner/experiment integration, combined validation and review.
- Environment agent: local setup, simulator API inspection, reproducibility documentation.
- MPC agent: objective repair, runtime grip parameter, feasible reference speed adaptation, focused tests.
- Estimator agent: physical estimator, uncertainty behavior, friction utility validation, focused tests.

Agents own separate files; shared interfaces are agreed before integration.

## Scope and scientific limits

The first deliverable is a working EKF-to-MPC experiment. A learned estimator, dynamic-model MPC, overtaking, and new racing-line optimization come after this gate. An unseen abrupt wet patch cannot be anticipated by a reactive estimator. Low tire loading provides weak friction information. Commanded steering/speed must not be mistaken for measured steering/acceleration. The simulator's friction parameterization may differ from real tire saturation, so conclusions are simulation-specific.

## Status

Started 2026-11-11; completion pass 2026-11-12. All four agent slots contributed to the initial implementation.

- Completed: MPC objective correction and runtime grip API; physical EKF with synthetic tests; offline CSV replay; spatial-friction runner integration; comparison CLI and result manifests; environment/readiness documentation.
- Validation: 50 tests passed in GitHub Actions, including real IPOPT solves; all dry, wet (mu=0.6) and mild-wet (mu=0.8) completion gates passed. Local NumPy tests also pass (27); CasADi remains absent locally.
- Completed evidence: pinned map bootstrap, historical 47.22 s diagnostic reproduction, corrected conservative baseline, all nine comparison runs, CSV logs, JSON summaries/manifests, and comparison plots. See VALIDATION_RESULTS.md and run 43207169981.
- Environment constraint: local shell DNS still cannot resolve GitHub/PyPI; full simulation validation ran in GitHub Actions instead.
- Remaining research: transient tyre-model MPC, calibrated uncertainty, noise/track robustness studies, a learned estimator, and an adaptive video. These are beyond the completed first EKF-to-MPC deliverable; the original demo remains historical.
- Scope finding: upstream friction scales a linear tyre response without nonlinear saturation; realistic peak-friction validation requires a more appropriate tyre model after this first experiment.


## Completion pass

Added measured-steering runner contract tests, collision/error cleanup, post-step
lap crossing accounting, checkpoint-only termination handling, safe map bootstrap,
CSV-to-EKF replay equivalence, a standalone synthetic validation command, experiment
plots, and a GitHub Actions workflow for full solver/simulator evidence. The
synthetic seed-7 report is committed; raw CSVs are regenerated by its command.
MPC failed-solve fallback now brakes even below 1 m/s.

The first EKF-to-MPC deliverable is implemented and validated in the declared
simulator scenarios. Adaptive MPC improves on the fixed-wet policy in both wet
trials, while fixed-dry remains faster and completes both. This is not evidence
of a general safety advantage or realistic peak-friction estimation.
