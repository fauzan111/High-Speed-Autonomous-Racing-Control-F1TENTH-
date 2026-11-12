# Environment and simulator integration

## Local setup

Use Python 3.12 (the system Python on this machine is 3.9). From the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
git clone -b v1.0.0 --depth 1 https://github.com/f1tenth/f1tenth_gym.git
git -C f1tenth_gym checkout 5a301bd0ae1ceaf7dec653e7549c8d099db58a6b
python -m pip install -e ./f1tenth_gym
python -m scripts.prepare_map --map Spielberg
python -m scripts.check_environment
python -m pytest -q
```

Prepare the map before running any controller: the project loads its racing line
before environment creation. `scripts.prepare_map` calls the simulator's
source-verified `find_track_dir` download helper, then copies `Spielberg.yaml` to
`Spielberg_map.yaml` only if the expected file is absent. It checks that the racing
line exists at the location expected by this project. Existing map configuration
is preserved. This avoids creating an environment just to trigger its download.

```bash
python -m scripts.prepare_map --map Spielberg
python -m scripts.check_environment
```

Map download and simulator initialization still require runtime verification in a
network-enabled shell. The upstream helper downloads from `api.f1tenth.org` and
extracts into the simulator checkout's `maps` directory.

## Status on 2026-11-12

- `.venv` uses Python 3.12.15 with NumPy 2.2.6 and pytest 9.1.1 plus pytest
  dependencies restored from the local wheel cache.
- Retried installation using `UV_CACHE_DIR=/tmp/f1tenth-uv uv pip install --python
  .venv/bin/python -r requirements.txt -r requirements-dev.txt`; PyPI DNS lookup
  failed. `git ls-remote` also could not resolve GitHub. No escalation was requested.
- CasADi and the simulator are not cached. Full solver checks, map downloads, and
  laps remain blocked by network access. This environment supports NumPy unit
  tests, not reproduction of the racing benchmark.
- The map preparation helper's local YAML compatibility and missing-asset checks
  passed two unit tests. Its real network/download path remains unvalidated.
- No new lap-time or adaptive-control performance claims have been validated.

## Verified simulator API

Inspected the `v1.0.0` branch at commit
`5a301bd0ae1ceaf7dec653e7549c8d099db58a6b` through the GitHub connector.
These interfaces are source-verified; runtime integration needs a simulator run.

To change grip on agent zero before a physics step:

```python
base = env.unwrapped
params = dict(base.sim.agents[0].params)
params["mu"] = mu_true
base.update_params(params, index=0)
```

Pass the complete dictionary: `RaceCar.update_params` replaces its parameter
dictionary instead of merging keys. Keep simulator ground truth separate from
the controller's friction estimate.

The dynamic single-track state is `[x, y, delta, v, psi, yaw_rate, beta]`.
Default original observations expose speed as `linear_vels_x[0]` and yaw rate
as `ang_vels_z[0]`; **`linear_vels_y` is hardcoded to zero**. They omit actual
steering and slip. Read `base.sim.agents[0].state[2]` for steering telemetry, or
configure a `features` observation with `delta`. Feature observations change the
format to an `agent_0` dictionary, so they require a runner adaptation.
Slip at state index 6 may be logged as diagnostic ground truth, but should not
silently become a measured estimator input. Steering command is distinct from
actual steering because the simulator models actuator rate limits.

The simulator's single-track model multiplies linear lateral tyre terms by mu;
it does not implement a nonlinear saturation friction circle. Identification
performance in this model is not evidence that real peak tyre friction can be
identified from gentle driving. Below 0.5 m/s it switches to kinematic dynamics.

Sources:
- [Environment parameter update](https://github.com/f1tenth/f1tenth_gym/blob/5a301bd0ae1ceaf7dec653e7549c8d099db58a6b/f1tenth_gym/envs/f110_env.py)
- [Vehicle state and parameter replacement](https://github.com/f1tenth/f1tenth_gym/blob/5a301bd0ae1ceaf7dec653e7549c8d099db58a6b/f1tenth_gym/envs/base_classes.py)
- [Observation implementations](https://github.com/f1tenth/f1tenth_gym/blob/5a301bd0ae1ceaf7dec653e7549c8d099db58a6b/f1tenth_gym/envs/observation.py)
- [Single-track dynamics](https://github.com/f1tenth/f1tenth_gym/blob/5a301bd0ae1ceaf7dec653e7549c8d099db58a6b/f1tenth_gym/envs/dynamic_models.py)

- [Map discovery and download helper](https://github.com/f1tenth/f1tenth_gym/blob/5a301bd0ae1ceaf7dec653e7549c8d099db58a6b/f1tenth_gym/envs/track/utils.py)


## Cloud validation completed

[Run 43207169981](https://github.com/fauzan111/High-Speed-Autonomous-Racing-Control-F1TENTH-/actions/runs/43207169981)
installed the pinned simulator, prepared Spielberg, passed 50 tests with no skips,
and completed the required dry/wet/mild-wet controller comparisons. This resolves
the simulator validation gate, while the local DNS/dependency limitation above
remains. Full source settings and measured results are in VALIDATION_RESULTS.md.
