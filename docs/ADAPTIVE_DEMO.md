# Adaptive-driving demo

The demo replays a completed adaptive experiment. It does not simulate, rerun
estimation, smooth the estimates, or change the recorded controller decisions.

## Generate the video

A separate rendering environment only needs the demo dependencies:

```bash
python3.12 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install -r requirements-demo.txt
python -m scripts.record_grip results/grip_wet/adaptive.csv --out results/adaptive-driving.mp4
```

Use `adaptive.csv` from an existing experiment or download the
`simulator-evidence-wet` artifact from
[validated run 43207169981](https://github.com/fauzan111/High-Speed-Autonomous-Racing-Control-F1TENTH-/actions/runs/43207169981).
The CSV is under `grip_wet/` inside the extracted archive. No simulator, map assets,
or CasADi installation is needed to render it. The map view is explicitly the
recorded route; its blue section shows reduced ground-truth friction.

Default output is 1280 x 720 H.264/YUV420P MP4, 24 fps, at 2x playback. It selects
lap zero through the first recorded finish crossing to show first exposure to the
wet patch. The title labels playback speed. Incomplete recordings are rejected
unless `--segment all` is explicitly supplied; that mode is labeled accordingly.

Three files are saved beside each other:

- `adaptive-driving.mp4`: browser-compatible video with fast-start metadata.
- `adaptive-driving.png`: representative preview from the reduced-grip portion.
- `adaptive-driving.json`: source SHA-256, selected times, codec, frame count,
  dimensions, playback speed, and uncertainty interpretation.

Existing outputs are protected. Use a fresh filename, or `--overwrite` deliberately.

## Create it in GitHub Actions

Run **Adaptive driving demo** from the Actions tab, choosing the branch that
contains the renderer. The default source run is 43207169981. The workflow
downloads its existing validated wet-run artifact, tests real H.264 encoding,
renders the first encounter, and uploads **adaptive-driving-demo**. The artifact
also includes the original CSV, experiment manifest and summary for provenance.
Artifact retention is finite; download the video/evidence for a durable copy.

If the source artifact has expired, run grip validation again and supply that
successful run's ID in the workflow input. The renderer never substitutes synthetic
motion for missing adaptive telemetry.

## What viewers see

The left panel shows the recorded route, reduced-grip section, driven trail and
current position. The right panels show speed, true friction, estimated friction,
the MPC budget, a two-sigma heuristic band and simulation time. Only observations
already recorded by the displayed time are used for the readout and trace.
Ground truth is for visualization only. The uncertainty band is not a calibrated
safety guarantee, and the footage represents the linear-tyre simulator.

The first-lap video duration should not be equated with the published flying-lap
benchmark. The simulation clock and source timestamps identify what is shown.
