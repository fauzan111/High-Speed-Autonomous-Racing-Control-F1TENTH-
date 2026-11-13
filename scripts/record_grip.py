"""Render an adaptive-driving CSV replay without importing the simulator.

python -m scripts.record_grip results/grip_wet/adaptive.csv --out results/adaptive-demo.mp4
Requires requirements-demo.txt. The default clip shows the first encounter lap.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

FIELDS = ('time', 'x', 'y', 'speed', 'true_mu', 'mu_hat', 'sigma', 'mu_used', 'lap')


def load_recording(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError('CSV requires: ' + ', '.join(FIELDS))
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError('Recording needs at least two samples')
    try:
        data = {key: np.array([float(row[key]) for row in rows]) for key in FIELDS}
    except (ValueError, TypeError) as exc:
        raise ValueError('Recording contains empty or nonnumeric telemetry') from exc
    if not all(np.isfinite(values).all() for values in data.values()):
        raise ValueError('Recording contains nonfinite telemetry')
    if not np.all(np.diff(data['time']) > 0):
        raise ValueError('Timestamps must increase strictly')
    if np.any(data['sigma'] < 0) or np.any(data['speed'] < 0):
        raise ValueError('Speed and uncertainty must be nonnegative')
    if any(np.any(data[key] <= 0) for key in ('true_mu', 'mu_hat', 'mu_used')):
        raise ValueError('Friction values must be positive')
    if np.any(data['lap'] < 0) or np.any(data['lap'] != np.floor(data['lap'])) or np.any(np.diff(data['lap']) < 0):
        raise ValueError('Lap counters must be nonnegative, integral and nondecreasing')
    return data


def select_segment(data, selection='first-lap'):
    if selection == 'all':
        return data
    if selection != 'first-lap':
        raise ValueError('Unknown segment')
    if data['lap'][0] != 0:
        raise ValueError('First encounter requires a recording starting on lap zero')
    crossings = np.flatnonzero(data['lap'] > 0)
    if not crossings.size:
        raise ValueError('No complete first lap recorded; use --segment all for a partial replay')
    # Include the actual crossing sample, without the rest of the flying lap.
    return {key: value[:int(crossings[0]) + 1] for key, value in data.items()}


def frame_indices(times, fps, playback_speed):
    if fps <= 0 or not math.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError('FPS and playback speed must be positive')
    count = math.ceil((times[-1] - times[0]) / playback_speed * fps) + 1
    targets = np.minimum(times[0] + np.arange(count) * playback_speed / fps, times[-1])
    # Telemetry always comes from an observation already recorded at that time.
    return np.searchsorted(times, targets, side='right') - 1


def find_encoder():
    binary = shutil.which('ffmpeg')
    if binary:
        return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError('Install requirements-demo.txt or make ffmpeg available') from exc


def render(input_path: Path, output: Path, *, fps=24, playback_speed=2.0,
           width=1280, height=720, segment='first-lap', overwrite=False):
    input_path, output = Path(input_path), Path(output)
    if width < 640 or height < 360 or width % 2 or height % 2:
        raise ValueError('Video size must be even and at least 640 x 360')
    if output.suffix.lower() != '.mp4':
        raise ValueError('Output must have an .mp4 extension')
    preview, metadata = output.with_suffix('.png'), output.with_suffix('.json')
    targets = [output, preview, metadata]
    if input_path.resolve() in [path.resolve() for path in targets]:
        raise ValueError('Output must not replace source telemetry')
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError('Demo output already exists; choose a new path or --overwrite')
    data = select_segment(load_recording(input_path), segment)
    indices = frame_indices(data['time'], fps, playback_speed)
    encoder = find_encoder()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    output.parent.mkdir(parents=True, exist_ok=True)
    bg, panel, ink = '#0b1220', '#111d30', '#e8eff7'
    yellow, blue, green = '#ffd166', '#55b7ff', '#67e8b4'
    scale = width / 1280
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=bg)
    grid = fig.add_gridspec(3, 2, left=.045, right=.975, bottom=.105, top=.82,
                           width_ratios=[1.05, 1], height_ratios=[.8, 1.5, 1], wspace=.18, hspace=.43)
    track = fig.add_subplot(grid[:, 0])
    hud = fig.add_subplot(grid[0, 1])
    grip = fig.add_subplot(grid[1, 1])
    speed_ax = fig.add_subplot(grid[2, 1], sharex=grip)
    for ax in [track, hud, grip, speed_ax]:
        ax.set_facecolor(panel)
        ax.tick_params(colors=ink, labelsize=8 * scale)
        for spine in ax.spines.values():
            spine.set_color('#344258')
    fig.text(.045, .925, 'ADAPTIVE DRIVING / GRIP ESTIMATION', color=ink,
             fontsize=21 * scale, weight='bold')
    caption = 'First encounter lap' if segment == 'first-lap' else 'Recorded samples (may include partial laps)'
    fig.text(.045, .875, f'{caption}  |  CSV replay  |  {playback_speed:g}x playback',
             color='#9cb0c9', fontsize=11 * scale)
    fig.text(.045, .035, 'Linear-tyre simulation  |  uncertainty band is a heuristic, not a safety guarantee',
             color='#9cb0c9', fontsize=9 * scale)
    track.set_aspect('equal', adjustable='datalim')
    track.plot(data['x'], data['y'], color='#344258', lw=8 * scale,
               solid_capstyle='round', label='Recorded route')
    variable_grip = np.ptp(data['true_mu']) > 1e-6
    reduced = data['true_mu'] < data['true_mu'].max() - 1e-6
    if variable_grip:
        track.plot(np.where(reduced, data['x'], np.nan), np.where(reduced, data['y'], np.nan),
                   color=blue, lw=8 * scale, solid_capstyle='round', label='Reduced grip (ground truth)')
    trail, = track.plot([], [], color=yellow, lw=2 * scale)
    car, = track.plot([], [], 'o', color='white', markeredgecolor=green, markersize=9 * scale, markeredgewidth=2)
    track.set_xticks([]); track.set_yticks([])
    track.set_title('Position / recorded route', color=ink, fontsize=11 * scale, loc='left', pad=12)
    track.legend(loc='lower left', facecolor=panel, labelcolor=ink, edgecolor='none', fontsize=8 * scale)
    hud.set_axis_off()
    status = hud.text(0, .9, '', color=blue, fontsize=12 * scale, weight='bold', va='top')
    readout = hud.text(0, .38, '', color=ink, fontsize=13 * scale, va='top', fontfamily='monospace')
    t = data['time'] - data['time'][0]
    grip_lines = [grip.plot([], [], color=color, lw=1.8, label=label)[0]
                  for color, label in [(ink, 'True mu'), (green, 'Estimated mu'), (yellow, 'MPC budget')]]
    band = PolyCollection([], facecolors=green, alpha=.12, edgecolors='none', label='+/-2 sigma (heuristic)')
    grip.add_collection(band)
    grip.set_xlim(0, t[-1])
    grip.set_ylim(min(0, float(np.min(data['mu_hat'] - 2*data['sigma']))),
                  max(1.2, float(np.max(data['mu_hat'] + 2*data['sigma']))) + .05)
    grip.set_ylabel('Friction parameter', color=ink, fontsize=9 * scale)
    grip.legend(loc='upper right', facecolor=panel, labelcolor=ink, edgecolor='none', fontsize=7 * scale, ncol=2)
    speed_line, = speed_ax.plot([], [], color=yellow, lw=1.8)
    speed_ax.set_ylim(0, max(1, data['speed'].max() * 1.15))
    speed_ax.set_ylabel('Speed [m/s]', color=ink, fontsize=9 * scale)
    speed_ax.set_xlabel('Simulation time from clip start [s]', color=ink, fontsize=9 * scale)
    cursors = [ax.axvline(0, color='#9cb0c9', lw=.8) for ax in [grip, speed_ax]]
    for ax in [grip, speed_ax]:
        ax.grid(alpha=.12, color='white')
    plt.setp(grip.get_xticklabels(), visible=False)

    def draw(index):
        history = np.unique(np.r_[np.arange(0, index + 1, max(1, (index + 1)//1200)), index])
        trail.set_data(data['x'][history], data['y'][history])
        car.set_data([data['x'][index]], [data['y'][index]])
        road = ('REDUCED GRIP' if reduced[index] else 'DRY SECTION') if variable_grip else 'CONSTANT GRIP'
        status.set_text(f'{road}  /  t = {t[index]:05.2f} s')
        readout.set_text(f"Speed {data['speed'][index]:4.2f} m/s    mu {data['mu_hat'][index]:4.2f}\n"
                         f"True  {data['true_mu'][index]:4.2f}         budget {data['mu_used'][index]:4.2f}")
        for line, field in zip(grip_lines, ['true_mu', 'mu_hat', 'mu_used']):
            line.set_data(t[history], data[field][history])
        lower = data['mu_hat'][history] - 2 * data['sigma'][history]
        upper = data['mu_hat'][history] + 2 * data['sigma'][history]
        band.set_verts([np.column_stack((np.r_[t[history], t[history][::-1]], np.r_[lower, upper[::-1]]))])
        speed_line.set_data(t[history], data['speed'][history])
        for cursor in cursors:
            cursor.set_xdata([t[index], t[index]])
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    temporary = tempfile.NamedTemporaryFile(suffix='.mp4', dir=output.parent, delete=False)
    temporary.close()
    temporary_path = Path(temporary.name)
    process = None
    try:
        with tempfile.TemporaryFile() as errors:
            command = [encoder, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
                       '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0',
                       '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                       '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary_path)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            try:
                for number, index in enumerate(indices):
                    process.stdin.write(draw(int(index)).tobytes())
                    if number % 120 == 0:
                        print(f'Rendering frame {number + 1}/{len(indices)}', flush=True)
            finally:
                process.stdin.close()
                code = process.wait()
            if code:
                errors.seek(0)
                raise RuntimeError('Video encoding failed: ' + errors.read().decode(errors='replace')[-2000:])
        # Verify the encoded file can be decoded before publishing any outputs.
        subprocess.run([encoder, '-v', 'error', '-i', str(temporary_path), '-f', 'null', '-'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        wet_indices = np.flatnonzero(reduced)
        preview_index = int(wet_indices[len(wet_indices)//2]) if wet_indices.size else len(t)//2
        draw(preview_index)
        fig.savefig(preview, dpi=100, facecolor=bg)
        report = dict(source_csv=str(input_path), source_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                      selection=segment, simulation_start_s=float(data['time'][0]),
                      simulation_end_s=float(data['time'][-1]), fps=fps, playback_speed=playback_speed,
                      frames=len(indices), encoded_duration_s=len(indices)/fps, width=width, height=height,
                      codec='h264', pixel_format='yuv420p', preview_time_s=float(data['time'][preview_index]),
                      ground_truth_usage='Visualization only; this renderer never runs estimation or control',
                      uncertainty='Plus/minus two sigma is a heuristic, not calibrated probability')
        metadata.write_text(json.dumps(report, indent=2) + '\n')
        temporary_path.replace(output)
        return report
    finally:
        plt.close(fig)
        temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, default=Path('results/adaptive-demo.mp4'))
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--playback-speed', type=float, default=2.0)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--segment', choices=['first-lap', 'all'], default='first-lap')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    print(json.dumps(render(args.input, args.out, fps=args.fps, playback_speed=args.playback_speed,
                            width=args.width, height=args.height, segment=args.segment,
                            overwrite=args.overwrite), indent=2))


if __name__ == '__main__':
    main()
