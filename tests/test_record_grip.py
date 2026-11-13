import csv

import numpy as np
import pytest

from scripts.record_grip import FIELDS, find_encoder, frame_indices, load_recording, render, select_segment


def recording(path):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for i in range(9):
            angle = i * np.pi / 3
            writer.writerow(dict(time=i * .5, x=np.cos(angle), y=np.sin(angle), speed=2.0,
                                 true_mu=.6 if 2 <= i <= 4 else 1.0489,
                                 mu_hat=.7 if 2 <= i <= 4 else 1.0, sigma=.1,
                                 mu_used=.5 if 2 <= i <= 4 else .8, lap=0 if i < 6 else 1))


def test_first_encounter_stops_at_crossing(tmp_path):
    source = tmp_path / 'adaptive.csv'
    recording(source)
    data = load_recording(source)
    selected = select_segment(data)
    assert selected['time'][-1] == 3.0
    assert len(selected['time']) == 7
    assert selected['lap'][-1] == 1
    assert select_segment(data, 'all')['time'][-1] == 4.0


def test_incomplete_recording_requires_explicit_selection(tmp_path):
    source = tmp_path / 'adaptive.csv'
    recording(source)
    data = load_recording(source)
    data['lap'][:] = 0
    with pytest.raises(ValueError, match='No complete first lap'):
        select_segment(data)
    assert select_segment(data, 'all') is data
    data['lap'][:] = 1
    with pytest.raises(ValueError, match='lap zero'):
        select_segment(data)


def test_playback_frames_use_past_observations_and_include_final_sample():
    times = np.array([0.0, .1, .25, .5, 1.0])
    indices = frame_indices(times, fps=4, playback_speed=2)
    assert list(indices) == [0, 3, 4]
    assert len(indices) == 3
    irregular = frame_indices(times, fps=3, playback_speed=2)
    assert list(irregular) == [0, 3, 4]
    for fps, speed in [(0, 1), (24, 0), (24, float('nan'))]:
        with pytest.raises(ValueError):
            frame_indices(times, fps, speed)


@pytest.mark.parametrize('column,value', [('sigma', '-1'), ('time', 'nan'), ('mu_hat', ''), ('lap', '.5')])
def test_invalid_telemetry_is_rejected(tmp_path, column, value):
    source = tmp_path / 'adaptive.csv'
    recording(source)
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[1][column] = value
    with source.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError):
        load_recording(source)


def test_output_protection_runs_before_optional_imports(tmp_path):
    source, output = tmp_path / 'adaptive.csv', tmp_path / 'demo.mp4'
    recording(source)
    output.write_bytes(b'keep this')
    with pytest.raises(FileExistsError):
        render(source, output)
    assert output.read_bytes() == b'keep this'


def test_real_h264_render_and_decode(tmp_path):
    pytest.importorskip('matplotlib')
    try:
        find_encoder()
    except RuntimeError:
        pytest.skip('Demo encoding dependency unavailable')
    source, output = tmp_path / 'adaptive.csv', tmp_path / 'demo.mp4'
    recording(source)
    report = render(source, output, fps=4, playback_speed=2, width=640, height=360)
    assert output.stat().st_size > 1000
    assert output.with_suffix('.png').stat().st_size > 1000
    assert output.with_suffix('.json').is_file()
    assert report['simulation_end_s'] == 3.0
    assert report['codec'] == 'h264'
    assert report['frames'] == 7
    assert len(report['source_sha256']) == 64
