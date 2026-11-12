import csv

import pytest

from scripts.replay_grip import replay


def recording(path, truth):
    with path.open('w', newline='') as stream:
        fields = ['time', 'speed', 'steering', 'yaw_rate', 'true_mu']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(30):
            writer.writerow(dict(time=i * 0.01, speed=4.0, steering=0.0,
                                 yaw_rate=0.0, true_mu=truth))


def test_true_friction_is_evaluation_only(tmp_path):
    source_a, source_b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    recording(source_a, 0.6)
    recording(source_b, 1.0)
    result_a = replay(source_a, tmp_path / 'out_a.csv')
    result_b = replay(source_b, tmp_path / 'out_b.csv')
    with (tmp_path / 'out_a.csv').open() as a, (tmp_path / 'out_b.csv').open() as b:
        for row_a, row_b in zip(csv.DictReader(a), csv.DictReader(b)):
            assert row_a['mu_hat'] == row_b['mu_hat']
            assert row_a['sigma'] == row_b['sigma']
    assert result_a['mu_rmse'] != result_b['mu_rmse']
    assert result_a['samples'] == 29


def test_replay_rejects_overwrite_and_bad_timestamps(tmp_path):
    source = tmp_path / 'input.csv'
    recording(source, 0.6)
    with pytest.raises(ValueError, match='differ'):
        replay(source, source)
    source.write_text('time,speed,steering,yaw_rate\n0,4,0,0\n0,4,0,0\n')
    output = tmp_path / 'out.csv'
    with pytest.raises(ValueError, match='increasing'):
        replay(source, output)
    assert not output.exists()
