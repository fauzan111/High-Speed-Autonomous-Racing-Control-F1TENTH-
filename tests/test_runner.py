"""Runner contract checks with a controlled environment; no simulator installed."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from racer.estimation.grip import GripSchedule
from racer.estimation.online import GripEstimator
from racer.raceline import Raceline


class FakeEnv:
    def __init__(self, crash=False):
        self.unwrapped = self
        self.agent = SimpleNamespace(params={'mu': 1.0, 'other': 42}, state=np.zeros(7))
        self.sim = SimpleNamespace(agents=[self.agent])
        self.updates = []
        self.actions = []
        self.closed = False
        self.crash = crash

    def observation(self):
        return dict(poses_x=[self.tick % 4], poses_y=[0.0], poses_theta=[0.0],
                    linear_vels_x=[4.0], ang_vels_z=[0.0],
                    collisions=[self.crash and self.tick > 1])

    def reset(self, **kwargs):
        self.tick = 0
        return self.observation(), {}

    def update_params(self, params, index):
        assert index == 0
        assert params['other'] == 42
        self.agent.params = params
        self.updates.append(params['mu'])

    def step(self, action):
        self.actions.append(action)
        self.tick += 1
        return self.observation(), 0.01, self.tick >= 4, False, {'checkpoint_done': [self.tick >= 4]}

    def close(self):
        self.closed = True


@pytest.fixture
def setup_runner(monkeypatch):
    gym = ModuleType('gymnasium')
    monkeypatch.setitem(sys.modules, 'gymnasium', gym)
    monkeypatch.setitem(sys.modules, 'f1tenth_gym', ModuleType('f1tenth_gym'))
    spec = importlib.util.spec_from_file_location('racer._runner_test', Path(__file__).parents[1] / 'racer/runner.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    line = Raceline([0, 1, 2, 3], [0, 1, 2, 3], [0]*4, [0]*4, [0]*4, [4]*4, [0]*4)
    # Synthetic progress loop uses four equidistant waypoints.
    line.length = 4.0
    monkeypatch.setattr(module.Raceline, 'load', lambda name: line)
    return module, gym


class Controller:
    name = 'test'

    def reset(self):
        self.estimates = []

    def set_grip(self, mu, sigma):
        self.estimates.append((mu, sigma))
        self.mu_used = mu - sigma

    def __call__(self, state):
        return 0.2, 4.0


def test_schedule_and_flying_lap_survive_checkpoint_signal(setup_runner):
    runner, gym = setup_runner
    env = FakeEnv()
    gym.make = lambda *a, **kw: env
    result = runner.run_episode(Controller(), grip_schedule=GripSchedule(((0.0, 1.0), (2.0, 0.6)), 4.0))
    assert result.completed
    assert result.lap_time == pytest.approx(0.04)
    assert env.tick == 8
    assert env.updates == [1.0, 0.6, 1.0, 0.6]
    assert result.grip_log[0]['steering'] == 0.0  # measured, not commanded 0.2
    assert env.closed


def test_collision_stops_and_closes_environment(setup_runner):
    runner, gym = setup_runner
    env = FakeEnv(crash=True)
    gym.make = lambda *a, **kw: env
    result = runner.run_episode(Controller())
    assert result.crashed and not result.completed
    assert env.closed


def test_controller_error_closes_environment(setup_runner):
    runner, gym = setup_runner
    env = FakeEnv()
    gym.make = lambda *a, **kw: env

    class Broken(Controller):
        def __call__(self, state):
            raise RuntimeError('failed')
    with pytest.raises(RuntimeError, match='failed'):
        runner.run_episode(Broken())
    assert env.closed


def test_schedule_ground_truth_does_not_enter_estimator(setup_runner):
    runner, gym = setup_runner
    estimates = []
    for mu in [0.6, 1.0]:
        env = FakeEnv()
        gym.make = lambda *a, **kw: env
        ctrl = Controller()
        runner.run_episode(ctrl, grip_schedule=GripSchedule(((0.0, mu),), 4.0),
                           grip_estimator=GripEstimator())
        estimates.append(ctrl.estimates)
    assert estimates[0] == estimates[1]
