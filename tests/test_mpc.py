"""MPC regressions independent of map downloads and the simulator."""
import numpy as np
import pytest

ca = pytest.importorskip("casadi")

from racer.controllers.base import CarState
from racer.controllers.mpc import MPCController
from racer.raceline import Raceline


@pytest.fixture
def controller():
    n = 80
    angle = np.arange(n) * 2 * np.pi / n
    radius = 10.0
    return MPCController(Raceline(
        radius * angle, radius * np.cos(angle), radius * np.sin(angle),
        angle + np.pi / 2, np.full(n, 1 / radius), np.full(n, 8.0), np.zeros(n),
    ), horizon=5)


def test_friction_slack_has_positive_objective_cost(controller):
    """Regression: previously slack was free, making the friction limit ineffective."""
    opti = controller.opti
    objective = ca.Function("objective", [opti.x, opti.p], [opti.f])
    initial = np.zeros(opti.nx)
    # Locate slack's entries symbolically so this checks the actual registered objective.
    select = ca.Function("select_slack", [opti.x], [controller.slack])
    # The selector values are the corresponding one-based decision-variable indices.
    decision_indices = np.asarray(select(np.arange(opti.nx) + 1)).astype(int).reshape(-1) - 1
    assert len(decision_indices) == controller.N
    with_slack = initial.copy()
    with_slack[decision_indices] = 1.0
    assert float(objective(with_slack, np.zeros(opti.np))) > float(objective(initial, np.zeros(opti.np)))


def test_default_dry_target_is_preserved(controller):
    assert controller._reference_speeds == pytest.approx(controller.rl.vx * controller.speed_cap)


def test_grip_updates_existing_solver_and_corner_speed(controller):
    solver = controller.opti
    old_speeds = controller._reference_speeds.copy()
    assert controller.set_grip(0.7, sigma=0.05) == pytest.approx(0.6)
    assert controller.opti is solver
    st = CarState(controller.rl.x[0], controller.rl.y[0], controller.rl.psi[0], 5.0)
    ref = controller._build_reference(st, st.yaw)
    assert controller.a_lat_max == pytest.approx(0.7 * 0.6 * 9.81)
    assert np.all(controller._reference_speeds < old_speeds)
    assert np.all(controller._reference_speeds ** 2 * np.abs(controller.rl.kappa) <= controller.a_lat_max + 1e-10)
    assert np.all(ref[3] <= np.sqrt(controller.a_lat_max / 0.1) + 1e-10)


def test_grip_profile_brakes_before_corner_and_across_finish(controller):
    controller.rl.kappa[:] = 0
    controller.rl.kappa[0] = 0.8
    controller.set_grip(0.4)
    controller._build_reference(CarState(10.0, 0.0, np.pi / 2, 5.0), np.pi / 2)
    v = controller._reference_speeds
    ds = np.diff(np.append(controller.rl.s, controller.rl.length))
    a = min(controller.p.a_max, controller.mu_used * 9.81 * controller.grip_frac)
    assert v[-1] < controller.rl.vx[-1] * controller.speed_cap
    assert np.all(v ** 2 <= np.roll(v, -1) ** 2 + 2 * a * ds + 1e-9)
    assert np.all(np.roll(v, -1) ** 2 <= v ** 2 + 2 * a * ds + 1e-9)


@pytest.mark.parametrize("args", [(float("nan"),), (0,), (0.6, -0.1), (0.6, 0.1, -1)])
def test_invalid_grip_is_rejected(controller, args):
    with pytest.raises(ValueError):
        controller.set_grip(*args)


def test_estimator_updates_preserve_solve_cadence_and_defer_profile_work(controller, monkeypatch):
    state = CarState(10.0, 0.0, np.pi / 2, 5.0)
    solves = []
    profile_updates = []
    update_profile = controller._update_speed_profile

    def counted_profile_update():
        profile_updates.append(controller._tick)
        update_profile()

    def counted_solve(st):
        solves.append(controller._tick)
        controller._build_reference(st, st.yaw)

    monkeypatch.setattr(controller, "_update_speed_profile", counted_profile_update)
    monkeypatch.setattr(controller, "_solve", counted_solve)
    for step in range(9):
        controller.set_grip(0.7 - step * 0.01)
        controller(state)
    assert solves == [0, 3, 6]
    assert profile_updates == solves
    assert controller._profile_dirty  # latest estimate waits for scheduled solve


def test_real_solver_responds_to_reduced_grip(controller):
    """Exercise IPOPT, parameter updates, warm starts and commanded braking."""
    state = CarState(10.0, 0.0, np.pi / 2, 7.5)
    dry_command = controller(state)
    assert controller.solve_calls == 1
    assert controller.solve_fails == 0
    assert np.all(np.isfinite(dry_command))
    controller.set_grip(0.3)
    # Held commands preserve cadence; the next scheduled solve uses new grip.
    assert controller(state) == dry_command
    assert controller(state) == dry_command
    wet_command = controller(state)
    assert controller.solve_calls == 2
    assert controller.solve_fails == 0
    assert wet_command[1] < dry_command[1] - 0.05
    assert controller.p.s_min <= wet_command[0] <= controller.p.s_max
    assert 0.0 <= wet_command[1] <= controller.p.v_max
    assert controller.opti.value(controller.p_a_lat_max) == pytest.approx(0.7 * 0.3 * 9.81)
    assert np.all(np.isfinite(controller._Xguess))
    assert np.all(np.isfinite(controller._Uguess))


def test_solver_failure_brakes_even_at_low_speed(controller, monkeypatch):
    def fail_solve(_):
        raise RuntimeError("forced non-convergence")

    monkeypatch.setattr(ca.Opti, "solve", fail_solve)
    controller._last_cmd = (0.1, 1.0)
    steering, speed = controller(CarState(10.0, 0.0, np.pi / 2, 0.4))
    assert steering == pytest.approx(0.1)
    assert speed == 0.0
    assert controller.solve_fails == 1
    assert controller._Xguess is None
    assert controller._Uguess is None
