"""
Nonlinear Model Predictive Control (Stage 2 — the centerpiece).

At every control tick the controller solves a finite-horizon optimal control
problem: "given where I am now, what sequence of steering + acceleration inputs
keeps me on the racing line at its target speed over the next ~2 seconds?" It
applies the first input, then re-solves at the next tick (receding horizon).

Unlike pure pursuit, MPC reasons about the vehicle model and future reference
*together*, so it plans the corner entry before the corner — it brakes and
turns ahead of time instead of understeering wide.

Model: kinematic bicycle (state = [X, Y, psi, v], input = [delta, a]), matched
to the F1TENTH wheelbase and actuator limits, discretized with RK4. Solved with
CasADi + IPOPT.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from .. import config
from ..raceline import Raceline
from .base import CarState, Controller


class MPCController(Controller):
    name = "mpc"

    def __init__(
        self,
        raceline: Raceline,
        params: config.VehicleParams | None = None,
        horizon: int = 25,          # number of stages N
        dt: float = 0.1,            # prediction step [s]
        resolve_every: int = 3,     # re-solve every k sim steps (sim dt=0.01)
        # cost weights
        q_pos: float = 8.0,         # contour/lag position tracking
        q_yaw: float = 0.5,         # heading tracking
        q_v: float = 0.3,           # speed tracking
        r_delta: float = 0.01,      # steering effort
        r_a: float = 0.002,         # accel effort
        rd_delta: float = 0.20,     # steering rate (smoothness)
        rd_a: float = 0.02,         # accel rate
        q_over: float = 40.0,       # penalty for exceeding raceline speed
        speed_cap: float = 0.97,    # scale raceline target speed
        grip_frac: float = 0.7,     # usable fraction of mu*g for lateral accel
    ):
        self.rl = raceline
        self.p = params or config.VehicleParams()
        self.N = horizon
        self.dt = dt
        self.resolve_every = resolve_every
        self.speed_cap = speed_cap
        self.a_lat_max = grip_frac * self.p.mu * 9.81

        self._build_solver(q_pos, q_yaw, q_v, r_delta, r_a, rd_delta, rd_a, q_over)
        self.reset()

    # ------------------------------------------------------------------ build
    def _f(self, x, u):
        """Continuous kinematic bicycle dynamics."""
        L = self.p.wheelbase
        psi, v = x[2], x[3]
        delta, a = u[0], u[1]
        return ca.vertcat(
            v * ca.cos(psi),
            v * ca.sin(psi),
            v / L * ca.tan(delta),
            a,
        )

    def _rk4(self, x, u):
        dt = self.dt
        k1 = self._f(x, u)
        k2 = self._f(x + dt / 2 * k1, u)
        k3 = self._f(x + dt / 2 * k2, u)
        k4 = self._f(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _build_solver(self, q_pos, q_yaw, q_v, r_delta, r_a, rd_delta, rd_a, q_over):
        N = self.N
        opti = ca.Opti()

        X = opti.variable(4, N + 1)   # states
        U = opti.variable(2, N)       # inputs [delta, a]
        slack = opti.variable(N)      # friction-circle slack (>= 0)

        x0 = opti.parameter(4)             # current state
        ref = opti.parameter(4, N)         # [Xr, Yr, psir, vr] for stages 1..N
        u_prev = opti.parameter(2)         # last applied input (rate penalty)

        opti.subject_to(X[:, 0] == x0)
        cost = 0
        for k in range(N):
            opti.subject_to(X[:, k + 1] == self._rk4(X[:, k], U[:, k]))

            ex = X[0, k + 1] - ref[0, k]
            ey = X[1, k + 1] - ref[1, k]
            cost += q_pos * (ex ** 2 + ey ** 2)
            cost += q_yaw * (1 - ca.cos(X[2, k + 1] - ref[2, k]))
            cost += q_v * (X[3, k + 1] - ref[3, k]) ** 2
            # One-sided penalty on exceeding the (grip-optimal) raceline speed.
            # This removes the kinematic model's perverse incentive to speed up
            # in order to gain yaw rate (psidot = v/L*tan(delta)) when steering
            # saturates in a tight corner.
            over = ca.fmax(0.0, X[3, k + 1] - ref[3, k])
            cost += q_over * over ** 2
            cost += r_delta * U[0, k] ** 2 + r_a * U[1, k] ** 2

            u_prevk = u_prev if k == 0 else U[:, k - 1]
            cost += rd_delta * (U[0, k] - u_prevk[0]) ** 2
            cost += rd_a * (U[1, k] - u_prevk[1]) ** 2

        opti.minimize(cost)

        # actuator / state limits
        opti.subject_to(opti.bounded(self.p.s_min, U[0, :], self.p.s_max))
        opti.subject_to(opti.bounded(-self.p.a_max, U[1, :], self.p.a_max))
        opti.subject_to(opti.bounded(0.0, X[3, :], self.p.v_max))

        # friction-circle limit: keep lateral acceleration within grip so the
        # kinematic plan stays feasible for the real (dynamic) car. This is what
        # forces the controller to brake *before* a tight corner. Implemented as
        # a SOFT constraint (slack + heavy penalty) so the NLP never becomes
        # infeasible near the limit — a hard bound makes IPOPT fail and freeze.
        L = self.p.wheelbase
        w_slack = 1e3
        for k in range(N):
            a_lat = X[3, k] ** 2 / L * ca.tan(U[0, k])
            opti.subject_to(a_lat <= self.a_lat_max + slack[k])
            opti.subject_to(a_lat >= -self.a_lat_max - slack[k])
            opti.subject_to(slack[k] >= 0)
            cost += w_slack * slack[k] ** 2
        # steering-rate limit between stages
        for k in range(1, N):
            dd = U[0, k] - U[0, k - 1]
            opti.subject_to(opti.bounded(self.p.sv_min * self.dt, dd,
                                         self.p.sv_max * self.dt))

        opti.solver("ipopt", {
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.max_iter": 150,
            "ipopt.sb": "yes",
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.mu_strategy": "adaptive",
        })

        self.opti = opti
        self.X, self.U, self.slack = X, U, slack
        self.p_x0, self.p_ref, self.p_uprev = x0, ref, u_prev

    # ------------------------------------------------------------------ run
    def reset(self) -> None:
        self._tick = 0
        self._u_prev = np.zeros(2)
        self._last_cmd = (0.0, 0.0)
        self._Xguess = None
        self._Uguess = None
        self._yaw_cont = None   # continuously-unwrapped heading
        self.solve_fails = 0
        self.solve_calls = 0

    def _continuous_yaw(self, raw_yaw: float) -> float:
        """Unwrap the measured heading so it never jumps by 2*pi between steps.
        A raw 2*pi jump would make the warm-start guess inconsistent with x0 and
        send the solver to a garbage 'unwinding' solution (full steering lock)."""
        if self._yaw_cont is None:
            self._yaw_cont = raw_yaw
        else:
            d = np.arctan2(np.sin(raw_yaw - self._yaw_cont),
                           np.cos(raw_yaw - self._yaw_cont))
            self._yaw_cont += d
        return self._yaw_cont

    def _build_reference(self, state: CarState, yaw_cont: float):
        # Space reference points by the arc length the car *should* cover each
        # step at the racing line's target speed (progress = integral of v dt).
        # Spacing by current speed instead would collapse the reference to a
        # tight cluster at low speed and pin the car there.
        rl = self.rl
        L = rl.length
        s_acc = rl.progress(state.x, state.y)
        s_query = np.empty(self.N)
        for k in range(self.N):
            v_here = np.interp(s_acc % L, rl.s, rl.vx, period=L) * self.speed_cap
            s_acc += max(v_here, 1.0) * self.dt
            s_query[k] = s_acc % L

        xr = np.interp(s_query, rl.s, rl.x, period=L)
        yr = np.interp(s_query, rl.s, rl.y, period=L)
        vxr = np.clip(np.interp(s_query, rl.s, rl.vx, period=L) * self.speed_cap,
                      0.0, self.p.v_max)
        # Interpolate heading through sin/cos so the raceline's psi wrap (it
        # passes through +/-pi mid-track) never injects a spurious 2*pi jump
        # into the reference — that jump used to make the MPC panic-brake. The
        # cost uses 1-cos(psi-psi_ref), which is wrap-safe, so no unwrap needed.
        sinr = np.interp(s_query, rl.s, np.sin(rl.psi), period=L)
        cosr = np.interp(s_query, rl.s, np.cos(rl.psi), period=L)
        psir = np.arctan2(sinr, cosr)
        # Put the reference heading on the same continuous branch as the car's
        # unwrapped heading so it agrees with x0 and the warm-start guess.
        psir = np.unwrap(np.concatenate(([yaw_cont], psir)))[1:]
        return np.vstack([xr, yr, psir, vxr])

    def __call__(self, state: CarState) -> tuple[float, float]:
        if self._tick % self.resolve_every == 0:
            self._solve(state)
        self._tick += 1
        return self._last_cmd

    def _reference_guess(self, x0, ref):
        """Physically meaningful initial guess: sit on the reference line at the
        reference speed, zero control. Keeps IPOPT starting near feasibility so
        it converges reliably (and breaks failure cascades)."""
        Xg = np.zeros((4, self.N + 1))
        Xg[:, 0] = x0
        Xg[:, 1:] = ref  # [Xr, Yr, psir, vr] for stages 1..N
        Ug = np.zeros((2, self.N))
        return Xg, Ug

    def _solve(self, state: CarState) -> None:
        yaw_cont = self._continuous_yaw(state.yaw)
        x0 = np.array([state.x, state.y, yaw_cont, max(state.v, 0.0)])
        ref = self._build_reference(state, yaw_cont)

        self.opti.set_value(self.p_x0, x0)
        self.opti.set_value(self.p_ref, ref)
        self.opti.set_value(self.p_uprev, self._u_prev)

        # Warm start from the previous (shifted) solution when available, else
        # from the reference trajectory.
        if self._Xguess is not None:
            Xg, Ug = self._Xguess, self._Uguess
        else:
            Xg, Ug = self._reference_guess(x0, ref)
        self.opti.set_initial(self.X, Xg)
        self.opti.set_initial(self.U, Ug)

        self.solve_calls += 1
        try:
            sol = self.opti.solve()
            Xs, Us = sol.value(self.X), sol.value(self.U)
        except RuntimeError:
            # Solver failed to converge. Do NOT trust the last iterate (it can
            # contain garbage-high speeds that overspeed the car into a wall).
            # Brake, hold the last good steering, and rebuild the guess from the
            # reference next tick so we don't get stuck failing on a bad guess.
            self.solve_fails += 1
            prev_delta = float(self._last_cmd[0])
            brake_v = max(state.v - 3.0, 1.0)
            self._last_cmd = (prev_delta, brake_v)
            self._Xguess = None
            self._Uguess = None
            return

        delta = float(np.clip(Us[0, 0], self.p.s_min, self.p.s_max))
        v_cmd = float(np.clip(Xs[3, 1], 0.0, self.p.v_max))  # optimal next speed
        self._last_cmd = (delta, v_cmd)
        self._u_prev = np.array([Us[0, 0], Us[1, 0]])

        # shift solution for next warm start
        self._Xguess = np.hstack([Xs[:, 1:], Xs[:, -1:]])
        self._Uguess = np.hstack([Us[:, 1:], Us[:, -1:]])
