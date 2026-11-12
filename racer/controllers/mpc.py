"""
Nonlinear Model Predictive Control (Stage 2 — the centerpiece).

At every control tick the controller solves a finite-horizon optimal control
problem: "given where I am now, what sequence of steering + acceleration inputs
keeps me on the racing line at its target speed over the next ~2 seconds?" It
applies the first input, then re-solves at the next tick (receding horizon).

Unlike pure pursuit, MPC reasons about the vehicle model and future reference
*together*, so it plans the corner entry before the corner — it brakes and
turns ahead of time instead of understeering wide.

Model: four-state bicycle with optional steady-state single-track yaw gain
and body-slip correction (state = [X, Y, psi, v], input = [delta, a]). This
approximates linear-tyre equilibrium, not full transient dynamics or tyre
saturation. Matched to F1TENTH parameters, integrated with RK4; CasADi + IPOPT.
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
        speed_cap: float = 0.70,    # scale raceline target speed
        grip_frac: float = 0.7,     # usable fraction of mu*g for lateral accel
        w_slack: float = 1e3,       # lateral-acceleration slack penalty [cost/(m/s^2)^2]
        understeer_compensation: bool = True,  # steady-state dynamic ST approximation
        adapt_dynamics: bool = False,  # experimental: use point estimate for tyre model
    ):
        self.rl = raceline
        self.p = params or config.VehicleParams()
        self.N = horizon
        self.dt = dt
        self.resolve_every = resolve_every
        self.speed_cap = speed_cap
        if not 0.0 < grip_frac <= 1.0:
            raise ValueError("grip_frac must be in (0, 1]")
        if not np.isfinite(w_slack) or w_slack < 0:
            raise ValueError("w_slack must be finite and nonnegative")
        self.w_slack = float(w_slack)
        self.understeer_compensation = understeer_compensation
        self.adapt_dynamics = adapt_dynamics
        self.model_mu = self.p.mu
        self.grip_frac = grip_frac
        self.mu_used = self.p.mu
        self._grip_adapted = False
        self._profile_dirty = False
        self.a_lat_max = grip_frac * self.mu_used * 9.81
        self._update_speed_profile()

        self._build_solver(q_pos, q_yaw, q_v, r_delta, r_a, rd_delta, rd_a, q_over)
        self.reset()

    # ------------------------------------------------------------------ build
    def _yaw_and_slip(self, speed, steering):
        """Steady-state ST approximation, with the simulator's low-speed switch.

        Setting yaw acceleration and slip-angle rate to zero in F1TENTH's
        linear-tyre ST equations (zero longitudinal load transfer) gives:
          delta = [L + v^2/(mu*g)*(1/C_Sf - 1/C_Sr)] * curvature
          beta = [lr - v^2/(mu*g*C_Sr)] * curvature
        This compensates both yaw gain and velocity direction. It is an
        equilibrium approximation, not transient tyre dynamics or a saturation
        model. Below 0.5 m/s retain the simulator's kinematic branch.
        """
        p = self.p
        if not self.understeer_compensation:
            return speed / p.wheelbase * ca.tan(steering), 0.0
        mu = self.p_mu
        effective_wheelbase = (p.wheelbase + speed ** 2 / (mu * 9.81)
                               * (1.0 / p.C_Sf - 1.0 / p.C_Sr))
        curvature = steering / effective_wheelbase
        dynamic_yaw_rate = speed * curvature
        dynamic_slip = (p.lr - speed ** 2 / (mu * 9.81 * p.C_Sr)) * curvature
        dynamic = ca.fabs(speed) >= 0.5
        return (ca.if_else(dynamic, dynamic_yaw_rate,
                           speed / p.wheelbase * ca.tan(steering)),
                ca.if_else(dynamic, dynamic_slip, 0.0))

    def _f(self, x, u):
        """Continuous bicycle dynamics, optionally corrected for steady tyre slip."""
        psi, v = x[2], x[3]
        delta, a = u[0], u[1]
        yaw_rate, slip = self._yaw_and_slip(v, delta)
        return ca.vertcat(
            v * ca.cos(psi + slip),
            v * ca.sin(psi + slip),
            yaw_rate,
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
        slack = opti.variable(N)      # lateral-acceleration slack (>= 0)

        x0 = opti.parameter(4)             # current state
        ref = opti.parameter(4, N)         # [Xr, Yr, psir, vr] for stages 1..N
        u_prev = opti.parameter(2)         # last applied input (rate penalty)
        a_lat_max = opti.parameter()       # updated without rebuilding the NLP
        self.p_mu = opti.parameter()      # physical-model point value, not grip budget

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

        # actuator / state limits
        opti.subject_to(opti.bounded(self.p.s_min, U[0, :], self.p.s_max))
        opti.subject_to(opti.bounded(-self.p.a_max, U[1, :], self.p.a_max))
        opti.subject_to(opti.bounded(0.0, X[3, :], self.p.v_max))

        # Soft lateral-acceleration bound, using the same predicted yaw gain
        # as the motion model. This reserves a fraction of mu*g for cornering;
        # it does not enforce a combined longitudinal/lateral friction circle.
        # Slack permits recovery from states already outside the envelope.
        # Zero is allowed for diagnostic reproduction of the historical free-
        # slack controller. It disables grip enforcement and is not a safe mode.
        w_slack = self.w_slack
        for k in range(N):
            yaw_rate, _ = self._yaw_and_slip(X[3, k], U[0, k])
            a_lat = X[3, k] * yaw_rate
            opti.subject_to(a_lat <= a_lat_max + slack[k])
            opti.subject_to(a_lat >= -a_lat_max - slack[k])
            opti.subject_to(slack[k] >= 0)
            cost += w_slack * slack[k] ** 2
        # steering-rate limit between stages
        for k in range(1, N):
            dd = U[0, k] - U[0, k - 1]
            opti.subject_to(opti.bounded(self.p.sv_min * self.dt, dd,
                                         self.p.sv_max * self.dt))

        # Register the complete objective, including friction slack penalties.
        opti.minimize(cost)
        opti.set_value(a_lat_max, self.a_lat_max)
        opti.set_value(self.p_mu, self.model_mu)
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
        self.p_a_lat_max = a_lat_max

    def set_grip(self, mu_hat: float, sigma: float = 0.0,
                 uncertainty_scale: float = 2.0) -> float:
        """Use a bounded lower-confidence grip estimate for future solves.

        The floor is a numerical operating assumption, not a safety guarantee.
        True simulator friction must never be passed here by adaptive experiments.
        By default the physical yaw/slip model remains at nominal friction;
        only the cornering budget and speed profile change. Experimental
        adapt_dynamics uses mu_hat, never its lower confidence bound.
        """
        values = np.asarray([mu_hat, sigma, uncertainty_scale], dtype=float)
        if not np.all(np.isfinite(values)) or mu_hat <= 0 or sigma < 0 or uncertainty_scale < 0:
            raise ValueError("need finite positive mu_hat and nonnegative sigma/uncertainty_scale")
        mu_used = float(np.clip(mu_hat - uncertainty_scale * sigma, 0.1, self.p.mu))
        self._profile_dirty |= not self._grip_adapted or mu_used != self.mu_used
        self.mu_used = mu_used
        if self.adapt_dynamics:
            # Uncertainty reduces the available grip budget, not expected tyre
            # cornering stiffness. This mode uses the un-margined point estimate.
            self.model_mu = float(np.clip(mu_hat, 0.1, self.p.mu))
        self.a_lat_max = self.grip_frac * self.mu_used * 9.81
        self._grip_adapted = True
        # Queue updates until the next scheduled solve. This preserves the same
        # solve cadence in fixed and adaptive arms and times profile work inside
        # the controller call rather than hiding it in the estimator callback.
        return self.mu_used

    def _update_speed_profile(self) -> None:
        """Cap corner speeds, then propagate braking/acceleration around the loop.

        This assumes the current grip everywhere in the preview, without access
        to a map of future wet patches. Limits are longitudinal/lateral bounds,
        not a combined tyre-force-circle guarantee.
        """
        rl = self.rl
        if not self._grip_adapted:
            # Preserve the shipped dry baseline target until adaptation is enabled.
            self._reference_speeds = np.clip(rl.vx * self.speed_cap, 0.0, self.p.v_max)
            return
        curve_limit = np.sqrt(self.a_lat_max / np.maximum(np.abs(rl.kappa), 1e-9))
        speeds = np.minimum(np.clip(rl.vx * self.speed_cap, 0.0, self.p.v_max), curve_limit)
        ds = np.diff(np.append(rl.s, rl.length))
        a_long = min(self.p.a_max, self.mu_used * 9.81 * self.grip_frac)
        # Two sweeps include constraints crossing the finish line. Squared-speed
        # propagation only lowers speeds, so the second sweep settles the loop.
        for _ in range(2):
            for i in range(len(speeds) - 1, -1, -1):
                j = (i + 1) % len(speeds)
                speeds[i] = min(speeds[i], np.sqrt(speeds[j] ** 2 + 2 * a_long * ds[i]))
            for i in range(len(speeds)):
                j = (i + 1) % len(speeds)
                speeds[j] = min(speeds[j], np.sqrt(speeds[i] ** 2 + 2 * a_long * ds[i]))
        self._reference_speeds = speeds

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
        if self._profile_dirty:
            self._update_speed_profile()
            self._profile_dirty = False
        # Space reference points by the arc length the car *should* cover each
        # step at the racing line's target speed (progress = integral of v dt).
        # Spacing by current speed instead would collapse the reference to a
        # tight cluster at low speed and pin the car there.
        rl = self.rl
        L = rl.length
        s_acc = rl.progress(state.x, state.y)
        s_query = np.empty(self.N)
        for k in range(self.N):
            v_here = np.interp(s_acc % L, rl.s, self._reference_speeds, period=L)
            s_acc += max(v_here, 1.0) * self.dt
            s_query[k] = s_acc % L

        xr = np.interp(s_query, rl.s, rl.x, period=L)
        yr = np.interp(s_query, rl.s, rl.y, period=L)
        vxr = np.clip(np.interp(s_query, rl.s, self._reference_speeds, period=L),
                      0.0, self.p.v_max)
        # Interpolate heading through sin/cos so the raceline's psi wrap (it
        # passes through +/-pi mid-track) never injects a spurious 2*pi jump
        # into the reference — that jump used to make the MPC panic-brake. The
        # cost uses 1-cos(psi-psi_ref), which is wrap-safe, so no unwrap needed.
        sinr = np.interp(s_query, rl.s, np.sin(rl.psi), period=L)
        cosr = np.interp(s_query, rl.s, np.cos(rl.psi), period=L)
        psir = np.arctan2(sinr, cosr)
        if self.understeer_compensation:
            curvature = np.interp(s_query, rl.s, rl.kappa, period=L)
            beta_ref = (self.p.lr - vxr ** 2 / (self.model_mu * 9.81 * self.p.C_Sr)) * curvature
            # Raceline headings describe velocity direction; the MPC state is
            # body yaw. Match their frames using the same steady-state slip.
            psir -= np.where(vxr >= 0.5, beta_ref, 0.0)
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

        self.opti.set_value(self.p_a_lat_max, self.a_lat_max)
        self.opti.set_value(self.p_mu, self.model_mu)
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
            brake_v = max(state.v - 3.0, 0.0)
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
