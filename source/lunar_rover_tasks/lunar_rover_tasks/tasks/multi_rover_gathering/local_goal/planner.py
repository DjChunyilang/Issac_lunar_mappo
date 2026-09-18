"""acados nonlinear differential-drive MPC; no simulator/global-map access."""

import time
from pathlib import Path

import numpy as np

from .contracts import DT, N, PlannerFeedback, TimedTrajectory, wrap


def dynamics(pose, control, feature, slope_scale):
    v, w = control
    scale = np.clip(
        feature[4]
        * np.exp(-abs(feature[1] * np.cos(pose[2]) + feature[2] * np.sin(pose[2])) * slope_scale),
        0.25,
        1.0,
    )
    v, w = v * scale, w * scale
    mid = pose[2] + 0.5 * DT * w
    return pose + DT * np.array([v * np.cos(mid), v * np.sin(mid), w]), np.array([v, w])


def shifted_controls(previous):
    tail = previous.controls[-1]
    brake = tail + np.clip(-tail, [-0.3, -0.6], [0.3, 0.6])
    return np.vstack([previous.controls[1:], brake])


def build_solver(cache_dir, slope_scale, *, build=True):
    # Lazy optional dependencies: legacy route works without either package.
    import casadi as ca
    from acados_template import AcadosOcp, AcadosOcpSolver

    ocp = AcadosOcp()
    model = ocp.model
    model.name = "local_goal_v1"
    x = ca.SX.sym("x", 5)  # pose and previous wheel-equivalent controls
    u = ca.SX.sym("u", 2)
    p = ca.SX.sym("p", 22 + 3 * 128)  # goal/dynamics/continuity, neighbors, observed free disks
    model.x, model.u, model.p = x, u, p
    scale = ca.fmin(
        1.0,
        ca.fmax(
            0.25, p[6] * ca.exp(-ca.fabs(p[4] * ca.cos(x[2]) + p[5] * ca.sin(x[2])) * slope_scale)
        ),
    )
    v, w = u[0] * scale, u[1] * scale
    mid = x[2] + 0.5 * DT * w
    model.disc_dyn_expr = ca.vertcat(
        x[0] + DT * v * ca.cos(mid), x[1] + DT * v * ca.sin(mid), x[2] + DT * w, u
    )
    yaw = x[2] - p[2]
    terminal = ca.vertcat(x[:2] - p[:2], ca.sin(yaw), 1 - ca.cos(yaw))
    model.cost_y_expr = ca.vertcat(terminal, u, u - x[3:5], x[:2] - p[7:9])
    model.cost_y_expr_e = terminal
    ocp.cost.cost_type = ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag([2.0, 2.0, 0.3, 0.3, 0.05, 0.03, 0.5, 0.2, 0.15, 0.15])
    ocp.cost.W_e = np.diag([12.0, 12.0, 1.0, 1.0])
    ocp.cost.yref = np.zeros(10)
    ocp.cost.yref_e = np.zeros(4)
    # The same observed free-space union is used at EVERY time node. Binding a
    # node to its old warm-start disk would silently force the previous path.
    terrain = ca.mmax(
        ca.vertcat(
            *[
                p[24 + 3 * j] ** 2 - ca.sumsqr(x[:2] - p[22 + 3 * j : 24 + 3 * j])
                for j in range(128)
            ]
        )
    )
    neighbors = [
        ca.sumsqr(x[:2] - p[13 + 3 * j : 15 + 3 * j]) - p[15 + 3 * j] ** 2 for j in range(3)
    ]
    safety = ca.vertcat(terrain, *neighbors)
    wheel = ca.vertcat((u[0] - 0.188 * u[1]) / 0.098, (u[0] + 0.188 * u[1]) / 0.098)
    model.con_h_expr = ca.vertcat(wheel, u - x[3:5], safety)
    model.con_h_expr_0 = model.con_h_expr
    model.con_h_expr_e = safety
    ocp.constraints.lh = np.r_[[-18.0, -18.0, -0.3, -0.6], np.zeros(4)]
    ocp.constraints.uh = np.r_[[18.0, 18.0, 0.3, 0.6], np.full(4, 1e8)]
    ocp.constraints.lh_0 = ocp.constraints.lh.copy()
    ocp.constraints.uh_0 = ocp.constraints.uh.copy()
    ocp.constraints.lh_e = np.zeros(4)
    ocp.constraints.uh_e = np.full(4, 1e8)
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([-1.15, -2.4])
    ocp.constraints.ubu = np.array([1.15, 2.4])
    ocp.constraints.x0 = np.zeros(5)
    ocp.parameter_values = np.zeros(22 + 3 * 128)
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * DT
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    # At most two terrain relinearizations, six SQP iterations each.
    ocp.solver_options.nlp_solver_max_iter = 6
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.levenberg_marquardt = 1e-3
    ocp.solver_options.nlp_solver_tol_stat = 1e-3
    ocp.solver_options.nlp_solver_tol_eq = 1e-4
    ocp.solver_options.nlp_solver_tol_ineq = 1e-4
    ocp.solver_options.print_level = 0
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    ocp.code_export_directory = str(cache_dir / "code")
    return AcadosOcpSolver(
        ocp, json_file=str(cache_dir / "ocp.json"), generate=build, build=build, verbose=False
    )


class NMPCPlanner:
    def __init__(self, solver, slope_scale, deadline=0.15):
        self.solver, self.slope_scale, self.deadline = solver, slope_scale, deadline
        self.reset()

    def reset(self):
        self.previous = None
        self.blocked = 0.0
        self.solver.reset()

    def rollout(self, pose, controls, memory, now):
        states, velocities = [pose.copy()], []
        for u in controls:
            features, _, _ = memory.lookup(states[-1][:2])
            state, velocity = dynamics(states[-1], u, features[0], self.slope_scale)
            states.append(state)
            velocities.append(velocity)
        return TimedTrajectory(
            np.array(states),
            np.array(controls),
            np.array(velocities),
            now + np.arange(N + 1) * DT,
            now,
        )

    def validate(self, plan, memory, neighbors, cap, previous_control):
        if not all(
            np.isfinite(a).all() for a in (plan.states, plan.controls, plan.velocities, plan.times)
        ):
            return "nonfinite"
        u = plan.controls
        if (abs(u[:, 0]) > cap + 1e-4).any() or (abs(u[:, 1]) > 2.4001).any():
            return "speed_limit"
        if (
            abs(u[:, 0, None] + u[:, 1, None] * np.array([-0.188, 0.188])) > 18 * 0.098 + 1e-4
        ).any():
            return "wheel_limit"
        if (
            abs(np.diff(np.vstack([previous_control, u]), axis=0)) > np.array([0.3001, 0.6001])
        ).any():
            return "slew_limit"
        # Swept check at 0.025 s (not just shooting nodes), also for neighbor crossing.
        for alpha in np.linspace(0, 1, 9):
            points = (1 - alpha) * plan.states[:-1, :2] + alpha * plan.states[1:, :2]
            if not memory.safe(points).all():
                return "unknown_or_impassable"
            for neighbor in neighbors:
                peer = (1 - alpha) * neighbor.positions[:-1] + alpha * neighbor.positions[1:]
                margin = (1 - alpha) * neighbor.radii[:-1] + alpha * neighbor.radii[1:]
                # Add a sub-sampling Lipschitz allowance for relative displacement.
                sweep = (
                    np.linalg.norm(
                        np.diff(plan.states[:, :2], axis=0) - np.diff(neighbor.positions, axis=0),
                        axis=1,
                    )
                    / 16
                )
                if (np.linalg.norm(points - peer, axis=1) < margin + sweep - 1e-4).any():
                    return "neighbor_swept_collision"
        return ""

    def stopped_prefix(
        self, pose, controls, memory, neighbors, cap, previous_control, now, deadline
    ):
        """Retain optimized controls only; append a rate-limited, validated stop.

        This neither searches for a new goal nor silently substitutes a route.
        Validate the complete horizon including the stationary tail and peers.
        """
        for length in range(N - 1, 0, -1):
            if time.monotonic() >= deadline:
                break
            candidate = controls.copy()
            last = candidate[length - 1].copy()
            for k in range(length, N):
                last = last + np.clip(-last, [-0.3, -0.6], [0.3, 0.6])
                candidate[k] = last
            if np.max(abs(candidate[-1])) > 1e-6:
                continue
            trajectory = self.rollout(pose, candidate, memory, now)
            if not self.validate(trajectory, memory, neighbors, cap, previous_control):
                return trajectory, length
        return None, 0

    def configure(self, warm, pose, request, memory, neighbors, previous_control, cap):
        """Refresh terrain parameters from the latest control rollout, not stale shooting nodes."""
        initial = np.r_[pose, previous_control]
        free_disks = memory.free_disks(pose[:2]).reshape(-1)
        self.solver.constraints_set(0, "lbx", initial)
        self.solver.constraints_set(0, "ubx", initial)
        for k in range(N + 1):
            feature, _, _ = memory.lookup(warm.states[k, :2])
            center, radius = memory.local_ball(warm.states[k, :2])
            peer = (
                np.concatenate([np.r_[n.positions[k], n.radii[k]] for n in neighbors])
                if neighbors
                else np.empty(0)
            )
            peer = np.r_[peer, np.tile([1000.0, 1000.0, 0.0], 3 - len(neighbors))]
            parameters = np.r_[
                request.target,
                cap,
                feature[0, 1:3],
                feature[0, 4],
                warm.states[k],
                center,
                radius,
                peer,
                free_disks,
            ]
            self.solver.set(k, "p", parameters)
            self.solver.set(
                k, "x", np.r_[warm.states[k], previous_control if k == 0 else warm.controls[k - 1]]
            )
            if k < N:
                distance = np.linalg.norm(warm.states[k, :2] - request.target[:2])
                bound = min(cap, distance * 2) if distance < 0.1 else cap
                self.solver.constraints_set(k, "lbu", np.array([-bound, -2.4]))
                self.solver.constraints_set(k, "ubu", np.array([bound, 2.4]))
                self.solver.set(k, "u", warm.controls[k])

    def plan(self, pose, request, memory, neighbors, previous_control, now):
        start = time.monotonic()
        cap = 0.0 if request.speed_cap < 0.05 else request.speed_cap
        if self.previous is not None:
            controls = shifted_controls(self.previous)
        else:
            controls, state, last = [], pose.copy(), previous_control.copy()
            for _ in range(N):
                delta = request.target[:2] - state[:2]
                distance = np.linalg.norm(delta)
                angle = wrap(np.arctan2(delta[1], delta[0]) - state[2])
                direction = 1 if abs(angle) <= np.pi / 2 else -1
                if direction < 0:
                    angle = wrap(angle + np.pi)
                desired = np.array(
                    [
                        direction * min(cap, distance * 2) * max(0, np.cos(angle)),
                        np.clip(
                            2 * (wrap(request.target[2] - state[2]) if distance < 0.1 else angle),
                            -2.4,
                            2.4,
                        ),
                    ]
                )
                last = last + np.clip(desired - last, [-0.3, -0.6], [0.3, 0.6])
                features, _, _ = memory.lookup(state[:2])
                nxt, _ = dynamics(state, last, features[0], self.slope_scale)
                if not memory.safe(np.linspace(state[:2], nxt[:2], 9)).all():
                    last = np.zeros(2)
                    nxt = state.copy()
                controls.append(last.copy())
                state = nxt
            controls = np.array(controls)
        warm = self.rollout(pose, controls, memory, now)
        self.configure(warm, pose, request, memory, neighbors, previous_control, cap)
        solver_exception = False
        try:
            status = int(self.solver.solve())
        except (RuntimeError, FloatingPointError, ValueError):
            status = -1
            solver_exception = True
        elapsed = time.monotonic() - start
        feedback = PlannerFeedback(solve_seconds=elapsed, solver_status=status, solver_passes=1)
        reason = (
            "solver_exception"
            if solver_exception
            else "solver_timeout"
            if elapsed > self.deadline
            else "solver_failure"
            if status not in (0, 2)
            else ""
        )
        if not reason:
            try:
                candidate = np.stack([self.solver.get(k, "u") for k in range(N)])
                if not np.isfinite(candidate).all():
                    reason = "nonfinite"
                else:
                    result = self.rollout(pose, candidate, memory, now)
                    shooting = np.stack([self.solver.get(k, "x")[:3] for k in range(N + 1)])
                    if not np.isfinite(shooting).all():
                        raise FloatingPointError("Nonfinite shooting states")
                    feedback.model_error_m = float(
                        np.linalg.norm(shooting[:, :2] - result.states[:, :2], axis=1).max()
                    )
                    # One bounded terrain refresh. The exported trajectory always
                    # comes from identical local lookup/dynamics used by validation.
                    if (
                        feedback.model_error_m > 0.01
                        and time.monotonic() - start < self.deadline * 0.5
                    ):
                        self.configure(
                            result, pose, request, memory, neighbors, previous_control, cap
                        )
                        status = int(self.solver.solve())
                        feedback.solver_status = status
                        feedback.solver_passes += 1
                        if status not in (0, 2):
                            reason = "solver_failure"
                        else:
                            candidate = np.stack([self.solver.get(k, "u") for k in range(N)])
                            if not np.isfinite(candidate).all():
                                reason = "nonfinite"
                            else:
                                result = self.rollout(pose, candidate, memory, now)
                                shooting = np.stack(
                                    [self.solver.get(k, "x")[:3] for k in range(N + 1)]
                                )
                                if not np.isfinite(shooting).all():
                                    raise FloatingPointError("Nonfinite shooting states")
                                feedback.model_error_m = float(
                                    np.linalg.norm(
                                        shooting[:, :2] - result.states[:, :2], axis=1
                                    ).max()
                                )
                    # Re-rollout with local terrain instead of trusting shooting equality tolerances.
                    if not reason:
                        reason = self.validate(result, memory, neighbors, cap, previous_control)
                        feedback.candidate_rejection = reason
                        if reason:
                            prefix, length = self.stopped_prefix(
                                pose,
                                candidate,
                                memory,
                                neighbors,
                                cap,
                                previous_control,
                                now,
                                start + self.deadline,
                            )
                            if prefix is not None:
                                result, feedback.prefix_steps, reason = prefix, length, ""
            except (RuntimeError, FloatingPointError, ValueError):
                reason = "solver_output_error"
            if time.monotonic() - start > self.deadline:
                reason = "solver_timeout"
        if not reason:
            self.previous = result
            translation_requested = (
                cap >= 0.05 and np.linalg.norm(request.target[:2] - pose[:2]) >= 0.1
            )
            self.blocked = (
                self.blocked + DT
                if translation_requested and np.linalg.norm(result.states[1, :2] - pose[:2]) < 1e-3
                else 0.0
            )
            feedback.status = (
                "feasible_prefix"
                if feedback.prefix_steps
                else "feasible"
                if status == 0
                else "feasible_iteration_limit"
            )
        else:
            feedback.reason = reason
            feedback.prefix_steps = 0
            self.blocked += DT
            if self.previous is not None:
                shifted = self.rollout(pose, shifted_controls(self.previous), memory, now)
                if not self.validate(shifted, memory, neighbors, cap, previous_control):
                    result = shifted
                    feedback.status, feedback.used_previous = "previous", True
                else:
                    result = None
            else:
                result = None
            if result is None:
                brake, last = [], previous_control.copy()
                for _ in range(N):
                    last = last + np.clip(-last, [-0.3, -0.6], [0.3, 0.6])
                    brake.append(last.copy())
                result = self.rollout(pose, np.array(brake), memory, now)
                feedback.status, feedback.emergency_brake = "brake", True
                # A limited brake is a last resort, NOT a certified feasible plan.
            self.previous = None if feedback.emergency_brake else result
        feedback.blocked_seconds = self.blocked
        feedback.solve_seconds = time.monotonic() - start
        return result, feedback


def track_trajectory(pose, trajectory):
    """Generic signed feed-forward plus pose feedback, independent of primitives."""
    error = trajectory.states[0] - pose
    v, w = trajectory.controls[0]
    forward = error[0] * np.cos(pose[2]) + error[1] * np.sin(pose[2])
    lateral = -error[0] * np.sin(pose[2]) + error[1] * np.cos(pose[2])
    return np.array([v + forward, w + 2 * wrap(error[2]) + np.sign(v) * lateral])
