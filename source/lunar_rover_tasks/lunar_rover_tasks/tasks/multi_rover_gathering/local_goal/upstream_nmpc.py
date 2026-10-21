"""Independent 40-step acados model for the immutable upstream proxy profile.

The shooting model proposes actuator commands. The public upstream planning
adapter re-rolls those commands through local lookup and independently reviews
body/contact terrain, stop reserve and identified neighbors before execution.
No global terrain or simulator handle is accepted here.
"""

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from .contracts import wrap
from .upstream_execution import (
    brake_sequence, limit_control_numpy, midpoint_numpy, rollout_numpy,
    terrain_gain_numpy,
)
from .upstream_spec import canonical, digest

NMPC_VERSION = "nmpc_upstream_v1"
MODEL_VERSION = "midpoint_directional_corridor_v1"
DISKS_PER_STAGE = 24
MAX_NEIGHBORS = 3
PARAMETER_SIZE = 11 + 3 * MAX_NEIGHBORS + 3 * DISKS_PER_STAGE + 3


def solver_identity(spec):
    return {"adapter": NMPC_VERSION, "model": MODEL_VERSION,
            "profile_fingerprint": spec.fingerprint,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "steps": spec.grid.steps, "dt": spec.grid.dt,
            "stage_support_disks": DISKS_PER_STAGE}


def build_upstream_ocp(spec):
    """Construct the actual constrained model; imports remain optional/lazy."""
    import casadi as ca
    from acados_template import AcadosOcp

    ocp = AcadosOcp(); model = ocp.model
    vehicle = spec.vehicle; dt = spec.grid.dt
    model.name = "upstream_" + digest(solver_identity(spec))[:12]
    x = ca.SX.sym("x", 5); u = ca.SX.sym("u", 2)
    p = ca.SX.sym("p", PARAMETER_SIZE)
    model.x, model.u, model.p = x, u, p
    # p: goal xyz, speed cap, feature[5], reference xy, identified peers[3x3], disks.
    gain = ca.fmin(1., ca.fmax(.25, p[8] * ca.exp(-ca.fabs(
        p[5] * ca.cos(x[2]) + p[6] * ca.sin(x[2])) * spec.terrain.slope_speed_scale)))
    actual = u * gain
    middle = x[2] + .5 * dt * actual[1]
    model.disc_dyn_expr = ca.vertcat(x[0] + dt * actual[0] * ca.cos(middle),
        x[1] + dt * actual[0] * ca.sin(middle), x[2] + dt * actual[1], u)
    goal = (x[:2] - p[:2]) / spec.rho_max
    yaw = x[2] - p[2]
    terminal = ca.vertcat(goal, ca.sin(yaw), 1 - ca.cos(yaw))
    control_scale = ca.DM([vehicle.forward_limit, vehicle.yaw_limit])
    rate_scale = ca.DM([vehicle.linear_acceleration * dt, vehicle.angular_acceleration * dt])
    model.cost_y_expr = ca.vertcat(terminal, u / control_scale, (u - x[3:5]) / rate_scale,
                                  (x[:2] - p[9:11]) / spec.rho_max)
    model.cost_y_expr_e = terminal
    ocp.cost.cost_type = ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag([2., 2., .3, .3, .05, .03, .01, .01, .15, .15])
    ocp.cost.W_e = np.diag([12., 12., 1., 1.])
    ocp.cost.yref = np.zeros(10); ocp.cost.yref_e = np.zeros(4)

    half = vehicle.effective_track / 2
    wheel = ca.vertcat((u[0] - half * u[1]) / vehicle.wheel_radius,
                      (u[0] + half * u[1]) / vehicle.wheel_radius)
    prior_wheel = ca.vertcat((x[3] - half * x[4]) / vehicle.wheel_radius,
                            (x[3] + half * x[4]) / vehicle.wheel_radius)
    lower = []; upper = []
    for i, (acc, dec) in enumerate(((vehicle.linear_acceleration, vehicle.linear_deceleration),
                                    (vehicle.angular_acceleration, vehicle.angular_deceleration))):
        prior = x[3 + i]
        # A physical direction switch passes exactly through zero. Freeze the
        # feasible warm-start mode within this SQP subproblem, rather than
        # differentiating a discontinuous permission-to-reverse at x_prev=0.
        mode = p[PARAMETER_SIZE - 3 + i]
        lower.append(ca.if_else(mode > 0., ca.fmax(0., prior - dec * dt), prior - acc * dt))
        upper.append(ca.if_else(mode < 0., ca.fmin(0., prior + dec * dt), prior + acc * dt))
    yaw_cap = ca.if_else(p[PARAMETER_SIZE - 1] == 0.,
                        vehicle.low_speed_yaw_limit, vehicle.yaw_limit)
    actuator = ca.vertcat(wheel, wheel - prior_wheel, u - ca.vertcat(*lower),
                         ca.vertcat(*upper) - u, yaw_cap - u[1], yaw_cap + u[1],
                         p[3] - u[0], p[3] + u[0])
    start = 11 + 3 * MAX_NEIGHBORS
    support = ca.mmax(ca.vertcat(*[p[start + 3*j + 2]**2 - ca.sumsqr(
        x[:2] - p[start + 3*j:start + 3*j + 2]) for j in range(DISKS_PER_STAGE)]))
    peers = [ca.sumsqr(x[:2] - p[11 + 3*j:13 + 3*j]) - p[13 + 3*j]**2
             for j in range(MAX_NEIGHBORS)]
    safety = ca.vertcat(support, *peers)
    model.con_h_expr = ca.vertcat(actuator, safety)
    model.con_h_expr_0 = model.con_h_expr
    model.con_h_expr_e = safety
    lower_h = np.r_[[-vehicle.wheel_limit]*2, [-vehicle.wheel_acceleration * dt]*2,
                    np.zeros(8 + 1 + MAX_NEIGHBORS)]
    upper_h = np.r_[[vehicle.wheel_limit]*2, [vehicle.wheel_acceleration * dt]*2,
                    np.full(8 + 1 + MAX_NEIGHBORS, 1e8)]
    ocp.constraints.lh = lower_h; ocp.constraints.uh = upper_h
    ocp.constraints.lh_0 = lower_h.copy(); ocp.constraints.uh_0 = upper_h.copy()
    ocp.constraints.lh_e = np.zeros(1 + MAX_NEIGHBORS)
    ocp.constraints.uh_e = np.full(1 + MAX_NEIGHBORS, 1e8)
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([-vehicle.reverse_limit, -vehicle.yaw_limit])
    ocp.constraints.ubu = np.array([vehicle.forward_limit, vehicle.yaw_limit])
    ocp.constraints.idxbx = np.array([3, 4])
    ocp.constraints.lbx = np.array([-vehicle.reverse_limit, -vehicle.yaw_limit])
    ocp.constraints.ubx = np.array([vehicle.forward_limit, vehicle.yaw_limit])
    ocp.constraints.x0 = np.zeros(5); ocp.parameter_values = np.zeros(PARAMETER_SIZE)
    options = ocp.solver_options
    options.N_horizon = spec.grid.steps; options.tf = spec.grid.horizon
    options.integrator_type = "DISCRETE"; options.nlp_solver_type = "SQP"
    options.nlp_solver_max_iter = 16
    options.qp_solver = "PARTIAL_CONDENSING_HPIPM"; options.hpipm_mode = "ROBUST"
    options.hessian_approx = "GAUSS_NEWTON"; options.levenberg_marquardt = 1e-3
    options.globalization = "MERIT_BACKTRACKING"
    options.globalization_alpha_min = .01; options.globalization_alpha_reduction = .5
    options.nlp_solver_tol_stat = 1e-3; options.nlp_solver_tol_eq = 1e-7
    options.nlp_solver_tol_ineq = 1e-7; options.print_level = 0
    return ocp


def build_upstream_solver(spec, cache_dir, *, build=True):
    from acados_template import AcadosOcpSolver

    identity = solver_identity(spec); cache = Path(cache_dir).resolve() / digest(identity)
    metadata = cache / "upstream_model_identity.json"
    if metadata.exists() and json.loads(metadata.read_text()) != identity:
        raise ValueError("Upstream acados cache identity mismatch")
    if not build and not metadata.exists():
        raise ValueError("Missing upstream acados cache identity")
    cache.mkdir(parents=True, exist_ok=True)
    ocp = build_upstream_ocp(spec); ocp.code_export_directory = str(cache / "code")
    solver = AcadosOcpSolver(ocp, json_file=str(cache / "ocp.json"),
                             generate=build, build=build, verbose=False)
    metadata.write_text(canonical(identity) + "\n")
    return solver


@dataclass(frozen=True)
class CorridorSupport:
    """Detached observed supports covering warm/reference and complete braking."""
    xy: np.ndarray
    radii: np.ndarray
    reference_nodes: int
    stop_nodes: int
    warm_known: bool
    stop_known: bool

    def stage_disks(self, point):
        disks = np.tile([10000., 10000., 0.], (DISKS_PER_STAGE, 1))
        if len(self.xy):
            _, indices = cKDTree(self.xy).query(point, k=min(DISKS_PER_STAGE, len(self.xy)))
            indices = np.atleast_1d(indices)
            disks[:len(indices)] = np.c_[self.xy[indices], self.radii[indices]]
        return disks


def corridor_support(inp, warm, stopping):
    """Select around the entire rolling reference and stop corridor, never ego-only.

    These disks constrain the center during optimization; full rectangle/contact
    support remains an independent output gate. No vehicle radius is subtracted
    from an individual 0.155 m observation support disk.
    """
    snapshot = inp.terrain.snapshot(); spec = inp.spec
    if not len(snapshot.xy):
        return CorridorSupport(np.empty((0, 2)), np.empty(0), len(warm.states),
                               len(stopping.states), False, False)
    samples = np.vstack([warm.states[:, :2], stopping.states[:, :2]])
    # Include supports over the oriented body corridor conservatively by its
    # circumscribed radius; this selection itself does not certify footprint.
    tree = cKDTree(samples)
    distance, _ = tree.query(snapshot.xy)
    f = snapshot.features
    good = ((distance <= spec.vehicle.effective_radius + snapshot.support_radius)
            & (np.linalg.norm(f[:, 1:3], axis=-1) <= spec.terrain.driving_slope)
            & (f[:, 4] >= spec.terrain.minimum_traversability))
    xy = snapshot.xy[good].copy()
    _, warm_known = inp.terrain.lookup(warm.states[:, :2])
    _, stop_known = inp.terrain.lookup(stopping.states[:, :2])
    return CorridorSupport(xy, np.full(len(xy), snapshot.support_radius),
                           len(warm.states), len(stopping.states), bool(warm_known.all()),
                           bool(stop_known.all()))


def local_rollout(inp, controls):
    def gain(pose):
        feature, known = inp.terrain.lookup(pose[:2])
        return terrain_gain_numpy(pose, feature[0], inp.spec) if known[0] else .25
    return rollout_numpy(inp.ego_pose, controls, inp.spec, inp.now, gain=gain)


def guided_controls(inp, yaw, cap):
    spec = inp.spec; state = np.array(inp.ego_pose, dtype=float); prior = inp.previous_control.copy()
    controls = []
    for _ in range(spec.grid.steps):
        delta = inp.intent.goal_xy - state[:2]; distance = np.linalg.norm(delta)
        angle = wrap(np.arctan2(delta[1], delta[0]) - state[2]) if distance > 1e-9 else 0.
        direction = 1. if abs(angle) <= np.pi / 2 else -1.
        if direction < 0:angle = wrap(angle + np.pi)
        speed_limit = cap if direction > 0 else min(cap, spec.vehicle.reverse_limit)
        # Stop-at-goal is explicit. A rolling target still receives a gentle
        # approach profile but is not treated as a mandatory 4 m stop wall.
        speed = min(speed_limit, np.sqrt(2 * spec.vehicle.linear_deceleration * distance))
        heading_error = wrap(yaw - state[2]) if distance < .15 else angle
        desired = [direction * speed * max(0., np.cos(angle)), 1.2 * heading_error]
        prior = limit_control_numpy(desired, prior, spec)
        feature, known = inp.terrain.lookup(state[:2])
        gain = terrain_gain_numpy(state, feature[0], spec) if known[0] else .25
        state, _ = midpoint_numpy(state, prior, spec, gain=gain)
        controls.append(prior.copy())
    return np.asarray(controls)


class UpstreamNmpcSolver:
    """Bounded two-pass solve with explicit deterministic recovery seeds."""

    def __init__(self, spec, cache_dir=None, *, solver=None, deadline=10.):
        self.spec = spec
        if not np.isfinite(deadline) or deadline <= 0:raise ValueError("Invalid NMPC deadline")
        self.deadline = float(deadline)
        if solver is None and cache_dir is None:raise ValueError("NMPC requires an explicit run cache directory")
        self.construction_seconds = 0.
        if solver is None:
            started = time.monotonic()
            self.solver = build_upstream_solver(spec, cache_dir)
            self.construction_seconds = time.monotonic() - started
        else:
            self.solver = solver

    def reset(self):
        self.solver.reset()

    def configure(self, inp, warm, yaw, cap, support):
        x0 = np.r_[inp.ego_pose, inp.previous_control]
        self.solver.constraints_set(0, "lbx", x0); self.solver.constraints_set(0, "ubx", x0)
        if len(inp.neighbors) > MAX_NEIGHBORS:raise ValueError("Upstream NMPC supports three identified neighbors")
        for k in range(self.spec.grid.steps + 1):
            feature, _ = inp.terrain.lookup(warm.states[k, :2])
            peers = np.tile([1000., 1000., 0.], (MAX_NEIGHBORS, 1))
            for j, neighbor in enumerate(inp.neighbors):
                position, radius = neighbor.at(inp.times[k]); peers[j] = [*position, radius]
            prior = inp.previous_control if k == 0 else warm.controls[k - 1]
            direction = np.sign(np.where(abs(prior) < 1e-10, 0., prior))
            command = warm.controls[min(k, self.spec.grid.steps - 1)]
            yaw_mode = 0. if abs(command[0]) <= self.spec.vehicle.low_speed_threshold else np.sign(command[0])
            parameters = np.r_[inp.intent.goal_xy, yaw, cap, feature[0], warm.states[k, :2],
                               peers.ravel(), support.stage_disks(warm.states[k, :2]).ravel(),
                               direction, yaw_mode]
            self.solver.set(k, "p", parameters)
            self.solver.set(k, "x", np.r_[warm.states[k], inp.previous_control if k == 0 else warm.controls[k - 1]])
            if k < self.spec.grid.steps:
                v = self.spec.vehicle
                lower = np.array([-v.reverse_limit, -v.yaw_limit])
                upper = np.array([v.forward_limit, v.yaw_limit])
                if yaw_mode == 0:
                    lower = np.array([-v.low_speed_threshold, -v.low_speed_yaw_limit])
                    upper = np.array([v.low_speed_threshold, v.low_speed_yaw_limit])
                elif yaw_mode > 0:lower[0] = v.low_speed_threshold + 1e-9
                else:upper[0] = -v.low_speed_threshold - 1e-9
                self.solver.constraints_set(k, "lbu", lower); self.solver.constraints_set(k, "ubu", upper)
                if k > 0:
                    low = np.where(direction < 0, [-v.reverse_limit, -v.yaw_limit], 0.)
                    high = np.where(direction > 0, [v.forward_limit, v.yaw_limit], 0.)
                    self.solver.constraints_set(k, "lbx", low); self.solver.constraints_set(k, "ubx", high)
                self.solver.set(k, "u", warm.controls[k])

    def solve(self, inp, yaw, cap, initial=None):
        # Reconstruct all solver-native memory each call; the public shifted
        # controls are the sole warm start, including after snapshot recovery.
        self.solver.reset()
        start = time.monotonic(); diagnostics = {"model": MODEL_VERSION, "solver_passes": 0}
        seed = guided_controls(inp, yaw, cap) if initial is None else np.array(initial, dtype=float)
        warm = local_rollout(inp, seed)
        stop = local_rollout(inp, brake_sequence(inp.previous_control, self.spec))
        result = None; reason = "solver_failure"; status = -1
        for attempt in range(2):
            support = corridor_support(inp, warm, stop)
            diagnostics.update(support_samples=len(support.xy), support_reference_nodes=support.reference_nodes,
                               support_stop_nodes=support.stop_nodes, support_warm_known=support.warm_known,
                               support_stop_known=support.stop_known)
            if not len(support.xy):reason = "unknown_terrain_support"; break
            self.configure(inp, warm, yaw, cap, support)
            try:
                status = int(self.solver.solve()); diagnostics["solver_passes"] += 1
                raw = np.stack([self.solver.get(k, "u") for k in range(self.spec.grid.steps)])
                shooting = np.stack([self.solver.get(k, "x")[:3] for k in range(self.spec.grid.steps + 1)])
            except (RuntimeError, ValueError, FloatingPointError):
                reason = "solver_exception"; break
            if time.monotonic() - start > self.deadline:reason = "solver_timeout"; break
            if not np.isfinite(raw).all() or not np.isfinite(shooting).all():reason = "nonfinite_solver_output"; break
            # The finite SQP tolerance is never permission to exceed actuator
            # bounds: return commands passed through the common exact projector.
            prior = inp.previous_control.copy(); controls = []
            for command in raw:
                prior = limit_control_numpy(command, prior, self.spec); controls.append(prior.copy())
            controls = np.asarray(controls); candidate = local_rollout(inp, controls)
            error = float(np.linalg.norm(shooting[:, :2] - candidate.states[:, :2], axis=-1).max())
            diagnostics.update(solver_status=status, model_error_m=error,
                               maximum_actuator_correction=float(np.abs(raw - controls).max()))
            if status in (0, 2):
                result = controls; reason = ""
                if error <= .01:break
                warm = candidate
            else:
                reason = "solver_failure"
                # One fresh deterministic goal seed and reset, not an unbounded
                # retry or a reuse of solver-native objects from another run.
                self.solver.reset(); warm = local_rollout(inp, guided_controls(inp, yaw, cap))
        diagnostics.update(solver_seconds=time.monotonic() - start, solver_status=status,
                           fallback_reason=reason)
        return result if not reason else None, diagnostics


# Importing this adapter never imports acados; only constructing its solver does.
from .upstream_planning import Candidate, PlanningAdapter


class UpstreamNmpcAdapter(PlanningAdapter):
    adapter_id = NMPC_VERSION
    state_version = 1

    def __init__(self, cfg, *, cache_dir=None, solver=None, deadline=10.):
        self._frozen = None
        self.engine = UpstreamNmpcSolver(cfg.spec, cache_dir, solver=solver, deadline=deadline)
        super().__init__(cfg)

    def _reset_adapter(self):
        self._frozen = None
        self.engine.reset()

    def _resolve(self, inp):
        intent = inp.intent
        key = digest({"goal": np.asarray(intent.goal_xy).tolist(), "hold": bool(intent.hold),
                      "timestamp": float(intent.timestamp), "expires_at": float(intent.expires_at),
                      "terminal_yaw": None if intent.terminal_yaw is None else float(intent.terminal_yaw),
                      "speed_cap": None if intent.speed_cap is None else float(intent.speed_cap)})
        if self._frozen is None or self._frozen["key"] != key:
            delta = intent.goal_xy - inp.ego_pose[:2]
            heading = float(inp.ego_pose[2]) if np.linalg.norm(delta) <= 1e-10 else float(np.arctan2(delta[1], delta[0]))
            if abs(wrap(heading - inp.ego_pose[2])) > np.pi / 2:heading = float(wrap(heading + np.pi))
            self._frozen = {"key": key,
                            "yaw": heading if intent.terminal_yaw is None else float(intent.terminal_yaw),
                            "cap": self.spec.vehicle.forward_limit if intent.speed_cap is None else float(intent.speed_cap)}
        return self._frozen["yaw"], self._frozen["cap"]

    def candidate_controls(self, inp):
        yaw, cap = self._resolve(inp)
        initial = None
        previous = self._previous_trajectory
        if previous is not None:
            elapsed = (inp.now - previous.times[0]) / self.spec.grid.dt
            shift = max(0, min(self.spec.grid.steps, int(round(elapsed))))
            if abs(elapsed - round(elapsed)) <= 1e-8:
                tail = brake_sequence(previous.controls[-1], self.spec)
                raw = np.vstack([previous.controls[shift:], tail, np.zeros((self.spec.grid.steps, 2))])[:self.spec.grid.steps]
                last = inp.previous_control.copy(); initial = []
                for command in raw:
                    last = limit_control_numpy(command, last, self.spec); initial.append(last.copy())
                initial = np.asarray(initial)
        controls, diagnostics = self.engine.solve(inp, yaw, cap, initial=initial)
        diagnostics.update(frozen_terminal_yaw=yaw, frozen_speed_cap=cap,
                           solver_construction_seconds=self.engine.construction_seconds,
                           solver_cache_fingerprint=digest(solver_identity(self.spec)))
        self.last_solver_diagnostics = diagnostics
        if controls is None:return []
        return [Candidate(controls, "nmpc", total_cost=None, diagnostics=diagnostics)]

    def plan(self, inp):
        self.last_solver_diagnostics = None
        output = super().plan(inp)
        detail = self.last_solver_diagnostics
        if detail is not None:
            output.diagnostics.update(detail)
            output.feedback.solver_status = detail.get("solver_status", -1)
            output.feedback.solver_passes = detail.get("solver_passes", 0)
            output.feedback.model_error_m = detail.get("model_error_m", 0.)
            # Keep the actual solver failure when the common layer had no
            # candidate. A valid rechecked old plan remains a valid fallback.
            if not output.valid and detail.get("fallback_reason"):
                output.feedback.reason = detail["fallback_reason"]
                output = replace(output, fallback_reason=detail["fallback_reason"])
        return output

    def _adapter_state(self):
        return {"frozen_constraints": None if self._frozen is None else dict(self._frozen),
                "model_identity": solver_identity(self.spec)}

    def _validate_adapter_state(self, state):
        if not isinstance(state, dict) or state.get("model_identity") != solver_identity(self.spec):
            raise ValueError("Upstream NMPC model/state mismatch")
        frozen = state.get("frozen_constraints")
        if frozen is not None:
            if (not isinstance(frozen, dict) or set(frozen) != {"key", "yaw", "cap"}
                    or not isinstance(frozen["key"], str) or len(frozen["key"]) != 64
                    or not np.isfinite([frozen["yaw"], frozen["cap"]]).all()
                    or not 0 <= frozen["cap"] <= self.spec.vehicle.forward_limit):
                raise ValueError("Invalid upstream NMPC frozen constraints")

    def _load_adapter_state(self, state):
        self._validate_adapter_state(state)
        frozen = state.get("frozen_constraints")
        self._frozen = None if frozen is None else dict(frozen)
        # Native SQP state is deliberately absent from the snapshot. Every solve
        # is configured from public physical states and reproducible controls.
        self.engine.reset()
