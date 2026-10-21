"""Versioned local discrete-CBF/QP filter; finite-point proxy checks only.

No simulator, global terrain, or uncommunicated peer state is accepted here.
Safety constraints have no slack. A failed brake is explicitly unverified.
"""

from dataclasses import dataclass, asdict
import time

import numpy as np
from scipy import sparse

from .contracts import DT, N
from .execution import dynamics, brake_controls, rollout, validate_trajectory, limit_control
from .protocol import ExecutionDecision, frozen_array, PlannerInput

VERSION = "discrete_cbf_qp_v1"
STATE_VERSION = 1


def execution_cap(item):
    # A zero HOLD target is not an instantaneous actuator speed limit.
    if item.intent.hold or item.now > item.intent.expires_at or item.intent.speed_cap is None:
        return 1.15
    return item.intent.speed_cap


@dataclass(frozen=True)
class SafetyInput:
    planner_input: object
    workspace_A: np.ndarray | None = None
    workspace_b: np.ndarray | None = None

    def __post_init__(self):
        if type(self.planner_input) is not PlannerInput:
            raise ValueError("SafetyInput requires detached PlannerInput")
        if (self.workspace_A is None) != (self.workspace_b is None):
            raise ValueError("Incomplete workspace boundary")
        if self.workspace_A is not None:
            A, b = np.asarray(self.workspace_A), np.asarray(self.workspace_b)
            if A.ndim != 2 or A.shape[1] != 2 or b.shape != (len(A),) or not (
                np.isfinite(A).all() and np.isfinite(b).all()
            ):
                raise ValueError("Invalid workspace boundary")
            lengths = np.linalg.norm(A, axis=1)
            if (lengths <= 0).any():
                raise ValueError("Invalid workspace plane normal")
            A, b = A / lengths[:, None], b / lengths
            object.__setattr__(self, "workspace_A", frozen_array(A))
            object.__setattr__(self, "workspace_b", frozen_array(b))


def position_jacobian(pose, control, feature, slope_scale):
    """Exact derivative of the existing midpoint model at fixed local terrain."""
    v, w = control
    scale = np.clip(feature[4] * np.exp(-abs(
        feature[1] * np.cos(pose[2]) + feature[2] * np.sin(pose[2])
    ) * slope_scale), .25, 1.)
    angle = pose[2] + .5 * DT * w * scale
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[DT * scale * c, -.5 * DT**2 * v * scale**2 * s],
                     [DT * scale * s, .5 * DT**2 * v * scale**2 * c]])


class SafetyProjector:
    def __init__(self, slope_scale=.75, *, deadline_s=None):
        import osqp
        if osqp.__version__ != "1.0.4":
            raise RuntimeError("Safety projection requires osqp==1.0.4")
        self.slope_scale = float(slope_scale)
        self.deadline_s = deadline_s
        if not np.isfinite(self.slope_scale) or (deadline_s is not None and (
            not np.isfinite(deadline_s) or deadline_s < 0
        )):
            raise ValueError("Invalid safety model/deadline")
        self.reset()

    def reset(self):
        self.last_control = None

    def state_dict(self):
        return {"version": VERSION, "state_version": STATE_VERSION,
                "slope_scale": self.slope_scale, "deadline_s": self.deadline_s,
                "last_control": None if self.last_control is None else self.last_control.copy()}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if any(state.get(k) != expected[k] for k in (
            "version", "state_version", "slope_scale", "deadline_s"
        )):
            raise ValueError("Safety snapshot version/config mismatch")
        value = state.get("last_control")
        if value is not None and (np.shape(value) != (2,) or not np.isfinite(value).all()):
            raise ValueError("Invalid safety warm control")
        self.last_control = None if value is None else np.asarray(value, dtype=float).copy()

    def _constraints(self, context, reference, nominal):
        item = context.planner_input
        terrain = item.terrain.conservative()
        feature = terrain.lookup(item.ego_pose[:2])[0][0]
        end = dynamics(item.ego_pose, reference, feature, self.slope_scale)[0][:2]
        jac = position_jacobian(item.ego_pose, reference, feature, self.slope_scale)
        A, upper, ids = [], [], []

        def inequality(row, bound, name):
            A.append(row); upper.append(bound); ids.append(name)

        cap = execution_cap(item)
        lo = np.maximum([-cap, -2.4], item.previous_control - [.3, .6])
        hi = np.minimum([cap, 2.4], item.previous_control + [.3, .6])
        if item.intent.hold or item.now > item.intent.expires_at:
            lo = np.maximum(lo, np.minimum(item.previous_control, 0))
            hi = np.minimum(hi, np.maximum(item.previous_control, 0))
        for axis in range(2):
            row = np.eye(2)[axis]
            inequality(row, hi[axis], f"actuator_upper:{axis}")
            inequality(-row, -lo[axis], f"actuator_lower:{axis}")
        for side in (-.188, .188):
            row = np.array([1., side])
            inequality(row, 18*.098, "wheel_upper")
            inequality(-row, 18*.098, "wheel_lower")
        disks = terrain.free_disks(item.ego_pose[:2])
        disks = disks[disks[:, 2] > 0]
        if not len(disks):
            raise ValueError("no_local_terrain_support")
        for fraction in np.linspace(0, 1, 9):
            point = item.ego_pose[:2] + fraction*(end-item.ego_pose[:2])
            J = fraction*jac
            values = disks[:, 2]**2 - np.sum((point-disks[:, :2])**2, axis=1)
            index = int(np.argmax(values))
            offset = point-disks[index, :2]
            gradient = -2*offset @ J
            # h(reference)+grad*(u-reference)>=0
            inequality(-gradient, values[index]-gradient@reference,
                       f"terrain:{fraction:.3f}:{index}")
            for neighbor in item.neighbors:
                p = neighbor.prediction
                peer = (1-fraction)*p.positions[0]+fraction*p.positions[1]
                radius = (1-fraction)*p.radii[0]+fraction*p.radii[1]
                relative = point-peer
                value = relative@relative-radius**2
                gradient = 2*relative@J
                # End-node discrete CBF; swept nodes require nonnegative h.
                initial = np.sum((item.ego_pose[:2]-p.positions[0])**2)-p.radii[0]**2
                minimum = max(0., .8*initial) if fraction == 1 else 0.
                inequality(-gradient, value-minimum-gradient@reference,
                           f"neighbor:{neighbor.sender_id}:{fraction:.3f}")
            if context.workspace_A is not None:
                for row, bound in zip(context.workspace_A, context.workspace_b):
                    inequality(row@J, bound-.14-row@point+row@J@reference,
                               f"workspace:{fraction:.3f}")
        return np.asarray(A), np.asarray(upper), ids

    def _verify(self, context, control, *, cbf=True):
        item = context.planner_input
        cap = execution_cap(item)
        if np.max(abs(limit_control(control, item.previous_control, cap)-control)) > 1e-5:
            return "actuator_limit"
        view = item.terrain.conservative()
        feature = view.lookup(item.ego_pose[:2])[0][0]
        end = dynamics(item.ego_pose, control, feature, self.slope_scale)[0][:2]
        points = item.ego_pose[:2]+np.linspace(0, 1, 9)[:, None]*(end-item.ego_pose[:2])
        if not view.safe(points).all():
            return "unknown_or_impassable"
        if context.workspace_A is not None and (
            points@context.workspace_A.T > context.workspace_b-.14+1e-6
        ).any():
            return "workspace_boundary"
        for neighbor in item.neighbors:
            p = neighbor.prediction
            f = np.linspace(0, 1, 9)
            peers = p.positions[0]+f[:, None]*(p.positions[1]-p.positions[0])
            radii = p.radii[0]+f*(p.radii[1]-p.radii[0])
            sweep = np.linalg.norm((end-item.ego_pose[:2])-(p.positions[1]-p.positions[0]))/16
            if (np.linalg.norm(points-peers, axis=1) < radii+sweep-1e-6).any():
                return f"neighbor_swept_collision:{neighbor.sender_id}"
            initial = np.sum((item.ego_pose[:2]-p.positions[0])**2)-p.radii[0]**2
            final = np.sum((end-p.positions[1])**2)-p.radii[1]**2
            if cbf and final < .8*initial-1e-6:
                return f"cbf_recheck:{neighbor.sender_id}"
        return ""

    def execution_prediction(self, context, control):
        item = context.planner_input
        controls = np.vstack([control, brake_controls(control)[:N-1]])
        trajectory = rollout(item.ego_pose, controls, item.terrain.conservative(),
                             item.now, self.slope_scale)
        reason = validate_trajectory(trajectory, item.terrain.conservative(),
            [n.prediction for n in item.neighbors],
            execution_cap(item),
            item.previous_control)
        if context.workspace_A is not None and (
            trajectory.states[:, :2]@context.workspace_A.T > context.workspace_b-.14+1e-6
        ).any():
            reason = "workspace_boundary"
        return None if reason else trajectory

    def project(self, context, nominal):
        import osqp
        started = time.monotonic()
        item = context.planner_input
        item.validate()
        nominal = np.asarray(nominal, dtype=float)
        if nominal.shape != (2,) or not np.isfinite(nominal).all():
            raise ValueError("state_estimation_invalid: nonfinite nominal control")
        deadline = float("inf") if self.deadline_s is None else started+self.deadline_s
        reason, qp_status, residual, ids = "", "not_needed", None, []
        expired = item.now > item.intent.expires_at
        reference = limit_control(nominal, item.previous_control,
            execution_cap(item))
        accepted = False
        if not item.intent.hold and not expired:
            for _ in range(3):
                if time.monotonic() >= deadline:
                    reason = "safety_timeout"; break
                try:
                    A, upper, ids = self._constraints(context, reference, nominal)
                except ValueError as error:
                    reason = str(error); break
                solver = osqp.OSQP()
                weights = 1/np.array([.3, .6])**2
                try:
                    solver.setup(P=sparse.diags(weights, format="csc"), q=-weights*nominal,
                        A=sparse.csc_matrix(A), l=np.full(len(upper), -np.inf), u=upper,
                        eps_abs=1e-6, eps_rel=1e-6, max_iter=4000,
                        adaptive_rho=False, polishing=True, verbose=False)
                    if self.last_control is not None:
                        solver.warm_start(x=self.last_control)
                    result = solver.solve(raise_error=False)
                except (osqp.OSQPException, ValueError, RuntimeError) as error:
                    qp_status = "solver_exception"
                    reason = f"qp_exception:{type(error).__name__}"; break
                if time.monotonic() >= deadline:
                    reason = "safety_timeout"; break
                qp_status = result.info.status
                residual = {key: float(value) if np.isfinite(value) else None
                            for key, value in (("primal", result.info.prim_res),
                                               ("dual", result.info.dual_res))}
                if qp_status != "solved" or result.x is None or not np.isfinite(result.x).all():
                    reason = f"qp:{qp_status}"; break
                reference = result.x.copy()
                reason = self._verify(context, reference)
                if time.monotonic() >= deadline:
                    reason = "safety_timeout"; break
                if not reason and np.max(A@reference-upper) <= 1e-6:
                    accepted = True; break
        if accepted:
            control = reference
            status = "unchanged" if np.max(abs(control-nominal)) <= 1e-6 else "projected"
        else:
            control = brake_controls(item.previous_control)[0]
            verify_reason = self._verify(context, control, cbf=False)
            status = "unverified_brake" if verify_reason else "verified_brake"
            reason = reason or ("expired_request" if expired else "hold_request")
            if verify_reason:
                reason += ";"+verify_reason
        self.last_control = control.copy()
        return ExecutionDecision(frozen_array(nominal), frozen_array(control), status,
            frozen_array(control-nominal), reason, status != "unverified_brake",
            tuple(ids), qp_status, residual, time.monotonic()-started)


def decision_record(decision):
    row = asdict(decision)
    for name in ("nominal_control", "applied_control", "correction"):
        value = row[name]
        row[name] = None if value is None else value.tolist()
    return row
