"""T05 versioned baselines; native primitives retain their discrete payload."""

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import time

import numpy as np
import torch

from ..action_interpreter import (
    decode_action,
    SPATIOTEMPORAL_ENDPOINTS,
    SPATIOTEMPORAL_SPEEDS,
    DIFFERENTIAL_REVERSE_ENDPOINTS,
    DIFFERENTIAL_SPIN_YAW_DELTAS,
    DIFFERENTIAL_YIELD_ENDPOINTS,
    DIFFERENTIAL_PRIMITIVE_ACTION_COUNT,
)
from ..gathering_env_cfg import PlannerCfg, TrajectoryGeneratorCfg, LowLevelControlCfg
from ..simple_controller import compute_control
from ..trajectory_generator import generate_trajectory
from .contracts import DT, N, VERSION, PlannerFeedback, TimedTrajectory, wrap
from .execution import (
    dynamics,
    rollout,
    brake_controls,
    limit_control,
    validate_trajectory,
    stopped_prefix,
    shifted_sequence,
)
from .protocol import (
    PROTOCOL_VERSION,
    NMPCAdapter,
    LocalIntent,
    PlannerOutput,
    PlannerInput,
    finite_array,
)


PRIMITIVE_VERSION = "exp156_primitive47_v1"


def primitive_library():
    decoded = decode_action(
        torch.arange(47).reshape(1, 47),
        torch.zeros(1, 47, 3),
        torch.zeros(1, 47),
        PlannerCfg(action_type="differential_trajectory_primitives", action_dim=47),
    )
    return {
        "version": PRIMITIVE_VERSION,
        "count": DIFFERENTIAL_PRIMITIVE_ACTION_COUNT,
        "forward_endpoints": SPATIOTEMPORAL_ENDPOINTS,
        "speeds": SPATIOTEMPORAL_SPEEDS,
        "reverse_endpoints": DIFFERENTIAL_REVERSE_ENDPOINTS,
        "reverse_speed": -0.45,
        "spin": tuple(float(x) for x in DIFFERENTIAL_SPIN_YAW_DELTAS),
        "yield": DIFFERENTIAL_YIELD_ENDPOINTS,
        "yield_speed": 0.45,
        "entries": [
            {
                "index": i,
                "local_xy": decoded.world_subgoal[0, i, :2].tolist(),
                "reference_speed": float(decoded.reference_speed[0, i]),
                "motion_direction": float(decoded.motion_direction[0, i]),
                "planned_yaw_delta": float(decoded.planned_yaw_delta[0, i]),
                "primitive_type": int(decoded.primitive_type[0, i]),
            }
            for i in range(47)
        ],
        "ordering": "hold, endpoint-major/speed-minor forward, reverse, spin, yield",
    }


def primitive_fingerprint():
    return hashlib.sha256(json.dumps(primitive_library(), sort_keys=True).encode()).hexdigest()


def intent_key(intent):
    return (
        tuple(intent.goal_xy),
        bool(intent.hold),
        intent.timestamp,
        intent.expires_at,
        intent.terminal_yaw,
        intent.speed_cap,
    )


class NMPCPolarAdapter(NMPCAdapter):
    adapter_id = "nmpc_polar_v1"

    def __init__(self, planner):
        super().__init__(planner)
        self.source_intent = self.resolved_intent = None

    def reset(self):
        super().reset()
        self.source_intent = self.resolved_intent = None

    def state_dict(self):
        state = super().state_dict()
        state.update(
            adapter=self.adapter_id,
            state_version=2,
            source_intent=deepcopy(self.source_intent),
            resolved_intent=deepcopy(self.resolved_intent),
        )
        return state

    def _legacy_state(self, state):
        if state.get("adapter") != self.adapter_id or state.get("state_version") != 2:
            raise ValueError("Polar NMPC state version mismatch")
        source, resolved = state.get("source_intent"), state.get("resolved_intent")
        if (source is None) != (resolved is None):
            raise ValueError("Missing frozen intent defaults")
        if source is not None:
            if not isinstance(source, LocalIntent) or not isinstance(resolved, LocalIntent):
                raise ValueError("Invalid cached intent")
            source.validate(source.timestamp)
            resolved.validate(resolved.timestamp)
            if (
                not np.array_equal(source.goal_xy, resolved.goal_xy)
                or source.hold != resolved.hold
                or source.timestamp != resolved.timestamp
                or source.expires_at != resolved.expires_at
                or resolved.terminal_yaw is None
                or resolved.speed_cap is None
                or (
                    source.terminal_yaw is not None and source.terminal_yaw != resolved.terminal_yaw
                )
                or (source.speed_cap is not None and source.speed_cap != resolved.speed_cap)
            ):
                raise ValueError("Cached defaults changed explicit constraints")
        legacy = {**state, "adapter": "nmpc_bridge_v1", "state_version": 1}
        super().validate_state_dict(legacy)
        return legacy

    def validate_state_dict(self, state):
        self._legacy_state(state)

    def load_state_dict(self, state):
        legacy = self._legacy_state(state)
        # Parent dynamically calls validate_state_dict; restore its pure fields here.
        self.reset()
        self._planner.previous = deepcopy(legacy["previous"])
        self._planner.blocked = float(legacy["blocked"])
        self.source_intent = deepcopy(state["source_intent"])
        self.resolved_intent = deepcopy(state["resolved_intent"])

    def plan(self, request):
        request.validate()
        intent = request.intent
        if request.now > intent.expires_at or intent.hold:
            return super().plan(request)
        if self.source_intent is None or intent_key(intent) != intent_key(self.source_intent):
            delta = intent.goal_xy - request.ego_pose[:2]
            yaw = float(request.ego_pose[2])
            if np.linalg.norm(delta) > 1e-6:
                bearing = np.arctan2(delta[1], delta[0])
                yaw = wrap(bearing + (np.pi if abs(wrap(bearing - yaw)) > np.pi / 2 else 0))
            self.source_intent = deepcopy(intent)
            self.resolved_intent = replace(
                intent,
                terminal_yaw=yaw if intent.terminal_yaw is None else intent.terminal_yaw,
                speed_cap=1.15 if intent.speed_cap is None else intent.speed_cap,
            )
        return super().plan(replace(request, intent=self.resolved_intent))


class NMPCRecoveryAdapter(NMPCPolarAdapter):
    adapter_id = "nmpc_recovery_v3"
    recovery_revision = "merit_recovery_v3"
    recovery_state_version = 3

    @property
    def last_diagnostics(self):
        return deepcopy(getattr(self._planner, "last_diagnostics", {}))

    def state_dict(self):
        state = super().state_dict()
        state.update(state_version=self.recovery_state_version, planner_revision=self.recovery_revision)
        return state

    def _legacy_state(self, state):
        if (state.get("adapter") != self.adapter_id or state.get("state_version") != self.recovery_state_version
            or state.get("planner_revision") != self.recovery_revision):
            raise ValueError("Recovery NMPC state version mismatch")
        return super()._legacy_state({**state, "state_version": 2,
                                     "planner_revision": "prefix_consistency_v2"})


class NMPCAdaptiveRecoveryAdapter(NMPCRecoveryAdapter):
    adapter_id = "nmpc_recovery_v4"
    recovery_revision = "merit_recovery_v4"
    recovery_state_version = 4


@dataclass(frozen=True)
class PrimitiveReplayInput:
    """Native discrete command plus common sensor context; not a polar action."""

    context: PlannerInput
    primitive_id: int
    library_version: str = PRIMITIVE_VERSION

    def validate(self):
        if type(self.context) is not PlannerInput:
            raise ValueError("Native primitive requires a protocol sensor context")
        self.context.validate()
        if (
            self.library_version != PRIMITIVE_VERSION
            or isinstance(self.primitive_id, bool)
            or not isinstance(self.primitive_id, (int, np.integer))
            or not 0 <= self.primitive_id < 47
        ):
            raise ValueError("Invalid frozen primitive payload")


class TrajectoryAdapter:
    """State and conservative fallback shared by non-solver planners."""

    adapter_id = "deterministic_line_v1"
    revision = "deterministic_v1"

    def __init__(self, slope_scale=0.75, *, flat=False):
        self.slope_scale, self.flat = float(slope_scale), bool(flat)
        self.identity = None
        self.previous = None
        self.blocked = 0.0
        self.last_diagnostics = {}

    def reset(self):
        self.previous = None
        self.blocked = 0.0
        self.last_diagnostics = {}

    def configuration(self):
        return {"slope_scale": self.slope_scale, "flat": self.flat}

    def bind(self, request):
        identity = (request.env_id, request.agent_id)
        if self.identity is not None and self.identity != identity:
            raise ValueError("Planner instance identity mismatch")
        self.identity = identity

    def state_dict(self):
        return deepcopy(
            {
                "protocol": PROTOCOL_VERSION,
                "adapter": self.adapter_id,
                "state_version": 1,
                "planner_revision": self.revision,
                "configuration": self.configuration(),
                "identity": self.identity,
                "previous": self.previous,
                "blocked": self.blocked,
            }
        )

    def validate_state_dict(self, state):
        expected = self.state_dict()
        for key in ("protocol", "adapter", "state_version", "planner_revision", "configuration"):
            if state.get(key) != expected[key]:
                raise ValueError(f"Planner state mismatch: {key}")
        identity = state.get("identity")
        if identity is not None and (
            not isinstance(identity, tuple)
            or len(identity) != 2
            or any(
                isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0
                for x in identity
            )
        ):
            raise ValueError("Invalid saved identity")
        if self.identity is not None and identity != self.identity:
            raise ValueError("Planner state identity mismatch")
        if not np.isfinite(state.get("blocked", np.nan)) or state["blocked"] < 0:
            raise ValueError("Invalid blocked time")
        p = state.get("previous")
        if "previous" not in state:
            raise ValueError("Missing trajectory")
        if p is not None:
            if not isinstance(p, TimedTrajectory) or p.version != VERSION:
                raise ValueError("Invalid previous trajectory")
            for key, shape in (
                ("states", (16, 3)),
                ("controls", (15, 2)),
                ("velocities", (15, 2)),
                ("times", (16,)),
            ):
                finite_array(getattr(p, key), shape, key)
            if not np.isfinite(p.timestamp) or not np.allclose(
                p.times, p.timestamp + np.arange(16) * DT, atol=1e-9, rtol=0
            ):
                raise ValueError("Invalid previous time grid")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.reset()
        self.identity, self.previous, self.blocked = (
            deepcopy(state["identity"]),
            deepcopy(state["previous"]),
            float(state["blocked"]),
        )

    def predict(self, request, controls):
        return rollout(
            request.ego_pose,
            controls,
            request.terrain,
            request.now,
            self.slope_scale,
            flat=self.flat,
        )

    def finish(self, request, candidates, *, reason="", force_brake=False):
        """Candidates are ordered; all output states come from physical prediction."""
        cap = (
            0.0
            if request.intent.hold
            else (1.15 if request.intent.speed_cap is None else request.intent.speed_cap)
        )
        peers = [n.prediction for n in request.neighbors]
        feedback = PlannerFeedback(reason=reason)
        result = None
        if not force_brake:
            for candidate in candidates:
                rejection = validate_trajectory(
                    candidate, request.terrain, peers, cap, request.previous_control
                )
                if not rejection:
                    result = candidate
                    feedback.status = "feasible"
                    break
                feedback.candidate_rejection = rejection
            if result is None and candidates:
                result, length = stopped_prefix(
                    request.ego_pose,
                    candidates[0].controls,
                    request.terrain,
                    peers,
                    cap,
                    request.previous_control,
                    request.now,
                    float("inf"),
                    self.slope_scale,
                    flat=self.flat,
                )
                if result is not None:
                    feedback.status, feedback.prefix_steps = "feasible_prefix", length
            if result is None and self.previous is not None:
                steps = max(0, int(round((request.now - self.previous.timestamp) / DT)))
                shifted = self.predict(request, shifted_sequence(self.previous.controls, steps))
                rejection = validate_trajectory(
                    shifted, request.terrain, peers, cap, request.previous_control
                )
                if not rejection:
                    result = shifted
                    feedback.status, feedback.used_previous = "previous", True
                else:
                    feedback.candidate_rejection = feedback.candidate_rejection or rejection
        if result is None:
            result = self.predict(request, brake_controls(request.previous_control))
            feedback.status, feedback.emergency_brake = "brake", True
            feedback.reason = reason or feedback.candidate_rejection or "no_feasible_plan"
        elif feedback.candidate_rejection and not feedback.reason:
            feedback.reason = feedback.candidate_rejection
        translating = (
            not request.intent.hold
            and np.linalg.norm(request.intent.goal_xy - request.ego_pose[:2]) >= 0.1
        )
        self.blocked = (
            self.blocked + DT
            if feedback.emergency_brake
            or (translating and np.linalg.norm(result.states[1, :2] - request.ego_pose[:2]) < 1e-3)
            else 0.0
        )
        self.previous = None if feedback.emergency_brake else result
        feedback.blocked_seconds = self.blocked
        output = PlannerOutput(result, feedback, not feedback.emergency_brake, feedback.reason)
        output.validate(request)
        return output


class DeterministicAdapter(TrajectoryAdapter):
    def __init__(self, geometry="line", slope_scale=0.75):
        if geometry not in ("line", "quintic"):
            raise ValueError("Unsupported deterministic geometry")
        super().__init__(slope_scale)
        self.geometry = geometry
        self.adapter_id = f"deterministic_{geometry}_v1"

    def configuration(self):
        return {**super().configuration(), "geometry": self.geometry}

    def _reference(self, request, primitive_id=None):
        pose = request.ego_pose
        pos = torch.tensor([[[pose[0], pose[1], 0.0]]], dtype=torch.float64)
        yaw = torch.tensor([[pose[2]]], dtype=torch.float64)
        generator = TrajectoryGeneratorCfg(
            geometry_method=self.geometry,
            n_trajectory_points=12,
            reference_speed=1.15,
            time_parameterization="arc_length_reference_speed",
        )
        if primitive_id is None:
            goal = torch.tensor([[[*request.intent.goal_xy, 0.0]]], dtype=torch.float64)
            return generate_trajectory(pos, goal, generator, DT, current_yaws=yaw)
        cfg = PlannerCfg(action_type="differential_trajectory_primitives", action_dim=47)
        decoded = decode_action(torch.tensor([[primitive_id]]), pos, yaw, cfg)
        return generate_trajectory(
            pos,
            decoded.world_subgoal,
            generator,
            DT,
            current_yaws=yaw,
            reference_speed=decoded.reference_speed,
            motion_direction=decoded.motion_direction,
            planned_yaw_delta=decoded.planned_yaw_delta,
            primitive_type=decoded.primitive_type,
        )

    def _candidate(self, request, primitive_id=None):
        reference = self._reference(request, primitive_id)
        cfg = LowLevelControlCfg(
            kinematic_model="differential_drive",
            tracking_point_mode="planning_time",
            max_linear_speed=1.15,
            max_angular_speed=2.4,
        )
        state, last = request.ego_pose.copy(), request.previous_control.copy()
        controls = []
        cap = 1.15 if request.intent.speed_cap is None else request.intent.speed_cap
        for k in range(N):
            # Query the original frozen reference at the next physical time.
            shifted = replace(reference, timestamps=reference.timestamps - k * DT)
            command = compute_control(
                torch.tensor([[[state[0], state[1], 0.0]]], dtype=torch.float64),
                torch.tensor([[state[2]]], dtype=torch.float64),
                shifted,
                cfg,
                DT,
            )
            desired = command.packed[0, 0].numpy()
            if (
                primitive_id is None
                and request.intent.terminal_yaw is not None
                and np.linalg.norm(state[:2] - request.intent.goal_xy) < 0.1
            ):
                desired[1] = np.clip(2 * wrap(request.intent.terminal_yaw - state[2]), -2.4, 2.4)
            last = limit_control(desired, last, cap)
            controls.append(last.copy())
            features, _, _ = request.terrain.lookup(state[:2])
            state, _ = dynamics(state, last, features[0], self.slope_scale)
        return self.predict(request, np.array(controls))

    def plan(self, request):
        start = time.monotonic()
        request.validate()
        self.bind(request)
        expired = request.now > request.intent.expires_at
        candidate = (
            self.predict(request, brake_controls(request.previous_control))
            if request.intent.hold
            else None
        )
        if not expired and candidate is None:
            candidate = self._candidate(request)
        result = self.finish(
            request,
            [] if candidate is None else [candidate],
            reason="expired_intent" if expired else "",
            force_brake=expired,
        )
        result.feedback.solve_seconds = time.monotonic() - start
        return result


class PrimitiveAdapter(DeterministicAdapter):
    adapter_id = "primitive47_adapter_v1"
    revision = PRIMITIVE_VERSION

    def __init__(self, slope_scale=0.75):
        super().__init__("quintic", slope_scale)
        self.adapter_id = "primitive47_adapter_v1"

    def configuration(self):
        return {**super().configuration(), "library_sha256": primitive_fingerprint()}

    def plan(self, payload):
        if not isinstance(payload, PrimitiveReplayInput):
            raise ValueError("Expected native PrimitiveReplayInput")
        payload.validate()
        request = payload.context
        self.bind(request)
        start = time.monotonic()
        expired = request.now > request.intent.expires_at
        # The native HOLD branch receives the same physical stopping semantics.
        context = replace(request, intent=replace(request.intent, hold=False, speed_cap=1.15))
        candidate = (
            self.predict(request, brake_controls(request.previous_control))
            if payload.primitive_id == 0
            else self._candidate(context, payload.primitive_id)
        )
        if payload.primitive_id == 0:
            context = replace(request, intent=LocalIntent.for_hold(request.ego_pose, request.now))
        result = self.finish(
            context, [candidate], reason="expired_intent" if expired else "", force_brake=expired
        )
        result.feedback.solve_seconds = time.monotonic() - start
        return result
