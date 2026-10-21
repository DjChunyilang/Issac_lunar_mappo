"""New-scale geometric and native discrete baselines, isolated from old libraries.

Geometry is a controller reference only. Every returned control has first been
fed through the shared actuator/model while tracking that reference; the common
planner then independently predicts and reviews its executable trajectory.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
import math

import numpy as np
import torch

from ..action_interpreter import PRIMITIVE_HOLD, PRIMITIVE_REVERSE, PRIMITIVE_SPIN, PRIMITIVE_YIELD
from ..gathering_env_cfg import LowLevelControlCfg, TrajectoryGeneratorCfg
from ..simple_controller import compute_control
from ..trajectory_generator import generate_trajectory
from .contracts import wrap
from .upstream_execution import limit_control_numpy, midpoint_numpy, terrain_gain_numpy
from .upstream_planning import Candidate, PlanningAdapter, UpstreamIntent, UpstreamPlannerInput
from .upstream_spec import UpstreamSpec, digest


PRIMITIVE_LIBRARY_VERSION = "upstream_primitive47_v1"
LEGACY_LIBRARY_FINGERPRINT = "c41f447bb35a957f80ab8a3cc522743e8ecec42d5d3f84b7d69fdab03fd5d68e"
# Frozen source numbers, not a reference to mutable historical decoder settings.
_FORWARD_ENDPOINTS = ((.4, -.4), (.4, 0.), (.4, .4), (.8, -.8), (.8, -.4),
                      (.8, 0.), (.8, .4), (.8, .8), (1.2, -.8), (1.2, -.4),
                      (1.2, 0.), (1.2, .4), (1.2, .8))
_FORWARD_SPEEDS = (.45, .8, 1.15)
_REVERSE_ENDPOINTS = ((-.4, -.4), (-.4, 0.), (-.4, .4))
_YIELD_ENDPOINTS = ((.8, .8), (.8, -.8))


def primitive_library(spec):
    """A new fingerprint binds the unchanged native ordering to the new profile."""
    if not isinstance(spec, UpstreamSpec):
        raise ValueError("Primitive library requires UpstreamSpec")
    displacement_scale = spec.rho_max / 1.6
    speed_scale = spec.vehicle.forward_limit / 1.15
    entries = []

    def append(xy, speed, direction, yaw_delta, category):
        entries.append({"index": len(entries), "local_xy": [x * displacement_scale for x in xy],
                        "reference_speed": speed, "motion_direction": direction,
                        "planned_yaw_delta": yaw_delta, "primitive_type": category})

    append((0., 0.), 0., 0., 0., PRIMITIVE_HOLD)
    for xy in _FORWARD_ENDPOINTS:
        for speed in _FORWARD_SPEEDS:
            append(xy, speed * speed_scale, 1., 0., 1)
    for xy in _REVERSE_ENDPOINTS:
        append(xy, -min(.45 * speed_scale, spec.vehicle.reverse_limit), -1., 0., PRIMITIVE_REVERSE)
    for yaw in (math.pi / 4, -math.pi / 4):
        append((0., 0.), 0., 0., yaw, PRIMITIVE_SPIN)
    for xy in _YIELD_ENDPOINTS:
        append(xy, .45 * speed_scale, 1., 0., PRIMITIVE_YIELD)
    return {"version": PRIMITIVE_LIBRARY_VERSION, "count": 47,
            "profile_fingerprint": spec.fingerprint,
            "source_library_sha256": LEGACY_LIBRARY_FINGERPRINT,
            "displacement_scale": displacement_scale, "forward_speed_scale": speed_scale,
            "reverse_speed_cap": spec.vehicle.reverse_limit, "entries": entries,
            "ordering": "hold, endpoint-major/speed-minor forward, reverse, spin, yield"}


def primitive_fingerprint(spec):
    return digest(primitive_library(spec))


@dataclass(frozen=True)
class UpstreamPrimitiveInput:
    """Native ID and library identity; a polar intent cannot encode this payload."""

    context: UpstreamPlannerInput
    primitive_id: int
    library_version: str
    library_fingerprint: str

    @classmethod
    def for_context(cls, context, primitive_id):
        return cls(context, primitive_id, PRIMITIVE_LIBRARY_VERSION,
                   primitive_fingerprint(context.spec))

    def validate(self):
        if type(self.context) is not UpstreamPlannerInput:
            raise ValueError("Native primitive requires upstream planner context")
        self.context.__post_init__()
        if (self.library_version != PRIMITIVE_LIBRARY_VERSION
                or self.library_fingerprint != primitive_fingerprint(self.context.spec)
                or isinstance(self.primitive_id, (bool, np.bool_))
                or not isinstance(self.primitive_id, (int, np.integer))
                or not 0 <= self.primitive_id < 47):
            raise ValueError("Native primitive ID/library fingerprint mismatch")


def _tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def _extend_reference(reference, horizon):
    """A rolling local target guides heading; reaching it does not demand a stop."""
    end_time = float(reference.timestamps[0, 0, -1])
    if end_time >= horizon or abs(float(reference.reference_speed[0, 0, -1])) < 1e-12:
        return reference
    speed = float(reference.reference_speed[0, 0, -1])
    heading = float(reference.headings[0, 0, -1])
    extra = reference.points[:, :, -1:].clone()
    extra[..., 0] += (horizon - end_time) * speed * math.cos(heading)
    extra[..., 1] += (horizon - end_time) * speed * math.sin(heading)
    fields = {"points": torch.cat((reference.points, extra), dim=2),
              "timestamps": torch.cat((reference.timestamps,
                                      torch.full_like(reference.timestamps[..., -1:], horizon)), dim=2)}
    for name in ("headings", "reference_speed", "motion_direction", "planned_yaw_delta", "primitive_type"):
        value = getattr(reference, name)
        fields[name] = None if value is None else torch.cat((value, value[..., -1:]), dim=2)
    return replace(reference, **fields)


class DeterministicAdapter(PlanningAdapter):
    """Physical arc-length references tracked using the existing controller."""

    state_version = 1

    def __init__(self, cfg, geometry="line"):
        if geometry not in ("line", "quintic"):
            raise ValueError("Unsupported upstream geometry")
        self.geometry = geometry
        self.adapter_id = f"{geometry}_upstream_v1"
        super().__init__(cfg)

    def _adapter_state(self):
        return {"geometry": self.geometry, "reference_version": "arc_length_upstream_v1"}

    def _validate_adapter_state(self, state):
        if state != self._adapter_state():
            raise ValueError("Geometric baseline state mismatch")

    def _reference(self, request, entry=None):
        pose = request.ego_pose
        cap = self.spec.vehicle.forward_limit if request.intent.speed_cap is None else request.intent.speed_cap
        position = _tensor([[[pose[0], pose[1], 0.]]])
        yaw = _tensor([[pose[2]]])
        goal = _tensor([[[*request.intent.goal_xy, 0.]]])
        delta = request.intent.goal_xy - pose[:2]
        bearing = math.atan2(delta[1], delta[0]) if np.linalg.norm(delta) > 1e-12 else pose[2]
        reverse = abs(wrap(bearing - pose[2])) > math.pi / 2
        direction = -1. if reverse else 1.
        speed = min(cap, self.spec.vehicle.reverse_limit if reverse else self.spec.vehicle.forward_limit)
        category = PRIMITIVE_REVERSE if reverse else 1
        yaw_delta = 0.
        if entry is not None:
            direction = entry["motion_direction"]
            speed = min(abs(entry["reference_speed"]), cap)
            category = entry["primitive_type"]
            yaw_delta = wrap(request.intent.terminal_yaw - pose[2]) if category == PRIMITIVE_SPIN else 0.
        generator = TrajectoryGeneratorCfg(
            geometry_method=self.geometry, n_trajectory_points=41,
            reference_speed=max(speed, 1e-12), time_parameterization="arc_length_reference_speed",
            rotation_reference_speed=self.spec.vehicle.low_speed_yaw_limit,
        )
        kwargs = dict(current_yaws=yaw, reference_speed=_tensor([[direction * speed]]))
        if self.geometry == "quintic":
            kwargs.update(motion_direction=_tensor([[direction]]), planned_yaw_delta=_tensor([[yaw_delta]]),
                          primitive_type=torch.tensor([[category]]))
        reference = generate_trajectory(position, goal, generator, self.spec.grid.dt, **kwargs)
        if self.geometry == "line":
            # The old line generator is pure geometry. Attach directional tracking
            # channels here without changing its historical behavior.
            shape = reference.reference_speed.shape
            reference = replace(reference,
                headings=reference.headings + (math.pi if reverse else 0.),
                motion_direction=torch.full(shape, direction, dtype=torch.float64),
                planned_yaw_delta=torch.zeros(shape, dtype=torch.float64),
                primitive_type=torch.full(shape, category, dtype=torch.long))
        if not request.intent.stop_at_goal and entry is None and np.linalg.norm(delta) > 1e-8:
            reference = _extend_reference(reference, self.spec.grid.horizon + self.spec.grid.dt)
        return reference

    def _controls(self, request, entry=None):
        reference = self._reference(request, entry)
        controller = LowLevelControlCfg(
            kinematic_model="differential_drive", tracking_point_mode="planning_time",
            max_linear_speed=self.spec.vehicle.forward_limit, max_angular_speed=self.spec.vehicle.yaw_limit,
        )
        state = request.ego_pose.copy()
        previous = request.previous_control.copy()
        controls = []
        cap = self.spec.vehicle.forward_limit if request.intent.speed_cap is None else request.intent.speed_cap
        for step in range(self.spec.grid.steps):
            shifted = replace(reference, timestamps=reference.timestamps - step * self.spec.grid.dt)
            command = compute_control(_tensor([[[state[0], state[1], 0.]]]), _tensor([[state[2]]]),
                                      shifted, controller, self.spec.grid.dt)
            desired = command.packed[0, 0].numpy()
            desired[0] = np.clip(desired[0], -min(cap, self.spec.vehicle.reverse_limit), cap)
            if (entry is None and request.intent.stop_at_goal and request.intent.terminal_yaw is not None
                    and np.linalg.norm(state[:2] - request.intent.goal_xy) < .2):
                desired[1] = 2 * wrap(request.intent.terminal_yaw - state[2])
            control = limit_control_numpy(desired, previous, self.spec)
            features, _ = request.terrain.lookup(state[:2])
            gain = terrain_gain_numpy(state, features[0], self.spec)
            state, _ = midpoint_numpy(state, control, self.spec, gain=gain)
            controls.append(control.copy())
            previous = control
        return np.asarray(controls)

    def candidate_controls(self, request):
        return [Candidate(self._controls(request), source=self.geometry,
                          diagnostics={"reference": "arc_length", "tracking": "planning_time"})]


class PrimitiveAdapter(DeterministicAdapter):
    """47 native actions with request-lifetime frozen endpoints and spin headings."""

    def __init__(self, cfg):
        self._native_payload = None
        self._source_key = None
        self._resolved = None
        super().__init__(cfg, "quintic")
        self.adapter_id = "primitive47_upstream_v1"

    def reset(self):
        super().reset()
        self._source_key = self._resolved = None
        self._native_payload = None

    def _adapter_state(self):
        return {**super()._adapter_state(), "library_version": PRIMITIVE_LIBRARY_VERSION,
                "library_fingerprint": primitive_fingerprint(self.spec),
                "source_key": deepcopy(self._source_key), "resolved": deepcopy(self._resolved)}

    def _validate_adapter_state(self, state):
        if not isinstance(state, dict):
            raise ValueError("Invalid primitive adapter snapshot")
        expected = self._adapter_state()
        for key in ("geometry", "reference_version", "library_version", "library_fingerprint"):
            if state.get(key) != expected[key]:
                raise ValueError("Native primitive snapshot library mismatch")
        key, resolved = state.get("source_key"), state.get("resolved")
        if (key is None) != (resolved is None):
            raise ValueError("Incomplete primitive request snapshot")
        if key is not None:
            if (not isinstance(key, tuple) or len(key) != 3 or not isinstance(key[0], int)
                    or isinstance(key[0], bool) or not 0 <= key[0] < 47
                    or not np.isfinite(key[1:]).all() or key[1] < 0 or key[2] < key[1]
                    or not isinstance(resolved, UpstreamIntent)):
                raise ValueError("Invalid frozen primitive request")
            if (resolved.timestamp, resolved.expires_at) != key[1:]:
                raise ValueError("Frozen primitive timestamp mismatch")
            resolved.__post_init__()
            if resolved.hold != (key[0] == 0) or not resolved.stop_at_goal or resolved.terminal_yaw is None:
                raise ValueError("Frozen primitive semantics mismatch")

    def _load_adapter_state(self, state):
        self._validate_adapter_state(state)
        self._source_key = deepcopy(state["source_key"])
        self._resolved = deepcopy(state["resolved"])

    def plan(self, payload):
        if type(payload) is not UpstreamPrimitiveInput:
            raise ValueError("Expected native UpstreamPrimitiveInput")
        payload.validate()
        request = payload.context
        key = (int(payload.primitive_id), request.intent.timestamp, request.intent.expires_at)
        entry = primitive_library(self.spec)["entries"][payload.primitive_id]
        # Use a trial resolution until the common base validates instance binding.
        resolved = self._resolved
        if key != self._source_key:
            origin=(request.information_pose if self.cfg.environment_execution_version!='not_connected'
                    and request.information_pose is not None else request.ego_pose)
            c, s = math.cos(origin[2]), math.sin(origin[2])
            goal = origin[:2] + np.array([[c, -s], [s, c]]) @ entry["local_xy"]
            resolved = replace(request.intent, goal_xy=goal, hold=entry["primitive_type"] == PRIMITIVE_HOLD,
                               terminal_yaw=wrap(origin[2] + entry["planned_yaw_delta"]),
                               stop_at_goal=True)
        # Safety speed constraints may tighten without changing the native goal.
        resolved = replace(resolved, speed_cap=request.intent.speed_cap)
        self._native_payload = entry
        try:
            result = super().plan(replace(request, intent=resolved))
        finally:
            self._native_payload = None
        self._source_key, self._resolved = key, deepcopy(resolved)
        # HOLD/expired requests bypass candidate generation but still retain the
        # exact native action and library identity in their execution evidence.
        return replace(result, diagnostics={**result.diagnostics,
                       "primitive_id": int(payload.primitive_id),
                       "library_version": PRIMITIVE_LIBRARY_VERSION,
                       "library_fingerprint": primitive_fingerprint(self.spec)})

    def candidate_controls(self, request):
        if self._native_payload is None:
            raise ValueError("Missing native primitive payload")
        entry = self._native_payload
        return [Candidate(self._controls(request, entry), source=f"primitive:{entry['index']}",
                          diagnostics={"primitive_id": entry["index"],
                                       "library_version": PRIMITIVE_LIBRARY_VERSION,
                                       "library_fingerprint": primitive_fingerprint(self.spec)})]
