"""Planner protocol v1: sensor-only inputs and versioned, solver-independent state.

No policy distribution or safety projection is implemented by this module.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .contracts import (
    DT,
    N,
    VERSION,
    LocalGoalRequest,
    NeighborPrediction,
    PlannerFeedback,
    TimedTrajectory,
)
from .memory import LocalTerrainMemory
from .planner import NMPCPlanner, build_solver, track_trajectory

PROTOCOL_VERSION = "planner_protocol_v1"
ADAPTER_ID = "nmpc_bridge_v1"
STATE_VERSION = 1


def finite_array(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite shape {shape}")
    return value


def frozen_array(value):
    value = np.asarray(value, dtype=float).copy()
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class LocalIntent:
    goal_xy: np.ndarray
    hold: bool
    timestamp: float
    expires_at: float
    terminal_yaw: float | None = None
    speed_cap: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "goal_xy", frozen_array(self.goal_xy))

    @classmethod
    def from_request(cls, request, *, hold=False):
        if request.version != VERSION:
            raise ValueError("Unsupported request version")
        finite_array(request.target, (3,), "target")
        return cls(
            request.target[:2],
            hold,
            request.timestamp,
            request.expires_at,
            float(request.target[2]),
            request.speed_cap,
        )

    @classmethod
    def for_hold(cls, pose, now):
        return cls.from_request(LocalGoalRequest.for_hold(pose, now), hold=True)

    def validate(self, now):
        finite_array(self.goal_xy, (2,), "goal_xy")
        if not isinstance(self.hold, (bool, np.bool_)):
            raise ValueError("hold must be boolean")
        if self.hold and self.terminal_yaw is None:
            raise ValueError("HOLD requires a frozen pose; use LocalIntent.for_hold")
        if not np.isfinite([self.timestamp, self.expires_at]).all():
            raise ValueError("Nonfinite intent time")
        if self.timestamp > now or self.expires_at < self.timestamp:
            raise ValueError("Invalid intent time interval")
        if self.terminal_yaw is not None and not np.isfinite(self.terminal_yaw):
            raise ValueError("Nonfinite terminal yaw")
        if self.speed_cap is not None and (
            not np.isfinite(self.speed_cap) or not 0 <= self.speed_cap <= 1.15
        ):
            raise ValueError("Invalid speed cap")


@dataclass(frozen=True)
class TerrainSnapshot:
    xy: np.ndarray
    features: np.ndarray
    observed_at: np.ndarray
    support_radius: float
    version: str = "local_terrain_samples_v1"

    def __post_init__(self):
        count = len(self.xy)
        for name, shape in (
            ("xy", (count, 2)),
            ("features", (count, 5)),
            ("observed_at", (count,)),
        ):
            finite_array(getattr(self, name), shape, name)
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        if (
            self.version != "local_terrain_samples_v1"
            or not np.isfinite(self.support_radius)
            or self.support_radius <= 0
        ):
            raise ValueError("Invalid terrain snapshot version/support")


class TerrainView:
    """Detached local-memory snapshot; queries cannot modify the live sensor memory."""

    __slots__ = ("__memory", "__conservative_ties")

    def __init__(self, memory):
        if type(memory) is not LocalTerrainMemory:
            raise ValueError("TerrainView requires sensor-only LocalTerrainMemory")
        # Whitelist sensor fields instead of copying arbitrary attributes or owner references.
        count = len(memory.samples)
        finite_array(memory.xy, (count, 2), "terrain samples")
        finite_array(memory.features, (count, 5), "terrain features")
        finite_array(memory._timestamps[:count], (count,), "terrain timestamps")
        self.__memory = LocalTerrainMemory(
            memory.capacity, memory.resolution, memory.support_radius,
            version=getattr(memory, "version", "quantized_latest_v1"),
        )
        self.__memory.observe(memory.xy, memory.features, 0.0)
        self.__memory._timestamps[:count] = memory._timestamps[:count]
        self.__conservative_ties = getattr(memory, "version", "quantized_latest_v1") != "quantized_latest_v1"

    def conservative(self):
        """Opt in to MPPI tie rules; preserve historical bridge query semantics."""
        if self.__conservative_ties:
            return self
        view = object.__new__(TerrainView)
        # Share only this already detached sensor snapshot, never the live owner.
        view.__memory = self.__memory
        view.__conservative_ties = True
        return view

    def lookup(self, xy):
        points = np.atleast_2d(xy)
        features, known, direct = self.__memory.lookup(points)
        # Resolve equidistant conflicting observations conservatively. This is
        # limited to detached protocol views; historical live memory is unchanged.
        memory = self.__memory
        if self.__conservative_ties and len(memory.samples) > 1:
            distances, _ = memory.tree.query(points, k=2)
            ties = np.isclose(distances[:, 0], distances[:, 1], atol=1e-7, rtol=0)
            features = features.copy()
            for index in np.flatnonzero(ties):
                d = np.linalg.norm(memory.xy - points[index], axis=-1)
                peer = memory.features[np.isclose(d, d.min(), atol=1e-7, rtol=0)]
                features[index, 1:3] = peer[np.linalg.norm(peer[:, 1:3], axis=-1).argmax(), 1:3]
                features[index, 4] = peer[:, 4].min()
        return features, known, direct

    def safe(self, xy):
        features, known, _ = self.lookup(xy)
        return known & (features[:, 4] >= 0.25) & (np.linalg.norm(features[:, 1:3], axis=-1) <= 0.8)

    def snapshot(self):
        memory = self.__memory
        return TerrainSnapshot(
            memory.xy,
            memory.features,
            memory._timestamps[: len(memory.samples)],
            memory.support_radius,
        )

    def local_ball(self, xy):
        return self.__memory.local_ball(xy)

    def free_disks(self, center, count=128):
        return self.__memory.free_disks(center, count)


@dataclass(frozen=True)
class IdentifiedNeighbor:
    sender_id: int
    prediction: NeighborPrediction

    def __post_init__(self):
        p = self.prediction
        object.__setattr__(
            self,
            "prediction",
            NeighborPrediction(
                frozen_array(p.positions),
                frozen_array(p.radii),
                p.timestamp,
                p.expires_at,
                p.version,
            ),
        )


@dataclass(frozen=True)
class PlannerInput:
    env_id: int
    agent_id: int
    ego_pose: np.ndarray
    ego_velocity: np.ndarray  # signed actual forward speed, actual yaw rate
    intent: LocalIntent
    terrain: TerrainView
    neighbors: tuple[IdentifiedNeighbor, ...]
    previous_control: np.ndarray  # last command sent to differential execution
    now: float
    times: np.ndarray  # absolute episode times, N+1
    version: str = PROTOCOL_VERSION

    def __post_init__(self):
        for name in ("ego_pose", "ego_velocity", "previous_control", "times"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        object.__setattr__(self, "neighbors", tuple(self.neighbors))

    def validate(self):
        if self.version != PROTOCOL_VERSION:
            raise ValueError("Unsupported planner protocol")
        for value in (self.env_id, self.agent_id):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError("Invalid environment/agent identity")
        if not np.isfinite(self.now) or self.now < 0:
            raise ValueError("Invalid planning time")
        finite_array(self.ego_pose, (3,), "ego_pose")
        finite_array(self.ego_velocity, (2,), "ego_velocity")
        finite_array(self.previous_control, (2,), "previous_control")
        finite_array(self.times, (N + 1,), "times")
        if not np.allclose(self.times, self.now + np.arange(N + 1) * DT, atol=1e-9, rtol=0):
            raise ValueError("Expected fixed 0.2 s / 15-step time grid")
        if type(self.terrain) is not TerrainView:
            raise ValueError("Expected sensor-only TerrainView")
        self.intent.validate(self.now)
        seen = {self.agent_id}
        for neighbor in self.neighbors:
            if (
                isinstance(neighbor.sender_id, bool)
                or not isinstance(neighbor.sender_id, (int, np.integer))
                or neighbor.sender_id < 0
                or neighbor.sender_id in seen
            ):
                raise ValueError("Invalid or duplicate neighbor sender")
            seen.add(neighbor.sender_id)
            p = neighbor.prediction
            if p.version != VERSION:
                raise ValueError("Unsupported neighbor prediction version")
            finite_array(p.positions, (N + 1, 2), "neighbor positions")
            finite_array(p.radii, (N + 1,), "neighbor radii")
            if (p.radii < 0).any() or not np.isfinite([p.timestamp, p.expires_at]).all():
                raise ValueError("Invalid neighbor uncertainty/time")
            if p.timestamp > self.now or p.expires_at < p.timestamp:
                raise ValueError("Invalid neighbor time interval")
            # Expired predictions are intentional conservative sensor-cache estimates.


@dataclass(frozen=True)
class PlannerOutput:
    trajectory: TimedTrajectory
    feedback: PlannerFeedback
    valid: bool
    fallback_reason: str = ""
    total_cost: float | None = None
    version: str = PROTOCOL_VERSION

    @property
    def predicted_states(self):
        return self.trajectory.states

    @property
    def nominal_controls(self):
        return self.trajectory.controls

    def validate(self, request):
        if self.version != PROTOCOL_VERSION or self.trajectory.version != VERSION:
            raise ValueError("Unsupported planner output version")
        if self.feedback.version != VERSION:
            raise ValueError("Unsupported feedback version")
        p = self.trajectory
        finite_array(p.states, (N + 1, 3), "predicted states")
        finite_array(p.controls, (N, 2), "nominal controls")
        finite_array(p.velocities, (N, 2), "predicted velocities")
        finite_array(p.times, (N + 1,), "trajectory times")
        if not np.allclose(p.times, request.times, atol=1e-9, rtol=0) or p.timestamp != request.now:
            raise ValueError("Trajectory time mismatch")
        if not np.allclose(p.states[0], request.ego_pose, atol=1e-9, rtol=0):
            raise ValueError("Trajectory initial pose mismatch")
        if not isinstance(self.valid, (bool, np.bool_)):
            raise ValueError("valid must be boolean")
        feasible = self.feedback.status in {
            "feasible",
            "feasible_prefix",
            "feasible_iteration_limit",
            "previous",
        }
        if bool(self.valid) != feasible or self.feedback.emergency_brake != (
            self.feedback.status == "brake"
        ):
            raise ValueError("Inconsistent planning validity/feedback")
        if self.fallback_reason != self.feedback.reason or (
            not self.valid and not self.fallback_reason
        ):
            raise ValueError("Inconsistent fallback reason")
        if self.total_cost is not None and not np.isfinite(self.total_cost):
            raise ValueError("Nonfinite total cost")


class Planner(Protocol):
    def plan(self, request: PlannerInput) -> PlannerOutput: ...
    def reset(self) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...


class NMPCAdapter:
    """Lossless bridge for complete legacy requests; low-dimensional defaults await T05."""

    def __init__(self, planner):
        self._planner = planner

    def reset(self):
        self._planner.reset()

    def state_dict(self):
        return deepcopy(
            {
                "protocol": PROTOCOL_VERSION,
                "adapter": ADAPTER_ID,
                "state_version": STATE_VERSION,
                "planner_revision": "prefix_consistency_v2",
                "previous": self._planner.previous,
                "blocked": self._planner.blocked,
            }
        )

    def validate_state_dict(self, state):
        for key, expected in (
            ("protocol", PROTOCOL_VERSION),
            ("adapter", ADAPTER_ID),
            ("state_version", STATE_VERSION),
            ("planner_revision", "prefix_consistency_v2"),
        ):
            if state.get(key) != expected:
                raise ValueError(f"Planner state mismatch: {key}")
        if (
            "previous" not in state
            or not np.isfinite(state.get("blocked", np.nan))
            or state["blocked"] < 0
        ):
            raise ValueError("Invalid planner warm state")
        p = state["previous"]
        if p is not None:
            if not isinstance(p, TimedTrajectory) or p.version != VERSION:
                raise ValueError("Invalid warm trajectory version")
            for name, shape in (
                ("states", (N + 1, 3)),
                ("controls", (N, 2)),
                ("velocities", (N, 2)),
                ("times", (N + 1,)),
            ):
                finite_array(getattr(p, name), shape, name)
            if not np.isfinite(p.timestamp) or not np.allclose(
                p.times, p.timestamp + np.arange(N + 1) * DT, atol=1e-9, rtol=0
            ):
                raise ValueError("Invalid warm trajectory times")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.reset()
        self._planner.previous = deepcopy(state["previous"])
        self._planner.blocked = float(state["blocked"])

    def _brake(self, request, reason):
        last = request.previous_control.copy()
        controls = []
        for _ in range(N):
            last += np.clip(-last, [-0.3, -0.6], [0.3, 0.6])
            controls.append(last.copy())
        trajectory = self._planner.rollout(
            request.ego_pose, np.array(controls), request.terrain, request.now
        )
        self._planner.previous = None
        self._planner.blocked += DT
        feedback = PlannerFeedback(
            status="brake",
            reason=reason,
            emergency_brake=True,
            blocked_seconds=self._planner.blocked,
        )
        return PlannerOutput(trajectory, feedback, False, reason)

    def plan(self, request):
        request.validate()
        if len(request.neighbors) > 3:
            raise ValueError("NMPC bridge supports at most three communicated neighbors")
        intent = request.intent
        if request.now > intent.expires_at:
            result = self._brake(request, "expired_intent")
        elif intent.terminal_yaw is None or (not intent.hold and intent.speed_cap is None):
            result = self._brake(request, "unsupported_intent")
        else:
            goal = LocalGoalRequest(
                np.r_[intent.goal_xy, intent.terminal_yaw],
                0.0 if intent.hold else intent.speed_cap,
                intent.timestamp,
                intent.expires_at,
            )
            trajectory, feedback = self._planner.plan(
                request.ego_pose,
                goal,
                request.terrain,
                [n.prediction for n in request.neighbors],
                request.previous_control,
                request.now,
            )
            result = PlannerOutput(
                trajectory, feedback, not feedback.emergency_brake, feedback.reason
            )
        result.validate(request)
        return result


def create_nmpc_adapters(count, cache_dir, slope_scale, *, solver_factory=build_solver):
    return [
        NMPCAdapter(
            NMPCPlanner(solver_factory(cache_dir, slope_scale, build=(i == 0)), slope_scale)
        )
        for i in range(count)
    ]


def plan_batch(planners: Sequence[Planner], inputs: Sequence[PlannerInput], *, pool=None):
    """One independent instance per input; preserve caller order, including sparse batches."""
    if len(planners) != len(inputs) or len({id(p) for p in planners}) != len(planners):
        raise ValueError("Batch requires one independent planner per input")
    identities = [(i.env_id, i.agent_id) for i in inputs]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate batch identity")
    for request in inputs:
        request.validate()
    if any(hasattr(p, "batch_kernel") for p in planners):
        outputs = [None] * len(inputs)
        groups = {}
        for index, planner in enumerate(planners):
            kernel = getattr(planner, "batch_kernel", None)
            if kernel is None:
                outputs[index] = planner.plan(inputs[index])
            else:
                key = (type(kernel), kernel.config)
                group = groups.setdefault(key, (kernel, []))
                group[1].append(index)
        for kernel, indices in groups.values():
            results = kernel.plan_batch(
                [planners[i] for i in indices], [inputs[i] for i in indices]
            )
            for index, result in zip(indices, results):
                outputs[index] = result
    elif pool is None:
        outputs = [p.plan(i) for p, i in zip(planners, inputs)]
    else:
        futures = [pool.submit(p.plan, i) for p, i in zip(planners, inputs)]
        outputs = [future.result() for future in futures]
    for request, output in zip(inputs, outputs):
        output.validate(request)
    return outputs


@dataclass(frozen=True)
class ExecutionDecision:
    nominal_control: np.ndarray
    applied_control: np.ndarray
    safety_status: str = "projection_disabled"
    correction: np.ndarray | None = None
    reason: str = ""
    model_feasible: bool = False
    constraint_ids: tuple[str, ...] = ()
    qp_status: str = "disabled"
    residual: dict | None = None
    seconds: float = 0.0


def execution_passthrough(pose, output):
    """Tracking followed by a disabled safety projection; no feasibility certificate."""
    nominal = finite_array(track_trajectory(pose, output.trajectory), (2,), "tracking control")
    return ExecutionDecision(frozen_array(nominal), frozen_array(nominal))
