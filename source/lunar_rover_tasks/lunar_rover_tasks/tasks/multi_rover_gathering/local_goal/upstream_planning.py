"""Independent 40-step planner protocol and public execution review.

All positions use the receiver's fixed odometry frame. ``now`` is control
activation time. Information-time braking reserves never add execution delay.
This module is independent of acados and the legacy 15-step planner protocol.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import math
import time

import numpy as np

from .contracts import PlannerFeedback, TimedTrajectory
from .upstream_execution import (
    brake_sequence, control_error, limit_control_numpy, rollout_numpy,
    stop_envelope, terrain_gain_numpy,
)
from .upstream_footprint import FootprintNeighbor, FootprintReviewer, FootprintTerrain, QUERY_V2, REVIEW_V2
from .upstream_grid import GridTerrain
from .upstream_spec import UpstreamRuntimeConfig, UpstreamSpec
from .upstream_timing import ActualControlHistory, HELD_TIMING_VERSION, HISTORY_TIMING_VERSION, review_history_reserve

PROTOCOL_VERSION = "planner_protocol_v2"


def _array(value, shape, name):
    result = np.asarray(value, dtype=float).copy()
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} requires finite shape {shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class UpstreamIntent:
    goal_xy: np.ndarray
    hold: bool
    timestamp: float
    expires_at: float
    terminal_yaw: float | None = None
    speed_cap: float | None = None
    stop_at_goal: bool = False

    def __post_init__(self):
        object.__setattr__(self, "goal_xy", _array(self.goal_xy, (2,), "Intent goal"))
        if type(self.hold) is not bool or type(self.stop_at_goal) is not bool:
            raise ValueError("Intent HOLD/stop_at_goal must be explicit boolean")
        if not np.isfinite([self.timestamp, self.expires_at]).all() or not 0 <= self.timestamp <= self.expires_at:
            raise ValueError("Invalid intent time/expiry")
        for name in ("terminal_yaw", "speed_cap"):
            value = getattr(self, name)
            if value is not None and (not np.isscalar(value) or not np.isfinite(value)):
                raise ValueError(f"Invalid intent {name}")
        if self.speed_cap is not None and self.speed_cap < 0:
            raise ValueError("Intent speed cap must be nonnegative")


@dataclass(frozen=True)
class UpstreamPlannerInput:
    env_id: int
    agent_id: int
    ego_pose: np.ndarray
    ego_velocity: np.ndarray
    intent: UpstreamIntent
    terrain: FootprintTerrain
    neighbors: tuple
    previous_control: np.ndarray
    now: float
    times: np.ndarray
    workspace: tuple | None
    spec: UpstreamSpec
    information_pose: np.ndarray | None = None
    information_time: float | None = None
    protocol_version: str = PROTOCOL_VERSION
    actual_history: ActualControlHistory | None = None

    def __post_init__(self):
        for name in ("env_id", "agent_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid {name}")
        if self.agent_id >= 4:
            raise ValueError("Upstream agent_id must identify one of four vehicles")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unsupported upstream planner protocol version")
        if type(self.spec) is not UpstreamSpec or self.spec.grid.steps != 40 or self.spec.grid.dt != .2:
            raise ValueError("Upstream planner requires the explicit 40 x 0.2 s grid")
        if type(self.intent) is not UpstreamIntent or type(self.terrain) not in (FootprintTerrain,GridTerrain):
            raise ValueError("Planner accepts only detached local terrain and a typed intent")
        for name, shape in (("ego_pose", (3,)), ("ego_velocity", (2,)), ("previous_control", (2,)), ("times", (41,))):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        if not np.isfinite(self.now) or self.now < 0 or self.intent.timestamp > self.now + 1e-10:
            raise ValueError("Invalid planner activation time or future intent")
        if not np.allclose(self.times, self.now + np.arange(41) * .2, atol=1e-10, rtol=0):
            raise ValueError("Planner time grid must start at control activation time")
        if self.terrain.profile_fingerprint != self.spec.fingerprint:
            raise ValueError("Planner terrain/profile mismatch")
        peers = tuple(self.neighbors)
        if any(type(peer) is not FootprintNeighbor for peer in peers):
            raise ValueError("Planner neighbors require explicit sender-bound predictions")
        if len({peer.sender_id for peer in peers}) != len(peers) or any(peer.sender_id == self.agent_id or peer.sender_id >= 4 for peer in peers):
            raise ValueError("Duplicate/self/unknown sender in planner neighbors")
        for peer in peers:
            reason = peer.contract_error(self.spec, self.spec.fingerprint)
            if reason:
                raise ValueError(reason)
        object.__setattr__(self, "neighbors", peers)
        if self.workspace is not None:
            bounds = _array(self.workspace, (4,), "Workspace (xmin,ymin,xmax,ymax)")
            if (bounds[2:] <= bounds[:2]).any():
                raise ValueError("Invalid workspace bounds")
            object.__setattr__(self, "workspace", tuple(bounds))
        if (self.information_pose is None) != (self.information_time is None):
            raise ValueError("Information pose and timestamp must be supplied together")
        if self.information_pose is not None:
            object.__setattr__(self, "information_pose", _array(self.information_pose, (3,), "information_pose"))
            if not np.isfinite(self.information_time) or not 0 <= self.information_time <= self.now:
                raise ValueError("Invalid information timestamp")
        if self.actual_history is not None:
            if type(self.actual_history) is not ActualControlHistory or self.information_pose is None:
                raise ValueError("Actual history requires typed feedback and information state")
            self.actual_history.validate(self.spec,self.env_id,self.agent_id)
            history=self.actual_history
            if (not math.isclose(history.times[0],self.information_time,abs_tol=1e-9)
                    or not math.isclose(history.times[-1],self.now,abs_tol=1e-9)
                    or not np.allclose(history.poses[0],self.information_pose,atol=1e-10,rtol=1e-8)
                    or not np.allclose(history.poses[-1],self.ego_pose,atol=1e-10,rtol=1e-8)
                    or not np.allclose(history.previous_at_activation,self.previous_control,atol=1e-10,rtol=1e-8)
                    or (len(history.controls) and not np.allclose(history.actual_velocities[-1],self.ego_velocity,atol=1e-10,rtol=1e-8))):
                raise ValueError("Actual history does not join information and activation feedback")
        if control_error(self.previous_control, self.previous_control, self.spec):
            raise ValueError("Illegal previous applied control")
        moving = abs(self.previous_control) > 1e-10
        if (abs(self.ego_velocity[~moving]) > 1e-8).any():
            raise ValueError("Actual velocity is inconsistent with previous applied control")
        gains = self.ego_velocity[moving] / self.previous_control[moving]
        if len(gains) and ((gains < .25 - 1e-6).any() or (gains > 1 + 1e-6).any() or np.ptp(gains) > 1e-6):
            raise ValueError("Actual velocity violates the terrain-scaled actuator model")
        if self.intent.speed_cap is not None and self.intent.speed_cap > self.spec.vehicle.forward_limit + 1e-10:
            raise ValueError("Intent speed cap exceeds physical limit")
        info_time = self.now if self.information_time is None else self.information_time
        if (self.terrain.readonly_snapshot().observed_at > info_time + 1e-10).any():
            raise ValueError("Terrain snapshot contains observations from the future")
        for peer in peers:
            if peer.times[0] > info_time + 1e-9 or peer.times[-1] < self.times[-1] - 1e-9:
                raise ValueError("Neighbor forecast does not cover information and planning times")


@dataclass(frozen=True)
class Candidate:
    controls: np.ndarray
    source: str = "candidate"
    total_cost: float | None = None
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "controls", _array(self.controls, (40, 2), "Candidate controls"))
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("Candidate source must be named")
        if self.total_cost is not None and not np.isfinite(self.total_cost):
            raise ValueError("Candidate cost must be finite or None")
        object.__setattr__(self, "diagnostics", deepcopy(self.diagnostics))


@dataclass(frozen=True)
class UpstreamPlannerOutput:
    trajectory: TimedTrajectory
    feedback: PlannerFeedback
    valid: bool
    fallback_reason: str
    total_cost: float | None = None
    diagnostics: dict = field(default_factory=dict)


class PlanningAdapter:
    """Per-vehicle state with shared strict review and explicit fallbacks.

    Subclasses implement ``candidate_controls`` and optional state/score hooks.
    Batch backends may call ``prepare`` once then ``finish`` on their outputs.
    Up to sixteen sampled candidates plus one updated mean receive full review;
    fallback and reserve reviews have separately reported counts.
    """

    adapter_id = "upstream_planning_base_v1"
    state_version = 1

    def __init__(self, cfg):
        if type(cfg) is not UpstreamRuntimeConfig:
            raise ValueError("Upstream planner requires independent runtime configuration")
        if cfg.clock_mode != "offline_simulated_latency":
            raise ValueError("Real-time planner execution requires a measured latency chain; only offline_simulated_latency is implemented")
        if cfg.execution_review_version != REVIEW_V2 or cfg.terrain_query_version != QUERY_V2:
            raise ValueError("Upstream planner requires footprint_review_v2 and batch_ties_v2")
        if cfg.timing_version not in {HELD_TIMING_VERSION,HISTORY_TIMING_VERSION}:
            raise ValueError("Unsupported upstream timing version")
        self.cfg = self.config = cfg
        self.spec = cfg.spec
        if cfg.review_acceleration_version == "cuda_sweep_pose_v1":
            from .upstream_flat_review import CudaSweepReviewer
            self.reviewer = CudaSweepReviewer(self.spec, version=REVIEW_V2)
        elif cfg.review_acceleration_version == "flat_grid_pose_cache_v2":
            from .upstream_flat_review import FlatGridPoseReviewer
            self.reviewer = FlatGridPoseReviewer(self.spec, version=REVIEW_V2)
        elif cfg.review_acceleration_version == "flat_frame_pose_cache_v3":
            from .upstream_flat_review import FlatFramePoseReviewer
            self.reviewer = FlatFramePoseReviewer(self.spec, version=REVIEW_V2)
        elif cfg.review_acceleration_version == "pose_scope_cache_v1":
            from .upstream_review_cache import PoseCachedReviewer
            self.reviewer = PoseCachedReviewer(self.spec, version=REVIEW_V2)
        elif cfg.review_acceleration_version == "disabled":
            self.reviewer = FootprintReviewer(self.spec, version=REVIEW_V2)
        else:
            raise ValueError("Unsupported footprint review acceleration version")
        self._identity = None
        self._previous_trajectory = None
        self._last_time = None

    def _gain(self, item):
        last_key = None; last_gain = None
        def gain(pose):
            nonlocal last_key, last_gain
            key = np.asarray(pose, dtype=float).tobytes()
            if key == last_key:return last_gain
            feature, known = item.terrain.lookup(pose[:2])
            # Unknown remains inadmissible in full review; use the existing
            # conservative gain only to produce a finite emergency prediction.
            last_gain = terrain_gain_numpy(pose, feature[0], self.spec) if known[0] else .25
            last_key = key
            return last_gain
        gain._stationary_repeatable = True
        return gain

    def validate_input(self, item, *, bind=False):
        if type(item) is not UpstreamPlannerInput:
            raise ValueError("Expected UpstreamPlannerInput")
        # Revalidate caller-owned mutable view metadata before touching state.
        item.__post_init__()
        expected_query=GridTerrain.query_version if self.cfg.obstacle_input is not None else QUERY_V2
        if item.spec.fingerprint != self.spec.fingerprint or item.terrain.query_version != expected_query:
            raise ValueError("Planner input profile/query version mismatch")
        if self.cfg.obstacle_input is not None and item.terrain.settings.metadata!=self.cfg.obstacle_input.metadata:
            raise ValueError('Planner grid input configuration mismatch')
        identity = (item.env_id, item.agent_id)
        if self._identity is not None and identity != self._identity:
            raise ValueError("Planner state belongs to another environment/vehicle")
        if self._last_time is not None and item.now < self._last_time - 1e-10:
            raise ValueError("Planner activation time moved backwards without reset")
        if self._last_time is not None:
            elapsed = (item.now - self._last_time) / self.spec.grid.dt
            if not math.isclose(elapsed, round(elapsed), abs_tol=1e-8):
                raise ValueError("Planner calls must advance by whole actual control steps")
        if self.cfg.timing_version == HELD_TIMING_VERSION and item.actual_history is not None:
            raise ValueError("Actual feedback history cannot enter the legacy held-control timing path")
        if self.cfg.timing_version == HISTORY_TIMING_VERSION and item.actual_history is None:
            raise ValueError("Actual-control timing requires explicit queue feedback history")
        if item.information_time is not None:
            steps = math.ceil(self.cfg.simulated_latency / self.spec.grid.dt - 1e-12)
            if not math.isclose(item.now, item.information_time + steps * self.spec.grid.dt, abs_tol=1e-9):
                raise ValueError("Information/activation time disagrees with rounded configured latency")
            if self.cfg.timing_version == HELD_TIMING_VERSION:
                held = rollout_numpy(item.information_pose, np.tile(item.previous_control, (steps, 1)), self.spec,
                                     item.information_time, gain=self._gain(item))
                if not np.allclose(held.states[-1], item.ego_pose, atol=1e-8, rtol=1e-8):
                    raise ValueError("Information-state latency prediction disagrees with actual activation pose")
        if bind:
            self._identity = identity
        return item

    def candidate_controls(self, item):
        raise NotImplementedError

    def score_trajectory(self, item, trajectory):
        """Return a reliable final physical cost or None; never reuse raw cost."""
        return None

    def prepare(self, item):
        self.validate_input(item, bind=True)
        if self.cfg.simulated_latency > 0 and item.information_pose is None:
            return self._brake(item, "missing_information_state", allow_valid=False)
        if item.now > item.intent.expires_at:
            return self._brake(item, "expired_intent", allow_valid=False)
        if item.intent.hold:
            return self._brake(item, "hold", allow_valid=True)
        return None

    def plan(self, item):
        prepared = self.prepare(item)
        if prepared is not None:
            return prepared
        start = time.perf_counter()
        candidates = self.candidate_controls(item)
        result = self.finish(item, candidates)
        result.feedback.solve_seconds = time.perf_counter() - start
        return result

    def _limited(self, desired, previous):
        last = previous.copy()
        result = []
        for target in desired:
            # Most adapters already generate legal controls. Preserve those
            # exactly and avoid solving the actuator polygons a second time.
            last = target.copy() if not control_error(target, last, self.spec) else limit_control_numpy(target, last, self.spec)
            result.append(last.copy())
        return np.asarray(result)

    def _stopping_controls(self, controls, previous, *, prefix_steps=None):
        limited = self._limited(controls, previous)
        if prefix_steps is None and np.max(abs(limited[-1])) < 1e-10:
            return limited, len(limited)
        # Longest prefix that leaves enough steps for a complete joint stop.
        initial = len(limited) if prefix_steps is None else min(prefix_steps, len(limited))
        for count in range(initial, -1, -1):
            last = limited[count - 1] if count else previous
            rates = np.array([self.spec.vehicle.linear_deceleration, self.spec.vehicle.angular_deceleration]) * self.spec.grid.dt
            if count + math.ceil(float(np.max(abs(last) / rates)) - 1e-9) > self.spec.grid.steps:
                continue
            brake = brake_sequence(last, self.spec)
            if count + len(brake) <= self.spec.grid.steps:
                return np.vstack([limited[:count], brake, np.zeros((self.spec.grid.steps - count - len(brake), 2))]), count
        raise ValueError("Current control cannot stop inside the planner horizon")

    def trajectory_from_controls(self, item, controls, *, prefix_steps=None):
        commands, prefix = self._stopping_controls(controls, item.previous_control, prefix_steps=prefix_steps)
        return rollout_numpy(item.ego_pose, commands, self.spec, item.now, gain=self._gain(item)), prefix

    def _review(self, item, trajectory):
        if item.intent.speed_cap is not None:
            # A lowered cap never clips the physical initial actuator state.
            # Its only admitted transient is the existing full brake sequence.
            brake = brake_sequence(item.previous_control, self.spec)
            limit = np.maximum(item.intent.speed_cap, np.pad(abs(brake[:, 0]), (0, max(0, 40 - len(brake))))[:40])
            if (abs(trajectory.controls[:, 0]) > limit + 1e-8).any():
                return "intent_speed_cap"
        if self.cfg.obstacle_input is not None and self.cfg.obstacle_input.graded_craters:
            # This hard check consumes only distance/sign. Known-footprint
            # certification, when enabled, belongs to the full reviewer below.
            distance,_=item.terrain.obstacle_distance(trajectory.states,kind='crater_large',include_known=False)
            if (distance<0).any():return "large_crater"
        if not self.cfg.predictive_footprint_checks:
            return ""
        workspace = None if item.workspace is None else np.asarray(item.workspace).reshape(2, 2)
        report = self.reviewer.trajectory(trajectory, item.previous_control, item.terrain, neighbors=item.neighbors,
                                          workspace=workspace, require_stop=True)
        return report.reason if not report.passed else ""

    def _reserve(self, item, first_control):
        if not self.cfg.predictive_footprint_checks:
            return "", {'information_reserve_verified': False, 'reserve_review_performed': False}
        if self.cfg.timing_version == HISTORY_TIMING_VERSION:
            return review_history_reserve(item,first_control,self.reviewer,self._gain(item))
        if self.cfg.simulated_latency > 0 and item.information_pose is None:
            return "missing_information_state", {}
        pose = item.ego_pose if item.information_pose is None else item.information_pose
        info_time = item.now if item.information_time is None else item.information_time
        reserve = stop_envelope(pose, item.previous_control, first_control, self.spec,
                                information_time=info_time, latency=self.cfg.simulated_latency, gain=self._gain(item))
        detail = {"information_time": info_time, "activation_time": reserve.activation_time,
                  "delay_steps": reserve.delay_steps, "reserve_steps": len(reserve.trajectory.controls),
                  "reserve_center_distance": reserve.conservative_center_distance,
                  "information_reserve_verified": False}
        if not reserve.fits_horizon:
            return "stop_reserve_exceeds_horizon", detail
        workspace = None if item.workspace is None else np.asarray(item.workspace).reshape(2, 2)
        report = self.reviewer.trajectory(reserve.trajectory, item.previous_control, item.terrain, neighbors=item.neighbors,
                                          workspace=workspace, require_stop=True, conservative_reserve=True)
        detail["information_reserve_verified"] = report.passed
        return ("" if report.passed else "stop_reserve:" + report.reason), detail

    def _output(self, item, trajectory, valid, reason, source, diagnostics, *, prefix=0):
        if not self.cfg.predictive_footprint_checks:
            diagnostics = dict(diagnostics, execution_review_passed=False,
                execution_review_performed=False, information_reserve_verified=False,
                unverified_brake=source in {'hold','brake'},
                hazard_response_version=self.cfg.hazard_response_version)
        feedback = PlannerFeedback(status=source if valid else "emergency_brake", reason=reason,
                                   used_previous=source == "previous_plan", emergency_brake=not valid,
                                   prefix_steps=prefix, candidate_rejection=diagnostics.get("candidate_rejection", ""))
        deferred=getattr(self,'_defer_output_scoring',False)
        cost = self.score_trajectory(item, trajectory) if valid and not deferred else None
        if self.cfg.obstacle_input is not None:
            diagnostics=dict(diagnostics,selected_cost_parts=(dict(self.selected_cost_parts) if valid else {}))
        if cost is not None and not np.isfinite(cost):
            raise ValueError("Final trajectory cost must be finite or None")
        output=UpstreamPlannerOutput(trajectory, feedback, valid, reason, cost,
                                     {**diagnostics, "selected_source": source, "protocol_version": PROTOCOL_VERSION,
                                      "adapter_id": self.adapter_id, "control_activation_time": item.now})
        if not deferred:self._commit_output(item,output)
        return output

    def _commit_output(self,item,output):
        """Commit only a completed output; grid MPPI scores its batch first."""
        self._last_time=item.now
        if output.valid:self._previous_trajectory=deepcopy(output.trajectory)

    def _brake(self, item, reason, *, allow_valid=False, diagnostics=None):
        controls = np.zeros((40, 2))
        trajectory, _ = self.trajectory_from_controls(item, controls, prefix_steps=0)
        review_reason = self._review(item, trajectory)
        reserve_reason, detail = self._reserve(item, trajectory.controls[0])
        valid = allow_valid and not review_reason and not reserve_reason
        detail.update(diagnostics or {})
        detail.update(execution_review_passed=not review_reason, execution_review_reason=review_reason,
                      reserve_review_reason=reserve_reason, unverified_brake=bool(review_reason or reserve_reason))
        return self._output(item, trajectory, valid, reason if not valid else "", "hold" if valid else "brake", detail)

    def finish(self, item, candidates, diagnostics=None):
        prepared = self.prepare(item)
        if prepared is not None:
            return prepared
        details = dict(diagnostics or {})
        details.update(candidate_reviews=0, sampled_candidate_reviews=0, updated_mean_reviews=0,
                       fallback_reviews=0, reserve_reviews=0, candidate_rejections=[])
        first = None
        for candidate in candidates:
            if type(candidate) is not Candidate:
                raise ValueError("Planner candidates require typed controls and source")
            mean = candidate.source == "updated_mean"
            if mean and details["updated_mean_reviews"] >= 1:
                continue
            if not mean and details["sampled_candidate_reviews"] >= 16:
                continue
            details["updated_mean_reviews" if mean else "sampled_candidate_reviews"] += 1
            details["candidate_reviews"] += 1
            trajectory, prefix = self.trajectory_from_controls(item, candidate.controls)
            if first is None:
                first = candidate.controls
            reason = self._review(item, trajectory)
            if not reason:
                details["reserve_reviews"] += 1
                reason, reserve_details = self._reserve(item, trajectory.controls[0])
                details.update(reserve_details)
            if not reason:
                details.update(candidate.diagnostics)
                details["execution_review_passed"] = True
                return self._output(item, trajectory, True, "", candidate.source, details, prefix=prefix)
            details["candidate_rejections"].append({"source": candidate.source, "reason": reason})
        details["candidate_rejection"] = details["candidate_rejections"][0]["reason"] if details["candidate_rejections"] else (details.get("failure_reason") or "no_candidate")
        if not self.cfg.predictive_footprint_checks:
            # Without the final reviewer, an old/prefixed plan has no current
            # MPPI feasibility mask. Do not silently treat it as admissible.
            return self._brake(item, details["candidate_rejection"], diagnostics=details)
        # A small deterministic fallback panel has its own recorded review
        # budget; every prefix includes a complete rate-limited static tail.
        if first is not None:
            for count in (20, 10, 5, 1):
                trajectory, actual_prefix = self.trajectory_from_controls(item, first, prefix_steps=count)
                details["fallback_reviews"] += 1
                reason = self._review(item, trajectory)
                if not reason:
                    details["reserve_reviews"] += 1
                    reason, reserve_details = self._reserve(item, trajectory.controls[0])
                    details.update(reserve_details)
                if not reason:
                    return self._output(item, trajectory, True, details["candidate_rejection"], "feasible_prefix", details, prefix=actual_prefix)
        if self._previous_trajectory is not None:
            shift = int(math.floor((item.now - self._previous_trajectory.timestamp) / self.spec.grid.dt + 1e-9))
            if 0 <= shift < 40:
                old = self._previous_trajectory.controls[shift:]
                controls = np.vstack([old, np.zeros((40 - len(old), 2))])
                trajectory, _ = self.trajectory_from_controls(item, controls)
                details["fallback_reviews"] += 1
                reason = self._review(item, trajectory)
                if not reason:
                    details["reserve_reviews"] += 1
                    reason, reserve_details = self._reserve(item, trajectory.controls[0])
                    details.update(reserve_details)
                if not reason:
                    return self._output(item, trajectory, True, details["candidate_rejection"], "previous_plan", details)
        return self._brake(item, details["candidate_rejection"], diagnostics=details)

    def _adapter_state(self):
        return {}

    def _validate_adapter_state(self, state):
        if state != {}:
            raise ValueError("Unexpected adapter state")

    def _load_adapter_state(self, state):
        pass

    def _reset_adapter(self):
        pass

    def reset(self):
        self._reset_adapter()
        self._identity = None
        self._previous_trajectory = None
        self._last_time = None

    def state_dict(self):
        return deepcopy({"contracts": self.cfg.metadata, "adapter_id": self.adapter_id,
                         "state_version": self.state_version, "identity": self._identity,
                         "last_time": self._last_time, "previous_trajectory": self._previous_trajectory,
                         "adapter_state": self._adapter_state()})

    def load_state_dict(self, state):
        incoming = deepcopy(state)
        self.cfg.validate_state_identity(incoming)
        if incoming.get("adapter_id") != self.adapter_id or incoming.get("state_version") != self.state_version:
            raise ValueError("Planner adapter/state version mismatch")
        identity = incoming.get("identity")
        if identity is not None and (not isinstance(identity, tuple) or len(identity) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in identity) or identity[1] >= 4):
            raise ValueError("Invalid restored planner identity")
        if self._identity is not None and identity != self._identity:
            raise ValueError("Cannot restore another vehicle's planner state")
        last_time = incoming.get("last_time")
        if last_time is not None and (not np.isfinite(last_time) or last_time < 0 or identity is None):
            raise ValueError("Invalid restored planner time")
        previous = incoming.get("previous_trajectory")
        if previous is not None:
            if type(previous) is not TimedTrajectory or identity is None or last_time is None:
                raise ValueError("Invalid restored planner trajectory")
            for name, shape in (("states", (41, 3)), ("controls", (40, 2)), ("velocities", (40, 2)), ("times", (41,))):
                _array(getattr(previous, name), shape, "Restored " + name)
            if not np.isfinite(previous.timestamp) or not 0 <= previous.timestamp <= last_time or not np.allclose(
                    previous.times, previous.timestamp + np.arange(41) * .2, atol=1e-10, rtol=0):
                raise ValueError("Invalid restored trajectory time grid")
            if np.max(abs(previous.controls[-1])) > 1e-8:
                raise ValueError("Restored trajectory lacks stationary tail")
            for step, command in enumerate(previous.controls):
                prior = previous.controls[step - 1] if step else command
                if control_error(command, prior, self.spec):
                    raise ValueError("Illegal restored trajectory control")
        extra = incoming.get("adapter_state")
        self._validate_adapter_state(extra)
        self._load_adapter_state(extra)
        self._identity, self._last_time, self._previous_trajectory = identity, last_time, previous
