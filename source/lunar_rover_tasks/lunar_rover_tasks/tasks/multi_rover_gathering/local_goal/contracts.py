"""Execution contracts; all poses are in the receiver's fixed odometry frame."""

from dataclasses import dataclass

import numpy as np

VERSION = 1
DT = 0.2
N = 15


def contract_metadata(cfg):
    if getattr(cfg, "runtime_family", None) == "upstream_scale_v1":
        from .upstream_spec import UpstreamRuntimeConfig
        if not isinstance(cfg, UpstreamRuntimeConfig):
            raise ValueError("Expected immutable upstream runtime configuration")
        return cfg.metadata
    contract = cfg.local_goal_contract
    supported = {
        "legacy_v1": ("local_goal_v1",),
        "finite_task_v2": ("local_goal_clock_v2", "local_intent_clock_v3"),
    }
    if (
        not isinstance(contract.task_contract_version, str)
        or contract.task_contract_version not in supported
        or contract.observation_version not in supported[contract.task_contract_version]
    ):
        raise ValueError("Unsupported task/observation contract combination")
    from .registry import adapter_metadata
    state_version = adapter_metadata(contract.planner_adapter, contract.planner_revision)
    if not isinstance(contract.training_paused, bool):
        raise ValueError("local_goal.training_paused must be boolean")
    metadata = {
        "task_contract_version": contract.task_contract_version,
        "observation_version": contract.observation_version,
        "action_version": contract.action_version,
        "planner_revision": contract.planner_revision,
    }
    if contract.safety_projection not in ("disabled", "discrete_cbf_qp_v1"):
        raise ValueError("Unsupported safety projection version")
    if contract.safety_projection != "disabled":
        if contract.planner_protocol != "planner_protocol_v1" or not contract.training_paused:
            raise ValueError("Safety projection requires paused planner protocol diagnostics")
        metadata["safety"] = {"version": contract.safety_projection, "state_version": 1,
                              "deadline_s": contract.safety_deadline_s, "radius": .14}
    if contract.nmpc_deadline_s != .15:
        if not np.isfinite(contract.nmpc_deadline_s) or contract.nmpc_deadline_s <= 0:
            raise ValueError("Invalid NMPC deadline")
        metadata["nmpc_deadline_s"] = contract.nmpc_deadline_s
    if contract.no_learning_script not in ("disabled", "leader_slot_v1", "leader_local_site_v2", "leader_local_site_v3", "leader_local_site_v4", "leader_local_site_v5", "leader_local_site_v6", "leader_local_site_v7", "leader_local_site_v8", "leader_local_site_v9", "leader_local_site_v10"):
        raise ValueError("Unsupported no-learning script")
    if contract.no_learning_script != "disabled":
        if not contract.training_paused or contract.action_version != "polar_hold_v2":
            raise ValueError("No-learning script requires paused polar contract")
        metadata["script_version"] = contract.no_learning_script
    if contract.planner_protocol not in ("legacy_direct", "planner_protocol_v1"):
        raise ValueError("Unsupported planner protocol")
    if contract.planner_protocol == "legacy_direct" and contract.planner_adapter != "nmpc_bridge_v1":
        raise ValueError("Legacy path requires original NMPC adapter")
    if contract.planner_protocol == "planner_protocol_v1":
        if contract.task_contract_version != "finite_task_v2" or not contract.training_paused:
            raise ValueError("Planner protocol requires paused finite_task_v2 diagnostics")
        metadata.update(
            planner_protocol=contract.planner_protocol,
            planner_adapter=contract.planner_adapter,
            planner_state_version=state_version,
        )
    if contract.planner_adapter.startswith("mppi_"):
        from .mppi import MPPIConfig
        metadata["mppi"] = MPPIConfig.from_contract(contract, cfg.terrain.slope_speed_scale).metadata()
        metadata["planner_seed"] = contract.planner_seed
    if contract.action_version not in ("local_goal_request_v1", "polar_hold_v2"):
        raise ValueError("Unsupported action version")
    hybrid = contract.action_version == "polar_hold_v2"
    if hybrid != (contract.observation_version == "local_intent_clock_v3"):
        raise ValueError("Action/observation version mismatch")
    if hybrid:
        if contract.planner_protocol != "planner_protocol_v1":
            raise ValueError("Polar HOLD requires the unified planner protocol")
        if (
            not np.isfinite([contract.rho_max, contract.beta_max]).all()
            or contract.rho_max <= 0
            or not 0 < contract.beta_max <= np.pi
        ):
            raise ValueError("Invalid polar action bounds")
        metadata.update(
            rho_max=contract.rho_max,
            beta_max=contract.beta_max,
            rollout_version="polar_hold_rollout_v1",
        )
    return metadata


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


@dataclass
class LocalGoalRequest:
    target: np.ndarray  # x, y, yaw, frozen at decision time
    speed_cap: float
    timestamp: float
    expires_at: float
    version: int = VERSION

    @classmethod
    def for_hold(cls, pose, now):
        """Diagnostic HOLD request; actual slew-limited stopping is the planner's job."""
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("HOLD requires a finite current pose")
        return cls(pose.copy(), 0.0, now, now + 1.0)

    @classmethod
    def from_action(cls, action, pose, now):
        a = np.asarray(action, dtype=float)
        if a.shape != (4,) or not np.isfinite(a).all() or (abs(a) > 1.000001).any():
            raise ValueError("Expected four finite normalized bounded request actions")
        xy = pose[:2] + rotation(pose[2]) @ (1.6 * a[:2])
        return cls(
            np.r_[xy, wrap(pose[2] + np.pi * a[2])], float(0.575 * (a[3] + 1)), now, now + 1.0
        )


@dataclass
class TimedTrajectory:
    states: np.ndarray  # [N+1, 3]: x, y, yaw
    controls: np.ndarray  # [N, 2]: wheel-equivalent v, omega before slope loss
    velocities: np.ndarray  # [N, 2]: predicted actual v, omega
    times: np.ndarray
    timestamp: float
    version: int = VERSION


@dataclass
class PlannerFeedback:
    status: str = "reset"
    reason: str = ""
    solve_seconds: float = 0.0
    solver_status: int = 0
    blocked_seconds: float = 0.0
    used_previous: bool = False
    emergency_brake: bool = False
    version: int = VERSION
    solver_passes: int = 0
    model_error_m: float = 0.0
    candidate_rejection: str = ""
    prefix_steps: int = 0


@dataclass
class NeighborPrediction:
    positions: np.ndarray  # N+1,2 in receiver odometry frame
    radii: np.ndarray  # includes 0.42 task spacing + uncertainty/tracking margin
    timestamp: float
    expires_at: float
    version: int = VERSION
