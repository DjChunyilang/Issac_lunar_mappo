"""Execution contracts; all poses are in the receiver's fixed odometry frame."""

from dataclasses import dataclass

import numpy as np

VERSION = 1
DT = 0.2
N = 15


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
