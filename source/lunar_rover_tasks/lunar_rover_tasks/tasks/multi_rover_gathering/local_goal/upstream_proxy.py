"""Versioned no-learning actuator proxy for S01 integration.

This core advances physical state only. It does not yet implement the S03
sensor/communication/task loop, and cannot be passed to the legacy trainer.
"""

from copy import deepcopy

import numpy as np
import torch

from .upstream_execution import control_error, limit_control_torch, midpoint_torch
from .upstream_spec import UpstreamRuntimeConfig


class UpstreamActuatorProxy:
    state_version = "upstream_actuator_proxy_state_v1"

    def __init__(self, cfg, initial_poses):
        if not isinstance(cfg, UpstreamRuntimeConfig):
            raise ValueError("Upstream actuator proxy requires its versioned runtime config")
        self.cfg = cfg; self.spec = cfg.spec
        self.shape = (cfg.simulation.num_envs, cfg.spec.task.n_agents)
        self.poses = self._array(initial_poses, (*self.shape, 3), "initial poses")
        self.previous_control = np.zeros((*self.shape, 2))
        self.actual_velocity = np.zeros((*self.shape, 2))
        self.times = np.zeros(self.shape[0])
        self.episodes = np.zeros(self.shape[0], dtype=np.int64)

    @staticmethod
    def _array(value, shape, name):
        array = np.asarray(value, dtype=float).copy()
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{name} requires finite shape {shape}")
        return array

    def step(self, desired, *, gains=None, active_envs=None):
        """Bound command, integrate, and record actual control for hot starts.

        Caller supplies legal local terrain gains. There is no delay added here;
        injected delay is represented by explicitly stepping held controls.
        """
        target = self._array(desired, (*self.shape, 2), "desired controls")
        gain = np.ones(self.shape) if gains is None else self._array(gains, self.shape, "terrain gains")
        active = np.ones(self.shape[0], dtype=bool) if active_envs is None else np.asarray(active_envs)
        if active.shape != (self.shape[0],) or active.dtype != np.bool_:
            raise ValueError("Active environments require a boolean mask")
        if ((gain < .25) | (gain > 1)).any():
            raise ValueError("Terrain gains outside [.25,1]")
        applied = limit_control_torch(torch.from_numpy(target), torch.from_numpy(self.previous_control), self.spec)
        poses, actual = midpoint_torch(torch.from_numpy(self.poses), applied, self.spec, gain=torch.from_numpy(gain))
        before = self.poses.copy(); activation = self.times.copy(); previous = self.previous_control.copy()
        # Publish only after all shape/value/model checks have succeeded.
        self.poses[active] = poses.numpy()[active]
        self.previous_control[active] = applied.numpy()[active]
        self.actual_velocity[active] = actual.numpy()[active]
        self.times[active] += self.spec.grid.dt
        distances = np.linalg.norm(self.poses[:, :, None, :2] - self.poses[:, None, :, :2], axis=-1)
        mask = np.triu(np.ones((self.shape[1], self.shape[1]), dtype=bool), 1)
        minimum = np.min(np.where(mask, distances, np.inf), axis=(-2, -1))
        return {"activation_time": activation, "before": before, "after": self.poses.copy(),
                "desired_control": target, "previous_control": previous,
                "applied_control": self.previous_control.copy(), "actual_velocity": self.actual_velocity.copy(),
                "active_envs": active.copy(), "geometry_overlap": minimum < 2 * self.spec.vehicle.physical_radius,
                "clearance_violation": minimum < 2 * self.spec.vehicle.effective_radius,
                "safety_projection": "not_integrated_at_S01", "task_termination": "not_integrated_at_S01"}

    def reset(self, env_ids, poses):
        ids = np.asarray(env_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(set(ids.tolist())) != len(ids) or (ids < 0).any() or (ids >= self.shape[0]).any():
            raise ValueError("Invalid asynchronous reset environment ids")
        fresh = self._array(poses, (len(ids), self.shape[1], 3), "reset poses")
        self.poses[ids] = fresh; self.previous_control[ids] = 0.; self.actual_velocity[ids] = 0.
        self.times[ids] = 0.; self.episodes[ids] += 1

    def state_dict(self):
        return {"contracts": self.cfg.metadata, "state_version": self.state_version,
                "poses": self.poses.copy(), "previous_control": self.previous_control.copy(),
                "actual_velocity": self.actual_velocity.copy(), "times": self.times.copy(),
                "episodes": self.episodes.copy()}

    def load_state_dict(self, state):
        self.cfg.validate_state_identity(state)
        if state.get("state_version") != self.state_version:
            raise ValueError("Upstream proxy state version mismatch")
        poses = self._array(state.get("poses"), (*self.shape, 3), "snapshot poses")
        previous = self._array(state.get("previous_control"), (*self.shape, 2), "snapshot controls")
        actual = self._array(state.get("actual_velocity"), (*self.shape, 2), "snapshot velocities")
        times = self._array(state.get("times"), (self.shape[0],), "snapshot times")
        episodes = np.asarray(state.get("episodes"))
        if (times < 0).any() or not np.allclose(times / self.spec.grid.dt, np.round(times / self.spec.grid.dt), atol=1e-7, rtol=0):
            raise ValueError("Snapshot time outside the control grid")
        if episodes.shape != (self.shape[0],) or episodes.dtype.kind not in "iu" or (episodes < 0).any():
            raise ValueError("Invalid snapshot episode counters")
        for command in previous.reshape(-1, 2):
            if control_error(command, command, self.spec):raise ValueError("Invalid snapshot actuator state")
        for command, velocity in zip(previous.reshape(-1, 2), actual.reshape(-1, 2)):
            nonzero = abs(command) > 1e-10
            if (abs(velocity[~nonzero]) > 1e-8).any():raise ValueError("Invalid snapshot velocity gain")
            ratios = velocity[nonzero] / command[nonzero]
            if len(ratios) and ((ratios < .25 - 1e-8).any() or (ratios > 1 + 1e-8).any() or np.ptp(ratios) > 1e-8):
                raise ValueError("Invalid snapshot velocity gain")
        # All validation precedes mutation, so rejected loads leave the run intact.
        self.poses = poses; self.previous_control = previous; self.actual_velocity = actual
        self.times = times; self.episodes = deepcopy(episodes)
