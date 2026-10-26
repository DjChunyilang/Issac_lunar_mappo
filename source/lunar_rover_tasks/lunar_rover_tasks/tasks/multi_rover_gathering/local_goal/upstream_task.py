"""Actual-state finite gathering and continuous parking contract for S03."""

from copy import deepcopy

import numpy as np

from .upstream_footprint import FootprintReport
from .upstream_spec import UpstreamSpec

TASK_VERSION = "finite_task_upstream_v1"
STATE_VERSION = "upstream_task_state_v1"
REASONS = {"none", "environment_invalid", "collision", "out_of_bounds", "success", "task_deadline", "terrain_failure"}


class UpstreamTask:
    """Task evaluation may see actual team state; it is never a planner input."""

    def __init__(self, spec, num_envs, episode_seconds=360., *, terrain_failure_terminal=False):
        if type(spec) is not UpstreamSpec:
            raise ValueError("Actual task requires an immutable upstream profile")
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("Invalid task environment count")
        if np.asarray(episode_seconds).dtype == bool:
            raise ValueError("Task deadline cannot be boolean")
        limits = np.broadcast_to(np.asarray(episode_seconds, float), (num_envs,)).copy()
        if not np.isfinite(limits).all() or (limits <= 0).any():
            raise ValueError("Invalid finite task deadline")
        self.spec, self.num_envs, self.deadlines = spec, num_envs, limits
        self.terrain_failure_terminal = terrain_failure_terminal
        self.hold_steps = round(spec.task.hold_duration / spec.grid.dt)
        self._count = np.zeros(num_envs, dtype=np.int64)
        self._anchors = np.full((num_envs, 4, 3), np.nan)
        self._last_time = np.zeros(num_envs)
        self._reason = np.full(num_envs, "none", dtype=object)

    @property
    def metadata(self):
        return {**({"terrain_failure_semantics":"known_terrain_terminal_v1"} if self.terrain_failure_terminal else {}), "task_version": TASK_VERSION, "state_version": STATE_VERSION,
                "profile_fingerprint": self.spec.fingerprint, "num_envs": self.num_envs,
                "episode_seconds": self.deadlines.tolist(), "hold_steps": self.hold_steps}

    def advance(self, poses, actual_velocity, parking_reports, times, *, environment_valid=None, active_envs=None, terrain_failed=None):
        shape = (self.num_envs, 4)
        poses, speed, times = np.asarray(poses, float), np.asarray(actual_velocity, float), np.asarray(times, float)
        if poses.shape != (*shape, 3) or speed.shape != (*shape, 2) or times.shape != (self.num_envs,):
            raise ValueError("Task requires actual [E,4,3] poses, [E,4,2] speeds and [E] times")
        if not np.isfinite(times).all() or (times < 0).any():
            raise ValueError("Invalid actual task time")
        active = np.ones(self.num_envs, bool) if active_envs is None else np.asarray(active_envs)
        valid = np.ones(self.num_envs, bool) if environment_valid is None else np.asarray(environment_valid)
        if active.shape != (self.num_envs,) or valid.shape != active.shape or active.dtype != bool or valid.dtype != bool:
            raise ValueError("Task validity/activity must be boolean [E]")
        if len(parking_reports) != self.num_envs or any(len(row) != 4 for row in parking_reports):
            raise ValueError("Task needs four individual parking reports per environment")
        if any(type(report) is not FootprintReport for row in parking_reports for report in row):
            raise ValueError("Task needs typed local footprint facts")
        if (active & (self._reason != "none")).any():
            raise ValueError("Ended task environments require asynchronous reset")
        if (times[active] < self._last_time[active] - 1e-10).any():
            raise ValueError("Actual task time moved backwards")
        elapsed = (times[active] - self._last_time[active]) / self.spec.grid.dt
        if not np.allclose(elapsed, 1., atol=1e-8, rtol=0):
            raise ValueError("Continuous task hold requires one actual control step")
        finite = np.isfinite(poses).all((1, 2)) & np.isfinite(speed).all((1, 2))
        valid = valid & finite
        failed = np.zeros(self.num_envs, bool) if terrain_failed is None else np.asarray(terrain_failed)
        if failed.shape != valid.shape or failed.dtype != bool or (failed.any() and not self.terrain_failure_terminal):
            raise ValueError('Known-terrain task failures require the explicit terminal semantics')
        clean = np.where(finite[:, None, None], poses, 0.)
        velocity = np.where(finite[:, None, None], speed, 0.)
        center = clean[..., :2].mean(1)
        distances = np.linalg.norm(clean[:, :, None, :2] - clean[:, None, :, :2], axis=-1)
        pair = np.triu(np.ones((4, 4), bool), 1)
        minimum = np.min(np.where(pair, distances, np.inf), axis=(1, 2))
        dmax = distances.max((1, 2))
        dispersion = ((clean[..., :2] - center[:, None])**2).sum(-1).mean(-1)
        vehicle, task = self.spec.vehicle, self.spec.task
        collision = minimum < 2 * vehicle.physical_radius - 1e-10
        clearance = minimum >= 2 * vehicle.effective_radius - 1e-10
        outside = ((abs(clean[..., 0]) + vehicle.effective_radius > self.spec.world_width / 2 + 1e-10)
                   | (abs(clean[..., 1]) + vehicle.effective_radius > self.spec.world_height / 2 + 1e-10)).any(-1)
        containment = (np.linalg.norm(clean[..., :2] - center[:, None], axis=-1)
                       + vehicle.effective_radius <= task.worksite_radius + 1e-10).all(-1)
        stopped = ((abs(velocity[..., 0]) <= task.linear_stop + 1e-10)
                   & (abs(velocity[..., 1]) <= task.angular_stop + 1e-10)
                   & (abs(velocity[..., 0]) + vehicle.physical_radius * abs(velocity[..., 1]) <= task.boundary_stop + 1e-10)).all(-1)
        terrain = np.asarray([[
            type(r.passed) is bool and r.passed and r.checked_poses >= 1
            and r.minimum_body_clearance is not None
            and np.isfinite([r.slope, r.roughness_rms, r.maximum_step, r.minimum_body_clearance]).all()
            and 0 <= r.slope <= self.spec.terrain.parking_slope + 1e-8
            and 0 <= r.roughness_rms <= self.spec.terrain.roughness_rms + 1e-8
            and 0 <= r.maximum_step <= self.spec.terrain.step_height + 1e-8
            and r.minimum_body_clearance >= -1e-8 for r in row] for row in parking_reports]).all(-1)
        instant = valid & ~failed & ~collision & ~outside & clearance & containment & stopped & terrain
        instant &= (dmax <= task.dmax + 1e-10) & (dispersion <= task.dispersion + 1e-10)
        count, anchors, reasons = self._count.copy(), self._anchors.copy(), self._reason.copy()
        drift_broken = np.zeros(self.num_envs, bool)
        for env in np.flatnonzero(active):
            if instant[env] and count[env]:
                position = np.linalg.norm(clean[env, :, :2] - anchors[env, :, :2], axis=-1)
                heading = clean[env, :, 2] - anchors[env, :, 2]
                heading = np.arctan2(np.sin(heading), np.cos(heading))
                drift_broken[env] = ((position > task.position_drift + 1e-10).any()
                                     or (abs(heading) > task.heading_drift + 1e-10).any())
            if not instant[env] or drift_broken[env]:
                count[env] = 0; anchors[env] = np.nan
            else:
                if not count[env]:anchors[env] = clean[env]
                count[env] += 1
            if not valid[env]:reasons[env] = "environment_invalid"
            elif collision[env]:reasons[env] = "collision"
            elif outside[env]:reasons[env] = "out_of_bounds"
            elif failed[env]:reasons[env] = "terrain_failure"
            elif count[env] >= self.hold_steps:reasons[env] = "success"
            elif times[env] >= self.deadlines[env] - 1e-10:reasons[env] = "task_deadline"
        self._count, self._anchors, self._reason = count, anchors, reasons
        self._last_time[active] = times[active]
        terminal = reasons != "none"
        return {"terminal_reason": reasons.copy(), "terminated": terminal,
                "ended_this_step": active & terminal,
                "truncated": np.zeros(self.num_envs, bool), "bootstrap_allowed": ~terminal,
                "learning_valid": active & valid, "hold_count": count.copy(), "instant_success_conditions": instant,
                "drift_broken": drift_broken, "dmax": dmax, "dispersion": dispersion,
                "minimum_center_distance": minimum, "clearance_satisfied": clearance,
                "contained": containment, "actual_stopped": stopped, "parking_terrain_satisfied": terrain,
                "environment_valid": valid, "remaining_time": np.clip((self.deadlines - times) / self.deadlines, 0., 1.)}

    def reset(self, env_ids):
        ids = np.asarray(env_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(set(ids.tolist())) != len(ids) or (ids < 0).any() or (ids >= self.num_envs).any():
            raise ValueError("Invalid asynchronous task reset IDs")
        self._count[ids] = 0; self._anchors[ids] = np.nan; self._last_time[ids] = 0.; self._reason[ids] = "none"

    def state_dict(self):
        return deepcopy({"contracts": self.metadata, "hold_count": self._count, "anchors": self._anchors,
                         "last_time": self._last_time, "terminal_reason": self._reason})

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("contracts") != self.metadata:
            raise ValueError("Task snapshot version/profile/deadline mismatch")
        count = np.asarray(state.get("hold_count"));anchors = np.asarray(state.get("anchors"), float)
        times = np.asarray(state.get("last_time"), float);reason = np.asarray(state.get("terminal_reason"), object)
        if count.shape != (self.num_envs,) or count.dtype.kind not in "iu" or (count < 0).any() or (count > self.hold_steps).any():
            raise ValueError("Invalid task hold counts")
        if anchors.shape != (self.num_envs, 4, 3) or times.shape != count.shape or reason.shape != count.shape:
            raise ValueError("Invalid task snapshot shapes")
        if not np.isfinite(times).all() or (times < 0).any() or not np.allclose(times / self.spec.grid.dt, np.round(times / self.spec.grid.dt), atol=1e-8, rtol=0):
            raise ValueError("Invalid restored actual task time")
        if any(x not in REASONS or (x == "terrain_failure" and not self.terrain_failure_terminal) for x in reason) or not np.isfinite(anchors[count > 0]).all() or not np.isnan(anchors[count == 0]).all():
            raise ValueError("Invalid restored task reason/anchors")
        if ((reason == "success") != (count == self.hold_steps)).any() or ((reason == "task_deadline") & (times < self.deadlines - 1e-10)).any():
            raise ValueError("Task reason disagrees with restored hold/time")
        if ((reason == "none") & (times >= self.deadlines - 1e-10)).any() or (times < count * self.spec.grid.dt - 1e-10).any():
            raise ValueError("Impossible restored task time/count")
        if (np.isin(reason, ["environment_invalid", "collision", "out_of_bounds", "terrain_failure"]) & (count != 0)).any():
            raise ValueError("Unsafe terminal state cannot retain a success hold")
        self._count, self._anchors, self._last_time, self._reason = count.copy(), anchors.copy(), times.copy(), reason.copy()
