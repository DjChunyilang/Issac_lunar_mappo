"""Full local footprint review and information-time stop corridor admission.

Only detached observed samples and identified predictions enter this module.
The rectangle checks terrain; circles check neighbors/workspace. Neither a
center support disk nor a geometric worksite circle certifies terrain support.
"""

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.spatial import cKDTree

from .contracts import TimedTrajectory, wrap
from .protocol import TerrainSnapshot
from .upstream_execution import control_error, limit_control_numpy, midpoint_numpy, stop_envelope, terrain_gain_numpy

REVIEW_V2 = "footprint_review_v2"
QUERY_V2 = "batch_ties_v2"
NEIGHBOR_V2 = "identified_neighbor_upstream_v2"
QUERY_BLOCK = 256
TIE_ENTRY_BUDGET = 65536


class FootprintTerrain:
    """Owned, read-only local point snapshot; no memory owner/global map refs."""

    __slots__ = ("_xy", "_features", "_timestamps", "_tree", "support_radius", "query_version", "profile_fingerprint",
                 "_uniform_features", "_review_witnesses", "_memory_cells")

    def __init__(self, snapshot, spec, *, query_version=QUERY_V2):
        if type(snapshot) is not TerrainSnapshot:
            raise ValueError("FootprintTerrain requires a detached sensor snapshot")
        if not math.isclose(snapshot.support_radius, spec.perception.support_radius, abs_tol=1e-12):
            raise ValueError("Footprint terrain support/profile mismatch")
        if query_version not in {"scalar_ties_v1", QUERY_V2}:
            raise ValueError("Unsupported footprint terrain query version")
        if len(snapshot.xy) > spec.perception.capacity:
            raise ValueError("Footprint terrain exceeds the profile memory capacity")
        self.query_version = query_version; self.profile_fingerprint = spec.fingerprint
        self._xy = snapshot.xy.copy(); self._features = snapshot.features.copy()
        self._timestamps = snapshot.observed_at.copy()
        for a in (self._xy, self._features, self._timestamps):
            a.setflags(write=False)
        self._tree = cKDTree(self._xy)
        self.support_radius = snapshot.support_radius
        self._uniform_features = bool(len(self._features) and np.all(self._features == self._features[:1]))
        # Only geometry/terrain witnesses live here; never dynamic acceptance.
        self._review_witnesses = {}
        self._memory_cells = None

    def with_observation_times(self, times):
        """New detached view sharing immutable geometry and its spatial index."""
        times = np.asarray(times, dtype=float)
        if times.shape != self._timestamps.shape or not np.isfinite(times).all():
            raise ValueError("Observation times must match the detached terrain")
        result = object.__new__(type(self))
        for name in ("_xy", "_features", "_tree", "support_radius", "query_version", "profile_fingerprint", "_uniform_features", "_memory_cells"):
            setattr(result, name, getattr(self, name))
        result._timestamps = times.copy(); result._timestamps.setflags(write=False)
        # Positive witnesses certify immutable geometry/attributes. Timestamp
        # refresh cannot invalidate the observed supporting subset. A failed
        # search is retried because frame grouping can change with timestamps.
        result._review_witnesses = {key: value for key, value in self._review_witnesses.items() if value is not None}
        return result

    @classmethod
    def _from_owned_arrays(cls, xy, features, timestamps, spec):
        """Internal transfer of fresh memory outputs; no caller may mutate them."""
        if (xy.shape != (len(timestamps), 2) or features.shape != (len(timestamps), 5)
                or len(xy) > spec.perception.capacity
                or not all(np.isfinite(a).all() for a in (xy, features, timestamps))):
            raise ValueError("Invalid owned terrain arrays")
        result = object.__new__(cls)
        result._xy, result._features, result._timestamps = xy, features, timestamps
        for a in (xy, features, timestamps):a.setflags(write=False)
        result.query_version = QUERY_V2; result.profile_fingerprint = spec.fingerprint
        result.support_radius = spec.perception.support_radius
        result._tree = cKDTree(xy)
        result._uniform_features = bool(len(features) and np.all(features == features[:1]))
        result._review_witnesses = {}; result._memory_cells = None
        return result

    def readonly_snapshot(self):
        """Internal borrowed view; public snapshot() keeps its detached contract."""
        from types import SimpleNamespace
        return SimpleNamespace(xy=self._xy, features=self._features,
                               observed_at=self._timestamps, support_radius=self.support_radius)

    def snapshot(self):
        return TerrainSnapshot(self._xy, self._features, self._timestamps, self.support_radius)

    def lookup(self, xy):
        points = np.atleast_2d(np.asarray(xy, dtype=float))
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Footprint query requires finite [M,2]")
        if not len(self._xy):
            return np.zeros((len(points), 5)), np.zeros(len(points), dtype=bool)
        if self._uniform_features:
            # Equal attributes make every nearest/tie selection identical.
            # Query support exactly as before: no filling unknown holes.
            distance, index = self._tree.query(points)
            return self._features[index].copy(), distance <= self.support_radius + 1e-10
        if self.query_version == QUERY_V2:
            features = np.empty((len(points), 5)); known = np.empty(len(points), dtype=bool)
            for start in range(0, len(points), QUERY_BLOCK):
                end = start + QUERY_BLOCK
                features[start:end], known[start:end] = self._batch_lookup(points[start:end])
            return features, known
        distance, index = self._tree.query(points)
        features = self._features[index].copy()
        known = distance <= self.support_radius + 1e-10
        if len(self._xy) > 1:
            nearest, _ = self._tree.query(points, k=2)
            ties = np.isclose(nearest[:, 0], nearest[:, 1], atol=1e-8, rtol=0)
            for i in np.flatnonzero(ties & known):
                ids = self._tree.query_ball_point(points[i], distance[i] + 1e-8)
                peer = self._features[ids]
                # Preserve dangerous heights, slopes and traversability in ties.
                features[i, 0] = peer[:, 0].max()
                features[i, 1:3] = peer[np.linalg.norm(peer[:, 1:3], axis=-1).argmax(), 1:3]
                features[i, 4] = peer[:, 4].min()
        return features, known

    def _batch_lookup(self, points):
        # Keep k=1 and k=2: the original source choice for channel 3 is part of
        # the five-channel contract, including otherwise identical ties.
        distance, index = self._tree.query(points)
        features = self._features[index].copy(); known = distance <= self.support_radius + 1e-10
        if len(self._xy) < 2:return features, known
        nearest, _ = self._tree.query(points, k=2)
        ties = np.flatnonzero(np.isclose(nearest[:, 0], nearest[:, 1], atol=1e-8, rtol=0) & known)
        if not len(ties):return features, known
        lengths = self._tree.query_ball_point(points[ties], distance[ties] + 1e-8, return_length=True)
        # Count first, then bound the ragged neighbor allocation even for a
        # memory full of coincident observations. No query x memory matrix.
        block = max(1, min(QUERY_BLOCK, TIE_ENTRY_BUDGET // int(lengths.max())))
        for start in range(0, len(ties), block):
            ids = ties[start:start + block]
            peers = self._tree.query_ball_point(points[ids], distance[ids] + 1e-8, return_sorted=False)
            sizes = lengths[start:start + block]
            flat = np.concatenate(peers).astype(np.int64)
            offsets = np.r_[0, np.cumsum(sizes[:-1])]
            parent = np.repeat(np.arange(len(peers)), sizes)
            values = self._features[flat]
            features[ids, 0] = np.maximum.reduceat(values[:, 0], offsets)
            features[ids, 4] = np.minimum.reduceat(values[:, 4], offsets)
            norms = np.linalg.norm(values[:, 1:3], axis=-1)
            maxima = np.maximum.reduceat(norms, offsets)
            picks = np.minimum.reduceat(np.where(norms == maxima[parent], np.arange(len(norms)), len(norms)), offsets)
            features[ids, 1:3] = values[picks, 1:3]
        return features, known


@dataclass(frozen=True)
class FootprintNeighbor:
    sender_id: int
    times: np.ndarray
    positions: np.ndarray
    radii: np.ndarray
    profile_fingerprint: str | None = None
    version: str = "identified_neighbor_upstream_v1"

    def __post_init__(self):
        # Radius already includes own/peer geometry and communication uncertainty.
        if isinstance(self.sender_id, bool) or not isinstance(self.sender_id, int) or self.sender_id < 0:
            raise ValueError("Invalid neighbor sender")
        count = len(self.times)
        if count < 2:
            raise ValueError("Neighbor prediction requires at least two times")
        for name, shape in (("times", (count,)), ("positions", (count, 2)), ("radii", (count,))):
            a = np.asarray(getattr(self, name), dtype=float).copy()
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError("Invalid neighbor shape/value")
            a.setflags(write=False); object.__setattr__(self, name, a)
        if (np.diff(self.times) <= 0).any() or (self.radii <= 0).any():
            raise ValueError("Invalid neighbor time/radius")

    @classmethod
    def for_spec(cls, sender_id, times, positions, radii, spec):
        result = cls(sender_id, times, positions, radii, spec.fingerprint, NEIGHBOR_V2)
        reason = result.contract_error(spec, spec.fingerprint)
        if reason:raise ValueError(reason)
        return result

    def contract_error(self, spec, fingerprint):
        if (self.radii < 2 * spec.vehicle.effective_radius - 1e-10).any():
            return "neighbor_radius_below_geometry"
        if self.version != NEIGHBOR_V2 or self.profile_fingerprint != fingerprint:
            return "neighbor_geometry_contract_mismatch"
        return ""

    def at(self, time):
        if time < self.times[0] - 1e-9 or time > self.times[-1] + 1e-9:
            raise ValueError("Neighbor forecast does not cover the queried time")
        return (np.array([np.interp(time, self.times, self.positions[:, j]) for j in range(2)]),
                float(np.interp(time, self.times, self.radii)))


@dataclass(frozen=True)
class FootprintReport:
    passed: bool
    reason: str = ""
    slope: float = 0.
    roughness_rms: float = 0.
    maximum_step: float = 0.
    minimum_body_clearance: float | None = None
    checked_poses: int = 0


def _grid(length, width, spacing):
    x = np.linspace(-length / 2, length / 2, math.ceil(length / spacing) + 1)
    y = np.linspace(-width / 2, width / 2, math.ceil(width / spacing) + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.stack([xx, yy], axis=-1).reshape(-1, 2), (len(x), len(y))


def _world(points, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return points @ np.array([[c, s], [-s, c]]) + pose[:2]


class FootprintReviewer:
    """Finite-point kinematic proxy review, not PhysX or real vehicle proof."""

    def __init__(self, spec, *, version=REVIEW_V2):
        if version not in {"footprint_review_v1", REVIEW_V2}:
            raise ValueError("Unsupported footprint review version")
        self.version = version; self._fingerprint = spec.fingerprint
        self.spec = spec; v = spec.vehicle
        self.body_points, self.body_shape = _grid(v.length + 2 * v.clearance_margin,
                                                  v.width + 2 * v.clearance_margin, v.sweep_spacing)
        contact, _ = _grid(v.contact_length, v.contact_width, v.sweep_spacing)
        self.contact_points = np.vstack([contact + center for center in v.contact_centers])
        self.points = np.vstack([self.contact_points, self.body_points])
        self.sweep_radius = max(np.linalg.norm(self.points, axis=-1))

    def pose(self, pose, terrain, *, parking=False):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all() or type(terrain) is not FootprintTerrain:
            raise ValueError("Footprint requires finite pose and sensor-only terrain")
        if self.version == REVIEW_V2 and terrain.profile_fingerprint != self._fingerprint:
            raise ValueError("Footprint terrain profile mismatch")
        features, known = terrain.lookup(_world(self.points, pose))
        if not known.all():
            return FootprintReport(False, "unknown_footprint", checked_poses=1)
        if (features[:, 4] < self.spec.terrain.minimum_traversability).any():
            return FootprintReport(False, "impassable_footprint", checked_poses=1)
        n = len(self.contact_points); contact_height = features[:n, 0]
        # Fit the supporting plane from finite contact areas, not the whole body
        # or global DEM. Linear slopes are removed before RMS/step computation.
        design = np.c_[self.contact_points, np.ones(n)]
        plane, _, rank, _ = np.linalg.lstsq(design, contact_height, rcond=None)
        if rank < 3:
            return FootprintReport(False, "degenerate_contact_plane", checked_poses=1)
        contact_residual = contact_height - design @ plane
        rms = float(np.sqrt(np.mean(contact_residual**2)))
        slope = float(max(np.linalg.norm(plane[:2]), np.linalg.norm(features[:, 1:3], axis=-1).max()))
        body_plane = np.c_[self.body_points, np.ones(len(self.body_points))] @ plane
        body_residual = (features[n:, 0] - body_plane).reshape(self.body_shape)
        step = float(max(np.max(abs(np.diff(body_residual, axis=0))),
                         np.max(abs(np.diff(body_residual, axis=1))), np.ptp(contact_residual)))
        clearance = float(self.spec.vehicle.ground_clearance - body_residual.max())
        limit = self.spec.terrain.parking_slope if parking else self.spec.terrain.driving_slope
        if slope > limit + 1e-8:reason = "parking_slope" if parking else "driving_slope"
        elif rms > self.spec.terrain.roughness_rms + 1e-8:reason = "contact_roughness"
        elif clearance < -1e-8:reason = "body_clearance"
        elif step > self.spec.terrain.step_height + 1e-8:reason = "terrain_step"
        else:reason = ""
        return FootprintReport(not reason, reason, slope, rms, step, clearance, 1)

    def trajectory(self, trajectory, previous, terrain, *, neighbors=(), workspace=None,
                   require_stop=False, parking_end=False, conservative_reserve=False):
        if not isinstance(trajectory, TimedTrajectory):
            raise ValueError("Expected shared TimedTrajectory")
        if not isinstance(conservative_reserve, (bool, np.bool_)):
            raise ValueError("Conservative reserve marker must be boolean")
        count = len(trajectory.controls)
        if count < 1:
            raise ValueError("Full footprint review requires a nonempty control trajectory")
        for a, shape in ((trajectory.states, (count + 1, 3)), (trajectory.controls, (count, 2)),
                         (trajectory.velocities, (count, 2)), (trajectory.times, (count + 1,))):
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError("Trajectory shape/nonfinite error")
        if not np.allclose(np.diff(trajectory.times), self.spec.grid.dt, atol=1e-10, rtol=0):
            raise ValueError("Trajectory grid mismatch")
        if len({n.sender_id for n in neighbors}) != len(neighbors):
            raise ValueError("Duplicate sender predictions")
        if self.version == REVIEW_V2:
            if type(terrain) is not FootprintTerrain or terrain.profile_fingerprint != self._fingerprint:
                raise ValueError("Footprint terrain profile mismatch")
            if not np.isfinite(trajectory.timestamp) or trajectory.timestamp < 0 or not math.isclose(trajectory.timestamp, trajectory.times[0], abs_tol=1e-9):
                raise ValueError("Trajectory information time mismatch")
            for neighbor in neighbors:
                if type(neighbor) is not FootprintNeighbor:raise ValueError("Expected identified footprint neighbor")
                reason = neighbor.contract_error(self.spec, self._fingerprint)
                if reason:return FootprintReport(False, reason)
        if workspace is not None:
            workspace = np.asarray(workspace, dtype=float)
            if workspace.shape != (2, 2) or not np.isfinite(workspace).all() or (workspace[1] <= workspace[0]).any():
                raise ValueError("Workspace requires receiver-frame [min,max] bounds")
        last = np.asarray(previous, dtype=float)
        stationary_review = getattr(self, '_stationary_trajectory', None)
        if stationary_review is not None:
            report = stationary_review(trajectory, last, terrain, neighbors, workspace, parking_end)
            if report is not None:return report
        flat_review = getattr(self, '_flat_trajectory', None)
        if flat_review is not None:
            report = flat_review(trajectory, last, terrain, neighbors, workspace, require_stop, parking_end)
            if report is not None:return report
        poses = 0
        for i, control in enumerate(trajectory.controls):
            reason = control_error(control, last, self.spec)
            if reason:return FootprintReport(False, reason, checked_poses=poses)
            # Conservative reserves integrate incoming controls. Both paths still
            # obey the same actuator restrictions; neither may invent states.
            if self.version == "footprint_review_v1":
                predicted, _ = midpoint_numpy(trajectory.states[i], trajectory.velocities[i], self.spec)
                if not np.allclose(predicted, trajectory.states[i + 1], atol=1e-7, rtol=1e-7):
                    return FootprintReport(False, "dynamics_inconsistent", checked_poses=poses)
            speed = trajectory.velocities[i]
            if ((np.abs(control) < 1e-10) & (np.abs(speed) > 1e-8)).any():
                return FootprintReport(False, "velocity_gain_inconsistent", checked_poses=poses)
            ratios = speed[np.abs(control) > 1e-10] / control[np.abs(control) > 1e-10]
            if len(ratios) and ((ratios < .25 - 1e-7).any() or (ratios > 1 + 1e-7).any() or np.ptp(ratios) > 1e-7):
                return FootprintReport(False, "velocity_gain_inconsistent", checked_poses=poses)
            if self.version == REVIEW_V2:
                features, known = terrain.lookup(trajectory.states[i, :2])
                if not known[0]:return FootprintReport(False, "unknown_footprint", checked_poses=poses)
                gain = terrain_gain_numpy(trajectory.states[i], features[0], self.spec)
                expected, velocity = midpoint_numpy(trajectory.states[i], control, self.spec, gain=gain)
                single = any(a.dtype == np.float32 for a in (trajectory.states, trajectory.controls, trajectory.velocities))
                atol, rtol = (1e-5, 1e-5) if single else (1e-10, 1e-8)
                if not np.allclose(speed, velocity, atol=atol, rtol=rtol):
                    return FootprintReport(False, "velocity_gain_inconsistent", checked_poses=poses)
                if not np.allclose(expected, trajectory.states[i + 1], atol=atol, rtol=rtol):
                    return FootprintReport(False, "dynamics_inconsistent", checked_poses=poses)
            delta = trajectory.states[i + 1] - trajectory.states[i]
            delta[2] = wrap(delta[2])
            displacement = np.linalg.norm(delta[:2]) + self.sweep_radius * abs(delta[2])
            intervals = max(8, math.ceil(displacement / self.spec.vehicle.sweep_spacing))
            relative_bound = 0.
            if neighbors:
                # Growing reachable radii change the clearance function even
                # when the sender position is stationary. Include that change
                # once in the finite-sampling bound, not as another age margin.
                bounds = []
                for neighbor in neighbors:
                    p0, r0 = neighbor.at(trajectory.times[i])
                    p1, r1 = neighbor.at(trajectory.times[i + 1])
                    bounds.append(np.linalg.norm(p1 - p0) + abs(r1 - r0))
                relative_bound = max(bounds)
                intervals = max(intervals, math.ceil((displacement + relative_bound) / self.spec.vehicle.sweep_spacing))
            alphas = np.linspace(0, 1, intervals + 1)
            # Optional vectorized proof checks the very same swept samples.
            # Any uncertain/boundary case falls through to the scalar path,
            # preserving the first failure and checked-pose accounting.
            clear_sweep = getattr(self, "_clear_sweep", None)
            sweep = False if clear_sweep is None else clear_sweep(
                trajectory.states[i], delta, alphas, terrain, workspace,
                neighbors, trajectory.times[i], relative_bound, intervals)
            if isinstance(sweep, FootprintReport):
                return replace(sweep, checked_poses=poses + sweep.checked_poses)
            if sweep:
                poses += len(alphas)
                last = control
                continue
            for alpha in alphas:
                pose = trajectory.states[i] + alpha * delta
                report = self.pose(pose, terrain)
                poses += 1
                if not report.passed:
                    return FootprintReport(False, report.reason, report.slope, report.roughness_rms,
                                           report.maximum_step, report.minimum_body_clearance, poses)
                if workspace is not None and ((pose[:2] - self.spec.vehicle.effective_radius < workspace[0]).any() or
                                               (pose[:2] + self.spec.vehicle.effective_radius > workspace[1]).any()):
                    return FootprintReport(False, "workspace_clearance", checked_poses=poses)
                time = trajectory.times[i] + alpha * self.spec.grid.dt
                for neighbor in neighbors:
                    position, radius = neighbor.at(time)
                    # Finite swept checks include the half-sampling relative
                    # displacement allowance, without re-adding message age.
                    allowance = (np.linalg.norm(delta[:2]) + relative_bound) / (2 * intervals)
                    if np.linalg.norm(pose[:2] - position) < radius + allowance - 1e-8:
                        return FootprintReport(False, "neighbor_swept_clearance", checked_poses=poses)
            last = control
        if require_stop and (not count or np.max(abs(trajectory.controls[-1])) > 1e-8):
            return FootprintReport(False, "missing_stationary_tail", checked_poses=poses)
        if parking_end:
            end = self.pose(trajectory.states[-1], terrain, parking=True)
            if not end.passed:return end
        return FootprintReport(True, checked_poses=poses)


@dataclass(frozen=True)
class AdmissionDecision:
    control: np.ndarray
    verified: bool
    status: str
    reason: str
    dynamic_speed_cap: float
    transient_overspeed: float
    envelope: object
    checked_candidates: int


def admit_control(desired, previous, pose, terrain, spec, *, information_time=0., latency=.5,
                  neighbors=(), workspace=None, hold=False, candidates=24, review_version=REVIEW_V2,
                  reviewer=None):
    """Choose a rate-limited control with an independently verified full stop.

    Failure returns the bounded brake with an explicit unverified marker. A
    smaller speed cap never clips actual velocity or the preceding command.
    """
    if isinstance(candidates, bool) or not isinstance(candidates, int) or candidates < 2:
        raise ValueError("Admission needs at least two candidates")
    if reviewer is None:
        reviewer = FootprintReviewer(spec, version=review_version)
    elif (not isinstance(reviewer, FootprintReviewer) or reviewer.spec != spec
          or reviewer.version != review_version):
        raise ValueError("Admission reviewer must use the same profile and review version")
    def local_gain(state):
        features, _ = terrain.lookup(state[:2])
        return terrain_gain_numpy(state, features[0], spec)
    brake = limit_control_numpy([0., 0.], previous, spec)
    nominal = brake if hold else limit_control_numpy(desired, previous, spec)
    targets = [brake] if hold else [nominal * alpha for alpha in np.linspace(1., 0., candidates)]
    failure = "no_admissible_control"; checked = 0; seen = set(); last_envelope = None
    for target in targets:
        control = brake if hold else limit_control_numpy(target, previous, spec)
        key = tuple(np.round(control, 10))
        if key in seen:continue
        seen.add(key); checked += 1
        envelope = stop_envelope(pose, previous, control, spec, information_time=information_time, latency=latency, gain=local_gain)
        last_envelope = envelope
        if not envelope.fits_horizon:
            failure = "stop_horizon_exceeded"; continue
        report = reviewer.trajectory(envelope.trajectory, previous, terrain, neighbors=neighbors,
                                     workspace=workspace, require_stop=True, conservative_reserve=True)
        if report.passed:
            status = "hold_brake" if hold else "accepted" if key == tuple(np.round(nominal, 10)) else "dynamic_limited"
            return AdmissionDecision(control, True, status, "", abs(float(control[0])), 0., envelope, checked)
        failure = report.reason
    if last_envelope is None or not np.array_equal(last_envelope.trajectory.controls[last_envelope.delay_steps], brake):
        last_envelope = stop_envelope(pose, previous, brake, spec, information_time=information_time, latency=latency, gain=local_gain)
    return AdmissionDecision(brake, False, "unverified_brake", failure, 0., abs(float(brake[0])), last_envelope, checked)
