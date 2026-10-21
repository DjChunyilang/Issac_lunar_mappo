"""S03 dense direct sensing and profile-bound local memory.

The sensor owner may query the terrain backend. Exported frames, memories and
planner views contain only observed point data, never backend/core references.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree
import torch

from ..terrain_features import query_terrain_features
from .protocol import TerrainSnapshot
from .upstream_footprint import FootprintTerrain
from .upstream_spec import UpstreamSpec

SENSOR_VERSION = "dense_sensor_frame_upstream_v1"
MEMORY_VERSION = "dense_memory_upstream_v1"


def _owned(value, shape, name, dtype=float):
    array = np.asarray(value, dtype=dtype).copy()
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} requires finite shape {shape}")
    array.setflags(write=False)
    return array


def world_to_odometry(xy, frame):
    """Owned NumPy points transformed to a receiver's fixed odometry frame."""
    frame = np.asarray(frame, dtype=float)
    if frame.shape != (3,) or not np.isfinite(frame).all():
        raise ValueError("Invalid fixed odometry frame")
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
        raise ValueError("Odometry transform requires finite [M,2] points")
    delta = points - frame[:2]
    c, s = np.cos(frame[2]), np.sin(frame[2])
    return delta @ np.array([[c, -s], [s, c]])


@dataclass(frozen=True)
class SensorFrame:
    env_id: int
    agent_id: int
    timestamp: float
    xy: np.ndarray
    features: np.ndarray
    valid: np.ndarray
    roughness_valid: np.ndarray
    sensor_pose: np.ndarray
    spec: UpstreamSpec
    map_identity: str
    version: str = SENSOR_VERSION

    def __post_init__(self):
        if type(self.spec) is not UpstreamSpec or self.version != SENSOR_VERSION:
            raise ValueError("Unsupported dense sensor profile/version")
        if (type(self.env_id) is not int or self.env_id < 0 or type(self.agent_id) is not int
                or not 0 <= self.agent_id < 4):
            raise ValueError("Invalid dense sensor identity")
        if not math.isfinite(self.timestamp) or self.timestamp < 0 or not isinstance(self.map_identity, str) or not self.map_identity:
            raise ValueError("Invalid dense sensor time/map identity")
        count = self.spec.perception.sample_count
        object.__setattr__(self, "xy", _owned(self.xy, (count, 2), "Sensor coordinates"))
        object.__setattr__(self, "features", _owned(self.features, (count, 5), "Sensor features"))
        object.__setattr__(self, "sensor_pose", _owned(self.sensor_pose, (3,), "Sensor pose"))
        for name in ("valid", "roughness_valid"):
            original = np.asarray(getattr(self, name))
            if original.dtype != np.bool_:
                raise ValueError("Sensor masks must be boolean")
            object.__setattr__(self, name, _owned(original, (count,), name, bool))
        if (self.roughness_valid & ~self.valid).any():
            raise ValueError("Roughness cannot be known outside the sensor support")
        if ((self.features[:, 4] < 0) | (self.features[:, 4] > 1)).any() or (self.features[:, 3] < 0).any():
            raise ValueError("Invalid observed terrain attribute")
        if (self.features[~self.valid] != 0).any():
            raise ValueError("Unknown sensor samples require zero placeholders")
        if (abs(self.xy / self.spec.perception.memory_resolution) > 2**40).any():
            raise ValueError("Sensor coordinates exceed bounded local indexing")
        p = self.spec.perception
        xx, yy = np.meshgrid(np.linspace(-p.reverse, p.forward, p.frame_shape[0]),
                             np.linspace(-p.lateral, p.lateral, p.frame_shape[1]), indexing="ij")
        c, s = np.cos(self.sensor_pose[2]), np.sin(self.sensor_pose[2])
        expected = np.c_[c * xx.ravel() - s * yy.ravel(), s * xx.ravel() + c * yy.ravel()] + self.sensor_pose[:2]
        if not np.allclose(self.xy, expected, atol=1e-9, rtol=1e-9):
            raise ValueError("Sensor positions must match the legal dense field of view")

    def snapshot(self):
        return TerrainSnapshot(self.xy[self.valid], self.features[self.valid],
                               np.full(int(self.valid.sum()), self.timestamp), self.spec.perception.support_radius)


class DenseSensor:
    """Global terrain access is confined to this sensor owner.

    Dense samples are queried directly. Actor grids are independent consumers
    of local memory and are never used as the sensor source.
    """
    version = SENSOR_VERSION

    def __init__(self, spec, terrain_cfg, *, runtime=None, map_identities=("flat_proxy",), device="cpu"):
        if type(spec) is not UpstreamSpec:
            raise ValueError("Dense sensing requires an immutable profile")
        self.spec = spec
        self._terrain_cfg = deepcopy(terrain_cfg)
        if terrain_cfg is not None and terrain_cfg.type == "lunar_scene_pack" and (
                not math.isfinite(terrain_cfg.raster_risk_slope_scale) or terrain_cfg.raster_risk_slope_scale <= 0
                or not math.isfinite(terrain_cfg.raster_risk_rms_scale_m) or terrain_cfg.raster_risk_rms_scale_m <= 0):
            raise ValueError("Raster sensor requires positive finite risk scales")
        self._runtime = runtime
        self._map_identities = tuple(map_identities)
        if not self._map_identities or any(not isinstance(x, str) or not x for x in self._map_identities):
            raise ValueError("Explicit sensor map identity required")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested sensor CUDA device unavailable")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Unsupported sensor device")
        p = spec.perception
        x = np.linspace(-p.reverse, p.forward, p.frame_shape[0])
        y = np.linspace(-p.lateral, p.lateral, p.frame_shape[1])
        xx, yy = np.meshgrid(x, y, indexing="ij")
        self._body_points = torch.as_tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float64, device=self.device)
        ix = np.clip(np.arange(len(x)) - 1, 0, len(x) - 3)
        iy = np.clip(np.arange(len(y)) - 1, 0, len(y) - 3)
        dx, dy = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        indices = (ix[:, None, None] + dx.ravel()) * len(y) + iy[None, :, None] + dy.ravel()
        self._patch_indices = torch.as_tensor(indices.reshape(-1, 9), device=self.device)
        design = np.c_[dx.ravel() * p.spacing, dy.ravel() * p.spacing, np.ones(9)]
        self._detrend = torch.as_tensor(np.eye(9) - design @ np.linalg.pinv(design), dtype=torch.float64, device=self.device)

    def sample(self, poses_world, odometry_frames, times):
        poses = torch.as_tensor(poses_world, dtype=torch.float64, device=self.device)
        frames = torch.as_tensor(odometry_frames, dtype=torch.float64, device=self.device)
        now = torch.as_tensor(times, dtype=torch.float64, device=self.device)
        if (poses.ndim != 3 or poses.shape[1:] != (4, 3) or frames.shape != poses.shape
                or now.shape != (poses.shape[0],) or not torch.isfinite(poses).all()
                or not torch.isfinite(frames).all() or not torch.isfinite(now).all() or (now < 0).any()):
            raise ValueError("Dense sensor requires finite [E,4,3] poses/frames and [E] times")
        if len(self._map_identities) not in (1, len(poses)):
            raise ValueError("Sensor map identities do not match environments")
        c, s = poses[..., 2].cos(), poses[..., 2].sin()
        bx, by = self._body_points[:, 0], self._body_points[:, 1]
        world = poses[..., None, :2] + torch.stack((c[..., None] * bx - s[..., None] * by,
                                                   s[..., None] * bx + c[..., None] * by), -1)
        cfg = self._terrain_cfg
        if cfg is None or cfg.type == "flat_proxy":
            features = torch.zeros((*world.shape[:-1], 5), dtype=torch.float64, device=self.device)
            features[..., 4] = 1.
            supported = torch.ones(world.shape[:-1], dtype=torch.bool, device=self.device)
        elif cfg.type == "lunar_scene_pack":
            if self._runtime is None or self._runtime.scene_bank is None or self._runtime.scene_indices is None:
                raise ValueError("Raster sensing requires a versioned scene runtime")
            query = self._runtime.scene_bank.query(world, self._runtime.scene_indices)
            native = query.features.to(device=self.device, dtype=torch.float64)
            supported = query.metrics_valid.to(device=self.device, dtype=torch.bool)
            slope = native[..., 1:3].norm(dim=-1)
            score = torch.exp(-slope / cfg.raster_risk_slope_scale - native[..., 3] / cfg.raster_risk_rms_scale_m)
            features = torch.cat((native, score[..., None]), -1)
        else:
            runtime = self._runtime.to(self.device, dtype=torch.float64) if self._runtime is not None else None
            features = query_terrain_features(world, cfg, runtime)
            supported = torch.isfinite(features).all(-1)
        if features.shape != (*world.shape[:-1], 5) or supported.shape != world.shape[:-1]:
            raise ValueError("Terrain backend sensor shape mismatch")
        supported &= torch.isfinite(features).all(-1)
        supported &= ((world[..., 0].abs() <= self.spec.world_width / 2)
                      & (world[..., 1].abs() <= self.spec.world_height / 2))
        height = torch.where(supported, features[..., 0], 0.)
        roughness_known = supported[..., self._patch_indices].all(-1)
        if cfg is None or cfg.type == "flat_proxy":
            roughness = torch.zeros_like(height)
        else:
            patches = height[..., self._patch_indices]
            residual = patches @ self._detrend.T
            roughness = residual.square().mean(-1).sqrt()
        # A finite backend placeholder does not certify its slope/RMS support.
        valid = supported & roughness_known
        features = features.clone(); features[..., 3] = roughness
        fc, fs = frames[..., 2].cos(), frames[..., 2].sin()
        delta = world - frames[..., None, :2]
        odom = torch.stack((fc[..., None] * delta[..., 0] + fs[..., None] * delta[..., 1],
                            -fs[..., None] * delta[..., 0] + fc[..., None] * delta[..., 1]), -1)
        gradient = features[..., 1:3].clone()
        features[..., 1] = fc[..., None] * gradient[..., 0] + fs[..., None] * gradient[..., 1]
        features[..., 2] = -fs[..., None] * gradient[..., 0] + fc[..., None] * gradient[..., 1]
        features = torch.where(valid[..., None], features, 0.)
        result = []
        for env in range(len(poses)):
            row = []
            for agent in range(4):
                frame = frames[env, agent].cpu().numpy()
                pose = poses[env, agent].cpu().numpy()
                local_pose = np.r_[world_to_odometry(pose[:2][None], frame)[0], pose[2] - frame[2]]
                row.append(SensorFrame(env, agent, float(now[env]), odom[env, agent].cpu().numpy(),
                    features[env, agent].cpu().numpy(), valid[env, agent].cpu().numpy(), valid[env, agent].cpu().numpy(),
                    local_pose, self.spec, self._map_identities[env if len(self._map_identities) > 1 else 0]))
            result.append(row)
        return result


class DenseTerrainMemory:
    """Bounded direct observations with profile-specific hazard preservation."""
    version = MEMORY_VERSION

    def __init__(self, spec):
        if type(spec) is not UpstreamSpec:
            raise ValueError("Dense memory requires an immutable profile")
        self.spec = spec
        self.reset()

    def reset(self):
        self._identity = None
        self._map_identity = None
        self._last_time = None
        self._observations = 0
        self._view = None
        self._last_frame = None

    def _key(self, point):
        cell = tuple(np.rint(point / self.spec.perception.memory_resolution).astype(np.int64))
        return (*cell, float(point[0]), float(point[1]))

    def _hazard(self, feature):
        terrain = self.spec.terrain
        return (feature[4] < terrain.minimum_traversability or np.linalg.norm(feature[1:3]) > terrain.driving_slope
                or feature[3] > terrain.roughness_rms)

    def _merge(self, a, b):
        merged = b.copy()
        merged[0] = max(a[0], b[0]); merged[3] = max(a[3], b[3]); merged[4] = min(a[4], b[4])
        if np.linalg.norm(a[1:3]) > np.linalg.norm(b[1:3]):merged[1:3] = a[1:3]
        # Conflicting heights at exactly the same direct position cannot be
        # certified as a traversable surface by choosing the higher value.
        if abs(a[0] - b[0]) > self.spec.terrain.step_height + 1e-8:merged[4] = 0.
        return merged

    def observe(self, frame):
        if type(frame) is not SensorFrame or frame.spec.fingerprint != self.spec.fingerprint or frame.version != SENSOR_VERSION:
            raise ValueError("Dense memory accepts only matching direct sensor frames")
        identity = frame.env_id, frame.agent_id
        if self._identity is not None and self._identity != identity:
            raise ValueError("Cannot mix another rover's private memory")
        if self._map_identity is not None and self._map_identity != frame.map_identity:
            raise ValueError("Cannot mix another map episode without reset")
        if self._last_time is not None and frame.timestamp < self._last_time:
            raise ValueError("Dense sensor time moved backwards")
        xy=frame.xy[frame.valid]; features=frame.features[frame.valid]
        if (self._last_frame is not None and self._view is not None
                and np.array_equal(self._last_frame[0],xy)
                and np.array_equal(self._last_frame[1],features)):
            # Re-observing the identical direct frame leaves the conservative
            # merge and insertion order unchanged. Fresh keys occupy the tail;
            # retained historical hazards keep their original timestamps.
            start=len(self._view._xy)-self._last_frame[2]
            times=self._view._timestamps.copy(); times[start:]=frame.timestamp
            view=self._view.with_observation_times(times)
            self._view=view
            self._identity,self._last_time=identity,frame.timestamp
            self._map_identity=frame.map_identity;self._observations+=1
            return
        if self._view is not None:
            previous=self._view.readonly_snapshot();xy=frame.xy[frame.valid];features=frame.features[frame.valid]
            if np.array_equal(previous.xy,xy) and np.array_equal(previous.features,features):
                # Identical complete direct measurements need only advance
                # observation times. Hazard values and exact direct positions
                # are unchanged; changing geometry/attributes uses full merge.
                view=self._view.with_observation_times(np.full(len(self._view._xy),frame.timestamp))
                self._view=view
                self._identity,self._last_time=identity,frame.timestamp
                self._map_identity=frame.map_identity;self._observations+=1
                self._last_frame=(xy.copy(),features.copy(),len(self._view._xy))
                return
        from .upstream_native import memory_merge_cpu
        merged = memory_merge_cpu(self.view(), xy, features, frame.timestamp, self.spec)
        if merged is None:
            merged = self._merge_reference(xy, features, frame.timestamp)
        points, attributes, times, fresh_count = merged
        view = self._make_view(points, attributes, times)
        self._view = view
        self._identity, self._last_time = identity, frame.timestamp
        self._map_identity = frame.map_identity
        self._observations += 1
        self._last_frame = (xy.copy(), features.copy(), fresh_count)

    def _merge_reference(self, xy, features, timestamp):
        """Authoritative scalar decisions at floating-point threshold ties."""
        old = self.view()
        previous_rows = OrderedDict((self._key(p), (p, f, float(t)))
            for p, f, t in zip(old._xy, old._features, old._timestamps, strict=True))
        fresh = OrderedDict(); cells = set(); by_cell = {}
        indices = np.rint(xy / self.spec.perception.memory_resolution).astype(np.int64)
        for cell, point, feature in zip(indices, xy, features, strict=True):
            key = (cell[0], cell[1], float(point[0]), float(point[1])); cells.add(key[:2])
            if key in fresh:feature = self._merge(fresh[key][1], feature)
            fresh[key] = (point.copy(), feature.copy(), timestamp)
            by_cell.setdefault(key[:2], []).append((point, feature))
        if len(fresh) > self.spec.perception.capacity:
            raise ValueError("Capacity cannot retain the complete direct frame")
        rows = OrderedDict()
        for key, value in previous_rows.items():
            keep = key[:2] not in cells or self._hazard(value[1])
            if not keep:
                old_point, old_feature, _ = value
                # A fresh point in an indexing cell is a distinct measurement.
                # Remove the observed local plane before detecting a step;
                # the new point must not erase the old step/hole evidence.
                keep = any(abs(old_feature[0] - feature[0] - feature[1:3] @ (old_point - point))
                           > self.spec.terrain.step_height + 1e-8 for point, feature in by_cell[key[:2]])
            if keep:rows[key] = value
        for key, value in fresh.items():
            old = rows.pop(key, None)
            if old is not None:value = value[0], self._merge(old[1], value[1]), value[2]
            rows[key] = value
        while len(rows) > self.spec.perception.capacity:rows.popitem(last=False)
        values = list(rows.values())
        return (np.asarray([x[0] for x in values], dtype=float).reshape(-1, 2),
                np.asarray([x[1] for x in values], dtype=float).reshape(-1, 5),
                np.asarray([x[2] for x in values], dtype=float), len(fresh))

    def _make_view(self, xy, features, observed):
        return FootprintTerrain._from_owned_arrays(xy, features, observed, self.spec)

    def view(self):
        if self._view is None:
            self._view = self._make_view(np.empty((0, 2)), np.empty((0, 5)), np.empty(0))
        return self._view

    def lookup(self, xy):
        points = np.atleast_2d(np.asarray(xy, dtype=float))
        features, known = self.view().lookup(points)
        snapshot = self.view().readonly_snapshot()
        direct = (self.view()._tree.query(points)[0] < 1e-5) if len(snapshot.xy) else np.zeros(len(points), dtype=bool)
        return features, known, direct

    def state_dict(self):
        snapshot = self.view().snapshot()
        return deepcopy({"version": self.version, "profile_fingerprint": self.spec.fingerprint,
                         "identity": self._identity, "last_time": self._last_time, "observations": self._observations,
                         "map_identity": self._map_identity,
                         "xy": snapshot.xy, "features": snapshot.features, "observed_at": snapshot.observed_at})

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("version") != self.version or state.get("profile_fingerprint") != self.spec.fingerprint:
            raise ValueError("Dense memory version/profile mismatch")
        identity, last, count = state.get("identity"), state.get("last_time"), state.get("observations")
        if type(count) is not int or count < 0:
            raise ValueError("Invalid dense memory observation count")
        if identity is not None and (type(identity) is not tuple or len(identity) != 2
                or any(type(x) is not int for x in identity) or identity[0] < 0 or not 0 <= identity[1] < 4):
            raise ValueError("Invalid dense memory identity")
        if self._identity is not None and self._identity != identity:
            raise ValueError("Cannot restore another rover's memory")
        map_identity = state.get("map_identity")
        if (identity is None and map_identity is not None) or (identity is not None and (not isinstance(map_identity, str) or not map_identity)):
            raise ValueError("Invalid restored sensor map identity")
        if self._map_identity is not None and self._map_identity != map_identity:
            raise ValueError("Cannot restore another map episode")
        if (identity is None) != (last is None) or (last is not None and (not math.isfinite(last) or last < 0 or count < 1)):
            raise ValueError("Invalid dense memory time/binding")
        xy = np.asarray(state.get("xy"), dtype=float)
        if xy.ndim != 2 or xy.shape[1:] != (2,) or len(xy) > self.spec.perception.capacity:
            raise ValueError("Invalid dense memory capacity/shape")
        xy = _owned(xy, (len(xy), 2), "Restored memory positions")
        features = _owned(state.get("features"), (len(xy), 5), "Restored memory features")
        times = _owned(state.get("observed_at"), (len(xy),), "Restored memory timestamps")
        if (len(xy) and identity is None) or (identity is None and count != 0) or (times < 0).any() or (len(xy) and (times > last).any()):
            raise ValueError("Invalid restored observation times")
        if ((features[:, 4] < 0) | (features[:, 4] > 1) | (features[:, 3] < 0)).any() or (abs(xy / self.spec.perception.memory_resolution) > 2**40).any():
            raise ValueError("Invalid restored terrain attributes/coordinates")
        if len(np.unique(xy, axis=0)) != len(xy):
            raise ValueError("Duplicate restored direct point")
        view = self._make_view(xy, features, times)
        self._view = view
        self._identity, self._last_time, self._observations = identity, last, count
        self._map_identity = map_identity
        self._last_frame = None
