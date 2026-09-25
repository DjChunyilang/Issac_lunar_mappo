"""Dataset-backed diagnostic runtime, with no legacy safety or learning semantics."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from . import FEATURE_SCHEMA
from .products import checksum
from .query import RasterTerrain


def require_feature_schema(checkpoint_metadata: dict):
    if checkpoint_metadata.get("terrain_feature_schema") != FEATURE_SCHEMA:
        raise ValueError("Missing/incompatible terrain semantics; legacy checkpoints require "
                         "an explicit task/observation migration and cannot be silently reused")


def load_scene(path: Path, *, verify_hashes: bool = True) -> RasterTerrain:
    scene = json.loads(path.read_text())
    if scene["schema"] != "lunar_scene_v1" or scene["terrain_feature_schema"] != FEATURE_SCHEMA:
        raise ValueError("Unsupported scene/feature schema")
    if scene["layer"] != "task_representation" or scene.get("synthetic_detail") is not False:
        raise ValueError("This loader accepts real-product views only")
    paths = {k: (path.parent / scene[k]["path"]).resolve() for k in ("dtm", "error")}
    if verify_hashes:
        for key, product in paths.items():
            if checksum(product) != scene[key]["sha256"]:
                raise ValueError(f"Changed scene product: {key}")
    return RasterTerrain(paths["dtm"], paths["error"], source_id=scene["source_id"],
                         origin_xy=scene["origin_xy_m"],
                         max_cache_bytes=scene.get("max_cache_bytes", 32*1024**2))


class TerrainProbeRuntime:
    """Kinematic point probe for integration checks, NOT a rover/task simulator.

    Truth is private to the runtime. Local samples emulate access to the orbital
    reconstruction, not a validated ground sensor. A separate method grants an
    orbital prior only if explicitly enabled. No safety/success label is emitted.
    """

    def __init__(self, terrain: RasterTerrain, *, half_width: int,
                 sensor_radius_m: float, orbital_prior: bool = False):
        if sensor_radius_m <= 0 or not np.isfinite(sensor_radius_m):
            raise ValueError("Explicit positive physical sensing radius required")
        if half_width < 1 or int(half_width) != half_width:
            raise ValueError("Explicit native-grid metric scale required")
        self._terrain = terrain
        self.half_width = half_width
        self.sensor_radius_m = sensor_radius_m
        self.orbital_prior_enabled = orbital_prior
        self.position = None

    def reset(self, local_xy_m):
        xy = np.asarray(local_xy_m, float)
        sample = self._terrain.query(xy, half_width=self.half_width)
        if not sample.height_valid:
            raise ValueError("Initial position has no terrain support")
        self.position = xy.copy()
        return dict(position_xy_m=self.position.copy(), terrain=asdict(sample),
                    role="simulator_diagnostic_truth_not_actor_observation")

    def observe_local(self, offsets_xy_m):
        if self.position is None:
            raise RuntimeError("Call reset first")
        offsets = np.asarray(offsets_xy_m, float)
        if offsets.ndim != 2 or offsets.shape[1] != 2 or not np.all(np.isfinite(offsets)):
            raise ValueError("Expected finite Nx2 local observation offsets")
        observations = []
        # Entire metric footprint must lie inside the explicitly allowed sensing disk.
        dx, dy = self._terrain.spacing
        for offset in offsets:
            sample = self._terrain.query(self.position + offset, half_width=self.half_width)
            center = np.asarray(sample.metric_center_local_xy_m) - self.position
            corners = center + np.array([[sx*self.half_width*dx, sy*self.half_width*dy]
                                         for sx in (-1, 1) for sy in (-1, 1)])
            if (np.linalg.norm(offset) > self.sensor_radius_m
                    or np.any(np.linalg.norm(corners, axis=1) > self.sensor_radius_m)):
                raise PermissionError("Requested sample/analysis support exceeds local access")
            observations.append(asdict(sample))
        return observations

    def query_orbital_prior(self, local_xy_m):
        if not self.orbital_prior_enabled:
            raise PermissionError("Orbital prior is not enabled for this runtime")
        return asdict(self._terrain.query(local_xy_m, half_width=self.half_width))

    def step(self, velocity_xy_m_s, dt_s: float):
        if self.position is None:
            raise RuntimeError("Call reset first")
        velocity = np.asarray(velocity_xy_m_s, float)
        if velocity.shape != (2,) or not np.all(np.isfinite(velocity)) or not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("Finite velocity and positive finite timestep required")
        target = self.position + velocity * dt_s
        # Sweep at <= half a native cell; no teleporting over a missing-cell strip.
        distance = np.linalg.norm(target - self.position)
        steps = max(1, int(np.ceil(distance / (min(self._terrain.spacing)/2))))
        supported = all(self._terrain.query(p, half_width=self.half_width).height_valid
                        for p in np.linspace(self.position, target, steps+1))
        if supported:
            self.position = target
        return dict(position_xy_m=self.position.copy(), moved=supported,
                    terrain=asdict(self._terrain.query(self.position, half_width=self.half_width)),
                    vehicle_safety="unknown", task_success=None,
                    scope="kinematic point-probe interface check only")
