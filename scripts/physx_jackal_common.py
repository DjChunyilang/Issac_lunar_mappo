"""Shared helpers for Clearpath Jackal PhysX tracking validation."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

JACKAL_PRIM_PATH = "/World/Jackal"
JACKAL_USD_RELATIVE_PATH = "Isaac/Robots/Clearpath/Jackal/jackal.usd"
JACKAL_WHEEL_DOF_NAMES = [
    "front_left_wheel_joint",
    "front_right_wheel_joint",
    "rear_left_wheel_joint",
    "rear_right_wheel_joint",
]
JACKAL_WHEEL_RADIUS = 0.098
JACKAL_TRACK_WIDTH = 0.376
JACKAL_MAX_LINEAR_SPEED = 1.2
JACKAL_MAX_ANGULAR_SPEED = 2.4
JACKAL_MAX_WHEEL_SPEED = 18.0

STRONG_LUNAR_CRATER_PROFILE = {
    "terrain": "strong_lunar_crater",
    "size": 9.0,
    "resolution": 72,
    "amplitude": 0.16,
    "wavelength": 2.8,
    "crater_count": 7,
    "crater_min_radius": 0.45,
    "crater_max_radius": 1.25,
    "crater_depth_to_diameter": 0.18,
    "crater_rim_height_to_diameter": 0.025,
    "crater_seed": 11,
}

STRONG_LUNAR_CRATER_PATH_OFFSETS = {
    "straight": (-0.5, -0.5),
    "circle": (1.0, -2.0),
    "sine": (0.5, 2.0),
}

TRACKING_THRESHOLDS = {
    "flat": {
        "rmse_cross_track_m": 0.20,
        "max_cross_track_m": 0.55,
        "path_completion_ratio": 0.90,
        "max_tilt_deg": 180.0,
    },
    "strong_lunar_crater": {
        "rmse_cross_track_m": 0.50,
        "max_cross_track_m": 1.10,
        "path_completion_ratio": 0.75,
        "max_tilt_deg": 35.0,
    },
}


@dataclass(frozen=True, slots=True)
class ReferencePath:
    profile: str
    points_xy: np.ndarray
    yaws: np.ndarray
    cumulative_s: np.ndarray

    @property
    def length_m(self) -> float:
        return float(self.cumulative_s[-1]) if len(self.cumulative_s) else 0.0


@dataclass(frozen=True, slots=True)
class TrackingControllerCfg:
    lookahead_m: float = 0.45
    k_heading: float = 1.6
    k_cross_track: float = 0.75
    angular_scale: float = 1.05
    target_speed_mps: float = 0.35
    max_linear_mps: float = JACKAL_MAX_LINEAR_SPEED
    max_angular_radps: float = JACKAL_MAX_ANGULAR_SPEED


class JackalSkidSteerController:
    """Convert unicycle commands to four Jackal wheel velocity targets."""

    def __init__(
        self,
        *,
        wheel_radius: float = JACKAL_WHEEL_RADIUS,
        track_width: float = JACKAL_TRACK_WIDTH,
        max_linear_speed: float = JACKAL_MAX_LINEAR_SPEED,
        max_angular_speed: float = JACKAL_MAX_ANGULAR_SPEED,
        max_wheel_speed: float = JACKAL_MAX_WHEEL_SPEED,
    ) -> None:
        if wheel_radius <= 0.0:
            raise ValueError("wheel_radius must be positive")
        if track_width <= 0.0:
            raise ValueError("track_width must be positive")
        if max_linear_speed < 0.0 or max_angular_speed < 0.0 or max_wheel_speed < 0.0:
            raise ValueError("speed limits must be non-negative")
        self.wheel_radius = float(wheel_radius)
        self.track_width = float(track_width)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.max_wheel_speed = float(max_wheel_speed)

    def forward(self, command: Iterable[float]) -> np.ndarray:
        command_arr = np.asarray(list(command), dtype=np.float64)
        if command_arr.shape != (2,):
            raise ValueError("command must contain [linear_mps, angular_radps]")
        linear = float(np.clip(command_arr[0], -self.max_linear_speed, self.max_linear_speed))
        angular = float(np.clip(command_arr[1], -self.max_angular_speed, self.max_angular_speed))
        left = (linear - 0.5 * angular * self.track_width) / self.wheel_radius
        right = (linear + 0.5 * angular * self.track_width) / self.wheel_radius
        wheels = np.array([left, right, left, right], dtype=np.float64)
        return np.clip(wheels, -self.max_wheel_speed, self.max_wheel_speed)


def resolve_path(path: str | Path) -> Path:
    from _common import ROOT

    output = Path(path)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def normalize_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_wxyz_to_yaw(quat: Iterable[float]) -> float:
    w, x, y, z = [float(v) for v in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_wxyz_to_tilt_deg(quat: Iterable[float]) -> float:
    w, x, y, z = [float(v) for v in quat]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_arg)
    return math.degrees(max(abs(roll), abs(pitch)))


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=np.float32)


def build_physics_scene(stage) -> None:
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(3500.0)
    sun.CreateAngleAttr(0.45)
    sun_xform = UsdGeom.Xformable(sun.GetPrim())
    sun_xform.ClearXformOpOrder()
    sun_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))

    sky = UsdLux.DomeLight.Define(stage, "/World/Sky")
    sky.CreateIntensityAttr(650.0)


def _bind_preview_material(stage, prim, path: str, color: tuple[float, float, float]) -> None:
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def add_flat_terrain(stage, size: float = 10.0) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, "/World/Terrain")
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.03))
    xform.AddScaleOp().Set(Gf.Vec3f(size, size, 0.06))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _bind_preview_material(stage, cube.GetPrim(), "/World/Materials/FlatTerrain", (0.42, 0.42, 0.38))


def _crater_layout_np(
    count: int,
    min_radius: float,
    max_radius: float,
    field_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = max(0, int(count))
    if count <= 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)
    if count == 1:
        return (
            np.zeros((1, 2), dtype=np.float32),
            np.full((1,), float(max_radius), dtype=np.float32),
        )
    index = np.arange(count, dtype=np.float32)
    seed_phase = float(seed) * 0.61803398875
    field_radius = 0.45 * float(field_size)
    radial = field_radius * np.sqrt((index + 0.5) / float(count))
    theta = index * 2.39996322973 + seed_phase
    centers = np.stack((radial * np.cos(theta), radial * np.sin(theta)), axis=-1)
    radius_mix = 0.5 + 0.5 * np.sin(index * 12.9898 + seed_phase)
    radii = float(min_radius) + (float(max_radius) - float(min_radius)) * radius_mix
    return centers.astype(np.float32), np.maximum(radii.astype(np.float32), 1.0e-3)


def lunar_crater_height_np(
    x: float | np.ndarray,
    y: float | np.ndarray,
    *,
    amplitude: float,
    wavelength: float,
    crater_count: int,
    crater_min_radius: float,
    crater_max_radius: float,
    crater_depth_to_diameter: float,
    crater_rim_height_to_diameter: float,
    crater_field_size: float,
    crater_seed: int,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    height = np.zeros(np.broadcast_shapes(x_arr.shape, y_arr.shape), dtype=np.float32)
    if amplitude != 0.0:
        k = 2.0 * math.pi / max(float(wavelength), 1.0e-6)
        height = height + float(amplitude) * (
            np.sin(k * x_arr) * np.cos(k * y_arr)
            + 0.45 * np.sin(1.7 * k * x_arr + 0.35) * np.sin(1.3 * k * y_arr)
        )
    centers, radii = _crater_layout_np(
        crater_count,
        crater_min_radius,
        crater_max_radius,
        crater_field_size,
        crater_seed,
    )
    for center, radius in zip(centers, radii, strict=True):
        distance = np.sqrt((x_arr - center[0]) ** 2 + (y_arr - center[1]) ** 2)
        diameter = 2.0 * float(radius)
        depth = float(crater_depth_to_diameter) * diameter
        rim_height = float(crater_rim_height_to_diameter) * diameter
        normalized = distance / max(float(radius), 1.0e-6)
        bowl = np.maximum(1.0 - normalized**2, 0.0) ** 2
        rim = np.exp(-((normalized - 1.0) / 0.22) ** 2)
        height = height - depth * bowl + rim_height * rim
    return height.astype(np.float32)


def strong_lunar_crater_height_np(x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
    profile = STRONG_LUNAR_CRATER_PROFILE
    return lunar_crater_height_np(
        x,
        y,
        amplitude=float(profile["amplitude"]),
        wavelength=float(profile["wavelength"]),
        crater_count=int(profile["crater_count"]),
        crater_min_radius=float(profile["crater_min_radius"]),
        crater_max_radius=float(profile["crater_max_radius"]),
        crater_depth_to_diameter=float(profile["crater_depth_to_diameter"]),
        crater_rim_height_to_diameter=float(profile["crater_rim_height_to_diameter"]),
        crater_field_size=float(profile["size"]),
        crater_seed=int(profile["crater_seed"]),
    )


def add_strong_lunar_crater_terrain(stage) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    profile = STRONG_LUNAR_CRATER_PROFILE
    size = float(profile["size"])
    resolution = int(profile["resolution"])
    xs = np.linspace(-size / 2.0, size / 2.0, resolution + 1)
    ys = np.linspace(-size / 2.0, size / 2.0, resolution + 1)
    points = []
    for y in ys:
        for x in xs:
            z = strong_lunar_crater_height_np(x, y)
            points.append(Gf.Vec3f(float(x), float(y), float(z)))

    row = resolution + 1
    counts = []
    indices = []
    for iy in range(resolution):
        for ix in range(resolution):
            i0 = iy * row + ix
            counts.append(4)
            indices.extend((i0, i0 + 1, i0 + row + 1, i0 + row))

    mesh = UsdGeom.Mesh.Define(stage, "/World/Terrain")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    try:
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        mesh_collision.CreateApproximationAttr("meshSimplification")
    except Exception:
        pass
    _bind_preview_material(stage, mesh.GetPrim(), "/World/Materials/StrongLunarCrater", (0.36, 0.35, 0.32))


def add_tracking_terrain(stage, terrain: str) -> None:
    if terrain == "flat":
        add_flat_terrain(stage, size=10.0)
    elif terrain == "strong_lunar_crater":
        add_strong_lunar_crater_terrain(stage)
    else:
        raise ValueError(f"Unsupported tracking terrain: {terrain}")


def terrain_height(terrain: str, x: float, y: float) -> float:
    if terrain == "flat":
        return 0.0
    if terrain == "strong_lunar_crater":
        return float(strong_lunar_crater_height_np(x, y))
    raise ValueError(f"Unsupported tracking terrain: {terrain}")


def generate_reference_path(profile: str, samples: int = 420) -> ReferencePath:
    samples = max(8, int(samples))
    if profile == "straight":
        x = np.linspace(-2.7, 2.7, samples, dtype=np.float64)
        y = np.zeros_like(x)
    elif profile == "circle":
        theta = np.linspace(-math.pi, math.pi, samples, endpoint=False, dtype=np.float64)
        radius = 1.55
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    elif profile == "sine":
        x = np.linspace(-2.8, 2.8, samples, dtype=np.float64)
        y = 0.65 * np.sin(2.0 * math.pi * (x + 2.8) / 3.0)
    else:
        raise ValueError(f"Unsupported tracking profile: {profile}")

    points = np.stack((x, y), axis=-1)
    deltas = np.diff(points, axis=0, prepend=points[:1])
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_s = np.concatenate(([0.0], np.cumsum(segment)))
    yaws = np.arctan2(deltas[:, 1], deltas[:, 0])
    if len(yaws) > 1:
        yaws[0] = yaws[1]
    return ReferencePath(profile=profile, points_xy=points, yaws=yaws, cumulative_s=cumulative_s)


def offset_reference_path(path: ReferencePath, offset_xy: Iterable[float]) -> ReferencePath:
    offset = np.asarray(list(offset_xy), dtype=np.float64)
    if offset.shape != (2,):
        raise ValueError("offset_xy must contain [x, y]")
    return ReferencePath(
        profile=path.profile,
        points_xy=path.points_xy + offset[None, :],
        yaws=path.yaws.copy(),
        cumulative_s=path.cumulative_s.copy(),
    )


def nearest_path_index(
    path: ReferencePath,
    xy: Iterable[float],
    *,
    start_index: int = 0,
    end_index: int | None = None,
) -> int:
    xy_arr = np.asarray(list(xy), dtype=np.float64)
    start = max(0, min(int(start_index), len(path.points_xy) - 1))
    end = len(path.points_xy) if end_index is None else max(start + 1, min(int(end_index), len(path.points_xy)))
    distances = np.linalg.norm(path.points_xy[start:end] - xy_arr[None, :2], axis=1)
    return int(start + np.argmin(distances))


def target_index_from_lookahead(path: ReferencePath, nearest_idx: int, lookahead_m: float) -> int:
    target_s = float(path.cumulative_s[nearest_idx]) + max(0.0, float(lookahead_m))
    return int(min(np.searchsorted(path.cumulative_s, target_s), len(path.cumulative_s) - 1))


def tracking_error(
    path: ReferencePath,
    xy: Iterable[float],
    yaw: float,
    *,
    nearest_idx: int | None = None,
    progress_index: int = 0,
) -> dict:
    xy_arr = np.asarray(list(xy), dtype=np.float64)
    idx = nearest_path_index(path, xy_arr, start_index=progress_index) if nearest_idx is None else int(nearest_idx)
    ref_xy = path.points_xy[idx]
    ref_yaw = float(path.yaws[idx])
    delta = xy_arr[:2] - ref_xy
    left_normal = np.array([-math.sin(ref_yaw), math.cos(ref_yaw)], dtype=np.float64)
    signed_cross_track = float(np.dot(delta, left_normal))
    cross_track = float(np.linalg.norm(delta))
    heading_error = float(normalize_angle(ref_yaw - yaw))
    completion = float(path.cumulative_s[idx] / max(path.length_m, 1.0e-6))
    return {
        "nearest_index": idx,
        "cross_track_m": cross_track,
        "signed_cross_track_m": signed_cross_track,
        "heading_error_rad": heading_error,
        "path_completion_ratio": completion,
    }


def compute_tracking_command(
    path: ReferencePath,
    xy: Iterable[float],
    yaw: float,
    cfg: TrackingControllerCfg,
    *,
    progress_index: int = 0,
    search_window: int = 80,
) -> tuple[np.ndarray, dict]:
    xy_arr = np.asarray(list(xy), dtype=np.float64)
    nearest_idx = nearest_path_index(
        path,
        xy_arr,
        start_index=progress_index,
        end_index=progress_index + max(8, int(search_window)),
    )
    target_idx = target_index_from_lookahead(path, nearest_idx, cfg.lookahead_m)
    target_xy = path.points_xy[target_idx]
    dx, dy = target_xy - xy_arr[:2]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    target_heading_error = math.atan2(local_y, max(local_x, 1.0e-6))
    error = tracking_error(path, xy_arr, yaw, nearest_idx=nearest_idx)
    remaining = max(path.length_m - float(path.cumulative_s[nearest_idx]), 0.0)
    speed_scale = min(1.0, max(0.25, remaining / max(cfg.lookahead_m, 1.0e-6)))
    if remaining < 0.25:
        speed_scale = 0.0
    linear = np.clip(cfg.target_speed_mps * speed_scale, -cfg.max_linear_mps, cfg.max_linear_mps)
    omega = cfg.angular_scale * (
        cfg.k_heading * target_heading_error
        - cfg.k_cross_track * error["signed_cross_track_m"] / max(cfg.lookahead_m, 1.0e-6)
    )
    angular = np.clip(omega, -cfg.max_angular_radps, cfg.max_angular_radps)
    details = {
        **error,
        "target_index": target_idx,
        "target_heading_error_rad": float(target_heading_error),
        "remaining_path_m": float(remaining),
        "command_linear_mps": float(linear),
        "command_angular_radps": float(angular),
    }
    return np.array([linear, angular], dtype=np.float64), details


def tracking_acceptance(metrics: dict, terrain: str) -> dict:
    thresholds = TRACKING_THRESHOLDS[terrain]
    checks = {
        "rmse_cross_track_m": float(metrics.get("rmse_cross_track_m", math.inf))
        <= thresholds["rmse_cross_track_m"],
        "max_cross_track_m": float(metrics.get("max_cross_track_m", math.inf))
        <= thresholds["max_cross_track_m"],
        "path_completion_ratio": float(metrics.get("path_completion_ratio", 0.0))
        >= thresholds["path_completion_ratio"],
        "max_tilt_deg": float(metrics.get("max_tilt_deg", math.inf)) <= thresholds["max_tilt_deg"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
    }


def set_camera() -> None:
    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(
        eye=[4.8, -5.4, 3.2],
        target=[0.0, 0.0, 0.25],
        camera_prim_path="/OmniverseKit_Persp",
    )


def get_assets_root(app) -> str:
    from isaacsim.storage.native import get_assets_root_path, get_assets_root_path_async

    assets_root = get_assets_root_path()
    if assets_root is None:
        assets_root = app.run_coroutine(get_assets_root_path_async())
    if assets_root is None:
        raise RuntimeError("Could not resolve Isaac Sim assets root path")
    return assets_root.rstrip("/")


def jackal_usd_path(assets_root: str) -> str:
    return f"{assets_root.rstrip('/')}/{JACKAL_USD_RELATIVE_PATH}"


def capture_viewport(app, output: str | Path) -> bool:
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport, next_viewport_frame_async

    output_path = resolve_path(output)
    viewport = get_active_viewport()
    if viewport is None:
        return False
    app.run_coroutine(next_viewport_frame_async(viewport, n_frames=5))
    capture = capture_viewport_to_file(viewport, file_path=str(output_path))
    ok = bool(app.run_coroutine(capture.wait_for_result(completion_frames=30)))
    for _ in range(10):
        app.update()
    return ok and output_path.exists()


def temporary_capture_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="physx_jackal_frames_"))
