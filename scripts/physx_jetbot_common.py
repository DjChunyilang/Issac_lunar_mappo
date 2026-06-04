"""Shared helpers for Isaac Sim Jetbot PhysX validation scripts."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

JETBOT_PRIM_PATH = "/World/Jetbot"
JETBOT_WHEEL_DOF_NAMES = ["left_wheel_joint", "right_wheel_joint"]
JETBOT_WHEEL_RADIUS = 0.0335
JETBOT_WHEEL_BASE = 0.118


def resolve_path(path: str | Path) -> Path:
    from _common import ROOT

    output = Path(path)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def quat_wxyz_to_yaw(quat: Iterable[float]) -> float:
    w, x, y, z = [float(v) for v in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_wxyz_to_tilt_deg(quat: Iterable[float]) -> float:
    w, x, y, z = [float(v) for v in quat]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_arg)
    return math.degrees(max(abs(roll), abs(pitch)))


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


def add_rough_terrain(
    stage,
    size: float = 7.0,
    resolution: int = 36,
    amplitude: float = 0.055,
    wavelength: float = 2.6,
) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    xs = np.linspace(-size / 2.0, size / 2.0, resolution + 1)
    ys = np.linspace(-size / 2.0, size / 2.0, resolution + 1)
    points = []
    for y in ys:
        for x in xs:
            z = amplitude * (
                math.sin(2.0 * math.pi * x / wavelength) * math.cos(2.0 * math.pi * y / wavelength)
                + 0.35 * math.sin(3.1 * x / wavelength + 0.4) * math.sin(2.3 * y / wavelength)
            )
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
    _bind_preview_material(stage, mesh.GetPrim(), "/World/Materials/RoughTerrain", (0.38, 0.36, 0.32))


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
    amplitude: float = 0.025,
    wavelength: float = 2.8,
    crater_count: int = 7,
    crater_min_radius: float = 0.35,
    crater_max_radius: float = 1.15,
    crater_depth_to_diameter: float = 0.06,
    crater_rim_height_to_diameter: float = 0.015,
    crater_field_size: float = 9.0,
    crater_seed: int = 11,
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


def add_lunar_crater_terrain(
    stage,
    size: float = 9.0,
    resolution: int = 64,
    amplitude: float = 0.025,
    wavelength: float = 2.8,
    crater_count: int = 7,
    crater_min_radius: float = 0.35,
    crater_max_radius: float = 1.15,
    crater_depth_to_diameter: float = 0.06,
    crater_rim_height_to_diameter: float = 0.015,
    crater_seed: int = 11,
) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    xs = np.linspace(-size / 2.0, size / 2.0, resolution + 1)
    ys = np.linspace(-size / 2.0, size / 2.0, resolution + 1)
    points = []
    for y in ys:
        for x in xs:
            z = lunar_crater_height_np(
                x,
                y,
                amplitude=amplitude,
                wavelength=wavelength,
                crater_count=crater_count,
                crater_min_radius=crater_min_radius,
                crater_max_radius=crater_max_radius,
                crater_depth_to_diameter=crater_depth_to_diameter,
                crater_rim_height_to_diameter=crater_rim_height_to_diameter,
                crater_field_size=size,
                crater_seed=crater_seed,
            )
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
    _bind_preview_material(stage, mesh.GetPrim(), "/World/Materials/LunarCraterTerrain", (0.36, 0.35, 0.32))


def set_camera() -> None:
    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(
        eye=[4.6, -5.2, 3.1],
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
    return assets_root


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


def make_gif_from_captures(frame_paths: list[Path], gif_path: Path, duration: float = 0.12) -> bool:
    if not frame_paths:
        return False
    import imageio.v2 as imageio

    frames = [imageio.imread(path) for path in frame_paths if path.exists()]
    if not frames:
        return False
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, duration=duration)
    return gif_path.exists()


def temporary_capture_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="physx_jetbot_frames_"))
