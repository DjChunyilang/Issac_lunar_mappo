#!/usr/bin/env python
"""Open an Isaac Sim scene with four visible proxy rovers on a flat plane."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280, help="Viewport width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Viewport height in pixels.")
    parser.add_argument(
        "--renderer",
        default="RealTimePathTracing",
        choices=("RayTracedLighting", "PathTracing", "RealTimePathTracing"),
        help="Isaac Sim renderer.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without opening the GUI window.")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=120.0,
        help="Seconds to keep the scene visible before closing. Ignored when --keep-open is set.",
    )
    parser.add_argument("--keep-open", action="store_true", help="Keep the window open until Ctrl+C.")
    parser.add_argument(
        "--capture",
        type=Path,
        default=Path("outputs/figures/isaac_render/proxy_rovers_scene.png"),
        help="PNG capture path. Use --no-capture to disable.",
    )
    parser.add_argument("--no-capture", action="store_true", help="Do not write a viewport screenshot.")
    parser.add_argument(
        "--stage-out",
        type=Path,
        default=Path("outputs/isaac_scenes/proxy_rovers_scene.usda"),
        help="USD stage export path.",
    )
    parser.add_argument("--warmup-frames", type=int, default=90, help="Frames to render before capture.")
    return parser.parse_args()


def _set_xform(schema, translate=(0.0, 0.0, 0.0), rotate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)) -> None:
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(schema.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rotate != (0.0, 0.0, 0.0):
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(*rotate))
    xformable.AddScaleOp().Set(Gf.Vec3f(*scale))


def _make_material(stage, path: str, color: tuple[float, float, float], roughness: float = 0.65):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_material(schema, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(schema.GetPrim()).Bind(material)


def _make_cube(stage, path: str, material, translate, scale, rotate=(0.0, 0.0, 0.0)):
    from pxr import UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_xform(cube, translate=translate, rotate=rotate, scale=scale)
    _bind_material(cube, material)
    return cube


def _make_cylinder(stage, path: str, material, translate, radius: float, height: float, axis: str = "Y"):
    from pxr import UsdGeom

    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(height))
    cylinder.CreateAxisAttr(axis)
    _set_xform(cylinder, translate=translate)
    _bind_material(cylinder, material)
    return cylinder


def _build_rover(stage, path: str, position, yaw_deg: float, body_material, wheel_material, marker_material) -> None:
    from pxr import UsdGeom

    rover = UsdGeom.Xform.Define(stage, path)
    _set_xform(rover, translate=position, rotate=(0.0, 0.0, yaw_deg))

    _make_cube(stage, f"{path}/Body", body_material, translate=(0.0, 0.0, 0.28), scale=(0.95, 0.58, 0.25))
    _make_cube(stage, f"{path}/Cabin", body_material, translate=(-0.12, 0.0, 0.50), scale=(0.35, 0.36, 0.20))
    _make_cube(stage, f"{path}/FrontMarker", marker_material, translate=(0.56, 0.0, 0.34), scale=(0.08, 0.42, 0.12))

    for wheel_id, x in enumerate((-0.32, 0.32)):
        for side_id, y in enumerate((-0.38, 0.38)):
            _make_cylinder(
                stage,
                f"{path}/Wheel_{wheel_id}_{side_id}",
                wheel_material,
                translate=(x, y, 0.17),
                radius=0.17,
                height=0.14,
                axis="Y",
            )


def build_scene(stage) -> None:
    from pxr import UsdGeom, UsdLux

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    ground_mat = _make_material(stage, "/World/Materials/Ground", (0.38, 0.40, 0.38))
    grid_mat = _make_material(stage, "/World/Materials/Grid", (0.12, 0.13, 0.13))
    wheel_mat = _make_material(stage, "/World/Materials/WheelBlack", (0.015, 0.015, 0.018), roughness=0.85)
    marker_mat = _make_material(stage, "/World/Materials/MarkerWhite", (0.93, 0.93, 0.85), roughness=0.4)
    rover_mats = [
        _make_material(stage, "/World/Materials/RoverBlue", (0.05, 0.30, 0.90)),
        _make_material(stage, "/World/Materials/RoverOrange", (0.95, 0.42, 0.08)),
        _make_material(stage, "/World/Materials/RoverGreen", (0.08, 0.58, 0.25)),
        _make_material(stage, "/World/Materials/RoverRed", (0.82, 0.08, 0.12)),
    ]

    _make_cube(stage, "/World/Ground", ground_mat, translate=(0.0, 0.0, -0.025), scale=(8.0, 8.0, 0.05))
    for line_id, offset in enumerate(range(-4, 5)):
        thickness = 0.025 if offset else 0.055
        _make_cube(
            stage,
            f"/World/Grid/X_{line_id}",
            grid_mat,
            translate=(0.0, float(offset), 0.004),
            scale=(8.0, thickness, 0.008),
        )
        _make_cube(
            stage,
            f"/World/Grid/Y_{line_id}",
            grid_mat,
            translate=(float(offset), 0.0, 0.005),
            scale=(thickness, 8.0, 0.008),
        )

    rover_specs = [
        ("/World/Rover_0", (-1.6, -1.6, 0.0), 45.0),
        ("/World/Rover_1", (1.6, -1.6, 0.0), 135.0),
        ("/World/Rover_2", (-1.6, 1.6, 0.0), -45.0),
        ("/World/Rover_3", (1.6, 1.6, 0.0), -135.0),
    ]
    for rover_id, (path, position, yaw_deg) in enumerate(rover_specs):
        _build_rover(stage, path, position, yaw_deg, rover_mats[rover_id], wheel_mat, marker_mat)

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(3200.0)
    sun.CreateAngleAttr(0.35)
    _set_xform(sun, rotate=(-45.0, 0.0, 35.0))

    fill = UsdLux.DomeLight.Define(stage, "/World/Sky")
    fill.CreateIntensityAttr(450.0)


def _capture_viewport(app, capture_path: Path) -> bool:
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport, next_viewport_frame_async

    capture_path.parent.mkdir(parents=True, exist_ok=True)
    viewport = get_active_viewport()
    if viewport is None:
        print("PROXY_ROVER_SCENE_CAPTURE_SKIPPED no_active_viewport", flush=True)
        return False

    app.run_coroutine(next_viewport_frame_async(viewport, n_frames=5))
    capture = capture_viewport_to_file(viewport, file_path=str(capture_path))
    result = app.run_coroutine(capture.wait_for_result(completion_frames=30))
    for _ in range(10):
        app.update()
    if bool(result) and capture_path.exists():
        print(f"PROXY_ROVER_SCENE_CAPTURED {capture_path}", flush=True)
        return True
    print(f"PROXY_ROVER_SCENE_CAPTURE_FAILED {capture_path}", flush=True)
    return False


def main() -> None:
    args = parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "width": args.width,
            "height": args.height,
            "renderer": args.renderer,
        }
    )

    try:
        import omni.usd
        from isaacsim.core.utils.viewports import set_camera_view

        stage = omni.usd.get_context().get_stage()
        build_scene(stage)

        set_camera_view(
            eye=[5.2, -6.2, 4.0],
            target=[0.0, 0.0, 0.2],
            camera_prim_path="/OmniverseKit_Persp",
        )

        args.stage_out.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(args.stage_out))
        print(f"PROXY_ROVER_STAGE_EXPORTED {args.stage_out}", flush=True)

        for _ in range(max(1, args.warmup_frames)):
            app.update()

        if not args.no_capture:
            _capture_viewport(app, args.capture)

        print(
            "PROXY_ROVER_SCENE_READY "
            f"rovers=4 ground=/World/Ground renderer={args.renderer} headless={args.headless}",
            flush=True,
        )

        if args.keep_open:
            print("PROXY_ROVER_SCENE_KEEP_OPEN press Ctrl+C in this terminal to close", flush=True)
            while True:
                app.update()
                time.sleep(1.0 / 60.0)

        end_time = time.monotonic() + max(0.0, args.duration_s)
        while time.monotonic() < end_time:
            app.update()
            time.sleep(1.0 / 60.0)
        print(f"PROXY_ROVER_SCENE_VISIBLE_SECONDS {args.duration_s:.1f}", flush=True)
    except KeyboardInterrupt:
        print("PROXY_ROVER_SCENE_INTERRUPTED", flush=True)
    finally:
        app.close()
        print("PROXY_ROVER_SCENE_CLOSED", flush=True)


if __name__ == "__main__":
    main()
