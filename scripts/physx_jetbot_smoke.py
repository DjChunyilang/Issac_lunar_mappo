#!/usr/bin/env python
"""Run a single official Jetbot PhysX wheeled-robot smoke test."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from physx_jetbot_common import (
    JETBOT_PRIM_PATH,
    JETBOT_WHEEL_BASE,
    JETBOT_WHEEL_DOF_NAMES,
    JETBOT_WHEEL_RADIUS,
    add_flat_terrain,
    add_rough_terrain,
    build_physics_scene,
    capture_viewport,
    get_assets_root,
    quat_wxyz_to_tilt_deg,
    resolve_path,
    set_camera,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=("flat", "rough"), default="flat")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--linear", type=float, default=0.18)
    parser.add_argument("--angular", type=float, default=0.0)
    parser.add_argument("--render", action="store_true", help="Open viewport and capture a screenshot.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--output",
        default="outputs/logs/physx_jetbot_smoke/jetbot_smoke.json",
        help="JSON metrics path.",
    )
    parser.add_argument(
        "--capture",
        default="outputs/figures/physx_jetbot_smoke/jetbot_smoke.png",
        help="Optional viewport screenshot path when --render is set.",
    )
    return parser.parse_args()


def _pose(robot) -> tuple[np.ndarray, np.ndarray]:
    positions, orientations = robot.get_world_poses()
    return positions.numpy()[0], orientations.numpy()[0]


def main() -> None:
    args = parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": not args.render,
            "width": args.width,
            "height": args.height,
            "renderer": "RealTimePathTracing" if args.render else "RayTracedLighting",
        }
    )

    try:
        import omni.timeline
        import omni.usd
        from isaacsim.robot.experimental.wheeled_robots.controllers import DifferentialController
        from isaacsim.robot.experimental.wheeled_robots.robots import WheeledRobot

        app.run_coroutine(omni.usd.get_context().new_stage_async())
        stage = omni.usd.get_context().get_stage()
        build_physics_scene(stage)
        if args.terrain == "rough":
            add_rough_terrain(stage)
        else:
            add_flat_terrain(stage)

        assets_root = get_assets_root(app)
        jetbot_usd = assets_root + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
        robot = WheeledRobot(
            paths=JETBOT_PRIM_PATH,
            wheel_dof_names=JETBOT_WHEEL_DOF_NAMES,
            usd_path=jetbot_usd,
            positions=np.array([0.0, 0.0, 0.16], dtype=np.float32),
        )
        controller = DifferentialController(
            wheel_radius=JETBOT_WHEEL_RADIUS,
            wheel_base=JETBOT_WHEEL_BASE,
            max_linear_speed=0.30,
            max_angular_speed=1.0,
            max_wheel_speed=12.0,
        )

        if args.render:
            set_camera()
        for _ in range(45):
            app.update()

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(30):
            app.update()

        start_pos, _ = _pose(robot)
        min_z = float(start_pos[2])
        max_tilt = 0.0
        wall_start = time.perf_counter()
        command = np.array([args.linear, args.angular], dtype=np.float64)
        for _ in range(args.steps):
            wheel_velocities = controller.forward(command)
            robot.apply_wheel_actions(wheel_velocities)
            app.update()
            pos, quat = _pose(robot)
            min_z = min(min_z, float(pos[2]))
            max_tilt = max(max_tilt, quat_wxyz_to_tilt_deg(quat))
        wall_time = time.perf_counter() - wall_start
        end_pos, end_quat = _pose(robot)
        timeline.stop()

        displacement = float(np.linalg.norm(end_pos[:2] - start_pos[:2]))
        sim_steps_per_s = float(args.steps / wall_time) if wall_time > 0.0 else math.inf
        captured = False
        if args.render:
            captured = capture_viewport(app, args.capture)

        ok = bool(np.isfinite(end_pos).all() and displacement > 0.02 and min_z > -0.10 and max_tilt < 85.0)
        summary = {
            "status": "ok" if ok else "failed",
            "terrain": args.terrain,
            "asset": jetbot_usd,
            "prim_path": JETBOT_PRIM_PATH,
            "wheel_dof_names": JETBOT_WHEEL_DOF_NAMES,
            "wheel_radius": JETBOT_WHEEL_RADIUS,
            "wheel_base": JETBOT_WHEEL_BASE,
            "steps": args.steps,
            "command": {"linear": args.linear, "angular": args.angular},
            "start_position": [float(v) for v in start_pos],
            "end_position": [float(v) for v in end_pos],
            "end_orientation_wxyz": [float(v) for v in end_quat],
            "displacement_xy": displacement,
            "min_z": min_z,
            "max_tilt_deg": max_tilt,
            "wall_time_s": wall_time,
            "sim_steps_per_s": sim_steps_per_s,
            "capture_path": args.capture if captured else None,
        }

        output_path = resolve_path(args.output)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        if not ok:
            raise SystemExit(1)
    finally:
        app.close()


if __name__ == "__main__":
    main()
