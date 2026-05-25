#!/usr/bin/env python
"""Evaluate a trained proxy policy in a four-Jetbot Isaac Sim PhysX scene."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import decode_action
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.simple_controller import compute_control
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import generate_trajectory
from physx_jetbot_common import (
    JETBOT_WHEEL_BASE,
    JETBOT_WHEEL_DOF_NAMES,
    JETBOT_WHEEL_RADIUS,
    add_flat_terrain,
    add_rough_terrain,
    build_physics_scene,
    capture_viewport,
    get_assets_root,
    make_gif_from_captures,
    quat_wxyz_to_tilt_deg,
    quat_wxyz_to_yaw,
    resolve_path,
    set_camera,
    temporary_capture_dir,
)
from play import _load_policy_players


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/exp_001_minimal.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/exp_001_minimal_proxy.pt")
    parser.add_argument("--terrain", choices=("flat", "rough"), default="rough")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--sim-steps-per-control", type=int, default=8)
    parser.add_argument("--max-linear", type=float, default=0.24)
    parser.add_argument("--max-angular", type=float, default=1.0)
    parser.add_argument("--scripted", action="store_true", help="Use deterministic gather action instead of a checkpoint.")
    parser.add_argument("--render", action="store_true", help="Open viewport and save screenshot/GIF.")
    parser.add_argument("--capture-interval", type=int, default=5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--output",
        default="outputs/logs/physx_four_jetbots/evaluation_metrics.json",
        help="JSON metrics path.",
    )
    parser.add_argument(
        "--capture",
        default="outputs/figures/physx_four_jetbots/evaluation_scene.png",
        help="Viewport screenshot path when --render is set.",
    )
    parser.add_argument(
        "--gif",
        default="outputs/videos/physx_four_jetbots/evaluation_rollout.gif",
        help="Viewport GIF path when --render is set.",
    )
    return parser.parse_args()


def _scripted_gather_action(env: MultiRoverGatheringCore) -> torch.Tensor:
    positions_xy = env.positions[..., :2]
    centroid_xy = positions_xy.mean(dim=1, keepdim=True)
    world_delta = centroid_xy - positions_xy
    cos_yaw = torch.cos(env.yaws)
    sin_yaw = torch.sin(env.yaws)
    local_x = cos_yaw * world_delta[..., 0] + sin_yaw * world_delta[..., 1]
    local_y = -sin_yaw * world_delta[..., 0] + cos_yaw * world_delta[..., 1]
    rho = torch.linalg.norm(torch.stack((local_x, local_y), dim=-1), dim=-1)
    rho = torch.clamp(rho, 0.0, env.cfg.planner.rho_max)
    beta = torch.atan2(local_y, local_x)
    beta = torch.clamp(beta, -env.cfg.planner.beta_max, env.cfg.planner.beta_max)
    return torch.stack(
        (
            2.0 * rho / env.cfg.planner.rho_max - 1.0,
            beta / env.cfg.planner.beta_max,
        ),
        dim=-1,
    )


def _pose(robot) -> tuple[np.ndarray, np.ndarray]:
    positions, orientations = robot.get_world_poses()
    return positions.numpy()[0], orientations.numpy()[0]


def _sync_env_from_robots(env: MultiRoverGatheringCore, robots, prev_xy: torch.Tensor | None) -> torch.Tensor:
    positions = []
    yaws = []
    tilts = []
    for robot in robots:
        pos, quat = _pose(robot)
        positions.append(pos)
        yaws.append(quat_wxyz_to_yaw(quat))
        tilts.append(quat_wxyz_to_tilt_deg(quat))
    pos_tensor = torch.tensor(np.asarray(positions), dtype=torch.float32, device=env.device)
    yaw_tensor = torch.tensor(yaws, dtype=torch.float32, device=env.device)
    env.positions[0] = pos_tensor
    env.yaws[0] = yaw_tensor
    if prev_xy is None:
        env.velocities_xy.zero_()
    else:
        env.velocities_xy[0] = (pos_tensor[:, :2] - prev_xy) / env.cfg.simulation.planning_dt
    env.angular_velocities.zero_()
    env.metrics = compute_team_metrics(env.positions, env.velocities_xy)
    return torch.tensor(tilts, dtype=torch.float32, device=env.device)


def _load_action_fn(args, cfg, device):
    if args.scripted:
        return None, "scripted"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    return _load_policy_players(checkpoint, cfg, device)


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
            add_rough_terrain(stage, size=8.0, amplitude=0.045)
        else:
            add_flat_terrain(stage, size=9.0)

        assets_root = get_assets_root(app)
        jetbot_usd = assets_root + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
        start_xy = np.array([[-1.4, -1.4], [1.4, -1.4], [-1.4, 1.4], [1.4, 1.4]], dtype=np.float32)
        robots = []
        controllers = []
        for robot_id, xy in enumerate(start_xy):
            robot = WheeledRobot(
                paths=f"/World/Jetbot_{robot_id}",
                wheel_dof_names=JETBOT_WHEEL_DOF_NAMES,
                usd_path=jetbot_usd,
                positions=np.array([xy[0], xy[1], 0.18], dtype=np.float32),
            )
            robots.append(robot)
            controllers.append(
                DifferentialController(
                    wheel_radius=JETBOT_WHEEL_RADIUS,
                    wheel_base=JETBOT_WHEEL_BASE,
                    max_linear_speed=args.max_linear,
                    max_angular_speed=args.max_angular,
                    max_wheel_speed=12.0,
                )
            )

        if args.render:
            set_camera()
        for _ in range(60):
            app.update()

        cfg = cfg_from_experiment(args.config)
        cfg.simulation.num_envs = 1
        cfg.simulation.device = "cpu"
        if args.terrain == "rough":
            cfg.terrain.type = "lunar_heightfield_proxy"
            cfg.terrain.amplitude = 0.12
        env = MultiRoverGatheringCore(cfg)
        action_fn, backend = _load_action_fn(args, cfg, env.device)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(45):
            app.update()

        prev_xy = None
        dmax_history = []
        dispersion_history = []
        max_tilt_history = []
        collision_count = 0
        frame_paths = []
        frame_dir = temporary_capture_dir()
        wall_start = time.perf_counter()

        for step_id in range(args.steps):
            tilts = _sync_env_from_robots(env, robots, prev_xy)
            prev_xy = env.positions[0, :, :2].detach().clone()
            actor_obs, _ = env.get_observations()
            with torch.no_grad():
                action = _scripted_gather_action(env) if action_fn is None else action_fn(actor_obs)
            decoded = decode_action(action, env.positions, env.yaws, env.cfg.planner)
            trajectory = generate_trajectory(
                env.positions,
                decoded.world_subgoal,
                env.cfg.trajectory_generator,
                env.cfg.simulation.planning_dt,
                current_yaws=env.yaws,
            )
            control = compute_control(env.positions, env.yaws, trajectory, env.cfg.low_level_control)

            for robot, controller, linear, angular in zip(
                robots,
                controllers,
                control.linear[0].detach().cpu().numpy(),
                control.angular[0].detach().cpu().numpy(),
                strict=True,
            ):
                command = np.array(
                    [
                        np.clip(float(linear), -args.max_linear, args.max_linear),
                        np.clip(float(angular), -args.max_angular, args.max_angular),
                    ],
                    dtype=np.float64,
                )
                robot.apply_wheel_actions(controller.forward(command))

            for _ in range(args.sim_steps_per_control):
                app.update()

            metrics = compute_team_metrics(env.positions, env.velocities_xy)
            dmax_history.append(float(metrics.dmax[0].detach().cpu()))
            dispersion_history.append(float(metrics.dispersion[0].detach().cpu()))
            max_tilt_history.append(float(tilts.max().detach().cpu()))
            distances = torch.cdist(env.positions[0, :, :2], env.positions[0, :, :2])
            collision_count += int(((distances < 0.22) & (distances > 0.0)).sum().detach().cpu().item() // 2)

            if args.render and step_id % max(1, args.capture_interval) == 0:
                frame_path = frame_dir / f"frame_{step_id:04d}.png"
                if capture_viewport(app, frame_path):
                    frame_paths.append(frame_path)

        wall_time = time.perf_counter() - wall_start
        _sync_env_from_robots(env, robots, prev_xy)
        final_metrics = compute_team_metrics(env.positions, env.velocities_xy)
        timeline.stop()

        screenshot_ok = capture_viewport(app, args.capture) if args.render else False
        gif_ok = False
        if args.render:
            gif_ok = make_gif_from_captures(frame_paths, resolve_path(args.gif), duration=0.14)

        final_positions = env.positions[0].detach().cpu().numpy()
        total_displacement = np.linalg.norm(final_positions[:, :2] - start_xy, axis=1)
        success = bool(float(final_metrics.dmax[0]) <= cfg.success_thresholds.dmax)
        summary = {
            "status": "ok",
            "backend": backend,
            "terrain": args.terrain,
            "asset": jetbot_usd,
            "n_agents": cfg.task.n_agents,
            "steps": args.steps,
            "sim_steps_per_control": args.sim_steps_per_control,
            "wall_time_s": wall_time,
            "control_steps_per_s": args.steps / wall_time if wall_time > 0.0 else math.inf,
            "physics_updates_per_s": (args.steps * args.sim_steps_per_control) / wall_time if wall_time > 0.0 else math.inf,
            "success": success,
            "final_dmax": float(final_metrics.dmax[0].detach().cpu()),
            "mean_dmax": float(np.mean(dmax_history)) if dmax_history else 0.0,
            "final_dispersion": float(final_metrics.dispersion[0].detach().cpu()),
            "mean_dispersion": float(np.mean(dispersion_history)) if dispersion_history else 0.0,
            "max_tilt_deg": float(np.max(max_tilt_history)) if max_tilt_history else 0.0,
            "collision_count": collision_count,
            "total_displacement_xy": [float(v) for v in total_displacement],
            "final_positions": final_positions.tolist(),
            "capture_path": args.capture if screenshot_ok else None,
            "gif_path": args.gif if gif_ok else None,
        }
        output_path = resolve_path(args.output)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        app.close()


if __name__ == "__main__":
    main()
