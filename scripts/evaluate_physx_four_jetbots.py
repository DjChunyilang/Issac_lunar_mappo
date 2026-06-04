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
    add_lunar_crater_terrain,
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


DEFAULT_OUTPUT = "outputs/logs/physx_four_jetbots/evaluation_metrics.json"
DEFAULT_CAPTURE = "outputs/figures/physx_four_jetbots/evaluation_scene.png"
DEFAULT_GIF = "outputs/videos/physx_four_jetbots/evaluation_rollout.gif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/exp_001_minimal.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/exp_001_minimal_proxy.pt")
    parser.add_argument("--terrain", choices=("flat", "rough", "lunar_crater"), default="lunar_crater")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--sim-steps-per-control", type=int, default=8)
    parser.add_argument("--max-linear", type=float, default=0.24)
    parser.add_argument("--max-angular", type=float, default=1.0)
    parser.add_argument("--start-spread", type=float, default=1.4)
    parser.add_argument("--start-jitter", type=float, default=0.0)
    parser.add_argument("--success-dmax", type=float, default=None)
    parser.add_argument("--collision-distance", type=float, default=0.22)
    parser.add_argument("--tilt-failure-deg", type=float, default=35.0)
    parser.add_argument("--phase-c-success-rate", type=float, default=0.9)
    parser.add_argument("--phase-c-collision-rate", type=float, default=0.02)
    parser.add_argument("--terrain-size", type=float, default=9.0)
    parser.add_argument("--terrain-resolution", type=int, default=64)
    parser.add_argument("--terrain-amplitude", type=float, default=0.025)
    parser.add_argument("--terrain-wavelength", type=float, default=2.8)
    parser.add_argument("--crater-count", type=int, default=7)
    parser.add_argument("--crater-min-radius", type=float, default=0.35)
    parser.add_argument("--crater-max-radius", type=float, default=1.15)
    parser.add_argument("--crater-depth-to-diameter", type=float, default=0.06)
    parser.add_argument("--crater-rim-height-to-diameter", type=float, default=0.015)
    parser.add_argument("--scripted", action="store_true", help="Use deterministic gather action instead of a checkpoint.")
    parser.add_argument("--render", action="store_true", help="Open viewport and save screenshot/GIF.")
    parser.add_argument("--render-all", action="store_true", help="Capture every episode instead of the first one only.")
    parser.add_argument("--capture-interval", type=int, default=5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Canonical run directory. When set, default output/capture/gif paths are placed under physx/.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="JSON metrics path.",
    )
    parser.add_argument(
        "--capture",
        default=DEFAULT_CAPTURE,
        help="Viewport screenshot path when --render is set.",
    )
    parser.add_argument(
        "--gif",
        default=DEFAULT_GIF,
        help="Viewport GIF path when --render is set.",
    )
    args = parser.parse_args()
    if args.run_dir:
        run_dir = resolve_path(args.run_dir)
        suffix = f"{args.terrain}_{'render' if args.render else 'headless'}"
        if args.output == DEFAULT_OUTPUT:
            args.output = str(run_dir / "physx" / "metrics" / f"{suffix}.json")
        if args.capture == DEFAULT_CAPTURE:
            args.capture = str(run_dir / "physx" / "figures" / f"{suffix}_scene.png")
        if args.gif == DEFAULT_GIF:
            args.gif = str(run_dir / "physx" / "videos" / f"{suffix}_rollout.gif")
    return args


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


def _terrain_profile(args: argparse.Namespace, episode_id: int) -> dict:
    return {
        "terrain": args.terrain,
        "size": args.terrain_size,
        "resolution": args.terrain_resolution,
        "amplitude": args.terrain_amplitude,
        "wavelength": args.terrain_wavelength,
        "crater_count": args.crater_count,
        "crater_min_radius": args.crater_min_radius,
        "crater_max_radius": args.crater_max_radius,
        "crater_depth_to_diameter": args.crater_depth_to_diameter,
        "crater_rim_height_to_diameter": args.crater_rim_height_to_diameter,
        "crater_seed": args.seed + episode_id,
        "scale_note": "meter-scale crater field for Jetbot/rover validation; based on small-crater lunar morphology ranges, not kilometer-scale basins",
    }


def _add_selected_terrain(stage, args: argparse.Namespace, episode_id: int) -> None:
    if args.terrain == "flat":
        add_flat_terrain(stage, size=args.terrain_size)
    elif args.terrain == "rough":
        add_rough_terrain(
            stage,
            size=args.terrain_size,
            resolution=max(12, args.terrain_resolution // 2),
            amplitude=args.terrain_amplitude,
            wavelength=args.terrain_wavelength,
        )
    else:
        add_lunar_crater_terrain(
            stage,
            size=args.terrain_size,
            resolution=args.terrain_resolution,
            amplitude=args.terrain_amplitude,
            wavelength=args.terrain_wavelength,
            crater_count=args.crater_count,
            crater_min_radius=args.crater_min_radius,
            crater_max_radius=args.crater_max_radius,
            crater_depth_to_diameter=args.crater_depth_to_diameter,
            crater_rim_height_to_diameter=args.crater_rim_height_to_diameter,
            crater_seed=args.seed + episode_id,
        )


def _episode_start_xy(args: argparse.Namespace, episode_id: int) -> np.ndarray:
    spread = float(args.start_spread)
    start_xy = np.array([[-spread, -spread], [spread, -spread], [-spread, spread], [spread, spread]], dtype=np.float32)
    if args.start_jitter > 0.0:
        rng = np.random.default_rng(args.seed + episode_id)
        start_xy += rng.normal(0.0, args.start_jitter, size=start_xy.shape).astype(np.float32)
    return start_xy


def phase_c_acceptance(
    success_rate: float,
    collision_rate: float,
    *,
    min_success_rate: float = 0.9,
    max_collision_rate: float = 0.02,
) -> dict:
    checks = {
        "success_rate": success_rate >= min_success_rate,
        "collision_rate": collision_rate <= max_collision_rate,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "success_rate": min_success_rate,
            "collision_rate": max_collision_rate,
        },
    }


def _run_episode(
    app,
    args: argparse.Namespace,
    cfg,
    action_fn,
    backend: str,
    jetbot_usd: str,
    episode_id: int,
) -> dict:
    import omni.timeline
    import omni.usd
    from isaacsim.robot.experimental.wheeled_robots.controllers import DifferentialController
    from isaacsim.robot.experimental.wheeled_robots.robots import WheeledRobot

    app.run_coroutine(omni.usd.get_context().new_stage_async())
    stage = omni.usd.get_context().get_stage()
    build_physics_scene(stage)
    _add_selected_terrain(stage, args, episode_id)

    start_xy = _episode_start_xy(args, episode_id)
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

    capture_enabled = args.render and (episode_id == 0 or args.render_all)
    if capture_enabled:
        set_camera()
    for _ in range(60):
        app.update()

    env = MultiRoverGatheringCore(cfg)
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
        collision_count += int(((distances < args.collision_distance) & (distances > 0.0)).sum().detach().cpu().item() // 2)

        if capture_enabled and step_id % max(1, args.capture_interval) == 0:
            frame_path = frame_dir / f"episode_{episode_id:03d}_frame_{step_id:04d}.png"
            if capture_viewport(app, frame_path):
                frame_paths.append(frame_path)

    wall_time = time.perf_counter() - wall_start
    _sync_env_from_robots(env, robots, prev_xy)
    final_metrics = compute_team_metrics(env.positions, env.velocities_xy)
    timeline.stop()

    screenshot_ok = capture_viewport(app, args.capture) if capture_enabled and episode_id == 0 else False
    gif_ok = False
    if capture_enabled and episode_id == 0:
        gif_ok = make_gif_from_captures(frame_paths, resolve_path(args.gif), duration=0.14)

    final_positions = env.positions[0].detach().cpu().numpy()
    total_displacement = np.linalg.norm(final_positions[:, :2] - start_xy, axis=1)
    max_tilt = float(np.max(max_tilt_history)) if max_tilt_history else 0.0
    success_dmax = cfg.success_thresholds.dmax if args.success_dmax is None else args.success_dmax
    success = bool(float(final_metrics.dmax[0]) <= success_dmax and max_tilt <= args.tilt_failure_deg)
    collision = collision_count > 0
    return {
        "episode": episode_id,
        "status": "ok",
        "backend": backend,
        "success": success,
        "collision": collision,
        "terrain_profile": _terrain_profile(args, episode_id),
        "start_positions": start_xy.tolist(),
        "wall_time_s": wall_time,
        "control_steps_per_s": args.steps / wall_time if wall_time > 0.0 else math.inf,
        "physics_updates_per_s": (args.steps * args.sim_steps_per_control) / wall_time if wall_time > 0.0 else math.inf,
        "final_dmax": float(final_metrics.dmax[0].detach().cpu()),
        "mean_dmax": float(np.mean(dmax_history)) if dmax_history else 0.0,
        "final_dispersion": float(final_metrics.dispersion[0].detach().cpu()),
        "mean_dispersion": float(np.mean(dispersion_history)) if dispersion_history else 0.0,
        "max_tilt_deg": max_tilt,
        "collision_count": collision_count,
        "total_displacement_xy": [float(v) for v in total_displacement],
        "final_positions": final_positions.tolist(),
        "capture_path": args.capture if screenshot_ok else None,
        "gif_path": args.gif if gif_ok else None,
    }


def _mean_or_none(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _summarize_episodes(
    args: argparse.Namespace,
    cfg,
    backend: str,
    jetbot_usd: str,
    episode_metrics: list[dict],
    wall_time_s: float,
) -> dict:
    success_rate = float(np.mean([float(item["success"]) for item in episode_metrics])) if episode_metrics else 0.0
    collision_rate = float(np.mean([float(item["collision"]) for item in episode_metrics])) if episode_metrics else 0.0
    acceptance = phase_c_acceptance(
        success_rate,
        collision_rate,
        min_success_rate=args.phase_c_success_rate,
        max_collision_rate=args.phase_c_collision_rate,
    )
    first = episode_metrics[0] if episode_metrics else {}
    summary = {
        "status": "ok",
        "backend": backend,
        "terrain": args.terrain,
        "terrain_profile": _terrain_profile(args, 0),
        "asset": jetbot_usd,
        "n_agents": cfg.task.n_agents,
        "episodes": len(episode_metrics),
        "steps": args.steps,
        "sim_steps_per_control": args.sim_steps_per_control,
        "wall_time_s": wall_time_s,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "mean_final_dmax": _mean_or_none([item["final_dmax"] for item in episode_metrics]),
        "mean_final_dispersion": _mean_or_none([item["final_dispersion"] for item in episode_metrics]),
        "mean_max_tilt_deg": _mean_or_none([item["max_tilt_deg"] for item in episode_metrics]),
        "mean_control_steps_per_s": _mean_or_none([item["control_steps_per_s"] for item in episode_metrics]),
        "mean_physics_updates_per_s": _mean_or_none([item["physics_updates_per_s"] for item in episode_metrics]),
        "phase_c_acceptance": acceptance,
        "episode_metrics": episode_metrics,
        "capture_path": first.get("capture_path"),
        "gif_path": first.get("gif_path"),
    }
    if len(episode_metrics) == 1:
        summary.update(
            {
                "success": first.get("success"),
                "final_dmax": first.get("final_dmax"),
                "mean_dmax": first.get("mean_dmax"),
                "final_dispersion": first.get("final_dispersion"),
                "mean_dispersion": first.get("mean_dispersion"),
                "max_tilt_deg": first.get("max_tilt_deg"),
                "collision_count": first.get("collision_count"),
                "total_displacement_xy": first.get("total_displacement_xy"),
                "final_positions": first.get("final_positions"),
            }
        )
    return summary


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
        assets_root = get_assets_root(app)
        jetbot_usd = assets_root + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
        cfg = cfg_from_experiment(args.config)
        cfg.simulation.num_envs = 1
        cfg.simulation.device = "cpu"
        if args.terrain == "flat":
            cfg.terrain.type = "flat_proxy"
            cfg.terrain.amplitude = 0.0
            cfg.terrain.crater_count = 0
        else:
            cfg.terrain.type = "lunar_crater_proxy" if args.terrain == "lunar_crater" else "lunar_heightfield_proxy"
            cfg.terrain.amplitude = args.terrain_amplitude
            cfg.terrain.wavelength = args.terrain_wavelength
            cfg.terrain.crater_count = args.crater_count if args.terrain == "lunar_crater" else 0
            cfg.terrain.crater_min_radius = args.crater_min_radius
            cfg.terrain.crater_max_radius = args.crater_max_radius
            cfg.terrain.crater_depth_to_diameter = args.crater_depth_to_diameter
            cfg.terrain.crater_rim_height_to_diameter = args.crater_rim_height_to_diameter
            cfg.terrain.crater_field_size = args.terrain_size
            cfg.terrain.crater_seed = args.seed
        probe_env = MultiRoverGatheringCore(cfg)
        action_fn, backend = _load_action_fn(args, cfg, probe_env.device)
        del probe_env

        wall_start = time.perf_counter()
        episode_metrics = [
            _run_episode(app, args, cfg, action_fn, backend, jetbot_usd, episode_id)
            for episode_id in range(max(1, args.episodes))
        ]
        summary = _summarize_episodes(args, cfg, backend, jetbot_usd, episode_metrics, time.perf_counter() - wall_start)
        output_path = resolve_path(args.output)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        app.close()


if __name__ == "__main__":
    main()
