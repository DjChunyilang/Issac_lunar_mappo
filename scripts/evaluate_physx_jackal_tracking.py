#!/usr/bin/env python
"""Run Clearpath Jackal PhysX trajectory tracking tests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from _common import ROOT
from physx_jackal_common import (
    JACKAL_MAX_ANGULAR_SPEED,
    JACKAL_MAX_LINEAR_SPEED,
    JACKAL_MAX_WHEEL_SPEED,
    JACKAL_PRIM_PATH,
    JACKAL_TRACK_WIDTH,
    JACKAL_WHEEL_DOF_NAMES,
    JACKAL_WHEEL_RADIUS,
    STRONG_LUNAR_CRATER_PROFILE,
    STRONG_LUNAR_CRATER_PATH_OFFSETS,
    JackalSkidSteerController,
    ReferencePath,
    TrackingControllerCfg,
    add_tracking_terrain,
    build_physics_scene,
    compute_tracking_command,
    capture_viewport,
    generate_reference_path,
    get_assets_root,
    jackal_usd_path,
    offset_reference_path,
    quat_wxyz_to_tilt_deg,
    quat_wxyz_to_yaw,
    resolve_path,
    set_camera,
    terrain_height,
    tracking_acceptance,
    yaw_to_quat_wxyz,
)

PROFILES = ("straight", "circle", "sine")
TERRAINS = ("flat", "strong_lunar_crater")
DEFAULT_RUN_ID = "jackal_tracking"


def _controller_grid(target_speed_mps: float) -> list[TrackingControllerCfg]:
    return [
        TrackingControllerCfg(lookahead_m=0.45, k_heading=1.6, k_cross_track=0.75, angular_scale=1.05, target_speed_mps=target_speed_mps),
        TrackingControllerCfg(lookahead_m=0.55, k_heading=1.8, k_cross_track=0.85, angular_scale=1.15, target_speed_mps=target_speed_mps),
        TrackingControllerCfg(lookahead_m=0.65, k_heading=1.9, k_cross_track=0.95, angular_scale=1.10, target_speed_mps=target_speed_mps),
        TrackingControllerCfg(lookahead_m=0.75, k_heading=1.6, k_cross_track=0.65, angular_scale=1.00, target_speed_mps=target_speed_mps),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=TERRAINS, default="flat")
    parser.add_argument("--profile", choices=(*PROFILES, "all"), default="all")
    parser.add_argument("--steps", type=int, default=660)
    parser.add_argument("--tune-steps", type=int, default=160)
    parser.add_argument("--sim-steps-per-control", type=int, default=4)
    parser.add_argument("--completion-stop-ratio", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--target-speed", type=float, default=0.35)
    parser.add_argument("--lookahead-m", type=float, default=None)
    parser.add_argument("--k-heading", type=float, default=None)
    parser.add_argument("--k-cross-track", type=float, default=None)
    parser.add_argument("--angular-scale", type=float, default=None)
    parser.add_argument(
        "--controller-json",
        default=None,
        help="Load selected controller parameters from a previous flat_tuning_grid.json.",
    )
    parser.add_argument("--wheel-radius", type=float, default=JACKAL_WHEEL_RADIUS)
    parser.add_argument("--track-width", type=float, default=JACKAL_TRACK_WIDTH)
    parser.add_argument("--max-linear", type=float, default=JACKAL_MAX_LINEAR_SPEED)
    parser.add_argument("--max-angular", type=float, default=JACKAL_MAX_ANGULAR_SPEED)
    parser.add_argument("--max-wheel-speed", type=float, default=JACKAL_MAX_WHEEL_SPEED)
    parser.add_argument("--tune-flat", action="store_true")
    parser.add_argument("--asset-smoke-only", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default=None, help="Accepted for checkpoint-evaluation compatibility.")
    parser.add_argument("--checkpoint", default=None, help="Accepted for checkpoint-evaluation compatibility.")
    return parser.parse_args()


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_run_dir(run_dir: str | None) -> Path:
    if run_dir:
        path = Path(run_dir)
        if not path.is_absolute():
            path = ROOT / path
    else:
        path = ROOT / "outputs" / "runs" / "physx_jackal_tracking" / f"{DEFAULT_RUN_ID}_{_utc_run_id()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _profiles_from_arg(profile: str) -> list[str]:
    return list(PROFILES) if profile == "all" else [profile]


def _pose(robot) -> tuple[np.ndarray, np.ndarray]:
    positions, orientations = robot.get_world_poses()
    return positions.numpy()[0], orientations.numpy()[0]


def _make_controller_cfg(args: argparse.Namespace, tuned: TrackingControllerCfg | None = None) -> TrackingControllerCfg:
    base = tuned or TrackingControllerCfg(target_speed_mps=args.target_speed)
    return TrackingControllerCfg(
        lookahead_m=float(args.lookahead_m if args.lookahead_m is not None else base.lookahead_m),
        k_heading=float(args.k_heading if args.k_heading is not None else base.k_heading),
        k_cross_track=float(args.k_cross_track if args.k_cross_track is not None else base.k_cross_track),
        angular_scale=float(args.angular_scale if args.angular_scale is not None else base.angular_scale),
        target_speed_mps=float(args.target_speed),
        max_linear_mps=float(args.max_linear),
        max_angular_radps=float(args.max_angular),
    )


def _load_controller_cfg(path: str | None, target_speed_mps: float) -> TrackingControllerCfg | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    values = payload.get("selected_controller") or payload.get("controller")
    if not isinstance(values, dict):
        raise ValueError(f"Controller JSON does not contain selected_controller: {resolved}")
    return TrackingControllerCfg(
        lookahead_m=float(values.get("lookahead_m", 0.45)),
        k_heading=float(values.get("k_heading", 1.6)),
        k_cross_track=float(values.get("k_cross_track", 0.75)),
        angular_scale=float(values.get("angular_scale", 1.05)),
        target_speed_mps=float(values.get("target_speed_mps", target_speed_mps)),
        max_linear_mps=float(values.get("max_linear_mps", JACKAL_MAX_LINEAR_SPEED)),
        max_angular_radps=float(values.get("max_angular_radps", JACKAL_MAX_ANGULAR_SPEED)),
    )


def _score_tracking_result(result: dict) -> float:
    return (
        float(result.get("rmse_cross_track_m", math.inf))
        + 0.20 * float(result.get("max_cross_track_m", math.inf))
        + 0.60 * max(0.0, 1.0 - float(result.get("path_completion_ratio", 0.0)))
        + 0.01 * float(result.get("max_tilt_deg", 0.0))
    )


def _plot_tracking(path: ReferencePath, records: list[dict], output_path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib unavailable: {exc}"

    actual = np.array([[row["x_m"], row["y_m"]] for row in records], dtype=np.float64)
    cross_track = np.array([row["cross_track_m"] for row in records], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(path.points_xy[:, 0], path.points_xy[:, 1], label="reference", linewidth=2.0)
    if len(actual):
        axes[0].plot(actual[:, 0], actual[:, 1], label="jackal", linewidth=1.6)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(cross_track, linewidth=1.6)
    axes[1].set_xlabel("control step")
    axes[1].set_ylabel("cross-track error [m]")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return None


def _write_timeseries(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "time_s",
        "x_m",
        "y_m",
        "z_m",
        "yaw_rad",
        "tilt_deg",
        "cross_track_m",
        "signed_cross_track_m",
        "heading_error_rad",
        "path_completion_ratio",
        "command_linear_mps",
        "command_angular_radps",
        "target_index",
        "ref_x_m",
        "ref_y_m",
        "left_wheel_radps",
        "right_wheel_radps",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field) for field in fields})


def _summarize_records(
    *,
    terrain: str,
    profile: str,
    records: list[dict],
    wall_time_s: float,
    controller_cfg: TrackingControllerCfg,
    path: ReferencePath,
    path_offset_xy: tuple[float, float],
) -> dict:
    cross = np.array([row["cross_track_m"] for row in records], dtype=np.float64)
    heading = np.array([abs(row["heading_error_rad"]) for row in records], dtype=np.float64)
    tilt = np.array([row["tilt_deg"] for row in records], dtype=np.float64)
    completion = max((row["path_completion_ratio"] for row in records), default=0.0)
    metrics = {
        "status": "ok",
        "terrain": terrain,
        "profile": profile,
        "steps": len(records),
        "reference_path_length_m": path.length_m,
        "wall_time_s": wall_time_s,
        "control_steps_per_s": len(records) / wall_time_s if wall_time_s > 0.0 else math.inf,
        "rmse_cross_track_m": float(np.sqrt(np.mean(cross**2))) if len(cross) else math.inf,
        "mean_cross_track_m": float(np.mean(cross)) if len(cross) else math.inf,
        "max_cross_track_m": float(np.max(cross)) if len(cross) else math.inf,
        "mean_heading_error_rad": float(np.mean(heading)) if len(heading) else math.inf,
        "max_tilt_deg": float(np.max(tilt)) if len(tilt) else math.inf,
        "path_completion_ratio": float(completion),
        "path_offset_xy": [float(path_offset_xy[0]), float(path_offset_xy[1])],
        "controller": asdict(controller_cfg),
    }
    metrics["acceptance"] = tracking_acceptance(metrics, terrain)
    metrics["passed"] = bool(metrics["acceptance"]["passed"])
    return metrics


def _run_tracking_profile(
    app,
    *,
    jackal_usd: str,
    terrain: str,
    profile: str,
    steps: int,
    sim_steps_per_control: int,
    controller_cfg: TrackingControllerCfg,
    wheel_controller: JackalSkidSteerController,
    output_dir: Path | None,
    render: bool,
    completion_stop_ratio: float,
) -> dict:
    import omni.timeline
    import omni.usd
    from isaacsim.robot.experimental.wheeled_robots.robots import WheeledRobot

    app.run_coroutine(omni.usd.get_context().new_stage_async())
    stage = omni.usd.get_context().get_stage()
    build_physics_scene(stage)
    add_tracking_terrain(stage, terrain)

    path = generate_reference_path(profile)
    path_offset_xy = (0.0, 0.0)
    if terrain == "strong_lunar_crater":
        path_offset_xy = STRONG_LUNAR_CRATER_PATH_OFFSETS[profile]
        path = offset_reference_path(path, path_offset_xy)
    start_xy = path.points_xy[0]
    start_yaw = float(path.yaws[0])
    start_z = terrain_height(terrain, float(start_xy[0]), float(start_xy[1])) + 0.22
    robot = WheeledRobot(
        paths=JACKAL_PRIM_PATH,
        wheel_dof_names=JACKAL_WHEEL_DOF_NAMES,
        usd_path=jackal_usd,
        positions=np.array([start_xy[0], start_xy[1], start_z], dtype=np.float32),
        orientations=yaw_to_quat_wxyz(start_yaw),
    )

    if render:
        set_camera()
    for _ in range(60):
        app.update()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(45):
        app.update()

    records: list[dict] = []
    progress_index = 0
    wall_start = time.perf_counter()
    for step_id in range(max(1, int(steps))):
        pos, quat = _pose(robot)
        yaw = quat_wxyz_to_yaw(quat)
        command, details = compute_tracking_command(path, pos[:2], yaw, controller_cfg, progress_index=progress_index)
        progress_index = max(progress_index, int(details["nearest_index"]))
        wheel_velocities = wheel_controller.forward(command)
        robot.apply_wheel_actions(wheel_velocities)
        for _ in range(max(1, int(sim_steps_per_control))):
            app.update()
        next_pos, next_quat = _pose(robot)
        next_yaw = quat_wxyz_to_yaw(next_quat)
        next_error = compute_tracking_command(
            path,
            next_pos[:2],
            next_yaw,
            controller_cfg,
            progress_index=progress_index,
        )[1]
        progress_index = max(progress_index, int(next_error["nearest_index"]))
        ref_xy = path.points_xy[int(next_error["nearest_index"])]
        records.append(
            {
                "step": step_id,
                "time_s": step_id * sim_steps_per_control * 0.05,
                "x_m": float(next_pos[0]),
                "y_m": float(next_pos[1]),
                "z_m": float(next_pos[2]),
                "yaw_rad": float(next_yaw),
                "tilt_deg": float(quat_wxyz_to_tilt_deg(next_quat)),
                "cross_track_m": float(next_error["cross_track_m"]),
                "signed_cross_track_m": float(next_error["signed_cross_track_m"]),
                "heading_error_rad": float(next_error["heading_error_rad"]),
                "path_completion_ratio": float(next_error["path_completion_ratio"]),
                "command_linear_mps": float(details["command_linear_mps"]),
                "command_angular_radps": float(details["command_angular_radps"]),
                "target_index": int(details["target_index"]),
                "ref_x_m": float(ref_xy[0]),
                "ref_y_m": float(ref_xy[1]),
                "left_wheel_radps": float(wheel_velocities[0]),
                "right_wheel_radps": float(wheel_velocities[1]),
            }
        )
        if float(next_error["path_completion_ratio"]) >= float(completion_stop_ratio):
            break

    wall_time_s = time.perf_counter() - wall_start
    timeline.stop()

    metrics = _summarize_records(
        terrain=terrain,
        profile=profile,
        records=records,
        wall_time_s=wall_time_s,
        controller_cfg=controller_cfg,
        path=path,
        path_offset_xy=path_offset_xy,
    )
    if output_dir is not None:
        metrics_dir = output_dir / "metrics"
        figures_dir = output_dir / "figures"
        timeseries_path = metrics_dir / f"{terrain}_{profile}_timeseries.csv"
        figure_path = figures_dir / f"{terrain}_{profile}_tracking.png"
        _write_timeseries(timeseries_path, records)
        plot_error = _plot_tracking(path, records, figure_path)
        metrics["artifacts"] = {
            "timeseries_csv": str(timeseries_path),
            "tracking_figure": str(figure_path) if plot_error is None else None,
            "tracking_figure_error": plot_error,
        }
        if render:
            scene_path = figures_dir / f"{terrain}_{profile}_scene.png"
            metrics["artifacts"]["scene_capture"] = str(scene_path) if capture_viewport(app, scene_path) else None
    return metrics


def _run_asset_smoke(app, *, jackal_usd: str, run_dir: Path) -> dict:
    import omni.usd
    from isaacsim.robot.experimental.wheeled_robots.robots import WheeledRobot

    app.run_coroutine(omni.usd.get_context().new_stage_async())
    stage = omni.usd.get_context().get_stage()
    build_physics_scene(stage)
    add_tracking_terrain(stage, "flat")
    robot = WheeledRobot(
        paths=JACKAL_PRIM_PATH,
        wheel_dof_names=JACKAL_WHEEL_DOF_NAMES,
        usd_path=jackal_usd,
        positions=np.array([0.0, 0.0, 0.22], dtype=np.float32),
    )
    for _ in range(50):
        app.update()
    result = {
        "status": "ok",
        "backend": "physx_jackal",
        "asset": jackal_usd,
        "prim_path": JACKAL_PRIM_PATH,
        "wheel_dof_names": JACKAL_WHEEL_DOF_NAMES,
        "resolved_dof_names": list(getattr(robot, "dof_names", [])),
    }
    output = run_dir / "physx" / "metrics" / "asset_smoke.json"
    _write_json(output, result)
    result["artifact"] = str(output)
    return result


def _run_tuning(
    app,
    *,
    jackal_usd: str,
    args: argparse.Namespace,
    wheel_controller: JackalSkidSteerController,
    run_dir: Path,
) -> tuple[TrackingControllerCfg, dict]:
    candidates = _controller_grid(args.target_speed)
    rows = []
    best_cfg = candidates[0]
    best_score = math.inf
    for cfg_index, cfg in enumerate(candidates):
        profile_results = []
        for profile in PROFILES:
            result = _run_tracking_profile(
                app,
                jackal_usd=jackal_usd,
                terrain="flat",
                profile=profile,
                steps=args.tune_steps,
                sim_steps_per_control=args.sim_steps_per_control,
                controller_cfg=cfg,
                wheel_controller=wheel_controller,
                output_dir=None,
                render=False,
                completion_stop_ratio=args.completion_stop_ratio,
            )
            profile_results.append(result)
        score = float(np.mean([_score_tracking_result(item) for item in profile_results]))
        row = {
            "candidate": cfg_index,
            "score": score,
            "controller": asdict(cfg),
            "profile_results": profile_results,
        }
        rows.append(row)
        if score < best_score:
            best_score = score
            best_cfg = cfg
    payload = {
        "status": "ok",
        "backend": "physx_jackal",
        "terrain": "flat",
        "selected_score": best_score,
        "selected_controller": asdict(best_cfg),
        "grid": rows,
    }
    output = run_dir / "physx" / "metrics" / "flat_tuning_grid.json"
    _write_json(output, payload)
    payload["artifact"] = str(output)
    return best_cfg, payload


def _write_manifest(run_dir: Path, summary: dict, command: list[str]) -> None:
    artifacts = summary.get("artifacts", {})
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "physx_jackal_tracking",
        "run": run_dir.name,
        "backend": "physx_jackal",
        "layout": "outputs/runs/<experiment>/<run>/physx/...",
        "command": command,
        "artifacts": artifacts,
        "summary": {
            "status": summary.get("status"),
            "passed": summary.get("passed"),
            "terrain": summary.get("terrain"),
            "profiles": summary.get("profiles"),
        },
    }
    _write_json(run_dir / "run_manifest.json", manifest)


def physx_acceptance(metrics: dict) -> dict:
    if "acceptance" in metrics and isinstance(metrics["acceptance"], dict):
        return metrics["acceptance"]
    return {
        "passed": bool(metrics.get("passed", False)),
        "checks": metrics.get("checks", {}),
        "thresholds": metrics.get("thresholds", {}),
    }


def physx_diagnostics(metrics: dict) -> dict:
    aggregate = metrics.get("aggregate", {}) if isinstance(metrics, dict) else {}
    return {
        "passed": metrics.get("passed") if isinstance(metrics, dict) else None,
        "mean_rmse_cross_track_m": aggregate.get("mean_rmse_cross_track_m"),
        "max_cross_track_m": aggregate.get("max_cross_track_m"),
        "min_path_completion_ratio": aggregate.get("min_path_completion_ratio"),
        "max_tilt_deg": aggregate.get("max_tilt_deg"),
    }


def _aggregate_results(profile_results: list[dict]) -> dict:
    return {
        "mean_rmse_cross_track_m": float(np.mean([item["rmse_cross_track_m"] for item in profile_results])),
        "max_cross_track_m": float(np.max([item["max_cross_track_m"] for item in profile_results])),
        "min_path_completion_ratio": float(np.min([item["path_completion_ratio"] for item in profile_results])),
        "max_tilt_deg": float(np.max([item["max_tilt_deg"] for item in profile_results])),
        "mean_control_steps_per_s": float(np.mean([item["control_steps_per_s"] for item in profile_results])),
    }


def main() -> None:
    args = parse_args()
    run_dir = _resolve_run_dir(args.run_dir)

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
        jackal_usd = jackal_usd_path(assets_root)
        if args.asset_smoke_only:
            summary = _run_asset_smoke(app, jackal_usd=jackal_usd, run_dir=run_dir)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return

        wheel_controller = JackalSkidSteerController(
            wheel_radius=args.wheel_radius,
            track_width=args.track_width,
            max_linear_speed=args.max_linear,
            max_angular_speed=args.max_angular,
            max_wheel_speed=args.max_wheel_speed,
        )
        tuning_payload = None
        tuned_cfg = _load_controller_cfg(args.controller_json, args.target_speed)
        if args.tune_flat:
            tuned_cfg, tuning_payload = _run_tuning(
                app,
                jackal_usd=jackal_usd,
                args=args,
                wheel_controller=wheel_controller,
                run_dir=run_dir,
            )

        controller_cfg = _make_controller_cfg(args, tuned=tuned_cfg)
        output_dir = run_dir / "physx"
        profiles = _profiles_from_arg(args.profile)
        profile_results = [
            _run_tracking_profile(
                app,
                jackal_usd=jackal_usd,
                terrain=args.terrain,
                profile=profile,
                steps=args.steps,
                sim_steps_per_control=args.sim_steps_per_control,
                controller_cfg=controller_cfg,
                wheel_controller=wheel_controller,
                output_dir=output_dir,
                render=args.render,
                completion_stop_ratio=args.completion_stop_ratio,
            )
            for profile in profiles
        ]
        aggregate = _aggregate_results(profile_results)
        passed = all(bool(item.get("passed")) for item in profile_results)
        artifacts = {
            "tracking_summary": str(
                Path(args.output)
                if args.output
                else run_dir / "physx" / "metrics" / "tracking_summary.json"
            ),
            "flat_tuning_grid": tuning_payload.get("artifact") if tuning_payload else None,
            "profile_artifacts": {
                item["profile"]: item.get("artifacts", {}) for item in profile_results
            },
        }
        summary = {
            "status": "ok",
            "backend": "physx_jackal",
            "asset": jackal_usd,
            "terrain": args.terrain,
            "terrain_profile": (
                {**STRONG_LUNAR_CRATER_PROFILE, "path_offsets": STRONG_LUNAR_CRATER_PATH_OFFSETS}
                if args.terrain == "strong_lunar_crater"
                else {"terrain": "flat"}
            ),
            "profiles": profiles,
            "steps": args.steps,
            "sim_steps_per_control": args.sim_steps_per_control,
            "controller": asdict(controller_cfg),
            "passed": passed,
            "aggregate": aggregate,
            "profile_results": profile_results,
            "tuning": tuning_payload,
            "artifacts": artifacts,
        }
        output_path = resolve_path(args.output) if args.output else run_dir / "physx" / "metrics" / "tracking_summary.json"
        _write_json(output_path, summary)
        summary["artifact"] = str(output_path)
        _write_manifest(run_dir, summary, command=["scripts/evaluate_physx_jackal_tracking.py"])
        _write_json(output_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        app.close()


if __name__ == "__main__":
    main()
