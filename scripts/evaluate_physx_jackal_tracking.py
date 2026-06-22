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
    JACKAL_ROOT_Z_OFFSET,
    JACKAL_TRACK_WIDTH,
    JACKAL_WHEEL_DOF_NAMES,
    JACKAL_WHEEL_RADIUS,
    STRONG_LUNAR_CRATER_PROFILE,
    STRONG_LUNAR_CRATER_PATH_OFFSETS,
    JackalSkidSteerController,
    ReferencePath,
    TrackingControllerCfg,
    TrackingControllerState,
    add_tracking_terrain,
    build_physics_scene,
    compute_chassis_servo_command,
    compute_tracking_command,
    capture_viewport,
    generate_reference_path,
    get_assets_root,
    jackal_usd_path,
    nearest_path_index,
    offset_reference_path,
    quat_wxyz_to_tilt_deg,
    quat_wxyz_to_yaw,
    resolve_path,
    set_camera,
    terrain_height,
    tracking_acceptance,
    tracking_error,
    yaw_to_quat_wxyz,
)

PROFILES = ("straight", "circle", "sine", "double_lane_change")
TERRAINS = ("flat", "strong_lunar_crater")
DEFAULT_RUN_ID = "jackal_tracking"
DEFAULT_PHYSICS_DT = 0.01


def _controller_grid(target_speed_mps: float) -> list[TrackingControllerCfg]:
    return [
        TrackingControllerCfg(
            mode="pure_pursuit",
            lookahead_m=0.35,
            k_heading=1.6,
            k_cross_track=0.75,
            angular_scale=1.05,
            speed_kp=0.20,
            speed_ki=0.02,
            yaw_rate_kp=0.50,
            yaw_rate_ki=0.02,
            target_speed_mps=target_speed_mps,
        ),
        TrackingControllerCfg(
            mode="pure_pursuit",
            lookahead_m=0.45,
            k_heading=1.8,
            k_cross_track=0.85,
            angular_scale=1.10,
            speed_kp=0.20,
            speed_ki=0.02,
            yaw_rate_kp=0.50,
            yaw_rate_ki=0.02,
            target_speed_mps=target_speed_mps,
        ),
        TrackingControllerCfg(
            mode="pure_pursuit",
            lookahead_m=0.55,
            k_heading=1.6,
            k_cross_track=0.80,
            angular_scale=1.05,
            speed_kp=0.25,
            speed_ki=0.02,
            yaw_rate_kp=0.60,
            yaw_rate_ki=0.02,
            target_speed_mps=target_speed_mps,
        ),
        TrackingControllerCfg(
            mode="pure_pursuit",
            lookahead_m=0.65,
            k_heading=1.6,
            k_cross_track=0.65,
            angular_scale=1.00,
            speed_kp=0.25,
            speed_ki=0.02,
            yaw_rate_kp=0.60,
            yaw_rate_ki=0.02,
            target_speed_mps=target_speed_mps,
        ),
    ]


def _stanley_coarse_grid(target_speed_mps: float) -> list[TrackingControllerCfg]:
    candidates: list[TrackingControllerCfg] = []
    for heading_gain in (0.9, 1.2, 1.5):
        for stanley_gain in (1.2, 1.6, 2.0):
            for lookahead_m in (0.28, 0.38):
                for yaw_rate_kp in (0.35, 0.60):
                    for speed_kp in (0.12, 0.25):
                        candidates.append(
                            TrackingControllerCfg(
                                mode="stanley_pid",
                                lookahead_m=lookahead_m,
                                heading_gain=heading_gain,
                                stanley_gain=stanley_gain,
                                curvature_feedforward_gain=0.0,
                                softening_speed_mps=0.12,
                                speed_kp=speed_kp,
                                speed_ki=0.02,
                                speed_kd=0.0,
                                yaw_rate_kp=yaw_rate_kp,
                                yaw_rate_ki=0.02,
                                yaw_rate_kd=0.0,
                                max_linear_servo_correction_mps=0.30,
                                max_angular_servo_correction_radps=1.0,
                                max_linear_accel_mps2=1.4,
                                max_angular_accel_radps2=5.0,
                                target_speed_mps=target_speed_mps,
                            )
                        )
    return candidates


def _unique_controller_grid(candidates: Iterable[TrackingControllerCfg]) -> list[TrackingControllerCfg]:
    seen: set[tuple] = set()
    unique: list[TrackingControllerCfg] = []
    for cfg in candidates:
        key = tuple(sorted(asdict(cfg).items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(cfg)
    return unique


def _stanley_refined_grid(base: TrackingControllerCfg, target_speed_mps: float) -> list[TrackingControllerCfg]:
    def build(**overrides: float) -> TrackingControllerCfg:
        values = asdict(base)
        values.update(overrides)
        values["target_speed_mps"] = target_speed_mps
        return TrackingControllerCfg(**values)

    candidates: list[TrackingControllerCfg] = [build()]
    for delta in (-0.12, 0.12):
        candidates.append(build(heading_gain=max(0.5, base.heading_gain + delta)))
        candidates.append(build(stanley_gain=max(0.2, base.stanley_gain + delta * 1.5)))
        candidates.append(build(lookahead_m=max(0.15, base.lookahead_m + delta)))
        candidates.append(build(yaw_rate_kp=max(0.1, base.yaw_rate_kp + delta * 0.8)))
    for delta in (-0.06, 0.06):
        candidates.append(build(speed_kp=max(0.0, base.speed_kp + delta)))
    candidates.append(
        build(
            heading_gain=max(0.5, base.heading_gain + 0.10),
            stanley_gain=max(0.2, base.stanley_gain + 0.15),
            yaw_rate_kp=max(0.1, base.yaw_rate_kp + 0.10),
        )
    )
    return _unique_controller_grid(candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=TERRAINS, default="flat")
    parser.add_argument("--profile", choices=(*PROFILES, "all"), default="all")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--tune-steps", type=int, default=160)
    parser.add_argument("--sim-steps-per-control", type=int, default=1)
    parser.add_argument("--physics-dt", type=float, default=DEFAULT_PHYSICS_DT)
    parser.add_argument("--completion-stop-ratio", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--target-speed", type=float, default=0.25)
    parser.add_argument("--controller-mode", choices=("stanley_pid", "pure_pursuit"), default="stanley_pid")
    parser.add_argument("--lookahead-m", type=float, default=None)
    parser.add_argument("--k-heading", type=float, default=None)
    parser.add_argument("--k-cross-track", type=float, default=None)
    parser.add_argument("--angular-scale", type=float, default=None)
    parser.add_argument("--heading-gain", type=float, default=None)
    parser.add_argument("--stanley-gain", type=float, default=None)
    parser.add_argument("--curvature-feedforward-gain", type=float, default=None)
    parser.add_argument("--softening-speed", type=float, default=None)
    parser.add_argument("--speed-kp", type=float, default=None)
    parser.add_argument("--speed-ki", type=float, default=None)
    parser.add_argument("--speed-kd", type=float, default=None)
    parser.add_argument("--yaw-rate-kp", type=float, default=None)
    parser.add_argument("--yaw-rate-ki", type=float, default=None)
    parser.add_argument("--yaw-rate-kd", type=float, default=None)
    parser.add_argument("--velocity-filter-tau", type=float, default=None)
    parser.add_argument("--max-linear-servo-correction", type=float, default=None)
    parser.add_argument("--max-angular-servo-correction", type=float, default=None)
    parser.add_argument("--max-linear-accel", type=float, default=None)
    parser.add_argument("--max-angular-accel", type=float, default=None)
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
    parser.add_argument("--tune-top-k", type=int, default=2)
    parser.add_argument("--step-test", action="store_true")
    parser.add_argument("--step-duration-s", type=float, default=16.0)
    parser.add_argument("--step-settle-s", type=float, default=2.0)
    parser.add_argument("--step-linear", type=float, default=0.25)
    parser.add_argument("--step-angular", type=float, default=0.60)
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


def _reset_robot_root(robot, position: np.ndarray, yaw: float) -> None:
    position = np.asarray(position, dtype=np.float32)
    orientation = yaw_to_quat_wxyz(float(yaw))
    robot.apply_wheel_actions(np.zeros(len(JACKAL_WHEEL_DOF_NAMES), dtype=np.float64))
    robot.set_world_poses(positions=position, orientations=orientation)
    try:
        robot.set_velocities(
            linear_velocities=np.zeros(3, dtype=np.float32),
            angular_velocities=np.zeros(3, dtype=np.float32),
        )
    except Exception:
        pass


def _make_controller_cfg(args: argparse.Namespace, tuned: TrackingControllerCfg | None = None) -> TrackingControllerCfg:
    base = tuned or TrackingControllerCfg(target_speed_mps=args.target_speed)
    return TrackingControllerCfg(
        mode=str(args.controller_mode if tuned is None else base.mode),
        lookahead_m=float(args.lookahead_m if args.lookahead_m is not None else base.lookahead_m),
        k_heading=float(args.k_heading if args.k_heading is not None else base.k_heading),
        k_cross_track=float(args.k_cross_track if args.k_cross_track is not None else base.k_cross_track),
        angular_scale=float(args.angular_scale if args.angular_scale is not None else base.angular_scale),
        heading_gain=float(
            args.heading_gain
            if args.heading_gain is not None
            else (args.k_heading if args.k_heading is not None and base.mode == "stanley_pid" else base.heading_gain)
        ),
        stanley_gain=float(args.stanley_gain if args.stanley_gain is not None else base.stanley_gain),
        curvature_feedforward_gain=float(
            args.curvature_feedforward_gain
            if args.curvature_feedforward_gain is not None
            else base.curvature_feedforward_gain
        ),
        softening_speed_mps=float(
            args.softening_speed if args.softening_speed is not None else base.softening_speed_mps
        ),
        speed_kp=float(args.speed_kp if args.speed_kp is not None else base.speed_kp),
        speed_ki=float(args.speed_ki if args.speed_ki is not None else base.speed_ki),
        speed_kd=float(args.speed_kd if args.speed_kd is not None else base.speed_kd),
        yaw_rate_kp=float(args.yaw_rate_kp if args.yaw_rate_kp is not None else base.yaw_rate_kp),
        yaw_rate_ki=float(args.yaw_rate_ki if args.yaw_rate_ki is not None else base.yaw_rate_ki),
        yaw_rate_kd=float(args.yaw_rate_kd if args.yaw_rate_kd is not None else base.yaw_rate_kd),
        velocity_filter_tau_s=float(
            args.velocity_filter_tau if args.velocity_filter_tau is not None else base.velocity_filter_tau_s
        ),
        max_linear_servo_correction_mps=float(
            args.max_linear_servo_correction
            if args.max_linear_servo_correction is not None
            else base.max_linear_servo_correction_mps
        ),
        max_angular_servo_correction_radps=float(
            args.max_angular_servo_correction
            if args.max_angular_servo_correction is not None
            else base.max_angular_servo_correction_radps
        ),
        max_linear_accel_mps2=float(
            args.max_linear_accel if args.max_linear_accel is not None else base.max_linear_accel_mps2
        ),
        max_angular_accel_radps2=float(
            args.max_angular_accel if args.max_angular_accel is not None else base.max_angular_accel_radps2
        ),
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
    mode = str(
        values.get(
            "mode",
            "stanley_pid" if "heading_gain" in values or "stanley_gain" in values else "pure_pursuit",
        )
    )
    return TrackingControllerCfg(
        mode=mode,
        lookahead_m=float(values.get("lookahead_m", 0.35)),
        k_heading=float(values.get("k_heading", 1.6)),
        k_cross_track=float(values.get("k_cross_track", 0.75)),
        angular_scale=float(values.get("angular_scale", 1.05)),
        heading_gain=float(values.get("heading_gain", values.get("k_heading", 1.0))),
        stanley_gain=float(values.get("stanley_gain", values.get("k_cross_track", 1.8))),
        curvature_feedforward_gain=float(values.get("curvature_feedforward_gain", 0.0)),
        softening_speed_mps=float(values.get("softening_speed_mps", 0.12)),
        speed_kp=float(values.get("speed_kp", 0.20)),
        speed_ki=float(values.get("speed_ki", 0.02)),
        speed_kd=float(values.get("speed_kd", 0.0)),
        yaw_rate_kp=float(values.get("yaw_rate_kp", 0.50)),
        yaw_rate_ki=float(values.get("yaw_rate_ki", 0.02)),
        yaw_rate_kd=float(values.get("yaw_rate_kd", 0.0)),
        velocity_filter_tau_s=float(values.get("velocity_filter_tau_s", 0.08)),
        max_linear_servo_correction_mps=float(values.get("max_linear_servo_correction_mps", 0.35)),
        max_angular_servo_correction_radps=float(values.get("max_angular_servo_correction_radps", 1.0)),
        max_linear_accel_mps2=float(values.get("max_linear_accel_mps2", 1.2)),
        max_angular_accel_radps2=float(values.get("max_angular_accel_radps2", 4.5)),
        target_speed_mps=float(values.get("target_speed_mps", target_speed_mps)),
        max_linear_mps=float(values.get("max_linear_mps", JACKAL_MAX_LINEAR_SPEED)),
        max_angular_radps=float(values.get("max_angular_radps", JACKAL_MAX_ANGULAR_SPEED)),
    )


def _score_tracking_result(result: dict) -> float:
    thresholds = result.get("acceptance", {}).get("thresholds", {})
    rmse_limit = max(float(thresholds.get("rmse_cross_track_m", 0.08)), 1.0e-6)
    max_limit = max(float(thresholds.get("max_cross_track_m", 0.18)), 1.0e-6)
    completion_limit = float(thresholds.get("path_completion_ratio", 0.99))
    rmse = float(result.get("rmse_cross_track_m", math.inf))
    max_error = float(result.get("max_cross_track_m", math.inf))
    completion = float(result.get("path_completion_ratio", 0.0))
    strict_penalty = (
        4.0 * max(0.0, rmse - rmse_limit) / rmse_limit
        + 2.0 * max(0.0, max_error - max_limit) / max_limit
        + 8.0 * max(0.0, completion_limit - completion)
    )
    return (
        rmse / rmse_limit
        + 0.50 * max_error / max_limit
        + 2.0 * max(0.0, completion_limit - completion)
        + strict_penalty
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
        "desired_speed_mps",
        "reference_linear_mps",
        "reference_angular_radps",
        "measured_speed_mps",
        "measured_yaw_rate_radps",
        "linear_speed_error_mps",
        "yaw_rate_error_radps",
        "linear_servo_correction_mps",
        "angular_servo_correction_radps",
        "stanley_correction_rad",
        "reference_curvature_radpm",
        "controller_mode",
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


def _write_step_timeseries(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "time_s",
        "phase",
        "x_m",
        "y_m",
        "z_m",
        "yaw_rad",
        "tilt_deg",
        "reference_linear_mps",
        "reference_angular_radps",
        "command_linear_mps",
        "command_angular_radps",
        "measured_speed_mps",
        "measured_yaw_rate_radps",
        "linear_speed_error_mps",
        "yaw_rate_error_radps",
        "linear_servo_correction_mps",
        "angular_servo_correction_radps",
        "measurement_valid",
        "left_wheel_radps",
        "right_wheel_radps",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field) for field in fields})


def _plot_step_response(records: list[dict], output_path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib unavailable: {exc}"

    if not records:
        return "no step response records"
    time_s = np.array([row["time_s"] for row in records], dtype=np.float64)
    ref_v = np.array([row["reference_linear_mps"] for row in records], dtype=np.float64)
    cmd_v = np.array([row["command_linear_mps"] for row in records], dtype=np.float64)
    meas_v = np.array([row["measured_speed_mps"] for row in records], dtype=np.float64)
    ref_w = np.array([row["reference_angular_radps"] for row in records], dtype=np.float64)
    cmd_w = np.array([row["command_angular_radps"] for row in records], dtype=np.float64)
    meas_w = np.array([row["measured_yaw_rate_radps"] for row in records], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True)
    axes[0].step(time_s, ref_v, where="post", label="reference", linewidth=1.5)
    axes[0].plot(time_s, cmd_v, label="servo command", linewidth=1.2)
    axes[0].plot(time_s, meas_v, label="measured", linewidth=1.4)
    axes[0].set_ylabel("linear speed [m/s]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].step(time_s, ref_w, where="post", label="reference", linewidth=1.5)
    axes[1].plot(time_s, cmd_w, label="servo command", linewidth=1.2)
    axes[1].plot(time_s, meas_w, label="measured", linewidth=1.4)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("yaw rate [rad/s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return None


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
    physics_dt: float,
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
    build_physics_scene(stage, physics_dt=physics_dt)
    add_tracking_terrain(stage, terrain)

    path = generate_reference_path(profile)
    path_offset_xy = (0.0, 0.0)
    if terrain == "strong_lunar_crater":
        path_offset_xy = STRONG_LUNAR_CRATER_PATH_OFFSETS[profile]
        path = offset_reference_path(path, path_offset_xy)
    start_xy = path.points_xy[0]
    start_yaw = float(path.yaws[0])
    start_z = terrain_height(terrain, float(start_xy[0]), float(start_xy[1])) + JACKAL_ROOT_Z_OFFSET
    start_position = np.array([start_xy[0], start_xy[1], start_z], dtype=np.float32)
    robot = WheeledRobot(
        paths=JACKAL_PRIM_PATH,
        wheel_dof_names=JACKAL_WHEEL_DOF_NAMES,
        usd_path=jackal_usd,
        positions=start_position,
        orientations=yaw_to_quat_wxyz(start_yaw),
    )

    if render:
        set_camera()
    for _ in range(60):
        app.update()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(8):
        app.update()
    _reset_robot_root(robot, start_position, start_yaw)

    records: list[dict] = []
    progress_index = 0
    control_dt = max(1, int(sim_steps_per_control)) * float(physics_dt)
    controller_state = TrackingControllerState()
    wall_start = time.perf_counter()
    for step_id in range(max(1, int(steps))):
        pos, quat = _pose(robot)
        yaw = quat_wxyz_to_yaw(quat)
        command, details = compute_tracking_command(
            path,
            pos[:2],
            yaw,
            controller_cfg,
            progress_index=progress_index,
            controller_state=controller_state,
            dt=control_dt,
        )
        progress_index = max(progress_index, int(details["nearest_index"]))
        wheel_velocities = wheel_controller.forward(command)
        robot.apply_wheel_actions(wheel_velocities)
        for _ in range(max(1, int(sim_steps_per_control))):
            app.update()
        next_pos, next_quat = _pose(robot)
        next_yaw = quat_wxyz_to_yaw(next_quat)
        next_nearest_idx = nearest_path_index(
            path,
            next_pos[:2],
            start_index=progress_index,
            end_index=progress_index + 80,  # matches search_window default in compute_tracking_command
        )
        next_error = tracking_error(
            path,
            next_pos[:2],
            next_yaw,
            nearest_idx=next_nearest_idx,
        )
        progress_index = max(progress_index, int(next_error["nearest_index"]))
        ref_xy = path.points_xy[int(next_error["nearest_index"])]
        records.append(
            {
                "step": step_id,
                "time_s": (step_id + 1) * control_dt,
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
                "desired_speed_mps": float(details.get("desired_speed_mps", 0.0)),
                "reference_linear_mps": float(details.get("reference_linear_mps", 0.0)),
                "reference_angular_radps": float(details.get("reference_angular_radps", 0.0)),
                "measured_speed_mps": float(details.get("measured_speed_mps", 0.0)),
                "measured_yaw_rate_radps": float(details.get("measured_yaw_rate_radps", 0.0)),
                "linear_speed_error_mps": float(details.get("linear_speed_error_mps", 0.0)),
                "yaw_rate_error_radps": float(details.get("yaw_rate_error_radps", 0.0)),
                "linear_servo_correction_mps": float(details.get("linear_servo_correction_mps", 0.0)),
                "angular_servo_correction_radps": float(details.get("angular_servo_correction_radps", 0.0)),
                "stanley_correction_rad": float(details.get("stanley_correction_rad", 0.0)),
                "reference_curvature_radpm": float(details.get("reference_curvature_radpm", 0.0)),
                "controller_mode": str(details.get("controller_mode", controller_cfg.mode)),
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
    metrics["physics_dt_s"] = float(physics_dt)
    metrics["control_dt_s"] = float(control_dt)
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


def _step_reference(
    time_s: float,
    *,
    duration_s: float,
    linear_mps: float,
    angular_radps: float,
) -> tuple[str, np.ndarray]:
    warmup_end = min(1.0, 0.20 * duration_s)
    angular_start = max(warmup_end, 0.45 * duration_s)
    stop_start = max(angular_start, 0.75 * duration_s)
    if time_s < warmup_end:
        return "hold_zero", np.array([0.0, 0.0], dtype=np.float64)
    if time_s < angular_start:
        return "linear_step", np.array([linear_mps, 0.0], dtype=np.float64)
    if time_s < stop_start:
        return "yaw_rate_step", np.array([linear_mps, angular_radps], dtype=np.float64)
    return "stop_step", np.array([0.0, 0.0], dtype=np.float64)


def _summarize_step_response(
    *,
    records: list[dict],
    physics_dt: float,
    sim_steps_per_control: int,
    settle_s: float,
    controller_cfg: TrackingControllerCfg,
    wall_time_s: float,
) -> dict:
    valid = [row for row in records if row.get("measurement_valid")]
    linear_error = np.array([row["linear_speed_error_mps"] for row in valid], dtype=np.float64)
    yaw_error = np.array([row["yaw_rate_error_radps"] for row in valid], dtype=np.float64)
    measured_speed = np.array([row["measured_speed_mps"] for row in valid], dtype=np.float64)
    measured_yaw_rate = np.array([row["measured_yaw_rate_radps"] for row in valid], dtype=np.float64)
    control_dt = max(1, int(sim_steps_per_control)) * float(physics_dt)
    return {
        "status": "ok",
        "backend": "physx_jackal",
        "test": "step_response",
        "physics_dt_s": float(physics_dt),
        "sim_steps_per_control": int(sim_steps_per_control),
        "control_dt_s": float(control_dt),
        "steps": len(records),
        "duration_s": float(records[-1]["time_s"]) if records else 0.0,
        "settle_s": float(settle_s),
        "wall_time_s": float(wall_time_s),
        "control_steps_per_s": len(records) / wall_time_s if wall_time_s > 0.0 else math.inf,
        "linear_speed_rmse_mps": float(np.sqrt(np.mean(linear_error**2))) if len(linear_error) else math.inf,
        "yaw_rate_rmse_radps": float(np.sqrt(np.mean(yaw_error**2))) if len(yaw_error) else math.inf,
        "max_abs_linear_speed_error_mps": float(np.max(np.abs(linear_error))) if len(linear_error) else math.inf,
        "max_abs_yaw_rate_error_radps": float(np.max(np.abs(yaw_error))) if len(yaw_error) else math.inf,
        "max_measured_speed_mps": float(np.max(measured_speed)) if len(measured_speed) else 0.0,
        "max_abs_measured_yaw_rate_radps": float(np.max(np.abs(measured_yaw_rate))) if len(measured_yaw_rate) else 0.0,
        "final_x_m": float(records[-1]["x_m"]) if records else 0.0,
        "final_y_m": float(records[-1]["y_m"]) if records else 0.0,
        "final_yaw_rad": float(records[-1]["yaw_rad"]) if records else 0.0,
        "controller": asdict(controller_cfg),
    }


def _run_step_response_test(
    app,
    *,
    jackal_usd: str,
    physics_dt: float,
    sim_steps_per_control: int,
    duration_s: float,
    settle_s: float,
    step_linear_mps: float,
    step_angular_radps: float,
    controller_cfg: TrackingControllerCfg,
    wheel_controller: JackalSkidSteerController,
    run_dir: Path,
    render: bool,
) -> dict:
    import omni.timeline
    import omni.usd
    from isaacsim.robot.experimental.wheeled_robots.robots import WheeledRobot

    app.run_coroutine(omni.usd.get_context().new_stage_async())
    stage = omni.usd.get_context().get_stage()
    build_physics_scene(stage, physics_dt=physics_dt)
    add_tracking_terrain(stage, "flat")

    robot = WheeledRobot(
        paths=JACKAL_PRIM_PATH,
        wheel_dof_names=JACKAL_WHEEL_DOF_NAMES,
        usd_path=jackal_usd,
        positions=np.array([0.0, 0.0, JACKAL_ROOT_Z_OFFSET], dtype=np.float32),
        orientations=yaw_to_quat_wxyz(0.0),
    )
    if render:
        set_camera()
    for _ in range(60):
        app.update()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    robot.apply_wheel_actions(wheel_controller.forward([0.0, 0.0]))
    for _ in range(max(1, int(math.ceil(max(0.0, float(settle_s)) / float(physics_dt))))):
        app.update()
    _reset_robot_root(robot, np.array([0.0, 0.0, JACKAL_ROOT_Z_OFFSET], dtype=np.float32), 0.0)

    control_dt = max(1, int(sim_steps_per_control)) * float(physics_dt)
    step_count = max(1, int(math.ceil(float(duration_s) / control_dt)))
    state = TrackingControllerState()
    records: list[dict] = []
    wall_start = time.perf_counter()
    for step_id in range(step_count):
        time_s = step_id * control_dt
        phase, reference = _step_reference(
            time_s,
            duration_s=float(duration_s),
            linear_mps=float(step_linear_mps),
            angular_radps=float(step_angular_radps),
        )
        pos, quat = _pose(robot)
        yaw = quat_wxyz_to_yaw(quat)
        command, details = compute_chassis_servo_command(
            reference,
            pos[:2],
            yaw,
            controller_cfg,
            controller_state=state,
            dt=control_dt,
        )
        wheel_velocities = wheel_controller.forward(command)
        records.append(
            {
                "step": step_id,
                "time_s": float(time_s),
                "phase": phase,
                "x_m": float(pos[0]),
                "y_m": float(pos[1]),
                "z_m": float(pos[2]),
                "yaw_rad": float(yaw),
                "tilt_deg": float(quat_wxyz_to_tilt_deg(quat)),
                "reference_linear_mps": float(details["reference_linear_mps"]),
                "reference_angular_radps": float(details["reference_angular_radps"]),
                "command_linear_mps": float(details["command_linear_mps"]),
                "command_angular_radps": float(details["command_angular_radps"]),
                "measured_speed_mps": float(details["measured_speed_mps"]),
                "measured_yaw_rate_radps": float(details["measured_yaw_rate_radps"]),
                "linear_speed_error_mps": float(details["linear_speed_error_mps"]),
                "yaw_rate_error_radps": float(details["yaw_rate_error_radps"]),
                "linear_servo_correction_mps": float(details["linear_servo_correction_mps"]),
                "angular_servo_correction_radps": float(details["angular_servo_correction_radps"]),
                "measurement_valid": bool(details["measurement_valid"]),
                "left_wheel_radps": float(wheel_velocities[0]),
                "right_wheel_radps": float(wheel_velocities[1]),
            }
        )
        robot.apply_wheel_actions(wheel_velocities)
        for _ in range(max(1, int(sim_steps_per_control))):
            app.update()

    wall_time_s = time.perf_counter() - wall_start
    timeline.stop()

    output_dir = run_dir / "physx"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    csv_path = metrics_dir / "step_response_timeseries.csv"
    figure_path = figures_dir / "step_response.png"
    _write_step_timeseries(csv_path, records)
    plot_error = _plot_step_response(records, figure_path)
    summary = _summarize_step_response(
        records=records,
        physics_dt=physics_dt,
        sim_steps_per_control=sim_steps_per_control,
        settle_s=settle_s,
        controller_cfg=controller_cfg,
        wall_time_s=wall_time_s,
    )
    summary["step_command"] = {
        "linear_mps": float(step_linear_mps),
        "angular_radps": float(step_angular_radps),
    }
    summary["artifacts"] = {
        "step_response_csv": str(csv_path),
        "step_response_figure": str(figure_path) if plot_error is None else None,
        "step_response_figure_error": plot_error,
    }
    output_path = metrics_dir / "step_response_summary.json"
    _write_json(output_path, summary)
    summary["artifact"] = str(output_path)
    return summary


def _run_asset_smoke(app, *, jackal_usd: str, run_dir: Path, physics_dt: float) -> dict:
    import omni.usd
    from isaacsim.robot.experimental.wheeled_robots.robots import WheeledRobot

    app.run_coroutine(omni.usd.get_context().new_stage_async())
    stage = omni.usd.get_context().get_stage()
    build_physics_scene(stage, physics_dt=physics_dt)
    add_tracking_terrain(stage, "flat")
    robot = WheeledRobot(
        paths=JACKAL_PRIM_PATH,
        wheel_dof_names=JACKAL_WHEEL_DOF_NAMES,
        usd_path=jackal_usd,
        positions=np.array([0.0, 0.0, JACKAL_ROOT_Z_OFFSET], dtype=np.float32),
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
        "physics_dt_s": float(physics_dt),
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
    rows = []
    best_cfg = TrackingControllerCfg(mode=args.controller_mode, target_speed_mps=args.target_speed)
    best_score = math.inf
    rounds: list[dict] = []

    def evaluate_candidates(phase: str, candidates: list[TrackingControllerCfg], start_index: int) -> int:
        nonlocal best_cfg, best_score
        phase_rows = []
        for local_index, cfg in enumerate(candidates):
            profile_results = []
            for profile in PROFILES:
                result = _run_tracking_profile(
                    app,
                    jackal_usd=jackal_usd,
                    terrain="flat",
                    profile=profile,
                    steps=args.tune_steps,
                    sim_steps_per_control=args.sim_steps_per_control,
                    physics_dt=args.physics_dt,
                    controller_cfg=cfg,
                    wheel_controller=wheel_controller,
                    output_dir=None,
                    render=False,
                    completion_stop_ratio=args.completion_stop_ratio,
                )
                profile_results.append(result)
            score = float(np.mean([_score_tracking_result(item) for item in profile_results]))
            row = {
                "candidate": start_index + local_index,
                "phase": phase,
                "score": score,
                "passed": all(bool(item.get("passed")) for item in profile_results),
                "controller": asdict(cfg),
                "profile_results": profile_results,
            }
            rows.append(row)
            phase_rows.append(row)
            if score < best_score:
                best_score = score
                best_cfg = cfg
        rounds.append(
            {
                "phase": phase,
                "candidate_count": len(candidates),
                "best_candidate": min(phase_rows, key=lambda item: item["score"])["candidate"] if phase_rows else None,
            }
        )
        return start_index + len(candidates)

    if args.controller_mode == "pure_pursuit":
        evaluate_candidates("pure_pursuit", _controller_grid(args.target_speed), 0)
    else:
        next_index = evaluate_candidates("coarse", _stanley_coarse_grid(args.target_speed), 0)
        top_k = max(1, int(args.tune_top_k))
        top_rows = sorted(rows, key=lambda item: item["score"])[:top_k]
        refined = _unique_controller_grid(
            candidate
            for row in top_rows
            for candidate in _stanley_refined_grid(
                TrackingControllerCfg(**row["controller"]),
                args.target_speed,
            )
        )
        evaluate_candidates("refined", refined, next_index)

    selected_row = min(rows, key=lambda item: item["score"]) if rows else None
    selected_passed = bool(selected_row.get("passed")) if selected_row else False
    payload = {
        "status": "ok",
        "backend": "physx_jackal",
        "terrain": "flat",
        "controller_mode": args.controller_mode,
        "selected_score": best_score,
        "selected_passed": selected_passed,
        "strict_thresholds": rows[0]["profile_results"][0]["acceptance"]["thresholds"] if rows else {},
        "selected_controller": asdict(best_cfg),
        "rounds": rounds,
        "candidate_count": len(rows),
        "grid": rows,
        "next_step_recommendation": (
            None
            if selected_passed
            else "Best flat candidate did not satisfy the strict gate; keep the strict thresholds and inspect per-profile failures before widening the search or moving to strong-terrain control."
        ),
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
    if args.physics_dt <= 0.0:
        raise ValueError("--physics-dt must be positive")
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
            summary = _run_asset_smoke(app, jackal_usd=jackal_usd, run_dir=run_dir, physics_dt=args.physics_dt)
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
        if args.step_test:
            summary = _run_step_response_test(
                app,
                jackal_usd=jackal_usd,
                physics_dt=args.physics_dt,
                sim_steps_per_control=args.sim_steps_per_control,
                duration_s=args.step_duration_s,
                settle_s=args.step_settle_s,
                step_linear_mps=args.step_linear,
                step_angular_radps=args.step_angular,
                controller_cfg=controller_cfg,
                wheel_controller=wheel_controller,
                run_dir=run_dir,
                render=args.render,
            )
            _write_manifest(run_dir, summary, command=["scripts/evaluate_physx_jackal_tracking.py"])
            print(json.dumps(summary, indent=2, sort_keys=True))
            return

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
                physics_dt=args.physics_dt,
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
            "physics_dt_s": float(args.physics_dt),
            "control_dt_s": float(max(1, int(args.sim_steps_per_control)) * float(args.physics_dt)),
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
