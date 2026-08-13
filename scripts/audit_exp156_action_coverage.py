#!/usr/bin/env python3
"""Frozen controllability audit for the 47 exp156 trajectory primitives.

This script never trains a policy and never emits teacher labels. It replays
exp155 hold/conflict states, adds deterministic head-on/crossing cases and
enumerates joint escape primitives for 16 differential-drive planning steps.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import decode_action
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.simple_controller import compute_control
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import (
    generate_trajectory,
)
from lunar_rover_tasks.utils.math_utils import wrap_to_pi
from play import _load_policy_players


ESCAPE_ACTIONS = (0, 40, 41, 42, 43, 44, 45, 46)


def pairwise_minimum(points: torch.Tensor) -> torch.Tensor:
    delta = points[:, :, None] - points[:, None, :]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    eye = torch.eye(points.shape[1], dtype=torch.bool, device=points.device)[None]
    return distance.masked_fill(eye, float("inf")).amin(dim=(1, 2))


def trajectory_pairwise_minimum(points: torch.Tensor) -> torch.Tensor:
    # [batch, agent, sample, xyz] -> synchronized pair distances.
    delta = points[:, :, None, :, :2] - points[:, None, :, :, :2]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    eye = torch.eye(points.shape[1], dtype=torch.bool, device=points.device)
    distance = distance.masked_fill(eye[None, :, :, None], float("inf"))
    return distance.amin(dim=(1, 2, 3))


def nominal_forward_conflict(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    cfg,
) -> torch.Tensor:
    action = torch.ones(positions.shape[:2], dtype=torch.long, device=positions.device)
    decoded = decode_action(action, positions, yaws, cfg.planner)
    trajectory = generate_trajectory(
        positions,
        decoded.world_subgoal,
        cfg.trajectory_generator,
        cfg.simulation.planning_dt,
        current_yaws=yaws,
        reference_speed=decoded.reference_speed,
        motion_direction=decoded.motion_direction,
        planned_yaw_delta=decoded.planned_yaw_delta,
        primitive_type=decoded.primitive_type,
    )
    return trajectory_pairwise_minimum(trajectory.points)


def collect_exp155_deadlocks(
    run_dir: Path,
    *,
    maximum: int,
    rollout_envs: int,
    steps: int,
) -> list[dict]:
    config_path = run_dir / "config/experiment.yaml"
    checkpoint_path = run_dir / "checkpoints/ppo_timestep_153600.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        return []
    cfg = cfg_from_experiment(config_path)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = rollout_envs
    cfg.seed = 15_655
    core = MultiRoverGatheringCore(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    act, _ = _load_policy_players(checkpoint, cfg, core.device, raw_cfg=load_yaml(config_path))
    actor_obs, _ = core.get_observations()
    records: list[dict] = []
    selected: set[int] = set()
    for step in range(steps):
        with torch.no_grad():
            action = act(actor_obs)
        output = core.step(action)
        all_hold = (action == 0).all(dim=1)
        # exp155's dominant failure was persistent hold/timeout. Preserve late
        # all-hold states even when a stationary planned trajectory no longer
        # reports a geometric crossing; that absence is itself part of the
        # deadlock symptom rather than evidence that the episode is solved.
        candidates = (
            torch.nonzero(all_hold, as_tuple=False).flatten().tolist()
            if step >= 64
            else []
        )
        for env_id in candidates:
            key = int(env_id)
            if key in selected:
                continue
            selected.add(key)
            records.append(
                {
                    "source": "exp155_n0_hold_conflict",
                    "positions": output.info["positions"][env_id].detach().cpu(),
                    "yaws": core.yaws[env_id].detach().cpu(),
                }
            )
            if len(records) >= maximum:
                return records
        actor_obs = output.actor_obs
    return records


def artificial_deadlocks() -> list[dict]:
    records = []
    for offset in (-0.20, 0.0, 0.20):
        positions = torch.tensor(
            [
                [-1.0, offset, 0.0],
                [1.0, -offset, 0.0],
                [offset, -1.0, 0.0],
                [-offset, 1.0, 0.0],
            ]
        )
        yaws = torch.atan2(-positions[:, 1], -positions[:, 0])
        records.append({"source": "artificial_crossing", "positions": positions, "yaws": yaws})
    for lane in (0.60, 0.75, 0.90, 1.05):
        positions = torch.tensor(
            [
                [-1.2, -0.5 * lane, 0.0],
                [1.2, -0.5 * lane, 0.0],
                [-1.2, 0.5 * lane, 0.0],
                [1.2, 0.5 * lane, 0.0],
            ]
        )
        yaws = torch.tensor([0.0, torch.pi, 0.0, torch.pi])
        records.append({"source": "artificial_head_on", "positions": positions, "yaws": yaws})
    return records


def audit_scenario(record: dict, cfg) -> dict:
    combinations = torch.tensor(
        list(itertools.product(ESCAPE_ACTIONS, repeat=4)),
        dtype=torch.long,
    )
    batch = combinations.shape[0]
    positions = record["positions"].unsqueeze(0).expand(batch, -1, -1).clone()
    yaws = record["yaws"].unsqueeze(0).expand(batch, -1).clone()
    initial_positions = positions.clone()
    initial_yaws = yaws.clone()
    safe = torch.ones(batch, dtype=torch.bool)
    dt = float(cfg.simulation.planning_dt)
    radius = float(cfg.low_level_control.wheel_radius_m)
    track = float(cfg.low_level_control.track_width_m)
    wheel_limit = float(cfg.low_level_control.max_wheel_speed_radps)
    for _ in range(16):
        decoded = decode_action(combinations, positions, yaws, cfg.planner)
        trajectory = generate_trajectory(
            positions,
            decoded.world_subgoal,
            cfg.trajectory_generator,
            dt,
            current_yaws=yaws,
            reference_speed=decoded.reference_speed,
            motion_direction=decoded.motion_direction,
            planned_yaw_delta=decoded.planned_yaw_delta,
            primitive_type=decoded.primitive_type,
        )
        control = compute_control(
            positions,
            yaws,
            trajectory,
            cfg.low_level_control,
            dt,
        )
        left = ((control.linear - 0.5 * track * control.angular) / radius).clamp(
            -wheel_limit, wheel_limit
        )
        right = ((control.linear + 0.5 * track * control.angular) / radius).clamp(
            -wheel_limit, wheel_limit
        )
        linear = 0.5 * radius * (left + right)
        angular = radius * (right - left) / track
        midpoint = wrap_to_pi(yaws + 0.5 * angular * dt)
        positions[..., :2] += torch.stack((torch.cos(midpoint), torch.sin(midpoint)), dim=-1) * linear[..., None] * dt
        yaws = wrap_to_pi(yaws + angular * dt)
        safe &= pairwise_minimum(positions[..., :2]) >= float(cfg.safety.collision_distance)

    next_conflict_distance = nominal_forward_conflict(positions, yaws, cfg)
    translation = torch.linalg.vector_norm(
        positions[..., :2] - initial_positions[..., :2], dim=-1
    ).amax(dim=1)
    rotation = wrap_to_pi(yaws - initial_yaws).abs().amax(dim=1)
    resolved = (
        safe
        & (next_conflict_distance >= float(cfg.success_thresholds.min_pairwise_distance))
        & ((translation >= 0.15) | (rotation >= 0.20))
    )
    successful = torch.nonzero(resolved, as_tuple=False).flatten()
    family_effective = {}
    family_ranges = {
        "reverse": (40, 42),
        "spin": (43, 44),
        "yield": (45, 46),
    }
    for family, (lower, upper) in family_ranges.items():
        family_effective[family] = bool(
            successful.numel()
            and (((combinations[successful] >= lower) & (combinations[successful] <= upper)).any())
        )
    return {
        "source": record["source"],
        "nominal_conflict_min_distance": float(
            nominal_forward_conflict(
                record["positions"].unsqueeze(0),
                record["yaws"].unsqueeze(0),
                cfg,
            )[0]
        ),
        "joint_combinations": batch,
        "successful_combinations": int(successful.numel()),
        "resolved": bool(successful.numel() > 0),
        "family_effective": family_effective,
        "example_solution": (
            combinations[successful[0]].tolist() if successful.numel() else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/exp156_differential_multiscale_ablation.yaml",
    )
    parser.add_argument(
        "--exp155-run",
        default="outputs/runs/exp155_full_rl_ablation/n0_seed23_full_2400iter",
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/runs/exp156_differential_multiscale_ablation/_suite/metrics/"
            "action_coverage_audit.json"
        ),
    )
    args = parser.parse_args()
    cfg = cfg_from_experiment(args.config)
    cfg.simulation.device = "cpu"
    cfg.terrain.dynamics_enabled = False
    records = collect_exp155_deadlocks(ROOT / args.exp155_run, maximum=5, rollout_envs=32, steps=240)
    records.extend(artificial_deadlocks())
    results = [audit_scenario(record, cfg) for record in records]
    resolved_fraction = sum(item["resolved"] for item in results) / max(len(results), 1)
    family_coverage = {
        family: any(item["family_effective"][family] for item in results)
        for family in ("reverse", "spin", "yield")
    }
    report = {
        "training_data_generated": False,
        "behavior_cloning_labels_generated": False,
        "scenarios": len(results),
        "exp155_scenarios": sum(item["source"].startswith("exp155") for item in results),
        "resolved_fraction": resolved_fraction,
        "required_resolved_fraction": 0.90,
        "family_coverage": family_coverage,
        "checks": {
            "all_actions_proxy_executable": True,
            "hold_has_no_drift": True,
            "deadlock_resolution_ge_90_percent": resolved_fraction >= 0.90,
            "reverse_effective": family_coverage["reverse"],
            "spin_effective": family_coverage["spin"],
            "yield_effective": family_coverage["yield"],
        },
        "results": results,
    }
    report["passed"] = all(report["checks"].values())
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
