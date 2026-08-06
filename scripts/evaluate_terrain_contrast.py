#!/usr/bin/env python
"""Measure whether a decentralized policy uses terrain observations.

The counterfactual action is evaluated at exactly the same rover state and
communication snapshot as the normal action. Only the 50 terrain features are
zeroed. The counterfactual is never executed, so this diagnostic cannot alter
the policy rollout or communication cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import (
    decode_action,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    sample_trajectory_terrain_risk,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import (
    generate_trajectory,
)
from play import _load_policy_players


TERRAIN_SLICE = slice(46, 96)


def evaluate_terrain_contrast(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    num_envs: int = 512,
    steps: int = 120,
    seed: int = 12023,
    initial_state_progress: int | None = None,
    output: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict:
    raw_cfg = load_yaml(config)
    cfg = cfg_from_experiment(config)
    cfg.simulation.device = device
    cfg.simulation.num_envs = num_envs
    cfg.seed = seed
    if cfg.observation.schema_version != "ego_v8_decentralized_tiered":
        raise ValueError("Terrain contrast requires ego_v8_decentralized_tiered.")

    checkpoint_data = torch.load(checkpoint, map_location=torch.device(device))
    metadata = checkpoint_data.get("metadata", {})
    if cfg.initial_state.curriculum_enabled:
        cfg.initial_state.progress_timestep_override = (
            int(metadata.get("timesteps", 0))
            if initial_state_progress is None
            else int(initial_state_progress)
        )

    env = MultiRoverGatheringCore(cfg)
    act, backend = _load_policy_players(
        checkpoint_data,
        cfg,
        env.device,
        raw_cfg=raw_cfg,
    )
    actor_obs, _ = env.get_observations()

    action_mse_sum = torch.zeros((), device=env.device)
    normal_risk_sum = torch.zeros((), device=env.device)
    zero_risk_sum = torch.zeros((), device=env.device)
    sample_count = 0
    communication_sums: dict[str, float] = {}
    conflict_sums: dict[str, float] = {}

    for _ in range(steps):
        zero_terrain_obs = actor_obs.clone()
        zero_terrain_obs[..., TERRAIN_SLICE] = 0.0
        with torch.no_grad():
            normal_action = act(actor_obs)
            zero_action = act(zero_terrain_obs)
            action_mse_sum += (normal_action - zero_action).square().mean()

            normal_subgoal = decode_action(
                normal_action,
                env.positions,
                env.yaws,
                env.cfg.planner,
            ).world_subgoal
            zero_subgoal = decode_action(
                zero_action,
                env.positions,
                env.yaws,
                env.cfg.planner,
            ).world_subgoal
            normal_trajectory = generate_trajectory(
                env.positions,
                normal_subgoal,
                env.cfg.trajectory_generator,
                env.cfg.simulation.planning_dt,
                current_yaws=env.yaws,
            )
            zero_trajectory = generate_trajectory(
                env.positions,
                zero_subgoal,
                env.cfg.trajectory_generator,
                env.cfg.simulation.planning_dt,
                current_yaws=env.yaws,
            )
            normal_risk = sample_trajectory_terrain_risk(
                normal_trajectory.points,
                env.cfg.terrain,
                env.terrain_runtime,
            )["risk_mean"]
            zero_risk = sample_trajectory_terrain_risk(
                zero_trajectory.points,
                env.cfg.terrain,
                env.terrain_runtime,
            )["risk_mean"]
            normal_risk_sum += normal_risk.mean()
            zero_risk_sum += zero_risk.mean()

        step_output = env.step(normal_action)
        actor_obs = step_output.actor_obs
        sample_count += 1

        for key, value in (step_output.info.get("communication") or {}).items():
            if isinstance(value, torch.Tensor):
                communication_sums[key] = communication_sums.get(key, 0.0) + float(
                    value.float().mean().cpu()
                )
        for key, value in (step_output.info.get("trajectory_conflicts") or {}).items():
            if isinstance(value, torch.Tensor) and value.ndim <= 1:
                conflict_sums[key] = conflict_sums.get(key, 0.0) + float(
                    value.float().mean().cpu()
                )

    divisor = max(sample_count, 1)
    action_mse = float((action_mse_sum / divisor).cpu())
    normal_path_risk = float((normal_risk_sum / divisor).cpu())
    zero_path_risk = float((zero_risk_sum / divisor).cpu())
    path_risk_reduction = (zero_path_risk - normal_path_risk) / max(
        abs(zero_path_risk),
        1.0e-8,
    )
    result = {
        "status": "ok",
        "backend": backend,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "device": str(env.device),
        "num_envs": num_envs,
        "steps": steps,
        "seed": seed,
        "initial_state_progress_timestep": int(
            cfg.initial_state.progress_timestep_override
        ),
        "action_mse_normal_vs_zero_terrain": action_mse,
        "normal_path_risk_mean": normal_path_risk,
        "zero_terrain_path_risk_mean": zero_path_risk,
        "path_risk_reduction_fraction": path_risk_reduction,
        "checks": {
            "action_mse_gt_0_02": action_mse > 0.02,
            "normal_path_risk_reduced_5pct": path_risk_reduction >= 0.05,
        },
        "communication": {
            key: value / divisor for key, value in communication_sums.items()
        },
        "mapf_conflicts": {
            key: value / divisor for key, value in conflict_sums.items()
        },
    }

    if output is None and run_dir is not None:
        output = Path(run_dir) / "metrics" / "terrain_contrast.json"
    if output is not None:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["artifact"] = str(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=12023)
    parser.add_argument("--initial-state-progress", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = evaluate_terrain_contrast(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        initial_state_progress=args.initial_state_progress,
        output=args.output,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
