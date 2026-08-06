#!/usr/bin/env python
"""Paired local action interventions on frozen exp125 planning states."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import decode_action
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from lunar_rover_tasks.tasks.multi_rover_gathering.mapf_diagnostics import (
    trajectory_pairwise_min_distance,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    sample_trajectory_terrain_risk,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import (
    generate_trajectory,
)
from play import _load_policy_players


EXPERIMENT_ID = "exp129_paired_action_interventions"


def _policy_digest(checkpoint: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(checkpoint["rover_0"]["policy"].items()):
        if isinstance(value, torch.Tensor):
            digest.update(key.encode())
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def planning_outcomes(
    env: MultiRoverGatheringCore,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-agent relative risk and trajectory nearest-neighbour margin."""

    decoded = decode_action(actions, env.positions, env.yaws, env.cfg.planner)
    trajectory = generate_trajectory(
        env.positions,
        decoded.world_subgoal,
        env.cfg.trajectory_generator,
        env.cfg.simulation.planning_dt,
        current_yaws=env.yaws,
    )
    risk = sample_trajectory_terrain_risk(
        trajectory.points,
        env.cfg.terrain,
        env.terrain_runtime,
    )["risk_mean"]
    straight_actions = actions.clone()
    straight_actions[..., 1] = 0.0
    straight_decoded = decode_action(
        straight_actions, env.positions, env.yaws, env.cfg.planner
    )
    straight_trajectory = generate_trajectory(
        env.positions,
        straight_decoded.world_subgoal,
        env.cfg.trajectory_generator,
        env.cfg.simulation.planning_dt,
        current_yaws=env.yaws,
    )
    reference_risk = sample_trajectory_terrain_risk(
        straight_trajectory.points,
        env.cfg.terrain,
        env.terrain_runtime,
    )["risk_mean"]
    relative_risk = risk - reference_risk
    pair_distance = trajectory_pairwise_min_distance(trajectory.points)
    eye = torch.eye(env.n_agents, dtype=torch.bool, device=env.device).unsqueeze(0)
    nearest_margin = pair_distance.masked_fill(eye, float("inf")).amin(dim=-1)
    return relative_risk, nearest_margin


def central_difference(
    minus: torch.Tensor,
    plus: torch.Tensor,
    minus_action: torch.Tensor,
    plus_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = plus_action - minus_action
    valid = denominator.abs() > 1.0e-6
    derivative = torch.where(valid, (plus - minus) / denominator.clamp_min(1.0e-6), 0.0)
    return derivative, valid


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten().cpu()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(torch.quantile(values, 0.90)),
    }


def analyze_paired_action_interventions(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    num_envs: int = 64,
    steps: int = 480,
    sample_interval: int = 8,
    action_delta: float = 0.15,
    seed: int = 17023,
    exploration_multiplier: float = 1.0,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    if sample_interval <= 0 or action_delta <= 0.0:
        raise ValueError("sample_interval and action_delta must be positive.")
    raw_cfg = load_yaml(config)
    cfg = cfg_from_experiment(config)
    cfg.simulation.device = device
    cfg.simulation.num_envs = num_envs
    cfg.seed = seed
    checkpoint_data = torch.load(checkpoint, map_location=torch.device(device))
    metadata = checkpoint_data.get("metadata") or {}
    if cfg.initial_state.curriculum_enabled:
        cfg.initial_state.progress_timestep_override = int(metadata.get("timesteps", 0))
    if cfg.planner.subgoal_filter.enabled:
        raise ValueError("Paired intervention requires the strict unfiltered B0 chain.")
    forbidden = (
        cfg.low_level_control.safety_projection_enabled,
        cfg.low_level_control.formation_center_correction_enabled,
        cfg.low_level_control.terminal_slot_capture_enabled,
        cfg.low_level_control.flat_geometry_capture_enabled,
    )
    if any(forbidden):
        raise ValueError("Paired intervention requires all execution overrides disabled.")

    env = MultiRoverGatheringCore(cfg)
    act, _ = _load_policy_players(checkpoint_data, cfg, env.device, raw_cfg=raw_cfg)
    actor_obs, _ = env.get_observations()
    probe_obs = actor_obs[: min(num_envs, 32)].clone()
    with torch.no_grad():
        probe_before = act(probe_obs).clone()
    digest_before = _policy_digest(checkpoint_data)
    log_std = checkpoint_data["rover_0"]["policy"]["log_std_parameter"]
    policy_std = log_std.detach().to(env.device).exp().view(1, 1, 2)
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed + 7919)

    risk_gradients: list[torch.Tensor] = []
    margin_gradients: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    baseline_risks: list[torch.Tensor] = []
    baseline_margins: list[torch.Tensor] = []
    for step in range(steps):
        with torch.no_grad():
            policy_mean = act(actor_obs)
            if step % sample_interval == 0:
                baseline_risk, baseline_margin = planning_outcomes(env, policy_mean)
                risk_gradient = torch.zeros(
                    (num_envs, env.n_agents, 2), device=env.device
                )
                margin_gradient = torch.zeros_like(risk_gradient)
                valid = torch.zeros_like(risk_gradient, dtype=torch.bool)
                for agent in range(env.n_agents):
                    for dimension in range(2):
                        minus_action = policy_mean.clone()
                        plus_action = policy_mean.clone()
                        minus_action[:, agent, dimension] = (
                            minus_action[:, agent, dimension] - action_delta
                        ).clamp(-1.0, 1.0)
                        plus_action[:, agent, dimension] = (
                            plus_action[:, agent, dimension] + action_delta
                        ).clamp(-1.0, 1.0)
                        minus_risk, minus_margin = planning_outcomes(env, minus_action)
                        plus_risk, plus_margin = planning_outcomes(env, plus_action)
                        risk_derivative, derivative_valid = central_difference(
                            minus_risk[:, agent],
                            plus_risk[:, agent],
                            minus_action[:, agent, dimension],
                            plus_action[:, agent, dimension],
                        )
                        margin_derivative, _ = central_difference(
                            minus_margin[:, agent],
                            plus_margin[:, agent],
                            minus_action[:, agent, dimension],
                            plus_action[:, agent, dimension],
                        )
                        risk_gradient[:, agent, dimension] = risk_derivative
                        margin_gradient[:, agent, dimension] = margin_derivative
                        valid[:, agent, dimension] = derivative_valid
                risk_gradients.append(risk_gradient.cpu())
                margin_gradients.append(margin_gradient.cpu())
                valid_masks.append(valid.cpu())
                baseline_risks.append(baseline_risk.cpu())
                baseline_margins.append(baseline_margin.cpu())

            noise = torch.randn(
                policy_mean.shape,
                generator=generator,
                device=env.device,
                dtype=policy_mean.dtype,
            )
            rollout_action = (
                policy_mean + exploration_multiplier * policy_std * noise
            ).clamp(-1.0, 1.0)
            output = env.step(rollout_action)
            actor_obs = output.actor_obs

    risk_gradient = torch.cat(risk_gradients, dim=0)
    margin_gradient = torch.cat(margin_gradients, dim=0)
    valid = torch.cat(valid_masks, dim=0).all(dim=-1)
    risk_norm = torch.linalg.norm(risk_gradient, dim=-1)
    margin_norm = torch.linalg.norm(margin_gradient, dim=-1)
    denominator = risk_norm * margin_norm
    alignment = torch.where(
        denominator > 1.0e-8,
        ((-risk_gradient) * margin_gradient).sum(dim=-1) / denominator.clamp_min(1.0e-8),
        torch.nan,
    )
    aligned = alignment[valid & torch.isfinite(alignment)]
    baseline_risk = torch.cat(baseline_risks, dim=0)
    baseline_margin = torch.cat(baseline_margins, dim=0)
    normalized_risk_response = action_delta * risk_norm / baseline_risk.std().clamp_min(1.0e-6)
    normalized_margin_response = action_delta * margin_norm / baseline_margin.std().clamp_min(1.0e-6)

    with torch.no_grad():
        probe_after = act(probe_obs)
    digest_after = _policy_digest(checkpoint_data)
    action_change = float((probe_after - probe_before).abs().amax().cpu())
    result: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "status": "diagnostic_complete",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "collection": {
            "seed": seed,
            "num_envs": num_envs,
            "steps": steps,
            "episode_coverage_seconds": steps * float(cfg.simulation.planning_dt),
            "sample_interval": sample_interval,
            "sampled_planning_steps": len(risk_gradients),
            "agent_state_samples": int(risk_norm.numel()),
            "action_delta": action_delta,
            "exploration_multiplier": exploration_multiplier,
        },
        "responses": {
            "relative_risk_gradient_norm": _summary(risk_norm[valid]),
            "nearest_margin_gradient_norm": _summary(margin_norm[valid]),
            "normalized_relative_risk_response": _summary(normalized_risk_response[valid]),
            "normalized_nearest_margin_response": _summary(
                normalized_margin_response[valid]
            ),
            "risk_descent_vs_margin_increase_cosine": _summary(aligned),
            "aligned_fraction_cosine_gt_0": float((aligned > 0.0).float().mean()),
            "strongly_aligned_fraction_cosine_gt_0_5": float(
                (aligned > 0.5).float().mean()
            ),
            "opposed_fraction_cosine_lt_minus_0_5": float(
                (aligned < -0.5).float().mean()
            ),
            "valid_central_difference_fraction": float(valid.float().mean()),
            "baseline_relative_risk_std": float(baseline_risk.std()),
            "baseline_nearest_margin_std": float(baseline_margin.std()),
        },
        "invariance": {
            "actor_digest_before": digest_before,
            "actor_digest_after": digest_after,
            "actor_probe_action_max_abs_change": action_change,
        },
        "decision_boundary": (
            "diagnostic_only_no_reward_or_training_change; use gradient alignment to decide "
            "whether a single terrain-safety credit hypothesis is coherent"
        ),
    }
    run_dir_path = (
        Path(run_dir)
        if run_dir is not None
        else ROOT / "outputs" / "runs" / EXPERIMENT_ID / "frozen_exp125_seed23"
    )
    if not run_dir_path.is_absolute():
        run_dir_path = ROOT / run_dir_path
    metrics_dir = run_dir_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "paired_action_interventions.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "producer": "scripts/analyze_paired_action_interventions.py",
        "status": result["status"],
        "source_checkpoint": str(checkpoint),
        "artifacts": {"metrics": str(metrics_path.relative_to(ROOT))},
    }
    (run_dir_path / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    result["artifact"] = str(metrics_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--sample-interval", type=int, default=8)
    parser.add_argument("--action-delta", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17023)
    parser.add_argument("--exploration-multiplier", type=float, default=1.0)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = analyze_paired_action_interventions(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        sample_interval=args.sample_interval,
        action_delta=args.action_delta,
        seed=args.seed,
        exploration_multiplier=args.exploration_multiplier,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
