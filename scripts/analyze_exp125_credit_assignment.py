#!/usr/bin/env python
"""Diagnose B0 team-reward credit assignment without changing the policy.

The shared-joint implementation computes one team reward and one GAE stream,
then repeats that advantage for all rover policy samples. This evaluator uses
the frozen critic's one-step TD residual as an advantage proxy and measures how
well it aligns with each rover's terrain and local safety outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from play import _load_policy_players
from train_skrl_mappo import SKRLValue


def pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().float().flatten()
    right = right.detach().float().flatten()
    finite = torch.isfinite(left) & torch.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.numel() < 2:
        return None
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum())
    if denominator <= 1.0e-12:
        return None
    return float((left * right).sum().div(denominator).cpu())


def rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    """Return Spearman correlation using average ranks for tied values."""

    left = left.detach().float().flatten()
    right = right.detach().float().flatten()
    finite = torch.isfinite(left) & torch.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.numel() < 2:
        return None
    def average_ranks(values: torch.Tensor) -> torch.Tensor:
        sorted_values, order = values.sort()
        _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
        starts = counts.cumsum(dim=0) - counts
        group_ranks = starts.float() + (counts.float() - 1.0) / 2.0
        sorted_ranks = torch.repeat_interleave(group_ranks, counts)
        ranks = torch.empty_like(values)
        ranks[order] = sorted_ranks
        return ranks

    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    return pearson_correlation(left_rank, right_rank)


def _load_critic(checkpoint: dict, cfg, device: torch.device) -> SKRLValue:
    metadata = checkpoint.get("metadata") or {}
    obs_space = gym.spaces.Box(
        low=-float("inf"),
        high=float("inf"),
        shape=(cfg.actor_obs_dim,),
        dtype=float,
    )
    state_space = gym.spaces.Box(
        low=-float("inf"),
        high=float("inf"),
        shape=(cfg.critic_state_dim,),
        dtype=float,
    )
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    critic = SKRLValue(
        obs_space,
        state_space,
        action_space,
        device,
        architecture=str(metadata.get("critic_architecture", "mlp_v1")),
    ).to(device)
    critic.load_state_dict(checkpoint["rover_0"]["value"])
    critic.eval()
    return critic


def _value(critic: SKRLValue, states: torch.Tensor) -> torch.Tensor:
    values, _ = critic.compute({"states": states}, role="value")
    return values.squeeze(-1)


def analyze_credit_assignment(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    num_envs: int = 512,
    steps: int = 120,
    seed: int = 13023,
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
        raise ValueError("Credit analysis requires ego_v8_decentralized_tiered.")

    checkpoint_data = torch.load(checkpoint, map_location=torch.device(device))
    metadata = checkpoint_data.get("metadata") or {}
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
    critic = _load_critic(checkpoint_data, cfg, env.device)
    actor_obs, critic_state = env.get_observations()
    gamma = float((raw_cfg.get("algorithm") or {}).get("gamma", 0.99))

    collected: dict[str, list[torch.Tensor]] = {
        "td_advantage_proxy": [],
        "relative_path_risk": [],
        "local_centroid_progress": [],
        "nearest_distance_change": [],
        "team_dmax_progress": [],
        "absolute_turn_action": [],
    }
    valid_team_samples = 0

    for _ in range(steps):
        with torch.no_grad():
            actions = act(actor_obs)
            current_value = _value(critic, critic_state)
            positions_before = env.positions.clone()
            velocities_before = env.velocities_xy.clone()
            metrics_before = compute_team_metrics(positions_before, velocities_before)
            centroid_distance_before = torch.linalg.norm(
                positions_before[..., :2] - metrics_before.centroid[:, None, :2],
                dim=-1,
            )

            step_output = env.step(actions)
            done = step_output.terminated | step_output.truncated
            valid = ~done
            next_value = _value(critic, step_output.critic_state)
            team_reward = step_output.rewards[:, 0]
            td = team_reward + gamma * next_value - current_value

            metrics_after = step_output.info["metrics"]
            positions_after = env.positions
            centroid_distance_after = torch.linalg.norm(
                positions_after[..., :2] - metrics_after.centroid[:, None, :2],
                dim=-1,
            )
            path_terrain = step_output.info.get("path_terrain") or {}
            relative_risk = path_terrain.get("relative_risk_mean")
            if relative_risk is None:
                raise ValueError(
                    "Credit analysis requires path_terrain_relative_cost to be enabled."
                )

            if valid.any():
                n_agents = cfg.task.n_agents
                valid_team_samples += int(valid.sum().cpu())
                collected["td_advantage_proxy"].append(
                    td[valid, None].expand(-1, n_agents).reshape(-1).cpu()
                )
                collected["relative_path_risk"].append(relative_risk[valid].reshape(-1).cpu())
                collected["local_centroid_progress"].append(
                    (centroid_distance_before[valid] - centroid_distance_after[valid])
                    .reshape(-1)
                    .cpu()
                )
                collected["nearest_distance_change"].append(
                    (
                        metrics_after.nearest_neighbor_distance[valid]
                        - metrics_before.nearest_neighbor_distance[valid]
                    )
                    .reshape(-1)
                    .cpu()
                )
                collected["team_dmax_progress"].append(
                    (metrics_before.dmax[valid] - metrics_after.dmax[valid])[:, None]
                    .expand(-1, n_agents)
                    .reshape(-1)
                    .cpu()
                )
                collected["absolute_turn_action"].append(
                    actions[valid, :, 1].abs().reshape(-1).cpu()
                )

            actor_obs = step_output.actor_obs
            critic_state = step_output.critic_state

    tensors = {
        key: torch.cat(parts) if parts else torch.empty(0)
        for key, parts in collected.items()
    }
    advantage = tensors["td_advantage_proxy"]
    relative_risk = tensors["relative_path_risk"]
    result = {
        "status": "ok",
        "backend": backend,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "device": str(env.device),
        "num_envs": num_envs,
        "steps": steps,
        "seed": seed,
        "valid_team_samples": valid_team_samples,
        "valid_agent_samples": int(advantage.numel()),
        "algorithm_semantics": {
            "team_reward_shared_by_all_agents": True,
            "gae_advantage_repeated_for_all_actor_samples": True,
            "per_agent_reward_or_advantage": False,
            "td_residual_used_as_advantage_proxy": True,
        },
        "means": {
            key: float(value.mean()) if value.numel() else None
            for key, value in tensors.items()
        },
        "correlations": {
            "relative_risk_vs_td_advantage": {
                "pearson": pearson_correlation(relative_risk, advantage),
                "rank": rank_correlation(relative_risk, advantage),
            },
            "relative_risk_vs_local_centroid_progress": {
                "pearson": pearson_correlation(
                    relative_risk, tensors["local_centroid_progress"]
                ),
                "rank": rank_correlation(
                    relative_risk, tensors["local_centroid_progress"]
                ),
            },
            "relative_risk_vs_nearest_distance_change": {
                "pearson": pearson_correlation(
                    relative_risk, tensors["nearest_distance_change"]
                ),
                "rank": rank_correlation(
                    relative_risk, tensors["nearest_distance_change"]
                ),
            },
            "relative_risk_vs_team_dmax_progress": {
                "pearson": pearson_correlation(
                    relative_risk, tensors["team_dmax_progress"]
                ),
                "rank": rank_correlation(
                    relative_risk, tensors["team_dmax_progress"]
                ),
            },
            "absolute_turn_vs_relative_risk": {
                "pearson": pearson_correlation(
                    tensors["absolute_turn_action"], relative_risk
                ),
                "rank": rank_correlation(
                    tensors["absolute_turn_action"], relative_risk
                ),
            },
        },
    }

    if output is None and run_dir is not None:
        output = Path(run_dir) / "metrics" / "credit_assignment.json"
    if output is not None:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["artifact"] = str(output_path)
        if run_dir is not None:
            manifest_path = Path(run_dir) / "run_manifest.json"
            if not manifest_path.is_absolute():
                manifest_path = ROOT / manifest_path
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                artifacts = manifest.setdefault("artifacts", {})
                artifacts["metrics_credit_assignment"] = str(
                    output_path.relative_to(ROOT)
                )
                manifest_path.write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8",
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=13023)
    parser.add_argument("--initial-state-progress", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = analyze_credit_assignment(
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
