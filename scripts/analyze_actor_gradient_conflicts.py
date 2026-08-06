#!/usr/bin/env python
"""Audit team-PPO versus terrain-credit Actor gradient interference offline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from skrl.multi_agents.torch.mappo.mappo import compute_gae

from _common import ROOT, cfg_from_experiment, load_yaml
from analyze_exp125_credit_assignment import _load_critic, _value
from analyze_joint_action_critic_feasibility import _parse_int_tuple, tensor_dict_digest
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from shared_policy_mappo import normalized_centered_credit_traces
from train_skrl_mappo import SKRLPolicy


EXPERIMENT_ID = "exp130_actor_gradient_conflict_audit"


def nearest_neighbor_safety_potential(
    nearest_neighbor_distance: torch.Tensor,
    safe_distance: float,
) -> torch.Tensor:
    """Return the existing min-pair success-gate potential for each agent."""

    return -torch.relu(
        torch.as_tensor(
            safe_distance,
            device=nearest_neighbor_distance.device,
            dtype=nearest_neighbor_distance.dtype,
        )
        - nearest_neighbor_distance
    )


def centered_safety_potential_credit(
    nearest_before: torch.Tensor,
    nearest_after: torch.Tensor,
    safe_distance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute raw and zero-sum per-agent progress in the safety potential."""

    raw_credit = nearest_neighbor_safety_potential(
        nearest_after, safe_distance
    ) - nearest_neighbor_safety_potential(nearest_before, safe_distance)
    return raw_credit, raw_credit - raw_credit.mean(dim=1, keepdim=True)


def agent_local_gather_progress(
    positions_before: torch.Tensor,
    positions_after: torch.Tensor,
) -> torch.Tensor:
    """Return per-agent progress toward the leave-one-out team centroid.

    The inputs are centralized diagnostic snapshots shaped ``[env, agent, xyz]``.
    The result is never exposed to the Actor; it is used only as an offline
    training-target identifiability diagnostic.
    """

    if positions_before.shape != positions_after.shape:
        raise ValueError("Gather-progress snapshots must have identical shapes.")
    if positions_before.ndim != 3 or positions_before.shape[-1] < 2:
        raise ValueError("Gather-progress snapshots must have shape [env, agent, >=2].")
    n_agents = positions_before.shape[1]
    if n_agents < 2:
        raise ValueError("Leave-one-out gather progress requires at least two agents.")

    def potential(positions: torch.Tensor) -> torch.Tensor:
        xy = positions[..., :2]
        other_centroid = (xy.sum(dim=1, keepdim=True) - xy) / float(n_agents - 1)
        return -torch.linalg.vector_norm(xy - other_centroid, dim=-1)

    return potential(positions_after) - potential(positions_before)


def nearest_neighbor_indices(positions: torch.Tensor) -> torch.Tensor:
    """Return deterministic nearest-neighbor indices for offline diagnostics."""

    if positions.ndim != 3 or positions.shape[-1] < 2:
        raise ValueError("Positions must have shape [env, agent, >=2].")
    n_agents = positions.shape[1]
    if n_agents < 2:
        raise ValueError("Nearest-neighbor diagnostics require at least two agents.")
    xy = positions[..., :2]
    distances = torch.cdist(xy, xy)
    diagonal = torch.eye(n_agents, device=positions.device, dtype=torch.bool)[None]
    return distances.masked_fill(diagonal, float("inf")).argmin(dim=-1)


def project_auxiliary_against_primary(
    primary: torch.Tensor,
    auxiliary: torch.Tensor,
) -> torch.Tensor:
    """Remove only the auxiliary component opposing the primary gradient."""

    dot = torch.dot(primary, auxiliary)
    if dot >= 0.0:
        return auxiliary
    return auxiliary - dot / primary.square().sum().clamp_min(1.0e-12) * primary


def gradient_metrics(primary: torch.Tensor, auxiliary: torch.Tensor) -> dict[str, float]:
    primary_norm = torch.linalg.norm(primary)
    auxiliary_norm = torch.linalg.norm(auxiliary)
    denominator = primary_norm * auxiliary_norm
    cosine = (
        float(torch.dot(primary, auxiliary).div(denominator).cpu())
        if denominator > 1.0e-12
        else 0.0
    )
    projected = project_auxiliary_against_primary(primary, auxiliary)
    combined = primary + 0.25 * projected
    combined_norm = torch.linalg.norm(combined)
    primary_alignment = (
        float(torch.dot(primary, combined).div(primary_norm * combined_norm).cpu())
        if primary_norm * combined_norm > 1.0e-12
        else 0.0
    )
    return {
        "cosine": cosine,
        "dot": float(torch.dot(primary, auxiliary).cpu()),
        "primary_norm": float(primary_norm.cpu()),
        "auxiliary_norm": float(auxiliary_norm.cpu()),
        "auxiliary_primary_norm_ratio": float(
            auxiliary_norm.div(primary_norm.clamp_min(1.0e-12)).cpu()
        ),
        "projected_primary_dot": float(torch.dot(primary, projected).cpu()),
        "primary_vs_combined_cosine_at_scale_0_25": primary_alignment,
    }


def _flatten_gradients(
    gradients: tuple[torch.Tensor | None, ...],
    parameters: tuple[torch.nn.Parameter, ...],
    names: tuple[str, ...],
    group: str,
) -> torch.Tensor:
    selected: list[torch.Tensor] = []
    for gradient, parameter, name in zip(gradients, parameters, names, strict=True):
        if group != "all" and not name.startswith(group + "."):
            continue
        selected.append(
            torch.zeros_like(parameter).flatten() if gradient is None else gradient.flatten()
        )
    if not selected:
        raise KeyError(f"No policy parameters found for gradient group {group!r}.")
    return torch.cat(selected)


def _policy_log_prob(
    policy: SKRLPolicy,
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    mean, outputs = policy.compute({"observations": observations}, role="policy")
    std = outputs["log_std"].exp()
    return torch.distributions.Normal(mean, std).log_prob(actions).sum(dim=-1, keepdim=True)


def _collect_seed_dataset(
    *,
    config: str | Path,
    checkpoint_data: dict[str, Any],
    device: str,
    num_envs: int,
    steps: int,
    rollout_length: int,
    seed: int,
    safety_credit_distance: float | None = None,
) -> dict[str, torch.Tensor]:
    if steps <= 0 or rollout_length <= 0:
        raise ValueError("steps and rollout_length must be positive.")
    raw_cfg = load_yaml(config)
    cfg = cfg_from_experiment(config)
    cfg.simulation.device = device
    cfg.simulation.num_envs = num_envs
    cfg.seed = seed
    metadata = checkpoint_data.get("metadata") or {}
    if cfg.initial_state.curriculum_enabled:
        cfg.initial_state.progress_timestep_override = int(metadata.get("timesteps", 0))
    env = MultiRoverGatheringCore(cfg)
    critic = _load_critic(checkpoint_data, cfg, env.device)
    obs_space = gym.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(cfg.actor_obs_dim,), dtype=float
    )
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    policy = SKRLPolicy(
        obs_space,
        action_space,
        env.device,
        architecture=str(metadata.get("actor_architecture", "branched_v5")),
    ).to(env.device)
    policy.load_state_dict(checkpoint_data["rover_0"]["policy"])
    policy.eval()
    actor_obs, critic_state = env.get_observations()
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed + 7919)
    sequences: dict[str, list[torch.Tensor]] = {
        "observations": [],
        "actions": [],
        "nearest_neighbor_actions": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
        "values": [],
        "credits": [],
        "terrain_raw_credits": [],
        "local_gather_credits": [],
        "planned_path_arc_length": [],
        "planned_path_horizon": [],
        "planned_path_reference_speed": [],
        "planned_tracking_point_arc_length": [],
        "actual_step_displacement": [],
        "safety_raw_credits": [],
        "safety_credits": [],
        "repeated_conflict_involvement": [],
        "next_values": [],
    }
    safe_distance = float(
        cfg.success_thresholds.min_pairwise_distance
        if safety_credit_distance is None
        else safety_credit_distance
    )
    if safe_distance <= 0.0:
        raise ValueError("Safety-potential audit requires a positive min-pairwise gate.")
    for _ in range(steps):
        with torch.no_grad():
            flat_obs = actor_obs.reshape(num_envs * cfg.task.n_agents, -1)
            mean, outputs = policy.compute({"observations": flat_obs}, role="policy")
            std = outputs["log_std"].exp()
            actions = (
                mean
                + std
                * torch.randn(
                    mean.shape,
                    generator=generator,
                    device=env.device,
                    dtype=mean.dtype,
                )
            ).clamp(-1.0, 1.0).reshape(num_envs, cfg.task.n_agents, 2)
            values = _value(critic, critic_state)
            sequences["observations"].append(actor_obs.clone())
            sequences["actions"].append(actions.clone())
            sequences["values"].append(values[:, None].clone())
            positions_before = env.positions.clone()
            nearest_indices = nearest_neighbor_indices(positions_before)
            nearest_actions = actions.gather(
                1,
                nearest_indices[..., None].expand(-1, -1, actions.shape[-1]),
            )
            sequences["nearest_neighbor_actions"].append(nearest_actions.clone())
            nearest_before = env.metrics.nearest_neighbor_distance.clone()
            output = env.step(actions)
            relative_risk = (output.info.get("path_terrain") or {}).get(
                "relative_risk_mean"
            )
            if not isinstance(relative_risk, torch.Tensor):
                raise RuntimeError("Gradient audit requires relative quintic path risk.")
            raw_credit = -relative_risk
            centered_credit = raw_credit - raw_credit.mean(dim=1, keepdim=True)
            positions_after = output.info.get("positions")
            if not isinstance(positions_after, torch.Tensor):
                raise RuntimeError("Gradient audit requires the pre-reset position snapshot.")
            local_gather_credit = agent_local_gather_progress(
                positions_before,
                positions_after,
            )
            trajectory = output.info["trajectory"]
            path_segments = torch.linalg.vector_norm(
                trajectory.points[..., 1:, :2] - trajectory.points[..., :-1, :2],
                dim=-1,
            )
            sequences["planned_path_arc_length"].append(
                path_segments.sum(dim=-1).clone()
            )
            sequences["planned_path_horizon"].append(
                trajectory.timestamps[..., -1].clone()
            )
            sequences["planned_path_reference_speed"].append(
                trajectory.reference_speed[..., 0].clone()
            )
            sequences["planned_tracking_point_arc_length"].append(
                path_segments[..., 0].clone()
            )
            sequences["actual_step_displacement"].append(
                torch.linalg.vector_norm(
                    positions_after[..., :2] - positions_before[..., :2], dim=-1
                ).clone()
            )
            safety_raw_credit, safety_credit = centered_safety_potential_credit(
                nearest_before,
                output.info["metrics"].nearest_neighbor_distance,
                safe_distance,
            )
            sequences["rewards"].append(output.rewards[:, :1].clone())
            sequences["terminated"].append(output.terminated[:, None].clone())
            sequences["truncated"].append(output.truncated[:, None].clone())
            sequences["credits"].append(centered_credit.clone())
            sequences["terrain_raw_credits"].append(raw_credit.clone())
            sequences["local_gather_credits"].append(local_gather_credit.clone())
            sequences["safety_raw_credits"].append(safety_raw_credit.clone())
            sequences["safety_credits"].append(safety_credit.clone())
            repeated_pairs = output.info["trajectory_conflicts"]["repeated"]
            repeated_involvement = repeated_pairs.any(dim=2) | repeated_pairs.any(
                dim=1
            )
            sequences["repeated_conflict_involvement"].append(
                repeated_involvement.float().clone()
            )
            sequences["next_values"].append(
                _value(critic, output.critic_state)[:, None].clone()
            )
            actor_obs = output.actor_obs
            critic_state = output.critic_state

    stacked = {key: torch.stack(values) for key, values in sequences.items()}
    team_advantage_segments: list[torch.Tensor] = []
    credit_trace_segments: list[torch.Tensor] = []
    safety_trace_segments: list[torch.Tensor] = []
    safety_active_rollout_segments: list[torch.Tensor] = []
    safety_environment_rollout_active_segments: list[torch.Tensor] = []
    gamma = float((raw_cfg.get("algorithm") or {}).get("gamma", 0.99))
    gae_lambda = float((raw_cfg.get("algorithm") or {}).get("gae_lambda", 0.95))
    for start in range(0, steps, rollout_length):
        # A complete 96 s episode contains 480 planning steps, which is not an
        # integer multiple of the training rollout length 64. The final partial
        # segment is still a valid finite-horizon diagnostic segment.
        end = min(start + rollout_length, steps)
        _, team_advantage = compute_gae(
            rewards=stacked["rewards"][start:end],
            terminated=stacked["terminated"][start:end],
            truncated=stacked["truncated"][start:end],
            values=stacked["values"][start:end],
            last_values=stacked["next_values"][end - 1],
            discount_factor=gamma,
            lambda_coefficient=gae_lambda,
            time_limit_bootstrap=False,
        )
        credit_trace = normalized_centered_credit_traces(
            stacked["credits"][start:end].permute(2, 0, 1).unsqueeze(-1),
            stacked["terminated"][start:end],
            stacked["truncated"][start:end],
            discount_factor=gamma,
            trace_lambda=0.95,
            time_limit_bootstrap=False,
        )
        safety_trace = normalized_centered_credit_traces(
            stacked["safety_credits"][start:end].permute(2, 0, 1).unsqueeze(-1),
            stacked["terminated"][start:end],
            stacked["truncated"][start:end],
            discount_factor=gamma,
            trace_lambda=0.95,
            time_limit_bootstrap=False,
        )
        team_advantage_segments.append(team_advantage)
        credit_trace_segments.append(credit_trace.permute(1, 2, 0, 3))
        safety_trace_segments.append(safety_trace.permute(1, 2, 0, 3))
        safety_active_rollout_segments.append(
            (stacked["safety_credits"][start:end].abs() > 1.0e-8).any().float()
        )
        safety_environment_rollout_active_segments.append(
            (stacked["safety_credits"][start:end].abs() > 1.0e-8)
            .any(dim=(0, 2))
            .float()
            .mean()
        )
    team_advantages = torch.cat(team_advantage_segments, dim=0)
    credit_traces = torch.cat(credit_trace_segments, dim=0)
    safety_traces = torch.cat(safety_trace_segments, dim=0)
    observations = stacked["observations"]
    actions = stacked["actions"]
    # Agent-major order exactly matches SharedPolicyMAPPO.update_joint.
    return {
        "observations": torch.cat(
            [observations[:, :, agent].reshape(-1, cfg.actor_obs_dim) for agent in range(cfg.task.n_agents)]
        ),
        "actions": torch.cat(
            [actions[:, :, agent].reshape(-1, 2) for agent in range(cfg.task.n_agents)]
        ),
        "nearest_neighbor_actions": torch.cat(
            [
                stacked["nearest_neighbor_actions"][:, :, agent].reshape(-1, 2)
                for agent in range(cfg.task.n_agents)
            ]
        ),
        "team_advantages": team_advantages.reshape(-1, 1).repeat(cfg.task.n_agents, 1),
        "terrain_credits": torch.cat(
            [credit_traces[:, :, agent].reshape(-1, 1) for agent in range(cfg.task.n_agents)]
        ),
        "terrain_raw_credits": torch.cat(
            [
                stacked["terrain_raw_credits"][:, :, agent].reshape(-1, 1)
                for agent in range(cfg.task.n_agents)
            ]
        ),
        "local_gather_credits": torch.cat(
            [
                stacked["local_gather_credits"][:, :, agent].reshape(-1, 1)
                for agent in range(cfg.task.n_agents)
            ]
        ),
        **{
            key: torch.cat(
                [stacked[key][:, :, agent].reshape(-1, 1) for agent in range(cfg.task.n_agents)]
            )
            for key in (
                "planned_path_arc_length",
                "planned_path_horizon",
                "planned_path_reference_speed",
                "planned_tracking_point_arc_length",
                "actual_step_displacement",
            )
        },
        "safety_credits": torch.cat(
            [safety_traces[:, :, agent].reshape(-1, 1) for agent in range(cfg.task.n_agents)]
        ),
        "safety_raw_credits": torch.cat(
            [
                stacked["safety_raw_credits"][:, :, agent].reshape(-1, 1)
                for agent in range(cfg.task.n_agents)
            ]
        ),
        "safety_centered_step_credits": torch.cat(
            [
                stacked["safety_credits"][:, :, agent].reshape(-1, 1)
                for agent in range(cfg.task.n_agents)
            ]
        ),
        "repeated_conflict_involvement": torch.cat(
            [
                stacked["repeated_conflict_involvement"][:, :, agent].reshape(
                    -1, 1
                )
                for agent in range(cfg.task.n_agents)
            ]
        ),
        "safety_centered_zero_sum_max_abs": stacked["safety_credits"]
        .sum(dim=-1)
        .abs()
        .max(),
        "safety_active_rollout_fraction": torch.stack(
            safety_active_rollout_segments
        ).mean(),
        "safety_rollout_active_flags": torch.stack(
            safety_active_rollout_segments
        ),
        "safety_environment_rollout_active_fractions": torch.stack(
            safety_environment_rollout_active_segments
        ),
        "safe_distance": torch.tensor(safe_distance, device=env.device),
        "policy": policy,
    }


def _summarize_batch_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    cosine = torch.tensor([item["cosine"] for item in items])
    ratio = torch.tensor([item["auxiliary_primary_norm_ratio"] for item in items])
    projected_dot = torch.tensor([item["projected_primary_dot"] for item in items])
    alignment = torch.tensor(
        [item["primary_vs_combined_cosine_at_scale_0_25"] for item in items]
    )
    return {
        "batches": len(items),
        "negative_cosine_fraction": float((cosine < 0.0).float().mean()),
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(cosine.median()),
        "cosine_p10": float(torch.quantile(cosine, 0.10)),
        "cosine_p90": float(torch.quantile(cosine, 0.90)),
        "norm_ratio_median": float(ratio.median()),
        "norm_ratio_p10": float(torch.quantile(ratio, 0.10)),
        "norm_ratio_p90": float(torch.quantile(ratio, 0.90)),
        "projected_primary_dot_min": float(projected_dot.min()),
        "primary_combined_alignment_min": float(alignment.min()),
        "primary_combined_alignment_mean": float(alignment.mean()),
    }


def analyze_actor_gradient_conflicts(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    num_envs: int = 128,
    steps: int = 512,
    rollout_length: int = 64,
    data_seeds: tuple[int, ...] = (18023, 19023),
    batch_size: int = 4096,
    batches: int = 32,
    batch_seed: int = 230,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_data = torch.load(checkpoint, map_location=torch.device(device))
    digest_before = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
    per_seed: dict[str, Any] = {}
    seed_conflict_rates: list[float] = []
    seed_norm_ratios: list[float] = []
    for data_seed in data_seeds:
        dataset = _collect_seed_dataset(
            config=config,
            checkpoint_data=checkpoint_data,
            device=device,
            num_envs=num_envs,
            steps=steps,
            rollout_length=rollout_length,
            seed=data_seed,
        )
        policy: SKRLPolicy = dataset.pop("policy")
        named_parameters = tuple(
            (name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad
        )
        names = tuple(item[0] for item in named_parameters)
        parameters = tuple(item[1] for item in named_parameters)
        groups = ("all", "terrain_encoder", "trunk")
        group_items: dict[str, list[dict[str, float]]] = {group: [] for group in groups}
        generator = torch.Generator(device=policy.device)
        generator.manual_seed(batch_seed + data_seed)
        sample_count = dataset["observations"].shape[0]
        for _ in range(batches):
            index = torch.randint(
                sample_count,
                (min(batch_size, sample_count),),
                generator=generator,
                device=policy.device,
            )
            observations = dataset["observations"][index]
            actions = dataset["actions"][index]
            team_advantages = dataset["team_advantages"][index]
            terrain_credits = dataset["terrain_credits"][index]
            team_log_prob = _policy_log_prob(policy, observations, actions)
            team_loss = -(team_advantages * team_log_prob).mean()
            team_gradients = torch.autograd.grad(
                team_loss, parameters, allow_unused=True
            )
            terrain_log_prob = _policy_log_prob(policy, observations, actions)
            terrain_loss = -(terrain_credits * terrain_log_prob).mean()
            terrain_gradients = torch.autograd.grad(
                terrain_loss, parameters, allow_unused=True
            )
            for group in groups:
                primary = _flatten_gradients(team_gradients, parameters, names, group)
                auxiliary = _flatten_gradients(terrain_gradients, parameters, names, group)
                group_items[group].append(gradient_metrics(primary, auxiliary))
        summaries = {
            group: _summarize_batch_metrics(items) for group, items in group_items.items()
        }
        per_seed[str(data_seed)] = {
            "actor_samples": sample_count,
            "groups": summaries,
        }
        seed_conflict_rates.append(summaries["all"]["negative_cosine_fraction"])
        seed_norm_ratios.append(summaries["all"]["norm_ratio_median"])

    digest_after = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
    checks = {
        "mean_conflict_fraction_ge_0_20": sum(seed_conflict_rates) / len(seed_conflict_rates)
        >= 0.20,
        "every_seed_conflict_fraction_ge_0_20": min(seed_conflict_rates) >= 0.20,
        "every_seed_norm_ratio_ge_0_05": min(seed_norm_ratios) >= 0.05,
        "every_seed_norm_ratio_le_20": max(seed_norm_ratios) <= 20.0,
        "actor_checkpoint_unchanged": digest_before == digest_after,
    }
    passed = all(checks.values())
    result: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "status": "allow_c2_plan_only" if passed else "stop_gradient_projection_direction",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "collection": {
            "num_envs": num_envs,
            "steps": steps,
            "rollout_length": rollout_length,
            "data_seeds": list(data_seeds),
            "batch_size": batch_size,
            "gradient_batches_per_seed": batches,
            "batch_seed": batch_seed,
        },
        "method": {
            "primary": "team MAPPO GAE policy gradient",
            "auxiliary": "exp126 centered relative-terrain credit trace gradient",
            "projection": "asymmetric primary-preserving PCGrad-inspired diagnostic",
            "training_or_optimizer_modified": False,
        },
        "per_seed": per_seed,
        "aggregate": {
            "mean_negative_cosine_fraction": sum(seed_conflict_rates)
            / len(seed_conflict_rates),
            "minimum_seed_negative_cosine_fraction": min(seed_conflict_rates),
            "median_norm_ratio_mean": sum(seed_norm_ratios) / len(seed_norm_ratios),
        },
        "checks": checks,
        "invariance": {
            "actor_digest_before": digest_before,
            "actor_digest_after": digest_after,
        },
        "decision": "draft_c2_screen_plan_only" if passed else "do_not_implement_c2",
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
    metrics_path = metrics_dir / "actor_gradient_conflicts.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "producer": "scripts/analyze_actor_gradient_conflicts.py",
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
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--data-seeds", type=_parse_int_tuple, default=(18023, 19023))
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--batch-seed", type=int, default=230)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = analyze_actor_gradient_conflicts(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        rollout_length=args.rollout_length,
        data_seeds=args.data_seeds,
        batch_size=args.batch_size,
        batches=args.batches,
        batch_seed=args.batch_seed,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
