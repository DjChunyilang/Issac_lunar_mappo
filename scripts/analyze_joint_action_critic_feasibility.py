#!/usr/bin/env python
"""Offline feasibility probe for continuous-action counterfactual credit.

This script freezes an exp125 Actor and Critic, collects stochastic on-policy
rollouts, and trains two diagnostic regressors on identical 16-step return
targets:

* a state-only critic ``Q_s(s)``;
* a joint-action critic ``Q_{sa}(s, a_1, ..., a_N)``.

Neither diagnostic model is used by the environment or policy. The probe only
asks whether joint actions carry enough held-out information to justify a later
counterfactual-credit design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from _common import ROOT, cfg_from_experiment, load_yaml
from analyze_exp125_credit_assignment import (
    _load_critic,
    _value,
    pearson_correlation,
    rank_correlation,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from play import _load_policy_players


EXPERIMENT_ID = "exp127_joint_action_critic_feasibility"


@dataclass(slots=True)
class RolloutDataset:
    states: torch.Tensor
    actions: torch.Tensor
    policy_means: torch.Tensor
    targets: torch.Tensor
    conflict_involvement: torch.Tensor
    collision_involvement: torch.Tensor
    nearest_distance_change: torch.Tensor
    relative_path_risk: torch.Tensor
    reward_terms: dict[str, torch.Tensor]
    communication: dict[str, float]

    @property
    def samples(self) -> int:
        return int(self.states.shape[0])


class DiagnosticCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass(slots=True)
class FittedDiagnosticCritic:
    model: DiagnosticCritic
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor
    training_loss: float
    parameter_count: int

    def artifact_state(self) -> dict[str, Any]:
        """Return every quantity required to reproduce unnormalised predictions."""

        return {
            "model": {
                key: value.detach().cpu() for key, value in self.model.state_dict().items()
            },
            "feature_mean": self.feature_mean.detach().cpu(),
            "feature_std": self.feature_std.detach().cpu(),
            "target_mean": self.target_mean.detach().cpu(),
            "target_std": self.target_std.detach().cpu(),
            "training_loss": self.training_loss,
            "parameter_count": self.parameter_count,
        }

    def predict(self, features: torch.Tensor, *, batch_size: int = 16384) -> torch.Tensor:
        predictions: list[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, features.shape[0], batch_size):
                batch = features[start : start + batch_size]
                normalized = (batch - self.feature_mean) / self.feature_std
                prediction = self.model(normalized)
                predictions.append(prediction * self.target_std + self.target_mean)
        return torch.cat(predictions) if predictions else torch.empty(0, device=features.device)


def tensor_dict_digest(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(key.encode("utf-8"))
        tensor = values[key].detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def compute_n_step_targets(
    rewards: torch.Tensor,
    done: torch.Tensor,
    next_values: torch.Tensor,
    *,
    horizon: int,
    gamma: float,
) -> torch.Tensor:
    """Compute fixed-horizon bootstrapped targets for ``[time, env]`` tensors."""

    if rewards.ndim != 2 or done.shape != rewards.shape or next_values.shape != rewards.shape:
        raise ValueError("rewards, done and next_values must share shape [time, env].")
    if horizon <= 0 or horizon > rewards.shape[0]:
        raise ValueError("horizon must be in [1, rollout_steps].")
    start_count = rewards.shape[0] - horizon + 1
    target = torch.zeros(
        (start_count, rewards.shape[1]),
        dtype=rewards.dtype,
        device=rewards.device,
    )
    alive = torch.ones_like(target, dtype=torch.bool)
    discount = 1.0
    for offset in range(horizon):
        reward_slice = rewards[offset : offset + start_count]
        done_slice = done[offset : offset + start_count]
        target = target + discount * reward_slice * alive
        alive = alive & ~done_slice
        discount *= gamma
    bootstrap = next_values[horizon - 1 : horizon - 1 + start_count]
    return target + discount * bootstrap * alive


def _policy_std(checkpoint: dict, n_agents: int, device: torch.device) -> torch.Tensor:
    log_std = checkpoint["rover_0"]["policy"].get("log_std_parameter")
    if not isinstance(log_std, torch.Tensor):
        raise KeyError("Checkpoint policy has no log_std_parameter.")
    return log_std.detach().to(device).exp().view(1, 1, -1).expand(1, n_agents, -1)


def _collect_rollout_dataset(
    *,
    config: str | Path,
    checkpoint_data: dict,
    device: str,
    num_envs: int,
    steps: int,
    seed: int,
    horizon: int,
    exploration_multiplier: float,
) -> tuple[RolloutDataset, Any, torch.Tensor, torch.Tensor]:
    raw_cfg = load_yaml(config)
    cfg = cfg_from_experiment(config)
    cfg.simulation.device = device
    cfg.simulation.num_envs = num_envs
    cfg.seed = seed
    metadata = checkpoint_data.get("metadata") or {}
    if cfg.initial_state.curriculum_enabled:
        cfg.initial_state.progress_timestep_override = int(metadata.get("timesteps", 0))
    if cfg.observation.schema_version != "ego_v8_decentralized_tiered":
        raise ValueError("Joint-action critic diagnostic requires the v8 schema.")
    if cfg.reward_coefficients.path_terrain_relative_cost == 0.0:
        raise ValueError("Diagnostic requires relative quintic risk to be enabled.")

    env = MultiRoverGatheringCore(cfg)
    act, _ = _load_policy_players(checkpoint_data, cfg, env.device, raw_cfg=raw_cfg)
    frozen_critic = _load_critic(checkpoint_data, cfg, env.device)
    actor_obs, critic_state = env.get_observations()
    probe_obs = actor_obs[: min(32, num_envs)].detach().clone()
    with torch.no_grad():
        probe_action_before = act(probe_obs).detach().clone()

    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed + 7919)
    std = _policy_std(checkpoint_data, cfg.task.n_agents, env.device)

    sequences: dict[str, list[torch.Tensor]] = {
        "states": [],
        "actions": [],
        "policy_means": [],
        "rewards": [],
        "done": [],
        "next_values": [],
        "conflict_involvement": [],
        "collision_involvement": [],
        "nearest_distance_change": [],
        "relative_path_risk": [],
    }
    communication_sums: dict[str, float] = {}
    reward_term_names = (
        "gather",
        "oracle",
        "energy",
        "safety",
        "terrain",
        "flatness",
        "motion",
        "consistency",
        "success_hold",
        "terminal",
        "total",
    )
    reward_term_weights = {
        "gather": float(cfg.reward_weights.gather),
        "oracle": float(cfg.reward_weights.oracle),
        "energy": float(cfg.reward_weights.energy),
        "safety": float(cfg.reward_weights.safety),
        "terrain": float(cfg.reward_weights.terrain),
        "flatness": float(cfg.reward_weights.flatness),
        "motion": float(cfg.reward_weights.motion),
        "consistency": float(cfg.reward_weights.consistency),
        "success_hold": 1.0,
        "terminal": float(cfg.reward_weights.terminal),
        "total": 1.0,
    }
    reward_term_sequences: dict[str, list[torch.Tensor]] = {
        name: [] for name in reward_term_names
    }

    for _ in range(steps):
        with torch.no_grad():
            policy_mean = act(actor_obs)
            noise = torch.randn(
                policy_mean.shape,
                generator=generator,
                device=env.device,
                dtype=policy_mean.dtype,
            )
            actions = (policy_mean + exploration_multiplier * std * noise).clamp(-1.0, 1.0)
            metrics_before = compute_team_metrics(env.positions, env.velocities_xy)

            sequences["states"].append(critic_state.detach().clone())
            sequences["actions"].append(actions.detach().clone())
            sequences["policy_means"].append(policy_mean.detach().clone())

            output = env.step(actions)
            done = output.terminated | output.truncated
            metrics_after = output.info["metrics"]
            conflict_active = output.info["trajectory_conflicts"]["active"]
            symmetric_conflict = conflict_active | conflict_active.transpose(1, 2)
            conflict_involvement = symmetric_conflict.sum(dim=-1).float()
            collision_involvement = (
                metrics_after.nearest_neighbor_distance
                < float(cfg.safety.collision_distance)
            ).float()
            path_terrain = output.info.get("path_terrain") or {}
            relative_risk = path_terrain.get("relative_risk_mean")
            if not isinstance(relative_risk, torch.Tensor):
                raise RuntimeError("Rollout did not expose relative path risk.")
            reward_terms = output.info["reward_terms"]
            for name in reward_term_names:
                reward_term_sequences[name].append(
                    (
                        getattr(reward_terms, name).detach().clone()
                        * reward_term_weights[name]
                    )
                )
            weighted_sum = sum(
                reward_term_sequences[name][-1]
                for name in reward_term_names
                if name != "total"
            )
            if not torch.allclose(
                weighted_sum,
                reward_term_sequences["total"][-1],
                atol=1.0e-5,
                rtol=1.0e-5,
            ):
                raise RuntimeError("Weighted reward-term decomposition does not sum to total.")

            sequences["rewards"].append(output.rewards[:, 0].detach().clone())
            sequences["done"].append(done.detach().clone())
            sequences["next_values"].append(
                _value(frozen_critic, output.critic_state).detach().clone()
            )
            sequences["conflict_involvement"].append(conflict_involvement.detach().clone())
            sequences["collision_involvement"].append(collision_involvement.detach().clone())
            sequences["nearest_distance_change"].append(
                (
                    metrics_after.nearest_neighbor_distance
                    - metrics_before.nearest_neighbor_distance
                ).detach().clone()
            )
            sequences["relative_path_risk"].append(relative_risk.detach().clone())

            for key, value in (output.info.get("communication") or {}).items():
                if isinstance(value, torch.Tensor):
                    communication_sums[key] = communication_sums.get(key, 0.0) + float(
                        value.float().mean().cpu()
                    )
            actor_obs = output.actor_obs
            critic_state = output.critic_state

    stacked = {key: torch.stack(values, dim=0) for key, values in sequences.items()}
    stacked_reward_terms = {
        key: torch.stack(values, dim=0) for key, values in reward_term_sequences.items()
    }
    gamma = float((raw_cfg.get("algorithm") or {}).get("gamma", 0.99))
    targets = compute_n_step_targets(
        stacked["rewards"],
        stacked["done"],
        stacked["next_values"],
        horizon=horizon,
        gamma=gamma,
    )
    starts = targets.shape[0]

    def flatten_time_env(value: torch.Tensor) -> torch.Tensor:
        return value[:starts].reshape(-1, *value.shape[2:]).cpu()

    dataset = RolloutDataset(
        states=flatten_time_env(stacked["states"]),
        actions=flatten_time_env(stacked["actions"]).reshape(-1, cfg.task.n_agents * 2),
        policy_means=flatten_time_env(stacked["policy_means"]).reshape(
            -1, cfg.task.n_agents * 2
        ),
        targets=targets.reshape(-1).cpu(),
        conflict_involvement=flatten_time_env(stacked["conflict_involvement"]),
        collision_involvement=flatten_time_env(stacked["collision_involvement"]),
        nearest_distance_change=flatten_time_env(stacked["nearest_distance_change"]),
        relative_path_risk=flatten_time_env(stacked["relative_path_risk"]),
        reward_terms={
            key: value[:starts].reshape(-1).cpu()
            for key, value in stacked_reward_terms.items()
        },
        communication={key: value / steps for key, value in communication_sums.items()},
    )
    return dataset, act, probe_obs, probe_action_before


def fit_diagnostic_critic(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
) -> FittedDiagnosticCritic:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    features = features.to(device)
    targets = targets.to(device)
    feature_mean = features.mean(dim=0)
    feature_std = features.std(dim=0).clamp_min(1.0e-5)
    target_mean = targets.mean()
    target_std = targets.std().clamp_min(1.0e-5)
    normalized_features = (features - feature_mean) / feature_std
    normalized_targets = (targets - target_mean) / target_std

    model = DiagnosticCritic(features.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1.0e-5)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 101)
    final_loss = float("nan")
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(features.shape[0], generator=generator, device=device)
        for start in range(0, features.shape[0], batch_size):
            index = permutation[start : start + batch_size]
            prediction = model(normalized_features[index])
            loss = (prediction - normalized_targets[index]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    return FittedDiagnosticCritic(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        training_loss=final_loss,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.detach().float().cpu()
    target = target.detach().float().cpu()
    mse = float((prediction - target).square().mean())
    variance = float((target - target.mean()).square().mean())
    return {
        "mse": mse,
        "rmse": mse**0.5,
        "r2": 1.0 - mse / max(variance, 1.0e-12),
        "prediction_target_pearson": pearson_correlation(prediction, target) or 0.0,
    }


def counterfactual_q_deltas(
    critic: FittedDiagnosticCritic,
    states: torch.Tensor,
    actions: torch.Tensor,
    policy_means: torch.Tensor,
    *,
    n_agents: int,
) -> torch.Tensor:
    actual_features = torch.cat((states, actions), dim=-1).to(critic.feature_mean.device)
    actual_q = critic.predict(actual_features)
    deltas: list[torch.Tensor] = []
    actions_device = actions.to(critic.feature_mean.device)
    means_device = policy_means.to(critic.feature_mean.device)
    states_device = states.to(critic.feature_mean.device)
    for agent in range(n_agents):
        counterfactual = actions_device.clone()
        start = 2 * agent
        counterfactual[:, start : start + 2] = means_device[:, start : start + 2]
        counterfactual_q = critic.predict(torch.cat((states_device, counterfactual), dim=-1))
        deltas.append(actual_q - counterfactual_q)
    return torch.stack(deltas, dim=1).cpu()


def _correlation_pair(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | None]:
    return {
        "pearson": pearson_correlation(left, right),
        "rank": rank_correlation(left, right),
    }


def analyze_joint_action_critic_feasibility(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    train_num_envs: int = 128,
    validation_num_envs: int = 64,
    train_steps: int = 480,
    validation_steps: int = 480,
    horizon: int = 16,
    train_seed: int = 14023,
    validation_seeds: tuple[int, ...] = (15023, 16023),
    model_seeds: tuple[int, ...] = (7, 17, 29),
    exploration_multiplier: float = 1.0,
    epochs: int = 30,
    batch_size: int = 4096,
    learning_rate: float = 3.0e-4,
    hidden_dim: int = 128,
    run_dir: str | Path | None = None,
) -> dict:
    checkpoint_path = Path(checkpoint)
    checkpoint_data = torch.load(checkpoint_path, map_location=torch.device(device))
    actor_digest_before = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])

    train_dataset, act, probe_obs, probe_action_before = _collect_rollout_dataset(
        config=config,
        checkpoint_data=checkpoint_data,
        device=device,
        num_envs=train_num_envs,
        steps=train_steps,
        seed=train_seed,
        horizon=horizon,
        exploration_multiplier=exploration_multiplier,
    )
    validation_datasets: dict[int, RolloutDataset] = {}
    for seed in validation_seeds:
        validation_datasets[seed], _, _, _ = _collect_rollout_dataset(
            config=config,
            checkpoint_data=checkpoint_data,
            device=device,
            num_envs=validation_num_envs,
            steps=validation_steps,
            seed=seed,
            horizon=horizon,
            exploration_multiplier=exploration_multiplier,
        )

    training_state = train_dataset.states.to(device)
    training_joint = torch.cat((train_dataset.states, train_dataset.actions), dim=-1).to(device)
    training_target = train_dataset.targets.to(device)
    fitted_pairs: list[tuple[FittedDiagnosticCritic, FittedDiagnosticCritic]] = []
    for seed in model_seeds:
        state_critic = fit_diagnostic_critic(
            training_state,
            training_target,
            device=torch.device(device),
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
        )
        joint_critic = fit_diagnostic_critic(
            training_joint,
            training_target,
            device=torch.device(device),
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
        )
        fitted_pairs.append((state_critic, joint_critic))

    validation_results: dict[str, Any] = {}
    mse_improvements: list[float] = []
    safety_rank_correlations: list[float] = []
    for validation_seed, dataset in validation_datasets.items():
        replica_results: list[dict[str, Any]] = []
        for model_seed, (state_critic, joint_critic) in zip(
            model_seeds, fitted_pairs, strict=True
        ):
            state_prediction = state_critic.predict(dataset.states.to(device)).cpu()
            joint_features = torch.cat((dataset.states, dataset.actions), dim=-1).to(device)
            joint_prediction = joint_critic.predict(joint_features).cpu()
            state_metrics = regression_metrics(state_prediction, dataset.targets)
            joint_metrics = regression_metrics(joint_prediction, dataset.targets)
            improvement = (state_metrics["mse"] - joint_metrics["mse"]) / max(
                state_metrics["mse"], 1.0e-12
            )
            delta_q = counterfactual_q_deltas(
                joint_critic,
                dataset.states,
                dataset.actions,
                dataset.policy_means,
                n_agents=dataset.conflict_involvement.shape[1],
            )
            conflict_correlation = _correlation_pair(
                delta_q.flatten(), dataset.conflict_involvement.flatten()
            )
            collision_correlation = _correlation_pair(
                delta_q.flatten(), dataset.collision_involvement.flatten()
            )
            nearest_correlation = _correlation_pair(
                delta_q.flatten(), dataset.nearest_distance_change.flatten()
            )
            risk_correlation = _correlation_pair(
                delta_q.flatten(), dataset.relative_path_risk.flatten()
            )
            safety_ranks = [
                value
                for value in (
                    conflict_correlation["rank"],
                    collision_correlation["rank"],
                    nearest_correlation["rank"],
                )
                if value is not None
            ]
            max_safety_rank = max((abs(value) for value in safety_ranks), default=0.0)
            mse_improvements.append(improvement)
            safety_rank_correlations.append(max_safety_rank)
            replica_results.append(
                {
                    "model_seed": model_seed,
                    "state_only": state_metrics,
                    "joint_action": joint_metrics,
                    "mse_improvement_fraction": improvement,
                    "counterfactual_delta_q": {
                        "abs_mean": float(delta_q.abs().mean()),
                        "std": float(delta_q.std()),
                        "vs_predicted_conflict_involvement": conflict_correlation,
                        "vs_collision_involvement": collision_correlation,
                        "vs_nearest_distance_change": nearest_correlation,
                        "vs_relative_path_risk": risk_correlation,
                        "max_abs_safety_rank_correlation": max_safety_rank,
                    },
                }
            )
        seed_improvements = [item["mse_improvement_fraction"] for item in replica_results]
        seed_safety = [
            item["counterfactual_delta_q"]["max_abs_safety_rank_correlation"]
            for item in replica_results
        ]
        validation_results[str(validation_seed)] = {
            "samples": dataset.samples,
            "communication": dataset.communication,
            "target_mean": float(dataset.targets.mean()),
            "target_std": float(dataset.targets.std()),
            "conflict_involvement_rate": float(
                (dataset.conflict_involvement > 0).float().mean()
            ),
            "collision_involvement_rate": float(
                (dataset.collision_involvement > 0).float().mean()
            ),
            "mean_mse_improvement_fraction": sum(seed_improvements) / len(seed_improvements),
            "mean_max_abs_safety_rank_correlation": sum(seed_safety) / len(seed_safety),
            "replicas": replica_results,
        }

    with torch.no_grad():
        probe_action_after = act(probe_obs)
    actor_digest_after = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
    actor_action_change = float((probe_action_after - probe_action_before).abs().amax().cpu())
    mean_mse_improvement = sum(mse_improvements) / len(mse_improvements)
    min_seed_mse_improvement = min(
        value["mean_mse_improvement_fraction"] for value in validation_results.values()
    )
    mean_safety_rank = sum(safety_rank_correlations) / len(safety_rank_correlations)
    min_seed_safety_rank = min(
        value["mean_max_abs_safety_rank_correlation"]
        for value in validation_results.values()
    )
    checks = {
        "joint_action_mse_improves_15pct": mean_mse_improvement >= 0.15,
        "every_validation_seed_improves_15pct": min_seed_mse_improvement >= 0.15,
        "counterfactual_safety_rank_ge_0_30": mean_safety_rank >= 0.30,
        "every_validation_seed_safety_rank_ge_0_30": min_seed_safety_rank >= 0.30,
        "actor_parameters_unchanged": actor_digest_before == actor_digest_after,
        "actor_outputs_unchanged": actor_action_change == 0.0,
    }
    result: dict[str, Any] = {
        "status": "feasible_for_c1" if all(checks.values()) else "stop_before_c1",
        "experiment": EXPERIMENT_ID,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "device": device,
        "collection": {
            "horizon": horizon,
            "gamma": float((load_yaml(config).get("algorithm") or {}).get("gamma", 0.99)),
            "exploration_multiplier": exploration_multiplier,
            "train_num_envs": train_num_envs,
            "train_steps": train_steps,
            "validation_num_envs": validation_num_envs,
            "validation_steps": validation_steps,
            "policy_std": _policy_std(
                checkpoint_data,
                train_dataset.conflict_involvement.shape[1],
                torch.device("cpu"),
            )[0].tolist(),
            "train_seed": train_seed,
            "train_samples": train_dataset.samples,
            "validation_seeds": list(validation_seeds),
            "model_seeds": list(model_seeds),
            "train_communication": train_dataset.communication,
        },
        "models": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "state_only_parameter_count": fitted_pairs[0][0].parameter_count,
            "joint_action_parameter_count": fitted_pairs[0][1].parameter_count,
        },
        "validation": validation_results,
        "aggregate": {
            "mean_mse_improvement_fraction": mean_mse_improvement,
            "minimum_seed_mse_improvement_fraction": min_seed_mse_improvement,
            "mean_max_abs_safety_rank_correlation": mean_safety_rank,
            "minimum_seed_max_abs_safety_rank_correlation": min_seed_safety_rank,
        },
        "invariance": {
            "actor_digest_before": actor_digest_before,
            "actor_digest_after": actor_digest_after,
            "actor_probe_action_max_abs_change": actor_action_change,
        },
        "checks": checks,
        "decision": (
            "allow_c1_planning_only" if all(checks.values()) else "stop_counterfactual_critic_direction"
        ),
    }

    if run_dir is None:
        run_dir_path = (
            ROOT
            / "outputs"
            / "runs"
            / EXPERIMENT_ID
            / "frozen_exp125_relative_quintic_seed23"
        )
    else:
        run_dir_path = Path(run_dir)
        if not run_dir_path.is_absolute():
            run_dir_path = ROOT / run_dir_path
    metrics_dir = run_dir_path / "metrics"
    artifacts_dir = run_dir_path / "artifacts"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "counterfactual_critic_feasibility.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    diagnostic_payload = {
        "diagnostic_only": True,
        "deployable_policy": False,
        "state_only": [pair[0].artifact_state() for pair in fitted_pairs],
        "joint_action": [pair[1].artifact_state() for pair in fitted_pairs],
        "model_seeds": list(model_seeds),
        "model_hyperparameters": {
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        "collection": result["collection"],
        "source_checkpoint": str(checkpoint),
    }
    artifact_path = artifacts_dir / "diagnostic_critics.pt"
    torch.save(diagnostic_payload, artifact_path)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "producer": "scripts/analyze_joint_action_critic_feasibility.py",
        "status": result["status"],
        "source_checkpoint": str(checkpoint),
        "artifacts": {
            "metrics": str(metrics_path.relative_to(ROOT)),
            "diagnostic_models": str(artifact_path.relative_to(ROOT)),
        },
    }
    (run_dir_path / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    result["artifact"] = str(metrics_path)
    return result


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-num-envs", type=int, default=128)
    parser.add_argument("--validation-num-envs", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=480)
    parser.add_argument("--validation-steps", type=int, default=480)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--train-seed", type=int, default=14023)
    parser.add_argument("--validation-seeds", type=_parse_int_tuple, default=(15023, 16023))
    parser.add_argument("--model-seeds", type=_parse_int_tuple, default=(7, 17, 29))
    parser.add_argument("--exploration-multiplier", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = analyze_joint_action_critic_feasibility(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        train_num_envs=args.train_num_envs,
        validation_num_envs=args.validation_num_envs,
        train_steps=args.train_steps,
        validation_steps=args.validation_steps,
        horizon=args.horizon,
        train_seed=args.train_seed,
        validation_seeds=args.validation_seeds,
        model_seeds=args.model_seeds,
        exploration_multiplier=args.exploration_multiplier,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
