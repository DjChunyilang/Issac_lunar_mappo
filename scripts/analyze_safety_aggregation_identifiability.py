#!/usr/bin/env python
"""Compare mean and worst-pair aggregation of the existing safety reward."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from analyze_joint_action_critic_feasibility import _parse_int_tuple, tensor_dict_digest
from analyze_reward_component_identifiability import (
    component_metrics,
    fit_multi_output_regressor,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from play import _load_policy_players


EXPERIMENT_ID = "exp138_safety_aggregation_identifiability"
TARGET_NAMES = ("safety_mean", "safety_worst_pair")


def safety_aggregation_targets(
    nearest_distance: torch.Tensor,
    collision: torch.Tensor,
    *,
    near_distance: float,
    near_coefficient: float,
    collision_coefficient: float,
    safety_weight: float,
) -> dict[str, torch.Tensor]:
    """Return current and candidate team safety terms from identical signals."""

    if nearest_distance.ndim != 2:
        raise ValueError("nearest_distance must have shape [environment, agent].")
    if collision.shape != nearest_distance.shape[:1]:
        raise ValueError("collision must have shape [environment].")
    per_agent_gap = torch.relu(float(near_distance) - nearest_distance)
    collision_penalty = collision.to(nearest_distance.dtype) * float(
        collision_coefficient
    )
    mean_penalty = per_agent_gap.mean(dim=-1)
    worst_penalty = per_agent_gap.amax(dim=-1)
    return {
        "safety_mean": -float(safety_weight)
        * (float(near_coefficient) * mean_penalty + collision_penalty),
        "safety_worst_pair": -float(safety_weight)
        * (float(near_coefficient) * worst_penalty + collision_penalty),
    }


def aggregation_gate(
    aggregate: dict[str, dict[str, float]],
    per_seed: dict[str, dict[str, Any]],
    *,
    actor_unchanged: bool,
    reconstruction_error: float,
) -> dict[str, Any]:
    deltas = {
        seed: values["components"]["safety_worst_pair"][
            "mean_mse_improvement_fraction"
        ]
        - values["components"]["safety_mean"]["mean_mse_improvement_fraction"]
        for seed, values in per_seed.items()
    }
    checks = {
        "worst_pair_mean_action_gain_ge_0_15": aggregate["safety_worst_pair"][
            "mean_mse_improvement_fraction"
        ]
        >= 0.15,
        "worst_pair_every_seed_action_gain_ge_0_15": aggregate[
            "safety_worst_pair"
        ]["minimum_seed_mse_improvement_fraction"]
        >= 0.15,
        "worst_pair_every_seed_gain_delta_ge_0_10": min(deltas.values()) >= 0.10,
        "worst_pair_every_seed_active_rate_ge_0_05": min(
            values["active_rate"]["safety_worst_pair"]
            for values in per_seed.values()
        )
        >= 0.05,
        "current_mean_target_reconstructed": reconstruction_error <= 1.0e-6,
        "actor_unchanged": actor_unchanged,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "worst_pair_action_gain": 0.15,
            "gain_delta_over_mean": 0.10,
            "target_active_rate": 0.05,
            "target_reconstruction_max_abs_error": 1.0e-6,
        },
        "per_seed_gain_delta": deltas,
    }


def _collect(
    *,
    config: str | Path,
    checkpoint_data: dict[str, Any],
    device: str,
    num_envs: int,
    steps: int,
    seed: int,
    exploration_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    raw_cfg = load_yaml(config)
    cfg = cfg_from_experiment(config)
    cfg.simulation.device = device
    cfg.simulation.num_envs = num_envs
    cfg.seed = seed
    metadata = checkpoint_data.get("metadata") or {}
    if cfg.initial_state.curriculum_enabled:
        cfg.initial_state.progress_timestep_override = int(metadata.get("timesteps", 0))
    env = MultiRoverGatheringCore(cfg)
    act, _ = _load_policy_players(checkpoint_data, cfg, env.device, raw_cfg=raw_cfg)
    actor_obs, critic_state = env.get_observations()
    policy_std = (
        checkpoint_data["rover_0"]["policy"]["log_std_parameter"]
        .detach()
        .to(env.device)
        .exp()
        .view(1, 1, 2)
    )
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed + 7919)
    states: list[torch.Tensor] = []
    actions_sequence: list[torch.Tensor] = []
    targets_sequence: list[torch.Tensor] = []
    reconstruction_error = 0.0
    active_counts = {name: 0 for name in TARGET_NAMES}
    sample_count = 0

    for _ in range(steps):
        with torch.no_grad():
            mean = act(actor_obs)
            action = (
                mean
                + float(exploration_multiplier)
                * policy_std
                * torch.randn(
                    mean.shape,
                    generator=generator,
                    device=env.device,
                    dtype=mean.dtype,
                )
            ).clamp(-1.0, 1.0)
            states.append(critic_state.detach().clone())
            actions_sequence.append(action.reshape(num_envs, -1).detach().clone())
            output = env.step(action)
            safety_targets = safety_aggregation_targets(
                output.info["metrics"].nearest_neighbor_distance,
                output.info["done"].collision,
                near_distance=float(cfg.safety.near_distance),
                near_coefficient=float(cfg.reward_coefficients.near_distance),
                collision_coefficient=float(
                    cfg.reward_coefficients.inter_agent_collision
                ),
                safety_weight=float(cfg.reward_weights.safety),
            )
            existing = (
                output.info["reward_terms"].safety
                * float(cfg.reward_weights.safety)
            )
            reconstruction_error = max(
                reconstruction_error,
                float((safety_targets["safety_mean"] - existing).abs().amax()),
            )
            target = torch.stack(
                [safety_targets[name] for name in TARGET_NAMES], dim=-1
            )
            targets_sequence.append(target.detach().clone())
            for index, name in enumerate(TARGET_NAMES):
                active_counts[name] += int((target[:, index].abs() > 1.0e-8).sum())
            sample_count += num_envs
            actor_obs = output.actor_obs
            critic_state = output.critic_state

    def flatten(values: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(values)
        return stacked.reshape(-1, stacked.shape[-1])

    evidence = {
        "samples": sample_count,
        "current_target_reconstruction_max_abs_error": reconstruction_error,
        "active_rate": {
            name: active_counts[name] / max(sample_count, 1) for name in TARGET_NAMES
        },
    }
    return flatten(states), flatten(actions_sequence), flatten(targets_sequence), evidence


def analyze_safety_aggregation_identifiability(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    train_num_envs: int = 128,
    validation_num_envs: int = 64,
    steps: int = 480,
    train_seed: int = 30023,
    validation_seeds: tuple[int, ...] = (31023, 32023),
    model_seeds: tuple[int, ...] = (7, 17, 29),
    exploration_multiplier: float = 1.0,
    epochs: int = 30,
    batch_size: int = 4096,
    learning_rate: float = 3.0e-4,
    hidden_dim: int = 128,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    checkpoint_data = torch.load(checkpoint, map_location=torch_device)
    actor_digest_before = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
    train_states, train_actions, train_targets, train_evidence = _collect(
        config=config,
        checkpoint_data=checkpoint_data,
        device=device,
        num_envs=train_num_envs,
        steps=steps,
        seed=train_seed,
        exploration_multiplier=exploration_multiplier,
    )
    validations = {
        seed: _collect(
            config=config,
            checkpoint_data=checkpoint_data,
            device=device,
            num_envs=validation_num_envs,
            steps=steps,
            seed=seed,
            exploration_multiplier=exploration_multiplier,
        )
        for seed in validation_seeds
    }
    fitted = []
    train_joint = torch.cat((train_states, train_actions), dim=-1)
    for seed in model_seeds:
        fitted.append(
            (
                fit_multi_output_regressor(
                    train_states,
                    train_targets,
                    device=torch_device,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    hidden_dim=hidden_dim,
                ),
                fit_multi_output_regressor(
                    train_joint,
                    train_targets,
                    device=torch_device,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    hidden_dim=hidden_dim,
                ),
            )
        )

    per_seed: dict[str, Any] = {}
    improvements = {name: [] for name in TARGET_NAMES}
    max_reconstruction_error = train_evidence[
        "current_target_reconstruction_max_abs_error"
    ]
    for validation_seed, (states, actions, targets, evidence) in validations.items():
        replicas = []
        seed_components: dict[str, Any] = {}
        max_reconstruction_error = max(
            max_reconstruction_error,
            evidence["current_target_reconstruction_max_abs_error"],
        )
        for model_seed, (state_model, joint_model) in zip(model_seeds, fitted, strict=True):
            state_prediction = state_model.predict(states.to(torch_device)).cpu()
            joint_prediction = joint_model.predict(
                torch.cat((states, actions), dim=-1).to(torch_device)
            ).cpu()
            components = {}
            for index, name in enumerate(TARGET_NAMES):
                state_metrics = component_metrics(state_prediction[:, index], targets[:, index])
                joint_metrics = component_metrics(joint_prediction[:, index], targets[:, index])
                gain = (state_metrics["mse"] - joint_metrics["mse"]) / max(
                    state_metrics["mse"], 1.0e-12
                )
                components[name] = {
                    "state_only": state_metrics,
                    "joint_action": joint_metrics,
                    "mse_improvement_fraction": gain,
                }
            replicas.append({"model_seed": model_seed, "components": components})
        for name in TARGET_NAMES:
            values = [row["components"][name]["mse_improvement_fraction"] for row in replicas]
            mean_value = sum(values) / len(values)
            improvements[name].append(mean_value)
            seed_components[name] = {
                "mean_mse_improvement_fraction": mean_value,
                "minimum_replica_mse_improvement_fraction": min(values),
            }
        per_seed[str(validation_seed)] = {
            "samples": evidence["samples"],
            "active_rate": evidence["active_rate"],
            "components": seed_components,
            "replicas": replicas,
        }

    aggregate = {
        name: {
            "mean_mse_improvement_fraction": sum(values) / len(values),
            "minimum_seed_mse_improvement_fraction": min(values),
        }
        for name, values in improvements.items()
    }
    actor_digest_after = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
    gate = aggregation_gate(
        aggregate,
        per_seed,
        actor_unchanged=actor_digest_before == actor_digest_after,
        reconstruction_error=max_reconstruction_error,
    )
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "status": (
            "allow_single_worst_pair_aggregation_plan"
            if gate["passed"]
            else "stop_safety_aggregation_change"
        ),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "collection": {
            "device": device,
            "steps": steps,
            "episode_coverage_seconds": steps * 0.2,
            "train_num_envs": train_num_envs,
            "validation_num_envs": validation_num_envs,
            "train_seed": train_seed,
            "validation_seeds": list(validation_seeds),
            "model_seeds": list(model_seeds),
            "exploration_multiplier": exploration_multiplier,
        },
        "method": {
            "current": "mean per-agent nearest-distance gap",
            "candidate": "maximum per-agent nearest-distance gap",
            "same_near_distance": True,
            "same_collision_term": True,
            "same_coefficients": True,
            "reward_or_execution_modified": False,
            "training_or_optimizer_modified": False,
        },
        "train_evidence": train_evidence,
        "validation": per_seed,
        "aggregate_components": aggregate,
        "gate": gate,
        "invariance": {
            "actor_digest_before": actor_digest_before,
            "actor_digest_after": actor_digest_after,
        },
        "decision": (
            "pre_register_one_4m_aggregation_only_screen"
            if gate["passed"]
            else "do_not_change_safety_aggregation"
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
    artifacts_dir = run_dir_path / "artifacts"
    config_dir = run_dir_path / "config"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_source = Path(config)
    if not config_source.is_absolute():
        config_source = ROOT / config_source
    config_snapshot = config_dir / "experiment.yaml"
    config_snapshot.write_text(config_source.read_text(encoding="utf-8"), encoding="utf-8")
    metrics_path = metrics_dir / "safety_aggregation_identifiability.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    models_path = artifacts_dir / "diagnostic_regressors.pt"
    torch.save(
        {
            "diagnostic_only": True,
            "deployable_policy": False,
            "target_order": list(TARGET_NAMES),
            "model_seeds": list(model_seeds),
            "state_only": [row[0].artifact_state() for row in fitted],
            "joint_action": [row[1].artifact_state() for row in fitted],
        },
        models_path,
    )
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "producer": "scripts/analyze_safety_aggregation_identifiability.py",
        "status": result["status"],
        "source_checkpoint": str(checkpoint),
        "artifacts": {
            "config": str(config_snapshot.relative_to(ROOT)),
            "metrics": str(metrics_path.relative_to(ROOT)),
            "diagnostic_models": str(models_path.relative_to(ROOT)),
        },
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
    parser.add_argument("--train-num-envs", type=int, default=128)
    parser.add_argument("--validation-num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--train-seed", type=int, default=30023)
    parser.add_argument("--validation-seeds", type=_parse_int_tuple, default=(31023, 32023))
    parser.add_argument("--model-seeds", type=_parse_int_tuple, default=(7, 17, 29))
    parser.add_argument("--exploration-multiplier", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = analyze_safety_aggregation_identifiability(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        train_num_envs=args.train_num_envs,
        validation_num_envs=args.validation_num_envs,
        steps=args.steps,
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
