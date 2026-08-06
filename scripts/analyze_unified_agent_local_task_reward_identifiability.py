#!/usr/bin/env python
"""Audit a unified per-rover gather/terrain/safety target offline.

The source B0 Actor is frozen. Matched regressors compare local observation
against local observation plus the rover's own action. This script cannot
authorize training; passing its gate only permits writing a subsequent plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from _common import ROOT
from analyze_actor_gradient_conflicts import _collect_seed_dataset
from analyze_joint_action_critic_feasibility import _parse_int_tuple, tensor_dict_digest
from analyze_reward_component_identifiability import (
    FittedMultiOutputRegressor,
    component_metrics,
    fit_multi_output_regressor,
)


EXPERIMENT_ID = "exp145_unified_agent_local_task_reward_identifiability"
RAW_TARGET_NAMES = ("local_gather_progress", "local_terrain_result", "local_safety_progress")
TARGET_NAMES = (*RAW_TARGET_NAMES, "unified_local_task")
DEFAULT_SOURCE_RUN = (
    ROOT
    / "outputs/runs/exp125_decentralized_tiered_b0_pure_rl"
    / "b0_screen_seed23_4m_relative_quintic"
)


def raw_targets(dataset: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the three pre-registered, unnormalized per-rover outcomes."""

    return torch.cat(
        (
            dataset["local_gather_credits"],
            dataset["terrain_raw_credits"],
            dataset["safety_raw_credits"],
        ),
        dim=-1,
    )


def fit_target_normalization(targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if targets.ndim != 2 or targets.shape[1] != len(RAW_TARGET_NAMES):
        raise ValueError("Raw local-task targets must have shape [sample, 3].")
    return targets.mean(dim=0), targets.std(dim=0).clamp_min(1.0e-8)


def build_targets(
    raw: torch.Tensor,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
) -> torch.Tensor:
    """Append the equal-weight normalized unified target to raw components."""

    if raw.ndim != 2 or raw.shape[1] != len(RAW_TARGET_NAMES):
        raise ValueError("Raw local-task targets must have shape [sample, 3].")
    unified = ((raw - normalization_mean) / normalization_std).sum(
        dim=-1, keepdim=True
    )
    return torch.cat((raw, unified), dim=-1)


def target_distribution(raw: torch.Tensor) -> dict[str, Any]:
    """Summarize coverage used by the pre-registered identifiability gates."""

    result: dict[str, Any] = {}
    for index, name in enumerate(RAW_TARGET_NAMES):
        values = raw[:, index]
        result[name] = {
            "mean": float(values.mean().cpu()),
            "std": float(values.std().cpu()),
            "active_rate": float((values.abs() > 1.0e-8).float().mean().cpu()),
            "positive_rate": float((values > 1.0e-8).float().mean().cpu()),
            "negative_rate": float((values < -1.0e-8).float().mean().cpu()),
        }
    return result


def unified_local_task_gate(
    *,
    aggregate: dict[str, dict[str, float]],
    validation: dict[str, dict[str, Any]],
    actor_parameters_unchanged: bool,
    actor_output_change: float,
    actor_observation_dim: int,
    own_action_input_dim: int,
) -> dict[str, Any]:
    """Apply exp145's pre-registered cross-seed stop gate."""

    checks: dict[str, bool] = {}
    for name in RAW_TARGET_NAMES:
        checks[f"{name}_mean_action_gain_ge_0_15"] = (
            aggregate[name]["mean_mse_improvement_fraction"] >= 0.15
        )
        checks[f"{name}_every_seed_action_gain_ge_0_15"] = (
            aggregate[name]["minimum_seed_mse_improvement_fraction"] >= 0.15
        )
        checks[f"{name}_every_seed_std_gt_1e_4"] = min(
            row["target_distribution"][name]["std"] for row in validation.values()
        ) > 1.0e-4
    checks["unified_mean_action_gain_ge_0_20"] = (
        aggregate["unified_local_task"]["mean_mse_improvement_fraction"] >= 0.20
    )
    checks["unified_every_seed_action_gain_ge_0_20"] = (
        aggregate["unified_local_task"]["minimum_seed_mse_improvement_fraction"]
        >= 0.20
    )
    checks["safety_every_seed_active_rate_ge_0_08"] = min(
        row["target_distribution"]["local_safety_progress"]["active_rate"]
        for row in validation.values()
    ) >= 0.08
    checks["terrain_every_seed_positive_rate_ge_0_10"] = min(
        row["target_distribution"]["local_terrain_result"]["positive_rate"]
        for row in validation.values()
    ) >= 0.10
    checks["terrain_every_seed_negative_rate_ge_0_10"] = min(
        row["target_distribution"]["local_terrain_result"]["negative_rate"]
        for row in validation.values()
    ) >= 0.10
    checks.update(
        {
            "actor_parameters_unchanged": actor_parameters_unchanged,
            "actor_outputs_unchanged": actor_output_change == 0.0,
            "strict_decentralized_observation_dim_101": actor_observation_dim == 101,
            "only_own_two_dimensional_action_added": own_action_input_dim == 103,
        }
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "raw_component_mean_and_worst_seed_action_gain": 0.15,
            "unified_mean_and_worst_seed_action_gain": 0.20,
            "raw_component_std_strictly_greater_than": 1.0e-4,
            "safety_active_rate": 0.08,
            "terrain_positive_rate": 0.10,
            "terrain_negative_rate": 0.10,
        },
    }


def _fit_pairs(
    observations: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    *,
    model_seeds: tuple[int, ...],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
) -> list[tuple[FittedMultiOutputRegressor, FittedMultiOutputRegressor]]:
    observation_action = torch.cat((observations, actions), dim=-1)
    fitted = []
    for seed in model_seeds:
        fitted.append(
            (
                fit_multi_output_regressor(
                    observations,
                    targets,
                    device=device,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    hidden_dim=hidden_dim,
                ),
                fit_multi_output_regressor(
                    observation_action,
                    targets,
                    device=device,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    hidden_dim=hidden_dim,
                ),
            )
        )
    return fitted


def analyze_unified_agent_local_task_reward_identifiability(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    train_num_envs: int = 128,
    validation_num_envs: int = 64,
    steps: int = 480,
    rollout_length: int = 64,
    train_seed: int = 39023,
    validation_seeds: tuple[int, ...] = (40023, 41023),
    model_seeds: tuple[int, ...] = (7, 17, 29),
    epochs: int = 30,
    batch_size: int = 4096,
    learning_rate: float = 3.0e-4,
    hidden_dim: int = 128,
    run_name: str = "frozen_exp125_seed23",
) -> dict[str, Any]:
    torch_device = torch.device(device)
    checkpoint_data = torch.load(checkpoint, map_location=torch_device)
    actor_digest_before = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])

    train_dataset = _collect_seed_dataset(
        config=config,
        checkpoint_data=checkpoint_data,
        device=device,
        num_envs=train_num_envs,
        steps=steps,
        rollout_length=rollout_length,
        seed=train_seed,
        safety_credit_distance=0.72,
    )
    train_policy = train_dataset.pop("policy")
    probe_observations = train_dataset["observations"][:32].detach().clone()
    with torch.no_grad():
        probe_before = train_policy.compute(
            {"observations": probe_observations}, role="policy"
        )[0].detach().clone()

    validation_datasets: dict[int, dict[str, torch.Tensor]] = {}
    for seed in validation_seeds:
        dataset = _collect_seed_dataset(
            config=config,
            checkpoint_data=checkpoint_data,
            device=device,
            num_envs=validation_num_envs,
            steps=steps,
            rollout_length=rollout_length,
            seed=seed,
            safety_credit_distance=0.72,
        )
        dataset.pop("policy")
        validation_datasets[seed] = dataset

    train_raw = raw_targets(train_dataset)
    normalization_mean, normalization_std = fit_target_normalization(train_raw)
    train_targets = build_targets(train_raw, normalization_mean, normalization_std)
    train_observations = train_dataset["observations"]
    train_actions = train_dataset["actions"]
    fitted = _fit_pairs(
        train_observations,
        train_actions,
        train_targets,
        model_seeds=model_seeds,
        device=torch_device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
    )

    improvements = {name: [] for name in TARGET_NAMES}
    validation: dict[str, Any] = {}
    for validation_seed, dataset in validation_datasets.items():
        observations = dataset["observations"]
        observation_action = torch.cat((observations, dataset["actions"]), dim=-1)
        raw = raw_targets(dataset)
        targets = build_targets(raw, normalization_mean, normalization_std)
        replicas: list[dict[str, Any]] = []
        for model_seed, (observation_model, action_model) in zip(
            model_seeds, fitted, strict=True
        ):
            observation_prediction = observation_model.predict(
                observations.to(torch_device)
            ).cpu()
            action_prediction = action_model.predict(
                observation_action.to(torch_device)
            ).cpu()
            components: dict[str, Any] = {}
            for index, name in enumerate(TARGET_NAMES):
                observation_metrics = component_metrics(
                    observation_prediction[:, index], targets[:, index]
                )
                action_metrics = component_metrics(
                    action_prediction[:, index], targets[:, index]
                )
                gain = (
                    observation_metrics["mse"] - action_metrics["mse"]
                ) / max(observation_metrics["mse"], 1.0e-12)
                components[name] = {
                    "observation_only": observation_metrics,
                    "observation_and_own_action": action_metrics,
                    "mse_improvement_fraction": gain,
                }
            replicas.append({"model_seed": model_seed, "components": components})

        seed_components: dict[str, Any] = {}
        for name in TARGET_NAMES:
            values = [
                replica["components"][name]["mse_improvement_fraction"]
                for replica in replicas
            ]
            mean_value = sum(values) / len(values)
            improvements[name].append(mean_value)
            seed_components[name] = {
                "mean_mse_improvement_fraction": mean_value,
                "minimum_replica_mse_improvement_fraction": min(values),
            }
        validation[str(validation_seed)] = {
            "samples": int(observations.shape[0]),
            "target_distribution": target_distribution(raw),
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
    with torch.no_grad():
        probe_after = train_policy.compute(
            {"observations": probe_observations}, role="policy"
        )[0]
    actor_digest_after = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
    actor_output_change = float((probe_after - probe_before).abs().amax().cpu())
    action_input_dim = int(train_observations.shape[-1] + train_actions.shape[-1])
    gate = unified_local_task_gate(
        aggregate=aggregate,
        validation=validation,
        actor_parameters_unchanged=actor_digest_before == actor_digest_after,
        actor_output_change=actor_output_change,
        actor_observation_dim=int(train_observations.shape[-1]),
        own_action_input_dim=action_input_dim,
    )
    status = (
        "allow_agent_local_advantage_plan_only"
        if gate["passed"]
        else "stop_unified_agent_local_reward"
    )

    run_dir = ROOT / "outputs/runs" / EXPERIMENT_ID / run_name
    metrics_path = run_dir / "metrics/unified_local_task_identifiability.json"
    models_path = run_dir / "artifacts/diagnostic_regressors.pt"
    config_snapshot = run_dir / "config/experiment.yaml"
    for directory in (metrics_path.parent, models_path.parent, config_snapshot.parent):
        directory.mkdir(parents=True, exist_ok=True)
    source_config = Path(config)
    if not source_config.is_absolute():
        source_config = ROOT / source_config
    config_snapshot.write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")

    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "run": run_name,
        "status": status,
        "config": str(source_config),
        "checkpoint": str(checkpoint),
        "collection": {
            "device": device,
            "steps": steps,
            "episode_coverage_seconds": steps * 0.2,
            "rollout_length": rollout_length,
            "final_partial_rollout_steps": steps % rollout_length,
            "train_num_envs": train_num_envs,
            "validation_num_envs": validation_num_envs,
            "train_seed": train_seed,
            "validation_seeds": list(validation_seeds),
            "model_seeds": list(model_seeds),
            "train_actor_samples": int(train_observations.shape[0]),
        },
        "method": {
            "target_order": list(TARGET_NAMES),
            "observation_only_dim": int(train_observations.shape[-1]),
            "observation_and_own_action_dim": action_input_dim,
            "hidden_layers": [hidden_dim, hidden_dim],
            "activation": "ELU",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "safe_distance_m": 0.72,
            "terminal_transition_uses_pre_reset_position_snapshot": True,
            "other_agent_actions_used": False,
            "oracle_or_global_state_in_actor_input": False,
            "reward_or_execution_modified": False,
            "training_authorized": False,
            "four_million_screen_authorized": False,
        },
        "target_normalization": {
            "fit_on_training_data_only": True,
            "raw_target_order": list(RAW_TARGET_NAMES),
            "mean": normalization_mean.detach().cpu().tolist(),
            "std": normalization_std.detach().cpu().tolist(),
            "unified_definition": "sum((raw - training_mean) / training_std)",
        },
        "train_target_distribution": target_distribution(train_raw),
        "validation": validation,
        "aggregate_components": aggregate,
        "gate": gate,
        "invariance": {
            "actor_digest_before": actor_digest_before,
            "actor_digest_after": actor_digest_after,
            "actor_probe_output_max_abs_change": actor_output_change,
        },
        "decision": (
            "write_agent_local_advantage_training_plan_but_do_not_train"
            if gate["passed"]
            else "stop_agent_local_reward_direction_without_retuning_or_training"
        ),
        "artifacts": {
            "metrics": str(metrics_path.relative_to(ROOT)),
            "diagnostic_models": str(models_path.relative_to(ROOT)),
            "config": str(config_snapshot.relative_to(ROOT)),
        },
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(
        {
            "diagnostic_only": True,
            "deployable_policy": False,
            "target_order": list(TARGET_NAMES),
            "raw_target_normalization_mean": normalization_mean.detach().cpu(),
            "raw_target_normalization_std": normalization_std.detach().cpu(),
            "model_seeds": list(model_seeds),
            "observation_only": [pair[0].artifact_state() for pair in fitted],
            "observation_and_own_action": [pair[1].artifact_state() for pair in fitted],
        },
        models_path,
    )
    manifest = {
        "schema_version": 1,
        "generated_at": result["generated_at"],
        "experiment": EXPERIMENT_ID,
        "run": run_name,
        "producer": "scripts/analyze_unified_agent_local_task_reward_identifiability.py",
        "command": " ".join(sys.argv),
        "status": status,
        "source_checkpoint": str(checkpoint),
        "artifacts": result["artifacts"],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    suite_dir = ROOT / "outputs/runs" / EXPERIMENT_ID / "_suite"
    suite_metrics = suite_dir / "metrics/audit_summary.json"
    suite_metrics.parent.mkdir(parents=True, exist_ok=True)
    suite_metrics.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (suite_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                **manifest,
                "run": "_suite",
                "artifacts": {
                    "audit_summary": str(suite_metrics.relative_to(ROOT)),
                    "source_run_manifest": str(
                        (run_dir / "run_manifest.json").relative_to(ROOT)
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(DEFAULT_SOURCE_RUN / "config/experiment.yaml")
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_SOURCE_RUN / "checkpoints/ppo_timestep_002048.pt"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-num-envs", type=int, default=128)
    parser.add_argument("--validation-num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--train-seed", type=int, default=39023)
    parser.add_argument(
        "--validation-seeds", type=_parse_int_tuple, default=(40023, 41023)
    )
    parser.add_argument("--model-seeds", type=_parse_int_tuple, default=(7, 17, 29))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--run-name", default="frozen_exp125_seed23")
    args = parser.parse_args()
    result = analyze_unified_agent_local_task_reward_identifiability(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        train_num_envs=args.train_num_envs,
        validation_num_envs=args.validation_num_envs,
        steps=args.steps,
        rollout_length=args.rollout_length,
        train_seed=args.train_seed,
        validation_seeds=args.validation_seeds,
        model_seeds=args.model_seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        run_name=args.run_name,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
