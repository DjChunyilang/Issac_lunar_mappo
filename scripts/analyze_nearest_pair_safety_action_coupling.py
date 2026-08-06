#!/usr/bin/env python
"""Audit whether nearest-pair actions jointly identify local safety progress."""

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


EXPERIMENT_ID = "exp146_nearest_pair_safety_action_coupling"
MODEL_NAMES = ("observation_only", "own_action", "neighbor_action", "pair_action")
DEFAULT_SOURCE_RUN = (
    ROOT
    / "outputs/runs/exp125_decentralized_tiered_b0_pure_rl"
    / "b0_screen_seed23_4m_relative_quintic"
)


def pair_safety_gate(
    *,
    aggregate: dict[str, dict[str, float]],
    validation: dict[str, dict[str, Any]],
    actor_parameters_unchanged: bool,
    actor_output_change: float,
    actor_observation_dim: int,
) -> dict[str, Any]:
    checks = {
        "pair_vs_observation_every_seed_ge_0_25": aggregate["pair_vs_observation"][
            "minimum_seed"
        ]
        >= 0.25,
        "own_given_neighbor_every_seed_ge_0_15": aggregate["own_given_neighbor"][
            "minimum_seed"
        ]
        >= 0.15,
        "neighbor_given_own_every_seed_ge_0_15": aggregate["neighbor_given_own"][
            "minimum_seed"
        ]
        >= 0.15,
        "neighbor_shuffle_degradation_every_seed_ge_0_10": aggregate[
            "neighbor_shuffle_degradation"
        ]["minimum_seed"]
        >= 0.10,
        "safety_every_seed_std_gt_1e_4": min(
            row["target_distribution"]["std"] for row in validation.values()
        )
        > 1.0e-4,
        "safety_every_seed_active_rate_ge_0_08": min(
            row["target_distribution"]["active_rate"] for row in validation.values()
        )
        >= 0.08,
        "actor_parameters_unchanged": actor_parameters_unchanged,
        "actor_outputs_unchanged": actor_output_change == 0.0,
        "strict_decentralized_observation_dim_101": actor_observation_dim == 101,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "pair_vs_observation": 0.25,
            "own_given_neighbor": 0.15,
            "neighbor_given_own": 0.15,
            "neighbor_shuffle_degradation": 0.10,
            "safety_std_strictly_greater_than": 1.0e-4,
            "safety_active_rate": 0.08,
        },
    }


def _features(dataset: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    observations = dataset["observations"]
    own_actions = dataset["actions"]
    neighbor_actions = dataset["nearest_neighbor_actions"]
    return {
        "observation_only": observations,
        "own_action": torch.cat((observations, own_actions), dim=-1),
        "neighbor_action": torch.cat((observations, neighbor_actions), dim=-1),
        "pair_action": torch.cat((observations, own_actions, neighbor_actions), dim=-1),
    }


def _fit_models(
    dataset: dict[str, torch.Tensor],
    *,
    model_seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
) -> dict[str, FittedMultiOutputRegressor]:
    targets = dataset["safety_raw_credits"]
    return {
        name: fit_multi_output_regressor(
            features,
            targets,
            device=device,
            seed=model_seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
        )
        for name, features in _features(dataset).items()
    }


def _fraction(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1.0e-12)


def analyze_nearest_pair_safety_action_coupling(
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
    policy = train_dataset.pop("policy")
    probe_observations = train_dataset["observations"][:32].detach().clone()
    with torch.no_grad():
        probe_before = policy.compute(
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

    fitted = [
        _fit_models(
            train_dataset,
            model_seed=seed,
            device=torch_device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
        )
        for seed in model_seeds
    ]
    metric_names = (
        "pair_vs_observation",
        "own_given_neighbor",
        "neighbor_given_own",
        "neighbor_shuffle_degradation",
    )
    per_seed_values = {name: [] for name in metric_names}
    validation: dict[str, Any] = {}
    for validation_seed, dataset in validation_datasets.items():
        targets = dataset["safety_raw_credits"]
        features = _features(dataset)
        generator = torch.Generator(device=targets.device)
        generator.manual_seed(validation_seed + 146)
        permutation = torch.randperm(
            targets.shape[0], generator=generator, device=targets.device
        )
        shuffled_pair_features = torch.cat(
            (
                dataset["observations"],
                dataset["actions"],
                dataset["nearest_neighbor_actions"][permutation],
            ),
            dim=-1,
        )
        replicas: list[dict[str, Any]] = []
        for model_seed, models in zip(model_seeds, fitted, strict=True):
            model_metrics = {
                name: component_metrics(
                    model.predict(features[name].to(torch_device)).cpu()[:, 0],
                    targets[:, 0],
                )
                for name, model in models.items()
            }
            shuffled_metrics = component_metrics(
                models["pair_action"]
                .predict(shuffled_pair_features.to(torch_device))
                .cpu()[:, 0],
                targets[:, 0],
            )
            mse = {name: row["mse"] for name, row in model_metrics.items()}
            gains = {
                "pair_vs_observation": _fraction(
                    mse["observation_only"] - mse["pair_action"],
                    mse["observation_only"],
                ),
                "own_given_neighbor": _fraction(
                    mse["neighbor_action"] - mse["pair_action"],
                    mse["neighbor_action"],
                ),
                "neighbor_given_own": _fraction(
                    mse["own_action"] - mse["pair_action"], mse["own_action"]
                ),
                "neighbor_shuffle_degradation": _fraction(
                    shuffled_metrics["mse"] - mse["pair_action"], mse["pair_action"]
                ),
            }
            replicas.append(
                {
                    "model_seed": model_seed,
                    "models": model_metrics,
                    "shuffled_neighbor_pair_model": shuffled_metrics,
                    "gains": gains,
                }
            )
        seed_gains: dict[str, Any] = {}
        for name in metric_names:
            values = [row["gains"][name] for row in replicas]
            mean_value = sum(values) / len(values)
            per_seed_values[name].append(mean_value)
            seed_gains[name] = {
                "mean": mean_value,
                "minimum_replica": min(values),
            }
        target = targets[:, 0]
        validation[str(validation_seed)] = {
            "samples": int(target.shape[0]),
            "target_distribution": {
                "mean": float(target.mean().cpu()),
                "std": float(target.std().cpu()),
                "active_rate": float((target.abs() > 1.0e-8).float().mean().cpu()),
            },
            "gains": seed_gains,
            "replicas": replicas,
        }

    aggregate = {
        name: {
            "mean": sum(values) / len(values),
            "minimum_seed": min(values),
        }
        for name, values in per_seed_values.items()
    }
    with torch.no_grad():
        probe_after = policy.compute(
            {"observations": probe_observations}, role="policy"
        )[0]
    actor_digest_after = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
    actor_output_change = float((probe_after - probe_before).abs().amax().cpu())
    observation_dim = int(train_dataset["observations"].shape[-1])
    gate = pair_safety_gate(
        aggregate=aggregate,
        validation=validation,
        actor_parameters_unchanged=actor_digest_before == actor_digest_after,
        actor_output_change=actor_output_change,
        actor_observation_dim=observation_dim,
    )
    status = (
        "allow_pair_local_safety_critic_plan_only"
        if gate["passed"]
        else "stop_pair_local_safety_credit_direction"
    )

    run_dir = ROOT / "outputs/runs" / EXPERIMENT_ID / run_name
    metrics_path = run_dir / "metrics/nearest_pair_safety_action_coupling.json"
    models_path = run_dir / "artifacts/diagnostic_regressors.pt"
    config_snapshot = run_dir / "config/experiment.yaml"
    for directory in (metrics_path.parent, models_path.parent, config_snapshot.parent):
        directory.mkdir(parents=True, exist_ok=True)
    source_config = Path(config)
    if not source_config.is_absolute():
        source_config = ROOT / source_config
    config_snapshot.write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
    generated_at = datetime.now(timezone.utc).isoformat()
    result = {
        "schema_version": 1,
        "generated_at": generated_at,
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
            "train_num_envs": train_num_envs,
            "validation_num_envs": validation_num_envs,
            "train_seed": train_seed,
            "validation_seeds": list(validation_seeds),
            "model_seeds": list(model_seeds),
            "train_actor_samples": int(train_dataset["observations"].shape[0]),
        },
        "method": {
            "model_inputs": {
                name: int(features.shape[-1])
                for name, features in _features(train_dataset).items()
            },
            "hidden_layers": [hidden_dim, hidden_dim],
            "activation": "ELU",
            "epochs": epochs,
            "batch_size": batch_size,
            "safe_distance_m": 0.72,
            "nearest_neighbor_selected_before_action": True,
            "tie_break": "lowest_agent_index",
            "neighbor_action_in_actor_or_communication": False,
            "reward_or_execution_modified": False,
            "training_authorized": False,
            "four_million_screen_authorized": False,
        },
        "validation": validation,
        "aggregate": aggregate,
        "gate": gate,
        "invariance": {
            "actor_digest_before": actor_digest_before,
            "actor_digest_after": actor_digest_after,
            "actor_probe_output_max_abs_change": actor_output_change,
        },
        "decision": (
            "write_pair_local_safety_critic_plan_but_do_not_train"
            if gate["passed"]
            else "do_not_implement_pair_local_safety_critic"
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
            "model_seeds": list(model_seeds),
            "models": {
                name: [replica[name].artifact_state() for replica in fitted]
                for name in MODEL_NAMES
            },
        },
        models_path,
    )
    manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "experiment": EXPERIMENT_ID,
        "run": run_name,
        "producer": "scripts/analyze_nearest_pair_safety_action_coupling.py",
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
    result = analyze_nearest_pair_safety_action_coupling(
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
