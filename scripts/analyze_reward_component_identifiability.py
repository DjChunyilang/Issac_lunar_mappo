#!/usr/bin/env python
"""Test whether frozen-policy joint actions identify individual reward terms.

The diagnostic is deliberately offline: it freezes the exp125 policy, collects
complete 96-second episodes, and fits matched state-only and state-plus-action
multi-output regressors. It does not alter MAPPO, rewards, or execution.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from _common import ROOT
from analyze_exp125_credit_assignment import pearson_correlation
from analyze_joint_action_critic_feasibility import (
    _collect_rollout_dataset,
    _parse_int_tuple,
    tensor_dict_digest,
)


EXPERIMENT_ID = "exp128_reward_component_identifiability"
CRITICAL_TERMS = ("gather", "safety", "terrain")
DIAGNOSTIC_TERMS = (
    "diagnostic_relative_path_risk",
    "diagnostic_predicted_conflict_involvement",
    "diagnostic_collision_involvement",
    "diagnostic_nearest_distance_change",
)


class MultiOutputRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass(slots=True)
class FittedMultiOutputRegressor:
    model: MultiOutputRegressor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor
    training_loss: float

    def predict(self, features: torch.Tensor, batch_size: int = 16384) -> torch.Tensor:
        predictions: list[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, features.shape[0], batch_size):
                batch = features[start : start + batch_size]
                normalized = (batch - self.feature_mean) / self.feature_std
                predictions.append(
                    self.model(normalized) * self.target_std + self.target_mean
                )
        return torch.cat(predictions, dim=0)

    def artifact_state(self) -> dict[str, Any]:
        return {
            "model": {
                key: value.detach().cpu() for key, value in self.model.state_dict().items()
            },
            "feature_mean": self.feature_mean.detach().cpu(),
            "feature_std": self.feature_std.detach().cpu(),
            "target_mean": self.target_mean.detach().cpu(),
            "target_std": self.target_std.detach().cpu(),
            "training_loss": self.training_loss,
        }


def fit_multi_output_regressor(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
) -> FittedMultiOutputRegressor:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    features = features.to(device)
    targets = targets.to(device)
    feature_mean = features.mean(dim=0)
    feature_std = features.std(dim=0).clamp_min(1.0e-5)
    target_mean = targets.mean(dim=0)
    target_std = targets.std(dim=0).clamp_min(1.0e-5)
    normalized_features = (features - feature_mean) / feature_std
    normalized_targets = (targets - target_mean) / target_std
    model = MultiOutputRegressor(
        features.shape[1], targets.shape[1], hidden_dim=hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1.0e-5)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 211)
    final_loss = float("nan")
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
    return FittedMultiOutputRegressor(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        training_loss=final_loss,
    )


def component_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.detach().float().cpu()
    target = target.detach().float().cpu()
    mse = float((prediction - target).square().mean())
    variance = float((target - target.mean()).square().mean())
    return {
        "mean": float(target.mean()),
        "std": float(target.std()),
        "active_rate": float((target.abs() > 1.0e-8).float().mean()),
        "mse": mse,
        "r2": 1.0 - mse / max(variance, 1.0e-12),
        "prediction_target_pearson": pearson_correlation(prediction, target) or 0.0,
    }


def identifiability_targets(dataset: Any, term_names: tuple[str, ...]) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for name in term_names:
        if name in dataset.reward_terms:
            values.append(dataset.reward_terms[name])
        elif name == "diagnostic_relative_path_risk":
            values.append(dataset.relative_path_risk.mean(dim=-1))
        elif name == "diagnostic_predicted_conflict_involvement":
            values.append(dataset.conflict_involvement.mean(dim=-1))
        elif name == "diagnostic_collision_involvement":
            values.append(dataset.collision_involvement.mean(dim=-1))
        elif name == "diagnostic_nearest_distance_change":
            values.append(dataset.nearest_distance_change.mean(dim=-1))
        else:
            raise KeyError(f"Unknown identifiability target: {name}")
    return torch.stack(values, dim=-1)


def analyze_reward_component_identifiability(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    train_num_envs: int = 128,
    validation_num_envs: int = 64,
    steps: int = 480,
    train_seed: int = 14023,
    validation_seeds: tuple[int, ...] = (15023, 16023),
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
    train_dataset, act, probe_obs, probe_action_before = _collect_rollout_dataset(
        config=config,
        checkpoint_data=checkpoint_data,
        device=device,
        num_envs=train_num_envs,
        steps=steps,
        seed=train_seed,
        horizon=1,
        exploration_multiplier=exploration_multiplier,
    )
    validation_datasets = {
        seed: _collect_rollout_dataset(
            config=config,
            checkpoint_data=checkpoint_data,
            device=device,
            num_envs=validation_num_envs,
            steps=steps,
            seed=seed,
            horizon=1,
            exploration_multiplier=exploration_multiplier,
        )[0]
        for seed in validation_seeds
    }
    term_names = tuple(train_dataset.reward_terms) + DIAGNOSTIC_TERMS
    train_targets = identifiability_targets(train_dataset, term_names)
    train_state = train_dataset.states
    train_joint = torch.cat((train_dataset.states, train_dataset.actions), dim=-1)
    fitted_pairs: list[
        tuple[FittedMultiOutputRegressor, FittedMultiOutputRegressor]
    ] = []
    for seed in model_seeds:
        fitted_pairs.append(
            (
                fit_multi_output_regressor(
                    train_state,
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

    validation: dict[str, Any] = {}
    component_seed_improvements: dict[str, list[float]] = {
        name: [] for name in term_names
    }
    for validation_seed, dataset in validation_datasets.items():
        targets = identifiability_targets(dataset, term_names)
        replicas: list[dict[str, Any]] = []
        for model_seed, (state_model, joint_model) in zip(
            model_seeds, fitted_pairs, strict=True
        ):
            state_prediction = state_model.predict(dataset.states.to(torch_device)).cpu()
            joint_prediction = joint_model.predict(
                torch.cat((dataset.states, dataset.actions), dim=-1).to(torch_device)
            ).cpu()
            components: dict[str, Any] = {}
            for index, name in enumerate(term_names):
                state_metrics = component_metrics(state_prediction[:, index], targets[:, index])
                joint_metrics = component_metrics(joint_prediction[:, index], targets[:, index])
                improvement = (state_metrics["mse"] - joint_metrics["mse"]) / max(
                    state_metrics["mse"], 1.0e-12
                )
                components[name] = {
                    "state_only": state_metrics,
                    "joint_action": joint_metrics,
                    "mse_improvement_fraction": improvement,
                }
            replicas.append({"model_seed": model_seed, "components": components})
        seed_components: dict[str, Any] = {}
        for name in term_names:
            improvements = [
                replica["components"][name]["mse_improvement_fraction"]
                for replica in replicas
            ]
            mean_improvement = sum(improvements) / len(improvements)
            component_seed_improvements[name].append(mean_improvement)
            seed_components[name] = {
                "mean_mse_improvement_fraction": mean_improvement,
                "minimum_replica_mse_improvement_fraction": min(improvements),
            }
        validation[str(validation_seed)] = {
            "samples": dataset.samples,
            "communication": dataset.communication,
            "components": seed_components,
            "replicas": replicas,
        }

    aggregate_components = {
        name: {
            "mean_mse_improvement_fraction": sum(values) / len(values),
            "minimum_seed_mse_improvement_fraction": min(values),
        }
        for name, values in component_seed_improvements.items()
    }
    checks = {
        f"{name}_mean_improvement_ge_15pct": aggregate_components[name][
            "mean_mse_improvement_fraction"
        ]
        >= 0.15
        for name in CRITICAL_TERMS
    }
    checks.update(
        {
            f"{name}_every_seed_improvement_ge_15pct": aggregate_components[name][
                "minimum_seed_mse_improvement_fraction"
            ]
            >= 0.15
            for name in CRITICAL_TERMS
        }
    )
    with torch.no_grad():
        probe_action_after = act(probe_obs)
    actor_digest_after = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
    actor_action_change = float((probe_action_after - probe_action_before).abs().amax().cpu())
    checks["actor_parameters_unchanged"] = actor_digest_before == actor_digest_after
    checks["actor_outputs_unchanged"] = actor_action_change == 0.0
    passed = all(checks.values())
    result: dict[str, Any] = {
        "status": "reward_credit_candidate_identifiable" if passed else "stop_reward_credit_candidate",
        "experiment": EXPERIMENT_ID,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "device": device,
        "collection": {
            "steps": steps,
            "episode_coverage_seconds": steps * 0.2,
            "train_num_envs": train_num_envs,
            "validation_num_envs": validation_num_envs,
            "train_seed": train_seed,
            "validation_seeds": list(validation_seeds),
            "model_seeds": list(model_seeds),
            "train_samples": train_dataset.samples,
            "exploration_multiplier": exploration_multiplier,
            "term_order": list(term_names),
        },
        "models": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "state_only_parameter_count": sum(
                parameter.numel() for parameter in fitted_pairs[0][0].model.parameters()
            ),
            "joint_action_parameter_count": sum(
                parameter.numel() for parameter in fitted_pairs[0][1].model.parameters()
            ),
        },
        "validation": validation,
        "aggregate_components": aggregate_components,
        "critical_terms": list(CRITICAL_TERMS),
        "checks": checks,
        "invariance": {
            "actor_digest_before": actor_digest_before,
            "actor_digest_after": actor_digest_after,
            "actor_probe_action_max_abs_change": actor_action_change,
        },
        "decision": "allow_single_credit_plan_only" if passed else "do_not_change_training_credit",
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
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "reward_component_identifiability.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    artifact_path = artifacts_dir / "diagnostic_regressors.pt"
    torch.save(
        {
            "diagnostic_only": True,
            "deployable_policy": False,
            "term_order": list(term_names),
            "model_seeds": list(model_seeds),
            "state_only": [pair[0].artifact_state() for pair in fitted_pairs],
            "joint_action": [pair[1].artifact_state() for pair in fitted_pairs],
            "source_checkpoint": str(checkpoint),
        },
        artifact_path,
    )
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "producer": "scripts/analyze_reward_component_identifiability.py",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-num-envs", type=int, default=128)
    parser.add_argument("--validation-num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=480)
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
    result = analyze_reward_component_identifiability(
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
