#!/usr/bin/env python
"""Measure whether one rover's action identifies its existing safety credit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from _common import ROOT
from analyze_actor_gradient_conflicts import _collect_seed_dataset, _parse_int_tuple
from analyze_joint_action_critic_feasibility import tensor_dict_digest
from analyze_reward_component_identifiability import (
    component_metrics,
    fit_multi_output_regressor,
)


EXPERIMENT_ID = "exp139_local_safety_credit_identifiability"
TARGET_NAMES = (
    "near_raw_credit",
    "near_centered_credit",
    "repeated_conflict_involvement",
)
EXP134_METRICS = (
    ROOT
    / "outputs"
    / "runs"
    / "exp134_near_credit_lead_time"
    / "frozen_exp125_seed23"
    / "metrics"
    / "near_credit_lead_time.json"
)


def _targets(dataset: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        (
            dataset["safety_raw_credits"],
            dataset["safety_centered_step_credits"],
            dataset["repeated_conflict_involvement"],
        ),
        dim=-1,
    )


def _activity(dataset: dict[str, torch.Tensor]) -> dict[str, float]:
    values = _targets(dataset)
    return {
        name: float((values[:, index].abs() > 1.0e-8).float().mean())
        for index, name in enumerate(TARGET_NAMES)
    }


def local_credit_gate(
    *,
    aggregate: dict[str, dict[str, float]],
    validation: dict[str, dict[str, Any]],
    exp134_checks: dict[str, bool],
    zero_sum_error: float,
    actor_unchanged: bool,
    actor_output_change: float,
) -> dict[str, Any]:
    required_exp134 = (
        "every_seed_collision_events_ge_20",
        "every_seed_collision_prior_near_fraction_ge_0_95",
        "every_seed_collision_lead_ge_1_fraction_ge_0_90",
        "every_seed_collision_lead_ge_2_fraction_ge_0_50",
        "every_seed_collision_lead_median_ge_2",
    )
    checks = {
        "near_raw_mean_action_gain_ge_0_15": aggregate["near_raw_credit"][
            "mean_mse_improvement_fraction"
        ]
        >= 0.15,
        "near_raw_every_seed_action_gain_ge_0_15": aggregate[
            "near_raw_credit"
        ]["minimum_seed_mse_improvement_fraction"]
        >= 0.15,
        "near_centered_mean_action_gain_ge_0_15": aggregate[
            "near_centered_credit"
        ]["mean_mse_improvement_fraction"]
        >= 0.15,
        "near_centered_every_seed_action_gain_ge_0_15": aggregate[
            "near_centered_credit"
        ]["minimum_seed_mse_improvement_fraction"]
        >= 0.15,
        "near_raw_every_seed_active_rate_ge_0_08": min(
            row["active_rate"]["near_raw_credit"] for row in validation.values()
        )
        >= 0.08,
        "near_centered_every_seed_active_rate_ge_0_10": min(
            row["active_rate"]["near_centered_credit"]
            for row in validation.values()
        )
        >= 0.10,
        "exp134_collision_lead_checks_hold": all(
            exp134_checks.get(name) is True for name in required_exp134
        ),
        "centered_credit_zero_sum": zero_sum_error <= 1.0e-6,
        "actor_parameters_unchanged": actor_unchanged,
        "actor_outputs_unchanged": actor_output_change == 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "raw_action_gain": 0.15,
            "centered_action_gain": 0.15,
            "raw_active_rate": 0.08,
            "centered_active_rate": 0.10,
            "zero_sum_max_abs": 1.0e-6,
        },
        "required_exp134_checks": list(required_exp134),
    }


def analyze_local_safety_credit_identifiability(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    train_num_envs: int = 128,
    validation_num_envs: int = 64,
    steps: int = 512,
    rollout_length: int = 64,
    train_seed: int = 36023,
    validation_seeds: tuple[int, ...] = (37023, 38023),
    model_seeds: tuple[int, ...] = (7, 17, 29),
    epochs: int = 30,
    batch_size: int = 4096,
    learning_rate: float = 3.0e-4,
    hidden_dim: int = 128,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    checkpoint_data = torch.load(checkpoint, map_location=torch_device)
    digest_before = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
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
    probe_obs = train_dataset["observations"][:32].detach().clone()
    with torch.no_grad():
        probe_before = train_policy.compute(
            {"observations": probe_obs}, role="policy"
        )[0].detach().clone()
    validation_datasets = {}
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

    train_targets = _targets(train_dataset)
    train_obs = train_dataset["observations"]
    train_local_action = torch.cat((train_obs, train_dataset["actions"]), dim=-1)
    fitted = []
    for seed in model_seeds:
        fitted.append(
            (
                fit_multi_output_regressor(
                    train_obs,
                    train_targets,
                    device=torch_device,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    hidden_dim=hidden_dim,
                ),
                fit_multi_output_regressor(
                    train_local_action,
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

    improvements = {name: [] for name in TARGET_NAMES}
    validation: dict[str, Any] = {}
    for validation_seed, dataset in validation_datasets.items():
        targets = _targets(dataset)
        observations = dataset["observations"]
        observations_actions = torch.cat((observations, dataset["actions"]), dim=-1)
        replicas = []
        seed_components = {}
        for model_seed, (state_model, action_model) in zip(
            model_seeds, fitted, strict=True
        ):
            state_prediction = state_model.predict(observations.to(torch_device)).cpu()
            action_prediction = action_model.predict(
                observations_actions.to(torch_device)
            ).cpu()
            components = {}
            for index, name in enumerate(TARGET_NAMES):
                state_metrics = component_metrics(
                    state_prediction[:, index], targets[:, index]
                )
                action_metrics = component_metrics(
                    action_prediction[:, index], targets[:, index]
                )
                gain = (state_metrics["mse"] - action_metrics["mse"]) / max(
                    state_metrics["mse"], 1.0e-12
                )
                components[name] = {
                    "observation_only": state_metrics,
                    "observation_and_own_action": action_metrics,
                    "mse_improvement_fraction": gain,
                }
            replicas.append({"model_seed": model_seed, "components": components})
        for name in TARGET_NAMES:
            values = [
                row["components"][name]["mse_improvement_fraction"]
                for row in replicas
            ]
            mean_value = sum(values) / len(values)
            improvements[name].append(mean_value)
            seed_components[name] = {
                "mean_mse_improvement_fraction": mean_value,
                "minimum_replica_mse_improvement_fraction": min(values),
            }
        raw_active = dataset["safety_raw_credits"].abs() > 1.0e-8
        repeated_active = dataset["repeated_conflict_involvement"] > 0.5
        validation[str(validation_seed)] = {
            "samples": int(observations.shape[0]),
            "active_rate": _activity(dataset),
            "repeated_given_raw_active": float(
                repeated_active[raw_active].float().mean()
                if raw_active.any()
                else torch.tensor(0.0)
            ),
            "raw_active_given_repeated": float(
                raw_active[repeated_active].float().mean()
                if repeated_active.any()
                else torch.tensor(0.0)
            ),
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
    exp134 = json.loads(EXP134_METRICS.read_text(encoding="utf-8"))
    with torch.no_grad():
        probe_after = train_policy.compute(
            {"observations": probe_obs}, role="policy"
        )[0]
    digest_after = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
    actor_output_change = float((probe_after - probe_before).abs().amax())
    zero_sum_error = max(
        float(dataset["safety_centered_zero_sum_max_abs"])
        for dataset in (train_dataset, *validation_datasets.values())
    )
    gate = local_credit_gate(
        aggregate=aggregate,
        validation=validation,
        exp134_checks=exp134["checks"],
        zero_sum_error=zero_sum_error,
        actor_unchanged=digest_before == digest_after,
        actor_output_change=actor_output_change,
    )
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "status": (
            "supports_c3_near_boundary_review"
            if gate["passed"]
            else "stop_local_safety_credit_reconsideration"
        ),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "collection": {
            "device": device,
            "steps": steps,
            "rollout_length": rollout_length,
            "train_num_envs": train_num_envs,
            "validation_num_envs": validation_num_envs,
            "train_seed": train_seed,
            "validation_seeds": list(validation_seeds),
            "model_seeds": list(model_seeds),
            "train_actor_samples": int(train_obs.shape[0]),
        },
        "method": {
            "observation_only_dim": int(train_obs.shape[-1]),
            "observation_and_own_action_dim": int(train_local_action.shape[-1]),
            "safe_distance_m": 0.72,
            "other_agent_actions_used": False,
            "oracle_or_global_state_used": False,
            "repeated_conflict_is_training_candidate": False,
            "reward_or_execution_modified": False,
            "training_or_optimizer_modified": False,
        },
        "train_activity": _activity(train_dataset),
        "validation": validation,
        "aggregate_components": aggregate,
        "exp134_collision_lead_evidence": {
            "source": str(EXP134_METRICS),
            "checks": exp134["checks"],
        },
        "gate": gate,
        "invariance": {
            "centered_zero_sum_max_abs": zero_sum_error,
            "actor_digest_before": digest_before,
            "actor_digest_after": digest_after,
            "actor_probe_output_max_abs_change": actor_output_change,
        },
        "decision": (
            "request_explicit_plan_revision_before_any_c3_near_training"
            if gate["passed"]
            else "do_not_reopen_c3_near"
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
    metrics_path = metrics_dir / "local_safety_credit_identifiability.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    models_path = artifacts_dir / "diagnostic_regressors.pt"
    torch.save(
        {
            "diagnostic_only": True,
            "deployable_policy": False,
            "target_order": list(TARGET_NAMES),
            "model_seeds": list(model_seeds),
            "observation_only": [row[0].artifact_state() for row in fitted],
            "observation_and_own_action": [row[1].artifact_state() for row in fitted],
        },
        models_path,
    )
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "producer": "scripts/analyze_local_safety_credit_identifiability.py",
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
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--train-seed", type=int, default=36023)
    parser.add_argument("--validation-seeds", type=_parse_int_tuple, default=(37023, 38023))
    parser.add_argument("--model-seeds", type=_parse_int_tuple, default=(7, 17, 29))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = analyze_local_safety_credit_identifiability(
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
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
