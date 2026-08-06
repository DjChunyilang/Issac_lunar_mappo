#!/usr/bin/env python
"""Audit whether a centralized state can predict near-horizon real collisions."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import ROOT, cfg_from_experiment, load_yaml
from analyze_joint_action_critic_feasibility import (
    _parse_int_tuple,
    _policy_std,
    tensor_dict_digest,
)
from play import _load_policy_players
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)


EXPERIMENT_ID = "exp141_collision_cost_value_feasibility"


def future_collision_labels(
    collision: torch.Tensor,
    done: torch.Tensor,
    *,
    horizon: int,
) -> torch.Tensor:
    """Return collision-within-horizon labels without crossing episode resets."""

    if collision.ndim != 2 or done.shape != collision.shape:
        raise ValueError("collision and done must share shape [time, environment].")
    if horizon <= 0 or horizon > collision.shape[0]:
        raise ValueError("horizon must be in [1, time].")
    starts = collision.shape[0] - horizon + 1
    labels = torch.zeros(
        (starts, collision.shape[1]),
        dtype=torch.bool,
        device=collision.device,
    )
    alive = torch.ones_like(labels)
    for offset in range(horizon):
        collision_slice = collision[offset : offset + starts].bool()
        done_slice = done[offset : offset + starts].bool()
        labels |= alive & collision_slice
        alive &= ~done_slice
    return labels


def binary_ranking_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    probability = probability.detach().float().flatten().cpu()
    target = target.detach().bool().flatten().cpu()
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        return {"auroc": 0.0, "average_precision": 0.0}
    order = torch.argsort(probability, descending=True)
    sorted_target = target[order].float()
    true_positive = sorted_target.cumsum(0)
    false_positive = (1.0 - sorted_target).cumsum(0)
    true_positive_rate = torch.cat(
        (torch.zeros(1), true_positive / positives)
    )
    false_positive_rate = torch.cat(
        (torch.zeros(1), false_positive / negatives)
    )
    auroc = float(torch.trapz(true_positive_rate, false_positive_rate))
    precision = true_positive / torch.arange(1, target.numel() + 1)
    average_precision = float(precision[sorted_target.bool()].mean())
    return {"auroc": auroc, "average_precision": average_precision}


class CollisionCostValue(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


@dataclass(slots=True)
class CollisionDataset:
    states: torch.Tensor
    labels: torch.Tensor
    collision_episodes: int
    completed_episodes: int
    actor_probe_observations: torch.Tensor
    actor_probe: torch.Tensor


@dataclass(slots=True)
class FittedCostValue:
    model: CollisionCostValue
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    positive_weight: torch.Tensor
    final_loss: float

    def predict_probability(self, features: torch.Tensor, batch_size: int = 16384) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, features.shape[0], batch_size):
                batch = features[start : start + batch_size].to(self.feature_mean.device)
                normalized = (batch - self.feature_mean) / self.feature_std
                weighted_logit = self.model(normalized)
                calibrated_logit = weighted_logit - self.positive_weight.log()
                outputs.append(torch.sigmoid(calibrated_logit).cpu())
        return torch.cat(outputs) if outputs else torch.empty(0)

    def artifact_state(self) -> dict[str, Any]:
        return {
            "model": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            "feature_mean": self.feature_mean.detach().cpu(),
            "feature_std": self.feature_std.detach().cpu(),
            "positive_weight": self.positive_weight.detach().cpu(),
            "final_loss": self.final_loss,
        }


def _collect_dataset(
    *,
    config: str | Path,
    checkpoint_data: dict,
    device: str,
    num_envs: int,
    steps: int,
    horizon: int,
    seed: int,
) -> CollisionDataset:
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
    probe_obs = actor_obs[: min(32, num_envs)].detach().clone()
    with torch.no_grad():
        actor_probe = act(probe_obs).detach().cpu()
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed + 7919)
    std = _policy_std(checkpoint_data, cfg.task.n_agents, env.device)
    states: list[torch.Tensor] = []
    collisions: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    for _ in range(steps):
        with torch.no_grad():
            mean = act(actor_obs)
            actions = (
                mean
                + std
                * torch.randn(
                    mean.shape,
                    generator=generator,
                    device=env.device,
                    dtype=mean.dtype,
                )
            ).clamp(-1.0, 1.0)
            states.append(critic_state.detach().clone())
            output = env.step(actions)
            done_flags = output.info["done"]
            collisions.append(done_flags.collision.detach().clone())
            dones.append(done_flags.done.detach().clone())
            actor_obs = output.actor_obs
            critic_state = output.critic_state
    collision_tensor = torch.stack(collisions)
    done_tensor = torch.stack(dones)
    labels = future_collision_labels(
        collision_tensor,
        done_tensor,
        horizon=horizon,
    )
    starts = labels.shape[0]
    state_tensor = torch.stack(states)[:starts]
    return CollisionDataset(
        states=state_tensor.reshape(-1, state_tensor.shape[-1]).cpu(),
        labels=labels.reshape(-1).float().cpu(),
        collision_episodes=int(collision_tensor.sum().cpu()),
        completed_episodes=int(done_tensor.sum().cpu()),
        actor_probe_observations=probe_obs.cpu(),
        actor_probe=actor_probe,
    )


def fit_cost_value(
    states: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
) -> FittedCostValue:
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("Cost-value fitting requires both positive and negative labels.")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    states = states.to(device)
    labels = labels.to(device)
    feature_mean = states.mean(dim=0)
    feature_std = states.std(dim=0).clamp_min(1.0e-5)
    normalized = (states - feature_mean) / feature_std
    positive_weight = (negatives / positives).to(device).clamp_min(1.0)
    model = CollisionCostValue(states.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1.0e-5)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 101)
    final_loss = float("nan")
    for _ in range(epochs):
        permutation = torch.randperm(states.shape[0], generator=generator, device=device)
        for start in range(0, states.shape[0], batch_size):
            index = permutation[start : start + batch_size]
            logit = model(normalized[index])
            loss = F.binary_cross_entropy_with_logits(
                logit,
                labels[index],
                pos_weight=positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    return FittedCostValue(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        positive_weight=positive_weight,
        final_loss=final_loss,
    )


def evaluation_metrics(
    probability: torch.Tensor,
    labels: torch.Tensor,
    *,
    baseline_probability: float,
) -> dict[str, float]:
    labels = labels.float().cpu()
    probability = probability.float().cpu()
    ranking = binary_ranking_metrics(probability, labels)
    brier = float((probability - labels).square().mean())
    baseline = torch.full_like(labels, float(baseline_probability))
    baseline_brier = float((baseline - labels).square().mean())
    return {
        **ranking,
        "positive_rate": float(labels.mean()),
        "brier_score": brier,
        "constant_baseline_brier_score": baseline_brier,
        "brier_improvement_fraction": (baseline_brier - brier)
        / max(baseline_brier, 1.0e-12),
    }


def gate_decision(result: dict) -> dict:
    validation = result["validation"]
    values = list(validation.values())
    checks = {
        "train_collision_episodes_ge_30": result["training"]["collision_episodes"] >= 30,
        "every_validation_seed_collision_episodes_ge_20": min(
            item["collision_episodes"] for item in values
        )
        >= 20,
        "every_validation_seed_positive_rate_ge_0_005": min(
            item["positive_rate"] for item in values
        )
        >= 0.005,
        "every_validation_seed_mean_auroc_ge_0_75": min(
            item["mean_auroc"] for item in values
        )
        >= 0.75,
        "every_validation_seed_mean_auprc_ge_3x_prevalence": all(
            item["mean_average_precision"] >= 3.0 * item["positive_rate"]
            for item in values
        ),
        "every_validation_seed_mean_brier_improvement_ge_0_15": min(
            item["mean_brier_improvement_fraction"] for item in values
        )
        >= 0.15,
        "actor_parameters_unchanged": result["invariance"]["actor_digest_before"]
        == result["invariance"]["actor_digest_after"],
        "actor_outputs_unchanged": result["invariance"][
            "actor_probe_output_max_abs_change"
        ]
        == 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "train_collision_episodes": 30,
            "validation_collision_episodes": 20,
            "positive_rate": 0.005,
            "auroc": 0.75,
            "auprc_prevalence_multiple": 3.0,
            "brier_improvement_fraction": 0.15,
        },
    }


def analyze_collision_cost_value_feasibility(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    train_num_envs: int = 512,
    validation_num_envs: int = 256,
    steps: int = 512,
    horizon: int = 64,
    train_seed: int = 39023,
    validation_seeds: tuple[int, ...] = (40023, 41023),
    model_seeds: tuple[int, ...] = (7, 17, 29),
    epochs: int = 30,
    batch_size: int = 4096,
    learning_rate: float = 3.0e-4,
    hidden_dim: int = 128,
    run_dir: str | Path | None = None,
) -> dict:
    checkpoint_data = torch.load(checkpoint, map_location=torch.device(device))
    actor_digest_before = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
    training = _collect_dataset(
        config=config,
        checkpoint_data=checkpoint_data,
        device=device,
        num_envs=train_num_envs,
        steps=steps,
        horizon=horizon,
        seed=train_seed,
    )
    validations = {
        seed: _collect_dataset(
            config=config,
            checkpoint_data=checkpoint_data,
            device=device,
            num_envs=validation_num_envs,
            steps=steps,
            horizon=horizon,
            seed=seed,
        )
        for seed in validation_seeds
    }
    fitted = [
        fit_cost_value(
            training.states,
            training.labels,
            device=torch.device(device),
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
        )
        for seed in model_seeds
    ]
    train_prevalence = float(training.labels.mean())
    validation_result: dict[str, Any] = {}
    for seed, dataset in validations.items():
        replicas = []
        for model_seed, model in zip(model_seeds, fitted, strict=True):
            metrics = evaluation_metrics(
                model.predict_probability(dataset.states),
                dataset.labels,
                baseline_probability=train_prevalence,
            )
            replicas.append({"model_seed": model_seed, **metrics})
        validation_result[str(seed)] = {
            "samples": int(dataset.labels.numel()),
            "collision_episodes": dataset.collision_episodes,
            "completed_episodes": dataset.completed_episodes,
            "positive_rate": float(dataset.labels.mean()),
            "mean_auroc": sum(item["auroc"] for item in replicas) / len(replicas),
            "mean_average_precision": sum(
                item["average_precision"] for item in replicas
            )
            / len(replicas),
            "mean_brier_improvement_fraction": sum(
                item["brier_improvement_fraction"] for item in replicas
            )
            / len(replicas),
            "replicas": replicas,
        }
    actor_digest_after = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
    probe_cfg = cfg_from_experiment(config)
    raw_cfg = load_yaml(config)
    probe_act, _ = _load_policy_players(
        checkpoint_data,
        probe_cfg,
        torch.device(device),
        raw_cfg=raw_cfg,
    )
    with torch.no_grad():
        probe_after = probe_act(training.actor_probe_observations.to(device)).cpu()
    probe_change = float((probe_after - training.actor_probe).abs().amax())
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "collection": {
            "device": device,
            "steps": steps,
            "horizon": horizon,
            "train_num_envs": train_num_envs,
            "validation_num_envs": validation_num_envs,
            "train_seed": train_seed,
            "validation_seeds": list(validation_seeds),
            "model_seeds": list(model_seeds),
        },
        "training": {
            "samples": int(training.labels.numel()),
            "collision_episodes": training.collision_episodes,
            "completed_episodes": training.completed_episodes,
            "positive_rate": train_prevalence,
        },
        "models": {
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "parameter_count": sum(parameter.numel() for parameter in fitted[0].model.parameters()),
        },
        "validation": validation_result,
        "invariance": {
            "actor_digest_before": actor_digest_before,
            "actor_digest_after": actor_digest_after,
            "actor_probe_output_max_abs_change": probe_change,
        },
        "method": {
            "target_crosses_episode_reset": False,
            "joint_action_used": False,
            "oracle_or_mapf_used": False,
            "policy_or_training_modified": False,
        },
    }
    result["gate"] = gate_decision(result)
    result["status"] = (
        "allow_lagrangian_plan_only"
        if result["gate"]["passed"]
        else "stop_constraint_critic_direction"
    )
    result["decision"] = (
        "plan_ppo_lagrangian_only"
        if result["gate"]["passed"]
        else "do_not_implement_cost_critic_or_lagrangian"
    )

    run_path = (
        Path(run_dir)
        if run_dir is not None
        else ROOT / "outputs" / "runs" / EXPERIMENT_ID / "frozen_exp125_seed23"
    )
    if not run_path.is_absolute():
        run_path = ROOT / run_path
    metrics_dir = run_path / "metrics"
    artifacts_dir = run_path / "artifacts"
    config_dir = run_path / "config"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_source = Path(config)
    if not config_source.is_absolute():
        config_source = ROOT / config_source
    (config_dir / "experiment.yaml").write_text(
        config_source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    metrics_path = metrics_dir / "collision_cost_value_feasibility.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    model_path = artifacts_dir / "diagnostic_cost_values.pt"
    torch.save(
        {
            "diagnostic_only": True,
            "deployable_policy": False,
            "models": [model.artifact_state() for model in fitted],
            "model_seeds": list(model_seeds),
        },
        model_path,
    )
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "producer": "scripts/analyze_collision_cost_value_feasibility.py",
        "status": result["status"],
        "source_checkpoint": str(checkpoint),
        "artifacts": {
            "config": str(config_dir / "experiment.yaml"),
            "metrics": str(metrics_path),
            "diagnostic_models": str(model_path),
        },
    }
    (run_path / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    result["artifact"] = str(metrics_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-num-envs", type=int, default=512)
    parser.add_argument("--validation-num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--train-seed", type=int, default=39023)
    parser.add_argument("--validation-seeds", type=_parse_int_tuple, default=(40023, 41023))
    parser.add_argument("--model-seeds", type=_parse_int_tuple, default=(7, 17, 29))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = analyze_collision_cost_value_feasibility(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        train_num_envs=args.train_num_envs,
        validation_num_envs=args.validation_num_envs,
        steps=args.steps,
        horizon=args.horizon,
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
