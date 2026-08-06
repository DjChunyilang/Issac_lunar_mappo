#!/usr/bin/env python
"""Audit an existing-gate, per-agent safety-potential credit offline.

The frozen exp125 policy and its team MAPPO advantages are reused. This script
does not alter the reward, optimizer, checkpoint, or online execution chain.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from _common import ROOT, cfg_from_experiment
from analyze_actor_gradient_conflicts import (
    _collect_seed_dataset,
    _flatten_gradients,
    _parse_int_tuple,
    _policy_log_prob,
    _summarize_batch_metrics,
    gradient_metrics,
)
from analyze_joint_action_critic_feasibility import tensor_dict_digest
from train_skrl_mappo import SKRLPolicy


EXPERIMENT_ID = "exp132_agent_safety_potential_credit"


@dataclass(frozen=True, slots=True)
class CreditAuditSpec:
    experiment_id: str
    metric_filename: str
    distance_m: float
    distance_source: str
    pass_status: str
    pass_decision: str
    raw_active_min: float
    centered_active_min: float
    positive_min: float
    negative_min: float
    trace_std_min: float = 0.90
    trace_std_max: float = 1.10
    norm_ratio_min: float = 0.05
    norm_ratio_max: float = 20.0
    terrain_conflict_min: float = 0.20
    active_rollout_fraction_min: float | None = None
    first_active_rollout_index_max: int | None = None
    environment_rollout_active_mean_min: float | None = None


def credit_audit_spec(config: str | Path, credit_kind: str) -> CreditAuditSpec:
    cfg = cfg_from_experiment(config)
    if credit_kind == "success_gate":
        return CreditAuditSpec(
            experiment_id=EXPERIMENT_ID,
            metric_filename="agent_safety_potential_credit.json",
            distance_m=float(cfg.success_thresholds.min_pairwise_distance),
            distance_source="existing success_thresholds.min_pairwise_distance",
            pass_status="allow_c3_plan_only",
            pass_decision="draft_c3_screen_plan_only",
            raw_active_min=0.01,
            centered_active_min=0.01,
            positive_min=0.0025,
            negative_min=0.0025,
        )
    if credit_kind == "near_distance":
        return CreditAuditSpec(
            experiment_id="exp133_agent_near_distance_credit",
            metric_filename="agent_near_distance_credit.json",
            distance_m=float(cfg.safety.near_distance),
            distance_source="existing safety.near_distance",
            pass_status="allow_c3_near_plan_only",
            pass_decision="draft_c3_near_screen_plan_only",
            raw_active_min=0.08,
            centered_active_min=0.10,
            positive_min=0.01,
            negative_min=0.01,
            active_rollout_fraction_min=0.875,
            first_active_rollout_index_max=1,
            environment_rollout_active_mean_min=0.25,
        )
    raise ValueError(f"Unsupported safety credit kind: {credit_kind!r}.")


def credit_statistics(dataset: dict[str, torch.Tensor]) -> dict[str, Any]:
    raw = dataset["safety_raw_credits"]
    centered = dataset["safety_centered_step_credits"]
    trace = dataset["safety_credits"]
    rollout_flags = dataset["safety_rollout_active_flags"].detach().cpu()
    environment_rollout_active = dataset[
        "safety_environment_rollout_active_fractions"
    ].detach().cpu()
    active_indices = torch.nonzero(rollout_flags > 0.5, as_tuple=False).flatten()
    epsilon = 1.0e-8
    return {
        "safe_distance_m": float(dataset["safe_distance"].detach().cpu()),
        "raw_active_fraction": float((raw.abs() > epsilon).float().mean().cpu()),
        "raw_positive_fraction": float((raw > epsilon).float().mean().cpu()),
        "raw_negative_fraction": float((raw < -epsilon).float().mean().cpu()),
        "centered_active_fraction": float(
            (centered.abs() > epsilon).float().mean().cpu()
        ),
        "raw_mean": float(raw.mean().cpu()),
        "raw_std": float(raw.std().cpu()),
        "trace_mean": float(trace.mean().cpu()),
        "trace_std": float(trace.std().cpu()),
        "active_rollout_fraction": float(
            dataset["safety_active_rollout_fraction"].detach().cpu()
        ),
        "first_active_rollout_index": (
            int(active_indices[0]) if active_indices.numel() else int(rollout_flags.numel())
        ),
        "environment_rollout_active_fraction_mean": float(
            environment_rollout_active.mean()
        ),
        "environment_rollout_active_fractions": [
            float(value) for value in environment_rollout_active
        ],
        "centered_zero_sum_max_abs": float(
            dataset["safety_centered_zero_sum_max_abs"].detach().cpu()
        ),
    }


def analyze_agent_safety_credit(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    num_envs: int = 128,
    steps: int = 512,
    rollout_length: int = 64,
    data_seeds: tuple[int, ...] = (20023, 21023),
    batch_size: int = 4096,
    batches: int = 32,
    batch_seed: int = 232,
    run_dir: str | Path | None = None,
    credit_kind: str = "success_gate",
) -> dict[str, Any]:
    spec = credit_audit_spec(config, credit_kind)
    checkpoint_data = torch.load(checkpoint, map_location=torch.device(device))
    digest_before = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
    per_seed: dict[str, Any] = {}
    raw_active_rates: list[float] = []
    centered_active_rates: list[float] = []
    positive_rates: list[float] = []
    negative_rates: list[float] = []
    safety_team_norm_ratios: list[float] = []
    terrain_safety_conflict_rates: list[float] = []
    zero_sum_errors: list[float] = []
    trace_stds: list[float] = []
    active_rollout_rates: list[float] = []
    first_active_rollout_indices: list[int] = []
    environment_rollout_active_means: list[float] = []

    for data_seed in data_seeds:
        dataset = _collect_seed_dataset(
            config=config,
            checkpoint_data=checkpoint_data,
            device=device,
            num_envs=num_envs,
            steps=steps,
            rollout_length=rollout_length,
            seed=data_seed,
            safety_credit_distance=spec.distance_m,
        )
        policy: SKRLPolicy = dataset.pop("policy")
        statistics = credit_statistics(dataset)
        named_parameters = tuple(
            (name, parameter)
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        )
        names = tuple(item[0] for item in named_parameters)
        parameters = tuple(item[1] for item in named_parameters)
        groups = ("all", "neighbor_encoder", "terrain_encoder", "trunk")
        team_safety_items: dict[str, list[dict[str, float]]] = {
            group: [] for group in groups
        }
        terrain_safety_items: dict[str, list[dict[str, float]]] = {
            group: [] for group in groups
        }
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
            safety_credits = dataset["safety_credits"][index]

            team_loss = -(
                team_advantages * _policy_log_prob(policy, observations, actions)
            ).mean()
            team_gradients = torch.autograd.grad(
                team_loss, parameters, allow_unused=True
            )
            terrain_loss = -(
                terrain_credits * _policy_log_prob(policy, observations, actions)
            ).mean()
            terrain_gradients = torch.autograd.grad(
                terrain_loss, parameters, allow_unused=True
            )
            safety_loss = -(
                safety_credits * _policy_log_prob(policy, observations, actions)
            ).mean()
            safety_gradients = torch.autograd.grad(
                safety_loss, parameters, allow_unused=True
            )
            for group in groups:
                team = _flatten_gradients(team_gradients, parameters, names, group)
                terrain = _flatten_gradients(
                    terrain_gradients, parameters, names, group
                )
                safety = _flatten_gradients(safety_gradients, parameters, names, group)
                team_safety_items[group].append(gradient_metrics(team, safety))
                terrain_safety_items[group].append(gradient_metrics(terrain, safety))

        team_safety = {
            group: _summarize_batch_metrics(items)
            for group, items in team_safety_items.items()
        }
        terrain_safety = {
            group: _summarize_batch_metrics(items)
            for group, items in terrain_safety_items.items()
        }
        per_seed[str(data_seed)] = {
            "actor_samples": sample_count,
            "credit": statistics,
            "team_vs_safety_gradient": team_safety,
            "terrain_vs_safety_gradient": terrain_safety,
        }
        raw_active_rates.append(statistics["raw_active_fraction"])
        centered_active_rates.append(statistics["centered_active_fraction"])
        positive_rates.append(statistics["raw_positive_fraction"])
        negative_rates.append(statistics["raw_negative_fraction"])
        zero_sum_errors.append(statistics["centered_zero_sum_max_abs"])
        trace_stds.append(statistics["trace_std"])
        active_rollout_rates.append(statistics["active_rollout_fraction"])
        first_active_rollout_indices.append(statistics["first_active_rollout_index"])
        environment_rollout_active_means.append(
            statistics["environment_rollout_active_fraction_mean"]
        )
        safety_team_norm_ratios.append(
            team_safety["all"]["norm_ratio_median"]
        )
        terrain_safety_conflict_rates.append(
            terrain_safety["all"]["negative_cosine_fraction"]
        )

    digest_after = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
    checks = {
        "every_seed_raw_active_fraction": min(raw_active_rates)
        >= spec.raw_active_min,
        "every_seed_centered_active_fraction": min(centered_active_rates)
        >= spec.centered_active_min,
        "every_seed_raw_positive_fraction": min(positive_rates) >= spec.positive_min,
        "every_seed_raw_negative_fraction": min(negative_rates) >= spec.negative_min,
        "every_seed_trace_std_in_range": min(trace_stds) >= spec.trace_std_min
        and max(trace_stds) <= spec.trace_std_max,
        "every_seed_safety_team_norm_ratio_in_range": min(safety_team_norm_ratios)
        >= spec.norm_ratio_min
        and max(safety_team_norm_ratios) <= spec.norm_ratio_max,
        "every_seed_terrain_safety_conflict_fraction": min(
            terrain_safety_conflict_rates
        )
        >= spec.terrain_conflict_min,
        "centered_credit_zero_sum_max_abs_le_1e_6": max(zero_sum_errors) <= 1.0e-6,
        "actor_checkpoint_unchanged": digest_before == digest_after,
    }
    if spec.active_rollout_fraction_min is not None:
        checks["every_seed_active_rollout_fraction"] = (
            min(active_rollout_rates) >= spec.active_rollout_fraction_min
        )
    if spec.first_active_rollout_index_max is not None:
        checks["every_seed_first_active_rollout_index"] = (
            min(active_rollout_rates) > 0.0
            and max(first_active_rollout_indices) <= spec.first_active_rollout_index_max
        )
    if spec.environment_rollout_active_mean_min is not None:
        checks["every_seed_environment_rollout_active_fraction_mean"] = (
            min(environment_rollout_active_means)
            >= spec.environment_rollout_active_mean_min
        )
    passed = all(checks.values())
    result: dict[str, Any] = {
        "experiment": spec.experiment_id,
        "status": spec.pass_status if passed else "stop_safety_credit_candidate",
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
            "credit_kind": credit_kind,
            "safe_distance_source": spec.distance_source,
            "safe_distance_m": spec.distance_m,
            "potential": "-relu(d_safe - nearest_neighbor_distance_i)",
            "step_credit": "potential(next_state) - potential(state)",
            "agent_centering": "subtract team mean at each environment step",
            "trace": "gamma=training gamma, lambda=0.95, joint normalization",
            "reward_or_execution_modified": False,
            "training_or_optimizer_modified": False,
        },
        "pre_registered_thresholds": {
            "raw_active_min": spec.raw_active_min,
            "centered_active_min": spec.centered_active_min,
            "positive_min": spec.positive_min,
            "negative_min": spec.negative_min,
            "trace_std_min": spec.trace_std_min,
            "trace_std_max": spec.trace_std_max,
            "norm_ratio_min": spec.norm_ratio_min,
            "norm_ratio_max": spec.norm_ratio_max,
            "terrain_conflict_min": spec.terrain_conflict_min,
            "active_rollout_fraction_min": spec.active_rollout_fraction_min,
            "first_active_rollout_index_max": spec.first_active_rollout_index_max,
            "environment_rollout_active_mean_min": (
                spec.environment_rollout_active_mean_min
            ),
        },
        "per_seed": per_seed,
        "aggregate": {
            "minimum_raw_active_fraction": min(raw_active_rates),
            "minimum_centered_active_fraction": min(centered_active_rates),
            "minimum_positive_fraction": min(positive_rates),
            "minimum_negative_fraction": min(negative_rates),
            "mean_safety_team_norm_ratio": sum(safety_team_norm_ratios)
            / len(safety_team_norm_ratios),
            "minimum_terrain_safety_conflict_fraction": min(
                terrain_safety_conflict_rates
            ),
            "minimum_active_rollout_fraction": min(active_rollout_rates),
            "maximum_first_active_rollout_index": max(first_active_rollout_indices),
            "minimum_environment_rollout_active_fraction_mean": min(
                environment_rollout_active_means
            ),
            "maximum_zero_sum_error": max(zero_sum_errors),
        },
        "checks": checks,
        "invariance": {
            "actor_digest_before": digest_before,
            "actor_digest_after": digest_after,
        },
        "decision": spec.pass_decision if passed else "do_not_train_c3",
    }
    run_dir_path = (
        Path(run_dir)
        if run_dir is not None
        else ROOT / "outputs" / "runs" / spec.experiment_id / "frozen_exp125_seed23"
    )
    if not run_dir_path.is_absolute():
        run_dir_path = ROOT / run_dir_path
    metrics_dir = run_dir_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    config_dir = run_dir_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_source = Path(config)
    if not config_source.is_absolute():
        config_source = ROOT / config_source
    config_snapshot = config_dir / "experiment.yaml"
    config_snapshot.write_text(config_source.read_text(encoding="utf-8"), encoding="utf-8")
    metrics_path = metrics_dir / spec.metric_filename
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    try:
        manifest_metrics_path = str(metrics_path.relative_to(ROOT))
    except ValueError:
        manifest_metrics_path = str(metrics_path)
    try:
        manifest_config_path = str(config_snapshot.relative_to(ROOT))
    except ValueError:
        manifest_config_path = str(config_snapshot)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": spec.experiment_id,
        "producer": "scripts/analyze_agent_safety_credit.py",
        "status": result["status"],
        "source_checkpoint": str(checkpoint),
        "device": device,
        "collection": result["collection"],
        "artifacts": {
            "config": manifest_config_path,
            "metrics": manifest_metrics_path,
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
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--data-seeds", type=_parse_int_tuple, default=(20023, 21023))
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--batch-seed", type=int, default=232)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--credit-kind",
        choices=("success_gate", "near_distance"),
        default="success_gate",
    )
    args = parser.parse_args()
    result = analyze_agent_safety_credit(
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
        credit_kind=args.credit_kind,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
