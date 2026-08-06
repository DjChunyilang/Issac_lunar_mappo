#!/usr/bin/env python
"""Match transient and pair-repeated trajectory conflicts to collision outcomes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from analyze_joint_action_critic_feasibility import _parse_int_tuple
from analyze_near_credit_lead_time import _distribution
from analyze_paired_action_interventions import _policy_digest
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from play import _load_policy_players


EXPERIMENT_ID = "exp135_repeated_conflict_outcomes"


def _bool_rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def extract_conflict_outcomes(
    *,
    predicted: torch.Tensor,
    pair_repeated: torch.Tensor,
    near_active: torch.Tensor,
    collision: torch.Tensor,
    done: torch.Tensor,
    outcome_window_steps: int = 4,
    collision_lookback_steps: int = 8,
) -> dict[str, Any]:
    """Summarize vehicle-level conflict events without crossing episode resets."""

    if predicted.ndim != 3:
        raise ValueError("predicted must have shape [time, environment, agent].")
    if pair_repeated.shape != predicted.shape or near_active.shape != predicted.shape:
        raise ValueError("pair_repeated and near_active must match predicted shape.")
    if collision.shape != predicted.shape:
        raise ValueError("collision must match predicted shape.")
    if done.shape != predicted.shape[:2]:
        raise ValueError("done must have shape [time, environment].")
    if outcome_window_steps < 0 or collision_lookback_steps <= 0:
        raise ValueError("Outcome window must be non-negative and lookback positive.")

    predicted = predicted.bool().cpu()
    pair_repeated = pair_repeated.bool().cpu()
    near_active = near_active.bool().cpu()
    collision = collision.bool().cpu()
    done = done.bool().cpu()
    event_outcomes: dict[str, list[bool]] = {"nonrepeated": [], "repeated": []}
    event_near: dict[str, list[bool]] = {"nonrepeated": [], "repeated": []}
    event_durations: dict[str, list[int]] = {"nonrepeated": [], "repeated": []}
    collision_any_recall: list[bool] = []
    collision_repeated_recall: list[bool] = []

    time_steps, num_envs, n_agents = predicted.shape
    for env_index in range(num_envs):
        episode_start = 0
        episode_ends = torch.nonzero(done[:, env_index], as_tuple=False).flatten().tolist()
        boundaries = [(int(value) + 1, True) for value in episode_ends]
        if not boundaries or boundaries[-1][0] < time_steps:
            boundaries.append((time_steps, False))
        for episode_end, completed in boundaries:
            if episode_end <= episode_start:
                continue
            for agent in range(n_agents):
                series = predicted[episode_start:episode_end, env_index, agent]
                repeated_series = pair_repeated[
                    episode_start:episode_end, env_index, agent
                ]
                near_series = near_active[episode_start:episode_end, env_index, agent]
                collision_series = collision[
                    episode_start:episode_end, env_index, agent
                ]
                local_time = 0
                while local_time < series.numel():
                    if not bool(series[local_time]):
                        local_time += 1
                        continue
                    event_start = local_time
                    while local_time < series.numel() and bool(series[local_time]):
                        local_time += 1
                    event_end = local_time
                    if not completed and event_end + outcome_window_steps > series.numel():
                        continue
                    category = (
                        "repeated"
                        if bool(repeated_series[event_start:event_end].any())
                        else "nonrepeated"
                    )
                    result_end = min(
                        series.numel(), event_end + outcome_window_steps
                    )
                    event_outcomes[category].append(
                        bool(collision_series[event_start:result_end].any())
                    )
                    event_near[category].append(
                        bool(near_series[event_start:event_end].any())
                    )
                    event_durations[category].append(event_end - event_start)

                for collision_time in torch.nonzero(
                    collision_series, as_tuple=False
                ).flatten().tolist():
                    start = max(0, int(collision_time) - collision_lookback_steps + 1)
                    end = int(collision_time) + 1
                    collision_any_recall.append(bool(series[start:end].any()))
                    collision_repeated_recall.append(
                        bool(repeated_series[start:end].any())
                    )
            episode_start = episode_end

    nonrepeated_rate = _bool_rate(event_outcomes["nonrepeated"])
    repeated_rate = _bool_rate(event_outcomes["repeated"])
    nonrepeated_outcome_count = sum(event_outcomes["nonrepeated"])
    repeated_outcome_count = sum(event_outcomes["repeated"])
    ratio_is_infinite = nonrepeated_rate == 0.0 and repeated_rate > 0.0
    return {
        "nonrepeated_events": len(event_outcomes["nonrepeated"]),
        "repeated_events": len(event_outcomes["repeated"]),
        "nonrepeated_collision_outcome_rate": nonrepeated_rate,
        "repeated_collision_outcome_rate": repeated_rate,
        "nonrepeated_collision_outcomes": nonrepeated_outcome_count,
        "repeated_collision_outcomes": repeated_outcome_count,
        "repeated_vs_nonrepeated_outcome_ratio": (
            None if ratio_is_infinite else repeated_rate / max(nonrepeated_rate, 1.0e-12)
        ),
        "repeated_vs_nonrepeated_ratio_is_infinite": ratio_is_infinite,
        "nonrepeated_near_coverage_rate": _bool_rate(event_near["nonrepeated"]),
        "repeated_near_coverage_rate": _bool_rate(event_near["repeated"]),
        "near_coverage_rate_difference": _bool_rate(event_near["repeated"])
        - _bool_rate(event_near["nonrepeated"]),
        "nonrepeated_duration_steps": _distribution(
            torch.tensor(event_durations["nonrepeated"], dtype=torch.long)
        ),
        "repeated_duration_steps": _distribution(
            torch.tensor(event_durations["repeated"], dtype=torch.long)
        ),
        "collision_involvement_events": len(collision_repeated_recall),
        "collision_any_conflict_recall": _bool_rate(collision_any_recall),
        "collision_repeated_conflict_recall": _bool_rate(
            collision_repeated_recall
        ),
        "outcome_window_steps": outcome_window_steps,
        "collision_lookback_steps": collision_lookback_steps,
    }


def collect_conflict_timeline(
    *,
    config: str | Path,
    checkpoint_data: dict[str, Any],
    device: str,
    num_envs: int,
    steps: int,
    seed: int,
) -> dict[str, torch.Tensor]:
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
    actor_obs, _ = env.get_observations()
    policy_std = (
        checkpoint_data["rover_0"]["policy"]["log_std_parameter"]
        .detach()
        .to(env.device)
        .exp()
        .view(1, 1, 2)
    )
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed + 7919)
    sequences: dict[str, list[torch.Tensor]] = {
        "predicted": [],
        "pair_repeated": [],
        "repeated_pairs": [],
        "near_active": [],
        "collision": [],
        "done": [],
        "success": [],
        "collision_done": [],
        "out_of_bounds": [],
        "timeout": [],
    }
    for _ in range(steps):
        with torch.no_grad():
            near_before = (
                env.metrics.nearest_neighbor_distance
                < float(cfg.safety.near_distance)
            )
            mean = act(actor_obs)
            actions = (
                mean
                + policy_std
                * torch.randn(
                    mean.shape,
                    generator=generator,
                    device=env.device,
                    dtype=mean.dtype,
                )
            ).clamp(-1.0, 1.0)
            output = env.step(actions)
            conflicts = output.info["trajectory_conflicts"]
            active_pairs = conflicts["active"]
            repeated_pairs = conflicts["repeated"]
            predicted = active_pairs.any(dim=2) | active_pairs.any(dim=1)
            pair_repeated = repeated_pairs.any(dim=2) | repeated_pairs.any(dim=1)
            done_flags = output.info["done"]
            near_after = output.info["metrics"].nearest_neighbor_distance
            collision = (
                near_after < float(cfg.safety.collision_distance)
            ) & done_flags.collision[:, None]
            sequences["predicted"].append(predicted.cpu())
            sequences["pair_repeated"].append(pair_repeated.cpu())
            sequences["repeated_pairs"].append(repeated_pairs.cpu())
            sequences["near_active"].append(near_before.cpu())
            sequences["collision"].append(collision.cpu())
            sequences["done"].append(done_flags.done.cpu())
            sequences["success"].append(done_flags.success.cpu())
            sequences["collision_done"].append(done_flags.collision.cpu())
            sequences["out_of_bounds"].append(done_flags.out_of_bounds.cpu())
            sequences["timeout"].append(done_flags.timeout.cpu())
            actor_obs = output.actor_obs
    return {key: torch.stack(values) for key, values in sequences.items()}


def analyze_repeated_conflict_outcomes(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    num_envs: int = 128,
    steps: int = 512,
    data_seeds: tuple[int, ...] = (26023, 27023),
    outcome_window_steps: int = 4,
    collision_lookback_steps: int = 8,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_data = torch.load(checkpoint, map_location=torch.device(device))
    digest_before = _policy_digest(checkpoint_data)
    per_seed: dict[str, Any] = {}
    for seed in data_seeds:
        timeline = collect_conflict_timeline(
            config=config,
            checkpoint_data=checkpoint_data,
            device=device,
            num_envs=num_envs,
            steps=steps,
            seed=seed,
        )
        per_seed[str(seed)] = extract_conflict_outcomes(
            predicted=timeline["predicted"],
            pair_repeated=timeline["pair_repeated"],
            near_active=timeline["near_active"],
            collision=timeline["collision"],
            done=timeline["done"],
            outcome_window_steps=outcome_window_steps,
            collision_lookback_steps=collision_lookback_steps,
        )
    digest_after = _policy_digest(checkpoint_data)
    values = list(per_seed.values())
    checks = {
        "every_seed_nonrepeated_events_ge_1000": min(
            item["nonrepeated_events"] for item in values
        )
        >= 1000,
        "every_seed_repeated_events_ge_500": min(
            item["repeated_events"] for item in values
        )
        >= 500,
        "every_seed_collision_events_ge_100": min(
            item["collision_involvement_events"] for item in values
        )
        >= 100,
        "every_seed_repeated_outcome_rate_ge_0_01": min(
            item["repeated_collision_outcome_rate"] for item in values
        )
        >= 0.01,
        "every_seed_repeated_vs_nonrepeated_ratio_ge_2": all(
            item["repeated_vs_nonrepeated_ratio_is_infinite"]
            or (
                item["repeated_vs_nonrepeated_outcome_ratio"] is not None
                and item["repeated_vs_nonrepeated_outcome_ratio"] >= 2.0
            )
            for item in values
        ),
        "every_seed_collision_repeated_recall_ge_0_80": min(
            item["collision_repeated_conflict_recall"] for item in values
        )
        >= 0.80,
        "every_seed_near_coverage_difference_ge_0_20": min(
            item["near_coverage_rate_difference"] for item in values
        )
        >= 0.20,
        "actor_checkpoint_unchanged": digest_before == digest_after,
    }
    passed = all(checks.values())
    result: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "status": (
            "retain_repeated_conflict_metric"
            if passed
            else "reject_conflict_as_architecture_gate"
        ),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "collection": {
            "device": device,
            "num_envs": num_envs,
            "steps": steps,
            "data_seeds": list(data_seeds),
            "outcome_window_steps": outcome_window_steps,
            "collision_lookback_steps": collision_lookback_steps,
        },
        "method": {
            "event": "continuous per-agent predicted-conflict involvement",
            "repeated": "event contains existing pair-level repeated flag",
            "collision_outcome": "during event or fixed post-event window",
            "reward_or_execution_modified": False,
            "training_or_optimizer_modified": False,
        },
        "per_seed": per_seed,
        "checks": checks,
        "invariance": {
            "actor_digest_before": digest_before,
            "actor_digest_after": digest_after,
        },
        "decision": (
            "use_repeated_conflicts_for_diagnostics_only"
            if passed
            else "do_not_use_conflict_metrics_to_enable_b2"
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
    config_dir = run_dir_path / "config"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_source = Path(config)
    if not config_source.is_absolute():
        config_source = ROOT / config_source
    config_snapshot = config_dir / "experiment.yaml"
    config_snapshot.write_text(config_source.read_text(encoding="utf-8"), encoding="utf-8")
    metrics_path = metrics_dir / "repeated_conflict_outcomes.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def artifact_path(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "producer": "scripts/analyze_repeated_conflict_outcomes.py",
        "status": result["status"],
        "source_checkpoint": str(checkpoint),
        "device": device,
        "collection": result["collection"],
        "artifacts": {
            "config": artifact_path(config_snapshot),
            "metrics": artifact_path(metrics_path),
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
    parser.add_argument("--data-seeds", type=_parse_int_tuple, default=(26023, 27023))
    parser.add_argument("--outcome-window-steps", type=int, default=4)
    parser.add_argument("--collision-lookback-steps", type=int, default=8)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = analyze_repeated_conflict_outcomes(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        data_seeds=args.data_seeds,
        outcome_window_steps=args.outcome_window_steps,
        collision_lookback_steps=args.collision_lookback_steps,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
