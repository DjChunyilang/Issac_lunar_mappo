#!/usr/bin/env python
"""Audit whether terminal collision credit can be assigned to participating agents."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from analyze_joint_action_critic_feasibility import _parse_int_tuple
from analyze_paired_action_interventions import _policy_digest
from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import scale_action
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from lunar_rover_tasks.utils.geometry_utils import pairwise_distances_xy
from play import _load_policy_players


EXPERIMENT_ID = "exp149_collision_participant_credit_feasibility"
DEFAULT_RUN_ID = "frozen_exp148_dual_checkpoint_dualseed"
HORIZONS = (1, 2, 4, 8, 16)
PRECOLLISION_STEPS = 16


def _distribution(values: list[float] | torch.Tensor) -> dict[str, float | int]:
    tensor = torch.as_tensor(values, dtype=torch.float32).flatten()
    tensor = tensor[torch.isfinite(tensor)]
    if tensor.numel() == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
        }
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "p90": float(torch.quantile(tensor, 0.90)),
    }


def _upper_triangle(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("pair matrix must end with equal agent dimensions")
    n_agents = matrix.shape[-1]
    mask = torch.triu(
        torch.ones((n_agents, n_agents), dtype=torch.bool, device=matrix.device),
        diagonal=1,
    )
    return matrix.bool() & mask


def _weighted_reward_terms(output: Any, cfg: Any) -> dict[str, torch.Tensor]:
    terms = output.info["reward_terms"]
    weights = cfg.reward_weights
    return {
        "gather": terms.gather * float(weights.gather),
        "safety": terms.safety * float(weights.safety),
        "terrain": terms.terrain * float(weights.terrain),
        "terminal": terms.terminal * float(weights.terminal),
        "total": terms.total,
    }


def collect_collision_timeline(
    *,
    config: str | Path,
    checkpoint_data: dict[str, Any],
    device: str,
    num_envs: int,
    steps: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Collect a deterministic frozen-policy timeline without changing training state."""

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
    sequences: dict[str, list[torch.Tensor]] = {
        "actions": [],
        "physical_actions": [],
        "near_before": [],
        "near_after": [],
        "dmax_before": [],
        "dmax_after": [],
        "repeated_pairs": [],
        "active_pairs": [],
        "positions_after": [],
        "done": [],
        "collision_done": [],
        "success_done": [],
        "gather_reward": [],
        "safety_reward": [],
        "terrain_reward": [],
        "terminal_reward": [],
        "total_reward": [],
    }
    for _ in range(steps):
        with torch.no_grad():
            near_before = env.metrics.nearest_neighbor_distance.clone()
            dmax_before = env.metrics.dmax.clone()
            actions = act(actor_obs).clamp(-1.0, 1.0)
            physical_actions = scale_action(actions, cfg.planner)
            output = env.step(actions)
            reward_terms = _weighted_reward_terms(output, cfg)
            conflicts = output.info["trajectory_conflicts"]
            done = output.info["done"]
            sequences["actions"].append(actions.cpu())
            sequences["physical_actions"].append(physical_actions.cpu())
            sequences["near_before"].append(near_before.cpu())
            sequences["near_after"].append(
                output.info["metrics"].nearest_neighbor_distance.cpu()
            )
            sequences["dmax_before"].append(dmax_before.cpu())
            sequences["dmax_after"].append(output.info["metrics"].dmax.cpu())
            sequences["repeated_pairs"].append(conflicts["repeated"].cpu())
            sequences["active_pairs"].append(conflicts["active"].cpu())
            sequences["positions_after"].append(output.info["positions"].cpu())
            sequences["done"].append(done.done.cpu())
            sequences["collision_done"].append(done.collision.cpu())
            sequences["success_done"].append(done.success.cpu())
            for name, value in reward_terms.items():
                sequences[f"{name}_reward"].append(value.cpu())
            actor_obs = output.actor_obs
    return {name: torch.stack(values) for name, values in sequences.items()}


def summarize_collision_timeline(
    timeline: dict[str, torch.Tensor],
    *,
    collision_distance: float,
    horizons: tuple[int, ...] = HORIZONS,
    precollision_steps: int = PRECOLLISION_STEPS,
) -> dict[str, Any]:
    """Summarize terminal collision participants and their preceding conflict history."""

    required = {
        "actions",
        "physical_actions",
        "near_before",
        "near_after",
        "dmax_before",
        "dmax_after",
        "repeated_pairs",
        "active_pairs",
        "positions_after",
        "done",
        "collision_done",
        "success_done",
        "gather_reward",
        "safety_reward",
        "terrain_reward",
        "terminal_reward",
        "total_reward",
    }
    missing = required - timeline.keys()
    if missing:
        raise ValueError(f"timeline is missing keys: {sorted(missing)}")
    done = timeline["done"].bool()
    collision_done = timeline["collision_done"].bool()
    if done.shape != collision_done.shape or done.ndim != 2:
        raise ValueError("done and collision_done must have shape [time, environment]")
    time_steps, num_envs = done.shape
    n_agents = int(timeline["actions"].shape[2])
    if n_agents < 2:
        raise ValueError("at least two agents are required")

    collision_events: list[tuple[int, int]] = [
        (int(t), int(e))
        for t, e in torch.nonzero(collision_done, as_tuple=False).tolist()
    ]
    participant_counts: list[float] = []
    pair_counts: list[float] = []
    nonparticipant_fractions: list[float] = []
    recall_by_horizon: dict[int, list[float]] = {h: [] for h in horizons}
    precision_8: list[float] = []
    repeated_lead_steps: list[float] = []
    participant_action_forward: list[float] = []
    nonparticipant_action_forward: list[float] = []
    participant_action_radius: list[float] = []
    nonparticipant_action_radius: list[float] = []
    participant_turn_abs: list[float] = []
    nonparticipant_turn_abs: list[float] = []
    participant_closing: list[float] = []
    nonparticipant_closing: list[float] = []
    precollision_dmax_progress: list[float] = []
    reward_pre: dict[str, list[float]] = {
        name: [] for name in ("gather", "safety", "terrain", "terminal", "total")
    }
    reward_terminal: dict[str, list[float]] = {
        name: [] for name in ("gather", "safety", "terrain", "terminal", "total")
    }
    profiles: dict[int, dict[str, list[float]]] = {
        offset: {
            "dmax_progress": [],
            "nearest_closing_mean": [],
            "action_forward_mean": [],
            "action_radius_mean": [],
            "repeated_pair_count": [],
            "gather_reward": [],
            "safety_reward": [],
            "terrain_reward": [],
            "terminal_reward": [],
            "total_reward": [],
        }
        for offset in range(-precollision_steps, 1)
    }

    last_done = [-1 for _ in range(num_envs)]
    event_index = {(t, e) for t, e in collision_events}
    for t in range(time_steps):
        for e in range(num_envs):
            if (t, e) in event_index:
                episode_start = last_done[e] + 1
                positions = timeline["positions_after"][t, e, :, :2]
                distances = pairwise_distances_xy(positions.unsqueeze(0))[0]
                collision_pairs = _upper_triangle(distances < float(collision_distance))
                pair_count = int(collision_pairs.sum())
                if pair_count == 0:
                    continue
                participants = collision_pairs.any(dim=0) | collision_pairs.any(dim=1)
                participant_count = int(participants.sum())
                nonparticipants = ~participants
                participant_counts.append(float(participant_count))
                pair_counts.append(float(pair_count))
                nonparticipant_fractions.append(1.0 - participant_count / n_agents)

                repeated = _upper_triangle(timeline["repeated_pairs"][:, e])
                for horizon in horizons:
                    start = max(episode_start, t - horizon)
                    hit = repeated[start:t].any(dim=0) if start < t else torch.zeros_like(collision_pairs)
                    recall_by_horizon[horizon].append(
                        float((hit & collision_pairs).sum()) / float(pair_count)
                    )
                start_8 = max(episode_start, t - 8)
                hit_8 = repeated[start_8:t].any(dim=0) if start_8 < t else torch.zeros_like(collision_pairs)
                hit_count = int(hit_8.sum())
                precision_8.append(
                    float((hit_8 & collision_pairs).sum()) / float(max(hit_count, 1))
                )
                for i, j in torch.nonzero(collision_pairs, as_tuple=False).tolist():
                    history = torch.nonzero(repeated[episode_start:t, i, j], as_tuple=False)
                    if history.numel():
                        earliest = episode_start + int(history[0, 0])
                        repeated_lead_steps.append(float(t - earliest))

                action_start = max(episode_start, t - 4)
                if action_start < t:
                    action_window = timeline["actions"][action_start:t, e]
                    physical_window = timeline["physical_actions"][action_start:t, e]
                    closing_window = (
                        timeline["near_before"][action_start:t, e]
                        - timeline["near_after"][action_start:t, e]
                    )
                    participant_action_forward.append(
                        float(action_window[:, participants, 0].mean())
                    )
                    participant_action_radius.append(
                        float(physical_window[:, participants, 0].mean())
                    )
                    participant_turn_abs.append(
                        float(physical_window[:, participants, 1].abs().mean())
                    )
                    participant_closing.append(
                        float(closing_window[:, participants].mean())
                    )
                    if nonparticipants.any():
                        nonparticipant_action_forward.append(
                            float(action_window[:, nonparticipants, 0].mean())
                        )
                        nonparticipant_action_radius.append(
                            float(physical_window[:, nonparticipants, 0].mean())
                        )
                        nonparticipant_turn_abs.append(
                            float(physical_window[:, nonparticipants, 1].abs().mean())
                        )
                        nonparticipant_closing.append(
                            float(closing_window[:, nonparticipants].mean())
                        )

                pre_start = max(episode_start, t - precollision_steps)
                precollision_dmax_progress.append(
                    float(
                        (
                            timeline["dmax_before"][pre_start:t, e]
                            - timeline["dmax_after"][pre_start:t, e]
                        ).sum()
                    )
                )
                for name in reward_pre:
                    reward_pre[name].append(
                        float(timeline[f"{name}_reward"][pre_start:t, e].sum())
                    )
                    reward_terminal[name].append(
                        float(timeline[f"{name}_reward"][t, e])
                    )
                for offset in range(-precollision_steps, 1):
                    step = t + offset
                    if step < episode_start or step < 0:
                        continue
                    profile = profiles[offset]
                    profile["dmax_progress"].append(
                        float(
                            timeline["dmax_before"][step, e]
                            - timeline["dmax_after"][step, e]
                        )
                    )
                    profile["nearest_closing_mean"].append(
                        float(
                            (
                                timeline["near_before"][step, e]
                                - timeline["near_after"][step, e]
                            ).mean()
                        )
                    )
                    profile["action_forward_mean"].append(
                        float(timeline["actions"][step, e, :, 0].mean())
                    )
                    profile["action_radius_mean"].append(
                        float(timeline["physical_actions"][step, e, :, 0].mean())
                    )
                    profile["repeated_pair_count"].append(
                        float(_upper_triangle(timeline["repeated_pairs"][step, e]).sum())
                    )
                    for name in reward_pre:
                        profile[f"{name}_reward"].append(
                            float(timeline[f"{name}_reward"][step, e])
                        )
            if bool(done[t, e]):
                last_done[e] = t

    recall_summary = {
        str(horizon): _distribution(values)
        for horizon, values in recall_by_horizon.items()
    }
    profile_summary = []
    for offset, values in profiles.items():
        row: dict[str, Any] = {"offset_steps": offset}
        for name, samples in values.items():
            row[name] = _distribution(samples)
        profile_summary.append(row)
    participant_action = {
        "normalized_forward": _distribution(participant_action_forward),
        "physical_radius_m": _distribution(participant_action_radius),
        "turn_abs_rad": _distribution(participant_turn_abs),
        "nearest_closing_m_per_step": _distribution(participant_closing),
    }
    nonparticipant_action = {
        "normalized_forward": _distribution(nonparticipant_action_forward),
        "physical_radius_m": _distribution(nonparticipant_action_radius),
        "turn_abs_rad": _distribution(nonparticipant_turn_abs),
        "nearest_closing_m_per_step": _distribution(nonparticipant_closing),
    }
    return {
        "collision_episodes": len(participant_counts),
        "participant_count": _distribution(participant_counts),
        "collision_pair_count": _distribution(pair_counts),
        "nonparticipant_fraction": _distribution(nonparticipant_fractions),
        "collision_pair_repeated_recall": recall_summary,
        "collision_pair_repeated_precision_h8": _distribution(precision_8),
        "first_repeated_lead_steps": _distribution(repeated_lead_steps),
        "precollision_dmax_progress": _distribution(precollision_dmax_progress),
        "precollision_reward_sum": {
            name: _distribution(values) for name, values in reward_pre.items()
        },
        "terminal_reward": {
            name: _distribution(values) for name, values in reward_terminal.items()
        },
        "participant_last4": participant_action,
        "nonparticipant_last4": nonparticipant_action,
        "relative_time_profile": profile_summary,
    }


def _combination_checks(summary: dict[str, Any], actor_unchanged: bool) -> dict[str, bool]:
    return {
        "collision_episodes_ge_100": summary["collision_episodes"] >= 100,
        "nonparticipant_fraction_mean_ge_0_25": (
            summary["nonparticipant_fraction"]["mean"] >= 0.25
        ),
        "participant_count_median_le_2": summary["participant_count"]["median"] <= 2.0,
        "repeated_recall_h8_ge_0_80": (
            summary["collision_pair_repeated_recall"]["8"]["mean"] >= 0.80
        ),
        "repeated_recall_h16_ge_0_90": (
            summary["collision_pair_repeated_recall"]["16"]["mean"] >= 0.90
        ),
        "first_repeated_lead_median_ge_4": (
            summary["first_repeated_lead_steps"]["median"] >= 4.0
        ),
        "actor_checkpoint_unchanged": actor_unchanged,
    }


def analyze_collision_participant_credit_feasibility(
    *,
    config: str | Path,
    checkpoints: tuple[str | Path, ...],
    device: str = "cuda",
    num_envs: int = 128,
    steps: int = 512,
    data_seeds: tuple[int, ...] = (32023, 33023),
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    if len(checkpoints) < 2:
        raise ValueError("at least two checkpoints are required")
    cfg = cfg_from_experiment(config)
    combinations: dict[str, Any] = {}
    for checkpoint in checkpoints:
        checkpoint_path = Path(checkpoint)
        checkpoint_data = torch.load(checkpoint_path, map_location=torch.device(device))
        digest_before = _policy_digest(checkpoint_data)
        checkpoint_label = checkpoint_path.stem
        for seed in data_seeds:
            timeline = collect_collision_timeline(
                config=config,
                checkpoint_data=checkpoint_data,
                device=device,
                num_envs=num_envs,
                steps=steps,
                seed=seed,
            )
            summary = summarize_collision_timeline(
                timeline,
                collision_distance=float(cfg.safety.collision_distance),
            )
            digest_after = _policy_digest(checkpoint_data)
            checks = _combination_checks(summary, digest_before == digest_after)
            combinations[f"{checkpoint_label}_seed{seed}"] = {
                "checkpoint": str(checkpoint),
                "checkpoint_label": checkpoint_label,
                "seed": seed,
                "summary": summary,
                "checks": checks,
                "passed": all(checks.values()),
                "actor_digest_before": digest_before,
                "actor_digest_after": digest_after,
            }
    all_passed = all(item["passed"] for item in combinations.values())
    result: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "status": (
            "participant_credit_feasible_plan_next_screen"
            if all_passed
            else "participant_credit_not_feasible_stop"
        ),
        "passed": all_passed,
        "config": str(config),
        "collection": {
            "device": device,
            "num_envs": num_envs,
            "steps": steps,
            "data_seeds": list(data_seeds),
            "deterministic_actor_mean": True,
            "checkpoints": [str(path) for path in checkpoints],
        },
        "thresholds": {
            "collision_episodes": 100,
            "nonparticipant_fraction_mean": 0.25,
            "participant_count_median": 2.0,
            "repeated_recall_h8": 0.80,
            "repeated_recall_h16": 0.90,
            "first_repeated_lead_median_steps": 4.0,
        },
        "method": {
            "collision_pairs": "actual terminal positions below collision_distance",
            "repeated_signal_used_for_training": False,
            "reward_modified": False,
            "execution_modified": False,
            "optimizer_modified": False,
        },
        "combinations": combinations,
        "decision": (
            "preregister_one_participant_specific_collision_credit_screen"
            if all_passed
            else "do_not_train_participant_specific_collision_credit"
        ),
    }

    run_dir_path = (
        Path(run_dir)
        if run_dir is not None
        else ROOT / "outputs" / "runs" / EXPERIMENT_ID / DEFAULT_RUN_ID
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
    metrics_path = metrics_dir / "collision_participant_credit_feasibility.json"
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
        "producer": "scripts/analyze_collision_participant_credit_feasibility.py",
        "status": result["status"],
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
    suite_dir = run_dir_path.parent / "_suite"
    suite_metrics_dir = suite_dir / "metrics"
    suite_metrics_dir.mkdir(parents=True, exist_ok=True)
    suite_summary_path = suite_metrics_dir / "suite_summary.json"
    suite_summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    suite_manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "producer": "scripts/analyze_collision_participant_credit_feasibility.py",
        "status": result["status"],
        "artifacts": {
            "suite_summary": artifact_path(suite_summary_path),
            "run_manifest": artifact_path(run_dir_path / "run_manifest.json"),
        },
    }
    (suite_dir / "run_manifest.json").write_text(
        json.dumps(suite_manifest, indent=2), encoding="utf-8"
    )
    result["artifact"] = str(metrics_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--data-seeds", type=_parse_int_tuple, default=(32023, 33023))
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = analyze_collision_participant_credit_feasibility(
        config=args.config,
        checkpoints=tuple(args.checkpoint),
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        data_seeds=args.data_seeds,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
