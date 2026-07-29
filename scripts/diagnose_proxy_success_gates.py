#!/usr/bin/env python
"""Diagnose per-episode success-gate failures for a proxy checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from play import _load_policy_players
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)


REASON_CODES = {
    -1: "unfinished",
    0: "success",
    1: "collision",
    2: "out_of_bounds",
    3: "timeout",
}


def _resolve_output(output: str | Path | None, run_dir: str | Path | None) -> Path | None:
    if output is None and run_dir is not None:
        output = Path(run_dir) / "metrics" / "success_gate_diagnostics.json"
    if output is None:
        return None
    path = Path(output)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _register_diagnostic_artifact(
    run_dir: str | Path | None,
    output_path: Path,
) -> None:
    """Atomically add a successful diagnostic to an existing run manifest.

    Diagnostics are optional post-processing, so a missing manifest must not
    turn a valid analysis into a failure.  When a run manifest exists, however,
    it remains the source-of-truth index for the generated JSON.
    """
    if run_dir is None:
        return
    resolved_run_dir = Path(run_dir)
    if not resolved_run_dir.is_absolute():
        resolved_run_dir = ROOT / resolved_run_dir
    manifest_path = resolved_run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"Run manifest must be a JSON object: {manifest_path}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        manifest["artifacts"] = artifacts
    try:
        artifact_path = str(output_path.relative_to(ROOT))
    except ValueError:
        artifact_path = str(output_path)
    artifacts["metrics_success_gate_diagnostic"] = artifact_path
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
        dir=manifest_path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, manifest_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _min(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return min(values)


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(1.0 for row in rows if bool(row.get(key))) / len(rows)


def _optional_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [bool(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "final_dmax_mean": _mean(rows, "final_dmax"),
        "final_dispersion_mean": _mean(rows, "final_dispersion"),
        "final_mean_speed_mean": _mean(rows, "final_mean_speed"),
        "final_min_pairwise_mean": _mean(rows, "final_min_pairwise"),
        "final_min_pairwise_min": _min(rows, "final_min_pairwise"),
        "final_success_hold_count_mean": _mean(rows, "final_success_hold_count"),
        "max_success_hold_count_mean": _mean(rows, "max_success_hold_count"),
        "final_terrain_speed_scale_mean": _mean(rows, "final_terrain_speed_scale"),
        "dmax_ok_rate": _rate(rows, "final_dmax_ok"),
        "dispersion_ok_rate": _rate(rows, "final_dispersion_ok"),
        "speed_ok_rate": _rate(rows, "final_speed_ok"),
        "min_pairwise_ok_rate": _rate(rows, "final_min_pairwise_ok"),
        "flatness_ok_rate": _optional_rate(rows, "final_flatness_ok"),
        "gather_point_is_flat_rate": _optional_rate(
            rows,
            "final_gather_point_is_flat",
        ),
        "final_gather_point_height_range_mean": _mean(
            rows,
            "final_gather_point_height_range",
        ),
        "final_gather_point_max_slope_mean": _mean(
            rows,
            "final_gather_point_max_slope",
        ),
        "final_gather_point_mean_slope_mean": _mean(
            rows,
            "final_gather_point_mean_slope",
        ),
        "instant_success_rate": _rate(rows, "final_instant_success"),
    }


def summarize_episode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_reason = {
        reason: [row for row in rows if row["done_reason"] == reason]
        for reason in REASON_CODES.values()
    }
    timeout_rows = by_reason["timeout"]
    gate_failure_counts = {
        "dmax": sum(1 for row in timeout_rows if not row["final_dmax_ok"]),
        "dispersion": sum(1 for row in timeout_rows if not row["final_dispersion_ok"]),
        "speed": sum(1 for row in timeout_rows if not row["final_speed_ok"]),
        "min_pairwise": sum(1 for row in timeout_rows if not row["final_min_pairwise_ok"]),
        "instant_success": sum(
            1 for row in timeout_rows if not row["final_instant_success"]
        ),
    }
    flatness_rows = [
        row for row in timeout_rows if row.get("final_flatness_ok") is not None
    ]
    if flatness_rows:
        gate_failure_counts["flatness"] = sum(
            1 for row in flatness_rows if not bool(row["final_flatness_ok"])
        )
    return {
        "num_envs": len(rows),
        "counts_by_reason": {reason: len(items) for reason, items in by_reason.items()},
        "all": _group_summary(rows),
        "by_reason": {
            reason: _group_summary(items) for reason, items in by_reason.items()
        },
        "timeout_final_gate_failure_counts": gate_failure_counts,
        "timeout_rows": timeout_rows,
    }


def diagnose_checkpoint(
    config: str | Path,
    checkpoint: str | Path,
    *,
    device: str | None = None,
    num_envs: int = 1024,
    steps: int = 320,
    seed: int | None = None,
    output: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    raw_cfg = load_yaml(config)
    cfg = cfg_from_experiment(config)
    cfg.simulation.num_envs = int(num_envs)
    if device is not None:
        cfg.simulation.device = device
    if seed is not None:
        cfg.seed = int(seed)

    map_location = torch.device(cfg.simulation.device)
    if map_location.type == "cuda" and not torch.cuda.is_available():
        map_location = torch.device("cpu")
    checkpoint_data = torch.load(checkpoint, map_location=map_location)
    metadata = checkpoint_data.get("metadata", {}) if isinstance(checkpoint_data, dict) else {}
    if cfg.planner.subgoal_filter.mode in {
        "terrain_safe_candidate_curriculum",
        "terrain_safe_candidate_constrained_curriculum",
        "terrain_safe_candidate_soft_progress_curriculum",
        "terrain_safe_candidate_mutual_progress_curriculum",
        "terrain_safe_candidate_hold_progress_curriculum",
    }:
        cfg.planner.subgoal_filter.progress_timestep_override = int(metadata.get("timesteps", 0))
        cfg.planner.subgoal_filter.deterministic_eval = True

    env = MultiRoverGatheringCore(cfg)
    act, backend = _load_policy_players(checkpoint_data, cfg, env.device, raw_cfg=raw_cfg)
    actor_obs, _ = env.get_observations()

    initial_dmax = env.metrics.dmax.detach().clone()
    initial_dispersion = env.metrics.dispersion.detach().clone()
    final_dmax = initial_dmax.clone()
    final_dispersion = initial_dispersion.clone()
    final_mean_speed = env.metrics.mean_speed.detach().clone()
    final_min_pairwise = env.metrics.nearest_neighbor_distance.amin(dim=-1).detach().clone()
    final_success_hold = env.success_hold_count.detach().clone()
    max_success_hold = env.success_hold_count.detach().clone()
    final_terrain_speed_scale = torch.ones(env.num_envs, device=env.device)
    initial_gather_point_flatness = env.evaluate_current_gather_point_flatness(env.metrics)
    initial_flatness_ok = (
        initial_gather_point_flatness.is_flat
        if cfg.gather_point.require_flat_for_success
        else torch.ones_like(initial_gather_point_flatness.is_flat)
    )
    final_flatness_ok = initial_flatness_ok.detach().clone()
    final_gather_point_is_flat = initial_gather_point_flatness.is_flat.detach().clone()
    final_gather_point_height_range = (
        initial_gather_point_flatness.height_range.detach().clone()
    )
    final_gather_point_max_slope = initial_gather_point_flatness.max_slope.detach().clone()
    final_gather_point_mean_slope = initial_gather_point_flatness.mean_slope.detach().clone()
    oracle_search_feasible = env.oracle_search_feasible.detach().clone()
    oracle_search_objective = env.oracle_search_objective.detach().clone()
    oracle_search_mean_distance = env.oracle_search_mean_distance.detach().clone()
    oracle_search_max_distance = env.oracle_search_max_distance.detach().clone()
    oracle_search_path_risk = env.oracle_search_path_risk.detach().clone()
    oracle_search_path_height_change = (
        env.oracle_search_path_height_change.detach().clone()
    )
    oracle_search_height_range = env.oracle_search_height_range.detach().clone()
    oracle_search_max_slope = env.oracle_search_max_slope.detach().clone()
    final_dmax_ok = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    final_dispersion_ok = torch.zeros_like(final_dmax_ok)
    final_speed_ok = torch.zeros_like(final_dmax_ok)
    final_min_pairwise_ok = torch.zeros_like(final_dmax_ok)
    final_instant_success = torch.zeros_like(final_dmax_ok)

    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    done_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    done_reason = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    for step_id in range(int(steps)):
        active_before = active.clone()
        with torch.no_grad():
            action = act(actor_obs)
        step_output = env.step(action)
        actor_obs = step_output.actor_obs
        metrics = step_output.info["metrics"]
        gates = step_output.info["success_gates"]
        done = step_output.info["done"]

        nearest = metrics.nearest_neighbor_distance.amin(dim=-1)
        success_hold = step_output.info["success_hold_count"]
        final_dmax = torch.where(active_before, metrics.dmax, final_dmax)
        final_dispersion = torch.where(active_before, metrics.dispersion, final_dispersion)
        final_mean_speed = torch.where(active_before, metrics.mean_speed, final_mean_speed)
        final_min_pairwise = torch.where(active_before, nearest, final_min_pairwise)
        final_success_hold = torch.where(active_before, success_hold, final_success_hold)
        max_success_hold = torch.maximum(
            max_success_hold,
            torch.where(active_before, success_hold, torch.zeros_like(success_hold)),
        )
        final_dmax_ok = torch.where(active_before, gates.dmax_ok, final_dmax_ok)
        final_dispersion_ok = torch.where(
            active_before,
            gates.dispersion_ok,
            final_dispersion_ok,
        )
        final_speed_ok = torch.where(active_before, gates.speed_ok, final_speed_ok)
        final_min_pairwise_ok = torch.where(
            active_before,
            gates.min_pairwise_ok,
            final_min_pairwise_ok,
        )
        final_flatness_ok = torch.where(
            active_before,
            gates.flatness_ok,
            final_flatness_ok,
        )
        gather_point_flatness = step_output.info["gather_point_flatness"]
        final_gather_point_is_flat = torch.where(
            active_before,
            gather_point_flatness.is_flat,
            final_gather_point_is_flat,
        )
        final_gather_point_height_range = torch.where(
            active_before,
            gather_point_flatness.height_range,
            final_gather_point_height_range,
        )
        final_gather_point_max_slope = torch.where(
            active_before,
            gather_point_flatness.max_slope,
            final_gather_point_max_slope,
        )
        final_gather_point_mean_slope = torch.where(
            active_before,
            gather_point_flatness.mean_slope,
            final_gather_point_mean_slope,
        )
        final_instant_success = torch.where(
            active_before,
            gates.instant_success,
            final_instant_success,
        )
        terrain_speed_scale = step_output.info.get("terrain_speed_scale")
        if terrain_speed_scale is not None:
            per_env_speed_scale = terrain_speed_scale.mean(dim=-1)
            final_terrain_speed_scale = torch.where(
                active_before,
                per_env_speed_scale,
                final_terrain_speed_scale,
            )

        first_done = done.done & active_before
        reason = torch.full_like(done_reason, -1)
        reason = torch.where(done.truncated, torch.full_like(reason, 3), reason)
        reason = torch.where(done.out_of_bounds, torch.full_like(reason, 2), reason)
        reason = torch.where(done.collision, torch.full_like(reason, 1), reason)
        reason = torch.where(done.success, torch.full_like(reason, 0), reason)
        done_step = torch.where(first_done, torch.full_like(done_step, step_id + 1), done_step)
        done_reason = torch.where(first_done, reason, done_reason)
        active = active & ~done.done
        if not active.any():
            break

    rows = []
    for env_id in range(env.num_envs):
        reason = int(done_reason[env_id].detach().cpu())
        rows.append(
            {
                "env_id": env_id,
                "done_reason": REASON_CODES.get(reason, "unknown"),
                "done_step": int(done_step[env_id].detach().cpu())
                if done_step[env_id] > 0
                else None,
                "initial_dmax": float(initial_dmax[env_id].detach().cpu()),
                "final_dmax": float(final_dmax[env_id].detach().cpu()),
                "initial_dispersion": float(initial_dispersion[env_id].detach().cpu()),
                "final_dispersion": float(final_dispersion[env_id].detach().cpu()),
                "final_mean_speed": float(final_mean_speed[env_id].detach().cpu()),
                "final_min_pairwise": float(final_min_pairwise[env_id].detach().cpu()),
                "final_success_hold_count": int(final_success_hold[env_id].detach().cpu()),
                "max_success_hold_count": int(max_success_hold[env_id].detach().cpu()),
                "final_terrain_speed_scale": float(
                    final_terrain_speed_scale[env_id].detach().cpu()
                ),
                "final_dmax_ok": bool(final_dmax_ok[env_id].detach().cpu()),
                "final_dispersion_ok": bool(final_dispersion_ok[env_id].detach().cpu()),
                "final_speed_ok": bool(final_speed_ok[env_id].detach().cpu()),
                "final_min_pairwise_ok": bool(final_min_pairwise_ok[env_id].detach().cpu()),
                "final_flatness_ok": bool(final_flatness_ok[env_id].detach().cpu()),
                "final_gather_point_is_flat": bool(
                    final_gather_point_is_flat[env_id].detach().cpu()
                ),
                "final_gather_point_height_range": float(
                    final_gather_point_height_range[env_id].detach().cpu()
                ),
                "final_gather_point_max_slope": float(
                    final_gather_point_max_slope[env_id].detach().cpu()
                ),
                "final_gather_point_mean_slope": float(
                    final_gather_point_mean_slope[env_id].detach().cpu()
                ),
                "final_instant_success": bool(final_instant_success[env_id].detach().cpu()),
            }
        )

    result = {
        "status": "ok",
        "backend": backend,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "device": str(env.device),
        "seed": cfg.seed,
        "steps": int(steps),
        "success_thresholds": {
            "dmax": cfg.success_thresholds.dmax,
            "dispersion": cfg.success_thresholds.dispersion,
            "speed": cfg.success_thresholds.speed,
            "hold_steps": cfg.success_thresholds.hold_steps,
            "min_pairwise_distance": cfg.success_thresholds.min_pairwise_distance,
        },
        "gather_point_flatness": {
            "require_flat_for_success": cfg.gather_point.require_flat_for_success,
            "radius": cfg.gather_point.flatness_radius,
            "rings": cfg.gather_point.flatness_rings,
            "samples_per_ring": cfg.gather_point.flatness_samples_per_ring,
            "max_height_range": cfg.gather_point.max_height_range,
            "max_slope": cfg.gather_point.max_slope,
        },
        "oracle_search": {
            "method": cfg.gather_point.search_method,
            "feasible_rate": float(
                oracle_search_feasible.float().mean().detach().cpu()
            ),
            "objective_mean": float(oracle_search_objective.mean().detach().cpu()),
            "mean_distance": float(oracle_search_mean_distance.mean().detach().cpu()),
            "max_distance": float(oracle_search_max_distance.mean().detach().cpu()),
            "path_risk_mean": float(oracle_search_path_risk.mean().detach().cpu()),
            "path_height_change_mean": float(
                oracle_search_path_height_change.mean().detach().cpu()
            ),
            "height_range_mean": float(
                oracle_search_height_range.mean().detach().cpu()
            ),
            "max_slope_mean": float(oracle_search_max_slope.mean().detach().cpu()),
        },
        "summary": summarize_episode_rows(rows),
        "episodes": rows,
    }

    output_path = _resolve_output(output, run_dir)
    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        result["artifact"] = str(output_path)
        _register_diagnostic_artifact(run_dir, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = diagnose_checkpoint(
        args.config,
        args.checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        output=args.output,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
