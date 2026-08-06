#!/usr/bin/env python
"""Audit the episode-level repeated-conflict trigger required before B2."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from _common import ROOT
from analyze_joint_action_critic_feasibility import _parse_int_tuple
from analyze_near_credit_lead_time import _distribution
from analyze_paired_action_interventions import _policy_digest
from analyze_repeated_conflict_outcomes import collect_conflict_timeline


EXPERIMENT_ID = "exp136_failed_episode_repeated_conflicts"


def summarize_failed_episode_repeats(
    *,
    repeated_pairs: torch.Tensor,
    done: torch.Tensor,
    success: torch.Tensor,
    collision_done: torch.Tensor,
    out_of_bounds: torch.Tensor,
    timeout: torch.Tensor,
) -> dict[str, Any]:
    """Count repeated-pair event onsets in completed episodes, including zeros."""

    if repeated_pairs.ndim != 4 or repeated_pairs.shape[-1] != repeated_pairs.shape[-2]:
        raise ValueError(
            "repeated_pairs must have shape [time, environment, agent, agent]."
        )
    expected = repeated_pairs.shape[:2]
    for name, value in (
        ("done", done),
        ("success", success),
        ("collision_done", collision_done),
        ("out_of_bounds", out_of_bounds),
        ("timeout", timeout),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}.")

    repeated_pairs = repeated_pairs.bool().cpu()
    done = done.bool().cpu()
    success = success.bool().cpu()
    collision_done = collision_done.bool().cpu()
    out_of_bounds = out_of_bounds.bool().cpu()
    timeout = timeout.bool().cpu()
    _, num_envs, n_agents, _ = repeated_pairs.shape
    previous = torch.zeros((num_envs, n_agents, n_agents), dtype=torch.bool)
    episode_counts = torch.zeros(num_envs, dtype=torch.long)
    rows: list[dict[str, Any]] = []
    for step in range(repeated_pairs.shape[0]):
        current = repeated_pairs[step]
        onset = current & ~previous
        episode_counts += onset.sum(dim=(1, 2))
        completed_ids = torch.nonzero(done[step], as_tuple=False).flatten().tolist()
        for env_id in completed_ids:
            is_success = bool(success[step, env_id])
            if is_success:
                reason = "success"
            elif bool(collision_done[step, env_id]):
                reason = "collision"
            elif bool(out_of_bounds[step, env_id]):
                reason = "out_of_bounds"
            elif bool(timeout[step, env_id]):
                reason = "timeout"
            else:
                reason = "failure_other"
            rows.append(
                {
                    "success": is_success,
                    "reason": reason,
                    "repeated_pair_conflict_events": int(episode_counts[env_id]),
                }
            )
        if completed_ids:
            env_ids = torch.tensor(completed_ids, dtype=torch.long)
            previous[env_ids] = False
            episode_counts[env_ids] = 0
        active_ids = ~done[step]
        previous[active_ids] = current[active_ids]

    failed_rows = [row for row in rows if not row["success"]]
    failed_counts = torch.tensor(
        [row["repeated_pair_conflict_events"] for row in failed_rows],
        dtype=torch.long,
    )
    by_reason: dict[str, Any] = {}
    for reason in ("collision", "out_of_bounds", "timeout", "failure_other"):
        reason_counts = torch.tensor(
            [
                row["repeated_pair_conflict_events"]
                for row in failed_rows
                if row["reason"] == reason
            ],
            dtype=torch.long,
        )
        if reason_counts.numel() == 0:
            continue
        by_reason[reason] = {
            "episodes": int(reason_counts.numel()),
            "with_repeated_fraction": float((reason_counts > 0).float().mean()),
            "event_count": _distribution(reason_counts),
        }
    return {
        "completed_episodes": len(rows),
        "success_episodes": sum(row["success"] for row in rows),
        "failed_episodes": len(failed_rows),
        "failed_with_repeated_fraction": float(
            (failed_counts > 0).float().mean() if failed_counts.numel() else 0.0
        ),
        "failed_repeated_event_count": _distribution(failed_counts),
        "by_reason": by_reason,
    }


def analyze_failed_episode_repeated_conflicts(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    num_envs: int = 128,
    steps: int = 512,
    data_seeds: tuple[int, ...] = (28023, 29023),
    run_dir: str | Path | None = None,
    output_experiment: str = EXPERIMENT_ID,
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
        per_seed[str(seed)] = summarize_failed_episode_repeats(
            repeated_pairs=timeline["repeated_pairs"],
            done=timeline["done"],
            success=timeline["success"],
            collision_done=timeline["collision_done"],
            out_of_bounds=timeline["out_of_bounds"],
            timeout=timeline["timeout"],
        )
    digest_after = _policy_digest(checkpoint_data)
    values = list(per_seed.values())
    checks = {
        "every_seed_failed_episodes_ge_100": min(
            item["failed_episodes"] for item in values
        )
        >= 100,
        "every_seed_failed_with_repeated_fraction_ge_0_20": min(
            item["failed_with_repeated_fraction"] for item in values
        )
        >= 0.20,
        "every_seed_failed_repeated_event_median_ge_2": min(
            item["failed_repeated_event_count"]["median"] for item in values
        )
        >= 2.0,
        "actor_checkpoint_unchanged": digest_before == digest_after,
    }
    trigger_met = all(checks.values())
    result: dict[str, Any] = {
        "experiment": output_experiment,
        "status": (
            "b2_conflict_trigger_met_base_not_converged"
            if trigger_met
            else "b2_conflict_trigger_not_met"
        ),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "collection": {
            "device": device,
            "num_envs": num_envs,
            "steps": steps,
            "data_seeds": list(data_seeds),
        },
        "method": {
            "event": "pair-level repeated false-to-true onset",
            "failed_episode": "completed episode with success=false",
            "right_censored_tail_excluded": True,
            "reward_or_execution_modified": False,
            "training_or_optimizer_modified": False,
        },
        "per_seed": per_seed,
        "checks": checks,
        "b2_enablement_matrix": {
            "repeated_conflict_metric_validated_by_exp135": True,
            "failed_episode_conflict_trigger": trigger_met,
            "b0_or_b1_base_convergence": False,
            "b2_implementation_allowed": False,
        },
        "invariance": {
            "actor_digest_before": digest_before,
            "actor_digest_after": digest_after,
        },
        "decision": "do_not_implement_b2_before_base_convergence",
    }
    run_dir_path = (
        Path(run_dir)
        if run_dir is not None
        else ROOT / "outputs" / "runs" / output_experiment / "frozen_exp125_seed23"
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
    metrics_path = metrics_dir / "failed_episode_repeated_conflicts.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def artifact_path(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": output_experiment,
        "producer": "scripts/analyze_failed_episode_repeated_conflicts.py",
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
    parser.add_argument("--data-seeds", type=_parse_int_tuple, default=(28023, 29023))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output-experiment", default=EXPERIMENT_ID)
    args = parser.parse_args()
    result = analyze_failed_episode_repeated_conflicts(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        data_seeds=args.data_seeds,
        run_dir=args.run_dir,
        output_experiment=args.output_experiment,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
