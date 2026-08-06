#!/usr/bin/env python
"""Measure whether the existing near-distance credit precedes safety failures."""

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
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from play import _load_policy_players


EXPERIMENT_ID = "exp134_near_credit_lead_time"


def _distribution(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().float().flatten().cpu()
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
        }
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p10": float(torch.quantile(values, 0.10)),
        "p90": float(torch.quantile(values, 0.90)),
    }


class AgentLeadTimeTracker:
    """Track per-agent near activation, conflict events, and collision lead time."""

    def __init__(self, num_envs: int, n_agents: int, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.episode_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        shape = (num_envs, n_agents)
        self.near_first_step = torch.full(shape, -1, dtype=torch.long, device=self.device)
        self.conflict_active = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.conflict_start_step = torch.full(
            shape, -1, dtype=torch.long, device=self.device
        )
        self.conflict_covered = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.conflict_delay = torch.full(shape, -1, dtype=torch.long, device=self.device)
        self.conflict_near_at_onset = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self._collision_leads: list[torch.Tensor] = []
        self._conflict_covered_events: list[torch.Tensor] = []
        self._conflict_delays: list[torch.Tensor] = []
        self._conflict_near_at_onset_events: list[torch.Tensor] = []

    def _record_conflict_end(self, ending: torch.Tensor) -> None:
        if not ending.any():
            return
        covered = self.conflict_covered[ending]
        self._conflict_covered_events.append(covered.detach().cpu())
        self._conflict_near_at_onset_events.append(
            self.conflict_near_at_onset[ending].detach().cpu()
        )
        if covered.any():
            self._conflict_delays.append(
                self.conflict_delay[ending][covered].detach().cpu()
            )

    def update(
        self,
        *,
        near_before: torch.Tensor,
        near_after: torch.Tensor,
        predicted_conflict_involvement: torch.Tensor,
        collision_involvement: torch.Tensor,
        done: torch.Tensor,
        near_distance: float,
    ) -> None:
        expected = self.near_first_step.shape
        for name, value in (
            ("near_before", near_before),
            ("near_after", near_after),
            ("predicted_conflict_involvement", predicted_conflict_involvement),
            ("collision_involvement", collision_involvement),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}.")
        if done.shape != (expected[0],):
            raise ValueError(f"done must have shape {(expected[0],)}, got {done.shape}.")

        current_step = self.episode_step[:, None]
        near_before_active = near_before < float(near_distance)
        first_before = near_before_active & (self.near_first_step < 0)
        self.near_first_step = torch.where(
            first_before, current_step, self.near_first_step
        )

        predicted = predicted_conflict_involvement.bool()
        onset = predicted & ~self.conflict_active
        self.conflict_start_step = torch.where(
            onset, current_step, self.conflict_start_step
        )
        self.conflict_covered = torch.where(
            onset, near_before_active, self.conflict_covered
        )
        self.conflict_delay = torch.where(
            onset & near_before_active,
            torch.zeros_like(self.conflict_delay),
            torch.where(onset, torch.full_like(self.conflict_delay, -1), self.conflict_delay),
        )
        self.conflict_near_at_onset = torch.where(
            onset, near_before_active, self.conflict_near_at_onset
        )
        newly_covered = predicted & ~self.conflict_covered & near_before_active
        self.conflict_covered = self.conflict_covered | newly_covered
        self.conflict_delay = torch.where(
            newly_covered,
            current_step - self.conflict_start_step,
            self.conflict_delay,
        )

        resolved = self.conflict_active & ~predicted
        self._record_conflict_end(resolved)
        self.conflict_active = predicted

        near_after_active = near_after < float(near_distance)
        first_after = near_after_active & (self.near_first_step < 0)
        self.near_first_step = torch.where(
            first_after, current_step + 1, self.near_first_step
        )
        if collision_involvement.any():
            lead = torch.where(
                self.near_first_step >= 0,
                current_step + 1 - self.near_first_step,
                torch.full_like(self.near_first_step, -1),
            )
            self._collision_leads.append(
                lead[collision_involvement.bool()].detach().cpu()
            )

        ending_on_done = self.conflict_active & done[:, None]
        self._record_conflict_end(ending_on_done)

        if done.any():
            env_ids = torch.nonzero(done, as_tuple=False).flatten()
            self.near_first_step[env_ids] = -1
            self.conflict_active[env_ids] = False
            self.conflict_start_step[env_ids] = -1
            self.conflict_covered[env_ids] = False
            self.conflict_delay[env_ids] = -1
            self.conflict_near_at_onset[env_ids] = False
            self.episode_step[env_ids] = 0
        self.episode_step = torch.where(done, self.episode_step, self.episode_step + 1)

        inactive = ~self.conflict_active
        self.conflict_start_step = torch.where(
            inactive, torch.full_like(self.conflict_start_step, -1), self.conflict_start_step
        )
        self.conflict_covered = self.conflict_covered & self.conflict_active
        self.conflict_delay = torch.where(
            inactive, torch.full_like(self.conflict_delay, -1), self.conflict_delay
        )
        self.conflict_near_at_onset = (
            self.conflict_near_at_onset & self.conflict_active
        )

    def summary(self) -> dict[str, Any]:
        collision_leads = (
            torch.cat(self._collision_leads)
            if self._collision_leads
            else torch.empty(0, dtype=torch.long)
        )
        conflict_covered = (
            torch.cat(self._conflict_covered_events)
            if self._conflict_covered_events
            else torch.empty(0, dtype=torch.bool)
        )
        conflict_near_at_onset = (
            torch.cat(self._conflict_near_at_onset_events)
            if self._conflict_near_at_onset_events
            else torch.empty(0, dtype=torch.bool)
        )
        conflict_delays = (
            torch.cat(self._conflict_delays)
            if self._conflict_delays
            else torch.empty(0, dtype=torch.long)
        )
        collision_count = int(collision_leads.numel())
        conflict_count = int(conflict_covered.numel())
        return {
            "collision_involvement_events": collision_count,
            "collision_prior_near_fraction": float(
                (collision_leads >= 0).float().mean() if collision_count else 0.0
            ),
            "collision_lead_ge_1_fraction": float(
                (collision_leads >= 1).float().mean() if collision_count else 0.0
            ),
            "collision_lead_ge_2_fraction": float(
                (collision_leads >= 2).float().mean() if collision_count else 0.0
            ),
            "collision_lead_steps": _distribution(collision_leads[collision_leads >= 0]),
            "predicted_conflict_events": conflict_count,
            "predicted_conflict_near_at_onset_fraction": float(
                conflict_near_at_onset.float().mean() if conflict_count else 0.0
            ),
            "predicted_conflict_covered_fraction": float(
                conflict_covered.float().mean() if conflict_count else 0.0
            ),
            "predicted_conflict_coverage_delay_steps": _distribution(conflict_delays),
        }


def collect_lead_time_seed(
    *,
    config: str | Path,
    checkpoint_data: dict[str, Any],
    device: str,
    num_envs: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
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
    tracker = AgentLeadTimeTracker(num_envs, cfg.task.n_agents, env.device)
    for _ in range(steps):
        with torch.no_grad():
            near_before = env.metrics.nearest_neighbor_distance.clone()
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
            active_pairs = output.info["trajectory_conflicts"]["active"]
            conflict_involvement = active_pairs.any(dim=2) | active_pairs.any(dim=1)
            near_after = output.info["metrics"].nearest_neighbor_distance
            done_flags = output.info["done"]
            collision_involvement = (
                near_after < float(cfg.safety.collision_distance)
            ) & done_flags.collision[:, None]
            tracker.update(
                near_before=near_before,
                near_after=near_after,
                predicted_conflict_involvement=conflict_involvement,
                collision_involvement=collision_involvement,
                done=done_flags.done,
                near_distance=float(cfg.safety.near_distance),
            )
            actor_obs = output.actor_obs
    result = tracker.summary()
    result["near_distance_m"] = float(cfg.safety.near_distance)
    result["collision_distance_m"] = float(cfg.safety.collision_distance)
    result["predicted_conflict_distance_m"] = float(
        cfg.success_thresholds.min_pairwise_distance
    )
    return result


def analyze_near_credit_lead_time(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    num_envs: int = 128,
    steps: int = 512,
    data_seeds: tuple[int, ...] = (24023, 25023),
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_data = torch.load(checkpoint, map_location=torch.device(device))
    digest_before = _policy_digest(checkpoint_data)
    per_seed = {
        str(seed): collect_lead_time_seed(
            config=config,
            checkpoint_data=checkpoint_data,
            device=device,
            num_envs=num_envs,
            steps=steps,
            seed=seed,
        )
        for seed in data_seeds
    }
    digest_after = _policy_digest(checkpoint_data)
    values = list(per_seed.values())
    checks = {
        "every_seed_collision_events_ge_20": min(
            item["collision_involvement_events"] for item in values
        )
        >= 20,
        "every_seed_predicted_conflict_events_ge_200": min(
            item["predicted_conflict_events"] for item in values
        )
        >= 200,
        "every_seed_collision_prior_near_fraction_ge_0_95": min(
            item["collision_prior_near_fraction"] for item in values
        )
        >= 0.95,
        "every_seed_collision_lead_ge_1_fraction_ge_0_90": min(
            item["collision_lead_ge_1_fraction"] for item in values
        )
        >= 0.90,
        "every_seed_collision_lead_ge_2_fraction_ge_0_50": min(
            item["collision_lead_ge_2_fraction"] for item in values
        )
        >= 0.50,
        "every_seed_collision_lead_median_ge_2": min(
            item["collision_lead_steps"]["median"] for item in values
        )
        >= 2.0,
        "every_seed_predicted_conflict_covered_fraction_ge_0_70": min(
            item["predicted_conflict_covered_fraction"] for item in values
        )
        >= 0.70,
        "every_seed_conflict_coverage_delay_median_le_8": max(
            item["predicted_conflict_coverage_delay_steps"]["median"]
            for item in values
        )
        <= 8.0,
        "actor_checkpoint_unchanged": digest_before == digest_after,
    }
    passed = all(checks.values())
    result: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "status": "allow_c3_near_plan_only" if passed else "stop_distance_safety_credit",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "collection": {
            "device": device,
            "num_envs": num_envs,
            "steps": steps,
            "data_seeds": list(data_seeds),
        },
        "method": {
            "near_signal": "existing safety.near_distance state",
            "predicted_conflict": "existing non-intervening quintic trajectory diagnostic",
            "collision": "existing safety.collision_distance termination",
            "reward_or_execution_modified": False,
            "training_or_optimizer_modified": False,
        },
        "per_seed": per_seed,
        "checks": checks,
        "invariance": {
            "actor_digest_before": digest_before,
            "actor_digest_after": digest_after,
        },
        "decision": "draft_c3_near_plan_only" if passed else "stop_distance_credit_line",
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
    metrics_path = metrics_dir / "near_credit_lead_time.json"
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
        "producer": "scripts/analyze_near_credit_lead_time.py",
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
    parser.add_argument("--data-seeds", type=_parse_int_tuple, default=(24023, 25023))
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = analyze_near_credit_lead_time(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        data_seeds=args.data_seeds,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
