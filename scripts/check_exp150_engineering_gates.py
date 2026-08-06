#!/usr/bin/env python
"""Write machine-readable engineering gates for the exp150 component screen."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from train_skrl_mappo import (
    collision_participant_centered_credit,
    install_actor_credit_rewards,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringSKRLEnv,
)


EXPERIMENT_ID = "exp150_collision_participant_actor_credit"
CONFIG = ROOT / "configs" / "experiment" / "exp150_collision_participant_actor_credit.yaml"
BASE_CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp148_decentralized_b0_trajectory_time_consistent.yaml"
)
DEFAULT_CPU_RUN = "smoke_cpu_seed23_32env_256steps"
DEFAULT_CUDA_RUN = "smoke_cuda_seed23_256env_256steps"


def _run_dir(run_name: str) -> Path:
    return ROOT / "outputs" / "runs" / EXPERIMENT_ID / run_name


def _load_summary(run_name: str) -> dict:
    return json.loads(
        (_run_dir(run_name) / "metrics" / "summary.json").read_text(encoding="utf-8")
    )


def _load_telemetry(run_name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (
            _run_dir(run_name) / "metrics" / "train_metrics.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _smoke_checks(summary: dict, telemetry: list[dict], *, device: str) -> tuple[dict, dict]:
    diagnostics = summary.get("training_diagnostics") or {}
    train_rows = [row for row in telemetry if row.get("phase") == "train"]
    maximum = lambda key, default: max(  # noqa: E731
        (float(row.get(key, default)) for row in train_rows),
        default=default,
    )
    active_rate = maximum("actor_credit_active_rate", 0.0)
    participant_rate = maximum("actor_credit_collision_participant_rate", 0.0)
    event_rate = maximum("actor_credit_collision_event_rate", 0.0)
    source_error = maximum("actor_credit_source_reconstruction_error", float("inf"))
    reward_error = maximum(
        "actor_credit_team_reward_preservation_error", float("inf")
    )
    policy_sum_error = maximum(
        "actor_credit_policy_step_sum_abs_max", float("inf")
    )
    allocation_error = maximum(
        "actor_credit_allocation_mean_error", float("inf")
    )
    collision_zero_sum_error = maximum(
        "actor_credit_collision_zero_sum_error", float("inf")
    )
    checks = {
        f"{device}_status_ok": summary.get("status") == "ok",
        f"{device}_device": str(summary.get("device", "")).startswith(device),
        f"{device}_policy_finite": diagnostics.get("policy_parameters_finite") is True,
        f"{device}_actor_updated": float(
            diagnostics.get("policy_parameter_delta_l2", 0.0)
        )
        > 0.0,
        f"{device}_neighbor_encoder_updated": float(
            diagnostics.get("neighbor_encoder_parameter_delta_l2", 0.0)
        )
        > 0.0,
        f"{device}_terrain_encoder_updated": float(
            diagnostics.get("terrain_encoder_parameter_delta_l2", 0.0)
        )
        > 0.0,
        f"{device}_action_non_degenerate": float(
            diagnostics.get("post_training_action_std", 0.0)
        )
        > 1.0e-4,
        f"{device}_credit_trace_nonzero": float(
            diagnostics.get("last_actor_credit_std", 0.0)
        )
        > 1.0e-4,
        f"{device}_credit_activated": active_rate > 0.0
        and participant_rate > 0.0
        and event_rate > 0.0,
        f"{device}_source_reconstruction": source_error <= 1.0e-5,
        f"{device}_team_reward_preserved": reward_error <= 1.0e-6,
        f"{device}_policy_step_zero_sum": policy_sum_error <= 1.0e-5,
        f"{device}_allocation_mean_preserved": allocation_error <= 1.0e-5,
        f"{device}_collision_credit_zero_sum": collision_zero_sum_error <= 1.0e-5,
        f"{device}_assignment": diagnostics.get("actor_credit_assignment")
        == "collision_participant_centered",
        f"{device}_pure_rl": int(diagnostics.get("bc_updates", -1)) == 0
        and float(diagnostics.get("bc_parameter_delta_l2", -1.0)) == 0.0,
        f"{device}_no_collision_constraint": diagnostics.get(
            "collision_constraint_enabled"
        )
        is False,
    }
    evidence = {
        "maximum_active_rate": active_rate,
        "maximum_participant_rate": participant_rate,
        "maximum_collision_event_rate": event_rate,
        "maximum_source_reconstruction_error": source_error,
        "maximum_team_reward_preservation_error": reward_error,
        "maximum_policy_step_sum_error": policy_sum_error,
        "maximum_allocation_mean_error": allocation_error,
        "maximum_collision_zero_sum_error": collision_zero_sum_error,
        "training_diagnostics": diagnostics,
    }
    return checks, evidence


def _semantic_checks() -> tuple[dict, dict]:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    base = cfg_from_experiment(BASE_CONFIG)
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]]]
    )
    allocation = collision_participant_centered_credit(
        positions,
        torch.tensor([True]),
        collision_distance=float(cfg.safety.collision_distance),
        collision_penalty=155.0,
    )
    expected = torch.tensor([[-155.0, -155.0, 155.0, 155.0]])
    formula_error = float((allocation["policy"] - expected).abs().max())

    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 16
    reference = MultiRoverGatheringSKRLEnv(cfg)
    credited_cfg = cfg_from_experiment(CONFIG)
    credited_cfg.simulation.device = "cpu"
    credited_cfg.simulation.num_envs = 16
    credited = MultiRoverGatheringSKRLEnv(credited_cfg)
    install_actor_credit_rewards(credited, assignment="collision_participant_centered")
    actions = {
        agent: torch.zeros((cfg.simulation.num_envs, 2))
        for agent in reference.possible_agents
    }
    ref_out = reference.step(actions)
    credited_out = credited.step(actions)
    ref_rewards = torch.stack(
        [ref_out[1][agent] for agent in reference.possible_agents], dim=1
    )
    credited_rewards = torch.stack(
        [credited_out[1][agent] for agent in credited.possible_agents], dim=1
    )
    environment_reward_error = float((ref_rewards - credited_rewards).abs().max())
    actor_obs_error = max(
        float((ref_out[0][agent] - credited_out[0][agent]).abs().max())
        for agent in reference.possible_agents
    )
    control_error = float(
        (
            ref_out[4]["control"].packed
            - credited_out[4]["control"].packed
        ).abs().max()
    )
    unchanged_sections = {
        name: asdict(getattr(cfg_from_experiment(CONFIG), name))
        == asdict(getattr(base, name))
        for name in (
            "observation",
            "reward_coefficients",
            "reward_weights",
            "safety",
            "success_thresholds",
            "planner",
            "trajectory_generator",
            "low_level_control",
        )
    }
    algorithm = raw.get("algorithm") or {}
    evidence = {
        "single_pair_formula_max_abs_error": formula_error,
        "single_pair_zero_sum_error": float(allocation["zero_sum_error"].max()),
        "single_pair_allocation_mean_error": float(
            allocation["allocation_mean_error"].max()
        ),
        "environment_reward_error": environment_reward_error,
        "actor_observation_error": actor_obs_error,
        "control_error": control_error,
        "unchanged_config_sections": unchanged_sections,
    }
    checks = {
        "actor_obs_dim_101": cfg.actor_obs_dim == 101,
        "strict_decentralized_schema": cfg.observation.schema_version
        == "ego_v8_decentralized_tiered",
        "assignment_is_collision_participant": algorithm.get(
            "actor_credit_assignment"
        )
        == "collision_participant_centered",
        "fixed_scale_0_25": float(algorithm.get("actor_credit_scale", -1.0)) == 0.25,
        "pure_rl_random_init": int(algorithm.get("bc_updates", -1)) == 0
        and algorithm.get("init_checkpoint") is None,
        "collision_constraint_disabled": algorithm.get(
            "collision_constraint_enabled"
        )
        is False,
        "single_pair_formula_exact": formula_error <= 1.0e-5,
        "single_pair_zero_sum": evidence["single_pair_zero_sum_error"] <= 1.0e-5,
        "single_pair_mean_preserved": evidence[
            "single_pair_allocation_mean_error"
        ]
        <= 1.0e-5,
        "environment_reward_unchanged": environment_reward_error <= 1.0e-7,
        "actor_observation_unchanged": actor_obs_error <= 1.0e-7,
        "control_unchanged": control_error <= 1.0e-7,
        "environment_config_matches_exp148": all(unchanged_sections.values()),
        "physical_trajectory_timing": cfg.trajectory_generator.time_parameterization
        == "arc_length_reference_speed",
        "planning_time_tracking": cfg.low_level_control.tracking_point_mode
        == "planning_time",
        "subgoal_filter_disabled": not cfg.planner.subgoal_filter.enabled,
        "safety_projection_disabled": not cfg.low_level_control.safety_projection_enabled,
    }
    return checks, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-run", default=DEFAULT_CPU_RUN)
    parser.add_argument("--cuda-run", default=DEFAULT_CUDA_RUN)
    args = parser.parse_args()
    cpu_checks, cpu_evidence = _smoke_checks(
        _load_summary(args.cpu_run), _load_telemetry(args.cpu_run), device="cpu"
    )
    cuda_checks, cuda_evidence = _smoke_checks(
        _load_summary(args.cuda_run), _load_telemetry(args.cuda_run), device="cuda"
    )
    semantic_checks, semantic_evidence = _semantic_checks()
    checks = {**semantic_checks, **cpu_checks, **cuda_checks}
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "status": "engineering_gate_passed" if all(checks.values()) else "engineering_gate_failed",
        "passed": all(checks.values()),
        "training_authorized": all(checks.values()),
        "forty_million_authorized": False,
        "checks": checks,
        "thresholds": {
            "formula_and_zero_sum_error": 1.0e-5,
            "team_reward_error": 1.0e-6,
            "execution_invariance_error": 1.0e-7,
            "action_std": 1.0e-4,
            "credit_trace_std": 1.0e-4,
        },
        "evidence": {
            "semantics": semantic_evidence,
            "cpu_smoke": cpu_evidence,
            "cuda_smoke": cuda_evidence,
        },
        "runs": {"cpu": args.cpu_run, "cuda": args.cuda_run},
    }
    suite_dir = ROOT / "outputs" / "runs" / EXPERIMENT_ID / "_suite"
    output = suite_dir / "metrics" / "engineering_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "producer": "scripts/check_exp150_engineering_gates.py",
        "status": payload["status"],
        "artifacts": {
            "engineering_gate": str(output.relative_to(ROOT)),
        },
    }
    (suite_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
