#!/usr/bin/env python
"""Write machine-readable engineering gates for the exp140 component screen."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from train_skrl_mappo import install_actor_credit_rewards
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringSKRLEnv,
)


EXPERIMENT_ID = "exp140_agent_local_near_credit"
CONFIG = ROOT / "configs" / "experiment" / "exp140_agent_local_near_credit.yaml"
BASE_CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp125_decentralized_tiered_b0_pure_rl_relative_quintic.yaml"
)
CENTERED_CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp126_decentralized_b0_centered_terrain_credit.yaml"
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
    maximum_active_rate = max(
        (float(row.get("actor_credit_active_rate", 0.0)) for row in train_rows),
        default=0.0,
    )
    maximum_source_error = max(
        (
            float(row.get("actor_credit_source_reconstruction_error", float("inf")))
            for row in train_rows
        ),
        default=float("inf"),
    )
    maximum_reward_error = max(
        (
            float(row.get("actor_credit_team_reward_preservation_error", float("inf")))
            for row in train_rows
        ),
        default=float("inf"),
    )
    policy_zero_sum_values = {
        float(row.get("actor_credit_policy_is_step_zero_sum", 1.0))
        for row in train_rows
    }
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
        f"{device}_credit_activated": maximum_active_rate > 0.0,
        f"{device}_source_reconstruction": maximum_source_error <= 1.0e-6,
        f"{device}_team_reward_preserved": maximum_reward_error <= 1.0e-6,
        f"{device}_policy_credit_not_step_zero_sum": policy_zero_sum_values == {0.0},
        f"{device}_assignment": diagnostics.get("actor_credit_assignment")
        == "near_potential_local",
        f"{device}_pure_rl": int(diagnostics.get("bc_updates", -1)) == 0
        and float(diagnostics.get("bc_parameter_delta_l2", -1.0)) == 0.0,
    }
    evidence = {
        "maximum_active_rate": maximum_active_rate,
        "maximum_source_reconstruction_error": maximum_source_error,
        "maximum_team_reward_preservation_error": maximum_reward_error,
        "policy_step_zero_sum_values": sorted(policy_zero_sum_values),
        "training_diagnostics": diagnostics,
    }
    return checks, evidence


def _semantic_checks() -> tuple[dict, dict]:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    base = cfg_from_experiment(BASE_CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 16
    env = MultiRoverGatheringSKRLEnv(cfg)
    nearest_before = env.core.metrics.nearest_neighbor_distance.clone()
    install_actor_credit_rewards(env, assignment="near_potential_local")
    actions = {
        agent: torch.zeros((cfg.simulation.num_envs, 2))
        for agent in env.possible_agents
    }
    _, rewards, _, _, info = env.step(actions)
    nearest_after = info["metrics"].nearest_neighbor_distance
    expected = -torch.relu(float(cfg.safety.near_distance) - nearest_after) + torch.relu(
        float(cfg.safety.near_distance) - nearest_before
    )
    credit = info["actor_credit"]
    reward_matrix = torch.stack([rewards[agent] for agent in env.possible_agents], dim=1)
    formula_error = float((credit["raw"] - expected).abs().amax())
    reward_error = float(
        (reward_matrix.mean(dim=1) - info["reward_terms"].total).abs().amax()
    )

    centered_cfg = cfg_from_experiment(CENTERED_CONFIG)
    centered_cfg.simulation.device = "cpu"
    centered_cfg.simulation.num_envs = 16
    centered_env = MultiRoverGatheringSKRLEnv(centered_cfg)
    install_actor_credit_rewards(centered_env, assignment="terrain_relative_centered")
    centered_actions = {
        agent: torch.zeros((centered_cfg.simulation.num_envs, 2))
        for agent in centered_env.possible_agents
    }
    _, _, _, _, centered_info = centered_env.step(centered_actions)
    centered_zero_sum_error = float(
        centered_info["actor_credit"]["policy"].sum(dim=1).abs().amax()
    )

    unchanged_sections = {
        "observation": asdict(cfg.observation) == asdict(base.observation),
        "reward_coefficients": asdict(cfg.reward_coefficients)
        == asdict(base.reward_coefficients),
        "reward_weights": asdict(cfg.reward_weights) == asdict(base.reward_weights),
        "safety": asdict(cfg.safety) == asdict(base.safety),
        "success_thresholds": asdict(cfg.success_thresholds)
        == asdict(base.success_thresholds),
        "planner": asdict(cfg.planner) == asdict(base.planner),
        "low_level_control": asdict(cfg.low_level_control)
        == asdict(base.low_level_control),
    }
    algorithm = raw.get("algorithm") or {}
    evidence = {
        "raw_formula_max_abs_error": formula_error,
        "source_reconstruction_max_abs_error": float(
            credit["source_reconstruction_error"].amax()
        ),
        "team_reward_max_abs_error": reward_error,
        "historical_centered_zero_sum_max_abs_error": centered_zero_sum_error,
        "unchanged_config_sections": unchanged_sections,
    }
    checks = {
        "actor_obs_dim_101": cfg.actor_obs_dim == 101,
        "strict_decentralized_schema": cfg.observation.schema_version
        == "ego_v8_decentralized_tiered",
        "assignment_is_local_near": algorithm.get("actor_credit_assignment")
        == "near_potential_local",
        "fixed_scale_0_25": float(algorithm.get("actor_credit_scale", -1.0)) == 0.25,
        "pure_rl_random_init": int(algorithm.get("bc_updates", -1)) == 0
        and algorithm.get("init_checkpoint") is None,
        "raw_formula_exact": formula_error <= 1.0e-6,
        "source_reconstruction_exact": evidence[
            "source_reconstruction_max_abs_error"
        ]
        <= 1.0e-6,
        "team_reward_unchanged": reward_error <= 1.0e-6,
        "policy_credit_not_step_zero_sum": credit["policy_is_step_zero_sum"] is False,
        "historical_centered_credit_preserved": centered_zero_sum_error <= 1.0e-6,
        "environment_config_matches_b0": all(unchanged_sections.values()),
        "subgoal_filter_disabled": not cfg.planner.subgoal_filter.enabled,
        "safety_projection_disabled": not cfg.low_level_control.safety_projection_enabled,
    }
    return checks, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-run", default=DEFAULT_CPU_RUN)
    parser.add_argument("--cuda-run", default=DEFAULT_CUDA_RUN)
    args = parser.parse_args()

    cpu_summary = _load_summary(args.cpu_run)
    cuda_summary = _load_summary(args.cuda_run)
    cpu_checks, cpu_evidence = _smoke_checks(
        cpu_summary,
        _load_telemetry(args.cpu_run),
        device="cpu",
    )
    cuda_checks, cuda_evidence = _smoke_checks(
        cuda_summary,
        _load_telemetry(args.cuda_run),
        device="cuda",
    )
    semantic_checks, semantic_evidence = _semantic_checks()
    checks = {**semantic_checks, **cpu_checks, **cuda_checks}
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "status": "engineering_gate_passed" if all(checks.values()) else "engineering_gate_failed",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "formula_error": 1.0e-6,
            "team_reward_error": 1.0e-6,
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
    output = (
        ROOT
        / "outputs"
        / "runs"
        / EXPERIMENT_ID
        / "_suite"
        / "metrics"
        / "engineering_gate.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
