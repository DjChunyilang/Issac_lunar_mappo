#!/usr/bin/env python
"""Write machine-readable engineering gates for exp142."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from _common import ROOT, cfg_from_experiment, load_yaml


EXPERIMENT_ID = "exp142_collision_lagrangian_component"
CONFIG = ROOT / "configs/experiment/exp142_collision_lagrangian_component.yaml"
BASE_CONFIG = ROOT / "configs/experiment/exp125_decentralized_tiered_b0_pure_rl_relative_quintic.yaml"


def _summary(run_name: str) -> dict:
    path = ROOT / "outputs/runs" / EXPERIMENT_ID / run_name / "metrics/summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _smoke_checks(summary: dict, *, device: str) -> tuple[dict, dict]:
    diagnostics = summary.get("training_diagnostics") or {}
    history = diagnostics.get("collision_constraint_history") or []
    finite_history = bool(history) and all(
        all(
            math.isfinite(float(row[key]))
            for key in (
                "episode_equivalent_collision_rate",
                "lagrangian_multiplier",
                "cost_value_loss",
            )
        )
        for row in history
    )
    checks = {
        f"{device}_status_ok": summary.get("status") == "ok",
        f"{device}_device": str(summary.get("device", "")).startswith(device),
        f"{device}_two_optimizers": diagnostics.get("optimizer_count") == 2,
        f"{device}_actor_updated": float(
            diagnostics.get("policy_parameter_delta_l2", 0.0)
        )
        > 0.0,
        f"{device}_reward_critic_updated": float(
            diagnostics.get("reward_critic_parameter_delta_l2", 0.0)
        )
        > 0.0,
        f"{device}_cost_critic_updated": float(
            diagnostics.get("collision_cost_value_parameter_delta_l2", 0.0)
        )
        > 0.0,
        f"{device}_all_expected_updates": int(
            diagnostics.get("collision_cost_critic_update_count", -1)
        )
        == int(diagnostics.get("joint_update_count", -2)),
        f"{device}_policy_finite": diagnostics.get("policy_parameters_finite") is True,
        f"{device}_cost_critic_finite": diagnostics.get(
            "collision_cost_value_parameters_finite"
        )
        is True,
        f"{device}_constraint_history_finite": finite_history,
        f"{device}_pure_rl": int(diagnostics.get("bc_updates", -1)) == 0
        and float(diagnostics.get("bc_parameter_delta_l2", -1.0)) == 0.0,
        f"{device}_no_actor_credit": diagnostics.get("actor_credit_assignment") == "none"
        and float(diagnostics.get("actor_credit_scale", -1.0)) == 0.0,
    }
    evidence = {
        "training_diagnostics": diagnostics,
        "history_length": len(history),
        "maximum_observed_collision_rate": max(
            (float(row["episode_equivalent_collision_rate"]) for row in history),
            default=0.0,
        ),
    }
    return checks, evidence


def _semantic_checks(cuda_run: str) -> tuple[dict, dict]:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    base = cfg_from_experiment(BASE_CONFIG)
    unchanged_sections = {
        name: asdict(getattr(cfg, name)) == asdict(getattr(base, name))
        for name in (
            "observation",
            "reward_coefficients",
            "reward_weights",
            "safety",
            "success_thresholds",
            "planner",
            "low_level_control",
            "initial_state",
        )
    }
    checkpoint_path = (
        ROOT / "outputs/runs" / EXPERIMENT_ID / cuda_run / "checkpoints/best.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    constraint = checkpoint.get("collision_constraint") or {}
    algorithm = raw.get("algorithm") or {}
    checks = {
        "actor_obs_dim_101": cfg.actor_obs_dim == 101,
        "critic_state_dim_54": cfg.critic_state_dim == 54,
        "strict_decentralized_schema": cfg.observation.schema_version
        == "ego_v8_decentralized_tiered",
        "environment_config_matches_b0": all(unchanged_sections.values()),
        "constraint_enabled": algorithm.get("collision_constraint_enabled") is True,
        "fixed_cost_definition_parameters": float(
            algorithm.get("collision_cost_limit", -1.0)
        )
        == 0.02
        and int(algorithm.get("collision_episode_steps", -1)) == 480,
        "fixed_dual_parameters": float(algorithm.get("lagrangian_init", -1.0)) == 0.0
        and float(algorithm.get("lagrangian_learning_rate", -1.0)) == 0.1
        and float(algorithm.get("lagrangian_max", -1.0)) == 2.0,
        "pure_rl_random_init": int(algorithm.get("bc_updates", -1)) == 0
        and algorithm.get("init_checkpoint") is None,
        "no_actor_credit": algorithm.get("actor_credit_assignment") == "none"
        and float(algorithm.get("actor_credit_scale", -1.0)) == 0.0,
        "subgoal_filter_disabled": not cfg.planner.subgoal_filter.enabled,
        "safety_projection_disabled": not cfg.low_level_control.safety_projection_enabled,
        "checkpoint_contains_cost_critic": isinstance(
            constraint.get("cost_value"), dict
        )
        and bool(constraint.get("cost_value")),
        "checkpoint_contains_finite_lambda": math.isfinite(
            float(constraint.get("lagrangian_multiplier", float("nan")))
        ),
        "execution_metadata_remains_101_dim": int(
            checkpoint.get("metadata", {}).get("actor_obs_dim", -1)
        )
        == 101,
    }
    evidence = {
        "unchanged_config_sections": unchanged_sections,
        "checkpoint_lagrangian_multiplier": constraint.get("lagrangian_multiplier"),
        "checkpoint_cost_value_tensor_count": len(constraint.get("cost_value") or {}),
    }
    return checks, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-run", default="cpu_smoke_32x128")
    parser.add_argument("--cuda-run", default="cuda_smoke_256x256")
    args = parser.parse_args()
    cpu_checks, cpu_evidence = _smoke_checks(_summary(args.cpu_run), device="cpu")
    cuda_checks, cuda_evidence = _smoke_checks(_summary(args.cuda_run), device="cuda")
    semantic_checks, semantic_evidence = _semantic_checks(args.cuda_run)
    checks = {**semantic_checks, **cpu_checks, **cuda_checks}
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "status": "engineering_gate_passed" if all(checks.values()) else "engineering_gate_failed",
        "passed": all(checks.values()),
        "checks": checks,
        "evidence": {
            "semantics": semantic_evidence,
            "cpu_smoke": cpu_evidence,
            "cuda_smoke": cuda_evidence,
        },
        "runs": {"cpu": args.cpu_run, "cuda": args.cuda_run},
    }
    output = ROOT / "outputs/runs" / EXPERIMENT_ID / "_suite/metrics/engineering_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
