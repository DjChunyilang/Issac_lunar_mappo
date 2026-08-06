#!/usr/bin/env python
"""Write machine-readable engineering gates for exp148 trajectory timing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.mapf_diagnostics import (
    trajectory_pairwise_min_distance,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.simple_controller import (
    interpolate_trajectory_point,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    sample_trajectory_terrain_risk,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import (
    generate_trajectory,
)


EXPERIMENT_ID = "exp148_trajectory_time_consistency_fix"
CONFIG = ROOT / "configs/experiment/exp148_decentralized_b0_trajectory_time_consistent.yaml"
BASE_CONFIG = ROOT / "configs/experiment/exp125_decentralized_tiered_b0_pure_rl_relative_quintic.yaml"


def _audit(run_name: str) -> dict:
    path = (
        ROOT
        / "outputs/runs"
        / EXPERIMENT_ID
        / run_name
        / "metrics/trajectory_execution_contract.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _all_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _config_checks() -> tuple[dict[str, bool], dict]:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    base = cfg_from_experiment(BASE_CONFIG)
    unchanged_sections = {
        name: asdict(getattr(cfg, name)) == asdict(getattr(base, name))
        for name in (
            "task",
            "initial_state",
            "planner",
            "terrain",
            "reward_weights",
            "reward_coefficients",
            "observation",
            "state",
            "safety",
            "gather_point",
            "success_thresholds",
        )
    }
    trajectory = asdict(cfg.trajectory_generator)
    base_trajectory = asdict(base.trajectory_generator)
    trajectory.pop("time_parameterization")
    base_trajectory.pop("time_parameterization")
    control = asdict(cfg.low_level_control)
    base_control = asdict(base.low_level_control)
    control.pop("tracking_point_mode")
    base_control.pop("tracking_point_mode")
    algorithm = raw.get("algorithm") or {}
    checks = {
        "actor_obs_dim_101": cfg.actor_obs_dim == 101,
        "strict_decentralized_schema": cfg.observation.schema_version
        == "ego_v8_decentralized_tiered",
        "arc_length_reference_speed_timing": cfg.trajectory_generator.time_parameterization
        == "arc_length_reference_speed",
        "planning_time_tracking": cfg.low_level_control.tracking_point_mode
        == "planning_time",
        "legacy_base_contract_preserved": base.trajectory_generator.time_parameterization
        == "planning_step"
        and base.low_level_control.tracking_point_mode == "fixed_index",
        "trajectory_geometry_config_unchanged": trajectory == base_trajectory,
        "control_config_unchanged_except_tracking_mode": control == base_control,
        "all_other_environment_sections_unchanged": all(unchanged_sections.values()),
        "pure_rl_random_initialization": int(algorithm.get("bc_updates", -1)) == 0
        and algorithm.get("init_checkpoint") is None,
        "subgoal_filter_disabled": not cfg.planner.subgoal_filter.enabled,
        "safety_projection_disabled": not cfg.low_level_control.safety_projection_enabled,
        "directional_projection_disabled": not cfg.low_level_control.projection_directional_agent_scale,
        "terminal_execution_overrides_disabled": not any(
            (
                cfg.low_level_control.success_zone_damping_enabled,
                cfg.low_level_control.formation_center_correction_enabled,
                cfg.low_level_control.terminal_slot_capture_enabled,
                cfg.low_level_control.flat_geometry_capture_enabled,
                cfg.task.dynamic_terminal_slot_goal_enabled,
                cfg.task.explicit_goal_in_execution,
            )
        ),
    }
    return checks, {"unchanged_sections": unchanged_sections}


def _numerical_contract_checks() -> tuple[dict[str, bool], dict]:
    base = cfg_from_experiment(BASE_CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    positions = torch.zeros(1, 2, 3)
    subgoals = torch.tensor([[[1.0, 0.4, 0.0], [0.7, -0.5, 0.0]]])
    yaws = torch.zeros(1, 2)
    legacy = generate_trajectory(
        positions,
        subgoals,
        base.trajectory_generator,
        base.simulation.planning_dt,
        current_yaws=yaws,
    )
    physical = generate_trajectory(
        positions,
        subgoals,
        cfg.trajectory_generator,
        cfg.simulation.planning_dt,
        current_yaws=yaws,
    )
    arc = torch.linalg.vector_norm(
        physical.points[..., 1:, :2] - physical.points[..., :-1, :2], dim=-1
    ).sum(dim=-1)
    expected_horizon = (arc / cfg.trajectory_generator.reference_speed).clamp_min(
        cfg.simulation.planning_dt
    )
    target = interpolate_trajectory_point(physical, cfg.simulation.planning_dt)
    legacy_risk = sample_trajectory_terrain_risk(legacy.points, cfg.terrain)
    physical_risk = sample_trajectory_terrain_risk(physical.points, cfg.terrain)

    crossing_points = torch.tensor(
        [[
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]]
    )
    unequal_timing = torch.tensor([[[0.0, 0.5, 1.0], [0.0, 1.0, 2.0]]])
    equal_timing = torch.tensor([[[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]])
    unequal_min = trajectory_pairwise_min_distance(crossing_points, unequal_timing)
    equal_min = trajectory_pairwise_min_distance(crossing_points, equal_timing)
    checks = {
        "trajectory_geometry_exactly_unchanged": torch.equal(legacy.points, physical.points),
        "trajectory_headings_exactly_unchanged": torch.equal(
            legacy.headings, physical.headings
        ),
        "physical_horizon_matches_arc_over_reference_speed": torch.allclose(
            physical.timestamps[..., -1], expected_horizon, atol=1.0e-7
        ),
        "timestamps_monotonic_and_finite": bool(
            torch.isfinite(physical.timestamps).all()
            and (physical.timestamps[..., 1:] >= physical.timestamps[..., :-1]).all()
        ),
        "planning_time_target_inside_path": bool(
            torch.isfinite(target).all()
            and (torch.linalg.vector_norm(target[..., :2] - positions[..., :2], dim=-1) > 0.0).all()
            and (
                torch.linalg.vector_norm(target[..., :2] - positions[..., :2], dim=-1)
                <= arc + 1.0e-6
            ).all()
        ),
        "terrain_risk_exactly_unchanged": all(
            torch.equal(legacy_risk[key], physical_risk[key]) for key in legacy_risk
        ),
        "time_alignment_rejects_false_conflict": float(unequal_min[0, 0, 1]) >= 0.49,
        "time_alignment_retains_true_conflict": float(equal_min[0, 0, 1]) == 0.0,
    }
    evidence = {
        "arc_length_m": arc.tolist(),
        "physical_horizon_s": physical.timestamps[..., -1].tolist(),
        "tracking_target": target.tolist(),
        "unequal_timing_min_distance_m": float(unequal_min[0, 0, 1]),
        "equal_timing_min_distance_m": float(equal_min[0, 0, 1]),
    }
    return checks, evidence


def _audit_checks(
    audit: dict,
    *,
    prefix: str,
    require_dual_seed: bool,
) -> tuple[dict[str, bool], dict]:
    validation = audit.get("validation") or {}
    ratios = [
        float(row["required_to_declared_horizon_ratio"]["median"])
        for row in validation.values()
    ]
    violations = [
        float(row["timestamp_speed_violation_rate"])
        for row in validation.values()
    ]
    contract = audit.get("fixed_contract") or {}
    invariance = audit.get("invariance") or {}
    checks = {
        f"{prefix}_finite": _all_finite(audit),
        f"{prefix}_expected_seed_count": len(validation) == (2 if require_dual_seed else 1),
        f"{prefix}_timestamp_speed_violation_le_0_05": bool(violations)
        and max(violations) <= 0.05,
        f"{prefix}_horizon_ratio_median_in_0_99_1_01": bool(ratios)
        and min(ratios) >= 0.99
        and max(ratios) <= 1.01,
        f"{prefix}_no_fixed_tracking_index": contract.get(
            "controller_tracking_point_index", "missing"
        )
        is None,
        f"{prefix}_physical_modes": contract.get("time_parameterization")
        == "arc_length_reference_speed"
        and contract.get("tracking_point_mode") == "planning_time",
        f"{prefix}_actor_parameters_unchanged": invariance.get("actor_digest_before")
        == invariance.get("actor_digest_after"),
        f"{prefix}_actor_outputs_unchanged": float(
            invariance.get("actor_probe_output_max_abs_change", -1.0)
        )
        == 0.0,
    }
    return checks, {
        "timestamp_speed_violation_rates": violations,
        "horizon_ratio_medians": ratios,
        "run": audit.get("run"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-run", default="cpu_smoke_8x16")
    parser.add_argument("--cuda-run", default="cuda_smoke_256x64")
    parser.add_argument("--formal-run", default="frozen_exp125_post_fix_dualseed")
    args = parser.parse_args()
    config_checks, config_evidence = _config_checks()
    numerical_checks, numerical_evidence = _numerical_contract_checks()
    cpu_checks, cpu_evidence = _audit_checks(
        _audit(args.cpu_run), prefix="cpu", require_dual_seed=False
    )
    cuda_checks, cuda_evidence = _audit_checks(
        _audit(args.cuda_run), prefix="cuda", require_dual_seed=False
    )
    formal_checks, formal_evidence = _audit_checks(
        _audit(args.formal_run), prefix="formal", require_dual_seed=True
    )
    checks = {
        **config_checks,
        **numerical_checks,
        **cpu_checks,
        **cuda_checks,
        **formal_checks,
    }
    passed = all(checks.values())
    generated_at = datetime.now(timezone.utc).isoformat()
    output = ROOT / "outputs/runs" / EXPERIMENT_ID / "_suite/metrics/engineering_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": generated_at,
        "experiment": EXPERIMENT_ID,
        "status": "engineering_gate_passed" if passed else "engineering_gate_failed",
        "passed": passed,
        "training_authorized": passed,
        "forty_million_authorized": False,
        "checks": checks,
        "evidence": {
            "config": config_evidence,
            "numerical_contract": numerical_evidence,
            "cpu_smoke": cpu_evidence,
            "cuda_smoke": cuda_evidence,
            "formal_frozen_audit": formal_evidence,
        },
        "runs": {
            "cpu": args.cpu_run,
            "cuda": args.cuda_run,
            "formal": args.formal_run,
        },
        "artifacts": {"engineering_gate": str(output.relative_to(ROOT))},
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "experiment": EXPERIMENT_ID,
        "run": "_suite",
        "producer": "scripts/check_exp148_engineering_gates.py",
        "command": " ".join(sys.argv),
        "status": payload["status"],
        "artifacts": payload["artifacts"],
    }
    (output.parents[1] / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
