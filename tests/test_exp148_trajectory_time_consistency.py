from __future__ import annotations

from dataclasses import asdict
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment  # noqa: E402


BASE_CONFIG = ROOT / "configs/experiment/exp125_decentralized_tiered_b0_pure_rl_relative_quintic.yaml"
EXP148_CONFIG = ROOT / "configs/experiment/exp148_decentralized_b0_trajectory_time_consistent.yaml"


def test_exp148_changes_only_trajectory_time_contract() -> None:
    base = cfg_from_experiment(BASE_CONFIG)
    candidate = cfg_from_experiment(EXP148_CONFIG)

    assert base.trajectory_generator.time_parameterization == "planning_step"
    assert base.low_level_control.tracking_point_mode == "fixed_index"
    assert candidate.trajectory_generator.time_parameterization == "arc_length_reference_speed"
    assert candidate.low_level_control.tracking_point_mode == "planning_time"

    candidate_trajectory = asdict(candidate.trajectory_generator)
    base_trajectory = asdict(base.trajectory_generator)
    candidate_trajectory.pop("time_parameterization")
    base_trajectory.pop("time_parameterization")
    assert candidate_trajectory == base_trajectory

    candidate_control = asdict(candidate.low_level_control)
    base_control = asdict(base.low_level_control)
    candidate_control.pop("tracking_point_mode")
    base_control.pop("tracking_point_mode")
    assert candidate_control == base_control

    for section in (
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
    ):
        assert asdict(getattr(candidate, section)) == asdict(getattr(base, section))


def test_exp148_keeps_strict_decentralized_execution_and_pure_rl() -> None:
    cfg = cfg_from_experiment(EXP148_CONFIG)
    assert cfg.actor_obs_dim == 101
    assert cfg.observation.schema_version == "ego_v8_decentralized_tiered"
    assert not cfg.planner.subgoal_filter.enabled
    assert not cfg.low_level_control.safety_projection_enabled
    assert not cfg.low_level_control.success_zone_damping_enabled
    assert not cfg.low_level_control.formation_center_correction_enabled
    assert not cfg.low_level_control.terminal_slot_capture_enabled
    assert not cfg.low_level_control.flat_geometry_capture_enabled
    assert not cfg.task.explicit_goal_in_execution
