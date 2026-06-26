from __future__ import annotations

import math

import pytest
import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import decode_action
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (
    MultiRoverGatheringEnvCfg,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.subgoal_filter import apply_subgoal_filter


def _cfg(*, enabled: bool = True) -> MultiRoverGatheringEnvCfg:
    cfg = MultiRoverGatheringEnvCfg()
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 1
    cfg.task.n_agents = 1
    cfg.planner.rho_max = 1.2
    cfg.planner.beta_max = math.pi / 4.0
    cfg.planner.subgoal_filter.enabled = enabled
    cfg.observation.communication_radius = 12.0
    cfg.safety.collision_distance = 0.28
    cfg.success_thresholds.min_pairwise_distance = 0.42
    return cfg


def _curriculum_cfg() -> MultiRoverGatheringEnvCfg:
    cfg = _cfg(enabled=True)
    cfg.planner.subgoal_filter.mode = "terrain_safe_candidate_curriculum"
    cfg.planner.subgoal_filter.rho_scales = [1.0]
    cfg.planner.subgoal_filter.beta_offsets_deg = [-30.0, 0.0, 30.0]
    cfg.planner.subgoal_filter.warmup_timesteps = 2048
    cfg.planner.subgoal_filter.ramp_timesteps = 4096
    cfg.planner.subgoal_filter.apply_probability_end = 0.60
    cfg.planner.subgoal_filter.score_scale_start = 0.15
    cfg.planner.subgoal_filter.score_scale_end = 0.75
    cfg.planner.subgoal_filter.deterministic_improvement_margin = 0.02
    cfg.terrain.type = "lunar_crater_proxy"
    cfg.terrain.crater_count = 1
    cfg.terrain.crater_min_radius = 1.0
    cfg.terrain.crater_max_radius = 1.0
    cfg.terrain.crater_depth_to_diameter = 0.24
    cfg.terrain.crater_rim_height_to_diameter = 0.0
    cfg.terrain.traversability_slope_scale = 0.20
    return cfg


def _decode(cfg: MultiRoverGatheringEnvCfg, positions: torch.Tensor, action: torch.Tensor):
    yaws = torch.zeros(positions.shape[:2])
    return decode_action(action, positions, yaws, cfg.planner), yaws


def test_disabled_filter_leaves_decoded_action_unchanged() -> None:
    cfg = _cfg(enabled=False)
    positions = torch.tensor([[[0.0, 0.0, 0.0]]])
    action = torch.tensor([[[0.2, -0.3]]])
    decoded, yaws = _decode(cfg, positions, action)

    result = apply_subgoal_filter(decoded, positions, yaws, cfg)

    assert not result.info["enabled"]
    assert torch.allclose(result.decoded.physical, decoded.physical)
    assert torch.allclose(result.decoded.world_subgoal, decoded.world_subgoal)
    assert torch.allclose(result.decoded.clipped_normalized, decoded.clipped_normalized)


def test_flat_filter_chooses_raw_intent_candidate() -> None:
    cfg = _cfg(enabled=True)
    positions = torch.tensor([[[0.0, 0.0, 0.0]]])
    action = torch.tensor([[[0.1, 0.2]]])
    decoded, yaws = _decode(cfg, positions, action)

    result = apply_subgoal_filter(decoded, positions, yaws, cfg)

    assert result.info["candidate_count"] == 10
    assert int(result.info["candidate_index"][0, 0]) == int(result.info["raw_candidate_index"])
    assert torch.allclose(result.decoded.physical, decoded.physical)


def test_crater_crossing_selects_low_risk_side_candidate() -> None:
    cfg = _cfg(enabled=True)
    cfg.planner.subgoal_filter.rho_scales = [1.0]
    cfg.planner.subgoal_filter.beta_offsets_deg = [-45.0, 0.0, 45.0]
    cfg.terrain.type = "lunar_crater_proxy"
    cfg.terrain.crater_count = 1
    cfg.terrain.crater_min_radius = 1.0
    cfg.terrain.crater_max_radius = 1.0
    cfg.terrain.crater_depth_to_diameter = 0.24
    cfg.terrain.crater_rim_height_to_diameter = 0.0
    cfg.terrain.traversability_slope_scale = 0.20
    positions = torch.tensor([[[-1.2, 0.4, 0.0]]])
    action = torch.tensor([[[1.0, 0.0]]])
    decoded, yaws = _decode(cfg, positions, action)

    result = apply_subgoal_filter(decoded, positions, yaws, cfg)

    assert int(result.info["candidate_index"][0, 0]) != int(result.info["raw_candidate_index"])
    assert abs(float(result.decoded.physical[0, 0, 1])) > 0.1
    assert result.info["filtered_path_terrain_risk_mean"][0, 0] < result.info["raw_path_terrain_risk_mean"][0, 0]


def test_endpoint_conflict_filter_increases_predicted_neighbor_distance() -> None:
    cfg = _cfg(enabled=True)
    cfg.task.n_agents = 2
    cfg.terrain.type = "flat_proxy"
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]]])
    action = torch.tensor([[[0.0, 0.0], [-1.0, 0.0]]])
    decoded, yaws = _decode(cfg, positions, action)
    raw_distance = torch.linalg.norm(decoded.world_subgoal[0, 0, :2] - positions[0, 1, :2])

    result = apply_subgoal_filter(decoded, positions, yaws, cfg)
    filtered_distance = torch.linalg.norm(
        result.decoded.world_subgoal[0, 0, :2] - positions[0, 1, :2]
    )

    assert filtered_distance > raw_distance
    assert result.info["endpoint_collision_violation"][0, 0] == pytest.approx(0.0)


def test_invisible_rover_does_not_affect_filter_result() -> None:
    cfg = _cfg(enabled=True)
    cfg.task.n_agents = 3
    cfg.observation.communication_radius = 2.0
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [8.0, 0.0, 0.0]]])
    changed = positions.clone()
    changed[0, 2, :2] = torch.tensor([8.0, 8.0])
    action = torch.tensor([[[0.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]])
    decoded, yaws = _decode(cfg, positions, action)
    changed_decoded, changed_yaws = _decode(cfg, changed, action)

    result = apply_subgoal_filter(decoded, positions, yaws, cfg)
    changed_result = apply_subgoal_filter(changed_decoded, changed, changed_yaws, cfg)

    assert torch.allclose(
        result.decoded.physical[0, 0],
        changed_result.decoded.physical[0, 0],
    )


def test_filtered_action_bounds_and_info_are_finite() -> None:
    cfg = _cfg(enabled=True)
    cfg.task.n_agents = 2
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.7, 0.1, 0.0]]])
    action = torch.tensor([[[0.8, 0.9], [0.4, -0.6]]])
    decoded, yaws = _decode(cfg, positions, action)

    result = apply_subgoal_filter(decoded, positions, yaws, cfg)

    assert result.decoded.physical.shape == (1, 2, 2)
    assert torch.all(result.decoded.physical[..., 0] >= 0.0)
    assert torch.all(result.decoded.physical[..., 0] <= cfg.planner.rho_max)
    assert torch.all(result.decoded.physical[..., 1].abs() <= cfg.planner.beta_max)
    for value in result.info.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all()


def test_curriculum_warmup_reports_scores_without_replacing_action() -> None:
    cfg = _curriculum_cfg()
    positions = torch.tensor([[[-1.2, 0.4, 0.0]]])
    action = torch.tensor([[[1.0, 0.0]]])
    decoded, yaws = _decode(cfg, positions, action)

    result = apply_subgoal_filter(
        decoded,
        positions,
        yaws,
        cfg,
        progress_timestep=0,
        deterministic=False,
    )

    assert not result.info["applied"].any()
    assert result.info["apply_probability"] == pytest.approx(0.0)
    assert result.info["score_scale"] == pytest.approx(0.15)
    assert torch.allclose(result.decoded.physical, decoded.physical)
    assert torch.isfinite(result.info["raw_score"]).all()
    assert torch.isfinite(result.info["suggested_score"]).all()

    deterministic_result = apply_subgoal_filter(
        decoded,
        positions,
        yaws,
        cfg,
        progress_timestep=0,
        deterministic=True,
    )
    assert not deterministic_result.info["applied"].any()
    assert torch.allclose(deterministic_result.decoded.physical, decoded.physical)


def test_curriculum_ramp_can_apply_low_risk_candidate_deterministically() -> None:
    cfg = _curriculum_cfg()
    positions = torch.tensor([[[-1.2, 0.4, 0.0]]])
    action = torch.tensor([[[1.0, 0.0]]])
    decoded, yaws = _decode(cfg, positions, action)

    result = apply_subgoal_filter(
        decoded,
        positions,
        yaws,
        cfg,
        progress_timestep=6144,
        deterministic=True,
    )

    assert result.info["applied"][0, 0]
    assert result.info["deterministic_applied"][0, 0]
    assert result.info["apply_probability"] == pytest.approx(0.60)
    assert result.info["score_scale"] == pytest.approx(0.75)
    assert result.info["filtered_path_terrain_risk_mean"][0, 0] < result.info[
        "raw_path_terrain_risk_mean"
    ][0, 0]


def test_curriculum_deterministic_rule_is_reproducible() -> None:
    cfg = _curriculum_cfg()
    positions = torch.tensor([[[-1.2, 0.2, 0.0]]])
    action = torch.tensor([[[1.0, 0.0]]])
    decoded, yaws = _decode(cfg, positions, action)

    first = apply_subgoal_filter(decoded, positions, yaws, cfg, progress_timestep=6144, deterministic=True)
    second = apply_subgoal_filter(decoded, positions, yaws, cfg, progress_timestep=6144, deterministic=True)

    assert torch.equal(first.info["candidate_index"], second.info["candidate_index"])
    assert torch.allclose(first.decoded.physical, second.decoded.physical)


def test_visible_neighbor_center_intent_ignores_invisible_rover() -> None:
    cfg = _cfg(enabled=True)
    cfg.task.n_agents = 3
    cfg.observation.communication_radius = 2.0
    cfg.planner.subgoal_filter.mode = "terrain_safe_candidate_curriculum"
    cfg.planner.subgoal_filter.visible_neighbor_center_weight = 10.0
    cfg.planner.subgoal_filter.warmup_timesteps = 0
    cfg.planner.subgoal_filter.ramp_timesteps = 1
    cfg.planner.subgoal_filter.apply_probability_end = 1.0
    cfg.planner.subgoal_filter.score_scale_start = 1.0
    cfg.planner.subgoal_filter.score_scale_end = 1.0
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [8.0, 0.0, 0.0]]])
    changed = positions.clone()
    changed[0, 2, :2] = torch.tensor([8.0, 8.0])
    action = torch.tensor([[[0.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]])
    decoded, yaws = _decode(cfg, positions, action)
    changed_decoded, changed_yaws = _decode(cfg, changed, action)

    result = apply_subgoal_filter(
        decoded,
        positions,
        yaws,
        cfg,
        progress_timestep=1,
        deterministic=True,
    )
    changed_result = apply_subgoal_filter(
        changed_decoded,
        changed,
        changed_yaws,
        cfg,
        progress_timestep=1,
        deterministic=True,
    )

    assert torch.allclose(
        result.decoded.physical[0, 0],
        changed_result.decoded.physical[0, 0],
    )


def _constrained_curriculum_cfg() -> MultiRoverGatheringEnvCfg:
    cfg = _cfg(enabled=True)
    cfg.task.n_agents = 2
    cfg.terrain.type = "flat_proxy"
    cfg.planner.subgoal_filter.mode = "terrain_safe_candidate_constrained_curriculum"
    cfg.planner.subgoal_filter.rho_scales = [1.0]
    cfg.planner.subgoal_filter.beta_offsets_deg = [-45.0, 0.0, 45.0]
    cfg.planner.subgoal_filter.intent_deviation_weight = 0.1
    cfg.planner.subgoal_filter.endpoint_near_weight = 10.0
    cfg.planner.subgoal_filter.endpoint_collision_weight = 2000.0
    cfg.planner.subgoal_filter.path_collision_weight = 2000.0
    cfg.planner.subgoal_filter.visible_neighbor_center_weight = 0.2
    cfg.planner.subgoal_filter.endpoint_safe_distance = 0.50
    cfg.planner.subgoal_filter.path_safe_distance = 0.42
    cfg.planner.subgoal_filter.hard_endpoint_near_filter = True
    cfg.planner.subgoal_filter.hard_path_collision_filter = True
    cfg.planner.subgoal_filter.hard_center_progress_filter = True
    cfg.planner.subgoal_filter.center_progress_slack = 0.45
    cfg.planner.subgoal_filter.safety_override_after_warmup = True
    cfg.planner.subgoal_filter.warmup_timesteps = 8
    cfg.planner.subgoal_filter.ramp_timesteps = 8
    cfg.planner.subgoal_filter.apply_probability_end = 0.0
    cfg.planner.subgoal_filter.score_scale_start = 1.0
    cfg.planner.subgoal_filter.score_scale_end = 1.0
    return cfg


def test_constrained_curriculum_warmup_does_not_override_unsafe_raw_action() -> None:
    cfg = _constrained_curriculum_cfg()
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]])
    action = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    decoded, yaws = _decode(cfg, positions, action)

    result = apply_subgoal_filter(
        decoded,
        positions,
        yaws,
        cfg,
        progress_timestep=0,
        deterministic=False,
    )

    assert not result.info["applied"][0, 0]
    assert not result.info["safety_override"][0, 0]
    assert torch.allclose(result.decoded.physical[0, 0], decoded.physical[0, 0])
    assert result.info["raw_endpoint_near_violation"][0, 0] > 0.0
    assert result.info["feasible_fraction"] > 0.0


def test_constrained_curriculum_safety_override_after_warmup_selects_safe_candidate() -> None:
    cfg = _constrained_curriculum_cfg()
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]])
    action = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    decoded, yaws = _decode(cfg, positions, action)

    result = apply_subgoal_filter(
        decoded,
        positions,
        yaws,
        cfg,
        progress_timestep=8,
        deterministic=False,
    )

    assert result.info["safety_override"][0, 0]
    assert result.info["applied"][0, 0]
    assert int(result.info["candidate_index"][0, 0]) != int(result.info["raw_candidate_index"])
    assert result.info["endpoint_near_violation"][0, 0] == pytest.approx(0.0)
    assert result.info["path_collision_violation"][0, 0] == pytest.approx(0.0)
    assert result.info["raw_endpoint_collision_violation"][0, 0] > 0.0
    assert result.info["candidate_feasible"][0, 0]
    assert result.info["safety_override_fraction"] > 0.0


def test_constrained_curriculum_ignores_invisible_rover_for_safety_constraints() -> None:
    cfg = _constrained_curriculum_cfg()
    cfg.task.n_agents = 3
    cfg.observation.communication_radius = 1.2
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [8.0, 0.0, 0.0]]])
    changed = positions.clone()
    changed[0, 2, :2] = torch.tensor([8.0, 8.0])
    action = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [-1.0, 0.0]]])
    decoded, yaws = _decode(cfg, positions, action)
    changed_decoded, changed_yaws = _decode(cfg, changed, action)

    result = apply_subgoal_filter(
        decoded,
        positions,
        yaws,
        cfg,
        progress_timestep=8,
        deterministic=False,
    )
    changed_result = apply_subgoal_filter(
        changed_decoded,
        changed,
        changed_yaws,
        cfg,
        progress_timestep=8,
        deterministic=False,
    )

    assert torch.allclose(result.decoded.physical[0, 0], changed_result.decoded.physical[0, 0])
    assert torch.equal(result.info["candidate_index"][0, 0], changed_result.info["candidate_index"][0, 0])
