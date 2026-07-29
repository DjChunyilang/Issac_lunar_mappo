from __future__ import annotations

import math

import pytest
import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (
    LowLevelControlCfg,
    SuccessThresholdsCfg,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.simple_controller import (
    ControlCommand,
    apply_control_safety_projection,
)


def _control(num_envs: int = 1, n_agents: int = 2) -> ControlCommand:
    return ControlCommand(
        linear=torch.ones(num_envs, n_agents),
        angular=torch.zeros(num_envs, n_agents),
    )


def test_control_safety_disabled_leaves_command_unchanged() -> None:
    cfg = LowLevelControlCfg()
    thresholds = SuccessThresholdsCfg()
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.55, 0.0, 0.0]]])
    yaws = torch.tensor([[0.0, math.pi]])
    control = _control()
    metrics = compute_team_metrics(positions, torch.zeros(1, 2, 2))

    result = apply_control_safety_projection(
        control,
        positions,
        yaws,
        metrics,
        cfg,
        thresholds,
        planning_dt=0.2,
        communication_radius=12.0,
    )

    assert not result.info["enabled"]
    assert torch.allclose(result.control.linear, control.linear)
    assert torch.allclose(result.control.angular, control.angular)
    assert not result.info["applied"].any()


def test_control_safety_projection_damps_closing_rovers() -> None:
    cfg = LowLevelControlCfg(
        safety_projection_enabled=True,
        projection_activation_distance=0.62,
        projection_stop_distance=0.36,
        projection_horizon_s=0.40,
        projection_strength=1.0,
        projection_min_linear_scale=0.20,
    )
    thresholds = SuccessThresholdsCfg(min_pairwise_distance=0.42)
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.55, 0.0, 0.0]]])
    yaws = torch.tensor([[0.0, math.pi]])
    control = _control()
    metrics = compute_team_metrics(positions, torch.zeros(1, 2, 2))

    result = apply_control_safety_projection(
        control,
        positions,
        yaws,
        metrics,
        cfg,
        thresholds,
        planning_dt=0.2,
        communication_radius=12.0,
    )

    assert result.info["enabled"]
    assert result.info["applied"].all()
    assert torch.all(result.control.linear < control.linear)
    assert torch.all(result.info["linear_scale"] >= cfg.projection_min_linear_scale)
    assert torch.all(result.info["pairwise_risk"] > 0.0)
    assert torch.isfinite(result.info["predicted_nearest_distance"]).all()


def test_control_safety_projection_ignores_far_rovers() -> None:
    cfg = LowLevelControlCfg(
        safety_projection_enabled=True,
        projection_activation_distance=0.62,
        projection_stop_distance=0.36,
        projection_horizon_s=0.40,
        projection_strength=1.0,
        projection_min_linear_scale=0.20,
    )
    positions = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    yaws = torch.tensor([[0.0, math.pi]])
    control = _control()
    metrics = compute_team_metrics(positions, torch.zeros(1, 2, 2))

    result = apply_control_safety_projection(
        control,
        positions,
        yaws,
        metrics,
        cfg,
        SuccessThresholdsCfg(),
        planning_dt=0.2,
        communication_radius=12.0,
    )

    assert torch.allclose(result.control.linear, control.linear)
    assert not result.info["applied"].any()
    assert torch.allclose(result.info["pairwise_risk"], torch.zeros_like(control.linear))


def test_closing_only_projection_ignores_nonclosing_near_rovers() -> None:
    cfg = LowLevelControlCfg(
        safety_projection_enabled=True,
        projection_activation_distance=0.62,
        projection_stop_distance=0.36,
        projection_horizon_s=0.40,
        projection_strength=1.0,
        projection_min_linear_scale=0.20,
        projection_damp_nonclosing_near=False,
    )
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.50, 0.0, 0.0]]])
    yaws = torch.tensor([[math.pi, 0.0]])
    control = _control()
    metrics = compute_team_metrics(positions, torch.zeros(1, 2, 2))

    result = apply_control_safety_projection(
        control,
        positions,
        yaws,
        metrics,
        cfg,
        SuccessThresholdsCfg(min_pairwise_distance=0.42),
        planning_dt=0.2,
        communication_radius=12.0,
    )

    assert torch.allclose(result.control.linear, control.linear)
    assert not result.info["applied"].any()
    assert torch.allclose(result.info["pairwise_risk"], torch.zeros_like(control.linear))


def test_directional_projection_damps_only_inward_moving_rover() -> None:
    cfg = LowLevelControlCfg(
        safety_projection_enabled=True,
        projection_activation_distance=0.62,
        projection_stop_distance=0.36,
        projection_horizon_s=0.40,
        projection_strength=1.0,
        projection_min_linear_scale=0.20,
        projection_damp_nonclosing_near=False,
        projection_directional_agent_scale=True,
    )
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.50, 0.0, 0.0]]])
    yaws = torch.tensor([[0.0, 0.0]])
    control = ControlCommand(
        linear=torch.tensor([[0.2, 0.0]]),
        angular=torch.zeros(1, 2),
    )
    metrics = compute_team_metrics(positions, torch.zeros(1, 2, 2))

    result = apply_control_safety_projection(
        control,
        positions,
        yaws,
        metrics,
        cfg,
        SuccessThresholdsCfg(min_pairwise_distance=0.42),
        planning_dt=0.2,
        communication_radius=12.0,
    )

    assert result.info["applied"][0, 0]
    assert not result.info["applied"][0, 1]
    assert result.control.linear[0, 0] < control.linear[0, 0]
    assert result.control.linear[0, 1] == pytest.approx(control.linear[0, 1])
    assert result.info["pairwise_risk"][0, 0] > 0.0
    assert result.info["pairwise_risk"][0, 1] == pytest.approx(0.0)


def test_directional_mask_projection_keeps_asymmetry_but_uses_full_pair_risk() -> None:
    base_cfg = dict(
        safety_projection_enabled=True,
        projection_activation_distance=0.62,
        projection_stop_distance=0.36,
        projection_horizon_s=0.40,
        projection_strength=1.0,
        projection_min_linear_scale=0.20,
        projection_damp_nonclosing_near=False,
        projection_directional_agent_scale=True,
    )
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.50, 0.0, 0.0]]])
    yaws = torch.tensor([[0.0, 0.0]])
    control = ControlCommand(
        linear=torch.tensor([[0.2, 0.0]]),
        angular=torch.zeros(1, 2),
    )
    metrics = compute_team_metrics(positions, torch.zeros(1, 2, 2))

    fraction = apply_control_safety_projection(
        control,
        positions,
        yaws,
        metrics,
        LowLevelControlCfg(**base_cfg, projection_directional_agent_scale_mode="fraction"),
        SuccessThresholdsCfg(min_pairwise_distance=0.42),
        planning_dt=0.2,
        communication_radius=12.0,
    )
    masked = apply_control_safety_projection(
        control,
        positions,
        yaws,
        metrics,
        LowLevelControlCfg(**base_cfg, projection_directional_agent_scale_mode="mask"),
        SuccessThresholdsCfg(min_pairwise_distance=0.42),
        planning_dt=0.2,
        communication_radius=12.0,
    )

    assert masked.info["applied"][0, 0]
    assert not masked.info["applied"][0, 1]
    assert masked.control.linear[0, 0] < fraction.control.linear[0, 0]
    assert masked.control.linear[0, 1] == pytest.approx(control.linear[0, 1])
    assert masked.info["pairwise_risk"][0, 0] > fraction.info["pairwise_risk"][0, 0]
    assert masked.info["pairwise_risk"][0, 1] == pytest.approx(0.0)


def test_hard_directional_projection_stops_only_the_inward_rover_at_clearance() -> None:
    cfg = LowLevelControlCfg(
        safety_projection_enabled=True,
        projection_activation_distance=0.75,
        projection_stop_distance=0.42,
        projection_horizon_s=0.60,
        projection_strength=1.0,
        projection_min_linear_scale=0.0,
        projection_damp_nonclosing_near=True,
        projection_directional_agent_scale=True,
        projection_directional_agent_scale_mode="mask",
    )
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.40, 0.0, 0.0]]])
    yaws = torch.zeros(1, 2)
    control = ControlCommand(
        linear=torch.tensor([[1.0, 0.0]]),
        angular=torch.zeros(1, 2),
    )
    metrics = compute_team_metrics(positions, torch.zeros(1, 2, 2))

    result = apply_control_safety_projection(
        control,
        positions,
        yaws,
        metrics,
        cfg,
        SuccessThresholdsCfg(min_pairwise_distance=0.42),
        planning_dt=0.2,
        communication_radius=12.0,
    )

    assert result.control.linear[0, 0] == pytest.approx(0.0)
    assert result.control.linear[0, 1] == pytest.approx(0.0)
    assert result.info["applied"][0, 0]
    assert not result.info["applied"][0, 1]


def test_success_zone_damping_scales_compact_safe_team() -> None:
    cfg = LowLevelControlCfg(
        success_zone_damping_enabled=True,
        success_zone_linear_scale=0.65,
    )
    thresholds = SuccessThresholdsCfg(
        dmax=1.25,
        dispersion=0.30,
        min_pairwise_distance=0.42,
    )
    positions = torch.tensor(
        [[[-0.3, -0.3, 0.0], [0.3, -0.3, 0.0], [-0.3, 0.3, 0.0], [0.3, 0.3, 0.0]]]
    )
    yaws = torch.zeros(1, 4)
    control = _control(n_agents=4)
    metrics = compute_team_metrics(positions, torch.zeros(1, 4, 2))

    result = apply_control_safety_projection(
        control,
        positions,
        yaws,
        metrics,
        cfg,
        thresholds,
        planning_dt=0.2,
        communication_radius=12.0,
    )

    assert result.info["success_zone_active"][0]
    assert result.info["success_zone_fraction"] == pytest.approx(1.0)
    assert torch.allclose(
        result.control.linear,
        torch.full_like(control.linear, cfg.success_zone_linear_scale),
    )
