from __future__ import annotations

import math

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import (
    apply_formation_center_correction,
    apply_flat_geometry_capture,
    apply_terminal_slot_capture,
    decode_action,
    polar_to_local_subgoal,
    scale_action,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import PlannerCfg


def test_scale_action_bounds() -> None:
    cfg = PlannerCfg(rho_max=2.0, beta_max=math.pi / 2)
    physical = scale_action(torch.tensor([[[-2.0, 2.0], [1.0, -1.0]]]), cfg)
    assert torch.all(physical[..., 0] >= 0.0)
    assert torch.all(physical[..., 0] <= 2.0)
    assert torch.all(physical[..., 1] >= -math.pi / 2)
    assert torch.all(physical[..., 1] <= math.pi / 2)


def test_polar_to_local_subgoal() -> None:
    physical = torch.tensor([[[1.0, 0.0], [1.0, math.pi / 2]]])
    local = polar_to_local_subgoal(physical)
    assert torch.allclose(local[0, 0], torch.tensor([1.0, 0.0]), atol=1.0e-6)
    assert torch.allclose(local[0, 1], torch.tensor([0.0, 1.0]), atol=1.0e-6)


def test_local_to_world_subgoal_from_decode() -> None:
    cfg = PlannerCfg(rho_max=2.0, beta_max=math.pi / 2)
    positions = torch.zeros(1, 1, 3)
    yaws = torch.tensor([[math.pi / 2]])
    action = torch.tensor([[[1.0, 0.0]]])
    decoded = decode_action(action, positions, yaws, cfg)
    assert torch.allclose(decoded.world_subgoal[0, 0, :2], torch.tensor([0.0, 2.0]), atol=1.0e-5)


def test_formation_center_correction_preserves_slot_offsets() -> None:
    cfg = PlannerCfg(rho_max=1.6, beta_max=math.pi / 2)
    positions = torch.tensor(
        [[[0.5, 0.2, 0.0], [0.7, 0.2, 0.0], [0.5, 0.4, 0.0], [0.7, 0.4, 0.0]]]
    )
    decoded = decode_action(
        torch.zeros(1, 4, 2),
        positions,
        torch.zeros(1, 4),
        cfg,
    )

    result = apply_formation_center_correction(
        decoded,
        centroid_xy=torch.tensor([[0.6, 0.3]]),
        dmax=torch.tensor([1.0]),
        dispersion=torch.tensor([0.30]),
        formation_center_xy=torch.tensor([[0.0, 0.0]]),
        dmax_threshold=1.25,
        dispersion_threshold=0.30,
        enabled=True,
        activation_dmax_multiplier=1.75,
        activation_dispersion_multiplier=1.75,
        max_offset=0.35,
        gain=0.5,
    )

    expected = torch.tensor([[-0.15652476, -0.07826238]])
    assert result.active.tolist() == [True]
    assert torch.allclose(result.offset_xy, expected, atol=1.0e-6)
    assert torch.allclose(
        result.decoded.world_subgoal[..., :2] - decoded.world_subgoal[..., :2],
        expected[:, None, :],
        atol=1.0e-6,
    )
    before_pairwise = torch.cdist(decoded.world_subgoal[..., :2], decoded.world_subgoal[..., :2])
    after_pairwise = torch.cdist(
        result.decoded.world_subgoal[..., :2],
        result.decoded.world_subgoal[..., :2],
    )
    assert torch.allclose(after_pairwise, before_pairwise, atol=1.0e-6)


def test_formation_center_correction_is_inactive_outside_terminal_zone() -> None:
    decoded = decode_action(
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2),
        PlannerCfg(),
    )
    result = apply_formation_center_correction(
        decoded,
        centroid_xy=torch.tensor([[0.5, 0.0]]),
        dmax=torch.tensor([2.3]),
        dispersion=torch.tensor([0.10]),
        formation_center_xy=torch.zeros(1, 2),
        dmax_threshold=1.25,
        dispersion_threshold=0.30,
        enabled=True,
        activation_dmax_multiplier=1.75,
        activation_dispersion_multiplier=1.75,
        max_offset=0.35,
        gain=0.55,
    )

    assert result.active.tolist() == [False]
    assert torch.allclose(result.offset_xy, torch.zeros_like(result.offset_xy))
    assert torch.allclose(result.decoded.world_subgoal, decoded.world_subgoal)


def test_formation_center_correction_can_target_only_flatness_failures() -> None:
    decoded = decode_action(
        torch.zeros(2, 2, 2),
        torch.zeros(2, 2, 3),
        torch.zeros(2, 2),
        PlannerCfg(),
    )
    result = apply_formation_center_correction(
        decoded,
        centroid_xy=torch.tensor([[0.4, 0.0], [0.4, 0.0]]),
        dmax=torch.tensor([1.0, 1.0]),
        dispersion=torch.tensor([0.2, 0.2]),
        formation_center_xy=torch.zeros(2, 2),
        dmax_threshold=1.25,
        dispersion_threshold=0.30,
        enabled=True,
        activation_dmax_multiplier=1.25,
        activation_dispersion_multiplier=1.25,
        max_offset=0.35,
        gain=0.5,
        flatness_ok=torch.tensor([True, False]),
        require_flatness_failure=True,
    )

    assert result.active.tolist() == [False, True]
    assert torch.allclose(result.offset_xy[0], torch.zeros(2))
    assert torch.allclose(result.offset_xy[1], torch.tensor([-0.175, 0.0]))


def test_terminal_slot_capture_blends_only_near_terminal_geometry() -> None:
    decoded = decode_action(
        torch.zeros(2, 2, 2),
        torch.zeros(2, 2, 3),
        torch.zeros(2, 2),
        PlannerCfg(),
    )
    slots = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    )
    result = apply_terminal_slot_capture(
        decoded,
        gather_slot_points=slots,
        dmax=torch.tensor([1.0, 2.3]),
        dispersion=torch.tensor([0.2, 0.2]),
        dmax_threshold=1.25,
        dispersion_threshold=0.30,
        enabled=True,
        activation_dmax_multiplier=1.75,
        activation_dispersion_multiplier=1.75,
        blend=0.65,
    )

    assert result.active.tolist() == [True, False]
    assert torch.allclose(
        result.decoded.world_subgoal[0],
        torch.lerp(decoded.world_subgoal[0], slots[0], 0.65),
    )
    assert torch.allclose(result.decoded.world_subgoal[1], decoded.world_subgoal[1])


def test_flat_geometry_capture_contracts_around_flat_actual_centroid_only() -> None:
    decoded = decode_action(
        torch.zeros(3, 2, 2),
        torch.zeros(3, 2, 3),
        torch.zeros(3, 2),
        PlannerCfg(),
    )
    slots = torch.tensor(
        [
            [[1.2, 1.0, 0.0], [0.8, 1.0, 0.0]],
            [[1.2, 1.0, 0.0], [0.8, 1.0, 0.0]],
            [[1.2, 1.0, 0.0], [0.8, 1.0, 0.0]],
        ]
    )
    result = apply_flat_geometry_capture(
        decoded,
        gather_slot_points=slots,
        centroid_xy=torch.tensor([[0.4, -0.2], [0.4, -0.2], [0.4, -0.2]]),
        dmax=torch.tensor([1.5, 1.5, 1.0]),
        dispersion=torch.tensor([0.2, 0.2, 0.2]),
        flatness_ok=torch.tensor([True, False, True]),
        dmax_threshold=1.25,
        dispersion_threshold=0.30,
        enabled=True,
        activation_dmax_multiplier=1.75,
        activation_dispersion_multiplier=1.75,
        blend=1.0,
    )

    assert result.active.tolist() == [True, False, False]
    assert torch.allclose(
        result.decoded.world_subgoal[0, :, :2],
        torch.tensor([[0.6, -0.2], [0.2, -0.2]]),
    )
    assert torch.allclose(result.decoded.world_subgoal[1:], decoded.world_subgoal[1:])
