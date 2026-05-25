from __future__ import annotations

import math

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import (
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

