"""Map normalized actor actions to local and world subgoals."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import PlannerCfg
from lunar_rover_tasks.utils.math_utils import rotate_2d


@dataclass(slots=True)
class DecodedAction:
    clipped_normalized: torch.Tensor
    physical: torch.Tensor
    local_subgoal_xy: torch.Tensor
    world_subgoal: torch.Tensor


def clip_action(action: torch.Tensor) -> torch.Tensor:
    return torch.clamp(action, -1.0, 1.0)


def scale_action(action: torch.Tensor, cfg: PlannerCfg) -> torch.Tensor:
    clipped = clip_action(action)
    rho = 0.5 * (clipped[..., 0] + 1.0) * cfg.rho_max
    beta = clipped[..., 1] * cfg.beta_max
    return torch.stack((rho, beta), dim=-1)


def polar_to_local_subgoal(physical_action: torch.Tensor) -> torch.Tensor:
    rho = physical_action[..., 0]
    beta = physical_action[..., 1]
    return torch.stack((rho * torch.cos(beta), rho * torch.sin(beta)), dim=-1)


def local_to_world_subgoal(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    local_subgoal_xy: torch.Tensor,
) -> torch.Tensor:
    world_delta = rotate_2d(local_subgoal_xy, yaws)
    z = torch.zeros_like(world_delta[..., :1])
    return positions + torch.cat((world_delta, z), dim=-1)


def decode_action(
    action: torch.Tensor,
    positions: torch.Tensor,
    yaws: torch.Tensor,
    cfg: PlannerCfg,
) -> DecodedAction:
    clipped = clip_action(action)
    physical = scale_action(clipped, cfg)
    local_subgoal_xy = polar_to_local_subgoal(physical)
    world_subgoal = local_to_world_subgoal(positions, yaws, local_subgoal_xy)
    return DecodedAction(
        clipped_normalized=clipped,
        physical=physical,
        local_subgoal_xy=local_subgoal_xy,
        world_subgoal=world_subgoal,
    )

