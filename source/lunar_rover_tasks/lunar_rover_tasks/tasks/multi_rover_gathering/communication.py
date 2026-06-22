"""Neighbor state sharing for decentralized actor observations."""

from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import ObservationCfg
from lunar_rover_tasks.utils.math_utils import wrap_to_pi


def compute_visibility_mask(
    positions: torch.Tensor,
    communication_radius: float,
) -> torch.Tensor:
    delta = positions[:, :, None, :2] - positions[:, None, :, :2]
    dist = torch.linalg.norm(delta, dim=-1)
    n_agents = positions.shape[1]
    self_mask = torch.eye(n_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
    return (dist <= communication_radius) & ~self_mask


def build_neighbor_features(
    positions: torch.Tensor,
    velocities_xy: torch.Tensor,
    yaws: torch.Tensor,
    communication_radius: float,
    cfg: ObservationCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_envs, n_agents, _ = positions.shape
    visibility = compute_visibility_mask(positions, communication_radius)
    pairwise_delta = positions[:, None, :, :2] - positions[:, :, None, :2]
    pairwise_vel = velocities_xy[:, None, :, :] - velocities_xy[:, :, None, :]
    yaw_delta = wrap_to_pi(yaws[:, None, :] - yaws[:, :, None])
    pairwise_dist = torch.linalg.norm(pairwise_delta, dim=-1)
    slot_count = min(cfg.max_neighbors, n_agents)
    masked_dist = pairwise_dist.masked_fill(~visibility, float("inf"))
    selected_dist, selected = torch.topk(
        masked_dist,
        k=slot_count,
        dim=-1,
        largest=False,
        sorted=True,
    )
    valid = torch.isfinite(selected_dist)
    gather_xy = selected[..., None].expand(-1, -1, -1, 2)
    selected_delta = torch.gather(pairwise_delta, dim=2, index=gather_xy)
    selected_vel = torch.gather(pairwise_vel, dim=2, index=gather_xy)
    selected_yaw = torch.gather(yaw_delta, dim=2, index=selected)
    selected_features = torch.cat(
        (
            selected_delta,
            selected_vel,
            torch.cos(selected_yaw).unsqueeze(-1),
            torch.sin(selected_yaw).unsqueeze(-1),
            valid.to(dtype=positions.dtype).unsqueeze(-1),
        ),
        dim=-1,
    )
    selected_features = torch.where(
        valid.unsqueeze(-1),
        selected_features,
        torch.zeros_like(selected_features),
    )

    features = torch.zeros(
        num_envs,
        n_agents,
        cfg.max_neighbors,
        cfg.neighbor_dim,
        dtype=positions.dtype,
        device=positions.device,
    )
    masks = torch.zeros(
        num_envs,
        n_agents,
        cfg.max_neighbors,
        dtype=positions.dtype,
        device=positions.device,
    )
    features[..., :slot_count, :] = selected_features
    masks[..., :slot_count] = valid.to(dtype=positions.dtype)
    return features.flatten(start_dim=2), masks
