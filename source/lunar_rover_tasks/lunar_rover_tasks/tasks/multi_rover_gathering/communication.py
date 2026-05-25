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

    pairwise_delta = positions[:, None, :, :2] - positions[:, :, None, :2]
    pairwise_vel = velocities_xy[:, None, :, :] - velocities_xy[:, :, None, :]
    yaw_delta = wrap_to_pi(yaws[:, None, :] - yaws[:, :, None])
    pairwise_dist = torch.linalg.norm(pairwise_delta, dim=-1)

    for env_id in range(num_envs):
        for agent_id in range(n_agents):
            candidates = torch.nonzero(visibility[env_id, agent_id], as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            order = torch.argsort(pairwise_dist[env_id, agent_id, candidates])
            selected = candidates[order[: cfg.max_neighbors]]
            slot_count = selected.numel()
            features[env_id, agent_id, :slot_count, 0:2] = pairwise_delta[
                env_id, agent_id, selected
            ]
            features[env_id, agent_id, :slot_count, 2:4] = pairwise_vel[env_id, agent_id, selected]
            features[env_id, agent_id, :slot_count, 4] = torch.cos(
                yaw_delta[env_id, agent_id, selected]
            )
            features[env_id, agent_id, :slot_count, 5] = torch.sin(
                yaw_delta[env_id, agent_id, selected]
            )
            features[env_id, agent_id, :slot_count, 6] = 1.0
            masks[env_id, agent_id, :slot_count] = 1.0
    return features.flatten(start_dim=2), masks

