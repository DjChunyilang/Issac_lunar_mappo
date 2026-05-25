"""Team metrics used by rewards, terminations, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lunar_rover_tasks.utils.geometry_utils import (
    masked_pairwise_max,
    mean_offdiag,
    pairwise_distances_xy,
)


@dataclass(slots=True)
class TeamMetrics:
    centroid: torch.Tensor
    dmax: torch.Tensor
    dispersion: torch.Tensor
    mean_pairwise_distance: torch.Tensor
    nearest_neighbor_distance: torch.Tensor
    mean_speed: torch.Tensor


def compute_team_metrics(positions: torch.Tensor, velocities_xy: torch.Tensor) -> TeamMetrics:
    centroid = positions.mean(dim=1)
    centered = positions[:, :, :2] - centroid[:, None, :2]
    dispersion = torch.mean(torch.sum(centered.square(), dim=-1), dim=-1)
    pairwise = pairwise_distances_xy(positions)
    dmax = masked_pairwise_max(pairwise)
    mean_pairwise_distance = mean_offdiag(pairwise)
    n_agents = positions.shape[1]
    eye = torch.eye(n_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
    nearest = pairwise.masked_fill(eye, float("inf")).amin(dim=-1)
    nearest_neighbor_distance = torch.where(torch.isinf(nearest), torch.zeros_like(nearest), nearest)
    mean_speed = torch.linalg.norm(velocities_xy, dim=-1).mean(dim=-1)
    return TeamMetrics(
        centroid=centroid,
        dmax=dmax,
        dispersion=dispersion,
        mean_pairwise_distance=mean_pairwise_distance,
        nearest_neighbor_distance=nearest_neighbor_distance,
        mean_speed=mean_speed,
    )


def success_rate(success: torch.Tensor) -> float:
    return float(success.float().mean().detach().cpu())

