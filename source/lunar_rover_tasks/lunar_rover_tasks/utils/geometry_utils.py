"""Geometry helpers for team statistics."""

from __future__ import annotations

import torch


def pairwise_distances_xy(positions: torch.Tensor) -> torch.Tensor:
    delta = positions[:, :, None, :2] - positions[:, None, :, :2]
    return torch.linalg.norm(delta, dim=-1)


def masked_pairwise_max(pairwise: torch.Tensor) -> torch.Tensor:
    n_agents = pairwise.shape[-1]
    mask = ~torch.eye(n_agents, dtype=torch.bool, device=pairwise.device).unsqueeze(0)
    return pairwise.masked_fill(~mask, 0.0).amax(dim=(-1, -2))


def mean_offdiag(pairwise: torch.Tensor) -> torch.Tensor:
    n_agents = pairwise.shape[-1]
    mask = ~torch.eye(n_agents, dtype=torch.bool, device=pairwise.device).unsqueeze(0)
    return pairwise.masked_select(mask).view(pairwise.shape[0], -1).mean(dim=-1)

