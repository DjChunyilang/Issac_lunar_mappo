"""Training-only oracle helpers."""

from __future__ import annotations

import torch


def compute_geometric_median(points: torch.Tensor, iterations: int = 32, eps: float = 1.0e-6) -> torch.Tensor:
    """Compute a batched Weiszfeld geometric median for points shaped ``[E, N, 3]``."""
    estimate = points.mean(dim=1)
    for _ in range(iterations):
        diff = points - estimate[:, None, :]
        dist = torch.linalg.norm(diff, dim=-1).clamp_min(eps)
        weights = 1.0 / dist
        estimate = (points * weights[..., None]).sum(dim=1) / weights.sum(dim=1, keepdim=True)
    return estimate


def compute_oracle_distances(positions: torch.Tensor, oracle_point: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(positions[:, :, :2] - oracle_point[:, None, :2], dim=-1)


def compute_mean_oracle_distance(positions: torch.Tensor, oracle_point: torch.Tensor) -> torch.Tensor:
    return compute_oracle_distances(positions, oracle_point).mean(dim=-1)


def build_oracle_features(
    positions: torch.Tensor,
    centroid: torch.Tensor,
    oracle_point: torch.Tensor,
) -> torch.Tensor:
    distances = compute_oracle_distances(positions, oracle_point)
    mean_distance = distances.mean(dim=-1, keepdim=True)
    centroid_gap = torch.linalg.norm(centroid[:, :2] - oracle_point[:, :2], dim=-1, keepdim=True)
    return torch.cat((oracle_point, distances, mean_distance, centroid_gap), dim=-1)

