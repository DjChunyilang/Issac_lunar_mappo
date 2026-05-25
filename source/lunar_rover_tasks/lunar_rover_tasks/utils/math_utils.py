"""Small tensor math helpers used by the proxy task."""

from __future__ import annotations

import math

import torch


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def yaw_to_matrix(yaw: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    row0 = torch.stack((cos_yaw, -sin_yaw), dim=-1)
    row1 = torch.stack((sin_yaw, cos_yaw), dim=-1)
    return torch.stack((row0, row1), dim=-2)


def rotate_2d(vec_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    rot = yaw_to_matrix(yaw)
    return torch.matmul(rot, vec_xy.unsqueeze(-1)).squeeze(-1)


def heading_from_delta(delta_xy: torch.Tensor, fallback: torch.Tensor | None = None) -> torch.Tensor:
    heading = torch.atan2(delta_xy[..., 1], delta_xy[..., 0])
    if fallback is None:
        return heading
    small = torch.linalg.norm(delta_xy, dim=-1) < 1.0e-6
    return torch.where(small, fallback, heading)


def seed_torch(seed: int, device: str) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def finite_or_raise(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def pi_tensor(device: torch.device | str) -> torch.Tensor:
    return torch.tensor(math.pi, dtype=torch.float32, device=device)

