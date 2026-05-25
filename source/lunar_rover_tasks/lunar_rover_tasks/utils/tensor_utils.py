"""Tensor construction helpers."""

from __future__ import annotations

import torch


def as_float_tensor(value, device: torch.device | str) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def zeros(shape: tuple[int, ...], device: torch.device | str) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float32, device=device)

