"""Training-only analytical PRD advantage utilities for exp159."""

from __future__ import annotations

import torch


def compute_analytical_prd_advantages(
    *,
    team_raw_advantages: torch.Tensor,
    loo_baseline: torch.Tensor,
    baseline_scale: float = 1.0,
) -> torch.Tensor:
    """Return jointly normalized single-step ALO-PRD advantages.

    ``team_raw_advantages`` is ``[T, E, 1]`` and ``loo_baseline`` is
    ``[T, E, A]``. No temporal recursion is applied to the LOO baseline.
    """

    if team_raw_advantages.ndim != 3 or team_raw_advantages.shape[-1] != 1:
        raise ValueError("team_raw_advantages must have shape [T, E, 1]")
    if loo_baseline.ndim != 3 or loo_baseline.shape[:2] != team_raw_advantages.shape[:2]:
        raise ValueError("loo_baseline must have shape [T, E, A]")
    if float(baseline_scale) != 1.0:
        raise ValueError("exp159 fixes baseline_scale at 1.0")
    raw = team_raw_advantages.unsqueeze(0) - loo_baseline.permute(2, 0, 1).unsqueeze(-1)
    return (raw - raw.mean()) / (raw.std() + 1.0e-8)
