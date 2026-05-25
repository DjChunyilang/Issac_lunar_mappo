"""Structured terrain features for proxy training and high-fidelity evaluation."""

from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import ObservationCfg, TerrainCfg


TERRAIN_FEATURE_NAMES = ("height", "slope_x", "slope_y", "roughness", "traversability")


def _is_flat(terrain_cfg: TerrainCfg | None) -> bool:
    return terrain_cfg is None or terrain_cfg.type == "flat_proxy" or terrain_cfg.amplitude == 0.0


def _base_features(xy: torch.Tensor, terrain_cfg: TerrainCfg | None) -> torch.Tensor:
    if _is_flat(terrain_cfg):
        return torch.zeros(*xy.shape[:-1], 5, dtype=xy.dtype, device=xy.device)

    assert terrain_cfg is not None
    wavelength = max(float(terrain_cfg.wavelength), 1.0e-6)
    amplitude = float(terrain_cfg.amplitude)
    k = 2.0 * torch.pi / wavelength
    x = xy[..., 0]
    y = xy[..., 1]
    phase = 0.35

    height = amplitude * (
        torch.sin(k * x) * torch.cos(k * y)
        + 0.45 * torch.sin(1.7 * k * x + phase) * torch.sin(1.3 * k * y)
    )
    slope_x = amplitude * (
        k * torch.cos(k * x) * torch.cos(k * y)
        + 0.45 * 1.7 * k * torch.cos(1.7 * k * x + phase) * torch.sin(1.3 * k * y)
    )
    slope_y = amplitude * (
        -k * torch.sin(k * x) * torch.sin(k * y)
        + 0.45 * 1.3 * k * torch.sin(1.7 * k * x + phase) * torch.cos(1.3 * k * y)
    )
    roughness = torch.sqrt(slope_x.square() + slope_y.square()) * float(terrain_cfg.roughness_scale)
    traversability = torch.exp(
        -roughness / max(float(terrain_cfg.traversability_slope_scale), 1.0e-6)
    )
    return torch.stack((height, slope_x, slope_y, roughness, traversability), dim=-1)


def _fit_dim(features: torch.Tensor, dim: int) -> torch.Tensor:
    if features.shape[-1] == dim:
        return features
    if features.shape[-1] > dim:
        return features[..., :dim]
    pad = torch.zeros(*features.shape[:-1], dim - features.shape[-1], dtype=features.dtype, device=features.device)
    return torch.cat((features, pad), dim=-1)


def query_height(xy: torch.Tensor, terrain_cfg: TerrainCfg | None = None) -> torch.Tensor:
    return _base_features(xy, terrain_cfg)[..., :1]


def build_terrain_features(
    positions: torch.Tensor,
    cfg: ObservationCfg,
    terrain_cfg: TerrainCfg | None = None,
) -> torch.Tensor:
    return _fit_dim(_base_features(positions[..., :2], terrain_cfg), cfg.terrain_dim)


def build_global_terrain_state(
    positions: torch.Tensor | None,
    dim: int,
    device: torch.device | str,
    terrain_cfg: TerrainCfg | None = None,
) -> torch.Tensor:
    if positions is None or _is_flat(terrain_cfg):
        num_envs = 1 if positions is None else positions.shape[0]
        return torch.zeros(num_envs, dim, dtype=torch.float32, device=device)
    local_features = _base_features(positions[..., :2], terrain_cfg)
    mean_features = local_features.mean(dim=1)
    return _fit_dim(mean_features, dim)
