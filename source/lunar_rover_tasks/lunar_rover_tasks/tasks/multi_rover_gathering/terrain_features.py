"""Structured terrain features for proxy training and high-fidelity evaluation."""

from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import ObservationCfg, TerrainCfg


TERRAIN_FEATURE_NAMES = ("height", "slope_x", "slope_y", "roughness", "traversability")


def is_flat_terrain(terrain_cfg: TerrainCfg | None) -> bool:
    return (
        terrain_cfg is None
        or terrain_cfg.type == "flat_proxy"
        or (terrain_cfg.amplitude == 0.0 and terrain_cfg.crater_count <= 0)
    )


def _is_flat(terrain_cfg: TerrainCfg | None) -> bool:
    return is_flat_terrain(terrain_cfg)


def _crater_layout(
    terrain_cfg: TerrainCfg,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = max(0, int(terrain_cfg.crater_count))
    if count <= 0:
        return (
            torch.zeros(0, 2, device=device, dtype=dtype),
            torch.zeros(0, device=device, dtype=dtype),
        )
    if count == 1:
        centers = torch.zeros(1, 2, device=device, dtype=dtype)
        radii = torch.full((1,), float(terrain_cfg.crater_max_radius), device=device, dtype=dtype)
        return centers, radii

    index = torch.arange(count, device=device, dtype=dtype)
    seed_phase = float(terrain_cfg.crater_seed) * 0.61803398875
    field_radius = 0.45 * float(terrain_cfg.crater_field_size)
    radial = field_radius * torch.sqrt((index + 0.5) / float(count))
    theta = index * 2.39996322973 + seed_phase
    centers = torch.stack((radial * torch.cos(theta), radial * torch.sin(theta)), dim=-1)
    radius_mix = 0.5 + 0.5 * torch.sin(index * 12.9898 + seed_phase)
    radii = float(terrain_cfg.crater_min_radius) + (
        float(terrain_cfg.crater_max_radius) - float(terrain_cfg.crater_min_radius)
    ) * radius_mix
    return centers, radii.clamp_min(1.0e-3)


def _heightfield_height(xy: torch.Tensor, terrain_cfg: TerrainCfg | None) -> torch.Tensor:
    if _is_flat(terrain_cfg):
        return torch.zeros(*xy.shape[:-1], dtype=xy.dtype, device=xy.device)

    assert terrain_cfg is not None
    wavelength = max(float(terrain_cfg.wavelength), 1.0e-6)
    amplitude = float(terrain_cfg.amplitude)
    k = 2.0 * torch.pi / wavelength
    x = xy[..., 0]
    y = xy[..., 1]
    phase = 0.35

    height = torch.zeros_like(x)
    if amplitude != 0.0:
        height = height + amplitude * (
            torch.sin(k * x) * torch.cos(k * y)
            + 0.45 * torch.sin(1.7 * k * x + phase) * torch.sin(1.3 * k * y)
        )

    if terrain_cfg.type == "lunar_crater_proxy" or terrain_cfg.crater_count > 0:
        centers, radii = _crater_layout(terrain_cfg, xy.device, xy.dtype)
        if centers.numel() > 0:
            delta = xy[..., None, :] - centers
            distance = torch.linalg.norm(delta, dim=-1)
            radius = radii.view(*((1,) * (distance.ndim - 1)), -1)
            diameter = 2.0 * radius
            depth = float(terrain_cfg.crater_depth_to_diameter) * diameter
            rim_height = float(terrain_cfg.crater_rim_height_to_diameter) * diameter
            normalized_distance = distance / radius.clamp_min(1.0e-6)
            bowl_profile = torch.clamp(1.0 - normalized_distance.square(), min=0.0).square()
            rim_width = 0.22
            rim_profile = torch.exp(-((normalized_distance - 1.0) / rim_width).square())
            height = height + (-depth * bowl_profile + rim_height * rim_profile).sum(dim=-1)
    return height


def _base_features(xy: torch.Tensor, terrain_cfg: TerrainCfg | None) -> torch.Tensor:
    if _is_flat(terrain_cfg):
        return torch.zeros(*xy.shape[:-1], 5, dtype=xy.dtype, device=xy.device)

    height = _heightfield_height(xy, terrain_cfg)
    eps = torch.tensor(0.05, dtype=xy.dtype, device=xy.device)
    dx = torch.zeros_like(xy)
    dy = torch.zeros_like(xy)
    dx[..., 0] = eps
    dy[..., 1] = eps
    slope_x = (_heightfield_height(xy + dx, terrain_cfg) - _heightfield_height(xy - dx, terrain_cfg)) / (2.0 * eps)
    slope_y = (_heightfield_height(xy + dy, terrain_cfg) - _heightfield_height(xy - dy, terrain_cfg)) / (2.0 * eps)
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


def query_terrain_features(xy: torch.Tensor, terrain_cfg: TerrainCfg | None = None) -> torch.Tensor:
    """Return height, slope_x, slope_y, roughness, traversability at xy points."""
    return _base_features(xy, terrain_cfg)


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
