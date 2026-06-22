"""Structured terrain features for proxy training and high-fidelity evaluation."""

from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import ObservationCfg, TerrainCfg


TERRAIN_FEATURE_NAMES = ("height", "slope_x", "slope_y", "roughness", "traversability")
LOCAL_TERRAIN_GRID_X = (-0.4, 0.0, 0.4, 0.8, 1.2)
LOCAL_TERRAIN_GRID_Y = (-0.8, -0.4, 0.0, 0.4, 0.8)
LOCAL_TERRAIN_GRID_CHANNELS = ("relative_height", "risk")


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


def local_terrain_grid_offsets(
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the fixed body-frame grid as [x_index, y_index, xy]."""
    x = torch.tensor(LOCAL_TERRAIN_GRID_X, device=device, dtype=dtype)
    y = torch.tensor(LOCAL_TERRAIN_GRID_Y, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


def local_terrain_grid_world_points(
    positions: torch.Tensor,
    yaws: torch.Tensor,
) -> torch.Tensor:
    """Transform the fixed body-frame grid into world-frame xy sample points."""
    offsets = local_terrain_grid_offsets(device=positions.device, dtype=positions.dtype)
    local_x = offsets[..., 0]
    local_y = offsets[..., 1]
    cos_yaw = torch.cos(yaws)[..., None, None]
    sin_yaw = torch.sin(yaws)[..., None, None]
    world_x = (
        positions[..., 0, None, None]
        + cos_yaw * local_x
        - sin_yaw * local_y
    )
    world_y = (
        positions[..., 1, None, None]
        + sin_yaw * local_x
        + cos_yaw * local_y
    )
    return torch.stack((world_x, world_y), dim=-1)


def build_local_terrain_grid(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    terrain_cfg: TerrainCfg | None = None,
) -> torch.Tensor:
    """Return [relative_height, risk] over the fixed body-frame 5x5 grid."""
    shape = (*positions.shape[:-1], len(LOCAL_TERRAIN_GRID_X), len(LOCAL_TERRAIN_GRID_Y), 2)
    if _is_flat(terrain_cfg):
        return torch.zeros(shape, dtype=positions.dtype, device=positions.device)

    sample_xy = local_terrain_grid_world_points(positions, yaws)
    sample_features = _base_features(sample_xy, terrain_cfg)
    base_height = _heightfield_height(positions[..., :2], terrain_cfg)[..., None, None]
    relative_height = sample_features[..., 0] - base_height
    risk = (1.0 - sample_features[..., 4]).clamp(0.0, 1.0)
    return torch.stack((relative_height, risk), dim=-1)


def flatten_local_terrain_grid(grid: torch.Tensor) -> torch.Tensor:
    """Flatten in x -> y -> channel order."""
    return grid.flatten(start_dim=-3)


def summarize_local_terrain_grid(grid: torch.Tensor) -> torch.Tensor:
    """Return the fixed 5-D centralized terrain summary for each environment."""
    relative_height = grid[..., 0]
    risk = grid[..., 1]
    reduce_dims = tuple(range(1, relative_height.ndim))
    mean_abs_height = relative_height.abs().mean(dim=reduce_dims)
    max_rise = relative_height.clamp_min(0.0).amax(dim=reduce_dims)
    max_descent = (-relative_height).clamp_min(0.0).amax(dim=reduce_dims)
    mean_risk = risk.mean(dim=reduce_dims)
    max_risk = risk.amax(dim=reduce_dims)
    return torch.stack(
        (mean_abs_height, max_rise, max_descent, mean_risk, max_risk),
        dim=-1,
    )


def build_terrain_features(
    positions: torch.Tensor,
    cfg: ObservationCfg,
    terrain_cfg: TerrainCfg | None = None,
) -> torch.Tensor:
    """Return the legacy 5-D under-rover features used by dynamics and reward."""
    del cfg
    return _base_features(positions[..., :2], terrain_cfg)


def build_global_terrain_state(
    positions: torch.Tensor | None,
    dim: int,
    device: torch.device | str,
    terrain_cfg: TerrainCfg | None = None,
    yaws: torch.Tensor | None = None,
) -> torch.Tensor:
    if positions is None or _is_flat(terrain_cfg):
        num_envs = 1 if positions is None else positions.shape[0]
        return torch.zeros(num_envs, dim, dtype=torch.float32, device=device)
    if yaws is None:
        yaws = torch.zeros(positions.shape[:-1], dtype=positions.dtype, device=positions.device)
    summary = summarize_local_terrain_grid(
        build_local_terrain_grid(positions, yaws, terrain_cfg)
    )
    return _fit_dim(summary, dim)
