"""Structured terrain features for proxy training and high-fidelity evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import ObservationCfg, TerrainCfg


TERRAIN_FEATURE_NAMES = ("height", "slope_x", "slope_y", "roughness", "traversability")
LOCAL_TERRAIN_GRID_X = (-0.4, 0.0, 0.4, 0.8, 1.2)
LOCAL_TERRAIN_GRID_Y = (-0.8, -0.4, 0.0, 0.4, 0.8)
LOCAL_TERRAIN_GRID_CHANNELS = ("relative_height", "risk")

# The v9 decentralized policy uses a bounded body-frame multi-scale map.  The
# near patch resolves the smallest configured crater rims, the middle patch
# covers the complete action horizon, and the coarse patch supplies directional
# context without constructing a global map.
MULTISCALE_TERRAIN_FINE_X = (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8)
MULTISCALE_TERRAIN_FINE_Y = (-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8)
MULTISCALE_TERRAIN_MEDIUM_X = (0.8, 1.2, 1.6)
MULTISCALE_TERRAIN_MEDIUM_Y = (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2)
MULTISCALE_TERRAIN_COARSE_X = (1.6, 2.4, 3.2, 4.0)
MULTISCALE_TERRAIN_COARSE_Y = (-2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4)
MULTISCALE_TERRAIN_DIMS = (126, 42, 56)
MULTISCALE_TERRAIN_DIM = sum(MULTISCALE_TERRAIN_DIMS)
MULTISCALE_SITE_BELIEF_DIMS = (189, 63, 84)
MULTISCALE_SITE_BELIEF_DIM = sum(MULTISCALE_SITE_BELIEF_DIMS)


@dataclass(slots=True)
class TerrainRuntime:
    """Per-environment procedural terrain parameters held fixed for one episode."""

    translation_xy: torch.Tensor
    yaw: torch.Tensor
    phase: torch.Tensor
    amplitude_scale: torch.Tensor
    crater_radius_scale: torch.Tensor
    crater_depth_scale: torch.Tensor
    topology_bucket: torch.Tensor

    def subset(self, env_ids: torch.Tensor) -> TerrainRuntime:
        return TerrainRuntime(
            translation_xy=self.translation_xy[env_ids],
            yaw=self.yaw[env_ids],
            phase=self.phase[env_ids],
            amplitude_scale=self.amplitude_scale[env_ids],
            crater_radius_scale=self.crater_radius_scale[env_ids],
            crater_depth_scale=self.crater_depth_scale[env_ids],
            topology_bucket=self.topology_bucket[env_ids],
        )

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> TerrainRuntime:
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return TerrainRuntime(
            translation_xy=self.translation_xy.to(**kwargs),
            yaw=self.yaw.to(**kwargs),
            phase=self.phase.to(**kwargs),
            amplitude_scale=self.amplitude_scale.to(**kwargs),
            crater_radius_scale=self.crater_radius_scale.to(**kwargs),
            crater_depth_scale=self.crater_depth_scale.to(**kwargs),
            topology_bucket=self.topology_bucket.to(device=device),
        )

    def clone(self) -> TerrainRuntime:
        return TerrainRuntime(
            translation_xy=self.translation_xy.clone(),
            yaw=self.yaw.clone(),
            phase=self.phase.clone(),
            amplitude_scale=self.amplitude_scale.clone(),
            crater_radius_scale=self.crater_radius_scale.clone(),
            crater_depth_scale=self.crater_depth_scale.clone(),
            topology_bucket=self.topology_bucket.clone(),
        )


@dataclass(slots=True)
class GatherPointFlatness:
    """Local terrain-patch measurements used by search and the success gate."""

    height_range: torch.Tensor
    max_slope: torch.Tensor
    mean_slope: torch.Tensor
    is_flat: torch.Tensor


@dataclass(slots=True)
class LocalFlatnessCenterSearch:
    """Best nearby valid success-footprint center for each environment."""

    target_xy: torch.Tensor
    found_flat: torch.Tensor
    flatness: GatherPointFlatness


def make_terrain_runtime(
    num_envs: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> TerrainRuntime:
    ones = torch.ones(num_envs, device=device, dtype=dtype)
    return TerrainRuntime(
        translation_xy=torch.zeros(num_envs, 2, device=device, dtype=dtype),
        yaw=torch.zeros(num_envs, device=device, dtype=dtype),
        phase=torch.zeros(num_envs, device=device, dtype=dtype),
        amplitude_scale=ones.clone(),
        crater_radius_scale=ones.clone(),
        crater_depth_scale=ones.clone(),
        topology_bucket=torch.zeros(num_envs, device=device, dtype=torch.long),
    )


def _uniform_(
    target: torch.Tensor,
    low: float,
    high: float,
    *,
    generator: torch.Generator,
) -> None:
    if high < low:
        raise ValueError(f"Terrain randomization range must satisfy min <= max, got {low} > {high}.")
    if high == low:
        target.fill_(low)
    else:
        target.uniform_(low, high, generator=generator)


def randomize_terrain_runtime(
    runtime: TerrainRuntime,
    env_ids: torch.Tensor,
    terrain_cfg: TerrainCfg,
    *,
    generator: torch.Generator,
) -> None:
    env_ids = env_ids.to(device=runtime.yaw.device, dtype=torch.long)
    count = int(env_ids.numel())
    if count == 0:
        return
    topology_profile = str(terrain_cfg.topology_profile).lower()
    if topology_profile == "mixed_bottleneck_mix":
        runtime.topology_bucket[env_ids] = env_ids.remainder(2)
    elif topology_profile == "runtime_bucketed":
        stage = str(terrain_cfg.topology_curriculum_stage).lower()
        if stage == "open":
            runtime.topology_bucket[env_ids] = 0
        elif stage == "mixed_bottleneck":
            runtime.topology_bucket[env_ids] = 1 + env_ids.remainder(2)
        else:
            raise ValueError(
                "runtime_bucketed terrain requires topology_curriculum_stage "
                f"open or mixed_bottleneck, got {terrain_cfg.topology_curriculum_stage}."
            )
    else:
        runtime.topology_bucket[env_ids] = 0
    if not terrain_cfg.randomize_per_reset:
        runtime.translation_xy[env_ids] = 0.0
        runtime.yaw[env_ids] = 0.0
        runtime.phase[env_ids] = 0.0
        runtime.amplitude_scale[env_ids] = 1.0
        runtime.crater_radius_scale[env_ids] = 1.0
        runtime.crater_depth_scale[env_ids] = 1.0
        return

    translation = torch.empty(count, 2, device=runtime.translation_xy.device, dtype=runtime.translation_xy.dtype)
    _uniform_(
        translation,
        -float(terrain_cfg.random_translation_m),
        float(terrain_cfg.random_translation_m),
        generator=generator,
    )
    runtime.translation_xy[env_ids] = translation
    values = torch.empty(count, device=runtime.yaw.device, dtype=runtime.yaw.dtype)
    _uniform_(
        values,
        -float(terrain_cfg.random_yaw_rad),
        float(terrain_cfg.random_yaw_rad),
        generator=generator,
    )
    runtime.yaw[env_ids] = values
    values.uniform_(0.0, 2.0 * torch.pi, generator=generator)
    runtime.phase[env_ids] = values
    _uniform_(
        values,
        float(terrain_cfg.amplitude_scale_min),
        float(terrain_cfg.amplitude_scale_max),
        generator=generator,
    )
    runtime.amplitude_scale[env_ids] = values
    _uniform_(
        values,
        float(terrain_cfg.crater_radius_scale_min),
        float(terrain_cfg.crater_radius_scale_max),
        generator=generator,
    )
    runtime.crater_radius_scale[env_ids] = values
    _uniform_(
        values,
        float(terrain_cfg.crater_depth_scale_min),
        float(terrain_cfg.crater_depth_scale_max),
        generator=generator,
    )
    runtime.crater_depth_scale[env_ids] = values


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
    runtime: TerrainRuntime | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = max(0, int(terrain_cfg.crater_count))
    if count <= 0:
        return (
            torch.zeros(0, 2, device=device, dtype=dtype),
            torch.zeros(0, device=device, dtype=dtype),
        )
    if count == 1:
        if runtime is None:
            centers = torch.zeros(1, 2, device=device, dtype=dtype)
            radii = torch.full((1,), float(terrain_cfg.crater_max_radius), device=device, dtype=dtype)
        else:
            centers = torch.zeros(runtime.yaw.shape[0], 1, 2, device=device, dtype=dtype)
            radii = (
                torch.full(
                    (runtime.yaw.shape[0], 1),
                    float(terrain_cfg.crater_max_radius),
                    device=device,
                    dtype=dtype,
                )
                * runtime.crater_radius_scale[:, None]
            )
        return centers, radii

    index = torch.arange(count, device=device, dtype=dtype)
    seed_phase = float(terrain_cfg.crater_seed) * 0.61803398875
    field_radius = 0.45 * float(terrain_cfg.crater_field_size)
    radial = field_radius * torch.sqrt((index + 0.5) / float(count))
    if runtime is None:
        theta = index * 2.39996322973 + seed_phase
        centers = torch.stack((radial * torch.cos(theta), radial * torch.sin(theta)), dim=-1)
        radius_mix = 0.5 + 0.5 * torch.sin(index * 12.9898 + seed_phase)
        radii = float(terrain_cfg.crater_min_radius) + (
            float(terrain_cfg.crater_max_radius) - float(terrain_cfg.crater_min_radius)
        ) * radius_mix
        return centers, radii.clamp_min(1.0e-3)

    phase = runtime.phase[:, None]
    theta = index[None, :] * 2.39996322973 + seed_phase + phase
    centers = torch.stack((radial[None, :] * torch.cos(theta), radial[None, :] * torch.sin(theta)), dim=-1)
    radius_mix = 0.5 + 0.5 * torch.sin(index[None, :] * 12.9898 + seed_phase + phase)
    radii = (
        float(terrain_cfg.crater_min_radius)
        + (float(terrain_cfg.crater_max_radius) - float(terrain_cfg.crater_min_radius))
        * radius_mix
    ) * runtime.crater_radius_scale[:, None]
    return centers, radii.clamp_min(1.0e-3)


def _runtime_scalar(value: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    return value.view(value.shape[0], *((1,) * (xy.ndim - 2)))


def _terrain_local_xy(xy: torch.Tensor, runtime: TerrainRuntime | None) -> torch.Tensor:
    if runtime is None:
        return xy
    if xy.shape[0] != runtime.yaw.shape[0]:
        raise ValueError(
            f"Terrain runtime has {runtime.yaw.shape[0]} environments, but xy has leading dim {xy.shape[0]}."
        )
    translation = runtime.translation_xy.view(
        runtime.translation_xy.shape[0],
        *((1,) * (xy.ndim - 2)),
        2,
    )
    centered = xy - translation
    yaw = _runtime_scalar(runtime.yaw, xy)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    local_x = cos_yaw * centered[..., 0] + sin_yaw * centered[..., 1]
    local_y = -sin_yaw * centered[..., 0] + cos_yaw * centered[..., 1]
    return torch.stack((local_x, local_y), dim=-1)


def _heightfield_height(
    xy: torch.Tensor,
    terrain_cfg: TerrainCfg | None,
    runtime: TerrainRuntime | None = None,
) -> torch.Tensor:
    if _is_flat(terrain_cfg):
        return torch.zeros(*xy.shape[:-1], dtype=xy.dtype, device=xy.device)

    assert terrain_cfg is not None
    wavelength = max(float(terrain_cfg.wavelength), 1.0e-6)
    amplitude = float(terrain_cfg.amplitude)
    k = 2.0 * torch.pi / wavelength
    local_xy = _terrain_local_xy(xy, runtime)
    x = local_xy[..., 0]
    y = local_xy[..., 1]
    phase = 0.35 if runtime is None else 0.35 + _runtime_scalar(runtime.phase, xy)
    amplitude_scale = 1.0 if runtime is None else _runtime_scalar(runtime.amplitude_scale, xy)

    height = torch.zeros_like(x)
    if amplitude != 0.0:
        height = height + amplitude * amplitude_scale * (
            torch.sin(k * x) * torch.cos(k * y)
            + 0.45 * torch.sin(1.7 * k * x + phase) * torch.sin(1.3 * k * y)
        )

    topology_profile = str(terrain_cfg.topology_profile).lower()
    if topology_profile not in {
        "procedural",
        "open",
        "mixed",
        "bottleneck",
        "mixed_bottleneck_mix",
        "runtime_bucketed",
    }:
        raise ValueError(f"Unsupported terrain topology_profile: {terrain_cfg.topology_profile}")
    if topology_profile in {"bottleneck", "mixed_bottleneck_mix", "runtime_bucketed"}:
        wall_half_width = max(float(terrain_cfg.bottleneck_wall_half_width), 1.0e-3)
        gap_half_width = max(float(terrain_cfg.bottleneck_gap_half_width), 1.0e-3)
        # A smooth ridge spans the map except for one central passage. Its
        # slope, not an invisible obstacle mask, makes the wall non-traversable.
        ridge = torch.exp(-0.5 * (x / wall_half_width).square())
        passage = 1.0 - torch.exp(-0.5 * (y / gap_half_width).pow(4))
        bottleneck_weight: torch.Tensor | float = 1.0
        if topology_profile == "mixed_bottleneck_mix":
            if runtime is None:
                raise ValueError(
                    "mixed_bottleneck_mix requires per-environment TerrainRuntime."
                )
            bottleneck_weight = _runtime_scalar(
                runtime.topology_bucket.to(dtype=xy.dtype), xy
            )
        elif topology_profile == "runtime_bucketed":
            if runtime is None:
                bottleneck_weight = (
                    0.0
                    if str(terrain_cfg.topology_curriculum_stage).lower() == "open"
                    else 1.0
                )
            else:
                bottleneck_weight = _runtime_scalar(
                    (runtime.topology_bucket == 2).to(dtype=xy.dtype), xy
                )
        height = height + (
            float(terrain_cfg.bottleneck_wall_height)
            * bottleneck_weight
            * ridge
            * passage
        )

    crater_profile_enabled = topology_profile != "open"
    if topology_profile == "runtime_bucketed":
        crater_profile_enabled = (
            bool((runtime.topology_bucket > 0).any())
            if runtime is not None
            else str(terrain_cfg.topology_curriculum_stage).lower() != "open"
        )
    if crater_profile_enabled and (
        terrain_cfg.type == "lunar_crater_proxy" or terrain_cfg.crater_count > 0
    ):
        centers, radii = _crater_layout(terrain_cfg, xy.device, xy.dtype, runtime)
        if centers.numel() > 0:
            if runtime is None:
                delta = local_xy[..., None, :] - centers
                radius = radii.view(*((1,) * (local_xy.ndim - 1)), -1)
                depth_scale = 1.0
            else:
                spatial_dims = (1,) * (local_xy.ndim - 2)
                centers = centers.view(centers.shape[0], *spatial_dims, centers.shape[1], 2)
                radius = radii.view(radii.shape[0], *spatial_dims, radii.shape[1])
                delta = local_xy[..., None, :] - centers
                depth_scale = _runtime_scalar(runtime.crater_depth_scale, xy)[..., None]
            distance = torch.linalg.norm(delta, dim=-1)
            diameter = 2.0 * radius
            depth = float(terrain_cfg.crater_depth_to_diameter) * diameter * depth_scale
            rim_height = float(terrain_cfg.crater_rim_height_to_diameter) * diameter
            normalized_distance = distance / radius.clamp_min(1.0e-6)
            bowl_profile = torch.clamp(1.0 - normalized_distance.square(), min=0.0).square()
            rim_width = 0.22
            rim_profile = torch.exp(-((normalized_distance - 1.0) / rim_width).square())
            crater_height = (-depth * bowl_profile + rim_height * rim_profile).sum(dim=-1)
            if topology_profile == "runtime_bucketed":
                if runtime is not None:
                    crater_height = crater_height * _runtime_scalar(
                        (runtime.topology_bucket > 0).to(dtype=xy.dtype), xy
                    )
            height = height + crater_height
    return height


def _base_features(
    xy: torch.Tensor,
    terrain_cfg: TerrainCfg | None,
    runtime: TerrainRuntime | None = None,
) -> torch.Tensor:
    if _is_flat(terrain_cfg):
        return torch.zeros(*xy.shape[:-1], 5, dtype=xy.dtype, device=xy.device)

    height = _heightfield_height(xy, terrain_cfg, runtime)
    eps = torch.tensor(0.05, dtype=xy.dtype, device=xy.device)
    dx = torch.zeros_like(xy)
    dy = torch.zeros_like(xy)
    dx[..., 0] = eps
    dy[..., 1] = eps
    slope_x = (
        _heightfield_height(xy + dx, terrain_cfg, runtime)
        - _heightfield_height(xy - dx, terrain_cfg, runtime)
    ) / (2.0 * eps)
    slope_y = (
        _heightfield_height(xy + dy, terrain_cfg, runtime)
        - _heightfield_height(xy - dy, terrain_cfg, runtime)
    ) / (2.0 * eps)
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


def query_height(
    xy: torch.Tensor,
    terrain_cfg: TerrainCfg | None = None,
    runtime: TerrainRuntime | None = None,
) -> torch.Tensor:
    return _base_features(xy, terrain_cfg, runtime)[..., :1]


def query_terrain_features(
    xy: torch.Tensor,
    terrain_cfg: TerrainCfg | None = None,
    runtime: TerrainRuntime | None = None,
) -> torch.Tensor:
    """Return height, slope_x, slope_y, roughness, traversability at xy points."""
    return _base_features(xy, terrain_cfg, runtime)


def gather_point_flatness_offsets(
    radius: float,
    *,
    rings: int,
    samples_per_ring: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a deterministic, symmetric disk footprint including its center."""
    if radius < 0.0:
        raise ValueError("Gather-point flatness radius must be non-negative.")
    if rings <= 0:
        raise ValueError("Gather-point flatness rings must be positive.")
    if samples_per_ring < 4:
        raise ValueError("Gather-point flatness samples_per_ring must be at least 4.")

    angles = (
        torch.arange(samples_per_ring, device=device, dtype=dtype)
        * (2.0 * torch.pi / float(samples_per_ring))
    )
    unit_ring = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    ring_radii = torch.linspace(
        float(radius) / float(rings),
        float(radius),
        rings,
        device=device,
        dtype=dtype,
    )
    ring_offsets = (ring_radii[:, None, None] * unit_ring[None, :, :]).reshape(-1, 2)
    return torch.cat((torch.zeros(1, 2, device=device, dtype=dtype), ring_offsets), dim=0)


def evaluate_gather_point_flatness(
    points_xy: torch.Tensor,
    terrain_cfg: TerrainCfg | None = None,
    runtime: TerrainRuntime | None = None,
    *,
    radius: float,
    rings: int,
    samples_per_ring: int,
    max_height_range: float,
    max_slope: float,
) -> GatherPointFlatness:
    """Measure whether the complete gathering footprint is sufficiently flat.

    ``points_xy`` may be shaped ``[E, 2]`` or ``[E, ..., 2]``. The returned
    tensors have shape ``points_xy.shape[:-1]``. A patch must satisfy both the
    height-range and maximum slope constraints; checking only the center slope
    would incorrectly classify symmetric crater bottoms as flat.
    """
    if points_xy.ndim < 2 or points_xy.shape[-1] != 2:
        raise ValueError("points_xy must have shape [num_envs, ..., 2].")
    if radius < 0.0:
        raise ValueError("Gather-point flatness radius must be non-negative.")
    if rings <= 0:
        raise ValueError("Gather-point flatness rings must be positive.")
    if samples_per_ring < 4:
        raise ValueError("Gather-point flatness samples_per_ring must be at least 4.")
    if max_height_range < 0.0:
        raise ValueError("Gather-point max_height_range must be non-negative.")
    if max_slope < 0.0:
        raise ValueError("Gather-point max_slope must be non-negative.")

    output_shape = points_xy.shape[:-1]
    if _is_flat(terrain_cfg):
        zeros = torch.zeros(output_shape, dtype=points_xy.dtype, device=points_xy.device)
        return GatherPointFlatness(
            height_range=zeros,
            max_slope=zeros.clone(),
            mean_slope=zeros.clone(),
            is_flat=torch.ones(output_shape, dtype=torch.bool, device=points_xy.device),
        )

    offsets = gather_point_flatness_offsets(
        radius,
        rings=rings,
        samples_per_ring=samples_per_ring,
        device=points_xy.device,
        dtype=points_xy.dtype,
    )
    offset_shape = (1,) * (points_xy.ndim - 1) + offsets.shape
    sample_xy = points_xy[..., None, :] + offsets.view(offset_shape)
    features = query_terrain_features(sample_xy, terrain_cfg, runtime)
    height = features[..., 0]
    slope = torch.linalg.norm(features[..., 1:3], dim=-1)
    height_range = height.amax(dim=-1) - height.amin(dim=-1)
    patch_max_slope = slope.amax(dim=-1)
    mean_patch_slope = slope.mean(dim=-1)
    is_flat = (height_range <= float(max_height_range)) & (
        patch_max_slope <= float(max_slope)
    )
    return GatherPointFlatness(
        height_range=height_range,
        max_slope=patch_max_slope,
        mean_slope=mean_patch_slope,
        is_flat=is_flat,
    )


def search_local_flatness_center(
    centers_xy: torch.Tensor,
    terrain_cfg: TerrainCfg | None = None,
    runtime: TerrainRuntime | None = None,
    *,
    search_radius: float,
    samples: int,
    flatness_radius: float,
    flatness_rings: int,
    flatness_samples_per_ring: int,
    max_height_range: float,
    max_slope: float,
) -> LocalFlatnessCenterSearch:
    """Find a nearby actually-flat footprint, preferring flatter/closer sites.

    The centre itself and an evenly spaced ring are evaluated with exactly the
    same complete-disk gate as task success.  A caller can retain its existing
    target when ``found_flat`` is false; this helper never treats the least
    bad non-flat candidate as valid.
    """
    if centers_xy.ndim != 2 or centers_xy.shape[-1] != 2:
        raise ValueError("centers_xy must have shape [num_envs, 2].")
    if search_radius < 0.0:
        raise ValueError("search_radius must be non-negative.")
    if samples < 4:
        raise ValueError("samples must be at least 4.")

    if search_radius == 0.0:
        candidates = centers_xy[:, None, :]
    else:
        angles = (
            torch.arange(samples, device=centers_xy.device, dtype=centers_xy.dtype)
            * (2.0 * torch.pi / float(samples))
        )
        ring = float(search_radius) * torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        candidates = torch.cat((centers_xy[:, None, :], centers_xy[:, None, :] + ring[None]), dim=1)

    flatness = evaluate_gather_point_flatness(
        candidates,
        terrain_cfg,
        runtime,
        radius=flatness_radius,
        rings=flatness_rings,
        samples_per_ring=flatness_samples_per_ring,
        max_height_range=max_height_range,
        max_slope=max_slope,
    )
    height_cost = flatness.height_range / max(float(max_height_range), 1.0e-6)
    slope_cost = flatness.max_slope / max(float(max_slope), 1.0e-6)
    terrain_cost = torch.maximum(height_cost, slope_cost)
    displacement = torch.linalg.norm(candidates - centers_xy[:, None, :], dim=-1)
    distance_cost = displacement / max(float(search_radius), 1.0e-6)
    # Feasible sites dominate infeasible ones. The small displacement term
    # makes an already-flat current centre stable and breaks ring ties.
    score = terrain_cost + 0.02 * distance_cost
    score = torch.where(flatness.is_flat, score, torch.full_like(score, float("inf")))
    found_flat = flatness.is_flat.any(dim=-1)
    selected = score.argmin(dim=-1)
    target_xy = candidates[torch.arange(centers_xy.shape[0], device=centers_xy.device), selected]
    return LocalFlatnessCenterSearch(
        target_xy=target_xy,
        found_flat=found_flat,
        flatness=flatness,
    )


def sample_path_terrain_risk(
    start_positions: torch.Tensor,
    target_positions: torch.Tensor,
    terrain_cfg: TerrainCfg | None = None,
    runtime: TerrainRuntime | None = None,
    *,
    num_samples: int = 5,
) -> dict[str, torch.Tensor]:
    """Summarize terrain risk along straight paths from current rover pose to subgoals.

    Returns per-rover tensors with shape ``[num_envs, num_agents]``:
    ``risk_mean``, ``risk_max`` and ``height_change_mean``. Flat terrain returns zeros.
    """
    shape = start_positions.shape[:-1]
    if _is_flat(terrain_cfg):
        zeros = torch.zeros(shape, dtype=start_positions.dtype, device=start_positions.device)
        return {
            "risk_mean": zeros,
            "risk_max": zeros,
            "height_change_mean": zeros,
        }
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    fractions = torch.linspace(
        1.0 / float(num_samples),
        1.0,
        num_samples,
        dtype=start_positions.dtype,
        device=start_positions.device,
    )
    start_xy = start_positions[..., :2]
    target_xy = target_positions[..., :2]
    sample_xy = start_xy[..., None, :] + (
        target_xy - start_xy
    )[..., None, :] * fractions.view(*([1] * (start_xy.ndim - 1)), num_samples, 1)
    features = query_terrain_features(sample_xy, terrain_cfg, runtime)
    risk = (1.0 - features[..., 4]).clamp(0.0, 1.0)
    start_height = query_height(start_xy, terrain_cfg, runtime)
    height_change = (features[..., 0] - start_height).abs()
    return {
        "risk_mean": risk.mean(dim=-1),
        "risk_max": risk.amax(dim=-1),
        "height_change_mean": height_change.mean(dim=-1),
    }


def sample_trajectory_terrain_risk(
    trajectory_points: torch.Tensor,
    terrain_cfg: TerrainCfg | None = None,
    runtime: TerrainRuntime | None = None,
) -> dict[str, torch.Tensor]:
    """Summarize risk on the actual generated trajectory samples.

    ``trajectory_points`` has shape ``[num_envs, num_agents, samples, 3]``.
    Unlike :func:`sample_path_terrain_risk`, this function does not replace the
    trajectory geometry with a straight segment to the endpoint.
    """
    if trajectory_points.ndim != 4 or trajectory_points.shape[-1] != 3:
        raise ValueError(
            "trajectory_points must have shape [num_envs, num_agents, samples, 3]."
        )
    if trajectory_points.shape[-2] < 1:
        raise ValueError("trajectory_points must contain at least one sample.")
    shape = trajectory_points.shape[:2]
    if _is_flat(terrain_cfg):
        zeros = torch.zeros(
            shape,
            dtype=trajectory_points.dtype,
            device=trajectory_points.device,
        )
        return {
            "risk_mean": zeros,
            "risk_max": zeros,
            "height_change_mean": zeros,
        }

    sample_xy = trajectory_points[..., :2]
    features = query_terrain_features(sample_xy, terrain_cfg, runtime)
    risk = (1.0 - features[..., 4]).clamp(0.0, 1.0)
    start_height = query_height(sample_xy[..., 0, :], terrain_cfg, runtime)
    height_change = (features[..., 0] - start_height).abs()
    return {
        "risk_mean": risk.mean(dim=-1),
        "risk_max": risk.amax(dim=-1),
        "height_change_mean": height_change.mean(dim=-1),
    }


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


def _body_grid_world_points(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    x_coordinates: tuple[float, ...],
    y_coordinates: tuple[float, ...],
) -> torch.Tensor:
    x = torch.tensor(x_coordinates, device=positions.device, dtype=positions.dtype)
    y = torch.tensor(y_coordinates, device=positions.device, dtype=positions.dtype)
    local_x, local_y = torch.meshgrid(x, y, indexing="ij")
    cos_yaw = torch.cos(yaws)[..., None, None]
    sin_yaw = torch.sin(yaws)[..., None, None]
    world_x = positions[..., 0, None, None] + cos_yaw * local_x - sin_yaw * local_y
    world_y = positions[..., 1, None, None] + sin_yaw * local_x + cos_yaw * local_y
    return torch.stack((world_x, world_y), dim=-1)


def _build_body_terrain_grid(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    x_coordinates: tuple[float, ...],
    y_coordinates: tuple[float, ...],
    terrain_cfg: TerrainCfg | None,
    runtime: TerrainRuntime | None,
) -> torch.Tensor:
    shape = (
        *positions.shape[:-1],
        len(x_coordinates),
        len(y_coordinates),
        len(LOCAL_TERRAIN_GRID_CHANNELS),
    )
    if _is_flat(terrain_cfg):
        return torch.zeros(shape, dtype=positions.dtype, device=positions.device)
    sample_xy = _body_grid_world_points(
        positions,
        yaws,
        x_coordinates,
        y_coordinates,
    )
    sample_features = _base_features(sample_xy, terrain_cfg, runtime)
    base_height = _heightfield_height(positions[..., :2], terrain_cfg, runtime)[..., None, None]
    relative_height = sample_features[..., 0] - base_height
    risk = (1.0 - sample_features[..., 4]).clamp(0.0, 1.0)
    return torch.stack((relative_height, risk), dim=-1)


def build_multiscale_local_terrain_grids(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    terrain_cfg: TerrainCfg | None = None,
    runtime: TerrainRuntime | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the fine, medium and coarse v9 body-frame terrain grids."""

    specifications = (
        (MULTISCALE_TERRAIN_FINE_X, MULTISCALE_TERRAIN_FINE_Y),
        (MULTISCALE_TERRAIN_MEDIUM_X, MULTISCALE_TERRAIN_MEDIUM_Y),
        (MULTISCALE_TERRAIN_COARSE_X, MULTISCALE_TERRAIN_COARSE_Y),
    )
    return tuple(
        _build_body_terrain_grid(
            positions,
            yaws,
            x_coordinates,
            y_coordinates,
            terrain_cfg,
            runtime,
        )
        for x_coordinates, y_coordinates in specifications
    )


def build_multiscale_local_terrain_observation(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    terrain_cfg: TerrainCfg | None = None,
    runtime: TerrainRuntime | None = None,
) -> torch.Tensor:
    """Flatten and concatenate the three v9 grids in fine-to-coarse order."""

    grids = build_multiscale_local_terrain_grids(
        positions,
        yaws,
        terrain_cfg,
        runtime,
    )
    observation = torch.cat(tuple(grid.flatten(start_dim=-3) for grid in grids), dim=-1)
    if observation.shape[-1] != MULTISCALE_TERRAIN_DIM:
        raise RuntimeError(
            f"Multi-scale terrain observation has dim {observation.shape[-1]}, "
            f"expected {MULTISCALE_TERRAIN_DIM}."
        )
    return observation


def build_multiscale_site_belief_observation(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    site_point: torch.Tensor,
    terrain_cfg: TerrainCfg | None = None,
    runtime: TerrainRuntime | None = None,
    *,
    site_radius: float = 0.75,
    potential_sigma: float = 2.0,
) -> torch.Tensor:
    """Append a spatial feasible-region potential to each terrain grid.

    This is the H1 diagnostic interface. ``site_point`` may be shared per
    environment with shape ``[E, 3]`` or supplied per rover as ``[E, A, 3]``.
    The channel is one throughout the feasible disk and decays smoothly with
    distance outside it; unlike a bearing-distance vector it remains aligned
    with the terrain cells consumed by the CNN.
    """
    if potential_sigma <= 0.0:
        raise ValueError("potential_sigma must be positive.")
    if site_radius < 0.0:
        raise ValueError("site_radius must be non-negative.")
    if site_point.ndim == positions.ndim - 1:
        target_xy = site_point[..., :2].unsqueeze(-2)
        target_xy = target_xy.expand(*positions.shape[:-1], 2)
    elif site_point.shape[:-1] == positions.shape[:-1]:
        target_xy = site_point[..., :2]
    else:
        raise ValueError(
            "site_point must have shape [E, 3] or match the per-rover position shape."
        )

    terrain_grids = build_multiscale_local_terrain_grids(
        positions,
        yaws,
        terrain_cfg,
        runtime,
    )
    specifications = (
        (MULTISCALE_TERRAIN_FINE_X, MULTISCALE_TERRAIN_FINE_Y),
        (MULTISCALE_TERRAIN_MEDIUM_X, MULTISCALE_TERRAIN_MEDIUM_Y),
        (MULTISCALE_TERRAIN_COARSE_X, MULTISCALE_TERRAIN_COARSE_Y),
    )
    grids: list[torch.Tensor] = []
    for terrain_grid, (x_coordinates, y_coordinates) in zip(
        terrain_grids,
        specifications,
        strict=True,
    ):
        sample_xy = _body_grid_world_points(
            positions,
            yaws,
            x_coordinates,
            y_coordinates,
        )
        distance = torch.linalg.vector_norm(
            sample_xy - target_xy[..., None, None, :],
            dim=-1,
        )
        outside = (distance - float(site_radius)).clamp_min(0.0)
        potential = torch.exp(-0.5 * (outside / float(potential_sigma)).square())
        grids.append(torch.cat((terrain_grid, potential.unsqueeze(-1)), dim=-1))

    observation = torch.cat(tuple(grid.flatten(start_dim=-3) for grid in grids), dim=-1)
    if observation.shape[-1] != MULTISCALE_SITE_BELIEF_DIM:
        raise RuntimeError(
            f"Multi-scale site-belief observation has dim {observation.shape[-1]}, "
            f"expected {MULTISCALE_SITE_BELIEF_DIM}."
        )
    return observation


def build_local_terrain_grid(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    terrain_cfg: TerrainCfg | None = None,
    runtime: TerrainRuntime | None = None,
) -> torch.Tensor:
    """Return [relative_height, risk] over the fixed body-frame 5x5 grid."""
    shape = (*positions.shape[:-1], len(LOCAL_TERRAIN_GRID_X), len(LOCAL_TERRAIN_GRID_Y), 2)
    if _is_flat(terrain_cfg):
        return torch.zeros(shape, dtype=positions.dtype, device=positions.device)

    sample_xy = local_terrain_grid_world_points(positions, yaws)
    sample_features = _base_features(sample_xy, terrain_cfg, runtime)
    base_height = _heightfield_height(positions[..., :2], terrain_cfg, runtime)[..., None, None]
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


def summarize_local_terrain_grid_per_agent(grid: torch.Tensor) -> torch.Tensor:
    """Return the five sender-local terrain statistics for each rover."""
    if grid.ndim < 5 or grid.shape[-1] != 2:
        raise ValueError(
            "local terrain grid must have shape [..., agents, x, y, 2], got "
            f"{tuple(grid.shape)}."
        )
    relative_height = grid[..., 0]
    risk = grid[..., 1]
    reduce_dims = (-2, -1)
    return torch.stack(
        (
            relative_height.abs().mean(dim=reduce_dims),
            relative_height.clamp_min(0.0).amax(dim=reduce_dims),
            (-relative_height).clamp_min(0.0).amax(dim=reduce_dims),
            risk.mean(dim=reduce_dims),
            risk.amax(dim=reduce_dims),
        ),
        dim=-1,
    )


def build_terrain_features(
    positions: torch.Tensor,
    cfg: ObservationCfg,
    terrain_cfg: TerrainCfg | None = None,
    runtime: TerrainRuntime | None = None,
) -> torch.Tensor:
    """Return the legacy 5-D under-rover features used by dynamics and reward."""
    del cfg
    return _base_features(positions[..., :2], terrain_cfg, runtime)


def build_global_terrain_state(
    positions: torch.Tensor | None,
    dim: int,
    device: torch.device | str,
    terrain_cfg: TerrainCfg | None = None,
    yaws: torch.Tensor | None = None,
    runtime: TerrainRuntime | None = None,
) -> torch.Tensor:
    if positions is None or _is_flat(terrain_cfg):
        num_envs = 1 if positions is None else positions.shape[0]
        return torch.zeros(num_envs, dim, dtype=torch.float32, device=device)
    if yaws is None:
        yaws = torch.zeros(positions.shape[:-1], dtype=positions.dtype, device=positions.device)
    summary = summarize_local_terrain_grid(
        build_local_terrain_grid(positions, yaws, terrain_cfg, runtime)
    )
    return _fit_dim(summary, dim)
