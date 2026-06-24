"""Terrain visualization helpers shared by proxy validation scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import TerrainCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    TerrainRuntime,
    query_height,
)


def height_grid_for_extent(
    terrain_cfg: TerrainCfg | None,
    xy_min: np.ndarray,
    xy_max: np.ndarray,
    resolution: int = 140,
    terrain_runtime: TerrainRuntime | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[float, float]]:
    """Sample the configured terrain height over an xy extent."""
    xy_min = np.asarray(xy_min, dtype=np.float32)
    xy_max = np.asarray(xy_max, dtype=np.float32)
    if np.any(xy_max <= xy_min):
        center = 0.5 * (xy_min + xy_max)
        xy_min = center - 1.0
        xy_max = center + 1.0

    xs = np.linspace(float(xy_min[0]), float(xy_max[0]), resolution, dtype=np.float32)
    ys = np.linspace(float(xy_min[1]), float(xy_max[1]), resolution, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    device = terrain_runtime.yaw.device if terrain_runtime is not None else torch.device("cpu")
    xy = torch.from_numpy(np.stack((grid_x, grid_y), axis=-1)).to(device)
    if terrain_runtime is not None:
        if terrain_runtime.yaw.numel() != 1:
            raise ValueError("Terrain visualization expects exactly one terrain runtime.")
        xy = xy.unsqueeze(0)
    with torch.no_grad():
        height = query_height(xy, terrain_cfg, terrain_runtime).squeeze(-1).cpu().numpy()
    if terrain_runtime is not None:
        height = height[0]

    h_min = float(np.nanmin(height))
    h_max = float(np.nanmax(height))
    if abs(h_max - h_min) < 1.0e-6:
        pad = max(1.0e-3, abs(h_min) * 0.1)
        h_min -= pad
        h_max += pad
    extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
    return height, extent, (h_min, h_max)


def add_height_heatmap(
    ax,
    height: np.ndarray,
    extent: tuple[float, float, float, float],
    value_range: tuple[float, float],
    alpha: float = 0.72,
    contour: bool = True,
):
    """Draw a height heatmap on an existing axes and return the image artist."""
    image = ax.imshow(
        height,
        extent=extent,
        origin="lower",
        cmap="terrain",
        vmin=value_range[0],
        vmax=value_range[1],
        alpha=alpha,
        zorder=0,
    )
    if contour and abs(value_range[1] - value_range[0]) > 1.0e-5:
        ax.contour(
            height,
            extent=extent,
            origin="lower",
            colors="black",
            linewidths=0.25,
            alpha=0.22,
            levels=8,
            zorder=1,
        )
    return image


def save_height_map(
    terrain_cfg: TerrainCfg | None,
    xy_min: np.ndarray,
    xy_max: np.ndarray,
    path: str | Path,
    title: str = "Terrain Height Heatmap",
    resolution: int = 180,
    terrain_runtime: TerrainRuntime | None = None,
) -> None:
    """Save a standalone terrain height heatmap figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, extent, value_range = height_grid_for_extent(
        terrain_cfg,
        xy_min,
        xy_max,
        resolution=resolution,
        terrain_runtime=terrain_runtime,
    )
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = add_height_heatmap(ax, height, extent, value_range, alpha=0.9, contour=True)
    fig.colorbar(image, ax=ax, label="height (m)")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(path, dpi=160)
    plt.close(fig)
