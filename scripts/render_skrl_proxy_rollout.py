#!/usr/bin/env python
"""Render proxy rollouts, including replay of saved upstream evaluation panels."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_mpl_config_dir = Path(os.environ.get("MPLCONFIGDIR", f"/tmp/isaac_mappo_matplotlib_{os.getuid()}"))
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_mpl_config_dir)

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Polygon

from _common import ROOT, cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from play import _load_policy_players
from terrain_viz import add_height_heatmap, height_grid_for_extent, save_height_map


FLAT_FOOTPRINT_COLOR = "#16a34a"
ROUGH_FOOTPRINT_COLOR = "#dc2626"
UNKNOWN_FOOTPRINT_COLOR = "#6b7280"


@dataclass(frozen=True, slots=True)
class FlatnessFrame:
    """Flatness diagnostics aligned with one rendered rollout state."""

    centroid_xy: np.ndarray
    centroid_is_flat: bool
    centroid_height_range: float
    centroid_max_slope: float
    centroid_mean_slope: float
    oracle_feasible: bool | None
    oracle_height_range: float | None
    oracle_max_slope: float | None


def _resolve(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _manifest_artifact_path(path: Path) -> str:
    if path.is_absolute() and path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT))
    return str(path)


def _merge_render_artifacts_into_manifest(
    run_dir: Path,
    *,
    gif_path: Path,
    terrain_height_path: Path,
    metrics_path: Path,
) -> Path | None:
    """Register completed render artifacts without creating or replacing a manifest."""
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    if not (
        gif_path.is_file()
        and terrain_height_path.is_file()
        and metrics_path.is_file()
    ):
        return None

    entries = {
            "metrics_proxy_rollout_render": _manifest_artifact_path(metrics_path),
            "figures_terrain_height": _manifest_artifact_path(
                terrain_height_path
            ),
            "videos_proxy_rollout": _manifest_artifact_path(gif_path),
        }
    artifact_suffix = gif_path.stem.replace("-", "_")
    entries.update(
        {
            f"metrics_{artifact_suffix}": _manifest_artifact_path(metrics_path),
            f"figures_{artifact_suffix}_terrain": _manifest_artifact_path(
                terrain_height_path
            ),
            f"videos_{artifact_suffix}": _manifest_artifact_path(gif_path),
        }
    )
    return _merge_artifacts_into_manifest(run_dir, entries)


def _merge_artifacts_into_manifest(run_dir: Path, entries: dict) -> Path | None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object.")
    artifacts = manifest.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError(f"{manifest_path} field 'artifacts' must be a JSON object.")
    artifacts.update(entries)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
        dir=manifest_path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temp_path, manifest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return manifest_path


def _mean_pairwise_xy(positions: np.ndarray) -> float:
    xy = positions[:, :2]
    diff = xy[:, None, :] - xy[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    mask = ~np.eye(xy.shape[0], dtype=bool)
    return float(distances[mask].mean())


def _dmax_xy(positions: np.ndarray) -> float:
    xy = positions[:, :2]
    diff = xy[:, None, :] - xy[None, :, :]
    return float(np.linalg.norm(diff, axis=-1).max())


def _mean_oracle_xy(positions: np.ndarray, oracle: np.ndarray) -> float:
    return float(np.linalg.norm(positions[:, :2] - oracle[None, :2], axis=-1).mean())


def _field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _first_scalar(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.size == 0:
        return default
    return float(array.reshape(-1)[0])


def _first_bool(value: Any, default: bool | None = None) -> bool | None:
    scalar = _first_scalar(value)
    return default if scalar is None else bool(scalar)


def _first_xy(value: Any, default: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            return np.zeros(2, dtype=np.float32)
        return np.asarray(default, dtype=np.float32).reshape(-1)[:2].copy()
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim > 1:
        array = array[0]
    return np.asarray(array, dtype=np.float32).reshape(-1)[:2].copy()


def _flatness_frame(
    centroid_xy: Any,
    flatness: Any,
    oracle_search: Any,
    *,
    fallback: FlatnessFrame | None = None,
) -> FlatnessFrame:
    centroid_default = fallback.centroid_xy if fallback is not None else None
    is_flat_default = fallback.centroid_is_flat if fallback is not None else False
    height_default = fallback.centroid_height_range if fallback is not None else 0.0
    max_slope_default = fallback.centroid_max_slope if fallback is not None else 0.0
    mean_slope_default = fallback.centroid_mean_slope if fallback is not None else 0.0
    oracle_feasible_default = fallback.oracle_feasible if fallback is not None else None
    oracle_height_default = fallback.oracle_height_range if fallback is not None else None
    oracle_slope_default = fallback.oracle_max_slope if fallback is not None else None
    return FlatnessFrame(
        centroid_xy=_first_xy(centroid_xy, centroid_default),
        centroid_is_flat=bool(
            _first_bool(_field(flatness, "is_flat"), is_flat_default)
        ),
        centroid_height_range=float(
            _first_scalar(_field(flatness, "height_range"), height_default)
        ),
        centroid_max_slope=float(
            _first_scalar(_field(flatness, "max_slope"), max_slope_default)
        ),
        centroid_mean_slope=float(
            _first_scalar(_field(flatness, "mean_slope"), mean_slope_default)
        ),
        oracle_feasible=_first_bool(
            _field(oracle_search, "feasible"),
            oracle_feasible_default,
        ),
        oracle_height_range=_first_scalar(
            _field(oracle_search, "height_range"),
            oracle_height_default,
        ),
        oracle_max_slope=_first_scalar(
            _field(oracle_search, "max_slope"),
            oracle_slope_default,
        ),
    )


def _oracle_search_from_env(env: MultiRoverGatheringCore) -> dict[str, Any]:
    return {
        "feasible": getattr(env, "oracle_search_feasible", None),
        "height_range": getattr(env, "oracle_search_height_range", None),
        "max_slope": getattr(env, "oracle_search_max_slope", None),
    }


def _status_color(status: bool | None) -> str:
    if status is None:
        return UNKNOWN_FOOTPRINT_COLOR
    return FLAT_FOOTPRINT_COLOR if status else ROUGH_FOOTPRINT_COLOR


def _add_flatness_footprints(
    ax,
    frame: FlatnessFrame,
    oracle_xy: np.ndarray,
    radius: float,
) -> tuple[Circle, Circle]:
    """Draw team-centroid and oracle gathering footprints on an axes."""
    footprint_radius = max(0.0, float(radius))
    centroid_color = _status_color(frame.centroid_is_flat)
    oracle_color = _status_color(frame.oracle_feasible)
    centroid_status = "flat" if frame.centroid_is_flat else "rough"
    if frame.oracle_feasible is None:
        oracle_status = "unknown"
    else:
        oracle_status = "feasible" if frame.oracle_feasible else "infeasible"

    centroid_circle = Circle(
        frame.centroid_xy,
        footprint_radius,
        facecolor=to_rgba(centroid_color, 0.14),
        edgecolor=centroid_color,
        linewidth=2.0,
        label=f"team footprint: {centroid_status}",
        zorder=2,
    )
    oracle_circle = Circle(
        np.asarray(oracle_xy, dtype=np.float32)[:2],
        footprint_radius,
        facecolor="none",
        edgecolor=oracle_color,
        linewidth=2.0,
        linestyle="--",
        label=f"oracle footprint: {oracle_status}",
        zorder=2,
    )
    ax.add_patch(centroid_circle)
    ax.add_patch(oracle_circle)
    ax.scatter(
        frame.centroid_xy[0],
        frame.centroid_xy[1],
        marker="x",
        s=58,
        color=centroid_color,
        linewidths=1.8,
        zorder=4,
    )
    return centroid_circle, oracle_circle


def _axis_limits(
    history: list[np.ndarray],
    oracle_history: list[np.ndarray],
    footprint_radius: float = 0.0,
) -> tuple[float, float, float, float]:
    points = [item[:, :2] for item in history]
    points.extend(item[None, :2] for item in oracle_history)
    stacked = np.concatenate(points, axis=0)
    low = stacked.min(axis=0)
    high = stacked.max(axis=0)
    span = np.maximum(high - low, 1.0)
    radius_margin = max(0.0, float(footprint_radius)) * 1.1
    margin = np.maximum(np.maximum(span * 0.18, 0.75), radius_margin)
    return low[0] - margin[0], high[0] + margin[0], low[1] - margin[1], high[1] + margin[1]


def _draw_frame(
    history: list[np.ndarray],
    oracle_history: list[np.ndarray],
    flatness_history: list[FlatnessFrame],
    step_index: int,
    output: Path,
    terrain_height: np.ndarray,
    terrain_extent: tuple[float, float, float, float],
    terrain_range: tuple[float, float],
    footprint_radius: float,
    *,
    upstream: dict | None = None,
) -> None:
    if upstream is not None:
        _draw_upstream_frame(history, step_index, output, upstream)
        return
    xmin, xmax, ymin, ymax = _axis_limits(
        history,
        oracle_history,
        footprint_radius,
    )
    current = history[step_index]
    current_oracle = oracle_history[step_index]
    flatness = flatness_history[step_index]
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=120, constrained_layout=True)
    heatmap = add_height_heatmap(
        ax,
        terrain_height,
        terrain_extent,
        terrain_range,
        alpha=0.72,
        contour=True,
    )
    fig.colorbar(
        heatmap,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="height (m)",
    )
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f97316", "#0891b2")
    for agent_id in range(current.shape[0]):
        trail = np.asarray([item[agent_id, :2] for item in history[: step_index + 1]])
        color = colors[agent_id % len(colors)]
        ax.plot(trail[:, 0], trail[:, 1], color=color, linewidth=1.6, alpha=0.8)
        ax.scatter(
            current[agent_id, 0],
            current[agent_id, 1],
            color=color,
            s=56,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        ax.text(
            current[agent_id, 0],
            current[agent_id, 1],
            str(agent_id),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            zorder=4,
        )
    _add_flatness_footprints(
        ax,
        flatness,
        current_oracle,
        footprint_radius,
    )
    ax.scatter(
        current_oracle[0],
        current_oracle[1],
        marker="*",
        s=170,
        color="#facc15",
        edgecolors="#713f12",
        linewidths=0.8,
        label="oracle point",
        zorder=3,
    )
    oracle_status = (
        "unknown"
        if flatness.oracle_feasible is None
        else "feasible"
        if flatness.oracle_feasible
        else "infeasible"
    )
    centroid_status = "flat" if flatness.centroid_is_flat else "rough"
    ax.set_title(
        (
            f"step {step_index} | pairwise {_mean_pairwise_xy(current):.2f} | "
            f"oracle distance {_mean_oracle_xy(current, current_oracle):.2f}\n"
            f"centroid {centroid_status}: Δh={flatness.centroid_height_range:.3f}, "
            f"slope={flatness.centroid_max_slope:.3f} | oracle {oracle_status}"
        ),
        fontsize=9,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=7)
    fig.savefig(output)
    plt.close(fig)


def _save_gif(
    history: list[np.ndarray],
    oracle_history: list[np.ndarray],
    flatness_history: list[FlatnessFrame],
    output: Path,
    *,
    terrain_cfg,
    terrain_runtime,
    footprint_radius: float,
    capture_interval: int,
    max_frames: int,
    duration_s: float,
    upstream: dict | None = None,
) -> int:
    if capture_interval < 1 or max_frames < 2:
        raise ValueError("Capture interval must be positive and max_frames at least two")
    frame_indices = list(range(0, len(history), max(1, capture_interval)))
    if frame_indices[-1] != len(history) - 1:
        frame_indices.append(len(history) - 1)
    if len(frame_indices) > max_frames:
        sample = np.linspace(0, len(frame_indices) - 1, max_frames).round().astype(int)
        frame_indices = [frame_indices[index] for index in sample]

    xmin, xmax, ymin, ymax = _axis_limits(
        history,
        oracle_history,
        footprint_radius,
    )
    if upstream is None:
        terrain_height, terrain_extent, terrain_range = height_grid_for_extent(
            terrain_cfg,
            np.asarray([xmin, ymin], dtype=np.float32),
            np.asarray([xmax, ymax], dtype=np.float32),
            resolution=180,
            terrain_runtime=terrain_runtime,
        )
    else:
        terrain_height, terrain_extent, terrain_range = upstream["terrain_grids"][0]
    with tempfile.TemporaryDirectory(prefix="skrl_proxy_gif_") as tmp:
        tmp_dir = Path(tmp)
        frames = []
        for frame_id, step_index in enumerate(frame_indices):
            frame_path = tmp_dir / f"frame_{frame_id:04d}.png"
            _draw_frame(
                history,
                oracle_history,
                flatness_history,
                step_index,
                frame_path,
                terrain_height,
                terrain_extent,
                terrain_range,
                footprint_radius,
                **({"upstream": upstream} if upstream is not None else {}),
            )
            frames.append(imageio.imread(frame_path))
        # Pillow's GIF writer uses milliseconds. Keep real elapsed time when
        # subsampling a recorded evaluation, including a short final interval.
        duration_ms = duration_s * 1000
        if upstream is not None:
            duration_ms = [
                (b - a) * upstream["dt"] * 1000
                for a, b in zip(frame_indices, frame_indices[1:])
            ] + [1000]  # Pause on the terminal state; this adds no simulation time.
        imageio.mimsave(output, frames, duration=duration_ms, loop=0)
    return len(frame_indices)


def _draw_upstream_frame(history, step_index: int, output: Path, metadata: dict) -> None:
    """Use the recorded world poses; no oracle or inferred terrain-failure subtype."""
    poses = np.asarray(history)
    current = poses[step_index]
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")
    fig = plt.figure(figsize=(12, 7), dpi=100, constrained_layout=True)
    layout = fig.add_gridspec(2, 3, width_ratios=(1.35, 1, 1))
    axes = [fig.add_subplot(layout[:, 0])]
    axes.extend(fig.add_subplot(layout[row, col]) for row in range(2) for col in (1, 2))
    length, width = metadata["vehicle"]["length"], metadata["vehicle"]["width"]
    corners = np.array([[length, width], [length, -width], [-length, -width], [-length, width]]) / 2
    for view, (ax, terrain) in enumerate(zip(axes, metadata["terrain_grids"])):
        height, extent, value_range = terrain
        heatmap = add_height_heatmap(ax, height, extent, value_range, alpha=.8, contour=True)
        agents = range(4) if view == 0 else (view - 1,)
        for agent in agents:
            color = colors[agent]
            trail = poses[:step_index + 1, agent, :2]
            ax.plot(trail[:, 0], trail[:, 1], color=color, linewidth=1.8, zorder=4)
            ax.scatter(*poses[0, agent, :2], marker="x", s=32, color=color, zorder=5)
            xy, yaw = current[agent, :2], current[agent, 2]
            if view == 0:
                ax.scatter(*xy, s=65, color=color, edgecolors="white", zorder=6)
                ax.annotate(str(agent), xy, color="white", ha="center", va="center", fontsize=8, zorder=7)
            else:
                rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                ax.add_patch(Polygon(corners @ rotation.T + xy, closed=True,
                                    facecolor=to_rgba(color, .18), edgecolor=color, linewidth=2, zorder=5))
                ax.arrow(*xy, np.cos(yaw) * length / 2, np.sin(yaw) * length / 2,
                         width=.06, head_width=.4, color=color, length_includes_head=True, zorder=6)
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="world x (m)", ylabel="world y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.2)
        if view == 0:
            ax.set_title(f"Team overview | Dmax {_dmax_xy(current):.2f} m", fontsize=10)
            fig.colorbar(heatmap, ax=ax, orientation="horizontal", shrink=.85, pad=.08, label="height (m)")
        else:
            agent = view - 1
            traveled = np.linalg.norm(np.diff(poses[:step_index + 1, agent, :2], axis=0), axis=-1).sum()
            ax.set_title(f"Rover {agent} | path {traveled:.2f} m\nheight {value_range[0]:.2f} to {value_range[1]:.2f} m", fontsize=9)
    elapsed = step_index * metadata["dt"]
    final_time = (len(history) - 1) * metadata["dt"]
    status = metadata["terminal_reason"] if step_index == len(history) - 1 else "running"
    fig.suptitle(f"Final evaluation | {metadata['panel']} | env {metadata['env_id']}\n"
                 f"t = {elapsed:.1f} / {final_time:.1f} s | {status}", fontsize=13)
    fig.supxlabel("Proxy replay | world terrain for visualization | x: start | rectangle: body | arrow: heading", fontsize=9)
    fig.savefig(output)
    plt.close(fig)


def _upstream_terrain_grids(history, terrain_cfg, terrain_runtime, vehicle):
    extents = [_axis_limits(history, [], footprint_radius=0)]
    positions = np.asarray(history)
    margin = max(vehicle["length"], vehicle["width"]) * .9
    for agent in range(4):
        low = positions[:, agent, :2].min(axis=0)
        high = positions[:, agent, :2].max(axis=0)
        center = (low + high) / 2
        half = max(float((high - low).max()) / 2 + margin, 7.)
        extents.append((center[0]-half, center[0]+half, center[1]-half, center[1]+half))
    return [height_grid_for_extent(
        terrain_cfg, np.array([extent[0], extent[2]]), np.array([extent[1], extent[3]]),
        resolution=240 if i == 0 else 180, terrain_runtime=terrain_runtime,
    ) for i, extent in enumerate(extents)]


def render_saved_evaluation(run_dir: str | Path, phase: str = "final", *,
                            capture_interval: int = 2, max_frames: int = 80,
                            from_capture: bool = False) -> dict:
    """Replay via the existing evaluator, then render through the existing GIF path."""
    from diagnose_upstream_environment import event_data
    from upstream_decision_evaluation import replay_saved_evaluation

    run_dir = _resolve(run_dir)
    stem = f"evaluation_{phase}_render"
    capture_path = run_dir / "metrics" / f"{stem}_capture.json"
    arrays_path = run_dir / "state" / f"{stem}.npz"
    metrics_path = run_dir / "metrics" / f"{stem}.json"
    video_dir = run_dir / "videos" / f"evaluation_{phase}"
    video_dir.mkdir(parents=True, exist_ok=True)
    if not from_capture:
        if capture_path.exists() or arrays_path.exists():
            raise FileExistsError(f"Capture already exists; use --from-capture: {capture_path}")
        replay = replay_saved_evaluation(run_dir, phase,
            progress=lambda name, elapsed: print(f"Replay {name}: {int(elapsed.sum())} base steps", flush=True))
        capture = {key: replay[key] for key in ("evaluation", "traces", "validation", "provenance")}
        capture.update(episodes={}, command=sys.argv, new_optimizer_steps=0)
        arrays = {}
        for name, panel_data in replay["panels"].items():
            for env_id, history in enumerate(panel_data["histories"]):
                key = f"{name}_env{env_id}"
                grids = _upstream_terrain_grids(history, panel_data["terrain_cfg"],
                                               panel_data["terrain_runtimes"][env_id], panel_data["vehicle"])
                arrays[f"{key}_poses"] = np.asarray(history)
                for i, (height, extent, value_range) in enumerate(grids):
                    arrays[f"{key}_height{i}"] = height
                    arrays[f"{key}_extent{i}"] = np.asarray(extent)
                    arrays[f"{key}_range{i}"] = np.asarray(value_range)
                episode = panel_data["episodes"][env_id]
                capture["episodes"][key] = dict(panel=name, env_id=env_id, dt=panel_data["dt"],
                    vehicle=panel_data["vehicle"], terminal_reason=episode["terminal_reason"],
                    steps=episode["steps"], scene_manifest=panel_data["scene_manifests"][env_id])
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arrays_path, **arrays)
        capture["arrays_sha256"] = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
        capture_path.write_text(json.dumps(event_data(capture), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    else:
        capture = json.loads(capture_path.read_text())
        if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != capture["arrays_sha256"]:
            raise ValueError("Saved evaluation capture checksum mismatch")
        for path, checksum in capture["provenance"]["sources"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Evaluation source changed since capture: {path}")
    if not capture["validation"]["matched"]:
        raise ValueError(f"Evaluation replay does not match original; retained evidence: {capture_path}")
    rendered = {}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for key, episode in capture["episodes"].items():
            gif_path = video_dir / f"{key}.gif"
            poster_path = video_dir / f"{key}.png"
            if gif_path.exists() or poster_path.exists():
                raise FileExistsError(gif_path if gif_path.exists() else poster_path)
            history = arrays[f"{key}_poses"]
            metadata = {**episode, "terrain_grids": [
                (arrays[f"{key}_height{i}"], tuple(arrays[f"{key}_extent{i}"]), tuple(arrays[f"{key}_range{i}"]))
                for i in range(5)
            ]}
            count = _save_gif(history, [], [], gif_path, terrain_cfg=None, terrain_runtime=None,
                footprint_radius=0., capture_interval=capture_interval, max_frames=max_frames,
                duration_s=episode["dt"], upstream=metadata)
            _draw_upstream_frame(history, len(history)-1, poster_path, metadata)
            rendered[key] = dict(gif=str(gif_path), poster=str(poster_path), frames=count,
                                 simulation_seconds=episode["steps"]*episode["dt"],
                                 terminal_reason=episode["terminal_reason"])
            print(f"Rendered {key}: {count} frames", flush=True)
    links = []
    for key, item in rendered.items():
        links.append(f'<article><h2>{html.escape(key)}</h2><p>{item["simulation_seconds"]:.1f} s · '
                     f'{html.escape(item["terminal_reason"])}</p><a href="{key}.gif">'
                     f'<img loading="lazy" src="{key}.png" alt="{html.escape(key)}"></a>'
                     f'<p><a href="{key}.gif">播放 GIF / Open animation</a></p></article>')
    index = video_dir / "index.html"
    index.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>最终评测动画</title><style>body{font-family:sans-serif;margin:2rem;background:#f5f6f8}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:1rem}'
        'article{padding:1rem;background:white;border-radius:8px}img{width:100%}h2{font-size:1rem}</style>'
        '<h1>最终评测 · 16 局动画</h1><p>点击预览图播放。原 checkpoint / seed 重放，结果已与原评测核对。'
        '全局图与四车局部图使用实际世界坐标；末帧停留 1 秒。Proxy，不是 Isaac / PhysX。</p>'
        '<main>' + ''.join(links) + '</main></html>')
    result = dict(status="ok", command=sys.argv, capture=str(capture_path), arrays=str(arrays_path),
                  index=str(index), videos=rendered, validation=capture["validation"],
                  replay_base_steps=capture["evaluation"]["actual_base_steps"],
                  new_base_steps_this_command=0 if from_capture else capture["evaluation"]["actual_base_steps"],
                  optimizer_updates=0)
    metrics_path.write_text(json.dumps(event_data(result), indent=2, allow_nan=False) + "\n")
    _merge_artifacts_into_manifest(run_dir, {
        f"metrics_{stem}": _manifest_artifact_path(metrics_path),
        f"videos_evaluation_{phase}": _manifest_artifact_path(index),
        f"state_{stem}": _manifest_artifact_path(arrays_path),
        f"metrics_{stem}_capture": _manifest_artifact_path(capture_path),
    })
    return result


def render_rollout(
    config: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    steps: int = 120,
    seed: int | None = None,
    output: str | Path | None = None,
    metrics_output: str | Path | None = None,
    terrain_output: str | Path | None = None,
    run_dir: str | Path | None = None,
    capture_interval: int = 2,
    max_frames: int = 80,
) -> dict[str, Any]:
    run_dir_path: Path | None = None
    if run_dir is not None:
        run_dir_path = _resolve(run_dir)
        output = output or run_dir_path / "videos" / "proxy_eval_rollout.gif"
        metrics_output = metrics_output or run_dir_path / "metrics" / "proxy_rollout_render.json"
    if output is None:
        output = "outputs/videos/proxy_eval_rollout.gif"
    if metrics_output is None:
        metrics_output = Path(output).with_suffix(".json")

    raw_cfg = load_yaml(config)
    cfg = cfg_from_experiment(config)
    cfg.simulation.num_envs = 1
    cfg.simulation.device = device
    if seed is not None:
        cfg.seed = seed

    map_location = torch.device(cfg.simulation.device)
    if map_location.type == "cuda" and not torch.cuda.is_available():
        map_location = torch.device("cpu")
    checkpoint_data = torch.load(checkpoint, map_location=map_location)
    metadata = checkpoint_data.get("metadata", {}) if isinstance(checkpoint_data, dict) else {}
    if cfg.planner.subgoal_filter.mode in {
        "terrain_safe_candidate_curriculum",
        "terrain_safe_candidate_constrained_curriculum",
        "terrain_safe_candidate_soft_progress_curriculum",
        "terrain_safe_candidate_mutual_progress_curriculum",
        "terrain_safe_candidate_hold_progress_curriculum",
    }:
        cfg.planner.subgoal_filter.progress_timestep_override = int(metadata.get("timesteps", 0))
        cfg.planner.subgoal_filter.deterministic_eval = True

    env = MultiRoverGatheringCore(cfg)
    rollout_terrain_runtime = env.terrain_runtime.clone()
    act, backend = _load_policy_players(checkpoint_data, cfg, env.device, raw_cfg=raw_cfg)
    actor_obs, _ = env.get_observations()

    initial_flatness = env.evaluate_current_gather_point_flatness(env.metrics)
    initial_flatness_frame = _flatness_frame(
        env.metrics.centroid,
        initial_flatness,
        _oracle_search_from_env(env),
    )
    history: list[np.ndarray] = [
        env.positions[0].detach().cpu().numpy().copy()
    ]
    oracle_history: list[np.ndarray] = [
        env.oracle_point[0].detach().cpu().numpy().copy()
    ]
    flatness_history: list[FlatnessFrame] = [initial_flatness_frame]
    reward_values: list[float] = []
    done_reason = "not_done"
    steps_executed = 0
    final_flatness_frame = initial_flatness_frame
    final_oracle = oracle_history[0]

    for step_id in range(max(1, steps)):
        with torch.no_grad():
            action = act(actor_obs)
        out = env.step(action)
        reward_values.append(float(out.rewards[0].mean().detach().cpu()))
        steps_executed = step_id + 1
        info = out.info
        metrics = info.get("metrics")
        step_flatness = info.get("gather_point_flatness")
        step_oracle_search = info.get("oracle_search", _oracle_search_from_env(env))
        step_oracle_value = info.get("oracle_point", env.oracle_point)
        step_oracle = _first_xy(step_oracle_value)
        centroid_xy = (
            metrics.centroid
            if metrics is not None
            else env.positions[..., :2].mean(dim=1)
        )
        final_flatness_frame = _flatness_frame(
            centroid_xy,
            step_flatness,
            step_oracle_search,
            fallback=final_flatness_frame,
        )
        final_oracle = step_oracle
        done = out.info["done"]
        if bool(done.done[0].detach().cpu()):
            if bool(done.success[0].detach().cpu()):
                done_reason = "success"
            elif bool(done.timeout[0].detach().cpu()):
                done_reason = "timeout"
            elif bool(done.collision[0].detach().cpu()):
                done_reason = "collision"
            elif bool(done.safety[0].detach().cpu()):
                done_reason = "safety"
            else:
                done_reason = "other"
            break
        actor_obs = out.actor_obs
        history.append(env.positions[0].detach().cpu().numpy().copy())
        oracle_history.append(step_oracle.copy())
        flatness_history.append(final_flatness_frame)

    gif_path = _resolve(output)
    footprint_radius = float(getattr(cfg.gather_point, "flatness_radius", 0.0))
    xmin, xmax, ymin, ymax = _axis_limits(
        history,
        oracle_history,
        footprint_radius,
    )
    terrain_height_path = (
        _resolve(terrain_output)
        if terrain_output is not None
        else run_dir_path / "figures" / "terrain_height_map.png"
        if run_dir_path is not None
        else gif_path.parent / "terrain_height_map.png"
    )
    save_height_map(
        cfg.terrain,
        np.asarray([xmin, ymin], dtype=np.float32),
        np.asarray([xmax, ymax], dtype=np.float32),
        terrain_height_path,
        title="Proxy Rollout Terrain Height",
        terrain_runtime=rollout_terrain_runtime,
    )
    frame_count = _save_gif(
        history,
        oracle_history,
        flatness_history,
        gif_path,
        terrain_cfg=cfg.terrain,
        terrain_runtime=rollout_terrain_runtime,
        footprint_radius=footprint_radius,
        capture_interval=capture_interval,
        max_frames=max_frames,
        duration_s=0.12,
    )
    first_positions = history[0]
    final_positions = history[-1]
    first_oracle = oracle_history[0]
    require_flat_for_success = bool(
        getattr(cfg.gather_point, "require_flat_for_success", False)
    )
    final_flatness_ok = (
        final_flatness_frame.centroid_is_flat
        if require_flat_for_success
        else True
    )
    flatness_config = {
        "radius": footprint_radius,
        "rings": int(getattr(cfg.gather_point, "flatness_rings", 0)),
        "samples_per_ring": int(
            getattr(cfg.gather_point, "flatness_samples_per_ring", 0)
        ),
        "max_height_range": float(
            getattr(cfg.gather_point, "max_height_range", 0.0)
        ),
        "max_slope": float(getattr(cfg.gather_point, "max_slope", 0.0)),
        "require_flat_for_success": require_flat_for_success,
    }
    final_flatness_result = {
        "is_flat": final_flatness_frame.centroid_is_flat,
        "effective_gate_ok": final_flatness_ok,
        "centroid_xy": final_flatness_frame.centroid_xy.tolist(),
        "height_range": final_flatness_frame.centroid_height_range,
        "max_slope": final_flatness_frame.centroid_max_slope,
        "mean_slope": final_flatness_frame.centroid_mean_slope,
    }
    oracle_search_result = {
        "method": str(getattr(cfg.gather_point, "search_method", "unknown")),
        "point": final_oracle.tolist(),
        "feasible": final_flatness_frame.oracle_feasible,
        "height_range": final_flatness_frame.oracle_height_range,
        "max_slope": final_flatness_frame.oracle_max_slope,
    }
    result = {
        "status": "ok",
        "backend": backend,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "device": str(env.device),
        "steps_requested": steps,
        "steps_executed": steps_executed,
        "done_reason": done_reason,
        "mean_reward": float(np.mean(reward_values)) if reward_values else None,
        "initial_mean_pairwise_distance": _mean_pairwise_xy(first_positions),
        "final_mean_pairwise_distance": _mean_pairwise_xy(final_positions),
        "initial_mean_oracle_distance": _mean_oracle_xy(first_positions, first_oracle),
        "final_mean_oracle_distance": _mean_oracle_xy(final_positions, final_oracle),
        "initial_dmax": _dmax_xy(first_positions),
        "final_dmax": _dmax_xy(final_positions),
        "flatness_footprint": flatness_config,
        "final_flatness_ok": final_flatness_ok,
        "final_gather_point_is_flat": final_flatness_frame.centroid_is_flat,
        "final_gather_point_height_range": final_flatness_frame.centroid_height_range,
        "final_gather_point_max_slope": final_flatness_frame.centroid_max_slope,
        "final_flatness": final_flatness_result,
        "oracle_search_feasible": final_flatness_frame.oracle_feasible,
        "oracle_gather_point_height_range": final_flatness_frame.oracle_height_range,
        "oracle_gather_point_max_slope": final_flatness_frame.oracle_max_slope,
        "oracle_search": oracle_search_result,
        "frame_count": frame_count,
        "gif_path": str(gif_path),
        "terrain_height_map": str(terrain_height_path),
    }
    metrics_path = _resolve(metrics_output)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest_path = (
        _merge_render_artifacts_into_manifest(
            run_dir_path,
            gif_path=gif_path,
            terrain_height_path=terrain_height_path,
            metrics_path=metrics_path,
        )
        if run_dir_path is not None
        else None
    )
    result["artifact"] = str(metrics_path)
    if manifest_path is not None:
        result["run_manifest"] = str(manifest_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--upstream-evaluation", choices=("final",), help="Replay saved upstream evaluation panels")
    parser.add_argument("--from-capture", action="store_true", help="Render saved replay poses without new environment steps")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--terrain-output", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--capture-interval", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=80)
    args = parser.parse_args()
    if args.upstream_evaluation:
        if not args.run_dir:
            parser.error("--upstream-evaluation requires --run-dir")
        result = render_saved_evaluation(args.run_dir, args.upstream_evaluation,
            capture_interval=args.capture_interval, max_frames=args.max_frames, from_capture=args.from_capture)
        print(json.dumps(result, indent=2))
        return
    if not args.config or not args.checkpoint or args.from_capture:
        parser.error("Legacy rollout requires --config and --checkpoint; --from-capture requires --upstream-evaluation")
    result = render_rollout(
        args.config,
        args.checkpoint,
        device=args.device,
        steps=args.steps,
        seed=args.seed,
        output=args.output,
        metrics_output=args.metrics_output,
        terrain_output=args.terrain_output,
        run_dir=args.run_dir,
        capture_interval=args.capture_interval,
        max_frames=args.max_frames,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
