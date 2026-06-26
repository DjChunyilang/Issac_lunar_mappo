#!/usr/bin/env python
"""Render a deterministic proxy rollout GIF from a SKRL MAPPO checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
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

from _common import ROOT, cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from play import _load_policy_players
from terrain_viz import add_height_heatmap, height_grid_for_extent, save_height_map


def _resolve(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


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


def _axis_limits(history: list[np.ndarray], oracle_history: list[np.ndarray]) -> tuple[float, float, float, float]:
    points = [item[:, :2] for item in history]
    points.extend(item[None, :2] for item in oracle_history)
    stacked = np.concatenate(points, axis=0)
    low = stacked.min(axis=0)
    high = stacked.max(axis=0)
    span = np.maximum(high - low, 1.0)
    margin = np.maximum(span * 0.18, 0.75)
    return low[0] - margin[0], high[0] + margin[0], low[1] - margin[1], high[1] + margin[1]


def _draw_frame(
    history: list[np.ndarray],
    oracle_history: list[np.ndarray],
    step_index: int,
    output: Path,
    terrain_height: np.ndarray,
    terrain_extent: tuple[float, float, float, float],
    terrain_range: tuple[float, float],
) -> None:
    xmin, xmax, ymin, ymax = _axis_limits(history, oracle_history)
    current = history[step_index]
    current_oracle = oracle_history[step_index]
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
    ax.scatter(
        current_oracle[0],
        current_oracle[1],
        marker="*",
        s=170,
        color="#facc15",
        edgecolors="#713f12",
        linewidths=0.8,
        label="oracle point",
        zorder=2,
    )
    ax.set_title(
        f"step {step_index} | pairwise {_mean_pairwise_xy(current):.2f} | oracle {_mean_oracle_xy(current, current_oracle):.2f}",
        fontsize=10,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.savefig(output)
    plt.close(fig)


def _save_gif(
    history: list[np.ndarray],
    oracle_history: list[np.ndarray],
    output: Path,
    *,
    terrain_cfg,
    terrain_runtime,
    capture_interval: int,
    max_frames: int,
    duration_s: float,
) -> int:
    frame_indices = list(range(0, len(history), max(1, capture_interval)))
    if frame_indices[-1] != len(history) - 1:
        frame_indices.append(len(history) - 1)
    if len(frame_indices) > max_frames:
        sample = np.linspace(0, len(frame_indices) - 1, max_frames).round().astype(int)
        frame_indices = [frame_indices[index] for index in sample]

    xmin, xmax, ymin, ymax = _axis_limits(history, oracle_history)
    terrain_height, terrain_extent, terrain_range = height_grid_for_extent(
        terrain_cfg,
        np.asarray([xmin, ymin], dtype=np.float32),
        np.asarray([xmax, ymax], dtype=np.float32),
        resolution=180,
        terrain_runtime=terrain_runtime,
    )
    with tempfile.TemporaryDirectory(prefix="skrl_proxy_gif_") as tmp:
        tmp_dir = Path(tmp)
        frames = []
        for frame_id, step_index in enumerate(frame_indices):
            frame_path = tmp_dir / f"frame_{frame_id:04d}.png"
            _draw_frame(
                history,
                oracle_history,
                step_index,
                frame_path,
                terrain_height,
                terrain_extent,
                terrain_range,
            )
            frames.append(imageio.imread(frame_path))
        imageio.mimsave(output, frames, duration=duration_s)
    return len(frame_indices)


def render_rollout(
    config: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    steps: int = 120,
    seed: int | None = None,
    output: str | Path | None = None,
    metrics_output: str | Path | None = None,
    run_dir: str | Path | None = None,
    capture_interval: int = 2,
    max_frames: int = 80,
) -> dict[str, Any]:
    if run_dir is not None:
        run_dir_path = _resolve(run_dir)
        output = output or run_dir_path / "videos" / "proxy_eval_rollout.gif"
        metrics_output = metrics_output or run_dir_path / "metrics" / "proxy_rollout_render.json"
    if output is None:
        output = "outputs/videos/proxy_eval_rollout.gif"
    if metrics_output is None:
        metrics_output = Path(output).with_suffix(".json")

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
    }:
        cfg.planner.subgoal_filter.progress_timestep_override = int(metadata.get("timesteps", 0))
        cfg.planner.subgoal_filter.deterministic_eval = True

    env = MultiRoverGatheringCore(cfg)
    rollout_terrain_runtime = env.terrain_runtime.clone()
    act, backend = _load_policy_players(checkpoint_data, cfg, env.device)
    actor_obs, _ = env.get_observations()

    history: list[np.ndarray] = []
    oracle_history: list[np.ndarray] = []
    reward_values: list[float] = []
    done_reason = "not_done"
    steps_executed = 0

    for step_id in range(max(1, steps)):
        history.append(env.positions[0].detach().cpu().numpy().copy())
        oracle_history.append(env.oracle_point[0].detach().cpu().numpy().copy())
        with torch.no_grad():
            action = act(actor_obs)
        out = env.step(action)
        reward_values.append(float(out.rewards[0].mean().detach().cpu()))
        steps_executed = step_id + 1
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

    gif_path = _resolve(output)
    xmin, xmax, ymin, ymax = _axis_limits(history, oracle_history)
    terrain_height_path = (
        _resolve(run_dir) / "figures" / "terrain_height_map.png"
        if run_dir is not None
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
        gif_path,
        terrain_cfg=cfg.terrain,
        terrain_runtime=rollout_terrain_runtime,
        capture_interval=capture_interval,
        max_frames=max_frames,
        duration_s=0.12,
    )
    first_positions = history[0]
    final_positions = history[-1]
    first_oracle = oracle_history[0]
    final_oracle = oracle_history[-1]
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
        "frame_count": frame_count,
        "gif_path": str(gif_path),
        "terrain_height_map": str(terrain_height_path),
    }
    metrics_path = _resolve(metrics_output)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["artifact"] = str(metrics_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--capture-interval", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=80)
    args = parser.parse_args()
    result = render_rollout(
        args.config,
        args.checkpoint,
        device=args.device,
        steps=args.steps,
        seed=args.seed,
        output=args.output,
        metrics_output=args.metrics_output,
        run_dir=args.run_dir,
        capture_interval=args.capture_interval,
        max_frames=args.max_frames,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
