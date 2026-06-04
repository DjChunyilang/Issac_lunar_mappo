#!/usr/bin/env python
"""Generate first-stage proxy-environment validation artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for cache_env, cache_dir in (
    ("MPLCONFIGDIR", "/tmp/isaac_mappo_matplotlib"),
    ("XDG_CACHE_HOME", "/tmp/isaac_mappo_cache"),
):
    os.environ.setdefault(cache_env, cache_dir)
    Path(os.environ[cache_env]).mkdir(parents=True, exist_ok=True)

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import ROOT, cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from terrain_viz import add_height_heatmap, height_grid_for_extent, save_height_map


def _resolve_output_root(path: str | Path) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    return output


def _scripted_gather_action(env: MultiRoverGatheringCore) -> torch.Tensor:
    positions_xy = env.positions[..., :2]
    centroid_xy = positions_xy.mean(dim=1, keepdim=True)
    world_delta = centroid_xy - positions_xy
    cos_yaw = torch.cos(env.yaws)
    sin_yaw = torch.sin(env.yaws)
    local_x = cos_yaw * world_delta[..., 0] + sin_yaw * world_delta[..., 1]
    local_y = -sin_yaw * world_delta[..., 0] + cos_yaw * world_delta[..., 1]
    rho = torch.linalg.norm(torch.stack((local_x, local_y), dim=-1), dim=-1)
    rho = torch.clamp(rho, 0.0, env.cfg.planner.rho_max)
    beta = torch.atan2(local_y, local_x)
    beta = torch.clamp(beta, -env.cfg.planner.beta_max, env.cfg.planner.beta_max)
    normalized_rho = 2.0 * rho / env.cfg.planner.rho_max - 1.0
    normalized_beta = beta / env.cfg.planner.beta_max
    return torch.stack((normalized_rho, normalized_beta), dim=-1)


def _tensor_to_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().flatten()]


def _save_rollout_curves(history: dict[str, list[float]], path: Path) -> None:
    steps = np.arange(1, len(history["dmax"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(steps, history["dmax"], marker="o", markersize=2)
    axes[0, 0].set_title("Team max distance")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("m")

    axes[0, 1].plot(steps, history["dispersion"], marker="o", markersize=2, color="tab:green")
    axes[0, 1].set_title("Team dispersion")
    axes[0, 1].set_xlabel("step")

    axes[1, 0].plot(steps, history["mean_reward"], color="tab:purple")
    axes[1, 0].set_title("Mean reward")
    axes[1, 0].set_xlabel("step")

    axes[1, 1].plot(steps, history["mean_speed"], label="mean speed")
    axes[1, 1].plot(steps, history["mean_linear_cmd"], label="linear cmd")
    axes[1, 1].plot(steps, history["mean_abs_angular_cmd"], label="|angular cmd|")
    axes[1, 1].set_title("Proxy control")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].legend()

    fig.suptitle("First-stage proxy rover rollout")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_observation_heatmap(actor_obs: torch.Tensor, cfg, path: Path) -> None:
    obs = actor_obs.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 3.5), constrained_layout=True)
    image = ax.imshow(obs, aspect="auto", cmap="coolwarm")
    boundaries = [
        cfg.observation.ego_dim,
        cfg.observation.ego_dim + cfg.observation.max_neighbors * cfg.observation.neighbor_dim,
        (
            cfg.observation.ego_dim
            + cfg.observation.max_neighbors * cfg.observation.neighbor_dim
            + cfg.observation.terrain_dim
        ),
    ]
    labels = ["ego", "neighbors", "terrain", "aggregation"]
    starts = [0, *boundaries]
    ends = [*boundaries, cfg.actor_obs_dim]
    for boundary in boundaries:
        ax.axvline(boundary - 0.5, color="black", linewidth=1.0)
    for label, start, end in zip(labels, starts, ends, strict=True):
        ax.text((start + end - 1) / 2.0, -0.8, label, ha="center", va="bottom", fontsize=9)
    ax.set_title("Actor observation layout for four rovers")
    ax.set_xlabel("observation feature index")
    ax.set_ylabel("rover index")
    ax.set_yticks(np.arange(obs.shape[0]))
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_trajectory_control_plot(
    positions: torch.Tensor,
    trajectory_points: torch.Tensor,
    linear: torch.Tensor,
    angular: torch.Tensor,
    path: Path,
    terrain_cfg=None,
) -> None:
    pos = positions.detach().cpu().numpy()
    traj = trajectory_points.detach().cpu().numpy()
    linear_np = linear.detach().cpu().numpy()
    angular_np = angular.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    all_xy = np.concatenate((pos[:, :2], traj.reshape(-1, traj.shape[-1])[:, :2]), axis=0)
    xy_min = all_xy.min(axis=0) - 0.5
    xy_max = all_xy.max(axis=0) + 0.5
    height_grid, height_extent, height_range = height_grid_for_extent(terrain_cfg, xy_min, xy_max)
    heatmap = add_height_heatmap(axes[0], height_grid, height_extent, height_range, alpha=0.62, contour=True)
    fig.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.04, label="height (m)")
    for agent_id in range(traj.shape[0]):
        axes[0].plot(traj[agent_id, :, 0], traj[agent_id, :, 1], marker="o", color=colors[agent_id])
        axes[0].scatter(pos[agent_id, 0], pos[agent_id, 1], color=colors[agent_id], s=50)
        axes[0].text(pos[agent_id, 0], pos[agent_id, 1], f"r{agent_id}", fontsize=9)
    axes[0].set_title("Generated line trajectories on height heatmap")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True, alpha=0.3)

    x = np.arange(traj.shape[0])
    axes[1].bar(x - 0.17, linear_np, width=0.34, label="linear")
    axes[1].bar(x + 0.17, angular_np, width=0.34, label="angular")
    axes[1].set_xticks(x)
    axes[1].set_xlabel("rover")
    axes[1].set_title("Simplified velocity commands")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.savefig(path, dpi=160)
    plt.close(fig)


def _figure_to_frame(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def _save_rollout_gif(position_history: list[np.ndarray], path: Path, terrain_cfg=None) -> None:
    stacked = np.concatenate(position_history, axis=0)
    xy_min = stacked[:, :2].min(axis=0) - 0.5
    xy_max = stacked[:, :2].max(axis=0) + 0.5
    height_grid, height_extent, height_range = height_grid_for_extent(terrain_cfg, xy_min, xy_max)
    frames = []
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for step_id, positions in enumerate(position_history):
        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        heatmap = add_height_heatmap(ax, height_grid, height_extent, height_range, alpha=0.70, contour=True)
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04, label="height (m)")
        for agent_id, color in enumerate(colors):
            trail = np.array([frame[agent_id, :2] for frame in position_history[: step_id + 1]])
            ax.plot(trail[:, 0], trail[:, 1], color=color, alpha=0.55)
            ax.scatter(positions[agent_id, 0], positions[agent_id, 1], color=color, s=55)
            ax.text(positions[agent_id, 0], positions[agent_id, 1], f"r{agent_id}", fontsize=9)
        centroid = positions[:, :2].mean(axis=0)
        ax.scatter(centroid[0], centroid[1], color="black", marker="x", s=70)
        ax.set_title(f"Proxy four-rover rollout step {step_id} with height heatmap")
        ax.set_xlim(xy_min[0], xy_max[0])
        ax.set_ylim(xy_min[1], xy_max[1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        frames.append(_figure_to_frame(fig))
        plt.close(fig)
    imageio.mimsave(path, frames, duration=0.18)


def run_validation(
    config: str,
    device: str,
    steps: int,
    output_root: str | Path = "outputs",
) -> dict:
    cfg = cfg_from_experiment(config)
    cfg.simulation.device = device
    env = MultiRoverGatheringCore(cfg)
    actor_obs, critic_state = env.get_observations()

    root = _resolve_output_root(output_root)
    log_dir = root / "logs" / "first_stage_validation"
    figure_dir = root / "figures" / "first_stage_validation"
    video_dir = root / "videos" / "first_stage_validation"
    for directory in (log_dir, figure_dir, video_dir):
        directory.mkdir(parents=True, exist_ok=True)

    history = {
        "dmax": [],
        "dispersion": [],
        "mean_speed": [],
        "mean_reward": [],
        "mean_linear_cmd": [],
        "mean_abs_angular_cmd": [],
    }
    position_history = [env.positions[0].detach().cpu().numpy().copy()]
    first_positions = env.positions[0].detach().clone()
    first_trajectory = None
    first_control = None
    done_step_env0 = None

    for step_id in range(steps):
        action = _scripted_gather_action(env)
        output = env.step(action)
        metrics = output.info["metrics"]
        control = output.info["control"]
        if first_trajectory is None:
            first_trajectory = output.info["trajectory"].points[0].detach().clone()
            first_control = control.packed[0].detach().clone()

        history["dmax"].append(float(metrics.dmax[0].detach().cpu()))
        history["dispersion"].append(float(metrics.dispersion[0].detach().cpu()))
        history["mean_speed"].append(float(metrics.mean_speed[0].detach().cpu()))
        history["mean_reward"].append(float(output.rewards[0].mean().detach().cpu()))
        history["mean_linear_cmd"].append(float(control.linear[0].mean().detach().cpu()))
        history["mean_abs_angular_cmd"].append(float(control.angular[0].abs().mean().detach().cpu()))

        done_env0 = bool((output.terminated[0] | output.truncated[0]).detach().cpu())
        if done_env0:
            done_step_env0 = step_id + 1
            break
        position_history.append(env.positions[0].detach().cpu().numpy().copy())

    rollout_path = figure_dir / "proxy_rollout_curves.png"
    heatmap_path = figure_dir / "observation_space_heatmap.png"
    trajectory_path = figure_dir / "trajectory_control_validation.png"
    terrain_height_path = figure_dir / "terrain_height_map.png"
    gif_path = video_dir / "proxy_rollout.gif"
    metrics_path = log_dir / "validation_metrics.json"

    _save_rollout_curves(history, rollout_path)
    _save_observation_heatmap(actor_obs[0], cfg, heatmap_path)
    assert first_trajectory is not None
    assert first_control is not None
    _save_trajectory_control_plot(
        first_positions,
        first_trajectory,
        first_control[:, 0],
        first_control[:, 1],
        trajectory_path,
        cfg.terrain,
    )
    stacked_positions = np.concatenate(position_history, axis=0)
    save_height_map(
        cfg.terrain,
        stacked_positions[:, :2].min(axis=0) - 0.5,
        stacked_positions[:, :2].max(axis=0) + 0.5,
        terrain_height_path,
    )
    _save_rollout_gif(position_history, gif_path, cfg.terrain)

    summary = {
        "status": "ok",
        "config": config,
        "device": str(env.device),
        "steps_requested": steps,
        "steps_recorded": len(history["dmax"]),
        "num_envs": cfg.simulation.num_envs,
        "n_agents": cfg.task.n_agents,
        "actor_obs_shape": list(actor_obs.shape),
        "critic_state_shape": list(critic_state.shape),
        "initial_positions_env0": position_history[0].tolist(),
        "final_positions_env0": position_history[-1].tolist(),
        "initial_dmax": history["dmax"][0],
        "final_dmax": history["dmax"][-1],
        "initial_dispersion": history["dispersion"][0],
        "final_dispersion": history["dispersion"][-1],
        "mean_speed": float(np.mean(history["mean_speed"])),
        "mean_reward": float(np.mean(history["mean_reward"])),
        "done_step_env0": done_step_env0,
        "artifacts": {
            "metrics": str(metrics_path),
            "rollout_curves": str(rollout_path),
            "observation_heatmap": str(heatmap_path),
            "trajectory_control": str(trajectory_path),
            "terrain_height_map": str(terrain_height_path),
            "rollout_gif": str(gif_path),
        },
        "history": history,
        "first_control_linear": _tensor_to_list(first_control[:, 0]),
        "first_control_angular": _tensor_to_list(first_control[:, 1]),
    }
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp_001_minimal.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    summary = run_validation(args.config, args.device, args.steps, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
