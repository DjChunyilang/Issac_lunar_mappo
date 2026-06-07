#!/usr/bin/env python
"""Evaluate a proxy gathering policy without rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import ROOT, cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from play import _load_policy_players


def _resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def evaluate_checkpoint(
    config: str | Path,
    checkpoint: str | Path,
    device: str | None = None,
    num_envs: int = 256,
    steps: int = 100,
    seed: int | None = None,
    output: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict:
    if run_dir is not None and output is None:
        output = Path(run_dir) / "metrics" / "final_eval_proxy.json"

    cfg = cfg_from_experiment(config)
    cfg.simulation.num_envs = num_envs
    if device is not None:
        cfg.simulation.device = device
    if seed is not None:
        cfg.seed = seed

    env = MultiRoverGatheringCore(cfg)
    checkpoint_data = torch.load(checkpoint, map_location=env.device)
    act, backend = _load_policy_players(checkpoint_data, cfg, env.device)
    actor_obs, _ = env.get_observations()

    initial_dmax = env.metrics.dmax.detach().clone()
    initial_dispersion = env.metrics.dispersion.detach().clone()
    final_dmax = initial_dmax.clone()
    final_dispersion = initial_dispersion.clone()
    final_mean_speed = env.metrics.mean_speed.detach().clone()
    final_success_hold_count = env.success_hold_count.detach().clone()
    max_success_hold_count = env.success_hold_count.detach().clone()
    final_terrain_speed_scale = torch.ones(env.num_envs, device=env.device)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    success_seen = torch.zeros_like(active)
    collision_seen = torch.zeros_like(active)
    timeout_seen = torch.zeros_like(active)
    done_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    first_collision_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    reward_sum = torch.tensor(0.0, device=env.device)
    reward_count = torch.tensor(0.0, device=env.device)
    nearest_sum = torch.tensor(0.0, device=env.device)
    nearest_count = torch.tensor(0.0, device=env.device)
    near_violation_count = torch.tensor(0.0, device=env.device)
    global_min_nearest = torch.tensor(float("inf"), device=env.device)
    terrain_height_sum = torch.tensor(0.0, device=env.device)
    terrain_height_count = torch.tensor(0.0, device=env.device)
    terrain_height_min = torch.tensor(float("inf"), device=env.device)
    terrain_height_max = torch.tensor(float("-inf"), device=env.device)
    terrain_roughness_sum = torch.tensor(0.0, device=env.device)
    terrain_roughness_max = torch.tensor(0.0, device=env.device)
    terrain_traversability_min = torch.tensor(float("inf"), device=env.device)
    terrain_speed_scale_sum = torch.tensor(0.0, device=env.device)
    terrain_speed_scale_count = torch.tensor(0.0, device=env.device)
    dmax_ok_sum = torch.tensor(0.0, device=env.device)
    dispersion_ok_sum = torch.tensor(0.0, device=env.device)
    speed_ok_sum = torch.tensor(0.0, device=env.device)
    instant_success_sum = torch.tensor(0.0, device=env.device)
    gate_sample_count = torch.tensor(0.0, device=env.device)

    for step_id in range(steps):
        active_before = active.clone()
        with torch.no_grad():
            action = act(actor_obs)
        step_output = env.step(action)
        actor_obs = step_output.actor_obs
        metrics = step_output.info["metrics"]
        done = step_output.info["done"]

        final_dmax = torch.where(active_before, metrics.dmax, final_dmax)
        final_dispersion = torch.where(active_before, metrics.dispersion, final_dispersion)
        final_mean_speed = torch.where(active_before, metrics.mean_speed, final_mean_speed)
        success_hold_count = step_output.info["success_hold_count"]
        final_success_hold_count = torch.where(active_before, success_hold_count, final_success_hold_count)
        max_success_hold_count = torch.maximum(
            max_success_hold_count,
            torch.where(active_before, success_hold_count, torch.zeros_like(success_hold_count)),
        )
        per_env_reward = step_output.rewards.mean(dim=-1)
        reward_sum = reward_sum + per_env_reward[active_before].sum()
        reward_count = reward_count + active_before.float().sum()
        success_gates = step_output.info["success_gates"]
        active_gate_count = active_before.float().sum()
        gate_sample_count = gate_sample_count + active_gate_count
        dmax_ok_sum = dmax_ok_sum + success_gates.dmax_ok[active_before].float().sum()
        dispersion_ok_sum = dispersion_ok_sum + success_gates.dispersion_ok[active_before].float().sum()
        speed_ok_sum = speed_ok_sum + success_gates.speed_ok[active_before].float().sum()
        instant_success_sum = instant_success_sum + success_gates.instant_success[active_before].float().sum()

        first_done = done.done & active_before
        done_step = torch.where(first_done, torch.full_like(done_step, step_id + 1), done_step)
        success_seen = success_seen | (done.success & active_before)
        first_collision = done.collision & active_before & ~collision_seen
        first_collision_step = torch.where(
            first_collision,
            torch.full_like(first_collision_step, step_id + 1),
            first_collision_step,
        )
        collision_seen = collision_seen | (done.collision & active_before)
        timeout_seen = timeout_seen | (done.timeout & active_before)
        nearest = metrics.nearest_neighbor_distance.amin(dim=-1)
        active_nearest = nearest[active_before]
        if active_nearest.numel() > 0:
            nearest_sum = nearest_sum + active_nearest.sum()
            nearest_count = nearest_count + torch.tensor(float(active_nearest.numel()), device=env.device)
            near_violation_count = near_violation_count + (active_nearest < env.cfg.safety.near_distance).float().sum()
            global_min_nearest = torch.minimum(global_min_nearest, active_nearest.amin())
        terrain_features = step_output.info.get("terrain_features")
        if terrain_features is not None:
            active_terrain = terrain_features[active_before].reshape(-1, terrain_features.shape[-1])
            if active_terrain.numel() > 0:
                heights = active_terrain[:, 0]
                roughness = active_terrain[:, 3]
                traversability = active_terrain[:, 4]
                terrain_height_sum = terrain_height_sum + heights.sum()
                terrain_height_count = terrain_height_count + torch.tensor(float(heights.numel()), device=env.device)
                terrain_height_min = torch.minimum(terrain_height_min, heights.amin())
                terrain_height_max = torch.maximum(terrain_height_max, heights.amax())
                terrain_roughness_sum = terrain_roughness_sum + roughness.sum()
                terrain_roughness_max = torch.maximum(terrain_roughness_max, roughness.amax())
                terrain_traversability_min = torch.minimum(terrain_traversability_min, traversability.amin())
        terrain_speed_scale = step_output.info.get("terrain_speed_scale")
        if terrain_speed_scale is not None:
            per_env_terrain_speed_scale = terrain_speed_scale.mean(dim=-1)
            final_terrain_speed_scale = torch.where(
                active_before,
                per_env_terrain_speed_scale,
                final_terrain_speed_scale,
            )
            active_speed_scale = terrain_speed_scale[active_before].reshape(-1)
            if active_speed_scale.numel() > 0:
                terrain_speed_scale_sum = terrain_speed_scale_sum + active_speed_scale.sum()
                terrain_speed_scale_count = terrain_speed_scale_count + torch.tensor(
                    float(active_speed_scale.numel()),
                    device=env.device,
                )
        active = active & ~done.done
        if not active.any():
            break

    initial_dmax_mean = initial_dmax.mean()
    final_dmax_mean = final_dmax.mean()
    timeout_count = int(timeout_seen.sum().detach().cpu())
    timeout_episode_metrics = {
        "count": timeout_count,
        "final_dmax_mean": float(final_dmax[timeout_seen].mean().detach().cpu()) if timeout_seen.any() else None,
        "final_dispersion_mean": float(final_dispersion[timeout_seen].mean().detach().cpu())
        if timeout_seen.any()
        else None,
        "final_mean_speed_mean": float(final_mean_speed[timeout_seen].mean().detach().cpu())
        if timeout_seen.any()
        else None,
        "mean_terrain_speed_scale": float(final_terrain_speed_scale[timeout_seen].mean().detach().cpu())
        if timeout_seen.any()
        else None,
    }
    hold_histogram = torch.bincount(
        max_success_hold_count.clamp(max=env.cfg.success_thresholds.hold_steps).detach().cpu(),
        minlength=env.cfg.success_thresholds.hold_steps + 1,
    )
    result = {
        "status": "ok",
        "backend": backend,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "device": str(env.device),
        "num_envs": env.num_envs,
        "steps": steps,
        "initial_dmax": float(initial_dmax_mean.detach().cpu()),
        "final_dmax": float(final_dmax_mean.detach().cpu()),
        "dmax_reduction_ratio": float((final_dmax_mean / initial_dmax_mean.clamp_min(1.0e-6)).detach().cpu()),
        "initial_dispersion": float(initial_dispersion.mean().detach().cpu()),
        "final_dispersion": float(final_dispersion.mean().detach().cpu()),
        "final_mean_speed": float(final_mean_speed.mean().detach().cpu()),
        "mean_reward": float((reward_sum / reward_count.clamp_min(1.0)).detach().cpu()),
        "success_rate": float(success_seen.float().mean().detach().cpu()),
        "collision_rate": float(collision_seen.float().mean().detach().cpu()),
        "timeout_rate": float(timeout_seen.float().mean().detach().cpu()),
        "finished_rate": float((~active).float().mean().detach().cpu()),
        "mean_done_step": float(done_step[done_step > 0].float().mean().detach().cpu())
        if (done_step > 0).any()
        else None,
        "min_nearest_distance": float(global_min_nearest.detach().cpu()) if torch.isfinite(global_min_nearest) else None,
        "mean_nearest_distance": float((nearest_sum / nearest_count.clamp_min(1.0)).detach().cpu()),
        "near_violation_rate": float((near_violation_count / nearest_count.clamp_min(1.0)).detach().cpu()),
        "first_collision_step_mean": float(first_collision_step[first_collision_step > 0].float().mean().detach().cpu())
        if (first_collision_step > 0).any()
        else None,
        "collision_episode_ids": torch.nonzero(collision_seen, as_tuple=False).flatten().detach().cpu().tolist(),
        "mean_terrain_height": float((terrain_height_sum / terrain_height_count.clamp_min(1.0)).detach().cpu()),
        "terrain_height_range": float((terrain_height_max - terrain_height_min).detach().cpu())
        if torch.isfinite(terrain_height_min) and torch.isfinite(terrain_height_max)
        else 0.0,
        "mean_roughness": float((terrain_roughness_sum / terrain_height_count.clamp_min(1.0)).detach().cpu()),
        "max_roughness": float(terrain_roughness_max.detach().cpu()),
        "min_traversability": float(terrain_traversability_min.detach().cpu())
        if torch.isfinite(terrain_traversability_min)
        else 1.0,
        "mean_terrain_speed_scale": float(
            (terrain_speed_scale_sum / terrain_speed_scale_count.clamp_min(1.0)).detach().cpu()
        ),
        "dmax_ok_rate": float((dmax_ok_sum / gate_sample_count.clamp_min(1.0)).detach().cpu()),
        "dispersion_ok_rate": float((dispersion_ok_sum / gate_sample_count.clamp_min(1.0)).detach().cpu()),
        "speed_ok_rate": float((speed_ok_sum / gate_sample_count.clamp_min(1.0)).detach().cpu()),
        "instant_success_rate": float((instant_success_sum / gate_sample_count.clamp_min(1.0)).detach().cpu()),
        "max_success_hold_count_mean": float(max_success_hold_count.float().mean().detach().cpu()),
        "final_success_hold_count_mean": float(final_success_hold_count.float().mean().detach().cpu()),
        "hold_count_histogram": {
            str(index): int(count.item()) for index, count in enumerate(hold_histogram)
        },
        "timeout_episode_metrics": timeout_episode_metrics,
    }
    output_path = _resolve_path(output)
    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        result["artifact"] = str(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp_001_minimal.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/exp_001_minimal_proxy.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = evaluate_checkpoint(
        args.config,
        args.checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        output=args.output,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
