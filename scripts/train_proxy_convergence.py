#!/usr/bin/env python
"""Warm-start and PPO training for proxy multi-rover gathering convergence."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass
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
import torch.nn.functional as F
import yaml
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency guard
    SummaryWriter = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import ROOT, cfg_from_experiment, ensure_output_dir, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from train import Actor, Critic


STRICT_THRESHOLDS = {
    "dmax_reduction_ratio": 0.2,
    "success_rate": 0.9,
    "collision_rate": 0.02,
    "timeout_rate": 0.0,
}


@dataclass(slots=True)
class Rollout:
    obs: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def _nearest_distances(positions: torch.Tensor) -> torch.Tensor:
    pairwise = torch.cdist(positions[..., :2], positions[..., :2])
    n_agents = positions.shape[1]
    eye = torch.eye(n_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
    return pairwise.masked_fill(eye, float("inf")).amin(dim=-1)


def scripted_gather_action(
    env: MultiRoverGatheringCore,
    stop_radius: float = 0.45,
    slow_distance: float | None = None,
    safety_aware: bool = True,
) -> torch.Tensor:
    positions_xy = env.positions[..., :2]
    centroid_xy = positions_xy.mean(dim=1, keepdim=True)
    if safety_aware:
        rel = positions_xy - centroid_xy
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
        fallback_angles = torch.linspace(0.0, 2.0 * torch.pi, env.n_agents + 1, device=env.device)[:-1]
        fallback = torch.stack((torch.cos(fallback_angles), torch.sin(fallback_angles)), dim=-1)
        unit = torch.where(dist > 1.0e-6, rel / dist.clamp_min(1.0e-6), fallback[None, :, :])
        target_xy = centroid_xy + unit * stop_radius
        world_delta = target_xy - positions_xy
        world_delta = torch.where(dist > stop_radius, world_delta, torch.zeros_like(world_delta))
        if slow_distance is not None and slow_distance > 0.0:
            nearest = _nearest_distances(env.positions).unsqueeze(-1)
            scale = torch.clamp((nearest - env.cfg.safety.collision_distance) / slow_distance, 0.0, 1.0)
            world_delta = world_delta * scale
    else:
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


def _act_deterministic(actor: Actor, actor_obs: torch.Tensor) -> torch.Tensor:
    flat_obs = actor_obs.reshape(-1, actor_obs.shape[-1])
    action = actor(flat_obs).mean
    return action.view(actor_obs.shape[0], actor_obs.shape[1], -1)


def _save_checkpoint(
    path: Path,
    actor: Actor,
    critic: Critic,
    raw_cfg: dict,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "cfg": raw_cfg,
            "metrics": metrics,
        },
        path,
    )


def evaluate_actor(
    actor: Actor,
    cfg,
    num_envs: int,
    steps: int,
    device: str,
    seed: int,
    capture_history: bool = False,
) -> tuple[dict, list[np.ndarray], dict[str, list[float]]]:
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.simulation.num_envs = num_envs
    eval_cfg.simulation.device = device
    eval_cfg.seed = seed
    env = MultiRoverGatheringCore(eval_cfg)
    actor_was_training = actor.training
    actor.eval()
    actor_obs, _ = env.get_observations()

    initial_dmax = env.metrics.dmax.detach().clone()
    initial_dispersion = env.metrics.dispersion.detach().clone()
    final_dmax = initial_dmax.clone()
    final_dispersion = initial_dispersion.clone()
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    success_seen = torch.zeros_like(active)
    collision_seen = torch.zeros_like(active)
    timeout_seen = torch.zeros_like(active)
    first_collision_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    reward_sum = torch.tensor(0.0, device=env.device)
    reward_count = torch.tensor(0.0, device=env.device)
    nearest_sum = torch.tensor(0.0, device=env.device)
    nearest_count = torch.tensor(0.0, device=env.device)
    near_violation_count = torch.tensor(0.0, device=env.device)
    global_min_nearest = torch.tensor(float("inf"), device=env.device)
    position_history: list[np.ndarray] = [env.positions[0].detach().cpu().numpy().copy()]
    curve_history = {
        "dmax": [float(initial_dmax[0].detach().cpu())],
        "reward": [],
        "nearest": [],
        "collision_rate": [],
    }

    for step_id in range(steps):
        active_before = active.clone()
        with torch.no_grad():
            action = _act_deterministic(actor, actor_obs)
        output = env.step(action)
        actor_obs = output.actor_obs
        metrics = output.info["metrics"]
        done = output.info["done"]
        final_dmax = torch.where(active_before, metrics.dmax, final_dmax)
        final_dispersion = torch.where(active_before, metrics.dispersion, final_dispersion)
        per_env_reward = output.rewards.mean(dim=-1)
        reward_sum = reward_sum + per_env_reward[active_before].sum()
        reward_count = reward_count + active_before.float().sum()
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
        active = active & ~done.done
        if capture_history:
            position_history.append(env.positions[0].detach().cpu().numpy().copy())
            curve_history["dmax"].append(float(metrics.dmax[0].detach().cpu()))
            curve_history["reward"].append(float(output.rewards[0].mean().detach().cpu()))
            curve_history["nearest"].append(float(metrics.nearest_neighbor_distance[0].amin().detach().cpu()))
            curve_history["collision_rate"].append(float(collision_seen.float().mean().detach().cpu()))
        if not active.any():
            break

    if actor_was_training:
        actor.train()
    initial_dmax_mean = initial_dmax.mean()
    final_dmax_mean = final_dmax.mean()
    metrics = {
        "initial_dmax": float(initial_dmax_mean.detach().cpu()),
        "final_dmax": float(final_dmax_mean.detach().cpu()),
        "dmax_reduction_ratio": float((final_dmax_mean / initial_dmax_mean.clamp_min(1.0e-6)).detach().cpu()),
        "initial_dispersion": float(initial_dispersion.mean().detach().cpu()),
        "final_dispersion": float(final_dispersion.mean().detach().cpu()),
        "mean_reward": float((reward_sum / reward_count.clamp_min(1.0)).detach().cpu()),
        "success_rate": float(success_seen.float().mean().detach().cpu()),
        "collision_rate": float(collision_seen.float().mean().detach().cpu()),
        "timeout_rate": float(timeout_seen.float().mean().detach().cpu()),
        "finished_rate": float((~active).float().mean().detach().cpu()),
        "min_nearest_distance": float(global_min_nearest.detach().cpu()) if torch.isfinite(global_min_nearest) else None,
        "mean_nearest_distance": float((nearest_sum / nearest_count.clamp_min(1.0)).detach().cpu()),
        "near_violation_rate": float((near_violation_count / nearest_count.clamp_min(1.0)).detach().cpu()),
        "first_collision_step_mean": float(first_collision_step[first_collision_step > 0].float().mean().detach().cpu())
        if (first_collision_step > 0).any()
        else None,
        "collision_episode_ids": torch.nonzero(collision_seen, as_tuple=False).flatten().detach().cpu().tolist(),
    }
    return metrics, position_history, curve_history


def _randomize_bc_state(env: MultiRoverGatheringCore) -> None:
    base_angles = torch.linspace(0.0, 2.0 * torch.pi, env.n_agents + 1, device=env.device)[:-1]
    base = torch.stack((torch.cos(base_angles), torch.sin(base_angles)), dim=-1)
    radius = torch.empty(env.num_envs, 1, 1, device=env.device).uniform_(
        0.25,
        4.0,
        generator=env.generator,
    )
    jitter = 0.35 * torch.randn(
        env.num_envs,
        env.n_agents,
        2,
        generator=env.generator,
        device=env.device,
    )
    centers = torch.empty(env.num_envs, 1, 2, device=env.device).uniform_(
        -1.0,
        1.0,
        generator=env.generator,
    )
    env.positions[..., :2] = centers + radius * base[None, :, :] + jitter
    env.positions[..., 2] = 0.0
    env.yaws = torch.empty_like(env.yaws).uniform_(-torch.pi, torch.pi, generator=env.generator)
    env.velocities_xy.zero_()
    env.angular_velocities.zero_()
    env.previous_physical_action.zero_()
    env.step_count.zero_()
    env.success_hold_count.zero_()
    env.oracle_point = env.positions.mean(dim=1)


def run_behavior_cloning(
    actor: Actor,
    cfg,
    steps: int,
    batch_size: int,
    learning_rate: float,
    teacher_stop_radius: float = 0.45,
    teacher_slow_distance: float = 0.40,
) -> list[dict]:
    if steps <= 0:
        return []
    bc_cfg = copy.deepcopy(cfg)
    bc_cfg.simulation.num_envs = max(bc_cfg.simulation.num_envs, int(np.ceil(batch_size / bc_cfg.task.n_agents)))
    env = MultiRoverGatheringCore(bc_cfg)
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    metrics: list[dict] = []
    samples_per_snapshot = env.num_envs * env.n_agents
    snapshots_per_batch = max(1, int(np.ceil(batch_size / samples_per_snapshot)))

    for step_id in range(steps):
        obs_items = []
        target_items = []
        for _ in range(snapshots_per_batch):
            _randomize_bc_state(env)
            actor_obs, _ = env.get_observations()
            target = scripted_gather_action(
                env,
                stop_radius=teacher_stop_radius,
                slow_distance=teacher_slow_distance,
                safety_aware=True,
            )
            obs_items.append(actor_obs.reshape(-1, actor_obs.shape[-1]).detach())
            target_items.append(target.reshape(-1, target.shape[-1]).detach())
        obs = torch.cat(obs_items, dim=0)[:batch_size]
        target = torch.cat(target_items, dim=0)[:batch_size]
        pred = actor(obs).mean
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        if step_id == 0 or (step_id + 1) % 100 == 0 or step_id + 1 == steps:
            metrics.append({"phase": "bc", "step": step_id + 1, "bc_loss": float(loss.detach().cpu())})
    return metrics


def collect_rollout(
    env: MultiRoverGatheringCore,
    actor: Actor,
    critic: Critic,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
) -> tuple[Rollout, dict]:
    obs_items = []
    state_items = []
    action_items = []
    log_prob_items = []
    reward_items = []
    done_items = []
    value_items = []
    dmax_items = []

    actor_obs, critic_state = env.get_observations()
    for _ in range(rollout_steps):
        flat_obs = actor_obs.reshape(-1, actor_obs.shape[-1])
        dist = actor(flat_obs)
        flat_action = torch.clamp(dist.sample(), -1.0, 1.0)
        flat_log_prob = dist.log_prob(flat_action).sum(dim=-1)
        action = flat_action.view(env.num_envs, env.n_agents, 2)
        value = critic(critic_state)
        output = env.step(action.detach())
        reward = output.rewards.mean(dim=-1)
        done = output.info["done"].done.float()

        obs_items.append(actor_obs.detach())
        state_items.append(critic_state.detach())
        action_items.append(action.detach())
        log_prob_items.append(flat_log_prob.detach().view(env.num_envs, env.n_agents))
        reward_items.append(reward.detach())
        done_items.append(done.detach())
        value_items.append(value.detach())
        dmax_items.append(output.info["metrics"].dmax.detach())
        actor_obs = output.actor_obs
        critic_state = output.critic_state

    rewards = torch.stack(reward_items)
    dones = torch.stack(done_items)
    values = torch.stack(value_items)
    with torch.no_grad():
        bootstrap = critic(critic_state)

    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(env.num_envs, device=env.device)
    for step_id in reversed(range(rollout_steps)):
        next_value = bootstrap if step_id == rollout_steps - 1 else values[step_id + 1]
        next_nonterminal = 1.0 - dones[step_id]
        delta = rewards[step_id] + gamma * next_value * next_nonterminal - values[step_id]
        last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
        advantages[step_id] = last_advantage
    returns = advantages + values
    rollout = Rollout(
        obs=torch.stack(obs_items),
        states=torch.stack(state_items),
        actions=torch.stack(action_items),
        log_probs=torch.stack(log_prob_items),
        rewards=rewards,
        dones=dones,
        values=values,
        returns=returns.detach(),
        advantages=advantages.detach(),
    )
    metrics = {
        "rollout_reward": float(rewards.mean().detach().cpu()),
        "rollout_dmax": float(torch.stack(dmax_items).mean().detach().cpu()),
        "rollout_done_rate": float(dones.mean().detach().cpu()),
    }
    return rollout, metrics


def ppo_update(
    rollout: Rollout,
    actor: Actor,
    critic: Critic,
    optimizer: torch.optim.Optimizer,
    clip_epsilon: float,
    ppo_epochs: int,
    mini_batches: int,
    value_loss_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    reference_actor: Actor | None = None,
    reference_policy_coef: float = 0.0,
) -> dict:
    obs = rollout.obs.reshape(-1, rollout.obs.shape[-1])
    actions = rollout.actions.reshape(-1, rollout.actions.shape[-1])
    old_log_probs = rollout.log_probs.reshape(-1)
    policy_advantages = rollout.advantages[:, :, None].expand(-1, -1, rollout.obs.shape[2]).reshape(-1)
    policy_advantages = (policy_advantages - policy_advantages.mean()) / policy_advantages.std().clamp_min(1.0e-6)

    states = rollout.states.reshape(-1, rollout.states.shape[-1])
    returns = rollout.returns.reshape(-1)
    value_count = states.shape[0]
    policy_count = obs.shape[0]
    last_metrics: dict[str, float] = {}

    for _ in range(ppo_epochs):
        policy_perm = torch.randperm(policy_count, device=obs.device)
        value_perm = torch.randperm(value_count, device=states.device)
        policy_chunks = torch.chunk(policy_perm, mini_batches)
        value_chunks = torch.chunk(value_perm, mini_batches)
        for policy_idx, value_idx in zip(policy_chunks, value_chunks):
            dist = actor(obs[policy_idx])
            new_log_probs = dist.log_prob(actions[policy_idx]).sum(dim=-1)
            ratio = torch.exp(new_log_probs - old_log_probs[policy_idx])
            adv = policy_advantages[policy_idx]
            policy_loss = -torch.minimum(
                ratio * adv,
                torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv,
            ).mean()
            entropy = dist.entropy().sum(dim=-1).mean()

            values = critic(states[value_idx])
            value_loss = F.mse_loss(values, returns[value_idx])
            reference_loss = torch.tensor(0.0, device=obs.device)
            if reference_actor is not None and reference_policy_coef > 0.0:
                with torch.no_grad():
                    reference_mean = reference_actor(obs[policy_idx]).mean
                reference_loss = F.mse_loss(dist.mean, reference_mean)
            loss = (
                policy_loss
                + value_loss_coef * value_loss
                + reference_policy_coef * reference_loss
                - entropy_coef * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), max_grad_norm)
            optimizer.step()
            last_metrics = {
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "reference_policy_loss": float(reference_loss.detach().cpu()),
            }
    return last_metrics


def _save_curves(eval_records: list[dict], path: Path) -> None:
    if not eval_records:
        return
    x = np.arange(len(eval_records))
    ratio = [item["dmax_reduction_ratio"] for item in eval_records]
    reward = [item["mean_reward"] for item in eval_records]
    success = [item["success_rate"] for item in eval_records]
    collision = [item["collision_rate"] for item in eval_records]
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), constrained_layout=True)
    axes[0].plot(x, ratio, marker="o")
    axes[0].axhline(STRICT_THRESHOLDS["dmax_reduction_ratio"], color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("final / initial dmax")
    axes[1].plot(x, reward, marker="o", color="tab:green")
    axes[1].set_ylabel("mean reward")
    axes[2].plot(x, success, marker="o", color="tab:purple")
    axes[2].axhline(STRICT_THRESHOLDS["success_rate"], color="tab:red", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("success rate")
    axes[3].plot(x, collision, marker="o", color="tab:orange")
    axes[3].axhline(STRICT_THRESHOLDS["collision_rate"], color="tab:red", linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("collision rate")
    axes[3].set_xlabel("evaluation")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_safety_diagnostics(eval_records: list[dict], curve_history: dict[str, list[float]], path: Path) -> None:
    if not eval_records:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    eval_x = np.arange(len(eval_records))
    axes[0, 0].plot(eval_x, [item.get("min_nearest_distance") or 0.0 for item in eval_records], marker="o")
    axes[0, 0].set_title("Eval min nearest distance")
    axes[0, 1].plot(eval_x, [item["collision_rate"] for item in eval_records], marker="o", color="tab:orange")
    axes[0, 1].axhline(STRICT_THRESHOLDS["collision_rate"], color="tab:red", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("Eval collision rate")
    axes[1, 0].scatter(
        [item["collision_rate"] for item in eval_records],
        [item["success_rate"] for item in eval_records],
        c=eval_x,
        cmap="viridis",
    )
    axes[1, 0].set_xlabel("collision rate")
    axes[1, 0].set_ylabel("success rate")
    axes[1, 0].set_title("Success / collision tradeoff")
    if curve_history.get("nearest"):
        axes[1, 1].plot(np.arange(len(curve_history["nearest"])), curve_history["nearest"], color="tab:green")
    axes[1, 1].set_title("Best rollout nearest distance")
    axes[1, 1].set_xlabel("step")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_rollout_gif(position_history: list[np.ndarray], path: Path) -> None:
    if len(position_history) < 2:
        return
    frames = []
    all_xy = np.concatenate([positions[:, :2] for positions in position_history], axis=0)
    center = all_xy.mean(axis=0)
    radius = max(3.0, float(np.max(np.linalg.norm(all_xy - center[None, :], axis=-1))) + 0.5)
    for step_id, positions in enumerate(position_history):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(positions[:, 0], positions[:, 1], c=["tab:blue", "tab:orange", "tab:green", "tab:red"], s=80)
        ax.scatter(positions[:, 0].mean(), positions[:, 1].mean(), c="black", marker="x", s=60)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"Proxy rollout step {step_id}")
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(path, frames, duration=0.12)


def _linear_schedule(start: float, end: float, index: int, total: int) -> float:
    if total <= 1:
        return end
    alpha = min(1.0, max(0.0, index / float(total - 1)))
    return (1.0 - alpha) * start + alpha * end


def strict_acceptance(metrics: dict, thresholds: dict | None = None, required_phase: str | None = None) -> dict:
    thresholds = thresholds or STRICT_THRESHOLDS
    checks = {
        "dmax_reduction_ratio": metrics["dmax_reduction_ratio"] <= thresholds["dmax_reduction_ratio"],
        "success_rate": metrics["success_rate"] >= thresholds["success_rate"],
        "collision_rate": metrics["collision_rate"] <= thresholds["collision_rate"],
        "timeout_rate": metrics["timeout_rate"] <= thresholds["timeout_rate"],
    }
    if required_phase is not None:
        checks["phase"] = metrics.get("phase") == required_phase
    return {"passed": all(checks.values()), "checks": checks, "thresholds": thresholds}


def _metric_is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def _write_tensorboard_scalars(writer, prefix: str, metrics: dict, step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        if _metric_is_finite_number(value):
            writer.add_scalar(f"{prefix}/{key}", float(value), step)


def _candidate_allowed(phase: str, best_source: str) -> bool:
    return best_source == "all" or phase == best_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp_004_proxy_convergence.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--mode", choices=("pure_rl", "bc_only", "bc_ppo", "weak_warmstart"), default=None)
    parser.add_argument("--best-source", choices=("all", "ppo"), default=None)
    parser.add_argument("--required-best-phase", choices=("ppo",), default=None)
    parser.add_argument("--total-env-steps", type=int, default=None)
    parser.add_argument("--bc-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coef-start", type=float, default=None)
    parser.add_argument("--entropy-coef-end", type=float, default=None)
    parser.add_argument("--teacher-stop-radius", type=float, default=None)
    parser.add_argument("--teacher-slow-distance", type=float, default=None)
    parser.add_argument("--reference-policy-coef-start", type=float, default=None)
    parser.add_argument("--reference-policy-coef-end", type=float, default=None)
    parser.add_argument("--safety-near-distance", type=float, default=None)
    parser.add_argument("--near-penalty-coef", type=float, default=None)
    parser.add_argument("--collision-penalty-coef", type=float, default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--strict-success-json", default=None)
    parser.add_argument("--tensorboard", choices=("auto", "on", "off"), default=None)
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    exp = raw_cfg.get("experiment", {})
    algo = raw_cfg.get("algorithm", {})
    cfg = cfg_from_experiment(args.config)
    if args.device is not None:
        cfg.simulation.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
        raw_cfg.setdefault("experiment", {})["seed"] = args.seed
    mode = args.mode or str(algo.get("mode", "bc_ppo"))
    best_source = args.best_source or str(algo.get("best_source", "all"))
    if mode == "bc_only":
        best_source = "all"
    required_best_phase = args.required_best_phase or algo.get("required_best_phase")
    if args.safety_near_distance is not None:
        cfg.safety.near_distance = args.safety_near_distance
    if args.near_penalty_coef is not None:
        cfg.reward_coefficients.near_distance = args.near_penalty_coef
    if args.collision_penalty_coef is not None:
        cfg.reward_coefficients.inter_agent_collision = args.collision_penalty_coef
    torch.manual_seed(cfg.seed)
    started_at = time.perf_counter()

    env = MultiRoverGatheringCore(cfg)
    actor = Actor(cfg.actor_obs_dim).to(env.device)
    critic = Critic(cfg.critic_state_dim).to(env.device)
    base_log_dir = args.log_dir or exp.get("log_dir", "outputs/logs/exp_004_proxy_convergence")
    log_dir = ensure_output_dir(base_log_dir)
    if args.run_name:
        log_dir = ensure_output_dir(log_dir / args.run_name)
    checkpoint_dir = ensure_output_dir(exp.get("checkpoint_dir", "outputs/checkpoints"))
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else checkpoint_dir / exp.get("checkpoint_name", "exp_004_proxy_converged.pt")
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    metrics_path = log_dir / "train_metrics.jsonl"
    eval_path = log_dir / "eval_metrics.json"
    curves_path = log_dir / "convergence_curves.png"
    safety_path = log_dir / "safety_diagnostics.png"
    gif_path = log_dir / "eval_rollout.gif"
    tensorboard_setting = args.tensorboard or str(raw_cfg.get("logging", {}).get("tensorboard", "auto")).lower()
    writer = None
    tensorboard_dir = log_dir / "tensorboard"
    if tensorboard_setting != "off":
        if SummaryWriter is not None:
            writer = SummaryWriter(str(tensorboard_dir))
        elif tensorboard_setting == "on":
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            (tensorboard_dir / "tensorboard_unavailable.txt").write_text(
                "torch.utils.tensorboard could not be imported in this environment.\n",
                encoding="utf-8",
            )

    total_env_steps = int(args.total_env_steps or exp.get("total_env_steps", 1_000_000))
    rollout_steps = int(exp.get("rollout_steps", 128))
    updates = 0 if total_env_steps <= 0 else max(1, total_env_steps // (env.num_envs * rollout_steps))
    bc_steps = int(args.bc_steps if args.bc_steps is not None else algo.get("bc_steps", 2000))
    if mode == "pure_rl":
        bc_steps = 0
    if mode == "bc_only":
        updates = 0
    bc_batch_size = int(algo.get("bc_batch_size", 8192))
    bc_learning_rate = float(algo.get("bc_learning_rate", 1.0e-3))
    learning_rate = float(args.learning_rate or algo.get("learning_rate", 3.0e-4))
    entropy_start = float(args.entropy_coef_start or algo.get("entropy_coef_start", 0.01))
    entropy_end = float(args.entropy_coef_end or algo.get("entropy_coef_end", 0.001))
    reference_start = float(args.reference_policy_coef_start or algo.get("reference_policy_coef_start", 0.0))
    reference_end = float(args.reference_policy_coef_end or algo.get("reference_policy_coef_end", 0.0))
    eval_interval = int(exp.get("eval_interval_updates", 4))
    eval_num_envs = int(exp.get("eval_num_envs", 256))
    eval_steps = int(exp.get("eval_steps", 100))
    teacher_stop_radius = float(args.teacher_stop_radius or algo.get("teacher_stop_radius", 0.45))
    teacher_slow_distance = float(args.teacher_slow_distance or algo.get("teacher_slow_distance", 0.40))

    train_records: list[dict] = []
    eval_records: list[dict] = []

    bc_records = run_behavior_cloning(
        actor,
        cfg,
        steps=bc_steps,
        batch_size=bc_batch_size,
        learning_rate=bc_learning_rate,
        teacher_stop_radius=teacher_stop_radius,
        teacher_slow_distance=teacher_slow_distance,
    )
    train_records.extend(bc_records)
    if writer is not None:
        for record in bc_records:
            _write_tensorboard_scalars(writer, "bc", record, int(record["step"]))

    baseline_metrics, baseline_positions, baseline_curve_history = evaluate_actor(
        actor,
        cfg,
        num_envs=eval_num_envs,
        steps=eval_steps,
        device=str(env.device),
        seed=cfg.seed + 1000,
        capture_history=True,
    )
    baseline_phase = "bc" if bc_steps > 0 else "initial"
    baseline_metrics.update({"phase": baseline_phase, "update": 0})
    eval_records.append(baseline_metrics)
    _write_tensorboard_scalars(writer, "eval", baseline_metrics, 0)
    best_metrics: dict | None = None
    best_positions: list[np.ndarray] = []
    best_curve_history: dict[str, list[float]] = {}
    best_ratio = float("inf")
    if _candidate_allowed(baseline_phase, best_source):
        best_metrics = baseline_metrics
        best_positions = baseline_positions
        best_curve_history = baseline_curve_history
        best_ratio = baseline_metrics["dmax_reduction_ratio"]
        _save_checkpoint(checkpoint_path, actor, critic, raw_cfg, baseline_metrics)

    reference_actor = None
    if mode in {"bc_ppo", "weak_warmstart"} and bc_steps > 0 and max(reference_start, reference_end) > 0.0:
        reference_actor = copy.deepcopy(actor).to(env.device)
        reference_actor.eval()
        for parameter in reference_actor.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=learning_rate)
    try:
        with metrics_path.open("w", encoding="utf-8") as stream:
            for record in train_records:
                stream.write(json.dumps(record) + "\n")

            for update_id in range(updates):
                rollout, rollout_metrics = collect_rollout(
                    env,
                    actor,
                    critic,
                    rollout_steps=rollout_steps,
                    gamma=float(algo.get("gamma", 0.99)),
                    gae_lambda=float(algo.get("gae_lambda", 0.95)),
                )
                entropy_coef = _linear_schedule(entropy_start, entropy_end, update_id, updates)
                reference_coef = _linear_schedule(reference_start, reference_end, update_id, updates)
                update_metrics = ppo_update(
                    rollout,
                    actor,
                    critic,
                    optimizer,
                    clip_epsilon=float(algo.get("clip_epsilon", 0.2)),
                    ppo_epochs=int(algo.get("ppo_epochs", 4)),
                    mini_batches=int(algo.get("mini_batches", 8)),
                    value_loss_coef=float(algo.get("value_loss_coef", 0.5)),
                    entropy_coef=entropy_coef,
                    max_grad_norm=float(algo.get("max_grad_norm", 0.5)),
                    reference_actor=reference_actor,
                    reference_policy_coef=reference_coef,
                )
                record = {
                    "phase": "ppo",
                    "update": update_id + 1,
                    "env_steps": (update_id + 1) * env.num_envs * rollout_steps,
                    "entropy_coef": entropy_coef,
                    "reference_policy_coef": reference_coef,
                    **rollout_metrics,
                    **update_metrics,
                }
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                _write_tensorboard_scalars(writer, "ppo", record, update_id + 1)

                should_eval = (update_id + 1) % eval_interval == 0 or update_id + 1 == updates
                if should_eval:
                    eval_metrics, positions, curve_history = evaluate_actor(
                        actor,
                        cfg,
                        num_envs=eval_num_envs,
                        steps=eval_steps,
                        device=str(env.device),
                        seed=cfg.seed + 1000,
                        capture_history=True,
                    )
                    eval_metrics.update({"phase": "ppo", "update": update_id + 1})
                    eval_records.append(eval_metrics)
                    _write_tensorboard_scalars(writer, "eval", eval_metrics, update_id + 1)
                    if _candidate_allowed("ppo", best_source) and eval_metrics["dmax_reduction_ratio"] <= best_ratio:
                        best_ratio = eval_metrics["dmax_reduction_ratio"]
                        best_metrics = eval_metrics
                        best_positions = positions
                        best_curve_history = curve_history
                        _save_checkpoint(checkpoint_path, actor, critic, raw_cfg, best_metrics)
    finally:
        if writer is not None:
            writer.flush()

    if best_metrics is None:
        best_metrics = eval_records[-1]
        best_positions = baseline_positions
        best_curve_history = baseline_curve_history
        _save_checkpoint(checkpoint_path, actor, critic, raw_cfg, best_metrics)
    _save_curves(eval_records, curves_path)
    _save_safety_diagnostics(eval_records, best_curve_history, safety_path)
    _save_rollout_gif(best_positions, gif_path)
    strict = strict_acceptance(best_metrics, required_phase=required_best_phase)
    if writer is not None:
        _write_tensorboard_scalars(writer, "best", best_metrics, int(best_metrics.get("update", 0)))
        writer.close()
    summary = {
        "status": "ok",
        "device": str(env.device),
        "mode": mode,
        "seed": cfg.seed,
        "best_source": best_source,
        "required_best_phase": required_best_phase,
        "bc_steps": bc_steps,
        "updates": updates,
        "wall_time_s": time.perf_counter() - started_at,
        "baseline_metrics": baseline_metrics,
        "best_metrics": best_metrics,
        "strict_acceptance": strict,
        "checkpoint_path": str(checkpoint_path),
        "train_metrics": str(metrics_path),
        "eval_metrics": str(eval_path),
        "convergence_curves": str(curves_path),
        "safety_diagnostics": str(safety_path),
        "eval_rollout_gif": str(gif_path),
        "tensorboard_dir": str(tensorboard_dir) if tensorboard_setting != "off" else None,
    }
    with eval_path.open("w", encoding="utf-8") as stream:
        json.dump({"summary": summary, "evaluations": eval_records}, stream, indent=2)
    if args.strict_success_json:
        strict_path = Path(args.strict_success_json)
        if not strict_path.is_absolute():
            strict_path = ROOT / strict_path
        strict_path.parent.mkdir(parents=True, exist_ok=True)
        strict_path.write_text(json.dumps(strict, indent=2), encoding="utf-8")
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
