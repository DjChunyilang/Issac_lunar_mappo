#!/usr/bin/env python
"""Warm-start and PPO training for proxy multi-rover gathering convergence."""

from __future__ import annotations

import argparse
import copy
import json
import os
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import ROOT, cfg_from_experiment, ensure_output_dir, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from train import Actor, Critic


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


def scripted_gather_action(env: MultiRoverGatheringCore) -> torch.Tensor:
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
    reward_sum = torch.tensor(0.0, device=env.device)
    reward_count = torch.tensor(0.0, device=env.device)
    position_history: list[np.ndarray] = [env.positions[0].detach().cpu().numpy().copy()]
    curve_history = {"dmax": [float(initial_dmax[0].detach().cpu())], "reward": []}

    for _ in range(steps):
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
        collision_seen = collision_seen | (done.collision & active_before)
        timeout_seen = timeout_seen | (done.timeout & active_before)
        active = active & ~done.done
        if capture_history:
            position_history.append(env.positions[0].detach().cpu().numpy().copy())
            curve_history["dmax"].append(float(metrics.dmax[0].detach().cpu()))
            curve_history["reward"].append(float(output.rewards[0].mean().detach().cpu()))
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
            target = scripted_gather_action(env)
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
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), max_grad_norm)
            optimizer.step()
            last_metrics = {
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
            }
    return last_metrics


def _save_curves(eval_records: list[dict], path: Path) -> None:
    if not eval_records:
        return
    x = np.arange(len(eval_records))
    ratio = [item["dmax_reduction_ratio"] for item in eval_records]
    reward = [item["mean_reward"] for item in eval_records]
    success = [item["success_rate"] for item in eval_records]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
    axes[0].plot(x, ratio, marker="o")
    axes[0].axhline(0.4, color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("final / initial dmax")
    axes[1].plot(x, reward, marker="o", color="tab:green")
    axes[1].set_ylabel("mean reward")
    axes[2].plot(x, success, marker="o", color="tab:purple")
    axes[2].set_ylabel("success rate")
    axes[2].set_xlabel("evaluation")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp_004_proxy_convergence.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--total-env-steps", type=int, default=None)
    parser.add_argument("--bc-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coef-start", type=float, default=None)
    parser.add_argument("--entropy-coef-end", type=float, default=None)
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    exp = raw_cfg.get("experiment", {})
    algo = raw_cfg.get("algorithm", {})
    cfg = cfg_from_experiment(args.config)
    if args.device is not None:
        cfg.simulation.device = args.device
    torch.manual_seed(cfg.seed)

    env = MultiRoverGatheringCore(cfg)
    actor = Actor(cfg.actor_obs_dim).to(env.device)
    critic = Critic(cfg.critic_state_dim).to(env.device)
    log_dir = ensure_output_dir(exp.get("log_dir", "outputs/logs/exp_004_proxy_convergence"))
    checkpoint_dir = ensure_output_dir(exp.get("checkpoint_dir", "outputs/checkpoints"))
    checkpoint_path = checkpoint_dir / exp.get("checkpoint_name", "exp_004_proxy_converged.pt")
    metrics_path = log_dir / "train_metrics.jsonl"
    eval_path = log_dir / "eval_metrics.json"
    curves_path = log_dir / "convergence_curves.png"
    gif_path = log_dir / "eval_rollout.gif"

    total_env_steps = int(args.total_env_steps or exp.get("total_env_steps", 1_000_000))
    rollout_steps = int(exp.get("rollout_steps", 128))
    updates = max(1, total_env_steps // (env.num_envs * rollout_steps))
    bc_steps = int(args.bc_steps if args.bc_steps is not None else algo.get("bc_steps", 2000))
    bc_batch_size = int(algo.get("bc_batch_size", 8192))
    bc_learning_rate = float(algo.get("bc_learning_rate", 1.0e-3))
    learning_rate = float(args.learning_rate or algo.get("learning_rate", 3.0e-4))
    entropy_start = float(args.entropy_coef_start or algo.get("entropy_coef_start", 0.01))
    entropy_end = float(args.entropy_coef_end or algo.get("entropy_coef_end", 0.001))
    eval_interval = int(exp.get("eval_interval_updates", 4))
    eval_num_envs = int(exp.get("eval_num_envs", 256))
    eval_steps = int(exp.get("eval_steps", 100))

    train_records: list[dict] = []
    eval_records: list[dict] = []

    bc_records = run_behavior_cloning(
        actor,
        cfg,
        steps=bc_steps,
        batch_size=bc_batch_size,
        learning_rate=bc_learning_rate,
    )
    train_records.extend(bc_records)

    best_metrics, best_positions, _ = evaluate_actor(
        actor,
        cfg,
        num_envs=eval_num_envs,
        steps=eval_steps,
        device=str(env.device),
        seed=cfg.seed + 1000,
        capture_history=True,
    )
    best_metrics.update({"phase": "bc", "update": 0})
    eval_records.append(best_metrics)
    _save_checkpoint(checkpoint_path, actor, critic, raw_cfg, best_metrics)
    best_ratio = best_metrics["dmax_reduction_ratio"]

    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=learning_rate)
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
            )
            record = {
                "phase": "ppo",
                "update": update_id + 1,
                "env_steps": (update_id + 1) * env.num_envs * rollout_steps,
                "entropy_coef": entropy_coef,
                **rollout_metrics,
                **update_metrics,
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()

            should_eval = (update_id + 1) % eval_interval == 0 or update_id + 1 == updates
            if should_eval:
                eval_metrics, positions, _ = evaluate_actor(
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
                if eval_metrics["dmax_reduction_ratio"] <= best_ratio:
                    best_ratio = eval_metrics["dmax_reduction_ratio"]
                    best_metrics = eval_metrics
                    best_positions = positions
                    _save_checkpoint(checkpoint_path, actor, critic, raw_cfg, best_metrics)

    _save_curves(eval_records, curves_path)
    _save_rollout_gif(best_positions, gif_path)
    summary = {
        "status": "ok",
        "device": str(env.device),
        "bc_steps": bc_steps,
        "updates": updates,
        "best_metrics": best_metrics,
        "checkpoint_path": str(checkpoint_path),
        "train_metrics": str(metrics_path),
        "eval_metrics": str(eval_path),
        "convergence_curves": str(curves_path),
        "eval_rollout_gif": str(gif_path),
    }
    with eval_path.open("w", encoding="utf-8") as stream:
        json.dump({"summary": summary, "evaluations": eval_records}, stream, indent=2)
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
