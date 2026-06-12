#!/usr/bin/env python
"""Run a short SKRL MAPPO smoke job on the first-stage proxy environment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from _common import cfg_from_experiment, ensure_output_dir, load_yaml
from _skrl_metadata import (
    DEFAULT_TRAINING_SEMANTICS,
    resolve_checkpoint_name,
    resolve_training_semantics,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringSKRLEnv
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.oracle import compute_mean_oracle_distance
from lunar_rover_tasks.tasks.multi_rover_gathering.termination import compute_success_gates

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.trainers.torch import SequentialTrainer


TRAINING_SEMANTICS = DEFAULT_TRAINING_SEMANTICS


class SKRLPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True, reduction="sum")
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), -0.5))

    def compute(self, inputs, role):
        mean = torch.tanh(self.net(inputs["observations"]))
        return mean, {"log_std": self.log_std_parameter.expand_as(mean)}


class SKRLValue(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)
        self.net = nn.Sequential(
            nn.Linear(self.num_states, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def parse_bool_config(value, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _validate_homogeneous_spaces(env: MultiRoverGatheringSKRLEnv) -> None:
    first = env.possible_agents[0]
    obs_shape = env.observation_spaces[first].shape
    action_shape = env.action_spaces[first].shape
    for agent_id in env.possible_agents[1:]:
        if env.observation_spaces[agent_id].shape != obs_shape:
            raise ValueError("Shared actor requires homogeneous agent observation spaces.")
        if env.action_spaces[agent_id].shape != action_shape:
            raise ValueError("Shared actor requires homogeneous agent action spaces.")


def build_skrl_mappo_models(
    env: MultiRoverGatheringSKRLEnv,
    *,
    shared_actor: bool = True,
    centralized_critic: bool = True,
    shared_value: bool = True,
) -> dict[str, dict[str, Model]]:
    """Build MAPPO models with project CTDE semantics.

    Policies consume per-agent local observations. Values consume the centralized state returned
    by ``env.state()`` / ``env.state_space``.
    """
    if not centralized_critic:
        raise ValueError("This project only wires SKRL MAPPO with a centralized critic state.")
    if shared_actor or shared_value:
        _validate_homogeneous_spaces(env)

    first_agent = env.possible_agents[0]
    shared_policy = (
        SKRLPolicy(
            env.observation_spaces[first_agent],
            env.action_spaces[first_agent],
            env.device,
        )
        if shared_actor
        else None
    )
    shared_critic = (
        SKRLValue(
            env.observation_spaces[first_agent],
            env.state_space,
            env.action_spaces[first_agent],
            env.device,
        )
        if shared_value
        else None
    )

    models: dict[str, dict[str, Model]] = {}
    for agent_id in env.possible_agents:
        models[agent_id] = {
            "policy": shared_policy
            if shared_actor
            else SKRLPolicy(
                env.observation_spaces[agent_id],
                env.action_spaces[agent_id],
                env.device,
            ),
            "value": shared_critic
            if shared_value
            else SKRLValue(
                env.observation_spaces[agent_id],
                env.state_space,
                env.action_spaces[agent_id],
                env.device,
            ),
        }
    return models


def build_skrl_mappo_memories(
    env: MultiRoverGatheringSKRLEnv,
    *,
    rollout_steps: int,
) -> dict[str, RandomMemory]:
    return {
        agent_id: RandomMemory(
            memory_size=rollout_steps,
            num_envs=env.num_envs,
            device=env.device,
        )
        for agent_id in env.possible_agents
    }


def _mean_float(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().mean().cpu())


def _stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten().cpu()
    return {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_max": float(values.max()),
    }


def _distance_threshold_ratios(prefix: str, values: torch.Tensor, threshold: float) -> dict[str, float]:
    values = values.detach().float().flatten().cpu()
    return {
        f"{prefix}_within_1_5x_success_threshold": float((values <= 1.5 * threshold).float().mean()),
        f"{prefix}_within_2x_success_threshold": float((values <= 2.0 * threshold).float().mean()),
        f"{prefix}_within_3x_success_threshold": float((values <= 3.0 * threshold).float().mean()),
    }


REWARD_COMPONENTS = (
    "gather",
    "oracle",
    "energy",
    "safety",
    "terrain",
    "motion",
    "consistency",
    "success_hold",
    "terminal",
)


def _std_float(values: torch.Tensor) -> float:
    return float(values.detach().float().std(unbiased=False).cpu())


def _action_telemetry(action: torch.Tensor, cfg=None) -> dict[str, float | bool]:
    action = action.detach().float()
    abs_action = action.abs()
    forward = action[..., 0]
    turn = action[..., 1]
    rho_max = float(getattr(getattr(cfg, "planner", None), "rho_max", 1.0))
    beta_max = float(getattr(getattr(cfg, "planner", None), "beta_max", 1.0))
    clipped_forward = forward.clamp(-1.0, 1.0)
    clipped_turn = turn.clamp(-1.0, 1.0)
    physical_rho = 0.5 * (clipped_forward + 1.0) * rho_max
    physical_beta = clipped_turn * beta_max
    telemetry = {
        "action_mean": _mean_float(action),
        "action_std": _std_float(action),
        "action_min": float(action.min().detach().cpu()),
        "action_max": float(action.max().detach().cpu()),
        "action_near_zero_fraction": _mean_float((abs_action < 0.05).float()),
        "action_saturation_fraction": _mean_float((abs_action > 0.95).float()),
        "action_near_zero": bool((abs_action < 0.05).float().mean() > 0.80),
        "action_saturated": bool((abs_action > 0.95).float().mean() > 0.20),
        "action_forward_std": _std_float(forward),
        "action_turn_std": _std_float(turn),
        "action_forward_low_saturation_fraction": _mean_float((forward <= -0.95).float()),
        "action_forward_high_saturation_fraction": _mean_float((forward >= 0.95).float()),
        "action_turn_left_saturation_fraction": _mean_float((turn <= -0.95).float()),
        "action_turn_right_saturation_fraction": _mean_float((turn >= 0.95).float()),
        "action_turn_abs_saturation_fraction": _mean_float((turn.abs() >= 0.95).float()),
        "physical_rho_std": _std_float(physical_rho),
        "physical_beta_std": _std_float(physical_beta),
        "physical_rho_max_config": rho_max,
        "physical_beta_max_config": beta_max,
        "physical_rho_low_fraction": _mean_float((physical_rho <= 0.05 * rho_max).float()),
        "physical_rho_high_fraction": _mean_float((physical_rho >= 0.95 * rho_max).float()),
        "physical_beta_abs_high_fraction": _mean_float((physical_beta.abs() >= 0.95 * beta_max).float()),
    }
    telemetry.update(_stats("action_forward", forward))
    telemetry.update(_stats("action_turn", turn))
    telemetry.update(_stats("physical_rho", physical_rho))
    telemetry.update(_stats("physical_beta", physical_beta))
    return telemetry


def _empty_done_counts() -> dict[str, int]:
    return {
        "success_done": 0,
        "timeout_done": 0,
        "collision_done": 0,
        "safety_done": 0,
        "other_done": 0,
    }


def _add_done_counts(counts: dict[str, int], done) -> None:
    success = done.success.detach().bool()
    timeout = done.timeout.detach().bool()
    collision = done.collision.detach().bool()
    out_of_bounds = done.out_of_bounds.detach().bool()
    known = success | timeout | collision | out_of_bounds
    counts["success_done"] += int(success.sum().cpu())
    counts["timeout_done"] += int(timeout.sum().cpu())
    counts["collision_done"] += int(collision.sum().cpu())
    counts["safety_done"] += int((collision | out_of_bounds).sum().cpu())
    counts["other_done"] += int((done.done.detach().bool() & ~known).sum().cpu())


def _reward_weight(cfg, component: str) -> float | None:
    if cfg is None:
        return None
    if component == "success_hold":
        return 1.0
    return float(getattr(cfg.reward_weights, component))


def _empty_reward_breakdown() -> dict[str, float | str | None]:
    metrics: dict[str, float | str | None] = {
        "progress_reward": None,
        "oracle_reward": None,
        "cohesion_pairwise_reward": None,
        "safety_penalty": None,
        "terrain_penalty": None,
        "terminal_success_reward": None,
        "reward_weighted_total": None,
        "reward_contribution_sum": None,
        "reward_positive_contribution_sum": None,
        "reward_negative_contribution_sum": None,
        "reward_abs_contribution_sum": None,
        "reward_dominant_positive_component": None,
        "reward_dominant_negative_component": None,
    }
    for component in REWARD_COMPONENTS:
        metrics[f"reward_raw_{component}"] = None
        metrics[f"reward_weight_{component}"] = None
        metrics[f"reward_contribution_{component}"] = None
        metrics[f"reward_abs_share_{component}"] = None
    return metrics


def _finalize_reward_summary(metrics: dict[str, float | str | None]) -> dict[str, float | str | None]:
    contributions = {
        component: metrics.get(f"reward_contribution_{component}")
        for component in REWARD_COMPONENTS
    }
    numeric_contributions = {
        component: float(value)
        for component, value in contributions.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    positive = {key: value for key, value in numeric_contributions.items() if value > 0.0}
    negative = {key: value for key, value in numeric_contributions.items() if value < 0.0}
    abs_sum = sum(abs(value) for value in numeric_contributions.values())
    metrics["reward_contribution_sum"] = sum(numeric_contributions.values())
    metrics["reward_positive_contribution_sum"] = sum(positive.values()) if positive else 0.0
    metrics["reward_negative_contribution_sum"] = sum(negative.values()) if negative else 0.0
    metrics["reward_abs_contribution_sum"] = abs_sum
    metrics["reward_dominant_positive_component"] = max(positive, key=positive.get) if positive else None
    metrics["reward_dominant_negative_component"] = min(negative, key=negative.get) if negative else None
    for component, value in numeric_contributions.items():
        metrics[f"reward_abs_share_{component}"] = abs(value) / abs_sum if abs_sum > 0.0 else 0.0
    return metrics


def _reward_breakdown(info: dict[str, Any], cfg=None) -> dict[str, float | str | None]:
    terms = info.get("reward_terms")
    if terms is None:
        return _empty_reward_breakdown()

    raw_values = {component: _mean_float(getattr(terms, component)) for component in REWARD_COMPONENTS}
    weights = {component: _reward_weight(cfg, component) for component in REWARD_COMPONENTS}
    contributions = {
        component: raw_values[component] * weights[component]
        if weights[component] is not None
        else None
        for component in REWARD_COMPONENTS
    }
    numeric_contributions = {
        component: value
        for component, value in contributions.items()
        if value is not None
    }
    positive = {key: value for key, value in numeric_contributions.items() if value > 0.0}
    negative = {key: value for key, value in numeric_contributions.items() if value < 0.0}
    abs_sum = sum(abs(value) for value in numeric_contributions.values())
    metrics = _empty_reward_breakdown()
    # The current reward implementation has gather/progress shaping, but no distinct pairwise
    # cohesion term beyond gather shaping, so cohesion_pairwise_reward stays intentionally null.
    metrics.update(
        {
            "progress_reward": raw_values["gather"],
            "oracle_reward": raw_values["oracle"],
            "safety_penalty": raw_values["safety"],
            "terrain_penalty": raw_values["terrain"],
            "terminal_success_reward": raw_values["terminal"],
            "reward_weighted_total": _mean_float(terms.total),
            "reward_contribution_sum": sum(numeric_contributions.values()),
            "reward_positive_contribution_sum": sum(positive.values()) if positive else 0.0,
            "reward_negative_contribution_sum": sum(negative.values()) if negative else 0.0,
            "reward_abs_contribution_sum": abs_sum,
            "reward_dominant_positive_component": max(positive, key=positive.get) if positive else None,
            "reward_dominant_negative_component": min(negative, key=negative.get) if negative else None,
        }
    )
    for component in REWARD_COMPONENTS:
        contribution = contributions[component]
        metrics[f"reward_raw_{component}"] = raw_values[component]
        metrics[f"reward_weight_{component}"] = weights[component]
        metrics[f"reward_contribution_{component}"] = contribution
        metrics[f"reward_abs_share_{component}"] = 0.0
    return _finalize_reward_summary(metrics)


def _accumulate_numeric_dict(
    sums: dict[str, float],
    counts: dict[str, int],
    metrics: dict[str, Any],
) -> None:
    for key, value in metrics.items():
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1


def _averaged_numeric_dict(sums: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    return {
        key: sums[key] / counts[key]
        for key in sums
        if counts.get(key, 0) > 0
    }


def _accumulate_numeric_metrics(telemetry_state: dict, group: str, metrics: dict[str, Any]) -> None:
    sums = telemetry_state.setdefault(f"{group}_metric_sums", {})
    counts = telemetry_state.setdefault(f"{group}_metric_counts", {})
    _accumulate_numeric_dict(sums, counts, metrics)


def _snapshot_numeric_metrics(telemetry_state: dict, group: str, target: str) -> None:
    sums = telemetry_state.setdefault(f"{group}_metric_sums", {})
    counts = telemetry_state.setdefault(f"{group}_metric_counts", {})
    averages = _averaged_numeric_dict(sums, counts)
    if averages:
        telemetry_state[target] = averages
    sums.clear()
    counts.clear()


def _tensor_or_none(value) -> torch.Tensor | None:
    return value if isinstance(value, torch.Tensor) else None


def finite_or_raise(name: str, value) -> None:
    tensor = _tensor_or_none(value)
    if tensor is not None:
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite values detected in {name}.")
        return
    if isinstance(value, dict):
        for item in value.values():
            finite_or_raise(name, item)


def install_nan_checks(env: MultiRoverGatheringSKRLEnv, telemetry_state: dict) -> None:
    actor_obs, critic_state = env.core.get_observations()
    finite_or_raise("actor_observation", actor_obs)
    finite_or_raise("critic_state", critic_state)

    original_step = env.step

    def checked_step(actions):
        finite_or_raise("action", actions)
        action_tensor = torch.stack([actions[agent] for agent in env.possible_agents], dim=1)
        action_metrics = _action_telemetry(action_tensor, env.cfg)
        telemetry_state["action"] = action_metrics
        _accumulate_numeric_metrics(telemetry_state, "action", action_metrics)
        observations, rewards, terminated, truncated, info = original_step(actions)
        finite_or_raise("actor_observation", observations)
        finite_or_raise("critic_state", env.state())
        finite_or_raise("reward", rewards)
        telemetry_state["step"] = int(telemetry_state.get("step", 0)) + 1
        reward_values = [
            reward.detach().float().mean()
            for reward in rewards.values()
            if isinstance(reward, torch.Tensor)
        ]
        telemetry_state["mean_reward"] = (
            float(torch.stack(reward_values).mean().detach().cpu())
            if reward_values
            else None
        )
        reward_metrics = _reward_breakdown(info, env.cfg)
        telemetry_state["reward_breakdown"] = reward_metrics
        _accumulate_numeric_metrics(telemetry_state, "reward", reward_metrics)
        done = info.get("done")
        if done is not None:
            _add_done_counts(telemetry_state["done_counts"], done)
        writer = telemetry_state.get("writer")
        interval = int(telemetry_state.get("interval", 0))
        if writer is not None and interval > 0 and telemetry_state["step"] % interval == 0:
            writer(telemetry_state["step"])
        return observations, rewards, terminated, truncated, info

    env.step = checked_step


def build_training_telemetry(
    env: MultiRoverGatheringSKRLEnv,
    *,
    timesteps: int,
    wall_time_s: float,
    checkpoint_path: Path,
    training_semantics: str,
    telemetry_state: dict,
    run_id: str,
    phase: str,
    random_baseline: dict | None = None,
    post_training_eval: dict | None = None,
    peak_cuda_memory_mb: float | None = None,
) -> dict:
    metrics = compute_team_metrics(env.core.positions, env.core.velocities_xy)
    mean_oracle_distance = compute_mean_oracle_distance(env.core.positions, env.core.oracle_point)
    success_gates = compute_success_gates(metrics, env.core.velocities_xy, env.cfg.success_thresholds)
    pairwise = metrics.mean_pairwise_distance
    oracle = mean_oracle_distance
    threshold = float(env.cfg.success_thresholds.dmax)
    telemetry = {
        "run_id": run_id,
        "phase": phase,
        "timesteps": timesteps,
        "wall_time_s": wall_time_s,
        "device": str(env.device),
        "cuda_available": torch.cuda.is_available(),
        "mean_reward": telemetry_state.get("mean_reward"),
        "episode_length": float(env.core.step_count.float().mean().detach().cpu()),
        "mean_pairwise_distance": float(metrics.mean_pairwise_distance.mean().detach().cpu()),
        "mean_oracle_distance": float(mean_oracle_distance.mean().detach().cpu()),
        "success_rate": float(success_gates.instant_success.float().mean().detach().cpu()),
        "nan_flag": False,
        "checkpoint_path": str(checkpoint_path),
        "training_semantics": training_semantics,
        "observation_schema_version": env.cfg.observation.schema_version,
        "success_threshold": {
            "dmax": float(env.cfg.success_thresholds.dmax),
            "dispersion": float(env.cfg.success_thresholds.dispersion),
            "speed": float(env.cfg.success_thresholds.speed),
            "hold_steps": int(env.cfg.success_thresholds.hold_steps),
        },
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
    }
    reward_metrics = dict(telemetry_state.get("reward_breakdown", _reward_breakdown({}, env.cfg)))
    reward_metrics.update(telemetry_state.get("reward_window", {}))
    reward_metrics = _finalize_reward_summary(reward_metrics)
    action_metrics = dict(telemetry_state.get("action", {}))
    action_metrics.update(telemetry_state.get("action_window", {}))
    telemetry.update(reward_metrics)
    telemetry.update(action_metrics)
    telemetry.update(telemetry_state.get("done_counts", _empty_done_counts()))
    telemetry.update(_stats("final_pairwise_distance", pairwise))
    telemetry.update(_stats("final_oracle_distance", oracle))
    telemetry.update(_distance_threshold_ratios("final_pairwise_distance", pairwise, threshold))
    telemetry.update(_distance_threshold_ratios("final_oracle_distance", oracle, threshold))
    if random_baseline is not None:
        telemetry["random_baseline"] = random_baseline
        telemetry.update(random_baseline)
    if post_training_eval is not None:
        telemetry["post_training_eval"] = post_training_eval
        telemetry.update(post_training_eval)
    return telemetry


def append_metrics_jsonl(output_dir: Path, metrics: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metrics, sort_keys=True) + "\n")
    return metrics_path


def _split_action(env: MultiRoverGatheringSKRLEnv, action: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        agent: action[:, index, :]
        for index, agent in enumerate(env.possible_agents)
    }


def _policy_actions(
    env: MultiRoverGatheringSKRLEnv,
    policy: Model,
    observations: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    actions = {}
    with torch.no_grad():
        for agent in env.possible_agents:
            mean, _ = policy.compute({"observations": observations[agent]}, role="policy")
            actions[agent] = mean.clamp(-1.0, 1.0)
    return actions


def evaluate_policy_signal(
    env: MultiRoverGatheringSKRLEnv,
    *,
    mode: str,
    policy: Model | None = None,
    max_steps: int | None = None,
) -> dict[str, float | int | None]:
    if mode not in {"random", "policy"}:
        raise ValueError(f"Unsupported evaluation mode: {mode}")
    if mode == "policy" and policy is None:
        raise ValueError("Policy evaluation requires a policy model.")

    observations, _ = env.reset(seed=env.cfg.seed)
    steps = max_steps or env.cfg.simulation.max_episode_steps
    reward_items = []
    action_sums: dict[str, float] = {}
    action_counts: dict[str, int] = {}
    reward_sums: dict[str, float] = {}
    reward_counts: dict[str, int] = {}
    success_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    first_done_step = torch.full((env.num_envs,), steps, dtype=torch.float32, device=env.device)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for step in range(1, steps + 1):
        if mode == "random":
            action_tensor = env.core.random_actions()
            actions = _split_action(env, action_tensor)
        else:
            actions = _policy_actions(env, policy, observations)
            action_tensor = torch.stack([actions[agent] for agent in env.possible_agents], dim=1)
        finite_or_raise("action", actions)
        _accumulate_numeric_dict(action_sums, action_counts, _action_telemetry(action_tensor, env.cfg))
        observations, rewards, terminated, truncated, info = env.step(actions)
        finite_or_raise("actor_observation", observations)
        finite_or_raise("reward", rewards)
        _accumulate_numeric_dict(reward_sums, reward_counts, _reward_breakdown(info, env.cfg))
        reward_items.extend(
            reward.detach().float().mean()
            for reward in rewards.values()
            if isinstance(reward, torch.Tensor)
        )
        done = info.get("done")
        if done is not None:
            done_now = done.done.detach().bool()
            success_seen |= done.success.detach().bool()
            first_done_step = torch.where(
                active & done_now,
                torch.full_like(first_done_step, float(step)),
                first_done_step,
            )
            active &= ~done_now

    metrics = compute_team_metrics(env.core.positions, env.core.velocities_xy)
    oracle_distance = compute_mean_oracle_distance(env.core.positions, env.core.oracle_point)
    mean_reward = (
        float(torch.stack(reward_items).mean().detach().cpu())
        if reward_items
        else None
    )
    prefix = "random" if mode == "random" else "eval"
    result = {
        f"{prefix}_mean_reward": mean_reward,
        f"{prefix}_mean_pairwise_distance": _mean_float(metrics.mean_pairwise_distance),
        f"{prefix}_mean_oracle_distance": _mean_float(oracle_distance),
        f"{prefix}_success_rate": _mean_float(success_seen.float()),
        f"{prefix}_episode_length": _mean_float(first_done_step),
    }
    result.update(
        {
            f"{prefix}_{key}": value
            for key, value in _averaged_numeric_dict(action_sums, action_counts).items()
        }
    )
    result.update(
        {
            f"{prefix}_{key}": value
            for key, value in _averaged_numeric_dict(reward_sums, reward_counts).items()
        }
    )
    return result


def skrl_mappo_checkpoint_payload(
    models: dict[str, dict[str, Model]],
    possible_agents: list[str],
    *,
    raw_cfg: dict,
    shared_actor: bool,
    centralized_critic: bool,
    shared_value: bool,
    timesteps: int,
    training_semantics: str = DEFAULT_TRAINING_SEMANTICS,
    observation_schema_version: str | None = None,
    device: str | None = None,
    checkpoint_path: str | None = None,
) -> dict:
    payload = {
        agent_id: {
            "policy": models[agent_id]["policy"].state_dict(),
            "value": models[agent_id]["value"].state_dict(),
        }
        for agent_id in possible_agents
    }
    experiment = raw_cfg.get("experiment", {}) if isinstance(raw_cfg.get("experiment", {}), dict) else {}
    algorithm = raw_cfg.get("algorithm", {}) if isinstance(raw_cfg.get("algorithm", {}), dict) else {}
    payload["metadata"] = {
        "training_semantics": training_semantics,
        "backend": "skrl.mappo",
        "experiment_name": experiment.get("name"),
        "algorithm_mode": algorithm.get("mode"),
        "observation_schema_version": observation_schema_version,
        "shared_actor": shared_actor,
        "centralized_critic": centralized_critic,
        "shared_value": shared_value,
        "timesteps": timesteps,
        "device": device,
        "checkpoint_path": checkpoint_path,
    }
    payload["cfg"] = raw_cfg
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp_001_minimal.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timesteps", type=int, default=128)
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    exp = raw_cfg.get("experiment", {})
    algo = raw_cfg.get("algorithm", {})
    cfg = cfg_from_experiment(args.config)
    cfg.simulation.device = args.device
    requested_device = torch.device(cfg.simulation.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this stage.")

    env = MultiRoverGatheringSKRLEnv(cfg)
    experiment_name = str(exp.get("name", Path(args.config).stem))
    run_id = f"{experiment_name}_{int(time.time())}"
    telemetry_dir = ensure_output_dir(Path("outputs/runs") / experiment_name)
    checkpoint_dir = ensure_output_dir(exp.get("checkpoint_dir", "outputs/checkpoints"))
    checkpoint_path = checkpoint_dir / resolve_checkpoint_name(raw_cfg, args.config)
    telemetry_state = {
        "mean_reward": None,
        "step": 0,
        "interval": 1000,
        "done_counts": _empty_done_counts(),
    }
    wrapped_env = wrap_env(env, wrapper="isaaclab-multi-agent", verbose=False)
    possible_agents = env.possible_agents
    empty_kwargs = {uid: {} for uid in possible_agents}
    shared_actor = parse_bool_config(algo.get("shared_actor"), default=True)
    centralized_critic = parse_bool_config(algo.get("centralized_critic"), default=True)
    shared_value = parse_bool_config(algo.get("shared_value"), default=True)
    training_semantics = resolve_training_semantics(raw_cfg)
    random_baseline = evaluate_policy_signal(
        MultiRoverGatheringSKRLEnv(cfg),
        mode="random",
    )

    models = build_skrl_mappo_models(
        env,
        shared_actor=shared_actor,
        centralized_critic=centralized_critic,
        shared_value=shared_value,
    )
    memories = build_skrl_mappo_memories(env, rollout_steps=int(exp.get("rollout_steps", 32)))

    agent = MAPPO(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        device=env.device,
        cfg={
            "rollouts": int(exp.get("rollout_steps", 32)),
            "learning_epochs": 1,
            "mini_batches": 1,
            "discount_factor": float(algo.get("gamma", 0.99)),
            "learning_rate": float(algo.get("learning_rate", 5.0e-4)),
            "learning_rate_scheduler_kwargs": empty_kwargs,
            "observation_preprocessor_kwargs": empty_kwargs,
            "state_preprocessor_kwargs": empty_kwargs,
            "value_preprocessor_kwargs": empty_kwargs,
            "entropy_loss_scale": 0.01,
            "value_loss_scale": 0.5,
            "random_timesteps": 0,
            "learning_starts": 0,
        },
    )

    def write_interval_telemetry(step: int) -> None:
        _snapshot_numeric_metrics(telemetry_state, "action", "action_window")
        _snapshot_numeric_metrics(telemetry_state, "reward", "reward_window")
        append_metrics_jsonl(
            telemetry_dir,
            build_training_telemetry(
                env,
                timesteps=step,
                wall_time_s=time.perf_counter() - telemetry_state["training_start_time"],
                checkpoint_path=checkpoint_path,
                training_semantics=training_semantics,
                telemetry_state=telemetry_state,
                run_id=run_id,
                phase="train",
                random_baseline=random_baseline,
            ),
        )

    telemetry_state["writer"] = write_interval_telemetry
    install_nan_checks(env, telemetry_state)
    trainer = SequentialTrainer(
        env=wrapped_env,
        agents=agent,
        cfg={
            "timesteps": args.timesteps,
            "headless": True,
            "disable_progressbar": True,
            "close_environment_at_exit": False,
        },
    )
    if env.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(env.device)
    start_time = time.perf_counter()
    telemetry_state["training_start_time"] = start_time
    trainer.train()
    wall_time_s = time.perf_counter() - start_time
    peak_cuda_memory_mb = (
        torch.cuda.max_memory_allocated(env.device) / (1024.0 * 1024.0)
        if env.device.type == "cuda"
        else None
    )
    post_training_eval = evaluate_policy_signal(
        MultiRoverGatheringSKRLEnv(cfg),
        mode="policy",
        policy=models[possible_agents[0]]["policy"],
    )
    _snapshot_numeric_metrics(telemetry_state, "action", "action_window")
    _snapshot_numeric_metrics(telemetry_state, "reward", "reward_window")

    torch.save(
        skrl_mappo_checkpoint_payload(
            models,
            possible_agents,
            raw_cfg=raw_cfg,
            training_semantics=training_semantics,
            observation_schema_version=cfg.observation.schema_version,
            shared_actor=shared_actor,
            centralized_critic=centralized_critic,
            shared_value=shared_value,
            timesteps=args.timesteps,
            device=str(env.device),
            checkpoint_path=str(checkpoint_path),
        ),
        checkpoint_path,
    )
    metrics_path = append_metrics_jsonl(
        telemetry_dir,
        build_training_telemetry(
            env,
            timesteps=args.timesteps,
            wall_time_s=wall_time_s,
            checkpoint_path=checkpoint_path,
            training_semantics=training_semantics,
            telemetry_state=telemetry_state,
            run_id=run_id,
            phase="final",
            random_baseline=random_baseline,
            post_training_eval=post_training_eval,
            peak_cuda_memory_mb=peak_cuda_memory_mb,
        ),
    )
    print(
        yaml.safe_dump(
            {
                "status": "ok",
                "backend": "skrl.mappo",
                "training_semantics": training_semantics,
                "shared_actor": shared_actor,
                "centralized_critic": centralized_critic,
                "shared_value": shared_value,
                "timesteps": args.timesteps,
                "device": str(env.device),
                "checkpoint_path": str(checkpoint_path),
                "metrics_path": str(metrics_path),
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
