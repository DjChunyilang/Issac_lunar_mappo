#!/usr/bin/env python
"""Run a short SKRL MAPPO smoke job on the first-stage proxy environment."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from _common import ROOT, cfg_from_experiment, ensure_output_dir, load_yaml
from _skrl_metadata import (
    DEFAULT_TRAINING_SEMANTICS,
    resolve_checkpoint_name,
    resolve_training_semantics,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringSKRLEnv
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.oracle import (
    compute_geometric_median,
    compute_mean_oracle_distance,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import query_terrain_features
from lunar_rover_tasks.tasks.multi_rover_gathering.termination import compute_success_gates

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.trainers.torch import SequentialTrainer

from shared_policy_mappo import SharedPolicyMAPPO


TRAINING_SEMANTICS = DEFAULT_TRAINING_SEMANTICS


class SKRLPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, initial_log_std: float = -0.5):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True, reduction="sum")
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), float(initial_log_std))
        )

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


def checkpoint_teacher_metadata(
    *,
    bc_updates: int,
    teacher_mode: str | None,
    teacher_stop_radius: float | None,
    teacher_max_rho: float | None,
) -> dict:
    enabled = bc_updates > 0
    return {
        "teacher_mode": teacher_mode if enabled else None,
        "teacher_stop_radius": teacher_stop_radius if enabled else None,
        "teacher_max_rho": teacher_max_rho if enabled else None,
    }


def _nearest_distances(positions: torch.Tensor) -> torch.Tensor:
    pairwise = torch.cdist(positions[..., :2], positions[..., :2])
    n_agents = positions.shape[1]
    eye = torch.eye(n_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
    return pairwise.masked_fill(eye, float("inf")).amin(dim=-1)


def scripted_gather_action(
    env,
    *,
    stop_radius: float = 0.45,
    slow_distance: float = 0.45,
    max_rho: float | None = None,
    visible_local: bool = False,
    terrain_scale: bool = False,
) -> torch.Tensor:
    """Return a safety-aware local subgoal action for BC warm-up."""
    positions_xy = env.positions[..., :2]
    if visible_local:
        pairwise = torch.cdist(positions_xy, positions_xy)
        eye = torch.eye(env.n_agents, dtype=torch.bool, device=env.device).unsqueeze(0)
        visible = (pairwise <= env.cfg.observation.communication_radius) & ~eye
        visible_f = visible.to(dtype=positions_xy.dtype)
        local_sum = torch.einsum("eij,ejd->eid", visible_f, positions_xy) + positions_xy
        local_count = visible_f.sum(dim=-1, keepdim=True) + 1.0
        centroid_xy = local_sum / local_count
        has_neighbor = visible.any(dim=-1, keepdim=True)
    else:
        centroid_xy = positions_xy.mean(dim=1, keepdim=True).expand_as(positions_xy)
        has_neighbor = torch.ones_like(positions_xy[..., :1], dtype=torch.bool)
    rel = positions_xy - centroid_xy
    dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
    fallback_angles = torch.linspace(
        0.0,
        2.0 * torch.pi,
        env.n_agents + 1,
        device=env.device,
    )[:-1]
    fallback = torch.stack((torch.cos(fallback_angles), torch.sin(fallback_angles)), dim=-1)
    unit = torch.where(
        dist > 1.0e-6,
        rel / dist.clamp_min(1.0e-6),
        fallback[None, :, :],
    )
    target_xy = centroid_xy + unit * stop_radius
    world_delta = torch.where(
        (dist > stop_radius) & has_neighbor,
        target_xy - positions_xy,
        torch.zeros_like(positions_xy),
    )
    if slow_distance > 0.0:
        nearest = _nearest_distances(env.positions).unsqueeze(-1)
        scale = torch.clamp(
            (nearest - env.cfg.safety.collision_distance) / slow_distance,
            0.0,
            1.0,
        )
        world_delta = world_delta * scale
    rho_limit = env.cfg.planner.rho_max if max_rho is None else min(
        float(max_rho),
        float(env.cfg.planner.rho_max),
    )
    world_norm = torch.linalg.vector_norm(world_delta, dim=-1, keepdim=True)
    world_delta = world_delta * torch.clamp(
        rho_limit / world_norm.clamp_min(1.0e-6),
        max=1.0,
    )
    if terrain_scale and env._terrain_dynamics_enabled:
        candidate_features = query_terrain_features(
            positions_xy + world_delta,
            env.cfg.terrain,
            env.terrain_runtime,
        )
        world_delta = world_delta * candidate_features[..., 4:5]

    cos_yaw = torch.cos(env.yaws)
    sin_yaw = torch.sin(env.yaws)
    local_x = cos_yaw * world_delta[..., 0] + sin_yaw * world_delta[..., 1]
    local_y = -sin_yaw * world_delta[..., 0] + cos_yaw * world_delta[..., 1]
    rho = torch.linalg.vector_norm(torch.stack((local_x, local_y), dim=-1), dim=-1)
    rho = rho.clamp(0.0, env.cfg.planner.rho_max)
    beta = torch.atan2(local_y, local_x).clamp(
        -env.cfg.planner.beta_max,
        env.cfg.planner.beta_max,
    )
    return torch.stack(
        (
            2.0 * rho / env.cfg.planner.rho_max - 1.0,
            beta / env.cfg.planner.beta_max,
        ),
        dim=-1,
    )


def _randomize_bc_state(
    env,
    *,
    visible_local: bool = False,
    yaw_noise_degrees: float | None = None,
    min_nearest_distance: float | None = None,
) -> None:
    env.randomize_terrain()
    base_angles = torch.linspace(
        0.0,
        2.0 * torch.pi,
        env.n_agents + 1,
        device=env.device,
    )[:-1]
    base = torch.stack((torch.cos(base_angles), torch.sin(base_angles)), dim=-1)
    pending = torch.arange(env.num_envs, device=env.device)
    for _ in range(32):
        count = int(pending.numel())
        if count == 0:
            break
        radius = torch.empty(count, 1, 1, device=env.device).uniform_(
            0.25,
            4.0,
            generator=env.generator,
        )
        jitter = 0.35 * torch.randn(
            count,
            env.n_agents,
            2,
            generator=env.generator,
            device=env.device,
        )
        centers = torch.empty(count, 1, 2, device=env.device).uniform_(
            -1.0,
            1.0,
            generator=env.generator,
        )
        env.positions[pending, :, :2] = centers + radius * base[None, :, :] + jitter
        if min_nearest_distance is None:
            pending = pending[:0]
        else:
            nearest = _nearest_distances(env.positions[pending])
            pending = pending[(nearest < min_nearest_distance).any(dim=-1)]
    if pending.numel() > 0:
        raise RuntimeError("Unable to sample collision-free BC states.")
    if env._terrain_dynamics_enabled:
        terrain_features = query_terrain_features(
            env.positions[..., :2],
            env.cfg.terrain,
            env.terrain_runtime,
        )
        env.positions[..., 2] = terrain_features[..., 0]
        env.last_terrain_features = terrain_features
    else:
        env.positions[..., 2] = 0.0
        env.last_terrain_features.zero_()
    env.last_terrain_speed_scale.fill_(1.0)
    env.last_height_delta.zero_()
    if visible_local and yaw_noise_degrees is not None:
        pairwise = torch.cdist(env.positions[..., :2], env.positions[..., :2])
        eye = torch.eye(env.n_agents, dtype=torch.bool, device=env.device).unsqueeze(0)
        visible = (pairwise <= env.cfg.observation.communication_radius) & ~eye
        visible_f = visible.to(dtype=env.positions.dtype)
        local_sum = (
            torch.einsum("eij,ejd->eid", visible_f, env.positions[..., :2])
            + env.positions[..., :2]
        )
        local_center = local_sum / (visible_f.sum(dim=-1, keepdim=True) + 1.0)
        delta = local_center - env.positions[..., :2]
        desired = torch.atan2(delta[..., 1], delta[..., 0])
        noise = torch.empty_like(desired).uniform_(
            -math.radians(yaw_noise_degrees),
            math.radians(yaw_noise_degrees),
            generator=env.generator,
        )
        env.yaws.copy_(desired + noise)
    else:
        env.yaws.uniform_(-torch.pi, torch.pi, generator=env.generator)
    env.velocities_xy.zero_()
    env.angular_velocities.zero_()
    env.previous_physical_action.zero_()
    env.step_count.zero_()
    env.success_hold_count.zero_()
    env.oracle_point.copy_(compute_geometric_median(env.positions))
    env.prev_mean_oracle_distance.copy_(
        compute_mean_oracle_distance(env.positions, env.oracle_point)
    )


def run_skrl_behavior_cloning(
    policy: Model,
    cfg,
    *,
    updates: int,
    batch_size: int,
    learning_rate: float,
    teacher_stop_radius: float = 0.45,
    teacher_slow_distance: float = 0.45,
    teacher_max_rho: float | None = None,
    teacher_mode: str = "global_centroid",
    teacher_terrain_scale: bool = False,
    bc_yaw_noise_degrees: float | None = None,
    bc_min_nearest_distance: float | None = None,
) -> list[dict[str, float | int | str]]:
    """Warm-start the shared SKRL actor; MAPPO itself remains teacher-loss free."""
    if updates <= 0:
        return []
    bc_cfg = copy.deepcopy(cfg)
    bc_cfg.simulation.num_envs = max(
        bc_cfg.simulation.num_envs,
        math.ceil(batch_size / bc_cfg.task.n_agents),
    )
    env = MultiRoverGatheringSKRLEnv(bc_cfg).core
    policy.to(env.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    samples_per_snapshot = env.num_envs * env.n_agents
    snapshots_per_batch = max(1, math.ceil(batch_size / samples_per_snapshot))
    records: list[dict[str, float | int | str]] = []
    policy.train()

    for update in range(1, updates + 1):
        observations = []
        targets = []
        for _ in range(snapshots_per_batch):
            _randomize_bc_state(
                env,
                visible_local=teacher_mode == "visible_local_centroid",
                yaw_noise_degrees=bc_yaw_noise_degrees,
                min_nearest_distance=bc_min_nearest_distance,
            )
            actor_obs, _ = env.get_observations()
            target = scripted_gather_action(
                env,
                stop_radius=teacher_stop_radius,
                slow_distance=teacher_slow_distance,
                max_rho=teacher_max_rho,
                visible_local=teacher_mode == "visible_local_centroid",
                terrain_scale=teacher_terrain_scale,
            )
            observations.append(actor_obs.reshape(-1, actor_obs.shape[-1]).detach())
            targets.append(target.reshape(-1, target.shape[-1]).detach())
        obs = torch.cat(observations, dim=0)[:batch_size]
        target = torch.cat(targets, dim=0)[:batch_size]
        prediction, _ = policy.compute({"observations": obs}, role="policy")
        loss = F.mse_loss(prediction, target)
        finite_or_raise("bc_loss", loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        records.append(
            {
                "phase": "bc",
                "update": update,
                "bc_loss": float(loss.detach().cpu()),
            }
        )
    return records


def build_mappo_config(
    algorithm: dict,
    experiment: dict,
    empty_kwargs: dict[str, dict],
    *,
    run_dir: Path | None = None,
) -> dict:
    """Map experiment YAML parameters to native SKRL MAPPO configuration."""
    cfg = {
        "rollouts": int(experiment.get("rollout_steps", 32)),
        "learning_epochs": int(algorithm.get("ppo_epochs", 1)),
        "mini_batches": int(algorithm.get("mini_batches", 1)),
        "discount_factor": float(algorithm.get("gamma", 0.99)),
        "gae_lambda": float(algorithm.get("gae_lambda", 0.95)),
        "learning_rate": float(algorithm.get("learning_rate", 5.0e-4)),
        "learning_rate_scheduler_kwargs": empty_kwargs,
        "observation_preprocessor_kwargs": empty_kwargs,
        "state_preprocessor_kwargs": empty_kwargs,
        "value_preprocessor_kwargs": empty_kwargs,
        "grad_norm_clip": float(algorithm.get("max_grad_norm", 0.5)),
        "ratio_clip": float(algorithm.get("clip_epsilon", 0.2)),
        "value_clip": float(algorithm.get("value_clip", algorithm.get("clip_epsilon", 0.2))),
        "entropy_loss_scale": float(
            algorithm.get("entropy_loss_scale", algorithm.get("entropy_coef_start", 0.01))
        ),
        "value_loss_scale": float(algorithm.get("value_loss_coef", 0.5)),
        "random_timesteps": 0,
        "learning_starts": 0,
    }
    if run_dir is not None:
        cfg["experiment"] = {
            "directory": str(run_dir),
            "experiment_name": "tensorboard",
            "write_interval": int(experiment.get("tensorboard_interval", 128)),
            "checkpoint_interval": 0,
        }
    return cfg


STRICT_THRESHOLDS = {
    "dmax_reduction_ratio": 0.20,
    "success_rate": 0.90,
    "collision_rate": 0.02,
    "timeout_rate": 0.0,
}

PURE_RL_LONG_THRESHOLDS = {
    "dmax_reduction_ratio": 0.45,
    "success_rate": 0.05,
    "collision_rate": 0.03,
}
SAFE_PROGRESS_LONG_THRESHOLDS = {
    "dmax_reduction_ratio": 0.30,
    "success_rate": 0.20,
}
BALANCED_PROGRESS_LONG_THRESHOLDS = {
    "dmax_reduction_ratio": 0.30,
    "success_rate": 0.05,
}


def proxy_acceptance(metrics: dict, thresholds: dict | None = None) -> dict:
    thresholds = thresholds or STRICT_THRESHOLDS
    checks = {
        "dmax_reduction_ratio": metrics["dmax_reduction_ratio"] <= thresholds["dmax_reduction_ratio"],
        "success_rate": metrics["success_rate"] >= thresholds["success_rate"],
        "collision_rate": metrics["collision_rate"] <= thresholds["collision_rate"],
        "timeout_rate": metrics["timeout_rate"] <= thresholds["timeout_rate"],
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": thresholds}


def pure_rl_long_acceptance(
    metrics: dict,
    thresholds: dict | None = None,
) -> dict:
    thresholds = thresholds or PURE_RL_LONG_THRESHOLDS
    checks = {
        "dmax_reduction_ratio": metrics["dmax_reduction_ratio"]
        <= thresholds["dmax_reduction_ratio"],
        "success_rate": metrics["success_rate"] >= thresholds["success_rate"],
        "collision_rate": metrics["collision_rate"] <= thresholds["collision_rate"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "timeout_rate_recorded_only": metrics.get("timeout_rate"),
    }


def checkpoint_rank(
    metrics: dict,
    thresholds: dict | None = None,
) -> tuple[bool, float, float, float, float, float]:
    thresholds = thresholds or STRICT_THRESHOLDS
    strict_pass = proxy_acceptance(metrics, thresholds)["passed"]
    violation = (
        max(0.0, float(metrics["dmax_reduction_ratio"]) / thresholds["dmax_reduction_ratio"] - 1.0)
        + max(0.0, 1.0 - float(metrics["success_rate"]) / thresholds["success_rate"])
        + max(0.0, float(metrics["collision_rate"]) / thresholds["collision_rate"] - 1.0)
        + (
            0.0
            if thresholds["timeout_rate"] == 0.0 and float(metrics["timeout_rate"]) == 0.0
            else float(metrics["timeout_rate"]) / max(thresholds["timeout_rate"], 1.0e-6)
        )
    )
    return (
        not strict_pass,
        violation,
        float(metrics.get("collision_rate", float("inf"))),
        float(metrics.get("timeout_rate", float("inf"))),
        -float(metrics.get("success_rate", 0.0)),
        float(metrics.get("dmax_reduction_ratio", float("inf"))),
    )


def pure_rl_long_checkpoint_rank(
    metrics: dict,
    thresholds: dict | None = None,
) -> tuple[bool, float, float, float, float, int]:
    thresholds = thresholds or PURE_RL_LONG_THRESHOLDS
    trend_pass = pure_rl_long_acceptance(metrics, thresholds)["passed"]
    violation = (
        max(
            0.0,
            float(metrics["dmax_reduction_ratio"])
            / thresholds["dmax_reduction_ratio"]
            - 1.0,
        )
        + max(
            0.0,
            1.0 - float(metrics["success_rate"]) / thresholds["success_rate"],
        )
        + max(
            0.0,
            float(metrics["collision_rate"]) / thresholds["collision_rate"] - 1.0,
        )
    )
    return (
        not trend_pass,
        violation,
        float(metrics.get("collision_rate", float("inf"))),
        -float(metrics.get("success_rate", 0.0)),
        float(metrics.get("dmax_reduction_ratio", float("inf"))),
        -int(metrics.get("candidate_timestep", 0)),
    )


def safe_progress_long_checkpoint_rank(
    metrics: dict,
    thresholds: dict | None = None,
) -> tuple[int, float, float, float, float, float, int]:
    thresholds = thresholds or SAFE_PROGRESS_LONG_THRESHOLDS
    if proxy_acceptance(metrics, STRICT_THRESHOLDS)["passed"]:
        return (
            0,
            0.0,
            float(metrics.get("collision_rate", float("inf"))),
            float(metrics.get("timeout_rate", float("inf"))),
            -float(metrics.get("success_rate", 0.0)),
            float(metrics.get("dmax_reduction_ratio", float("inf"))),
            -int(metrics.get("candidate_timestep", 0)),
        )
    trend_pass = (
        float(metrics["success_rate"]) >= thresholds["success_rate"]
        and float(metrics["dmax_reduction_ratio"]) <= thresholds["dmax_reduction_ratio"]
    )
    if trend_pass:
        return (
            1,
            0.0,
            float(metrics.get("collision_rate", float("inf"))),
            float(metrics.get("timeout_rate", float("inf"))),
            -float(metrics.get("success_rate", 0.0)),
            float(metrics.get("dmax_reduction_ratio", float("inf"))),
            -int(metrics.get("candidate_timestep", 0)),
        )
    violation = (
        max(
            0.0,
            float(metrics["dmax_reduction_ratio"])
            / thresholds["dmax_reduction_ratio"]
            - 1.0,
        )
        + max(
            0.0,
            1.0 - float(metrics["success_rate"]) / thresholds["success_rate"],
        )
        + max(0.0, float(metrics.get("collision_rate", 0.0)) / 0.02 - 1.0)
        + float(metrics.get("timeout_rate", 0.0))
    )
    return (
        2,
        violation,
        float(metrics.get("collision_rate", float("inf"))),
        float(metrics.get("timeout_rate", float("inf"))),
        -float(metrics.get("success_rate", 0.0)),
        float(metrics.get("dmax_reduction_ratio", float("inf"))),
        -int(metrics.get("candidate_timestep", 0)),
    )


def balanced_progress_long_checkpoint_rank(
    metrics: dict,
    thresholds: dict | None = None,
) -> tuple[int, float, float, float, float, float, int]:
    thresholds = thresholds or BALANCED_PROGRESS_LONG_THRESHOLDS
    if proxy_acceptance(metrics, STRICT_THRESHOLDS)["passed"]:
        return (
            0,
            0.0,
            float(metrics.get("collision_rate", float("inf"))),
            float(metrics.get("timeout_rate", float("inf"))),
            -float(metrics.get("success_rate", 0.0)),
            float(metrics.get("dmax_reduction_ratio", float("inf"))),
            -int(metrics.get("candidate_timestep", 0)),
        )
    has_minimum_progress = float(metrics.get("success_rate", 0.0)) >= thresholds["success_rate"]
    if has_minimum_progress:
        return (
            1,
            max(
                0.0,
                float(metrics.get("dmax_reduction_ratio", float("inf")))
                / thresholds["dmax_reduction_ratio"]
                - 1.0,
            ),
            float(metrics.get("collision_rate", float("inf"))),
            float(metrics.get("timeout_rate", float("inf"))),
            -float(metrics.get("success_rate", 0.0)),
            float(metrics.get("dmax_reduction_ratio", float("inf"))),
            -int(metrics.get("candidate_timestep", 0)),
        )
    violation = (
        max(
            0.0,
            float(metrics.get("dmax_reduction_ratio", float("inf")))
            / thresholds["dmax_reduction_ratio"]
            - 1.0,
        )
        + max(
            0.0,
            1.0 - float(metrics.get("success_rate", 0.0)) / thresholds["success_rate"],
        )
        + max(0.0, float(metrics.get("collision_rate", 0.0)) / 0.02 - 1.0)
        + float(metrics.get("timeout_rate", 0.0))
    )
    return (
        2,
        violation,
        float(metrics.get("collision_rate", float("inf"))),
        float(metrics.get("timeout_rate", float("inf"))),
        -float(metrics.get("success_rate", 0.0)),
        float(metrics.get("dmax_reduction_ratio", float("inf"))),
        -int(metrics.get("candidate_timestep", 0)),
    )


def subgoal_filter_metadata(cfg) -> dict[str, Any]:
    filter_cfg = cfg.planner.subgoal_filter
    return {
        "enabled": bool(filter_cfg.enabled),
        "mode": str(filter_cfg.mode),
        "rho_scales": [float(value) for value in filter_cfg.rho_scales],
        "beta_offsets_deg": [float(value) for value in filter_cfg.beta_offsets_deg],
        "path_samples": int(filter_cfg.path_samples),
        "score_weights": {
            "intent_deviation": float(filter_cfg.intent_deviation_weight),
            "path_terrain_mean": float(filter_cfg.path_terrain_mean_weight),
            "path_terrain_max": float(filter_cfg.path_terrain_max_weight),
            "path_height_change": float(filter_cfg.path_height_change_weight),
            "subgoal_terrain": float(filter_cfg.subgoal_terrain_weight),
            "endpoint_near": float(filter_cfg.endpoint_near_weight),
            "endpoint_collision": float(filter_cfg.endpoint_collision_weight),
            "path_near": float(filter_cfg.path_near_weight),
            "path_collision": float(filter_cfg.path_collision_weight),
            "visible_neighbor_center": float(filter_cfg.visible_neighbor_center_weight),
        },
        "constraints": {
            "endpoint_safe_distance": float(filter_cfg.endpoint_safe_distance),
            "path_safe_distance": float(filter_cfg.path_safe_distance),
            "hard_endpoint_near_filter": bool(filter_cfg.hard_endpoint_near_filter),
            "hard_path_collision_filter": bool(filter_cfg.hard_path_collision_filter),
            "hard_center_progress_filter": bool(filter_cfg.hard_center_progress_filter),
            "center_progress_slack": float(filter_cfg.center_progress_slack),
            "hard_constraint_penalty": float(filter_cfg.hard_constraint_penalty),
            "safety_override_after_warmup": bool(
                filter_cfg.safety_override_after_warmup
            ),
        },
        "schedule": {
            "warmup_timesteps": int(filter_cfg.warmup_timesteps),
            "ramp_timesteps": int(filter_cfg.ramp_timesteps),
            "apply_probability_end": float(filter_cfg.apply_probability_end),
            "score_scale_start": float(filter_cfg.score_scale_start),
            "score_scale_end": float(filter_cfg.score_scale_end),
            "deterministic_improvement_margin": float(
                filter_cfg.deterministic_improvement_margin
            ),
        },
    }


def terrain_sanity_metrics(cfg, device: torch.device | str, samples_per_axis: int = 61) -> dict:
    extent = 0.5 * float(cfg.terrain.crater_field_size)
    axis = torch.linspace(-extent, extent, samples_per_axis, device=device)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    features = query_terrain_features(xy, cfg.terrain)
    directions = torch.linspace(0.0, 2.0 * torch.pi, 17, device=device)[:-1]
    direction_xy = torch.stack((directions.cos(), directions.sin()), dim=-1)
    directional_slope = (
        features[:, None, 1:3] * direction_xy[None, :, :]
    ).sum(dim=-1).abs()
    speed_scale = features[:, None, 4] * torch.exp(
        -directional_slope * float(cfg.terrain.slope_speed_scale)
    )
    speed_scale = speed_scale.clamp(
        min=float(cfg.terrain.min_speed_scale),
        max=1.0,
    )
    height = features[:, 0]
    return {
        "height_min": float(height.amin().cpu()),
        "height_max": float(height.amax().cpu()),
        "height_range": float((height.amax() - height.amin()).cpu()),
        "min_traversability": float(features[:, 4].amin().cpu()),
        "mean_traversability": float(features[:, 4].mean().cpu()),
        "mean_speed_scale": float(speed_scale.mean().cpu()),
    }


def candidate_eval_seed(training_seed: int, eval_seed_offset: int) -> int:
    return training_seed + eval_seed_offset


def final_eval_seed(training_seed: int, eval_seed_offset: int) -> int:
    return training_seed + eval_seed_offset + 10000


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
    initial_log_std: float = -0.5,
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
            initial_log_std,
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
                initial_log_std,
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
        path_terrain = info.get("path_terrain")
        if path_terrain is not None:
            path_metrics = {
                "path_terrain_risk_mean": float(
                    path_terrain["risk_mean"].detach().float().mean().cpu()
                ),
                "path_terrain_risk_max": float(
                    path_terrain["risk_max"].detach().float().amax().cpu()
                ),
                "path_height_change_mean": float(
                    path_terrain["height_change_mean"].detach().float().mean().cpu()
                ),
            }
            telemetry_state["path_terrain"] = path_metrics
            _accumulate_numeric_metrics(telemetry_state, "path_terrain", path_metrics)
        action_filter = info.get("action_filter")
        if action_filter is not None:
            filter_metrics = {
                "filter_candidate_count": int(action_filter.get("candidate_count", 0)),
                "filter_applied_fraction": float(
                    action_filter["applied"].detach().float().mean().cpu()
                ),
                "filter_raw_path_terrain_risk_mean": float(
                    action_filter["raw_path_terrain_risk_mean"].detach().float().mean().cpu()
                ),
                "filter_filtered_path_terrain_risk_mean": float(
                    action_filter["filtered_path_terrain_risk_mean"].detach().float().mean().cpu()
                ),
                "filter_path_terrain_risk_reduction_mean": float(
                    action_filter["path_terrain_risk_reduction"].detach().float().mean().cpu()
                ),
                "filter_subgoal_deviation_mean": float(
                    action_filter["subgoal_deviation"].detach().float().mean().cpu()
                ),
                "filter_suggested_subgoal_deviation_mean": float(
                    action_filter["suggested_subgoal_deviation"].detach().float().mean().cpu()
                ),
                "filter_endpoint_near_violation_mean": float(
                    action_filter["endpoint_near_violation"].detach().float().mean().cpu()
                ),
                "filter_endpoint_collision_violation_mean": float(
                    action_filter["endpoint_collision_violation"].detach().float().mean().cpu()
                ),
                "filter_endpoint_collision_violation_fraction": float(
                    (action_filter["endpoint_collision_violation"] > 0.0)
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                ),
                "filter_path_near_violation_mean": float(
                    action_filter["path_near_violation"].detach().float().mean().cpu()
                ),
                "filter_path_collision_violation_mean": float(
                    action_filter["path_collision_violation"].detach().float().mean().cpu()
                ),
                "filter_path_collision_violation_fraction": float(
                    (action_filter["path_collision_violation"] > 0.0)
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                ),
                "filter_raw_endpoint_near_violation_mean": float(
                    action_filter["raw_endpoint_near_violation"]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                ),
                "filter_raw_endpoint_collision_violation_mean": float(
                    action_filter["raw_endpoint_collision_violation"]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                ),
                "filter_raw_path_near_violation_mean": float(
                    action_filter["raw_path_near_violation"].detach().float().mean().cpu()
                ),
                "filter_raw_path_collision_violation_mean": float(
                    action_filter["raw_path_collision_violation"]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                ),
                "filter_candidate_feasible_fraction": float(
                    action_filter["candidate_feasible"].detach().float().mean().cpu()
                ),
                "filter_feasible_fraction": float(
                    action_filter.get("feasible_fraction", 0.0)
                ),
                "filter_safety_override_fraction": float(
                    action_filter.get("safety_override_fraction", 0.0)
                ),
                "filter_candidate_index_mean": float(
                    action_filter["candidate_index"].detach().float().mean().cpu()
                ),
                "filter_raw_score_mean": float(
                    action_filter["raw_score"].detach().float().mean().cpu()
                ),
                "filter_filtered_score_mean": float(
                    action_filter["filtered_score"].detach().float().mean().cpu()
                ),
                "filter_score_margin_mean": float(
                    action_filter["score_margin"].detach().float().mean().cpu()
                ),
                "filter_deterministic_applied_fraction": float(
                    action_filter["deterministic_applied"].detach().float().mean().cpu()
                ),
                "filter_schedule_progress_step": float(
                    action_filter.get("schedule_progress_step", 0)
                ),
                "filter_apply_probability": float(action_filter.get("apply_probability", 0.0)),
                "filter_score_scale": float(action_filter.get("score_scale", 0.0)),
            }
            histogram = action_filter.get("candidate_index_histogram")
            if isinstance(histogram, torch.Tensor):
                for index, fraction in enumerate(histogram.detach().float().cpu().tolist()):
                    filter_metrics[f"filter_candidate_index_{index}_fraction"] = float(fraction)
            telemetry_state["action_filter"] = filter_metrics
            _accumulate_numeric_metrics(telemetry_state, "action_filter", filter_metrics)
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
    training_diagnostics: dict | None = None,
    peak_cuda_memory_mb: float | None = None,
) -> dict:
    metrics = compute_team_metrics(env.core.positions, env.core.velocities_xy)
    mean_oracle_distance = compute_mean_oracle_distance(env.core.positions, env.core.oracle_point)
    success_gates = compute_success_gates(metrics, env.core.velocities_xy, env.cfg.success_thresholds)
    pairwise = metrics.mean_pairwise_distance
    oracle = mean_oracle_distance
    threshold = float(env.cfg.success_thresholds.dmax)
    nearest = metrics.nearest_neighbor_distance.amin(dim=-1)
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
        "final_nearest_neighbor_distance": float(nearest.mean().detach().cpu()),
        "mean_oracle_distance": float(mean_oracle_distance.mean().detach().cpu()),
        "success_rate": float(success_gates.instant_success.float().mean().detach().cpu()),
        "safe_success_rate": float(success_gates.instant_success.float().mean().detach().cpu()),
        "min_pairwise_ok_rate": float(success_gates.min_pairwise_ok.float().mean().detach().cpu()),
        "nan_flag": False,
        "checkpoint_path": str(checkpoint_path),
        "training_semantics": training_semantics,
        "observation_schema_version": env.cfg.observation.schema_version,
        "actor_obs_dim": env.cfg.actor_obs_dim,
        "critic_state_dim": env.cfg.critic_state_dim,
        "success_threshold": {
            "dmax": float(env.cfg.success_thresholds.dmax),
            "dispersion": float(env.cfg.success_thresholds.dispersion),
            "speed": float(env.cfg.success_thresholds.speed),
            "hold_steps": int(env.cfg.success_thresholds.hold_steps),
            "min_pairwise_distance": float(env.cfg.success_thresholds.min_pairwise_distance),
        },
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
    }
    reward_metrics = dict(telemetry_state.get("reward_breakdown", _reward_breakdown({}, env.cfg)))
    reward_metrics.update(telemetry_state.get("reward_window", {}))
    reward_metrics = _finalize_reward_summary(reward_metrics)
    action_metrics = dict(telemetry_state.get("action", {}))
    action_metrics.update(telemetry_state.get("action_window", {}))
    path_metrics = dict(telemetry_state.get("path_terrain", {}))
    path_metrics.update(telemetry_state.get("path_terrain_window", {}))
    filter_metrics = dict(telemetry_state.get("action_filter", {}))
    filter_metrics.update(telemetry_state.get("action_filter_window", {}))
    telemetry.update(reward_metrics)
    telemetry.update(action_metrics)
    telemetry.update(path_metrics)
    telemetry.update(filter_metrics)
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
    if training_diagnostics is not None:
        telemetry["training_diagnostics"] = training_diagnostics
        telemetry.update(training_diagnostics)
    return telemetry


def append_metrics_jsonl(
    output_dir: Path,
    metrics: dict,
    *,
    filename: str = "metrics.jsonl",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / filename
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
    actor_obs_dim: int | None = None,
    critic_state_dim: int | None = None,
    device: str | None = None,
    checkpoint_path: str | None = None,
    extra_metadata: dict | None = None,
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
    terrain = raw_cfg.get("terrain", {}) if isinstance(raw_cfg.get("terrain", {}), dict) else {}
    payload["metadata"] = {
        "training_semantics": training_semantics,
        "backend": "skrl.mappo",
        "experiment_name": experiment.get("name"),
        "algorithm_mode": algorithm.get("mode"),
        "observation_schema_version": observation_schema_version,
        "actor_obs_dim": actor_obs_dim,
        "critic_state_dim": critic_state_dim,
        "shared_actor": shared_actor,
        "centralized_critic": centralized_critic,
        "shared_value": shared_value,
        "timesteps": timesteps,
        "device": device,
        "checkpoint_path": checkpoint_path,
        "terrain_randomize_per_reset": bool(
            terrain.get("randomize_per_reset", False)
        ),
        "terrain_randomization": {
            key: terrain.get(key)
            for key in (
                "random_translation_m",
                "random_yaw_rad",
                "amplitude_scale_min",
                "amplitude_scale_max",
                "crater_radius_scale_min",
                "crater_radius_scale_max",
                "crater_depth_scale_min",
                "crater_depth_scale_max",
            )
            if key in terrain
        },
    }
    if extra_metadata:
        payload["metadata"].update(extra_metadata)
    payload["cfg"] = raw_cfg
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp_001_minimal.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--timesteps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-layout", choices=("legacy", "run"), default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--eval-num-envs", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--eval-seed-offset", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--bc-updates", type=int, default=None)
    parser.add_argument("--bc-batch-size", type=int, default=None)
    parser.add_argument(
        "--selection-gate",
        choices=(
            "screen",
            "strict",
            "pure_rl_long",
            "safe_progress_long",
            "balanced_progress_long",
        ),
        default="strict",
    )
    parser.add_argument("--bc-only", action="store_true")
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    exp = raw_cfg.get("experiment", {})
    algo = raw_cfg.get("algorithm", {})
    cfg = cfg_from_experiment(args.config)
    if args.device is not None:
        cfg.simulation.device = args.device
        raw_cfg.setdefault("experiment", {})["device"] = args.device
    if args.seed is not None:
        cfg.seed = args.seed
        raw_cfg.setdefault("experiment", {})["seed"] = args.seed
    if args.num_envs is not None:
        cfg.simulation.num_envs = args.num_envs
        raw_cfg.setdefault("experiment", {})["num_envs"] = args.num_envs
    if args.rollout_steps is not None:
        exp["rollout_steps"] = args.rollout_steps
    if args.bc_updates is not None:
        algo["bc_updates"] = args.bc_updates
    if args.bc_batch_size is not None:
        algo["bc_batch_size"] = args.bc_batch_size
    torch.manual_seed(cfg.seed)
    requested_device = torch.device(cfg.simulation.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this stage.")

    experiment_name = str(exp.get("name", Path(args.config).stem))
    output_layout = args.output_layout or str(exp.get("output_layout", "legacy")).lower()
    run_id = args.run_name or f"{experiment_name}_{int(time.time())}"
    if output_layout == "run":
        if args.run_name is None:
            raise SystemExit("--run-name is required when --output-layout run is used.")
        run_dir = ensure_output_dir(Path("outputs/runs") / experiment_name / args.run_name)
        config_dir = run_dir / "config"
        checkpoint_dir = run_dir / "checkpoints"
        telemetry_dir = run_dir / "metrics"
        for directory in (config_dir, checkpoint_dir, telemetry_dir, run_dir / "tensorboard"):
            directory.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "experiment.yaml"
        config_path.write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")
        checkpoint_path = checkpoint_dir / "best.pt"
        metrics_filename = "train_metrics.jsonl"
    else:
        run_dir = ensure_output_dir(Path("outputs/runs") / experiment_name)
        config_path = Path(args.config)
        telemetry_dir = run_dir
        checkpoint_dir = ensure_output_dir(exp.get("checkpoint_dir", "outputs/checkpoints"))
        checkpoint_path = checkpoint_dir / resolve_checkpoint_name(raw_cfg, args.config)
        metrics_filename = "metrics.jsonl"

    env = MultiRoverGatheringSKRLEnv(cfg)
    training_seed = cfg.seed
    telemetry_state = {
        "mean_reward": None,
        "step": 0,
        "interval": int(exp.get("telemetry_interval", exp.get("rollout_steps", 128))),
        "done_counts": _empty_done_counts(),
    }
    wrapped_env = wrap_env(env, wrapper="isaaclab-multi-agent", verbose=False)
    if hasattr(wrapped_env, "_seed"):
        wrapped_env._seed = training_seed
    possible_agents = env.possible_agents
    empty_kwargs = {uid: {} for uid in possible_agents}
    shared_actor = parse_bool_config(algo.get("shared_actor"), default=True)
    centralized_critic = parse_bool_config(algo.get("centralized_critic"), default=True)
    shared_value = parse_bool_config(algo.get("shared_value"), default=True)
    training_semantics = resolve_training_semantics(raw_cfg)
    eval_num_envs = int(
        args.eval_num_envs
        if args.eval_num_envs is not None
        else exp.get("eval_num_envs", 256)
    )
    eval_steps = int(
        args.eval_steps
        if args.eval_steps is not None
        else exp.get("eval_steps", cfg.simulation.max_episode_steps)
    )
    eval_seed_offset = int(
        args.eval_seed_offset
        if args.eval_seed_offset is not None
        else exp.get("eval_seed_offset", 1000)
    )
    baseline_cfg = copy.deepcopy(cfg)
    baseline_cfg.simulation.num_envs = eval_num_envs
    baseline_cfg.seed = training_seed + eval_seed_offset
    random_baseline = evaluate_policy_signal(
        MultiRoverGatheringSKRLEnv(baseline_cfg),
        mode="random",
        max_steps=eval_steps,
    )

    models = build_skrl_mappo_models(
        env,
        shared_actor=shared_actor,
        centralized_critic=centralized_critic,
        shared_value=shared_value,
        initial_log_std=float(algo.get("initial_log_std", -0.5)),
    )
    policy = models[possible_agents[0]]["policy"]
    policy.to(env.device)
    random_initial_policy_parameters = [
        parameter.detach().clone()
        for parameter in policy.parameters()
    ]
    random_initial_first_layer_weight = policy.net[0].weight.detach().clone()
    bc_updates = int(algo.get("bc_updates", algo.get("bc_steps", 0)))
    bc_batch_size = int(algo.get("bc_batch_size", 8192))
    bc_learning_rate = float(algo.get("bc_learning_rate", 1.0e-3))
    teacher_mode: str | None = None
    teacher_stop_radius: float | None = None
    teacher_slow_distance: float | None = None
    teacher_max_rho: float | None = None
    if bc_updates > 0:
        teacher_mode = str(algo.get("teacher_mode", "global_centroid"))
        teacher_stop_radius = float(algo.get("teacher_stop_radius", 0.45))
        teacher_slow_distance = float(algo.get("teacher_slow_distance", 0.45))
        teacher_max_rho = (
            float(algo["teacher_max_rho"])
            if algo.get("teacher_max_rho") is not None
            else None
        )
        bc_records = run_skrl_behavior_cloning(
            policy,
            cfg,
            updates=bc_updates,
            batch_size=bc_batch_size,
            learning_rate=bc_learning_rate,
            teacher_stop_radius=teacher_stop_radius,
            teacher_slow_distance=teacher_slow_distance,
            teacher_max_rho=teacher_max_rho,
            teacher_mode=teacher_mode,
            teacher_terrain_scale=parse_bool_config(
                algo.get("teacher_terrain_scale"),
                default=False,
            ),
            bc_yaw_noise_degrees=(
                float(algo["bc_yaw_noise_degrees"])
                if algo.get("bc_yaw_noise_degrees") is not None
                else None
            ),
            bc_min_nearest_distance=(
                float(algo["bc_min_nearest_distance"])
                if algo.get("bc_min_nearest_distance") is not None
                else None
            ),
        )
    else:
        bc_records = []
    for record in bc_records:
        append_metrics_jsonl(telemetry_dir, record, filename=metrics_filename)

    memories = build_skrl_mappo_memories(env, rollout_steps=int(exp.get("rollout_steps", 32)))
    update_mode = str(algo.get("update_mode", "per_agent"))
    agent_class = SharedPolicyMAPPO if update_mode == "shared_joint" else MAPPO
    if update_mode == "shared_joint" and not (shared_actor and shared_value):
        raise ValueError("algorithm.update_mode=shared_joint requires shared_actor/shared_value.")
    agent_kwargs = dict(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        device=env.device,
        cfg=build_mappo_config(
            algo,
            exp,
            empty_kwargs,
            run_dir=run_dir if output_layout == "run" else None,
        ),
    )
    if agent_class is SharedPolicyMAPPO:
        agent_kwargs["entropy_loss_scale_end"] = float(
            algo.get(
                "entropy_loss_scale_end",
                algo.get("entropy_loss_scale", algo.get("entropy_coef_start", 0.0)),
            )
        )
        if algo.get("entropy_schedule_timesteps") is not None:
            agent_kwargs["entropy_schedule_timesteps"] = int(
                algo["entropy_schedule_timesteps"]
            )
    agent = agent_class(**agent_kwargs)
    initial_policy_parameters = [
        parameter.detach().clone()
        for parameter in policy.parameters()
    ]
    initial_first_layer_weight = policy.net[0].weight.detach().clone()

    if args.bc_only:
        if output_layout != "run":
            raise SystemExit("--bc-only requires --output-layout run.")
        from evaluate_proxy_policy import evaluate_checkpoint

        bc_checkpoint = checkpoint_dir / "bc_only.pt"
        torch.save(
            skrl_mappo_checkpoint_payload(
                models,
                possible_agents,
                raw_cfg=raw_cfg,
                training_semantics=training_semantics,
                observation_schema_version=cfg.observation.schema_version,
                actor_obs_dim=cfg.actor_obs_dim,
                critic_state_dim=cfg.critic_state_dim,
                shared_actor=shared_actor,
                centralized_critic=centralized_critic,
                shared_value=shared_value,
                timesteps=0,
                device=str(env.device),
                checkpoint_path=str(bc_checkpoint),
                extra_metadata={
                    "phase": "bc",
                    "update_mode": update_mode,
                    "communication_radius": cfg.observation.communication_radius,
                    "subgoal_filter": subgoal_filter_metadata(cfg),
                    "bc_updates": bc_updates,
                    "entropy_schedule_timesteps": algo.get(
                        "entropy_schedule_timesteps"
                    ),
                    **checkpoint_teacher_metadata(
                        bc_updates=bc_updates,
                        teacher_mode=teacher_mode,
                        teacher_stop_radius=teacher_stop_radius,
                        teacher_max_rho=teacher_max_rho,
                    ),
                },
            ),
            bc_checkpoint,
        )
        shutil.copy2(bc_checkpoint, checkpoint_path)
        final_eval = evaluate_checkpoint(
            config=config_path,
            checkpoint=checkpoint_path,
            device=str(env.device),
            num_envs=eval_num_envs,
            steps=eval_steps,
            seed=final_eval_seed(training_seed, eval_seed_offset),
            output=telemetry_dir / "final_eval_proxy.json",
            run_dir=run_dir,
        )
        strict = proxy_acceptance(final_eval)
        parameter_delta_sq = torch.zeros((), device=env.device)
        for initial, current in zip(
            random_initial_policy_parameters,
            policy.parameters(),
            strict=True,
        ):
            parameter_delta_sq += (current.detach() - initial).square().sum()
        terrain_start = (
            cfg.observation.ego_dim
            + cfg.observation.max_neighbors * cfg.observation.neighbor_dim
        )
        terrain_end = terrain_start + cfg.observation.terrain_dim
        first_layer_delta = policy.net[0].weight.detach() - random_initial_first_layer_weight
        policy_eval_cfg = copy.deepcopy(cfg)
        policy_eval_cfg.simulation.num_envs = eval_num_envs
        policy_eval_cfg.seed = training_seed + eval_seed_offset
        post_training_eval = evaluate_policy_signal(
            MultiRoverGatheringSKRLEnv(policy_eval_cfg),
            mode="policy",
            policy=policy,
            max_steps=eval_steps,
        )
        diagnostics = {
            "policy_parameter_delta_l2": float(torch.sqrt(parameter_delta_sq).cpu()),
            "terrain_input_weight_delta_l2": float(
                torch.linalg.vector_norm(
                    first_layer_delta[:, terrain_start:terrain_end]
                ).cpu()
            ),
            "bc_parameter_delta_l2": float(torch.sqrt(parameter_delta_sq).cpu()),
            "bc_updates": bc_updates,
            "bc_initial_loss": bc_records[0]["bc_loss"] if bc_records else None,
            "bc_final_loss": bc_records[-1]["bc_loss"] if bc_records else None,
            "post_training_action_std": post_training_eval.get("eval_action_std"),
            "update_mode": update_mode,
            "optimizer_count": 0,
            "joint_update_count": 0,
            "critic_update_count": 0,
        }
        summary = {
            "status": "ok",
            "phase": "bc_only",
            "backend": "skrl.mappo",
            "training_semantics": training_semantics,
            "seed": training_seed,
            "num_envs": cfg.simulation.num_envs,
            "timesteps": 0,
            "env_steps": 0,
            "device": str(env.device),
            "checkpoint_path": str(checkpoint_path),
            "observation_schema_version": cfg.observation.schema_version,
            "actor_obs_dim": cfg.actor_obs_dim,
            "critic_state_dim": cfg.critic_state_dim,
            "update_mode": update_mode,
            "communication_radius": cfg.observation.communication_radius,
            "terrain_sanity": terrain_sanity_metrics(cfg, env.device),
            "bc": {
                "updates": bc_updates,
                "batch_size": bc_batch_size,
                "learning_rate": bc_learning_rate,
                "records": bc_records,
            },
            "training_diagnostics": diagnostics,
            "post_training_eval": post_training_eval,
            "final_eval": final_eval,
            "strict_acceptance": strict,
        }
        summary_path = telemetry_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (telemetry_dir / "eval_metrics.json").write_text(
            json.dumps({"best_candidate": str(bc_checkpoint), "evaluations": [final_eval]}, indent=2),
            encoding="utf-8",
        )
        (telemetry_dir / "strict_acceptance.json").write_text(
            json.dumps(strict, indent=2),
            encoding="utf-8",
        )
        (telemetry_dir / "checkpoint_status.json").write_text(
            json.dumps(
                {
                    "state": "proxy_passed" if strict["passed"] else "candidate",
                    "checkpoint": str(checkpoint_path),
                    "proxy_evaluation": str(telemetry_dir / "final_eval_proxy.json"),
                    "strict_acceptance": strict,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "experiment": experiment_name,
                    "run": args.run_name,
                    "producer": "scripts/train_skrl_mappo.py",
                    "command": " ".join(sys.argv),
                    "summary": {
                        "status": "candidate" if not strict["passed"] else "proxy_passed",
                        "phase": "bc_only",
                        "seed": training_seed,
                        "communication_radius": cfg.observation.communication_radius,
                    },
                    "artifacts": {
                        "config": str(config_path.relative_to(ROOT)),
                        "checkpoint_best": str(checkpoint_path.relative_to(ROOT)),
                        "metrics_summary": str(summary_path.relative_to(ROOT)),
                        "metrics_train": str(
                            (telemetry_dir / metrics_filename).relative_to(ROOT)
                        ),
                        "metrics_final_eval": str(
                            (telemetry_dir / "final_eval_proxy.json").relative_to(ROOT)
                        ),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(yaml.safe_dump(summary, sort_keys=False))
        return

    def write_interval_telemetry(step: int) -> None:
        _snapshot_numeric_metrics(telemetry_state, "action", "action_window")
        _snapshot_numeric_metrics(telemetry_state, "reward", "reward_window")
        _snapshot_numeric_metrics(telemetry_state, "path_terrain", "path_terrain_window")
        _snapshot_numeric_metrics(telemetry_state, "action_filter", "action_filter_window")
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
            filename=metrics_filename,
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
    checkpoint_interval = int(
        args.checkpoint_interval
        if args.checkpoint_interval is not None
        else exp.get("checkpoint_interval", 0)
    )
    candidate_paths: list[Path] = []

    def save_candidate(timestep: int) -> Path:
        candidate_path = checkpoint_dir / f"ppo_timestep_{timestep:06d}.pt"
        torch.save(
            skrl_mappo_checkpoint_payload(
                models,
                possible_agents,
                raw_cfg=raw_cfg,
                training_semantics=training_semantics,
                observation_schema_version=cfg.observation.schema_version,
                actor_obs_dim=cfg.actor_obs_dim,
                critic_state_dim=cfg.critic_state_dim,
                shared_actor=shared_actor,
                centralized_critic=centralized_critic,
                shared_value=shared_value,
                timesteps=timestep,
                device=str(env.device),
                checkpoint_path=str(candidate_path),
                extra_metadata={
                    "phase": "ppo",
                    "update_mode": update_mode,
                    "communication_radius": cfg.observation.communication_radius,
                    "subgoal_filter": subgoal_filter_metadata(cfg),
                    "bc_updates": bc_updates,
                    "bc_batch_size": bc_batch_size,
                    "bc_learning_rate": bc_learning_rate,
                    "entropy_schedule_timesteps": algo.get(
                        "entropy_schedule_timesteps"
                    ),
                    **checkpoint_teacher_metadata(
                        bc_updates=bc_updates,
                        teacher_mode=teacher_mode,
                        teacher_stop_radius=teacher_stop_radius,
                        teacher_max_rho=teacher_max_rho,
                    ),
                },
            ),
            candidate_path,
        )
        candidate_paths.append(candidate_path)
        return candidate_path

    if checkpoint_interval > 0:
        original_post_interaction = agent.post_interaction

        def post_interaction_with_checkpoint(*, timestep: int, timesteps: int) -> None:
            original_post_interaction(timestep=timestep, timesteps=timesteps)
            completed_timestep = timestep + 1
            if completed_timestep % checkpoint_interval == 0:
                save_candidate(completed_timestep)

        agent.post_interaction = post_interaction_with_checkpoint

    if env.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(env.device)
    start_time = time.perf_counter()
    telemetry_state["training_start_time"] = start_time
    trainer.train()
    if not candidate_paths or candidate_paths[-1].stem != f"ppo_timestep_{args.timesteps:06d}":
        save_candidate(args.timesteps)
    wall_time_s = time.perf_counter() - start_time
    peak_cuda_memory_mb = (
        torch.cuda.max_memory_allocated(env.device) / (1024.0 * 1024.0)
        if env.device.type == "cuda"
        else None
    )
    policy_eval_cfg = copy.deepcopy(cfg)
    policy_eval_cfg.simulation.num_envs = eval_num_envs
    policy_eval_cfg.seed = training_seed + eval_seed_offset
    post_training_eval = evaluate_policy_signal(
        MultiRoverGatheringSKRLEnv(policy_eval_cfg),
        mode="policy",
        policy=policy,
        max_steps=eval_steps,
    )
    parameter_delta_sq = torch.zeros((), device=env.device)
    for initial, current in zip(initial_policy_parameters, policy.parameters(), strict=True):
        parameter_delta_sq = parameter_delta_sq + (current.detach() - initial).square().sum()
    terrain_start = (
        cfg.observation.ego_dim
        + cfg.observation.max_neighbors * cfg.observation.neighbor_dim
    )
    terrain_end = terrain_start + cfg.observation.terrain_dim
    first_layer_delta = policy.net[0].weight.detach() - initial_first_layer_weight
    bc_parameter_delta_sq = torch.zeros((), device=env.device)
    for initial, after_bc in zip(
        random_initial_policy_parameters,
        initial_policy_parameters,
        strict=True,
    ):
        bc_parameter_delta_sq = bc_parameter_delta_sq + (after_bc - initial).square().sum()
    training_diagnostics = {
        "policy_parameter_delta_l2": float(torch.sqrt(parameter_delta_sq).cpu()),
        "terrain_input_weight_delta_l2": float(
            torch.linalg.vector_norm(first_layer_delta[:, terrain_start:terrain_end]).cpu()
        ),
        "bc_parameter_delta_l2": float(torch.sqrt(bc_parameter_delta_sq).cpu()),
        "bc_updates": bc_updates,
        "bc_initial_loss": bc_records[0]["bc_loss"] if bc_records else None,
        "bc_final_loss": bc_records[-1]["bc_loss"] if bc_records else None,
        "post_training_action_std": post_training_eval.get("eval_action_std"),
        "update_mode": update_mode,
        "optimizer_count": int(getattr(agent, "optimizer_count", len(agent.optimizers))),
        "joint_update_count": int(getattr(agent, "joint_update_count", 0)),
        "critic_update_count": int(getattr(agent, "critic_update_count", 0)),
        "last_actor_sample_count": int(getattr(agent, "last_actor_sample_count", 0)),
        "last_critic_sample_count": int(getattr(agent, "last_critic_sample_count", 0)),
    }
    _snapshot_numeric_metrics(telemetry_state, "action", "action_window")
    _snapshot_numeric_metrics(telemetry_state, "reward", "reward_window")
    _snapshot_numeric_metrics(telemetry_state, "path_terrain", "path_terrain_window")
    _snapshot_numeric_metrics(telemetry_state, "action_filter", "action_filter_window")

    candidate_evaluations: list[dict] = []
    best_candidate = candidate_paths[-1]
    final_eval: dict | None = None
    strict = {"passed": False, "checks": {}, "thresholds": STRICT_THRESHOLDS}
    if output_layout == "run":
        from evaluate_proxy_policy import evaluate_checkpoint

        for candidate_path in candidate_paths:
            candidate_eval_path = telemetry_dir / f"candidate_{candidate_path.stem}_eval.json"
            evaluation = evaluate_checkpoint(
                config=config_path,
                checkpoint=candidate_path,
                device=str(env.device),
                num_envs=eval_num_envs,
                steps=eval_steps,
                seed=candidate_eval_seed(training_seed, eval_seed_offset),
                output=candidate_eval_path,
                run_dir=run_dir,
            )
            evaluation["candidate_timestep"] = int(candidate_path.stem.rsplit("_", 1)[-1])
            candidate_evaluations.append(evaluation)
        selection_thresholds = (
            {
                "dmax_reduction_ratio": 0.30,
                "success_rate": 0.50,
                "collision_rate": 0.03,
                "timeout_rate": 0.50,
            }
            if args.selection_gate == "screen"
            else PURE_RL_LONG_THRESHOLDS
            if args.selection_gate == "pure_rl_long"
            else SAFE_PROGRESS_LONG_THRESHOLDS
            if args.selection_gate == "safe_progress_long"
            else BALANCED_PROGRESS_LONG_THRESHOLDS
            if args.selection_gate == "balanced_progress_long"
            else STRICT_THRESHOLDS
        )
        ranker = (
            pure_rl_long_checkpoint_rank
            if args.selection_gate == "pure_rl_long"
            else safe_progress_long_checkpoint_rank
            if args.selection_gate == "safe_progress_long"
            else balanced_progress_long_checkpoint_rank
            if args.selection_gate == "balanced_progress_long"
            else checkpoint_rank
        )
        best_evaluation = min(
            candidate_evaluations,
            key=lambda item: ranker(item, selection_thresholds),
        )
        best_candidate = Path(best_evaluation["checkpoint"])
        shutil.copy2(best_candidate, checkpoint_path)
        final_eval = evaluate_checkpoint(
            config=config_path,
            checkpoint=checkpoint_path,
            device=str(env.device),
            num_envs=eval_num_envs,
            steps=eval_steps,
            seed=final_eval_seed(training_seed, eval_seed_offset),
            output=telemetry_dir / "final_eval_proxy.json",
            run_dir=run_dir,
        )
        strict = proxy_acceptance(final_eval)
        (telemetry_dir / "eval_metrics.json").write_text(
            json.dumps(
                {
                    "best_candidate": str(best_candidate),
                    "evaluations": candidate_evaluations,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (telemetry_dir / "strict_acceptance.json").write_text(
            json.dumps(strict, indent=2),
            encoding="utf-8",
        )
        (telemetry_dir / "checkpoint_status.json").write_text(
            json.dumps(
                {
                    "state": "proxy_passed" if strict["passed"] else "candidate",
                    "checkpoint": str(checkpoint_path),
                    "proxy_evaluation": str(telemetry_dir / "final_eval_proxy.json"),
                    "strict_acceptance": strict,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        shutil.copy2(best_candidate, checkpoint_path)

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
            training_diagnostics=training_diagnostics,
            peak_cuda_memory_mb=peak_cuda_memory_mb,
        ),
        filename=metrics_filename,
    )
    summary = {
        "status": "ok",
        "backend": "skrl.mappo",
        "training_semantics": training_semantics,
        "shared_actor": shared_actor,
        "centralized_critic": centralized_critic,
        "shared_value": shared_value,
        "update_mode": update_mode,
        "communication_radius": cfg.observation.communication_radius,
        "subgoal_filter": subgoal_filter_metadata(cfg),
        "seed": training_seed,
        "num_envs": cfg.simulation.num_envs,
        "timesteps": args.timesteps,
        "env_steps": args.timesteps * cfg.simulation.num_envs,
        "rollout_steps": int(exp.get("rollout_steps", 32)),
        "device": str(env.device),
        "wall_time_s": wall_time_s,
        "checkpoint_interval": checkpoint_interval,
        "candidate_count": len(candidate_paths),
        "selection_gate": args.selection_gate,
        "best_candidate": str(best_candidate),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "observation_schema_version": cfg.observation.schema_version,
        "actor_obs_dim": cfg.actor_obs_dim,
        "critic_state_dim": cfg.critic_state_dim,
        "terrain_sanity": terrain_sanity_metrics(cfg, env.device),
        "terrain_randomization": {
            "randomize_per_reset": bool(cfg.terrain.randomize_per_reset),
            "random_translation_m": float(cfg.terrain.random_translation_m),
            "random_yaw_rad": float(cfg.terrain.random_yaw_rad),
            "amplitude_scale": [
                float(cfg.terrain.amplitude_scale_min),
                float(cfg.terrain.amplitude_scale_max),
            ],
            "crater_radius_scale": [
                float(cfg.terrain.crater_radius_scale_min),
                float(cfg.terrain.crater_radius_scale_max),
            ],
            "crater_depth_scale": [
                float(cfg.terrain.crater_depth_scale_min),
                float(cfg.terrain.crater_depth_scale_max),
            ],
        },
        "bc": {
            "updates": bc_updates,
            "batch_size": bc_batch_size,
            "learning_rate": bc_learning_rate,
            "records": bc_records,
        },
        "mappo": {
            key: value
            for key, value in build_mappo_config(algo, exp, empty_kwargs).items()
            if key
            not in {
                "learning_rate_scheduler_kwargs",
                "observation_preprocessor_kwargs",
                "state_preprocessor_kwargs",
                "value_preprocessor_kwargs",
            }
        },
        "training_diagnostics": training_diagnostics,
        "post_training_eval": post_training_eval,
        "final_eval": final_eval,
        "strict_acceptance": strict,
    }
    if output_layout == "run":
        summary_path = telemetry_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment": experiment_name,
            "run": args.run_name,
            "producer": "scripts/train_skrl_mappo.py",
            "command": " ".join(sys.argv),
            "summary": {
                "status": "candidate" if not strict["passed"] else "proxy_passed",
                "seed": training_seed,
                "device": str(env.device),
                "timesteps": args.timesteps,
                "env_steps": args.timesteps * cfg.simulation.num_envs,
                "observation_schema_version": cfg.observation.schema_version,
                "actor_obs_dim": cfg.actor_obs_dim,
                "critic_state_dim": cfg.critic_state_dim,
                "strict_passed": strict["passed"],
            },
            "artifacts": {
                "config": str(config_path.relative_to(ROOT)),
                "checkpoint_best": str(checkpoint_path.relative_to(ROOT)),
                "metrics_summary": str(summary_path.relative_to(ROOT)),
                "metrics_train": str(metrics_path.relative_to(ROOT)),
                "metrics_eval": str((telemetry_dir / "eval_metrics.json").relative_to(ROOT)),
                "metrics_final_eval": str((telemetry_dir / "final_eval_proxy.json").relative_to(ROOT)),
                "metrics_strict": str((telemetry_dir / "strict_acceptance.json").relative_to(ROOT)),
                "checkpoint_status": str(
                    (telemetry_dir / "checkpoint_status.json").relative_to(ROOT)
                ),
                "tensorboard": str((run_dir / "tensorboard").relative_to(ROOT)),
            },
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
    print(
        yaml.safe_dump(
            summary,
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
