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
    validate_checkpoint_compatibility,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringSKRLEnv
from lunar_rover_tasks.tasks.multi_rover_gathering.communication import compute_visibility_mask
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.oracle import compute_mean_oracle_distance
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import query_terrain_features
from lunar_rover_tasks.tasks.multi_rover_gathering.termination import compute_success_gates
from lunar_rover_tasks.utils.geometry_utils import pairwise_distances_xy

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.trainers.torch import SequentialTrainer

from shared_policy_mappo import SharedPolicyMAPPO


TRAINING_SEMANTICS = DEFAULT_TRAINING_SEMANTICS

ACTOR_ARCHITECTURES = {
    "mlp_v1",
    "branched_v1",
    "branched_v2",
    "branched_v3",
    "branched_v4",
    "branched_v5",
    "branched_v6_graph_attention",
}
CRITIC_ARCHITECTURES = {"mlp_v1", "structured_v1", "structured_v2"}
ACTOR_OBSERVATION_SLICES_V3 = {
    "ego": (0, 10),
    "neighbors": (10, 31),
    "terrain": (31, 81),
    "aggregation": (81, 86),
}
ACTOR_OBSERVATION_SLICES_V4 = {
    **ACTOR_OBSERVATION_SLICES_V3,
    "terminal_gate": (86, 91),
}
ACTOR_OBSERVATION_SLICES_V5 = {
    **ACTOR_OBSERVATION_SLICES_V3,
    "gather_site_goal": (86, 89),
}
ACTOR_OBSERVATION_SLICES_V6 = {
    **ACTOR_OBSERVATION_SLICES_V3,
    "gather_site_and_slot_goal": (86, 92),
}
ACTOR_OBSERVATION_SLICES_V8 = {
    "ego": (0, 10),
    "neighbors": (10, 46),
    "terrain": (46, 96),
    "aggregation": (96, 101),
}
ACTOR_OBSERVATION_SLICES = ACTOR_OBSERVATION_SLICES_V3
CRITIC_STATE_SLICES_V3 = {
    "agents": (0, 32),
    "team": (32, 40),
    "terrain": (40, 45),
    "oracle": (45, 54),
}
CRITIC_STATE_SLICES_V4 = {
    "agents": (0, 32),
    "team": (32, 41),
    "terrain": (41, 46),
    "oracle": (46, 55),
}
CRITIC_STATE_SLICES = CRITIC_STATE_SLICES_V3


def _actor_slices_for_dim(num_observations: int) -> dict[str, tuple[int, int]]:
    if int(num_observations) == 86:
        return ACTOR_OBSERVATION_SLICES_V3
    if int(num_observations) == 91:
        return ACTOR_OBSERVATION_SLICES_V4
    if int(num_observations) == 89:
        return ACTOR_OBSERVATION_SLICES_V5
    if int(num_observations) == 92:
        return ACTOR_OBSERVATION_SLICES_V6
    if int(num_observations) == 101:
        return ACTOR_OBSERVATION_SLICES_V8
    raise ValueError(f"Unsupported actor observation dim: {num_observations}.")


def _critic_slices_for_dim(num_states: int) -> dict[str, tuple[int, int]]:
    if int(num_states) == 54:
        return CRITIC_STATE_SLICES_V3
    if int(num_states) == 55:
        return CRITIC_STATE_SLICES_V4
    raise ValueError(f"Unsupported critic state dim: {num_states}.")


def _slice_metadata(slices: dict[str, tuple[int, int]]) -> dict[str, dict[str, int]]:
    return {
        name: {"start": start, "end": end, "dim": end - start}
        for name, (start, end) in slices.items()
    }


def observation_slices_metadata(actor_obs_dim: int = 86) -> dict[str, dict[str, int]]:
    return _slice_metadata(_actor_slices_for_dim(actor_obs_dim))


def critic_state_slices_metadata(critic_state_dim: int = 54) -> dict[str, dict[str, int]]:
    metadata = _slice_metadata(_critic_slices_for_dim(critic_state_dim))
    metadata["agents"]["shape_agents"] = 4
    metadata["agents"]["shape_features"] = 8
    return metadata


def normalize_actor_architecture(value: Any | None) -> str:
    architecture = "mlp_v1" if value is None else str(value)
    if architecture not in ACTOR_ARCHITECTURES:
        raise ValueError(
            "algorithm.actor_architecture must be one of: "
            f"{', '.join(sorted(ACTOR_ARCHITECTURES))}."
        )
    return architecture


def normalize_critic_architecture(value: Any | None) -> str:
    architecture = "mlp_v1" if value is None else str(value)
    if architecture not in CRITIC_ARCHITECTURES:
        raise ValueError(
            "algorithm.critic_architecture must be one of: "
            f"{', '.join(sorted(CRITIC_ARCHITECTURES))}."
        )
    return architecture


class GraphAttentionNeighborEncoder(nn.Module):
    """Permutation-invariant one-hop aggregation over three cached neighbors.

    Each cached message keeps the fixed 12-dimensional physical schema.  The
    final component is message quality ``q``; ``q <= 0`` marks an invalid
    cache slot.  No sender index or state outside the observation is consumed.
    """

    def __init__(
        self,
        *,
        neighbor_dim: int = 12,
        num_neighbors: int = 3,
        node_dim: int = 32,
        ego_dim: int = 32,
        num_heads: int = 4,
        head_dim: int = 12,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.neighbor_dim = int(neighbor_dim)
        self.num_neighbors = int(num_neighbors)
        self.node_dim = int(node_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.output_dim = self.num_heads * self.head_dim
        self.eps = float(eps)

        self.node_encoder = nn.Sequential(
            nn.Linear(self.neighbor_dim, self.node_dim),
            nn.ELU(),
        )
        self.query_projection = nn.Linear(ego_dim, self.output_dim)
        self.key_projection = nn.Linear(self.node_dim, self.output_dim)
        self.value_projection = nn.Linear(self.node_dim, self.output_dim)

    def forward(
        self,
        flat_neighbors: torch.Tensor,
        ego_embedding: torch.Tensor,
    ) -> torch.Tensor:
        expected = self.num_neighbors * self.neighbor_dim
        if flat_neighbors.shape[-1] != expected:
            raise ValueError(
                f"GraphAttentionNeighborEncoder expects {expected} flattened "
                f"neighbor features, got {flat_neighbors.shape[-1]}."
            )
        if ego_embedding.shape[-1] != self.query_projection.in_features:
            raise ValueError(
                "GraphAttentionNeighborEncoder received an incompatible ego embedding: "
                f"expected {self.query_projection.in_features}, got "
                f"{ego_embedding.shape[-1]}."
            )

        neighbors = flat_neighbors.reshape(
            *flat_neighbors.shape[:-1],
            self.num_neighbors,
            self.neighbor_dim,
        )
        quality = neighbors[..., -1].clamp(min=0.0, max=1.0)
        valid = quality > 0.0
        nodes = self.node_encoder(neighbors)

        leading_shape = nodes.shape[:-2]
        queries = self.query_projection(ego_embedding).reshape(
            *leading_shape,
            self.num_heads,
            self.head_dim,
        )
        keys = self.key_projection(nodes).reshape(
            *leading_shape,
            self.num_neighbors,
            self.num_heads,
            self.head_dim,
        )
        values = self.value_projection(nodes).reshape(
            *leading_shape,
            self.num_neighbors,
            self.num_heads,
            self.head_dim,
        )
        scores = torch.einsum("...hd,...nhd->...nh", queries, keys)
        scores = scores / math.sqrt(float(self.head_dim))

        # A masked, quality-weighted normalization avoids NaN when no cached
        # neighbor is valid and makes q participate directly in attention.
        masked_scores = scores.masked_fill(~valid.unsqueeze(-1), -1.0e9)
        centered_scores = masked_scores - masked_scores.amax(dim=-2, keepdim=True)
        unnormalized = (
            torch.exp(centered_scores)
            * valid.unsqueeze(-1).to(scores.dtype)
            * quality.unsqueeze(-1).to(scores.dtype)
        )
        weights = unnormalized / unnormalized.sum(dim=-2, keepdim=True).clamp_min(
            self.eps
        )
        aggregated = (weights.unsqueeze(-1) * values).sum(dim=-3)
        return aggregated.reshape(*leading_shape, self.output_dim)


class SKRLPolicy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        initial_log_std: float = -0.5,
        architecture: str = "mlp_v1",
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True, reduction="sum")
        self.architecture = normalize_actor_architecture(architecture)
        if self.architecture == "mlp_v1":
            self.net = nn.Sequential(
                nn.Linear(self.num_observations, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, self.num_actions),
            )
        elif self.architecture == "branched_v1":
            if self.num_observations != 86:
                raise ValueError(
                    "branched_v1 actor expects the fixed ego_v3 86-dim observation."
                )
            self.ego_encoder = nn.Sequential(nn.Linear(10, 32), nn.ELU())
            self.neighbor_encoder = nn.Sequential(nn.Linear(21, 48), nn.ELU())
            self.terrain_encoder = nn.Sequential(nn.Linear(50, 64), nn.ELU())
            self.aggregation_encoder = nn.Sequential(nn.Linear(5, 16), nn.ELU())
            self.trunk = nn.Sequential(
                nn.Linear(160, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, self.num_actions),
            )
        elif self.architecture == "branched_v2":
            if self.num_observations != 91:
                raise ValueError(
                    "branched_v2 actor expects the ego_v4 91-dim terminal-gate observation."
                )
            self.ego_encoder = nn.Sequential(nn.Linear(10, 32), nn.ELU())
            self.neighbor_encoder = nn.Sequential(nn.Linear(21, 48), nn.ELU())
            self.terrain_encoder = nn.Sequential(nn.Linear(50, 64), nn.ELU())
            self.aggregation_encoder = nn.Sequential(nn.Linear(5, 16), nn.ELU())
            self.terminal_gate_encoder = nn.Sequential(nn.Linear(5, 16), nn.ELU())
            self.trunk = nn.Sequential(
                nn.Linear(176, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, self.num_actions),
            )
        elif self.architecture == "branched_v3":
            if self.num_observations != 89:
                raise ValueError(
                    "branched_v3 actor expects the 89-dim single-goal observation."
                )
            self.ego_encoder = nn.Sequential(nn.Linear(10, 32), nn.ELU())
            self.neighbor_encoder = nn.Sequential(nn.Linear(21, 48), nn.ELU())
            self.terrain_encoder = nn.Sequential(nn.Linear(50, 64), nn.ELU())
            self.aggregation_encoder = nn.Sequential(nn.Linear(5, 16), nn.ELU())
            self.gather_site_goal_encoder = nn.Sequential(nn.Linear(3, 16), nn.ELU())
            self.trunk = nn.Sequential(
                nn.Linear(176, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, self.num_actions),
            )
        elif self.architecture == "branched_v4":
            if self.num_observations != 92:
                raise ValueError(
                    "branched_v4 actor expects the ego_v7 92-dim site-and-slot observation."
                )
            self.ego_encoder = nn.Sequential(nn.Linear(10, 32), nn.ELU())
            self.neighbor_encoder = nn.Sequential(nn.Linear(21, 48), nn.ELU())
            self.terrain_encoder = nn.Sequential(nn.Linear(50, 64), nn.ELU())
            self.aggregation_encoder = nn.Sequential(nn.Linear(5, 16), nn.ELU())
            self.gather_site_and_slot_goal_encoder = nn.Sequential(
                nn.Linear(6, 16),
                nn.ELU(),
            )
            self.trunk = nn.Sequential(
                nn.Linear(176, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, self.num_actions),
            )
        elif self.architecture == "branched_v5":
            if self.num_observations != 101:
                raise ValueError(
                    "branched_v5 actor expects the 101-dim decentralized tiered observation."
                )
            self.ego_encoder = nn.Sequential(nn.Linear(10, 32), nn.ELU())
            self.neighbor_encoder = nn.Sequential(nn.Linear(36, 48), nn.ELU())
            self.terrain_encoder = nn.Sequential(nn.Linear(50, 64), nn.ELU())
            self.aggregation_encoder = nn.Sequential(nn.Linear(5, 16), nn.ELU())
            self.trunk = nn.Sequential(
                nn.Linear(160, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, self.num_actions),
            )
        else:
            if self.num_observations != 101:
                raise ValueError(
                    "branched_v6_graph_attention actor expects the 101-dim "
                    "decentralized tiered observation."
                )
            self.ego_encoder = nn.Sequential(nn.Linear(10, 32), nn.ELU())
            self.neighbor_encoder = GraphAttentionNeighborEncoder()
            self.terrain_encoder = nn.Sequential(nn.Linear(50, 64), nn.ELU())
            self.aggregation_encoder = nn.Sequential(nn.Linear(5, 16), nn.ELU())
            self.trunk = nn.Sequential(
                nn.Linear(160, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, self.num_actions),
            )
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), float(initial_log_std))
        )

    def compute(self, inputs, role):
        observations = inputs["observations"]
        if self.architecture == "mlp_v1":
            logits = self.net(observations)
        else:
            slices = _actor_slices_for_dim(self.num_observations)
            ego_start, ego_end = slices["ego"]
            neighbor_start, neighbor_end = slices["neighbors"]
            terrain_start, terrain_end = slices["terrain"]
            aggregation_start, aggregation_end = slices["aggregation"]
            ego_encoded = self.ego_encoder(observations[..., ego_start:ego_end])
            neighbor_observations = observations[..., neighbor_start:neighbor_end]
            neighbor_encoded = (
                self.neighbor_encoder(neighbor_observations, ego_encoded)
                if self.architecture == "branched_v6_graph_attention"
                else self.neighbor_encoder(neighbor_observations)
            )
            encoded_parts = [
                ego_encoded,
                neighbor_encoded,
                self.terrain_encoder(observations[..., terrain_start:terrain_end]),
                self.aggregation_encoder(
                    observations[..., aggregation_start:aggregation_end]
                ),
            ]
            if self.architecture == "branched_v2":
                terminal_start, terminal_end = slices["terminal_gate"]
                encoded_parts.append(
                    self.terminal_gate_encoder(
                        observations[..., terminal_start:terminal_end]
                    )
                )
            elif self.architecture == "branched_v3":
                goal_start, goal_end = slices["gather_site_goal"]
                encoded_parts.append(
                    self.gather_site_goal_encoder(observations[..., goal_start:goal_end])
                )
            elif self.architecture == "branched_v4":
                goal_start, goal_end = slices["gather_site_and_slot_goal"]
                encoded_parts.append(
                    self.gather_site_and_slot_goal_encoder(
                        observations[..., goal_start:goal_end]
                    )
                )
            encoded = torch.cat(
                encoded_parts,
                dim=-1,
            )
            logits = self.trunk(encoded)
        mean = torch.tanh(logits)
        return mean, {"log_std": self.log_std_parameter.expand_as(mean)}

    def terrain_input_weight(self) -> torch.Tensor:
        if self.architecture == "mlp_v1":
            terrain_start, terrain_end = _actor_slices_for_dim(self.num_observations)["terrain"]
            return self.net[0].weight[:, terrain_start:terrain_end]
        return self.terrain_encoder[0].weight


class SKRLValue(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        architecture: str = "mlp_v1",
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)
        self.architecture = normalize_critic_architecture(architecture)
        if self.architecture == "mlp_v1":
            self.net = nn.Sequential(
                nn.Linear(self.num_states, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, 1),
            )
        elif self.architecture == "structured_v1":
            if self.num_states != 54:
                raise ValueError("structured_v1 critic expects the fixed 54-dim state.")
            self.agent_encoder = nn.Sequential(nn.Linear(8, 32), nn.ELU())
            self.team_encoder = nn.Sequential(nn.Linear(8, 32), nn.ELU())
            self.terrain_encoder = nn.Sequential(nn.Linear(5, 16), nn.ELU())
            self.oracle_encoder = nn.Sequential(nn.Linear(9, 32), nn.ELU())
            self.value_trunk = nn.Sequential(
                nn.Linear(144, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, 1),
            )
        else:
            if self.num_states != 55:
                raise ValueError(
                    "structured_v2 critic expects the ego_v4 55-dim terminal-gate state."
                )
            self.agent_encoder = nn.Sequential(nn.Linear(8, 32), nn.ELU())
            self.team_encoder = nn.Sequential(nn.Linear(9, 32), nn.ELU())
            self.terrain_encoder = nn.Sequential(nn.Linear(5, 16), nn.ELU())
            self.oracle_encoder = nn.Sequential(nn.Linear(9, 32), nn.ELU())
            self.value_trunk = nn.Sequential(
                nn.Linear(144, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, 1),
            )

    def compute(self, inputs, role):
        states = inputs["states"]
        if self.architecture == "mlp_v1":
            return self.net(states), {}
        slices = _critic_slices_for_dim(self.num_states)
        agents_start, agents_end = slices["agents"]
        team_start, team_end = slices["team"]
        terrain_start, terrain_end = slices["terrain"]
        oracle_start, oracle_end = slices["oracle"]
        agent_features = states[..., agents_start:agents_end].reshape(-1, 4, 8)
        encoded_agents = self.agent_encoder(agent_features)
        agent_mean = encoded_agents.mean(dim=1)
        agent_max = encoded_agents.amax(dim=1)
        encoded = torch.cat(
            (
                agent_mean,
                agent_max,
                self.team_encoder(states[..., team_start:team_end]),
                self.terrain_encoder(states[..., terrain_start:terrain_end]),
                self.oracle_encoder(states[..., oracle_start:oracle_end]),
            ),
            dim=-1,
        )
        return self.value_trunk(encoded), {}


def terrain_input_weight_snapshot(policy: SKRLPolicy) -> torch.Tensor:
    return policy.terrain_input_weight().detach().clone()


def terrain_input_weight_delta_l2(
    policy: SKRLPolicy,
    snapshot: torch.Tensor,
) -> float:
    delta = policy.terrain_input_weight().detach() - snapshot
    return float(torch.linalg.vector_norm(delta).cpu())


def module_parameter_snapshot(module: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def module_parameter_delta_l2(
    module: nn.Module,
    snapshot: list[torch.Tensor],
) -> float:
    parameters = list(module.parameters())
    if len(parameters) != len(snapshot):
        raise ValueError("Module parameter structure changed after the snapshot.")
    delta_sq = torch.zeros((), device=parameters[0].device if parameters else "cpu")
    for current, initial in zip(parameters, snapshot, strict=True):
        delta_sq = delta_sq + (current.detach() - initial).square().sum()
    return float(torch.sqrt(delta_sq).cpu())


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
    teacher_center_step: float | None = None,
) -> dict:
    enabled = bc_updates > 0
    metadata = {
        "teacher_mode": teacher_mode if enabled else None,
        "teacher_stop_radius": teacher_stop_radius if enabled else None,
        "teacher_max_rho": teacher_max_rho if enabled else None,
    }
    # Preserve the old pure-RL checkpoint metadata shape.  The extra field is
    # meaningful only for BC records that actually configure the translating
    # ring teacher.
    if enabled and teacher_center_step is not None:
        metadata["teacher_center_step"] = teacher_center_step
    return metadata


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
    teacher_mode: str = "global_centroid",
    teacher_center_step: float = 0.65,
) -> torch.Tensor:
    """Return a safety-aware local subgoal action for BC warm-up.

    ``oracle_ring`` is only meaningful for an execution-goal configuration: it
    assigns each rover a radial slot around the terrain-aware planned point.
    ``oracle_translating_ring`` first translates the current formation center
    toward that point while shrinking it into the same ring. ``oracle_slots``
    follows the fixed, minimum-travel assignment of symmetric slots planned
    by the environment; ``terminal_flat_slots`` uses the actor-visible dynamic
    flat-site slots when that terminal contract activates. The ring radius
    preserves a nonzero pairwise separation instead of training every rover
    to collide at the same point.
    """
    supported_modes = {
        "global_centroid",
        "visible_local_centroid",
        "oracle_ring",
        "oracle_translating_ring",
        "oracle_slots",
        "terminal_flat_slots",
    }
    if teacher_mode not in supported_modes:
        raise ValueError(
            "teacher_mode must be one of: "
            f"{', '.join(sorted(supported_modes))}."
        )
    positions_xy = env.positions[..., :2]
    if teacher_mode in {"oracle_slots", "terminal_flat_slots"}:
        if env.cfg.observation.schema_version not in {
            "ego_v6_gather_slot_goal",
            "ego_v7_gather_site_and_slot_goal",
        }:
            raise ValueError(
                f"teacher_mode={teacher_mode} requires "
                "a gather-slot execution schema."
            )
        if teacher_mode == "terminal_flat_slots":
            if not env.cfg.task.dynamic_terminal_slot_goal_enabled:
                raise ValueError(
                    "teacher_mode=terminal_flat_slots requires "
                    "task.dynamic_terminal_slot_goal_enabled=true."
                )
            target_points = env.execution_slot_points
        else:
            target_points = env.gather_slot_points
        target_error = target_points[..., :2] - positions_xy
        world_delta = torch.where(
            torch.linalg.vector_norm(target_error, dim=-1, keepdim=True) > 1.0e-3,
            target_error,
            torch.zeros_like(target_error),
        )
    elif teacher_mode in {"oracle_ring", "oracle_translating_ring"}:
        if not env.cfg.task.explicit_goal_in_execution:
            raise ValueError(
                "teacher_mode=oracle_ring requires task.explicit_goal_in_execution=true."
            )
        if teacher_mode == "oracle_ring":
            anchor_xy = env.oracle_point[:, None, :2].expand_as(positions_xy)
        else:
            formation_center = positions_xy.mean(dim=1, keepdim=True)
            oracle_delta = env.oracle_point[:, None, :2] - formation_center
            oracle_distance = torch.linalg.vector_norm(
                oracle_delta,
                dim=-1,
                keepdim=True,
            )
            if teacher_center_step <= 0.0:
                raise ValueError("teacher_center_step must be positive.")
            center_shift = oracle_delta * torch.clamp(
                float(teacher_center_step) / oracle_distance.clamp_min(1.0e-6),
                max=1.0,
            )
            anchor_xy = (formation_center + center_shift).expand_as(positions_xy)
        rel = positions_xy - anchor_xy
        dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        fallback_angles = torch.linspace(
            0.0,
            2.0 * torch.pi,
            env.n_agents + 1,
            device=env.device,
        )[:-1]
        fallback = torch.stack(
            (torch.cos(fallback_angles), torch.sin(fallback_angles)),
            dim=-1,
        )
        unit = torch.where(
            dist > 1.0e-6,
            rel / dist.clamp_min(1.0e-6),
            fallback[None, :, :],
        )
        target_xy = anchor_xy + unit * stop_radius
        target_error = target_xy - positions_xy
        world_delta = torch.where(
            torch.linalg.vector_norm(target_error, dim=-1, keepdim=True) > 1.0e-3,
            target_error,
            torch.zeros_like(target_error),
        )
    elif visible_local:
        visible = compute_visibility_mask(
            env.positions,
            float(env.cfg.observation.communication_radius),
        )
        visible_f = visible.to(dtype=positions_xy.dtype)
        local_sum = torch.einsum("eij,ejd->eid", visible_f, positions_xy) + positions_xy
        local_count = visible_f.sum(dim=-1, keepdim=True) + 1.0
        centroid_xy = local_sum / local_count
        has_neighbor = visible.any(dim=-1, keepdim=True)
    else:
        centroid_xy = positions_xy.mean(dim=1, keepdim=True).expand_as(positions_xy)
        has_neighbor = torch.ones_like(positions_xy[..., :1], dtype=torch.bool)
    if teacher_mode not in {
        "oracle_ring",
        "oracle_translating_ring",
        "oracle_slots",
        "terminal_flat_slots",
    }:
        rel = positions_xy - centroid_xy
        dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        fallback_angles = torch.linspace(
            0.0,
            2.0 * torch.pi,
            env.n_agents + 1,
            device=env.device,
        )[:-1]
        fallback = torch.stack(
            (torch.cos(fallback_angles), torch.sin(fallback_angles)),
            dim=-1,
        )
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
    terminal_state_fraction: float = 0.0,
    terminal_spawn_radius_min: float = 0.35,
    terminal_spawn_radius_max: float = 0.65,
    terminal_jitter_std: float = 0.04,
) -> None:
    if not 0.0 <= terminal_state_fraction <= 1.0:
        raise ValueError("terminal_state_fraction must be in [0, 1].")
    if terminal_spawn_radius_min <= 0.0:
        raise ValueError("terminal_spawn_radius_min must be positive.")
    if terminal_spawn_radius_max < terminal_spawn_radius_min:
        raise ValueError(
            "terminal_spawn_radius_max must be >= terminal_spawn_radius_min."
        )
    if terminal_jitter_std < 0.0:
        raise ValueError("terminal_jitter_std must be non-negative.")
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
        if terminal_state_fraction > 0.0:
            terminal_mask = torch.rand(
                count,
                1,
                1,
                device=env.device,
                generator=env.generator,
            ) < terminal_state_fraction
            terminal_radius = torch.empty(count, 1, 1, device=env.device).uniform_(
                terminal_spawn_radius_min,
                terminal_spawn_radius_max,
                generator=env.generator,
            )
            radius = torch.where(terminal_mask, terminal_radius, radius)
            jitter_scale = torch.where(
                terminal_mask,
                torch.full_like(radius, terminal_jitter_std),
                torch.full_like(radius, 0.35),
            )
        else:
            jitter_scale = torch.full_like(radius, 0.35)
        jitter = jitter_scale * torch.randn(
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
        visible = compute_visibility_mask(
            env.positions,
            float(env.cfg.observation.communication_radius),
        )
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
    env.refresh_oracle_point()


def _collect_on_policy_tail_bc_samples(
    policy: Model,
    env,
    *,
    rollout_steps: int,
    teacher_stop_radius: float,
    teacher_slow_distance: float,
    teacher_max_rho: float | None,
    teacher_mode: str,
    teacher_terrain_scale: bool,
    teacher_center_step: float,
    dmax_multiplier: float,
    dispersion_multiplier: float,
    min_teacher_disagreement: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect teacher labels from policy-visited near-terminal states.

    Random reset snapshots contain little of the late geometry/terrain
    coupling that causes the 96-second timeout failures.  This collector
    freezes the current policy, rolls it out, and retains only states close to
    both geometric terminal gates.  The actor observation and the teacher
    label are captured before the same action, preserving the execution-goal
    contract without reward or termination changes.
    """
    if rollout_steps <= 0:
        raise ValueError("bc_on_policy_rollout_steps must be positive.")
    if dmax_multiplier < 1.0:
        raise ValueError("bc_on_policy_dmax_multiplier must be >= 1.0.")
    if dispersion_multiplier < 1.0:
        raise ValueError("bc_on_policy_dispersion_multiplier must be >= 1.0.")
    if min_teacher_disagreement < 0.0:
        raise ValueError("bc_on_policy_min_teacher_disagreement must be non-negative.")

    was_training = policy.training
    policy.eval()
    actor_obs, _ = env.reset()
    observations: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(rollout_steps):
            metrics = compute_team_metrics(env.positions, env.velocities_xy)
            near_terminal = (
                metrics.dmax
                <= float(env.cfg.success_thresholds.dmax) * dmax_multiplier
            ) & (
                metrics.dispersion
                <= float(env.cfg.success_thresholds.dispersion) * dispersion_multiplier
            )
            teacher = scripted_gather_action(
                env,
                stop_radius=teacher_stop_radius,
                slow_distance=teacher_slow_distance,
                max_rho=teacher_max_rho,
                visible_local=teacher_mode == "visible_local_centroid",
                terrain_scale=teacher_terrain_scale,
                teacher_mode=teacher_mode,
                teacher_center_step=teacher_center_step,
            )
            action, _ = policy.compute(
                {"observations": actor_obs.reshape(-1, actor_obs.shape[-1])},
                role="policy",
            )
            action = action.reshape(env.num_envs, env.n_agents, -1)
            if min_teacher_disagreement > 0.0:
                disagreement = (action - teacher).square().mean(dim=(1, 2))
                near_terminal = near_terminal & (
                    disagreement >= min_teacher_disagreement
                )
            if near_terminal.any():
                observations.append(actor_obs[near_terminal].reshape(-1, actor_obs.shape[-1]))
                targets.append(teacher[near_terminal].reshape(-1, teacher.shape[-1]))
            step_output = env.step(action)
            actor_obs = step_output.actor_obs
    policy.train(was_training)
    if not observations:
        raise RuntimeError(
            "On-policy BC rollout produced no near-terminal samples; "
            "increase rollout steps or terminal multipliers."
        )
    return torch.cat(observations).detach(), torch.cat(targets).detach()


def _collect_teacher_rollout_tail_bc_samples(
    env,
    *,
    rollout_steps: int,
    teacher_stop_radius: float,
    teacher_slow_distance: float,
    teacher_max_rho: float | None,
    teacher_mode: str,
    teacher_terrain_scale: bool,
    teacher_center_step: float,
    dmax_multiplier: float,
    dispersion_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect terminal labels along the scripted teacher's own trajectory.

    The direct fixed-slot and dynamic-flat teachers can complete more 96-second
    episodes than the learned policy.  Unlike on-policy correction data, this
    reservoir retains the states on the teacher's successful approach into the
    terminal region, while preserving the same actor observation and action
    contract used at execution time.
    """
    if rollout_steps <= 0:
        raise ValueError("bc_teacher_rollout_steps must be positive.")
    if dmax_multiplier < 1.0:
        raise ValueError("bc_teacher_dmax_multiplier must be >= 1.0.")
    if dispersion_multiplier < 1.0:
        raise ValueError("bc_teacher_dispersion_multiplier must be >= 1.0.")

    actor_obs, _ = env.reset()
    observations: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(rollout_steps):
            metrics = compute_team_metrics(env.positions, env.velocities_xy)
            near_terminal = (
                metrics.dmax
                <= float(env.cfg.success_thresholds.dmax) * dmax_multiplier
            ) & (
                metrics.dispersion
                <= float(env.cfg.success_thresholds.dispersion)
                * dispersion_multiplier
            )
            teacher = scripted_gather_action(
                env,
                stop_radius=teacher_stop_radius,
                slow_distance=teacher_slow_distance,
                max_rho=teacher_max_rho,
                visible_local=teacher_mode == "visible_local_centroid",
                terrain_scale=teacher_terrain_scale,
                teacher_mode=teacher_mode,
                teacher_center_step=teacher_center_step,
            )
            if near_terminal.any():
                observations.append(actor_obs[near_terminal].reshape(-1, actor_obs.shape[-1]))
                targets.append(teacher[near_terminal].reshape(-1, teacher.shape[-1]))
            actor_obs = env.step(teacher).actor_obs
    if not observations:
        raise RuntimeError(
            "Teacher rollout produced no near-terminal samples; increase rollout "
            "steps or terminal multipliers."
        )
    return torch.cat(observations).detach(), torch.cat(targets).detach()


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
    teacher_center_step: float = 0.65,
    bc_yaw_noise_degrees: float | None = None,
    bc_min_nearest_distance: float | None = None,
    bc_terminal_state_fraction: float = 0.0,
    bc_terminal_spawn_radius_min: float = 0.35,
    bc_terminal_spawn_radius_max: float = 0.65,
    bc_terminal_jitter_std: float = 0.04,
    bc_on_policy_rollout_steps: int = 0,
    bc_on_policy_tail_fraction: float = 0.0,
    bc_on_policy_dmax_multiplier: float = 2.0,
    bc_on_policy_dispersion_multiplier: float = 2.0,
    bc_on_policy_min_teacher_disagreement: float = 0.0,
    bc_teacher_rollout_steps: int = 0,
    bc_teacher_tail_fraction: float = 0.0,
    bc_teacher_dmax_multiplier: float = 2.0,
    bc_teacher_dispersion_multiplier: float = 2.0,
    bc_on_policy_anchor_base_policy: bool = False,
) -> list[dict[str, float | int | str]]:
    """Warm-start the shared SKRL actor; MAPPO itself remains teacher-loss free."""
    if updates <= 0:
        return []
    if teacher_mode not in {
        "global_centroid",
        "visible_local_centroid",
        "oracle_ring",
        "oracle_translating_ring",
        "oracle_slots",
        "terminal_flat_slots",
    }:
        raise ValueError(
            "teacher_mode must be one of: global_centroid, "
            "visible_local_centroid, oracle_ring, oracle_translating_ring, "
            "oracle_slots, terminal_flat_slots."
        )
    if not 0.0 <= bc_on_policy_tail_fraction <= 1.0:
        raise ValueError("bc_on_policy_tail_fraction must be in [0, 1].")
    if bc_on_policy_tail_fraction > 0.0 and bc_on_policy_rollout_steps <= 0:
        raise ValueError(
            "bc_on_policy_rollout_steps must be positive when "
            "bc_on_policy_tail_fraction is nonzero."
        )
    if not 0.0 <= bc_teacher_tail_fraction <= 1.0:
        raise ValueError("bc_teacher_tail_fraction must be in [0, 1].")
    if bc_teacher_tail_fraction > 0.0 and bc_teacher_rollout_steps <= 0:
        raise ValueError(
            "bc_teacher_rollout_steps must be positive when "
            "bc_teacher_tail_fraction is nonzero."
        )
    if bc_on_policy_tail_fraction > 0.0 and bc_teacher_tail_fraction > 0.0:
        raise ValueError(
            "Use either on-policy or teacher-rollout tail BC in one run; "
            "mixing both reservoirs obscures the terminal supervision source."
        )
    tail_fraction = max(bc_on_policy_tail_fraction, bc_teacher_tail_fraction)
    if bc_on_policy_anchor_base_policy and tail_fraction <= 0.0:
        raise ValueError(
            "bc_on_policy_anchor_base_policy requires a nonzero "
            "tail BC fraction."
        )
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
    tail_obs: torch.Tensor | None = None
    tail_targets: torch.Tensor | None = None
    tail_source: str | None = None
    anchor_policy: Model | None = None
    if bc_on_policy_tail_fraction > 0.0:
        tail_obs, tail_targets = _collect_on_policy_tail_bc_samples(
            policy,
            env,
            rollout_steps=bc_on_policy_rollout_steps,
            teacher_stop_radius=teacher_stop_radius,
            teacher_slow_distance=teacher_slow_distance,
            teacher_max_rho=teacher_max_rho,
            teacher_mode=teacher_mode,
            teacher_terrain_scale=teacher_terrain_scale,
            teacher_center_step=teacher_center_step,
            dmax_multiplier=bc_on_policy_dmax_multiplier,
            dispersion_multiplier=bc_on_policy_dispersion_multiplier,
            min_teacher_disagreement=bc_on_policy_min_teacher_disagreement,
        )
        tail_source = "on_policy"
    elif bc_teacher_tail_fraction > 0.0:
        tail_obs, tail_targets = _collect_teacher_rollout_tail_bc_samples(
            env,
            rollout_steps=bc_teacher_rollout_steps,
            teacher_stop_radius=teacher_stop_radius,
            teacher_slow_distance=teacher_slow_distance,
            teacher_max_rho=teacher_max_rho,
            teacher_mode=teacher_mode,
            teacher_terrain_scale=teacher_terrain_scale,
            teacher_center_step=teacher_center_step,
            dmax_multiplier=bc_teacher_dmax_multiplier,
            dispersion_multiplier=bc_teacher_dispersion_multiplier,
        )
        tail_source = "teacher_rollout"
    if bc_on_policy_anchor_base_policy and tail_obs is not None:
        # Preserve the source policy away from the diagnosed terminal states.
        # This makes the BC update a local correction instead of re-anchoring
        # global approach behaviour to the scripted teacher.
        anchor_policy = copy.deepcopy(policy).eval()

    for update in range(1, updates + 1):
        observations = []
        targets = []
        for _ in range(snapshots_per_batch):
            _randomize_bc_state(
                env,
                visible_local=teacher_mode == "visible_local_centroid",
                yaw_noise_degrees=bc_yaw_noise_degrees,
                min_nearest_distance=bc_min_nearest_distance,
                terminal_state_fraction=bc_terminal_state_fraction,
                terminal_spawn_radius_min=bc_terminal_spawn_radius_min,
                terminal_spawn_radius_max=bc_terminal_spawn_radius_max,
                terminal_jitter_std=bc_terminal_jitter_std,
            )
            actor_obs, _ = env.get_observations()
            if anchor_policy is None:
                target = scripted_gather_action(
                    env,
                    stop_radius=teacher_stop_radius,
                    slow_distance=teacher_slow_distance,
                    max_rho=teacher_max_rho,
                    visible_local=teacher_mode == "visible_local_centroid",
                    terrain_scale=teacher_terrain_scale,
                    teacher_mode=teacher_mode,
                    teacher_center_step=teacher_center_step,
                )
            else:
                with torch.no_grad():
                    target, _ = anchor_policy.compute(
                        {"observations": actor_obs.reshape(-1, actor_obs.shape[-1])},
                        role="policy",
                    )
                target = target.reshape(env.num_envs, env.n_agents, -1)
            observations.append(actor_obs.reshape(-1, actor_obs.shape[-1]).detach())
            targets.append(target.reshape(-1, target.shape[-1]).detach())
        normal_count = batch_size
        if tail_obs is not None and tail_targets is not None:
            tail_count = int(round(batch_size * tail_fraction))
            normal_count = batch_size - tail_count
            if tail_count > 0:
                indices = torch.randint(
                    tail_obs.shape[0],
                    (tail_count,),
                    device=env.device,
                    generator=env.generator,
                )
                observations.append(tail_obs[indices])
                targets.append(tail_targets[indices])
        obs = torch.cat(observations, dim=0)[:normal_count]
        target = torch.cat(targets, dim=0)[:normal_count]
        if tail_obs is not None and tail_targets is not None and tail_count > 0:
            obs = torch.cat((obs, observations[-1]), dim=0)
            target = torch.cat((target, targets[-1]), dim=0)
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
                "bc_on_policy_tail_samples": (
                    int(tail_obs.shape[0]) if tail_source == "on_policy" else 0
                ),
                "bc_teacher_rollout_tail_samples": (
                    int(tail_obs.shape[0]) if tail_source == "teacher_rollout" else 0
                ),
                "bc_on_policy_anchor_base_policy": int(
                    bc_on_policy_anchor_base_policy
                ),
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
PROGRESS_PRESERVING_LONG_THRESHOLDS = {
    "dmax_reduction_ratio": 0.30,
    "success_rate": 0.20,
    "collision_rate": 0.08,
    "timeout_rate": 0.80,
}
SUCCESS_PROGRESS_LONG_THRESHOLDS = {
    "dmax_reduction_ratio": 0.20,
    "success_rate": 0.50,
    "collision_rate": 0.10,
    "timeout_rate": 0.20,
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
    timeout = float(metrics["timeout_rate"])
    timeout_threshold = float(thresholds["timeout_rate"])
    timeout_violation = (
        timeout
        if timeout_threshold == 0.0
        else max(0.0, timeout / max(timeout_threshold, 1.0e-6) - 1.0)
    )
    violation = (
        max(0.0, float(metrics["dmax_reduction_ratio"]) / thresholds["dmax_reduction_ratio"] - 1.0)
        + max(0.0, 1.0 - float(metrics["success_rate"]) / thresholds["success_rate"])
        + max(0.0, float(metrics["collision_rate"]) / thresholds["collision_rate"] - 1.0)
        + timeout_violation
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


def progress_preserving_long_checkpoint_rank(
    metrics: dict,
    thresholds: dict | None = None,
) -> tuple[int, float, float, float, float, float, int]:
    thresholds = thresholds or PROGRESS_PRESERVING_LONG_THRESHOLDS
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
    success = float(metrics.get("success_rate", 0.0))
    dmax = float(metrics.get("dmax_reduction_ratio", float("inf")))
    collision = float(metrics.get("collision_rate", float("inf")))
    timeout = float(metrics.get("timeout_rate", float("inf")))
    has_progress = (
        success >= thresholds["success_rate"]
        or dmax <= thresholds["dmax_reduction_ratio"]
    )
    progress_violation = (
        max(0.0, dmax / thresholds["dmax_reduction_ratio"] - 1.0)
        + max(0.0, 1.0 - success / thresholds["success_rate"])
    )
    safety_violation = (
        max(0.0, collision / thresholds["collision_rate"] - 1.0)
        + max(0.0, timeout / thresholds["timeout_rate"] - 1.0)
    )
    return (
        1 if has_progress else 2,
        progress_violation,
        safety_violation,
        collision,
        timeout,
        -success,
        -int(metrics.get("candidate_timestep", 0)),
    )


def success_progress_long_checkpoint_rank(
    metrics: dict,
    thresholds: dict | None = None,
) -> tuple[int, float, float, float, float, float, int]:
    thresholds = thresholds or SUCCESS_PROGRESS_LONG_THRESHOLDS
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
    success = float(metrics.get("success_rate", 0.0))
    dmax = float(metrics.get("dmax_reduction_ratio", float("inf")))
    collision = float(metrics.get("collision_rate", float("inf")))
    timeout = float(metrics.get("timeout_rate", float("inf")))
    strong_progress = (
        success >= thresholds["success_rate"]
        and dmax <= thresholds["dmax_reduction_ratio"]
    )
    moderate_progress = success >= 0.20 or dmax <= 0.30
    progress_violation = (
        max(0.0, dmax / thresholds["dmax_reduction_ratio"] - 1.0)
        + max(0.0, 1.0 - success / thresholds["success_rate"])
    )
    safety_violation = (
        max(0.0, collision / thresholds["collision_rate"] - 1.0)
        + max(0.0, timeout / thresholds["timeout_rate"] - 1.0)
    )
    if strong_progress:
        return (
            1,
            safety_violation,
            -success,
            collision,
            timeout,
            dmax,
            -int(metrics.get("candidate_timestep", 0)),
        )
    return (
        2 if moderate_progress else 3,
        progress_violation,
        safety_violation,
        collision,
        timeout,
        -success,
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
            "mutual_path_near": float(filter_cfg.mutual_path_near_weight),
            "mutual_path_collision": float(filter_cfg.mutual_path_collision_weight),
            "visible_neighbor_center": float(filter_cfg.visible_neighbor_center_weight),
            "center_progress": float(filter_cfg.center_progress_weight),
            "hold_zone_rho": float(filter_cfg.hold_zone_rho_weight),
            "hold_zone_spacing": float(filter_cfg.hold_zone_spacing_weight),
        },
        "constraints": {
            "endpoint_safe_distance": float(filter_cfg.endpoint_safe_distance),
            "path_safe_distance": float(filter_cfg.path_safe_distance),
            "hard_endpoint_near_filter": bool(filter_cfg.hard_endpoint_near_filter),
            "hard_path_collision_filter": bool(filter_cfg.hard_path_collision_filter),
            "hard_center_progress_filter": bool(filter_cfg.hard_center_progress_filter),
            "center_progress_slack": float(filter_cfg.center_progress_slack),
            "center_progress_margin": float(filter_cfg.center_progress_margin),
            "hard_constraint_penalty": float(filter_cfg.hard_constraint_penalty),
            "safety_override_after_warmup": bool(
                filter_cfg.safety_override_after_warmup
            ),
            "collision_override_after_warmup": bool(
                filter_cfg.collision_override_after_warmup
            ),
            "hold_zone_override_after_warmup": bool(
                filter_cfg.hold_zone_override_after_warmup
            ),
            "hold_zone_dmax_multiplier": float(filter_cfg.hold_zone_dmax_multiplier),
            "hold_zone_dispersion_multiplier": float(
                filter_cfg.hold_zone_dispersion_multiplier
            ),
            "hold_zone_pairwise_distance": float(filter_cfg.hold_zone_pairwise_distance),
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


def control_safety_metadata(cfg) -> dict[str, Any]:
    control_cfg = cfg.low_level_control
    return {
        "safety_projection_enabled": bool(control_cfg.safety_projection_enabled),
        "projection_activation_distance": float(control_cfg.projection_activation_distance),
        "projection_stop_distance": float(control_cfg.projection_stop_distance),
        "projection_horizon_s": float(control_cfg.projection_horizon_s),
        "projection_strength": float(control_cfg.projection_strength),
        "projection_min_linear_scale": float(control_cfg.projection_min_linear_scale),
        "projection_damp_nonclosing_near": bool(
            control_cfg.projection_damp_nonclosing_near
        ),
        "projection_directional_agent_scale": bool(
            control_cfg.projection_directional_agent_scale
        ),
        "projection_directional_agent_scale_mode": str(
            control_cfg.projection_directional_agent_scale_mode
        ),
        "success_zone_damping_enabled": bool(control_cfg.success_zone_damping_enabled),
        "success_zone_dmax_multiplier": float(control_cfg.success_zone_dmax_multiplier),
        "success_zone_dispersion_multiplier": float(
            control_cfg.success_zone_dispersion_multiplier
        ),
        "success_zone_linear_scale": float(control_cfg.success_zone_linear_scale),
        "formation_center_correction_enabled": bool(
            control_cfg.formation_center_correction_enabled
        ),
        "formation_center_activation_dmax_multiplier": float(
            control_cfg.formation_center_activation_dmax_multiplier
        ),
        "formation_center_activation_dispersion_multiplier": float(
            control_cfg.formation_center_activation_dispersion_multiplier
        ),
        "formation_center_correction_max_offset": float(
            control_cfg.formation_center_correction_max_offset
        ),
        "formation_center_correction_gain": float(
            control_cfg.formation_center_correction_gain
        ),
        "formation_center_correction_require_flatness_failure": bool(
            control_cfg.formation_center_correction_require_flatness_failure
        ),
        "terminal_slot_capture_enabled": bool(control_cfg.terminal_slot_capture_enabled),
        "terminal_slot_capture_dmax_multiplier": float(
            control_cfg.terminal_slot_capture_dmax_multiplier
        ),
        "terminal_slot_capture_dispersion_multiplier": float(
            control_cfg.terminal_slot_capture_dispersion_multiplier
        ),
        "terminal_slot_capture_blend": float(control_cfg.terminal_slot_capture_blend),
        "flat_geometry_capture_enabled": bool(control_cfg.flat_geometry_capture_enabled),
        "flat_geometry_capture_dmax_multiplier": float(
            control_cfg.flat_geometry_capture_dmax_multiplier
        ),
        "flat_geometry_capture_dispersion_multiplier": float(
            control_cfg.flat_geometry_capture_dispersion_multiplier
        ),
        "flat_geometry_capture_blend": float(control_cfg.flat_geometry_capture_blend),
        "flat_geometry_capture_dynamic_assignment": bool(
            control_cfg.flat_geometry_capture_dynamic_assignment
        ),
    }


def environment_geometry_metadata(cfg) -> dict[str, Any]:
    """Return map-scale metadata that does not affect checkpoint compatibility."""

    world_half_width = float(cfg.safety.world_xy_limit)
    return {
        "world_xy_limit": world_half_width,
        "map_size_m": 2.0 * world_half_width,
        "terrain_crater_field_size": float(cfg.terrain.crater_field_size),
    }


def initial_state_metadata(cfg) -> dict[str, Any]:
    initial_state = cfg.initial_state
    return {
        "spawn_radius_min": float(initial_state.spawn_radius_min),
        "spawn_radius_max": float(initial_state.spawn_radius_max),
        "center_xy_range": float(initial_state.center_xy_range),
        "jitter_std": float(initial_state.jitter_std),
        "curriculum_enabled": bool(initial_state.curriculum_enabled),
        "curriculum_start_spawn_radius_min": float(
            initial_state.curriculum_start_spawn_radius_min
        ),
        "curriculum_start_spawn_radius_max": float(
            initial_state.curriculum_start_spawn_radius_max
        ),
        "curriculum_start_center_xy_range": float(
            initial_state.curriculum_start_center_xy_range
        ),
        "curriculum_start_jitter_std": float(initial_state.curriculum_start_jitter_std),
        "curriculum_warmup_timesteps": int(initial_state.curriculum_warmup_timesteps),
        "curriculum_ramp_timesteps": int(initial_state.curriculum_ramp_timesteps),
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
    actor_architecture: str = "mlp_v1",
    critic_architecture: str = "mlp_v1",
) -> dict[str, dict[str, Model]]:
    """Build MAPPO models with project CTDE semantics.

    Policies consume per-agent local observations. Values consume the centralized state returned
    by ``env.state()`` / ``env.state_space``.
    """
    if not centralized_critic:
        raise ValueError("This project only wires SKRL MAPPO with a centralized critic state.")
    if shared_actor or shared_value:
        _validate_homogeneous_spaces(env)
    actor_architecture = normalize_actor_architecture(actor_architecture)
    critic_architecture = normalize_critic_architecture(critic_architecture)

    first_agent = env.possible_agents[0]
    shared_policy = (
        SKRLPolicy(
            env.observation_spaces[first_agent],
            env.action_spaces[first_agent],
            env.device,
            initial_log_std,
            architecture=actor_architecture,
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
            architecture=critic_architecture,
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
                architecture=actor_architecture,
            ),
            "value": shared_critic
            if shared_value
            else SKRLValue(
                env.observation_spaces[agent_id],
                env.state_space,
                env.action_spaces[agent_id],
                env.device,
                architecture=critic_architecture,
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
    "flatness",
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
    timeout = done.truncated.detach().bool()
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
            if "reference_risk_mean" in path_terrain:
                path_metrics["path_terrain_reference_risk_mean"] = float(
                    path_terrain["reference_risk_mean"]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                )
                path_metrics["path_terrain_relative_risk_mean"] = float(
                    path_terrain["relative_risk_mean"]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                )
            telemetry_state["path_terrain"] = path_metrics
            _accumulate_numeric_metrics(telemetry_state, "path_terrain", path_metrics)
        actor_credit = info.get("actor_credit")
        if actor_credit is not None:
            centered = actor_credit["centered"].detach().float()
            policy_credit = actor_credit.get("policy", centered).detach().float()
            raw_credit = actor_credit.get("raw", policy_credit).detach().float()
            actor_credit_metrics = {
                "actor_credit_abs_mean": float(policy_credit.abs().mean().cpu()),
                "actor_credit_std": float(policy_credit.std().cpu()),
                "actor_credit_active_rate": float(
                    (raw_credit.abs() > 1.0e-8).float().mean().cpu()
                ),
                "actor_credit_zero_sum_error": float(
                    centered.sum(dim=1).abs().amax().cpu()
                ),
                "actor_credit_policy_step_sum_abs_max": float(
                    policy_credit.sum(dim=1).abs().amax().cpu()
                ),
                "actor_credit_policy_is_step_zero_sum": float(
                    bool(actor_credit.get("policy_is_step_zero_sum", True))
                ),
                "actor_credit_source_reconstruction_error": float(
                    actor_credit.get(
                        "source_reconstruction_error",
                        torch.zeros((), device=policy_credit.device),
                    )
                    .detach()
                    .float()
                    .amax()
                    .cpu()
                ),
                "actor_credit_team_reward_preservation_error": float(
                    actor_credit["team_reward_preservation_error"]
                    .detach()
                    .float()
                    .amax()
                    .cpu()
                ),
            }
            for info_key, metric_key in (
                ("collision_participant_rate", "actor_credit_collision_participant_rate"),
                ("collision_event", "actor_credit_collision_event_rate"),
                ("allocation_mean_error", "actor_credit_allocation_mean_error"),
                ("credit_zero_sum_error", "actor_credit_collision_zero_sum_error"),
            ):
                value = actor_credit.get(info_key)
                if isinstance(value, torch.Tensor):
                    actor_credit_metrics[metric_key] = float(
                        value.detach().float().mean().cpu()
                    )
            telemetry_state["actor_credit"] = actor_credit_metrics
            _accumulate_numeric_metrics(
                telemetry_state,
                "actor_credit",
                actor_credit_metrics,
            )
        centroid_flatness = info.get("centroid_flatness_reward")
        if centroid_flatness is not None:
            centroid_flatness_values = (
                torch.stack(
                    (
                        centroid_flatness["cost"].detach().float().mean(),
                        centroid_flatness["progress"].detach().float().mean(),
                        centroid_flatness["activation"].detach().float().mean(),
                    )
                )
                .cpu()
                .tolist()
            )
            centroid_flatness_metrics = {
                "centroid_flatness_cost_mean": float(centroid_flatness_values[0]),
                "centroid_flatness_progress_mean": float(centroid_flatness_values[1]),
                "centroid_flatness_activation_mean": float(centroid_flatness_values[2]),
            }
            telemetry_state["centroid_flatness"] = centroid_flatness_metrics
            _accumulate_numeric_metrics(
                telemetry_state,
                "centroid_flatness",
                centroid_flatness_metrics,
            )
        communication = info.get("communication")
        if communication is not None:
            communication_metrics = {
                key: float(value.detach().float().mean().cpu())
                for key, value in communication.items()
                if isinstance(value, torch.Tensor)
            }
            telemetry_state["communication"] = communication_metrics
            _accumulate_numeric_metrics(
                telemetry_state,
                "communication",
                communication_metrics,
            )
        trajectory_conflicts = info.get("trajectory_conflicts")
        if trajectory_conflicts is not None:
            conflict_metrics = {
                key: float(value.detach().float().mean().cpu())
                for key, value in trajectory_conflicts.items()
                if isinstance(value, torch.Tensor) and value.ndim <= 1
            }
            telemetry_state["mapf_conflicts"] = conflict_metrics
            _accumulate_numeric_metrics(
                telemetry_state,
                "mapf_conflicts",
                conflict_metrics,
            )
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
                "filter_mutual_path_near_violation_mean": float(
                    action_filter["mutual_path_near_violation"]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                ),
                "filter_mutual_path_collision_violation_mean": float(
                    action_filter["mutual_path_collision_violation"]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                ),
                "filter_mutual_path_collision_violation_fraction": float(
                    (action_filter["mutual_path_collision_violation"] > 0.0)
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
                "filter_raw_mutual_path_near_violation_mean": float(
                    action_filter["raw_mutual_path_near_violation"]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                ),
                "filter_raw_mutual_path_collision_violation_mean": float(
                    action_filter["raw_mutual_path_collision_violation"]
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
                "filter_collision_override_fraction": float(
                    action_filter.get("collision_override_fraction", 0.0)
                ),
                "filter_raw_visible_center_cost_mean": float(
                    action_filter["raw_visible_center_cost"].detach().float().mean().cpu()
                ),
                "filter_filtered_visible_center_cost_mean": float(
                    action_filter["filtered_visible_center_cost"]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                ),
                "filter_suggested_visible_center_cost_mean": float(
                    action_filter["suggested_visible_center_cost"]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                ),
                "filter_center_progress_regression_mean": float(
                    action_filter["center_progress_regression"]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                ),
                "filter_hold_zone_activation_mean": float(
                    action_filter["hold_zone_activation"].detach().float().mean().cpu()
                ),
                "filter_hold_zone_rho_cost_mean": float(
                    action_filter["hold_zone_rho_cost"].detach().float().mean().cpu()
                ),
                "filter_hold_zone_spacing_violation_mean": float(
                    action_filter["hold_zone_spacing_violation"]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                ),
                "filter_raw_hold_zone_rho_cost_mean": float(
                    action_filter["raw_hold_zone_rho_cost"].detach().float().mean().cpu()
                ),
                "filter_raw_hold_zone_spacing_violation_mean": float(
                    action_filter["raw_hold_zone_spacing_violation"]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
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
        control_safety = info.get("control_safety")
        if control_safety is not None:
            safety_metrics = {
                "control_safety_enabled": float(bool(control_safety.get("enabled", False))),
                "control_safety_applied_fraction": float(
                    control_safety["applied"].detach().float().mean().cpu()
                ),
                "control_safety_linear_scale_mean": float(
                    control_safety["linear_scale"].detach().float().mean().cpu()
                ),
                "control_safety_linear_scale_min": float(
                    control_safety["linear_scale"].detach().float().amin().cpu()
                ),
                "control_safety_pairwise_risk_mean": float(
                    control_safety["pairwise_risk"].detach().float().mean().cpu()
                ),
                "control_safety_predicted_nearest_mean": float(
                    control_safety["predicted_nearest_distance"].detach().float().mean().cpu()
                ),
                "control_safety_success_zone_fraction": float(
                    control_safety["success_zone_active"].detach().float().mean().cpu()
                ),
            }
            telemetry_state["control_safety"] = safety_metrics
            _accumulate_numeric_metrics(telemetry_state, "control_safety", safety_metrics)
        formation_center_correction = info.get("formation_center_correction")
        if formation_center_correction is not None:
            offset_norm = torch.linalg.norm(
                formation_center_correction["offset_xy"].detach().float(),
                dim=-1,
            )
            correction_metrics = {
                "formation_center_correction_active_fraction": float(
                    formation_center_correction["active"].detach().float().mean().cpu()
                ),
                "formation_center_correction_offset_mean": float(offset_norm.mean().cpu()),
                "formation_center_correction_offset_max": float(offset_norm.amax().cpu()),
            }
            telemetry_state["formation_center_correction"] = correction_metrics
            _accumulate_numeric_metrics(
                telemetry_state,
                "formation_center_correction",
                correction_metrics,
            )
        terminal_slot_capture = info.get("terminal_slot_capture")
        if terminal_slot_capture is not None:
            capture_metrics = {
                "terminal_slot_capture_active_fraction": float(
                    terminal_slot_capture["active"].detach().float().mean().cpu()
                ),
            }
            telemetry_state["terminal_slot_capture"] = capture_metrics
            _accumulate_numeric_metrics(
                telemetry_state,
                "terminal_slot_capture",
                capture_metrics,
            )
        flat_geometry_capture = info.get("flat_geometry_capture")
        if flat_geometry_capture is not None:
            capture_metrics = {
                "flat_geometry_capture_active_fraction": float(
                    flat_geometry_capture["active"].detach().float().mean().cpu()
                ),
            }
            telemetry_state["flat_geometry_capture"] = capture_metrics
            _accumulate_numeric_metrics(
                telemetry_state,
                "flat_geometry_capture",
                capture_metrics,
            )
        kinematics = info.get("kinematics")
        if kinematics is not None:
            turning_radius = kinematics["turning_radius"].detach().float()
            finite_turning_radius = turning_radius[torch.isfinite(turning_radius)]
            turning_radius_mean = (
                float(finite_turning_radius.mean().cpu())
                if finite_turning_radius.numel() > 0
                else 0.0
            )
            kinematics_metrics = {
                "kinematic_model_bicycle": float(
                    str(kinematics.get("kinematic_model", "unicycle")) == "bicycle"
                ),
                "steering_angle_abs_mean": float(
                    kinematics["steering_angle"].detach().float().abs().mean().cpu()
                ),
                "steering_angle_abs_max": float(
                    kinematics["steering_angle"].detach().float().abs().amax().cpu()
                ),
                "actual_yaw_rate_abs_mean": float(
                    kinematics["actual_yaw_rate"].detach().float().abs().mean().cpu()
                ),
                "turning_radius_finite_fraction": float(
                    torch.isfinite(turning_radius).float().mean().cpu()
                ),
                "turning_radius_mean": turning_radius_mean,
            }
            telemetry_state["kinematics"] = kinematics_metrics
            _accumulate_numeric_metrics(telemetry_state, "kinematics", kinematics_metrics)
        writer = telemetry_state.get("writer")
        interval = int(telemetry_state.get("interval", 0))
        if writer is not None and interval > 0 and telemetry_state["step"] % interval == 0:
            writer(telemetry_state["step"])
        return observations, rewards, terminated, truncated, info

    env.step = checked_step


def collision_participant_centered_credit(
    positions: torch.Tensor,
    collision_done: torch.Tensor,
    *,
    collision_distance: float,
    collision_penalty: float,
) -> dict[str, torch.Tensor]:
    """Reallocate the existing team collision terminal term across participants.

    The returned policy credit is the zero-sum residual between an allocation
    that penalizes only actual collision participants and the original team
    term copied to every agent. It is training-only and does not alter rewards.
    """

    if positions.ndim != 3 or positions.shape[-1] < 2:
        raise ValueError("positions must have shape [environment, agent, xyz].")
    if collision_done.shape != positions.shape[:1]:
        raise ValueError("collision_done must have shape [environment].")
    if collision_distance <= 0.0 or collision_penalty <= 0.0:
        raise ValueError("collision_distance and collision_penalty must be positive.")
    num_envs, n_agents = positions.shape[:2]
    distances = pairwise_distances_xy(positions[..., :2])
    eye = torch.eye(n_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
    colliding_pairs = (
        (distances < float(collision_distance))
        & ~eye
        & collision_done[:, None, None].bool()
    )
    participants = colliding_pairs.any(dim=2)
    participant_count = participants.sum(dim=1)
    missing = collision_done.bool() & (participant_count == 0)
    if missing.any():
        missing_ids = torch.nonzero(missing, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            "Collision termination has no actual participant under the configured "
            f"collision distance for environments {missing_ids}."
        )
    team_component = -float(collision_penalty) * collision_done.float()
    scale = float(n_agents) / participant_count.clamp_min(1).to(positions.dtype)
    allocated = (
        team_component[:, None]
        * scale[:, None]
        * participants.to(dtype=positions.dtype)
    )
    policy_credit = allocated - team_component[:, None]
    # Remove the tiny float32 remainder created by ratios such as 4 / 3. The
    # allocation itself remains available for an independent mean-preservation
    # audit; the Actor residual is required to be numerically step-zero-sum.
    policy_credit = policy_credit - policy_credit.mean(dim=1, keepdim=True)
    allocation_mean_error = (allocated.mean(dim=1) - team_component).abs()
    zero_sum_error = policy_credit.sum(dim=1).abs()
    return {
        "raw": policy_credit,
        "policy": policy_credit,
        "centered": policy_credit,
        "participants": participants,
        "participant_count": participant_count,
        "allocated": allocated,
        "team_component": team_component,
        "allocation_mean_error": allocation_mean_error,
        "zero_sum_error": zero_sum_error,
    }


def install_actor_credit_rewards(
    env: MultiRoverGatheringSKRLEnv,
    *,
    assignment: str,
) -> None:
    """Attach training-only Actor credit without changing environment rewards."""

    supported = {
        "terrain_relative_centered",
        "near_potential_local",
        "collision_participant_centered",
    }
    if assignment not in supported:
        raise ValueError(
            "actor_credit_assignment must be one of "
            f"{sorted(supported)} (or 'none' before installation)."
        )
    original_step = env.step

    def credited_step(actions):
        nearest_before = (
            env.core.metrics.nearest_neighbor_distance.detach().clone()
            if assignment == "near_potential_local"
            else None
        )
        observations, rewards, terminated, truncated, info = original_step(actions)
        if assignment == "terrain_relative_centered":
            path_terrain = info.get("path_terrain") or {}
            relative_risk = path_terrain.get("relative_risk_mean")
            if not isinstance(relative_risk, torch.Tensor):
                raise RuntimeError(
                    "terrain_relative_centered credit requires relative quintic path risk."
                )
            # Lower selected-vs-reference risk is better. Centering across
            # vehicles makes this historical credit exactly zero-sum.
            raw_credit = -relative_risk
            policy_credit = raw_credit - raw_credit.mean(dim=1, keepdim=True)
            centered_credit = policy_credit
            policy_is_step_zero_sum = True
            source_reconstruction_error = torch.zeros(
                raw_credit.shape[0],
                dtype=raw_credit.dtype,
                device=raw_credit.device,
            )
            credit_extras: dict[str, Any] = {}
        elif assignment == "near_potential_local":
            if nearest_before is None:
                raise RuntimeError("Missing pre-step nearest-neighbor distances.")
            metrics = info.get("metrics")
            nearest_after = getattr(metrics, "nearest_neighbor_distance", None)
            if not isinstance(nearest_after, torch.Tensor):
                raise RuntimeError(
                    "near_potential_local credit requires post-step nearest distances."
                )
            near_distance = float(env.cfg.safety.near_distance)
            before_potential = -torch.relu(near_distance - nearest_before)
            after_potential = -torch.relu(near_distance - nearest_after)
            raw_credit = after_potential - before_potential
            policy_credit = raw_credit
            centered_credit = raw_credit - raw_credit.mean(dim=1, keepdim=True)
            policy_is_step_zero_sum = False
            source_reconstruction_error = (
                raw_credit.mean(dim=1)
                - (after_potential.mean(dim=1) - before_potential.mean(dim=1))
            ).abs()
            credit_extras = {}
        else:
            done = info.get("done")
            collision_done = getattr(done, "collision", None)
            positions = info.get("positions")
            if not isinstance(collision_done, torch.Tensor) or not isinstance(
                positions, torch.Tensor
            ):
                raise RuntimeError(
                    "collision_participant_centered credit requires collision flags "
                    "and pre-reset positions."
                )
            coefficients = env.cfg.reward_coefficients
            weights = env.cfg.reward_weights
            collision_penalty = (
                float(weights.safety) * float(coefficients.inter_agent_collision)
                + float(weights.terminal) * float(coefficients.failure_penalty)
            )
            allocation = collision_participant_centered_credit(
                positions,
                collision_done,
                collision_distance=float(env.cfg.safety.collision_distance),
                collision_penalty=collision_penalty,
            )
            raw_credit = allocation["raw"]
            policy_credit = allocation["policy"]
            centered_credit = allocation["centered"]
            policy_is_step_zero_sum = True
            source_reconstruction_error = allocation["allocation_mean_error"]
            credit_extras = {
                "collision_participant_rate": allocation["participants"]
                .float()
                .mean(dim=1),
                "collision_event": collision_done.float(),
                "allocation_mean_error": allocation["allocation_mean_error"],
                "credit_zero_sum_error": allocation["zero_sum_error"],
                "collision_penalty": torch.full_like(
                    collision_done.float(), collision_penalty
                ),
            }
        team_reward_before = torch.stack(
            [rewards[agent] for agent in env.possible_agents], dim=1
        ).mean(dim=1)
        info["actor_credit"] = {
            "assignment": assignment,
            "raw": raw_credit.detach().clone(),
            "centered": centered_credit.detach().clone(),
            "policy": policy_credit.detach().clone(),
            "policy_is_step_zero_sum": policy_is_step_zero_sum,
            "source_reconstruction_error": source_reconstruction_error.detach().clone(),
            # Rewards are returned untouched. Keep an explicit zero diagnostic
            # so the training screen can enforce this invariant.
            "team_reward_preservation_error": torch.zeros_like(team_reward_before),
            **credit_extras,
        }
        return observations, rewards, terminated, truncated, info

    env.step = credited_step


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
    gather_point_flatness = env.core.evaluate_current_gather_point_flatness(metrics)
    flatness_ok = (
        gather_point_flatness.is_flat
        if env.cfg.gather_point.require_flat_for_success
        else torch.ones_like(gather_point_flatness.is_flat)
    )
    success_gates = compute_success_gates(
        metrics,
        env.core.velocities_xy,
        env.cfg.success_thresholds,
        flatness_ok=flatness_ok,
    )
    pairwise = metrics.mean_pairwise_distance
    oracle = mean_oracle_distance
    threshold = float(env.cfg.success_thresholds.dmax)
    nearest = metrics.nearest_neighbor_distance.amin(dim=-1)
    effective_initial_state = env.core._effective_initial_state_values()
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
        "dmax_mean": float(metrics.dmax.mean().detach().cpu()),
        "final_nearest_neighbor_distance": float(nearest.mean().detach().cpu()),
        "mean_oracle_distance": float(mean_oracle_distance.mean().detach().cpu()),
        "success_rate": float(success_gates.instant_success.float().mean().detach().cpu()),
        "safe_success_rate": float(success_gates.instant_success.float().mean().detach().cpu()),
        "min_pairwise_ok_rate": float(success_gates.min_pairwise_ok.float().mean().detach().cpu()),
        "flatness_ok_rate": float(success_gates.flatness_ok.float().mean().detach().cpu()),
        "gather_point_height_range_mean": float(
            gather_point_flatness.height_range.mean().detach().cpu()
        ),
        "gather_point_max_slope_mean": float(
            gather_point_flatness.max_slope.mean().detach().cpu()
        ),
        "oracle_search_feasible_rate": float(
            env.core.oracle_search_feasible.float().mean().detach().cpu()
        ),
        "oracle_search_objective_mean": float(
            env.core.oracle_search_objective.mean().detach().cpu()
        ),
        "nan_flag": False,
        "checkpoint_path": str(checkpoint_path),
        "training_semantics": training_semantics,
        "observation_schema_version": env.cfg.observation.schema_version,
        "actor_obs_dim": env.cfg.actor_obs_dim,
        "critic_state_dim": env.cfg.critic_state_dim,
        "kinematic_model": env.cfg.low_level_control.kinematic_model,
        "trajectory_geometry_method": env.cfg.trajectory_generator.geometry_method,
        "initial_state_curriculum_enabled": bool(
            env.cfg.initial_state.curriculum_enabled
        ),
        "initial_state_progress_timestep": int(
            env.cfg.initial_state.progress_timestep_override
        ),
        "initial_state_effective_spawn_radius_min": effective_initial_state[0],
        "initial_state_effective_spawn_radius_max": effective_initial_state[1],
        "initial_state_effective_center_xy_range": effective_initial_state[2],
        "initial_state_effective_jitter_std": effective_initial_state[3],
        "success_threshold": {
            "dmax": float(env.cfg.success_thresholds.dmax),
            "dispersion": float(env.cfg.success_thresholds.dispersion),
            "speed": float(env.cfg.success_thresholds.speed),
            "hold_steps": int(env.cfg.success_thresholds.hold_steps),
            "min_pairwise_distance": float(env.cfg.success_thresholds.min_pairwise_distance),
        },
        "gather_point_flatness": {
            "require_flat_for_success": bool(
                env.cfg.gather_point.require_flat_for_success
            ),
            "radius": float(env.cfg.gather_point.flatness_radius),
            "max_height_range": float(env.cfg.gather_point.max_height_range),
            "max_slope": float(env.cfg.gather_point.max_slope),
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
    actor_credit_metrics = dict(telemetry_state.get("actor_credit", {}))
    actor_credit_metrics.update(telemetry_state.get("actor_credit_window", {}))
    centroid_flatness_metrics = dict(
        telemetry_state.get("centroid_flatness", {})
    )
    centroid_flatness_metrics.update(
        telemetry_state.get("centroid_flatness_window", {})
    )
    filter_metrics = dict(telemetry_state.get("action_filter", {}))
    filter_metrics.update(telemetry_state.get("action_filter_window", {}))
    control_safety_metrics = dict(telemetry_state.get("control_safety", {}))
    control_safety_metrics.update(telemetry_state.get("control_safety_window", {}))
    formation_center_correction_metrics = dict(
        telemetry_state.get("formation_center_correction", {})
    )
    formation_center_correction_metrics.update(
        telemetry_state.get("formation_center_correction_window", {})
    )
    terminal_slot_capture_metrics = dict(telemetry_state.get("terminal_slot_capture", {}))
    terminal_slot_capture_metrics.update(
        telemetry_state.get("terminal_slot_capture_window", {})
    )
    flat_geometry_capture_metrics = dict(telemetry_state.get("flat_geometry_capture", {}))
    flat_geometry_capture_metrics.update(
        telemetry_state.get("flat_geometry_capture_window", {})
    )
    kinematics_metrics = dict(telemetry_state.get("kinematics", {}))
    kinematics_metrics.update(telemetry_state.get("kinematics_window", {}))
    communication_metrics = dict(telemetry_state.get("communication", {}))
    communication_metrics.update(telemetry_state.get("communication_window", {}))
    mapf_conflict_metrics = dict(telemetry_state.get("mapf_conflicts", {}))
    mapf_conflict_metrics.update(telemetry_state.get("mapf_conflicts_window", {}))
    telemetry.update(reward_metrics)
    telemetry.update(action_metrics)
    telemetry.update(path_metrics)
    telemetry.update(actor_credit_metrics)
    telemetry.update(centroid_flatness_metrics)
    telemetry.update(filter_metrics)
    telemetry.update(control_safety_metrics)
    telemetry.update(formation_center_correction_metrics)
    telemetry.update(terminal_slot_capture_metrics)
    telemetry.update(flat_geometry_capture_metrics)
    telemetry.update(kinematics_metrics)
    telemetry.update(communication_metrics)
    telemetry.update(mapf_conflict_metrics)
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
    collision_cost_value: Model | None = None,
    lagrangian_multiplier: float | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    payload = {
        agent_id: {
            "policy": models[agent_id]["policy"].state_dict(),
            "value": models[agent_id]["value"].state_dict(),
        }
        for agent_id in possible_agents
    }
    if collision_cost_value is not None:
        if lagrangian_multiplier is None:
            raise ValueError(
                "lagrangian_multiplier is required with collision_cost_value."
            )
        payload["collision_constraint"] = {
            "cost_value": collision_cost_value.state_dict(),
            "lagrangian_multiplier": float(lagrangian_multiplier),
        }
    experiment = raw_cfg.get("experiment", {}) if isinstance(raw_cfg.get("experiment", {}), dict) else {}
    algorithm = raw_cfg.get("algorithm", {}) if isinstance(raw_cfg.get("algorithm", {}), dict) else {}
    terrain = raw_cfg.get("terrain", {}) if isinstance(raw_cfg.get("terrain", {}), dict) else {}
    low_level_control = (
        raw_cfg.get("low_level_control", {})
        if isinstance(raw_cfg.get("low_level_control", {}), dict)
        else {}
    )
    trajectory_generator = (
        raw_cfg.get("trajectory_generator", {})
        if isinstance(raw_cfg.get("trajectory_generator", {}), dict)
        else {}
    )
    model_actor_architecture = getattr(
        models[possible_agents[0]]["policy"],
        "architecture",
        "mlp_v1",
    )
    model_critic_architecture = getattr(
        models[possible_agents[0]]["value"],
        "architecture",
        "mlp_v1",
    )
    actor_architecture = normalize_actor_architecture(
        algorithm.get("actor_architecture", model_actor_architecture)
    )
    critic_architecture = normalize_critic_architecture(
        algorithm.get("critic_architecture", model_critic_architecture)
    )
    payload["metadata"] = {
        "training_semantics": training_semantics,
        "backend": "skrl.mappo",
        "experiment_name": experiment.get("name"),
        "algorithm_mode": algorithm.get("mode"),
        "actor_architecture": actor_architecture,
        "critic_architecture": critic_architecture,
        "observation_slices": observation_slices_metadata(
            int(actor_obs_dim or 86)
        ),
        "critic_state_slices": critic_state_slices_metadata(
            int(critic_state_dim or 54)
        ),
        "kinematic_model": str(low_level_control.get("kinematic_model", "unicycle")),
        "wheelbase_m": float(low_level_control.get("wheelbase_m", 0.65)),
        "max_steer_angle_rad": float(
            low_level_control.get("max_steer_angle_rad", 0.610865)
        ),
        "trajectory_geometry_method": str(
            trajectory_generator.get("geometry_method", "line")
        ),
        "observation_schema_version": observation_schema_version,
        "actor_obs_dim": actor_obs_dim,
        "critic_state_dim": critic_state_dim,
        "shared_actor": shared_actor,
        "centralized_critic": centralized_critic,
        "shared_value": shared_value,
        "timesteps": timesteps,
        "device": device,
        "checkpoint_path": checkpoint_path,
        "collision_constraint_enabled": collision_cost_value is not None,
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


def initialize_skrl_mappo_models_from_checkpoint(
    checkpoint_path: str | Path,
    models: dict[str, dict[str, Model]],
    possible_agents: list[str],
    cfg,
    *,
    actor_architecture: str,
    critic_architecture: str,
    device: torch.device,
) -> dict[str, Any]:
    """Load compatible policy/value weights while intentionally resetting optimizers.

    This is a warm-start, not a resume: rollout memories, optimizer moments and
    scheduler state are deliberately rebuilt for the active environment/control
    configuration.  Shared target modules are loaded only once from rover_0.
    """
    resolved = Path(checkpoint_path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    if not resolved.is_file():
        raise FileNotFoundError(f"Initial checkpoint does not exist: {resolved}")
    checkpoint = torch.load(resolved, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Initial checkpoint payload must be a mapping.")
    metadata = validate_checkpoint_compatibility(
        checkpoint,
        cfg,
        expected_actor_architecture=actor_architecture,
        expected_critic_architecture=critic_architecture,
    )
    missing_agents = [agent_id for agent_id in possible_agents if agent_id not in checkpoint]
    if missing_agents:
        raise KeyError(
            "Initial checkpoint is missing SKRL agent entries: "
            + ", ".join(missing_agents)
        )

    loaded_modules: set[int] = set()
    for agent_id in possible_agents:
        source = checkpoint[agent_id]
        if not isinstance(source, dict) or "policy" not in source or "value" not in source:
            raise KeyError(
                f"Initial checkpoint entry {agent_id!r} must contain policy and value state dicts."
            )
        for model_name in ("policy", "value"):
            target = models[agent_id][model_name]
            if id(target) in loaded_modules:
                continue
            target.load_state_dict(source[model_name], strict=True)
            loaded_modules.add(id(target))

    return {
        "init_checkpoint": str(resolved),
        "init_checkpoint_source_timestep": int(metadata.get("timesteps", 0)),
        "init_checkpoint_source_training_semantics": metadata.get("training_semantics"),
    }


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
        "--init-checkpoint",
        default=None,
        help=(
            "Compatible SKRL checkpoint whose policy/value weights initialize this run; "
            "optimizer and rollout state are reset."
        ),
    )
    parser.add_argument(
        "--selection-gate",
        choices=(
            "screen",
            "strict",
            "pure_rl_long",
            "safe_progress_long",
            "balanced_progress_long",
            "progress_preserving_long",
            "success_progress_long",
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
    init_checkpoint_value = args.init_checkpoint or algo.get("init_checkpoint")
    if cfg.observation.schema_version == "ego_v8_decentralized_tiered":
        if int(algo.get("bc_updates", algo.get("bc_steps", 0))) != 0:
            raise SystemExit(
                "ego_v8_decentralized_tiered is a pure-RL contract and requires bc_updates=0."
            )
        if init_checkpoint_value is not None and str(init_checkpoint_value).strip():
            raise SystemExit(
                "ego_v8_decentralized_tiered requires random initialization; "
                "init_checkpoint must be null."
            )
    init_checkpoint_path: Path | None = None
    if init_checkpoint_value is not None and str(init_checkpoint_value).strip():
        init_checkpoint_path = Path(str(init_checkpoint_value))
        if not init_checkpoint_path.is_absolute():
            init_checkpoint_path = ROOT / init_checkpoint_path
        if not init_checkpoint_path.is_file():
            raise SystemExit(f"--init-checkpoint does not exist: {init_checkpoint_path}")
        algo["init_checkpoint"] = str(init_checkpoint_path)
        # BC teacher updates would overwrite the supplied policy before PPO.
        # An explicit --bc-updates remains available for intentional re-BC.
        if args.bc_updates is None:
            algo["bc_updates"] = 0
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
    actor_architecture = normalize_actor_architecture(
        algo.get("actor_architecture", "mlp_v1")
    )
    critic_architecture = normalize_critic_architecture(
        algo.get("critic_architecture", "mlp_v1")
    )
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
        actor_architecture=actor_architecture,
        critic_architecture=critic_architecture,
    )
    policy = models[possible_agents[0]]["policy"]
    policy.to(env.device)
    random_initial_policy_parameters = [
        parameter.detach().clone()
        for parameter in policy.parameters()
    ]
    random_initial_terrain_weight = terrain_input_weight_snapshot(policy)
    init_checkpoint_metadata: dict[str, Any] = {}
    if init_checkpoint_path is not None:
        init_checkpoint_metadata = initialize_skrl_mappo_models_from_checkpoint(
            init_checkpoint_path,
            models,
            possible_agents,
            cfg,
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            device=env.device,
        )
    bc_updates = int(algo.get("bc_updates", algo.get("bc_steps", 0)))
    bc_batch_size = int(algo.get("bc_batch_size", 8192))
    bc_learning_rate = float(algo.get("bc_learning_rate", 1.0e-3))
    teacher_mode: str | None = None
    teacher_stop_radius: float | None = None
    teacher_slow_distance: float | None = None
    teacher_max_rho: float | None = None
    teacher_center_step: float | None = None
    if bc_updates > 0:
        teacher_mode = str(algo.get("teacher_mode", "global_centroid"))
        teacher_stop_radius = float(algo.get("teacher_stop_radius", 0.45))
        teacher_slow_distance = float(algo.get("teacher_slow_distance", 0.45))
        teacher_max_rho = (
            float(algo["teacher_max_rho"])
            if algo.get("teacher_max_rho") is not None
            else None
        )
        teacher_center_step = float(algo.get("teacher_center_step", 0.65))
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
            teacher_center_step=teacher_center_step,
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
            bc_terminal_state_fraction=float(
                algo.get("bc_terminal_state_fraction", 0.0)
            ),
            bc_terminal_spawn_radius_min=float(
                algo.get("bc_terminal_spawn_radius_min", 0.35)
            ),
            bc_terminal_spawn_radius_max=float(
                algo.get("bc_terminal_spawn_radius_max", 0.65)
            ),
            bc_terminal_jitter_std=float(
                algo.get("bc_terminal_jitter_std", 0.04)
            ),
            bc_on_policy_rollout_steps=int(
                algo.get("bc_on_policy_rollout_steps", 0)
            ),
            bc_on_policy_tail_fraction=float(
                algo.get("bc_on_policy_tail_fraction", 0.0)
            ),
            bc_on_policy_dmax_multiplier=float(
                algo.get("bc_on_policy_dmax_multiplier", 2.0)
            ),
            bc_on_policy_dispersion_multiplier=float(
                algo.get("bc_on_policy_dispersion_multiplier", 2.0)
            ),
            bc_on_policy_min_teacher_disagreement=float(
                algo.get("bc_on_policy_min_teacher_disagreement", 0.0)
            ),
            bc_teacher_rollout_steps=int(algo.get("bc_teacher_rollout_steps", 0)),
            bc_teacher_tail_fraction=float(
                algo.get("bc_teacher_tail_fraction", 0.0)
            ),
            bc_teacher_dmax_multiplier=float(
                algo.get("bc_teacher_dmax_multiplier", 2.0)
            ),
            bc_teacher_dispersion_multiplier=float(
                algo.get("bc_teacher_dispersion_multiplier", 2.0)
            ),
            bc_on_policy_anchor_base_policy=parse_bool_config(
                algo.get("bc_on_policy_anchor_base_policy"),
                default=False,
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
        actor_credit_assignment = str(
            algo.get("actor_credit_assignment", "none")
        )
        actor_credit_scale = float(algo.get("actor_credit_scale", 0.0))
        if actor_credit_assignment == "none" and actor_credit_scale != 0.0:
            raise ValueError(
                "actor_credit_scale must be zero when actor_credit_assignment is none."
            )
        if actor_credit_assignment != "none" and actor_credit_scale <= 0.0:
            raise ValueError(
                "actor_credit_scale must be positive when actor credit is enabled."
            )
        agent_kwargs["actor_credit_scale"] = actor_credit_scale
        agent_kwargs["actor_credit_trace_lambda"] = float(
            algo.get("actor_credit_trace_lambda", 0.95)
        )
        agent_kwargs["actor_credit_gradient_mode"] = str(
            algo.get("actor_credit_gradient_mode", "additive_advantage")
        )
        collision_constraint_enabled = parse_bool_config(
            algo.get("collision_constraint_enabled"),
            default=False,
        )
        agent_kwargs["collision_constraint_enabled"] = collision_constraint_enabled
        if collision_constraint_enabled:
            if actor_credit_assignment != "none":
                raise ValueError(
                    "Collision constraint screening cannot be combined with Actor credit."
                )
            first_agent = possible_agents[0]
            agent_kwargs["collision_cost_value"] = SKRLValue(
                env.observation_spaces[first_agent],
                env.state_space,
                env.action_spaces[first_agent],
                env.device,
                architecture="mlp_v1",
            )
            agent_kwargs["collision_cost_discount_factor"] = float(
                algo.get("collision_cost_discount_factor", 0.99)
            )
            agent_kwargs["collision_cost_gae_lambda"] = float(
                algo.get("collision_cost_gae_lambda", 0.95)
            )
            agent_kwargs["collision_cost_limit"] = float(
                algo.get("collision_cost_limit", 0.02)
            )
            agent_kwargs["collision_episode_steps"] = int(
                algo.get("collision_episode_steps", cfg.simulation.max_episode_steps)
            )
            agent_kwargs["lagrangian_init"] = float(
                algo.get("lagrangian_init", 0.0)
            )
            agent_kwargs["lagrangian_learning_rate"] = float(
                algo.get("lagrangian_learning_rate", 0.1)
            )
            agent_kwargs["lagrangian_max"] = float(
                algo.get("lagrangian_max", 2.0)
            )
            agent_kwargs["collision_cost_value_learning_rate"] = float(
                algo.get("collision_cost_value_learning_rate", 3.0e-4)
            )
            agent_kwargs["collision_cost_value_loss_scale"] = float(
                algo.get("collision_cost_value_loss_coef", 0.5)
            )
    else:
        actor_credit_assignment = str(algo.get("actor_credit_assignment", "none"))
        if actor_credit_assignment != "none":
            raise ValueError("Actor credit assignment requires update_mode=shared_joint.")
        collision_constraint_enabled = parse_bool_config(
            algo.get("collision_constraint_enabled"),
            default=False,
        )
        if collision_constraint_enabled:
            raise ValueError("Collision constraint requires update_mode=shared_joint.")
    agent = agent_class(**agent_kwargs)
    initial_policy_parameters = [
        parameter.detach().clone()
        for parameter in policy.parameters()
    ]
    initial_terrain_weight = terrain_input_weight_snapshot(policy)
    initial_neighbor_encoder = module_parameter_snapshot(
        getattr(policy, "neighbor_encoder", nn.Identity())
    )
    initial_terrain_encoder = module_parameter_snapshot(
        getattr(policy, "terrain_encoder", nn.Identity())
    )
    initial_reward_critic = module_parameter_snapshot(models[possible_agents[0]]["value"])
    initial_collision_cost_value = (
        module_parameter_snapshot(agent.collision_cost_value)
        if getattr(agent, "collision_constraint_enabled", False)
        else []
    )

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
                collision_cost_value=getattr(agent, "collision_cost_value", None),
                lagrangian_multiplier=getattr(agent, "lagrangian_multiplier", None),
                extra_metadata={
                    "phase": "bc",
                    "update_mode": update_mode,
                    "communication_radius": cfg.observation.communication_radius,
                    "environment_geometry": environment_geometry_metadata(cfg),
                    "initial_state": initial_state_metadata(cfg),
                    "subgoal_filter": subgoal_filter_metadata(cfg),
                    "control_safety": control_safety_metadata(cfg),
                    "bc_updates": bc_updates,
                    "entropy_schedule_timesteps": algo.get(
                        "entropy_schedule_timesteps"
                    ),
                    **init_checkpoint_metadata,
                    **checkpoint_teacher_metadata(
                        bc_updates=bc_updates,
                        teacher_mode=teacher_mode,
                        teacher_stop_radius=teacher_stop_radius,
                        teacher_max_rho=teacher_max_rho,
                        teacher_center_step=teacher_center_step,
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
                terrain_input_weight_delta_l2(policy, random_initial_terrain_weight)
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
            "environment_geometry": environment_geometry_metadata(cfg),
            "initial_state": initial_state_metadata(cfg),
            "actor_architecture": actor_architecture,
            "critic_architecture": critic_architecture,
            "kinematic_model": cfg.low_level_control.kinematic_model,
            "trajectory_geometry_method": cfg.trajectory_generator.geometry_method,
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
        _snapshot_numeric_metrics(telemetry_state, "actor_credit", "actor_credit_window")
        _snapshot_numeric_metrics(
            telemetry_state,
            "centroid_flatness",
            "centroid_flatness_window",
        )
        _snapshot_numeric_metrics(telemetry_state, "action_filter", "action_filter_window")
        _snapshot_numeric_metrics(telemetry_state, "control_safety", "control_safety_window")
        _snapshot_numeric_metrics(
            telemetry_state,
            "formation_center_correction",
            "formation_center_correction_window",
        )
        _snapshot_numeric_metrics(
            telemetry_state,
            "terminal_slot_capture",
            "terminal_slot_capture_window",
        )
        _snapshot_numeric_metrics(
            telemetry_state,
            "flat_geometry_capture",
            "flat_geometry_capture_window",
        )
        _snapshot_numeric_metrics(telemetry_state, "kinematics", "kinematics_window")
        _snapshot_numeric_metrics(
            telemetry_state,
            "communication",
            "communication_window",
        )
        _snapshot_numeric_metrics(
            telemetry_state,
            "mapf_conflicts",
            "mapf_conflicts_window",
        )
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
    if actor_credit_assignment != "none":
        install_actor_credit_rewards(env, assignment=actor_credit_assignment)
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
                collision_cost_value=getattr(agent, "collision_cost_value", None),
                lagrangian_multiplier=getattr(agent, "lagrangian_multiplier", None),
                extra_metadata={
                    "phase": "ppo",
                    "update_mode": update_mode,
                    "communication_radius": cfg.observation.communication_radius,
                    "environment_geometry": environment_geometry_metadata(cfg),
                    "initial_state": initial_state_metadata(cfg),
                    "subgoal_filter": subgoal_filter_metadata(cfg),
                    "control_safety": control_safety_metadata(cfg),
                    "bc_updates": bc_updates,
                    "actor_credit_assignment": actor_credit_assignment,
                    "actor_credit_scale": float(
                        getattr(agent, "actor_credit_scale", 0.0)
                    ),
                    "actor_credit_trace_lambda": float(
                        getattr(agent, "actor_credit_trace_lambda", 0.0)
                    ),
                    "bc_batch_size": bc_batch_size,
                    "bc_learning_rate": bc_learning_rate,
                    "entropy_schedule_timesteps": algo.get(
                        "entropy_schedule_timesteps"
                    ),
                    **init_checkpoint_metadata,
                    **checkpoint_teacher_metadata(
                        bc_updates=bc_updates,
                        teacher_mode=teacher_mode,
                        teacher_stop_radius=teacher_stop_radius,
                        teacher_max_rho=teacher_max_rho,
                        teacher_center_step=teacher_center_step,
                    ),
                },
            ),
            candidate_path,
        )
        candidate_paths.append(candidate_path)
        return candidate_path

    # Keep the exact initialized policy as a selectable baseline.  This makes
    # warm-start screening falsifiable: PPO may improve it, but cannot hide a
    # regression by comparing only post-update checkpoints.
    if init_checkpoint_path is not None:
        save_candidate(0)

    original_post_interaction = agent.post_interaction
    use_initial_state_curriculum = bool(cfg.initial_state.curriculum_enabled)
    if use_initial_state_curriculum:
        cfg.initial_state.progress_timestep_override = 0
        env.core.reset()

    def post_interaction_with_housekeeping(*, timestep: int, timesteps: int) -> None:
        original_post_interaction(timestep=timestep, timesteps=timesteps)
        completed_timestep = timestep + 1
        if use_initial_state_curriculum:
            cfg.initial_state.progress_timestep_override = completed_timestep
        if checkpoint_interval > 0 and completed_timestep % checkpoint_interval == 0:
            save_candidate(completed_timestep)

    if checkpoint_interval > 0 or use_initial_state_curriculum:
        agent.post_interaction = post_interaction_with_housekeeping

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
            terrain_input_weight_delta_l2(policy, initial_terrain_weight)
        ),
        "neighbor_encoder_parameter_delta_l2": module_parameter_delta_l2(
            getattr(policy, "neighbor_encoder", nn.Identity()),
            initial_neighbor_encoder,
        ),
        "terrain_encoder_parameter_delta_l2": module_parameter_delta_l2(
            getattr(policy, "terrain_encoder", nn.Identity()),
            initial_terrain_encoder,
        ),
        "policy_parameters_finite": all(
            bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()
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
        "reward_critic_parameter_delta_l2": module_parameter_delta_l2(
            models[possible_agents[0]]["value"],
            initial_reward_critic,
        ),
        "collision_constraint_enabled": bool(
            getattr(agent, "collision_constraint_enabled", False)
        ),
        "collision_cost_critic_update_count": int(
            getattr(agent, "collision_cost_critic_update_count", 0)
        ),
        "collision_cost_value_parameter_delta_l2": (
            module_parameter_delta_l2(
                agent.collision_cost_value,
                initial_collision_cost_value,
            )
            if getattr(agent, "collision_constraint_enabled", False)
            else 0.0
        ),
        "collision_cost_value_parameters_finite": (
            all(
                bool(torch.isfinite(parameter).all())
                for parameter in agent.collision_cost_value.parameters()
            )
            if getattr(agent, "collision_constraint_enabled", False)
            else True
        ),
        "last_collision_cost_value_loss": float(
            getattr(agent, "last_collision_cost_value_loss", 0.0)
        ),
        "last_collision_episode_equivalent_rate": float(
            getattr(agent, "last_collision_episode_equivalent_rate", 0.0)
        ),
        "lagrangian_multiplier": float(
            getattr(agent, "lagrangian_multiplier", 0.0)
        ),
        "last_lagrangian_multiplier_applied": float(
            getattr(agent, "last_lagrangian_multiplier_applied", 0.0)
        ),
        "collision_constraint_history": list(
            getattr(agent, "collision_constraint_history", [])
        ),
        "last_actor_sample_count": int(getattr(agent, "last_actor_sample_count", 0)),
        "last_critic_sample_count": int(getattr(agent, "last_critic_sample_count", 0)),
        "actor_credit_assignment": actor_credit_assignment,
        "actor_credit_scale": float(getattr(agent, "actor_credit_scale", 0.0)),
        "actor_credit_trace_lambda": float(
            getattr(agent, "actor_credit_trace_lambda", 0.0)
        ),
        "actor_credit_gradient_mode": str(
            getattr(agent, "actor_credit_gradient_mode", "additive_advantage")
        ),
        "last_actor_credit_abs_mean": float(
            getattr(agent, "last_actor_credit_abs_mean", 0.0)
        ),
        "last_actor_credit_std": float(
            getattr(agent, "last_actor_credit_std", 0.0)
        ),
        "last_actor_gradient_conflict_fraction": float(
            getattr(agent, "last_actor_gradient_conflict_fraction", 0.0)
        ),
        "last_actor_gradient_cosine_mean": float(
            getattr(agent, "last_actor_gradient_cosine_mean", 0.0)
        ),
        "last_actor_gradient_projected_dot_min": float(
            getattr(agent, "last_actor_gradient_projected_dot_min", 0.0)
        ),
        "last_actor_gradient_combined_cosine_min": float(
            getattr(agent, "last_actor_gradient_combined_cosine_min", 1.0)
        ),
        "last_actor_gradient_norm_cap_scale_mean": float(
            getattr(agent, "last_actor_gradient_norm_cap_scale_mean", 1.0)
        ),
    }
    _snapshot_numeric_metrics(telemetry_state, "action", "action_window")
    _snapshot_numeric_metrics(telemetry_state, "reward", "reward_window")
    _snapshot_numeric_metrics(telemetry_state, "path_terrain", "path_terrain_window")
    _snapshot_numeric_metrics(telemetry_state, "actor_credit", "actor_credit_window")
    _snapshot_numeric_metrics(
        telemetry_state,
        "centroid_flatness",
        "centroid_flatness_window",
    )
    _snapshot_numeric_metrics(telemetry_state, "action_filter", "action_filter_window")
    _snapshot_numeric_metrics(telemetry_state, "control_safety", "control_safety_window")
    _snapshot_numeric_metrics(
        telemetry_state,
        "formation_center_correction",
        "formation_center_correction_window",
    )
    _snapshot_numeric_metrics(
        telemetry_state,
        "terminal_slot_capture",
        "terminal_slot_capture_window",
    )
    _snapshot_numeric_metrics(
        telemetry_state,
        "flat_geometry_capture",
        "flat_geometry_capture_window",
    )
    _snapshot_numeric_metrics(telemetry_state, "kinematics", "kinematics_window")
    _snapshot_numeric_metrics(
        telemetry_state,
        "communication",
        "communication_window",
    )
    _snapshot_numeric_metrics(
        telemetry_state,
        "mapf_conflicts",
        "mapf_conflicts_window",
    )

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
            else PROGRESS_PRESERVING_LONG_THRESHOLDS
            if args.selection_gate == "progress_preserving_long"
            else SUCCESS_PROGRESS_LONG_THRESHOLDS
            if args.selection_gate == "success_progress_long"
            else STRICT_THRESHOLDS
        )
        ranker = (
            pure_rl_long_checkpoint_rank
            if args.selection_gate == "pure_rl_long"
            else safe_progress_long_checkpoint_rank
            if args.selection_gate == "safe_progress_long"
            else balanced_progress_long_checkpoint_rank
            if args.selection_gate == "balanced_progress_long"
            else progress_preserving_long_checkpoint_rank
            if args.selection_gate == "progress_preserving_long"
            else success_progress_long_checkpoint_rank
            if args.selection_gate == "success_progress_long"
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
        "actor_architecture": actor_architecture,
        "critic_architecture": critic_architecture,
        "update_mode": update_mode,
        "communication_radius": cfg.observation.communication_radius,
        "environment_geometry": environment_geometry_metadata(cfg),
        "initial_state": initial_state_metadata(cfg),
        "subgoal_filter": subgoal_filter_metadata(cfg),
        "control_safety": control_safety_metadata(cfg),
        "kinematic_model": cfg.low_level_control.kinematic_model,
        "trajectory_geometry_method": cfg.trajectory_generator.geometry_method,
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
