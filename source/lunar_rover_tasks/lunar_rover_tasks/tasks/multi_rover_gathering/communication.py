"""Neighbor state sharing for decentralized actor observations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import ObservationCfg
from lunar_rover_tasks.utils.math_utils import wrap_to_pi


def compute_visibility_mask(
    positions: torch.Tensor,
    communication_radius: float,
) -> torch.Tensor:
    """Return non-self neighbor visibility.

    A non-positive radius is treated as the temporary "unlimited communication"
    setting used by engineering probes, not as a zero-meter visibility range.
    """

    delta = positions[:, :, None, :2] - positions[:, None, :, :2]
    dist = torch.linalg.norm(delta, dim=-1)
    n_agents = positions.shape[1]
    self_mask = torch.eye(n_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
    if float(communication_radius) <= 0.0:
        return (~self_mask).expand(positions.shape[0], -1, -1)
    return (dist <= communication_radius) & ~self_mask


def build_neighbor_features(
    positions: torch.Tensor,
    velocities_xy: torch.Tensor,
    yaws: torch.Tensor,
    communication_radius: float,
    cfg: ObservationCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_envs, n_agents, _ = positions.shape
    visibility = compute_visibility_mask(positions, communication_radius)
    pairwise_delta = positions[:, None, :, :2] - positions[:, :, None, :2]
    pairwise_vel = velocities_xy[:, None, :, :] - velocities_xy[:, :, None, :]
    yaw_delta = wrap_to_pi(yaws[:, None, :] - yaws[:, :, None])
    pairwise_dist = torch.linalg.norm(pairwise_delta, dim=-1)
    slot_count = min(cfg.max_neighbors, n_agents)
    masked_dist = pairwise_dist.masked_fill(~visibility, float("inf"))
    selected_dist, selected = torch.topk(
        masked_dist,
        k=slot_count,
        dim=-1,
        largest=False,
        sorted=True,
    )
    valid = torch.isfinite(selected_dist)
    gather_xy = selected[..., None].expand(-1, -1, -1, 2)
    selected_delta = torch.gather(pairwise_delta, dim=2, index=gather_xy)
    selected_vel = torch.gather(pairwise_vel, dim=2, index=gather_xy)
    selected_yaw = torch.gather(yaw_delta, dim=2, index=selected)
    selected_features = torch.cat(
        (
            selected_delta,
            selected_vel,
            torch.cos(selected_yaw).unsqueeze(-1),
            torch.sin(selected_yaw).unsqueeze(-1),
            valid.to(dtype=positions.dtype).unsqueeze(-1),
        ),
        dim=-1,
    )
    selected_features = torch.where(
        valid.unsqueeze(-1),
        selected_features,
        torch.zeros_like(selected_features),
    )

    features = torch.zeros(
        num_envs,
        n_agents,
        cfg.max_neighbors,
        cfg.neighbor_dim,
        dtype=positions.dtype,
        device=positions.device,
    )
    masks = torch.zeros(
        num_envs,
        n_agents,
        cfg.max_neighbors,
        dtype=positions.dtype,
        device=positions.device,
    )
    features[..., :slot_count, :] = selected_features
    masks[..., :slot_count] = valid.to(dtype=positions.dtype)
    return features.flatten(start_dim=2), masks


@dataclass(slots=True)
class CommunicationSnapshot:
    """Actor-visible messages selected exclusively from a communication cache."""

    features: torch.Tensor
    masks: torch.Tensor
    ages: torch.Tensor
    full_messages: torch.Tensor
    sender_indices: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def _world_to_body(vector: torch.Tensor, receiver_yaw: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(receiver_yaw)
    sin_yaw = torch.sin(receiver_yaw)
    return torch.stack(
        (
            cos_yaw * vector[..., 0] + sin_yaw * vector[..., 1],
            -sin_yaw * vector[..., 0] + cos_yaw * vector[..., 1],
        ),
        dim=-1,
    )


class TieredCommunicationCache:
    """Stateful 12 m full / distance-dependent sparse communication cache.

    Cache mutation is explicit through :meth:`reset` and :meth:`advance`.
    Observation reads call :meth:`snapshot`, which is deliberately pure so
    repeated reads cannot refresh a remote message or change its age.
    """

    base_message_dim = 12
    intent_message_dim = 16
    differential_intent_message_dim = 17

    def __init__(
        self,
        *,
        num_envs: int,
        n_agents: int,
        max_neighbors: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        full_radius_m: float = 12.0,
        map_max_distance_m: float = 25.0 * math.sqrt(2.0),
        min_sparse_period_s: float = 1.0,
        max_sparse_period_s: float = 4.0,
        include_plan_intent: bool = False,
        include_plan_yaw: bool = False,
    ) -> None:
        if full_radius_m <= 0.0:
            raise ValueError("full_radius_m must be positive.")
        if map_max_distance_m <= full_radius_m:
            raise ValueError("map_max_distance_m must exceed full_radius_m.")
        if min_sparse_period_s <= 0.0 or max_sparse_period_s < min_sparse_period_s:
            raise ValueError("sparse communication periods are invalid.")
        self.num_envs = int(num_envs)
        self.n_agents = int(n_agents)
        self.max_neighbors = int(max_neighbors)
        self.device = torch.device(device)
        self.dtype = dtype
        self.full_radius_m = float(full_radius_m)
        self.map_max_distance_m = float(map_max_distance_m)
        self.min_sparse_period_s = float(min_sparse_period_s)
        self.max_sparse_period_s = float(max_sparse_period_s)
        self.include_plan_intent = bool(include_plan_intent)
        self.include_plan_yaw = bool(include_plan_yaw)
        if self.include_plan_yaw and not self.include_plan_intent:
            raise ValueError("include_plan_yaw requires include_plan_intent.")
        self.message_dim = (
            self.differential_intent_message_dim
            if self.include_plan_yaw
            else self.intent_message_dim
            if self.include_plan_intent
            else self.base_message_dim
        )
        pair_shape = (self.num_envs, self.n_agents, self.n_agents)
        self.features = torch.zeros(
            *pair_shape,
            self.message_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.valid = torch.zeros(pair_shape, device=self.device, dtype=torch.bool)
        self.full = torch.zeros_like(self.valid)
        self.age = torch.zeros(pair_shape, device=self.device, dtype=self.dtype)
        self.update_period = torch.ones(pair_shape, device=self.device, dtype=self.dtype)
        self.last_distance = torch.zeros(pair_shape, device=self.device, dtype=self.dtype)
        self._self_mask = torch.eye(
            self.n_agents,
            device=self.device,
            dtype=torch.bool,
        ).unsqueeze(0)

    def _period(self, distance: torch.Tensor) -> torch.Tensor:
        alpha = ((distance - self.full_radius_m) / (
            self.map_max_distance_m - self.full_radius_m
        )).clamp(0.0, 1.0)
        return self.min_sparse_period_s + (
            self.max_sparse_period_s - self.min_sparse_period_s
        ) * alpha

    def _pair_state(
        self,
        positions: torch.Tensor,
        velocities_xy: torch.Tensor,
        yaws: torch.Tensor,
        terrain_summary: torch.Tensor,
        committed_world_subgoal: torch.Tensor | None = None,
        committed_reference_speed: torch.Tensor | None = None,
        coordination_token: torch.Tensor | None = None,
        committed_planned_yaw_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        receiver_yaw = yaws[:, :, None]
        delta_world = positions[:, None, :, :2] - positions[:, :, None, :2]
        relative_velocity_world = (
            velocities_xy[:, None, :, :] - velocities_xy[:, :, None, :]
        )
        delta_body = _world_to_body(delta_world, receiver_yaw)
        velocity_body = _world_to_body(relative_velocity_world, receiver_yaw)
        yaw_delta = wrap_to_pi(yaws[:, None, :] - receiver_yaw)
        sender_terrain = terrain_summary[:, None, :, :].expand(
            -1,
            self.n_agents,
            -1,
            -1,
        )
        quality = torch.ones(
            *delta_body.shape[:-1],
            1,
            device=self.device,
            dtype=self.dtype,
        )
        parts = [
            delta_body,
            velocity_body,
            torch.cos(yaw_delta).unsqueeze(-1),
            torch.sin(yaw_delta).unsqueeze(-1),
            sender_terrain,
            quality,
        ]
        if self.include_plan_intent:
            if committed_world_subgoal is None:
                committed_world_subgoal = positions[..., :2]
            if committed_reference_speed is None:
                committed_reference_speed = torch.zeros_like(yaws)
            if coordination_token is None:
                coordination_token = torch.zeros_like(yaws)
            if committed_planned_yaw_delta is None:
                committed_planned_yaw_delta = torch.zeros_like(yaws)
            sender_plan_world = committed_world_subgoal[:, None, :, :2]
            receiver_position = positions[:, :, None, :2]
            sender_plan_body = _world_to_body(
                sender_plan_world - receiver_position,
                receiver_yaw,
            )
            sender_speed = committed_reference_speed[:, None, :, None].expand(
                -1,
                self.n_agents,
                -1,
                -1,
            )
            sender_token = coordination_token[:, None, :, None].expand_as(sender_speed)
            parts.extend((sender_plan_body, sender_speed))
            if self.include_plan_yaw:
                sender_plan_yaw = committed_planned_yaw_delta[
                    :, None, :, None
                ].expand_as(sender_speed)
                parts.append(sender_plan_yaw)
            parts.append(sender_token)
        features = torch.cat(
            (
                *parts,
            ),
            dim=-1,
        )
        return features, torch.linalg.norm(delta_world, dim=-1)

    def _clear_restricted_payload(
        self,
        features: torch.Tensor,
        restricted: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if restricted is None:
            features[..., 2:4] = 0.0
            features[..., 6:11] = 0.0
            if self.include_plan_intent:
                features[..., 12 : self.message_dim] = 0.0
            return features
        features[..., 2:4] = torch.where(
            restricted.unsqueeze(-1),
            torch.zeros_like(features[..., 2:4]),
            features[..., 2:4],
        )
        features[..., 6:11] = torch.where(
            restricted.unsqueeze(-1),
            torch.zeros_like(features[..., 6:11]),
            features[..., 6:11],
        )
        if self.include_plan_intent:
            features[..., 12 : self.message_dim] = torch.where(
                restricted.unsqueeze(-1),
                torch.zeros_like(features[..., 12 : self.message_dim]),
                features[..., 12 : self.message_dim],
            )
        return features

    def reset(
        self,
        env_ids: torch.Tensor,
        positions: torch.Tensor,
        velocities_xy: torch.Tensor,
        yaws: torch.Tensor,
        terrain_summary: torch.Tensor,
        committed_world_subgoal: torch.Tensor | None = None,
        committed_reference_speed: torch.Tensor | None = None,
        coordination_token: torch.Tensor | None = None,
        committed_planned_yaw_delta: torch.Tensor | None = None,
    ) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        current, distance = self._pair_state(
            positions[env_ids],
            velocities_xy[env_ids],
            yaws[env_ids],
            terrain_summary[env_ids],
            committed_world_subgoal=(
                committed_world_subgoal[env_ids]
                if committed_world_subgoal is not None
                else None
            ),
            committed_reference_speed=(
                committed_reference_speed[env_ids]
                if committed_reference_speed is not None
                else None
            ),
            committed_planned_yaw_delta=(
                committed_planned_yaw_delta[env_ids]
                if committed_planned_yaw_delta is not None
                else None
            ),
            coordination_token=(
                coordination_token[env_ids]
                if coordination_token is not None
                else None
            ),
        )
        nonself = (~self._self_mask).expand(env_ids.numel(), -1, -1)
        full = (distance <= self.full_radius_m) & nonself
        sparse = nonself & ~full
        current = self._clear_restricted_payload(current, sparse)
        current[..., 11] = torch.where(
            full,
            torch.ones_like(distance),
            torch.full_like(distance, 0.5),
        )
        current = torch.where(nonself.unsqueeze(-1), current, torch.zeros_like(current))
        self.features[env_ids] = current
        self.valid[env_ids] = nonself
        self.full[env_ids] = full
        self.age[env_ids] = 0.0
        self.update_period[env_ids] = torch.where(
            full,
            torch.zeros_like(distance),
            self._period(distance),
        )
        self.last_distance[env_ids] = distance

    def advance(
        self,
        *,
        dt: float,
        positions: torch.Tensor,
        velocities_xy: torch.Tensor,
        yaws: torch.Tensor,
        terrain_summary: torch.Tensor,
        committed_world_subgoal: torch.Tensor | None = None,
        committed_reference_speed: torch.Tensor | None = None,
        committed_planned_yaw_delta: torch.Tensor | None = None,
        coordination_token: torch.Tensor | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("communication dt must be positive.")
        current, distance = self._pair_state(
            positions,
            velocities_xy,
            yaws,
            terrain_summary,
            committed_world_subgoal=committed_world_subgoal,
            committed_reference_speed=committed_reference_speed,
            coordination_token=coordination_token,
            committed_planned_yaw_delta=committed_planned_yaw_delta,
        )
        nonself = (~self._self_mask).expand(self.num_envs, -1, -1)
        now_full = (distance <= self.full_radius_m) & nonself
        leaving = self.full & ~now_full & nonself
        staying_sparse = ~self.full & ~now_full & nonself
        self.age = torch.where(nonself, self.age + float(dt), self.age)

        # Full-range messages are refreshed every real planning step.
        self.features = torch.where(now_full.unsqueeze(-1), current, self.features)
        self.features[..., 11] = torch.where(
            now_full,
            torch.ones_like(self.age),
            self.features[..., 11],
        )
        self.age = torch.where(now_full, torch.zeros_like(self.age), self.age)
        self.update_period = torch.where(
            now_full,
            torch.zeros_like(self.update_period),
            self.update_period,
        )

        # Leaving the full range immediately removes forbidden payload fields
        # without transmitting a fresh far-range pose.
        restricted = leaving | staying_sparse
        self.features = self._clear_restricted_payload(self.features, restricted)
        leaving_period = self._period(distance)
        self.update_period = torch.where(leaving, leaving_period, self.update_period)

        due = staying_sparse & (self.age >= self.update_period)
        sparse_current = current.clone()
        sparse_current = self._clear_restricted_payload(sparse_current)
        sparse_current[..., 11] = 0.5
        self.features = torch.where(due.unsqueeze(-1), sparse_current, self.features)
        self.age = torch.where(due, torch.zeros_like(self.age), self.age)
        self.update_period = torch.where(due, self._period(distance), self.update_period)

        sparse = ~now_full & nonself
        decayed_quality = 0.5 * torch.exp(
            -self.age / self.update_period.clamp_min(torch.finfo(self.dtype).eps)
        )
        self.features[..., 11] = torch.where(
            sparse,
            decayed_quality,
            self.features[..., 11],
        )
        self.features = torch.where(
            nonself.unsqueeze(-1),
            self.features,
            torch.zeros_like(self.features),
        )
        self.valid = nonself.clone()
        self.full = now_full
        self.last_distance = torch.where(nonself, distance, self.last_distance)

    def snapshot(self) -> CommunicationSnapshot:
        slot_count = min(self.max_neighbors, max(self.n_agents - 1, 0))
        output = torch.zeros(
            self.num_envs,
            self.n_agents,
            self.max_neighbors,
            self.message_dim,
            device=self.device,
            dtype=self.dtype,
        )
        masks = torch.zeros(
            self.num_envs,
            self.n_agents,
            self.max_neighbors,
            device=self.device,
            dtype=self.dtype,
        )
        ages = torch.zeros_like(masks)
        full_messages = torch.zeros_like(masks, dtype=torch.bool)
        sender_indices = torch.full(
            masks.shape,
            -1,
            device=self.device,
            dtype=torch.long,
        )
        if slot_count > 0:
            distance = torch.linalg.norm(self.features[..., :2], dim=-1)
            sender_tiebreak = torch.arange(
                self.n_agents,
                device=self.device,
                dtype=self.dtype,
            ).view(1, 1, -1) * 1.0e-6
            ranking_distance = (distance + sender_tiebreak).masked_fill(
                ~self.valid,
                float("inf"),
            )
            selected_distance, selected = torch.topk(
                ranking_distance,
                k=slot_count,
                dim=-1,
                largest=False,
                sorted=True,
            )
            selected_valid = torch.isfinite(selected_distance)
            gather_features = selected[..., None].expand(-1, -1, -1, self.message_dim)
            selected_features = torch.gather(self.features, dim=2, index=gather_features)
            selected_age = torch.gather(self.age, dim=2, index=selected)
            selected_full = torch.gather(self.full, dim=2, index=selected)
            selected_features = torch.where(
                selected_valid.unsqueeze(-1),
                selected_features,
                torch.zeros_like(selected_features),
            )
            output[..., :slot_count, :] = selected_features
            masks[..., :slot_count] = selected_valid.to(self.dtype)
            ages[..., :slot_count] = torch.where(
                selected_valid,
                selected_age,
                torch.zeros_like(selected_age),
            )
            full_messages[..., :slot_count] = selected_full & selected_valid
            sender_indices[..., :slot_count] = torch.where(
                selected_valid,
                selected,
                torch.full_like(selected, -1),
            )

        nonself = self.valid
        pair_count = nonself.sum(dim=(1, 2)).clamp_min(1)
        far = nonself & ~self.full
        diagnostics = {
            "full_message_ratio": self.full.sum(dim=(1, 2)).to(self.dtype) / pair_count,
            "sparse_message_ratio": far.sum(dim=(1, 2)).to(self.dtype) / pair_count,
            "mean_message_age": self.age.masked_fill(~nonself, 0.0).sum(dim=(1, 2))
            / pair_count,
            "mean_update_period": self.update_period.masked_fill(
                ~nonself,
                0.0,
            ).sum(dim=(1, 2))
            / pair_count,
            "far_pair_ratio": far.sum(dim=(1, 2)).to(self.dtype) / pair_count,
        }
        return CommunicationSnapshot(
            features=output.flatten(start_dim=2),
            masks=masks,
            ages=ages,
            full_messages=full_messages,
            sender_indices=sender_indices,
            diagnostics=diagnostics,
        )


def build_cached_aggregation_features(
    snapshot: CommunicationSnapshot,
    *,
    map_max_distance_m: float,
    max_message_age_s: float = 4.0,
) -> torch.Tensor:
    neighbor_slots = snapshot.masks.shape[-1]
    if neighbor_slots <= 0 or snapshot.features.shape[-1] % neighbor_slots != 0:
        raise ValueError("Communication snapshot has incompatible feature/mask shapes.")
    message_dim = snapshot.features.shape[-1] // neighbor_slots
    messages = snapshot.features.reshape(
        *snapshot.features.shape[:2],
        neighbor_slots,
        message_dim,
    )
    mask = snapshot.masks
    count = mask.sum(dim=-1, keepdim=True)
    safe_count = count.clamp_min(1.0)
    distance = torch.linalg.norm(messages[..., :2], dim=-1)
    mean_distance = (distance * mask).sum(dim=-1, keepdim=True) / safe_count
    max_distance = distance.masked_fill(mask <= 0.0, 0.0).amax(dim=-1, keepdim=True)
    mean_quality = (messages[..., 11] * mask).sum(dim=-1, keepdim=True) / safe_count
    mean_age = (snapshot.ages * mask).sum(dim=-1, keepdim=True) / safe_count
    features = torch.cat(
        (
            count / float(max(snapshot.masks.shape[-1], 1)),
            mean_distance / float(map_max_distance_m),
            max_distance / float(map_max_distance_m),
            mean_quality,
            mean_age / float(max_message_age_s),
        ),
        dim=-1,
    )
    return torch.where(count > 0.0, features, torch.zeros_like(features))
