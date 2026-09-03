"""Online Active-DSTC belief runtime for the RL-mainline Actor.

The runtime never chooses or modifies a rover action. It converts local terrain
measurements and one-hop delta/event communication into per-rover site beliefs
and an immutable all-rover certificate. The shared Actor remains responsible
for every physical action throughout the episode.
"""

from __future__ import annotations

from typing import Iterable

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import (
    SPATIOTEMPORAL_ENDPOINTS,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (
    ActiveDSTCCfg,
    GatherPointCfg,
    TerrainCfg,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.site_commitment import (
    ProposalBatchMessage,
    SiteBeliefCache,
    SiteCertificate,
    build_site_witnesses,
    combined_proposal_set_digest,
    extract_local_site_proposals,
    proposal_batch_digest,
    select_site_certificate,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    TerrainRuntime,
)


Clock = dict[tuple[int, str], tuple[int, int, int]]
ACTIVE_SCAN_OFFSETS = ((0.0, 0.0), *SPATIOTEMPORAL_ENDPOINTS)


def _cache_signature(cache: SiteBeliefCache) -> tuple:
    return tuple(sorted(cache.event_clock.items()))


class ActiveDSTCRuntime:
    """Finite decentralized site belief replicated across vector environments."""

    def __init__(
        self,
        *,
        num_envs: int,
        n_agents: int,
        device: torch.device,
        cfg: ActiveDSTCCfg,
        gather_cfg: GatherPointCfg,
    ) -> None:
        self.num_envs = int(num_envs)
        self.n_agents = int(n_agents)
        self.device = device
        self.cfg = cfg
        self.gather_cfg = gather_cfg
        self.initial_positions = torch.zeros(
            num_envs, n_agents, 3, device=device
        )
        self.initial_yaws = torch.zeros(num_envs, n_agents, device=device)
        self.target_points = torch.zeros(num_envs, n_agents, 3, device=device)
        self.target_valid = torch.zeros(
            num_envs, n_agents, dtype=torch.bool, device=device
        )
        self.committed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.committed_centers = torch.zeros(num_envs, 3, device=device)
        self.committed_steps = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )
        self.known_source_fraction = torch.zeros(
            num_envs, n_agents, device=device
        )
        self.potential = torch.zeros(num_envs, device=device)
        self.scan_versions = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.delta_records = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.full_records = torch.zeros_like(self.delta_records)
        self._caches = [
            [self._new_cache(agent) for agent in range(n_agents)]
            for _ in range(num_envs)
        ]
        self._link_clocks: list[list[list[Clock]]] = [
            [[{} for _ in range(n_agents)] for _ in range(n_agents)]
            for _ in range(num_envs)
        ]

    def _new_cache(self, agent: int) -> SiteBeliefCache:
        return SiteBeliefCache(
            agent_id=agent,
            n_agents=self.n_agents,
            max_entries_per_source=int(self.cfg.max_entries_per_source),
            ttl_steps=int(self.cfg.belief_ttl_steps),
        )

    def reset(
        self,
        env_ids: torch.Tensor,
        positions: torch.Tensor,
        yaws: torch.Tensor,
        terrain_cfg: TerrainCfg,
        terrain_runtime: TerrainRuntime,
    ) -> None:
        ids = env_ids.to(device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        self.initial_positions[ids] = positions[ids]
        self.initial_yaws[ids] = yaws[ids]
        self.target_points[ids] = 0.0
        self.target_valid[ids] = False
        self.committed[ids] = False
        self.committed_centers[ids] = 0.0
        self.committed_steps[ids] = -1
        self.known_source_fraction[ids] = 0.0
        self.potential[ids] = 0.0
        self.scan_versions[ids] = 0
        self.delta_records[ids] = 0
        self.full_records[ids] = 0
        for environment in ids.tolist():
            self._caches[environment] = [
                self._new_cache(agent) for agent in range(self.n_agents)
            ]
            self._link_clocks[environment] = [
                [{} for _ in range(self.n_agents)]
                for _ in range(self.n_agents)
            ]
        self.update(
            positions,
            yaws,
            terrain_cfg,
            terrain_runtime,
            step_counts=torch.zeros_like(self.scan_versions),
            env_ids=ids,
            force=True,
        )

    def _select_certificate(
        self,
        cache: SiteBeliefCache,
        rover_positions: list[tuple[float, float]],
    ) -> SiteCertificate | None:
        groups = cache.proposals_by_source()
        batches = [
            ProposalBatchMessage(0, source, proposal_batch_digest(group))
            for source, group in enumerate(groups)
        ]
        witnesses = build_site_witnesses(
            groups,
            pose_uncertainty_m=float(self.cfg.pose_uncertainty_m),
            min_support_rovers=1,
        )
        return select_site_certificate(
            witnesses,
            rover_positions,
            epoch=0,
            proposal_set_digest=combined_proposal_set_digest(batches),
            distance_weight=float(self.cfg.distance_score_weight),
            support_weight=float(self.cfg.support_score_weight),
        )

    def _exchange(
        self,
        environment: int,
        adjacency: torch.Tensor,
    ) -> tuple[int, int]:
        caches = self._caches[environment]
        delta_records = 0
        full_records = 0
        for _ in range(int(self.cfg.forwarding_rounds)):
            messages = [
                [
                    caches[sender].delta_message(
                        (
                            self._link_clocks[environment][sender][receiver]
                            if _cache_signature(caches[sender])
                            == _cache_signature(caches[receiver])
                            else {}
                        )
                    )
                    for receiver in range(self.n_agents)
                ]
                for sender in range(self.n_agents)
            ]
            full_messages = [cache.message() for cache in caches]
            for receiver in range(self.n_agents):
                for sender in range(self.n_agents):
                    if receiver == sender or not bool(adjacency[receiver, sender]):
                        continue
                    message = messages[sender][receiver]
                    delta_records += len(message.records) + len(message.tombstones)
                    full_records += len(full_messages[sender].records)
                    caches[receiver].merge_delta(message)
            for receiver in range(self.n_agents):
                receiver_clock = dict(caches[receiver].event_clock)
                for sender in range(self.n_agents):
                    if receiver != sender and bool(adjacency[receiver, sender]):
                        self._link_clocks[environment][sender][receiver] = (
                            receiver_clock
                        )
        return delta_records, full_records

    def update(
        self,
        positions: torch.Tensor,
        yaws: torch.Tensor,
        terrain_cfg: TerrainCfg,
        terrain_runtime: TerrainRuntime,
        *,
        step_counts: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        force: bool = False,
    ) -> torch.Tensor:
        if env_ids is None:
            mask = ~self.committed
            if not force:
                mask &= step_counts.remainder(
                    max(int(self.cfg.scan_interval_steps), 1)
                ) == 0
            ids = torch.nonzero(mask, as_tuple=False).flatten()
        else:
            ids = env_ids.to(device=self.device, dtype=torch.long)
            ids = ids[~self.committed[ids]]
            if not force:
                ids = ids[
                    step_counts[ids].remainder(
                        max(int(self.cfg.scan_interval_steps), 1)
                    )
                    == 0
                ]
        if ids.numel() == 0:
            return ids

        proposals = extract_local_site_proposals(
            positions[ids],
            yaws[ids],
            terrain_cfg,
            terrain_runtime.subset(ids),
            epoch=0,
            max_candidates_per_rover=int(self.cfg.max_candidates_per_rover),
            verification_radius_m=float(self.cfg.verification_radius_m),
            required_radius_m=float(self.cfg.required_flat_radius_m),
            flatness_rings=int(self.cfg.flatness_rings),
            flatness_samples_per_ring=int(self.cfg.flatness_samples_per_ring),
            required_flatness_rings=int(self.gather_cfg.flatness_rings),
            required_flatness_samples_per_ring=int(
                self.gather_cfg.flatness_samples_per_ring
            ),
            max_height_range_m=float(self.gather_cfg.max_height_range),
            max_slope=float(self.gather_cfg.max_slope),
            nms_distance_m=float(self.cfg.nms_distance_m),
            id_quantization_m=float(self.cfg.id_quantization_m),
            source_frame_positions=self.initial_positions[ids],
            source_frame_yaws=self.initial_yaws[ids],
            candidate_offsets_body=ACTIVE_SCAN_OFFSETS,
        )
        adjacency = torch.cdist(
            positions[ids, :, :2], positions[ids, :, :2]
        ) <= 12.0
        for local_index, environment in enumerate(ids.tolist()):
            version = int(self.scan_versions[environment])
            step = int(step_counts[environment])
            for source in range(self.n_agents):
                cache = self._caches[environment][source]
                cache.observe_local(
                    proposals[local_index][source],
                    source_version=version,
                    observed_step=step,
                )
                cache.expire(current_step=step)
            sent, full = self._exchange(environment, adjacency[local_index])
            self.delta_records[environment] += int(sent)
            self.full_records[environment] += int(full)
            rover_positions = [
                tuple(float(value) for value in point[:2])
                for point in positions[environment]
            ]
            local_certificates = [
                self._select_certificate(cache, rover_positions)
                for cache in self._caches[environment]
            ]
            for agent, certificate in enumerate(local_certificates):
                groups = self._caches[environment][agent].proposals_by_source()
                known = sum(bool(group) for group in groups) / float(self.n_agents)
                self.known_source_fraction[environment, agent] = known
                if certificate is None:
                    self.target_valid[environment, agent] = False
                    self.target_points[environment, agent] = 0.0
                else:
                    self.target_valid[environment, agent] = True
                    self.target_points[environment, agent, :2] = torch.tensor(
                        certificate.center_xy,
                        dtype=positions.dtype,
                        device=self.device,
                    )
                    self.target_points[environment, agent, 2] = 0.0
            complete = all(item is not None for item in local_certificates)
            if complete:
                site_ids = {item.site_id for item in local_certificates if item}
                digests = {
                    item.proposal_set_digest for item in local_certificates if item
                }
                if len(site_ids) == 1 and len(digests) == 1:
                    certificate = local_certificates[0]
                    assert certificate is not None
                    self.committed[environment] = True
                    self.committed_steps[environment] = step
                    self.committed_centers[environment, :2] = torch.tensor(
                        certificate.center_xy,
                        dtype=positions.dtype,
                        device=self.device,
                    )
                    self.committed_centers[environment, 2] = 0.0
                    self.target_points[environment, :, :2] = (
                        self.committed_centers[environment, None, :2]
                    )
                    self.target_points[environment, :, 2] = 0.0
                    self.target_valid[environment] = True
            self.scan_versions[environment] += 1
        self._refresh_potential(ids)
        return ids

    def _refresh_potential(self, env_ids: Iterable[int] | torch.Tensor) -> None:
        ids = (
            env_ids.to(device=self.device, dtype=torch.long)
            if isinstance(env_ids, torch.Tensor)
            else torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        )
        if ids.numel() == 0:
            return
        known = self.known_source_fraction[ids].mean(dim=-1)
        valid = self.target_valid[ids].float().mean(dim=-1)
        self.potential[ids] = 0.5 * known + 0.5 * valid + self.committed[
            ids
        ].float()

    def mean_committed_distance(self, positions: torch.Tensor) -> torch.Tensor:
        distance = torch.linalg.vector_norm(
            positions[..., :2] - self.committed_centers[:, None, :2], dim=-1
        ).mean(dim=-1)
        return torch.where(self.committed, distance, torch.zeros_like(distance))

    def critic_site_points(self, positions: torch.Tensor) -> torch.Tensor:
        valid = self.target_valid.to(dtype=positions.dtype)
        summed = (self.target_points * valid[..., None]).sum(dim=1)
        count = valid.sum(dim=1, keepdim=True)
        local_mean = summed / count.clamp_min(1.0)
        fallback = positions.mean(dim=1)
        site = torch.where(count > 0.0, local_mean, fallback)
        return torch.where(
            self.committed[:, None], self.committed_centers, site
        )

    def diagnostics(self) -> dict[str, torch.Tensor]:
        reduction = 1.0 - self.delta_records.float() / self.full_records.clamp_min(
            1
        ).float()
        return {
            "target_valid_fraction": self.target_valid.float().mean(dim=-1).clone(),
            "known_source_fraction": self.known_source_fraction.mean(dim=-1).clone(),
            "committed": self.committed.clone(),
            "committed_step": self.committed_steps.clone(),
            "potential": self.potential.clone(),
            "delta_records": self.delta_records.clone(),
            "full_records": self.full_records.clone(),
            "delta_record_reduction": reduction,
        }


__all__ = ["ACTIVE_SCAN_OFFSETS", "ActiveDSTCRuntime"]
