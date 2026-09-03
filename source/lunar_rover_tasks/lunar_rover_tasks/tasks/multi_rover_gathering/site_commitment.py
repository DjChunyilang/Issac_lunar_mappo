"""Decentralized flat-site proposals, robust witnesses and commit replicas.

This module implements the training-free H0 core of D-STC.  A proposal is
created only from one rover's local terrain footprint.  Proposal association
is geometric and permutation invariant; local candidate slot numbers never
cross the communication boundary.  The commit replica is deliberately
conservative: every rover must sign the same immutable proposal-set digest and
site id before a site becomes committed.

The module does not query an Oracle point and does not select or modify a rover
action.  Its output is a task-condition candidate for a later goal-conditioned
Actor, not an execution-time safety or planning override.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Sequence

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import TerrainCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    MULTISCALE_TERRAIN_COARSE_X,
    MULTISCALE_TERRAIN_COARSE_Y,
    MULTISCALE_TERRAIN_FINE_X,
    MULTISCALE_TERRAIN_FINE_Y,
    MULTISCALE_TERRAIN_MEDIUM_X,
    MULTISCALE_TERRAIN_MEDIUM_Y,
    TerrainRuntime,
    evaluate_gather_point_flatness,
)


@dataclass(frozen=True, slots=True)
class SiteProposal:
    """One rover-local terrain claim expressed in a comparison frame.

    ``local_center_xy`` is the stable source-frame coordinate used to build the
    proposal id. ``center_xy`` may be transformed into any receiver frame; all
    witness computations are SE(2)-equivariant and do not require a world frame.
    """

    epoch: int
    source_id: int
    proposal_id: str
    local_center_xy: tuple[float, float]
    center_xy: tuple[float, float]
    verification_radius_m: float
    required_radius_m: float
    terrain_cost: float
    height_range_m: float
    max_slope: float

    @property
    def center_margin_m(self) -> float:
        """Nominal radius in which a required footprint remains verified."""

        return self.verification_radius_m - self.required_radius_m


@dataclass(frozen=True, slots=True)
class SiteWitness:
    """A non-empty conservative intersection of compatible proposals."""

    site_id: str
    center_xy: tuple[float, float]
    required_radius_m: float
    support_ids: tuple[str, ...]
    source_ids: tuple[int, ...]
    terrain_cost: float
    containment_slack_m: float

    @property
    def support_count(self) -> int:
        return len(self.source_ids)


@dataclass(frozen=True, slots=True)
class SiteCertificate:
    """Deterministically selected common site task condition."""

    epoch: int
    site_id: str
    proposal_set_digest: str
    center_xy: tuple[float, float]
    required_radius_m: float
    support_ids: tuple[str, ...]
    source_ids: tuple[int, ...]
    selection_score: float
    containment_slack_m: float


@dataclass(frozen=True, slots=True)
class ProposalBatchMessage:
    epoch: int
    source_id: int
    payload_digest: str


@dataclass(frozen=True, slots=True)
class CommitVote:
    epoch: int
    voter_id: int
    proposal_set_digest: str
    site_id: str


@dataclass(frozen=True, slots=True)
class SiteBeliefRecord:
    proposal: SiteProposal
    source_version: int
    observed_step: int


@dataclass(frozen=True, slots=True)
class SiteBeliefMessage:
    epoch: int
    sender_id: int
    records: tuple[SiteBeliefRecord, ...]


@dataclass(frozen=True, slots=True)
class SiteBeliefTombstone:
    """Versioned deletion event for one proposal record."""

    epoch: int
    source_id: int
    proposal_id: str
    source_version: int
    observed_step: int


@dataclass(frozen=True, slots=True)
class SiteBeliefDeltaMessage:
    """Only belief events not acknowledged on one directed link."""

    epoch: int
    sender_id: int
    records: tuple[SiteBeliefRecord, ...]
    tombstones: tuple[SiteBeliefTombstone, ...]


def _sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _candidate_offsets(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the unique multiscale sample centers in deterministic order."""

    values: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for x_values, y_values in (
        (MULTISCALE_TERRAIN_FINE_X, MULTISCALE_TERRAIN_FINE_Y),
        (MULTISCALE_TERRAIN_MEDIUM_X, MULTISCALE_TERRAIN_MEDIUM_Y),
        (MULTISCALE_TERRAIN_COARSE_X, MULTISCALE_TERRAIN_COARSE_Y),
    ):
        for x in x_values:
            for y in y_values:
                key = (round(float(x), 6), round(float(y), 6))
                if key not in seen:
                    seen.add(key)
                    values.append(key)
    return torch.tensor(values, device=device, dtype=dtype)


def _body_to_frame(local_xy: torch.Tensor, positions: torch.Tensor, yaws: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(yaws)[..., None]
    sine = torch.sin(yaws)[..., None]
    x = cosine * local_xy[..., 0] - sine * local_xy[..., 1]
    y = sine * local_xy[..., 0] + cosine * local_xy[..., 1]
    return torch.stack((x, y), dim=-1) + positions[..., None, :2]


def _frame_to_body(points_xy: torch.Tensor, positions: torch.Tensor, yaws: torch.Tensor) -> torch.Tensor:
    delta = points_xy - positions[..., None, :2]
    cosine = torch.cos(yaws)[..., None]
    sine = torch.sin(yaws)[..., None]
    return torch.stack(
        (
            cosine * delta[..., 0] + sine * delta[..., 1],
            -sine * delta[..., 0] + cosine * delta[..., 1],
        ),
        dim=-1,
    )


def transform_proposal(
    proposal: SiteProposal,
    *,
    rotation_rad: float,
    translation_xy: tuple[float, float],
) -> SiteProposal:
    """Apply an SE(2) frame change without changing physical proposal identity."""

    cosine = math.cos(float(rotation_rad))
    sine = math.sin(float(rotation_rad))
    x, y = proposal.center_xy
    tx, ty = translation_xy
    return SiteProposal(
        epoch=proposal.epoch,
        source_id=proposal.source_id,
        proposal_id=proposal.proposal_id,
        local_center_xy=proposal.local_center_xy,
        center_xy=(cosine * x - sine * y + tx, sine * x + cosine * y + ty),
        verification_radius_m=proposal.verification_radius_m,
        required_radius_m=proposal.required_radius_m,
        terrain_cost=proposal.terrain_cost,
        height_range_m=proposal.height_range_m,
        max_slope=proposal.max_slope,
    )


def extract_local_site_proposals(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    terrain_cfg: TerrainCfg,
    terrain_runtime: TerrainRuntime,
    *,
    epoch: int = 0,
    max_candidates_per_rover: int = 4,
    verification_radius_m: float = 1.25,
    required_radius_m: float = 0.75,
    flatness_rings: int = 4,
    flatness_samples_per_ring: int = 16,
    required_flatness_rings: int = 3,
    required_flatness_samples_per_ring: int = 12,
    max_height_range_m: float = 0.18,
    max_slope: float = 0.25,
    nms_distance_m: float = 0.80,
    id_quantization_m: float = 0.05,
    source_frame_positions: torch.Tensor | None = None,
    source_frame_yaws: torch.Tensor | None = None,
    candidate_offsets_body: Sequence[tuple[float, float]] | torch.Tensor | None = None,
) -> list[list[list[SiteProposal]]]:
    """Extract at most ``K`` source-local flat-region proposals per rover.

    The queried points are exactly the current multiscale body-frame sample
    centers.  Each center is accepted only when a larger verification disk is
    flat.  The excess radius is later consumed by candidate disagreement and
    relative-pose uncertainty; no terrain outside the rover-local footprint is
    used for a proposal.
    """

    if positions.ndim != 3 or positions.shape[-1] < 2:
        raise ValueError("positions must have shape [E, A, >=2].")
    if yaws.shape != positions.shape[:2]:
        raise ValueError("yaws must have shape [E, A].")
    if max_candidates_per_rover <= 0:
        raise ValueError("max_candidates_per_rover must be positive.")
    if verification_radius_m <= required_radius_m:
        raise ValueError("verification_radius_m must exceed required_radius_m.")
    if nms_distance_m < 0.0 or id_quantization_m <= 0.0:
        raise ValueError("NMS distance and id quantization are invalid.")

    if candidate_offsets_body is None:
        offsets = _candidate_offsets(device=positions.device, dtype=positions.dtype)
    else:
        offsets = torch.as_tensor(
            candidate_offsets_body,
            device=positions.device,
            dtype=positions.dtype,
        )
        if offsets.ndim != 2 or offsets.shape[-1] != 2 or offsets.shape[0] == 0:
            raise ValueError("candidate_offsets_body must have shape [M, 2].")
    expanded_offsets = offsets.view(1, 1, -1, 2).expand(
        positions.shape[0], positions.shape[1], -1, -1
    )
    centers = _body_to_frame(expanded_offsets, positions, yaws)
    if source_frame_positions is None:
        source_frame_positions = positions
    if source_frame_yaws is None:
        source_frame_yaws = yaws
    if source_frame_positions.shape != positions.shape:
        raise ValueError("source_frame_positions must match positions.")
    if source_frame_yaws.shape != yaws.shape:
        raise ValueError("source_frame_yaws must match yaws.")
    source_frame_centers = _frame_to_body(
        centers,
        source_frame_positions,
        source_frame_yaws,
    )
    flatness = evaluate_gather_point_flatness(
        centers,
        terrain_cfg,
        terrain_runtime,
        radius=float(verification_radius_m),
        rings=int(flatness_rings),
        samples_per_ring=int(flatness_samples_per_ring),
        max_height_range=float(max_height_range_m),
        max_slope=float(max_slope),
    )
    # The larger verification footprint is not a mathematical substitute for
    # the task's exact discrete success-gate samples.  Re-evaluate every source
    # proposal with the required footprint so an association can never move a
    # centroid onto a location that no rover actually verified with that gate.
    required_flatness = evaluate_gather_point_flatness(
        centers,
        terrain_cfg,
        terrain_runtime,
        radius=float(required_radius_m),
        rings=int(required_flatness_rings),
        samples_per_ring=int(required_flatness_samples_per_ring),
        max_height_range=float(max_height_range_m),
        max_slope=float(max_slope),
    )
    height_cost = flatness.height_range / max(float(max_height_range_m), 1.0e-9)
    slope_cost = flatness.max_slope / max(float(max_slope), 1.0e-9)
    terrain_cost = torch.maximum(height_cost, slope_cost)
    distance_cost = torch.linalg.vector_norm(offsets, dim=-1)
    distance_cost = distance_cost / max(float(distance_cost.max()), 1.0e-9)
    score = terrain_cost + 0.02 * distance_cost.view(1, 1, -1)
    available = flatness.is_flat & required_flatness.is_flat

    selected_indices: list[torch.Tensor] = []
    selected_valid: list[torch.Tensor] = []
    offset_distance = torch.cdist(offsets, offsets)
    for _ in range(max_candidates_per_rover):
        masked = score.masked_fill(~available, float("inf"))
        index = masked.argmin(dim=-1)
        valid = torch.isfinite(masked.gather(-1, index[..., None]).squeeze(-1))
        selected_indices.append(index)
        selected_valid.append(valid)
        suppress = offset_distance[index] < float(nms_distance_m)
        available = available & ~suppress & valid[..., None]

    result: list[list[list[SiteProposal]]] = []
    for environment in range(positions.shape[0]):
        environment_proposals: list[list[SiteProposal]] = []
        for source in range(positions.shape[1]):
            source_proposals: list[SiteProposal] = []
            for rank, (indices, validity) in enumerate(zip(selected_indices, selected_valid)):
                if not bool(validity[environment, source]):
                    continue
                index = int(indices[environment, source])
                local = source_frame_centers[environment, source, index]
                frame_center = centers[environment, source, index]
                qx = int(round(float(local[0]) / id_quantization_m))
                qy = int(round(float(local[1]) / id_quantization_m))
                proposal_id = _sha256(
                    {
                        "epoch": int(epoch),
                        "source": int(source),
                        "local_center_q": [qx, qy],
                        "verification_radius_mm": int(round(1000.0 * verification_radius_m)),
                    }
                )[:20]
                del rank  # rank must never enter identity or transmitted semantics.
                source_proposals.append(
                    SiteProposal(
                        epoch=int(epoch),
                        source_id=int(source),
                        proposal_id=proposal_id,
                        local_center_xy=(float(local[0]), float(local[1])),
                        center_xy=(float(frame_center[0]), float(frame_center[1])),
                        verification_radius_m=float(verification_radius_m),
                        required_radius_m=float(required_radius_m),
                        terrain_cost=float(terrain_cost[environment, source, index]),
                        height_range_m=float(flatness.height_range[environment, source, index]),
                        max_slope=float(flatness.max_slope[environment, source, index]),
                    )
                )
            environment_proposals.append(source_proposals)
        result.append(environment_proposals)
    return result


def _circle_intersections(
    center_a: tuple[float, float],
    radius_a: float,
    center_b: tuple[float, float],
    radius_b: float,
) -> list[tuple[float, float]]:
    ax, ay = center_a
    bx, by = center_b
    dx, dy = bx - ax, by - ay
    distance = math.hypot(dx, dy)
    tolerance = 1.0e-10
    if distance <= tolerance:
        return []
    if distance > radius_a + radius_b + tolerance:
        return []
    if distance < abs(radius_a - radius_b) - tolerance:
        return []
    along = (radius_a**2 - radius_b**2 + distance**2) / (2.0 * distance)
    height_sq = max(radius_a**2 - along**2, 0.0)
    height = math.sqrt(height_sq)
    mid_x = ax + along * dx / distance
    mid_y = ay + along * dy / distance
    perpendicular_x = -dy / distance
    perpendicular_y = dx / distance
    first = (mid_x + height * perpendicular_x, mid_y + height * perpendicular_y)
    if height <= tolerance:
        return [first]
    return [first, (mid_x - height * perpendicular_x, mid_y - height * perpendicular_y)]


def robust_disc_intersection_witness(
    centers: Sequence[tuple[float, float]],
    radii: Sequence[float],
    *,
    tolerance: float = 1.0e-8,
) -> tuple[tuple[float, float], float] | None:
    """Return a point and minimum slack for a finite intersection of disks."""

    if not centers or len(centers) != len(radii):
        raise ValueError("centers and radii must be non-empty and equally sized.")
    if any(radius < 0.0 for radius in radii):
        return None
    # Dykstra projection of the arithmetic mean onto the disk intersection is
    # unique and SE(2)-equivariant.  Using a lexicographically selected circle
    # intersection would make the task condition depend on the world axes.
    mean = (
        sum(center[0] for center in centers) / len(centers),
        sum(center[1] for center in centers) / len(centers),
    )
    point = mean
    corrections = [(0.0, 0.0) for _ in centers]
    for _ in range(256):
        previous_cycle = point
        for index, (center, radius) in enumerate(zip(centers, radii)):
            y = (
                point[0] + corrections[index][0],
                point[1] + corrections[index][1],
            )
            dx, dy = y[0] - center[0], y[1] - center[1]
            distance = math.hypot(dx, dy)
            if distance <= radius or distance <= 1.0e-15:
                projected = y
            else:
                scale = radius / distance
                projected = (center[0] + scale * dx, center[1] + scale * dy)
            corrections[index] = (y[0] - projected[0], y[1] - projected[1])
            point = projected
        if math.dist(point, previous_cycle) <= 1.0e-12:
            break
    projected_slack = min(
        radius - math.dist(point, center) for center, radius in zip(centers, radii)
    )
    if projected_slack >= -float(tolerance):
        return point, projected_slack

    # The analytic candidates are a conservative numerical fallback for nearly
    # tangent disks where finite Dykstra iterations may miss tolerance.
    candidates = [mean, *centers]
    for first in range(len(centers)):
        for second in range(first + 1, len(centers)):
            candidates.extend(
                _circle_intersections(
                    centers[first], radii[first], centers[second], radii[second]
                )
            )
    valid: list[tuple[float, tuple[float, ...], tuple[float, float]]] = []
    for point in candidates:
        slack = min(
            radius - math.dist(point, center)
            for center, radius in zip(centers, radii)
        )
        if slack >= -float(tolerance):
            squared_distance = sum(math.dist(point, center) ** 2 for center in centers)
            invariant_tie_break = tuple(math.dist(point, center) for center in centers)
            valid.append((squared_distance, invariant_tie_break, point))
    if not valid:
        return None
    _, _, point = min(valid, key=lambda item: (item[0], item[1]))
    slack = min(radius - math.dist(point, center) for center, radius in zip(centers, radii))
    return point, slack


def build_site_witnesses(
    proposals_by_source: Sequence[Sequence[SiteProposal]],
    *,
    pose_uncertainty_m: float,
    min_support_rovers: int = 1,
) -> list[SiteWitness]:
    """Associate proposals through a conservative common-center certificate.

    Every supporting proposal contributes a disk with radius
    ``R_verify - R_required - epsilon_pose``.  Contracting by pose uncertainty
    guarantees that a returned witness cannot be created solely by uncertainty
    dilation.  Candidate slot numbers are absent from this computation.
    """

    if pose_uncertainty_m < 0.0:
        raise ValueError("pose_uncertainty_m must be non-negative.")
    if min_support_rovers <= 0:
        raise ValueError("min_support_rovers must be positive.")
    flat = [proposal for group in proposals_by_source for proposal in group]
    witnesses: dict[tuple[str, ...], SiteWitness] = {}
    for anchor in sorted(flat, key=lambda proposal: proposal.proposal_id):
        anchor_radius = anchor.center_margin_m - pose_uncertainty_m
        if anchor_radius < 0.0:
            continue
        selected = [anchor]
        for source_id, group in enumerate(proposals_by_source):
            if source_id == anchor.source_id:
                continue
            compatible = []
            for candidate in group:
                candidate_radius = candidate.center_margin_m - pose_uncertainty_m
                if candidate_radius < 0.0:
                    continue
                # The certificate center remains the anchor's source-verified
                # point. Supporting proposals may strengthen the evidence only
                # when their contracted verified region contains that point.
                if math.dist(anchor.center_xy, candidate.center_xy) <= candidate_radius + 1.0e-8:
                    compatible.append((math.dist(anchor.center_xy, candidate.center_xy), candidate))
            if compatible:
                selected.append(min(compatible, key=lambda item: (item[0], item[1].proposal_id))[1])
        if len(selected) < min_support_rovers:
            continue
        center = anchor.center_xy
        slack = min(
            item.center_margin_m
            - pose_uncertainty_m
            - math.dist(center, item.center_xy)
            for item in selected
        )
        if slack < -1.0e-8:
            continue
        support_ids = tuple(sorted(item.proposal_id for item in selected))
        source_ids = tuple(sorted(item.source_id for item in selected))
        site_id = _sha256({"epoch": anchor.epoch, "support_ids": support_ids})[:24]
        witness = SiteWitness(
            site_id=site_id,
            center_xy=center,
            required_radius_m=max(item.required_radius_m for item in selected),
            support_ids=support_ids,
            source_ids=source_ids,
            terrain_cost=max(item.terrain_cost for item in selected),
            containment_slack_m=float(slack),
        )
        existing = witnesses.get(support_ids)
        if existing is None or (
            witness.terrain_cost,
            witness.site_id,
        ) < (
            existing.terrain_cost,
            existing.site_id,
        ):
            witnesses[support_ids] = witness
    return sorted(witnesses.values(), key=lambda item: item.site_id)


def proposal_batch_digest(proposals: Iterable[SiteProposal]) -> str:
    rows = [
        {
            "epoch": item.epoch,
            "source_id": item.source_id,
            "proposal_id": item.proposal_id,
            "local_center_mm": [
                int(round(1000.0 * item.local_center_xy[0])),
                int(round(1000.0 * item.local_center_xy[1])),
            ],
            "verification_radius_mm": int(round(1000.0 * item.verification_radius_m)),
            "required_radius_mm": int(round(1000.0 * item.required_radius_m)),
        }
        for item in sorted(proposals, key=lambda proposal: proposal.proposal_id)
    ]
    return _sha256(rows)


def combined_proposal_set_digest(
    messages: Iterable[ProposalBatchMessage],
) -> str:
    rows = [
        (message.epoch, message.source_id, message.payload_digest)
        for message in sorted(messages, key=lambda item: item.source_id)
    ]
    return _sha256(rows)


def select_site_certificate(
    witnesses: Sequence[SiteWitness],
    rover_positions_xy: Sequence[tuple[float, float]],
    *,
    epoch: int,
    proposal_set_digest: str,
    distance_weight: float = 0.03,
    support_weight: float = 0.02,
) -> SiteCertificate | None:
    """Select a site by terrain quality, team travel cost and stable site id."""

    if distance_weight < 0.0 or support_weight < 0.0:
        raise ValueError("distance_weight and support_weight must be non-negative.")
    if not witnesses:
        return None
    ranked = []
    for witness in witnesses:
        travel = sum(
            math.dist(position, witness.center_xy) for position in rover_positions_xy
        ) / max(len(rover_positions_xy), 1)
        score = (
            witness.terrain_cost
            + distance_weight * travel
            - support_weight * float(max(witness.support_count - 1, 0))
        )
        ranked.append((score, witness.site_id, witness))
    score, _, selected = min(ranked, key=lambda item: (item[0], item[1]))
    return SiteCertificate(
        epoch=int(epoch),
        site_id=selected.site_id,
        proposal_set_digest=str(proposal_set_digest),
        center_xy=selected.center_xy,
        required_radius_m=selected.required_radius_m,
        support_ids=selected.support_ids,
        source_ids=selected.source_ids,
        selection_score=float(score),
        containment_slack_m=selected.containment_slack_m,
    )


class SiteCommitReplica:
    """Non-Byzantine all-rover commit register with immutable epoch votes."""

    def __init__(self, *, agent_id: int, n_agents: int, epoch: int = 0) -> None:
        if not 0 <= agent_id < n_agents:
            raise ValueError("agent_id must be within the team.")
        self.agent_id = int(agent_id)
        self.n_agents = int(n_agents)
        self.epoch = int(epoch)
        self._batches: dict[int, ProposalBatchMessage] = {}
        self._batch_conflict = False
        self._votes: dict[int, CommitVote] = {}
        self._vote_conflict = False
        self.committed_site_id: str | None = None

    @property
    def ready(self) -> bool:
        return not self._batch_conflict and len(self._batches) == self.n_agents

    @property
    def proposal_set_digest(self) -> str | None:
        if not self.ready:
            return None
        return combined_proposal_set_digest(self._batches.values())

    @property
    def conflicted(self) -> bool:
        return self._batch_conflict or self._vote_conflict

    def receive_batch(self, message: ProposalBatchMessage) -> None:
        if message.epoch != self.epoch or not 0 <= message.source_id < self.n_agents:
            return
        previous = self._batches.get(message.source_id)
        if previous is not None and previous.payload_digest != message.payload_digest:
            self._batch_conflict = True
            return
        self._batches[message.source_id] = message

    def make_vote(self, site_id: str) -> CommitVote:
        digest = self.proposal_set_digest
        if digest is None:
            raise RuntimeError("Cannot vote before all immutable source batches arrive.")
        return CommitVote(
            epoch=self.epoch,
            voter_id=self.agent_id,
            proposal_set_digest=digest,
            site_id=str(site_id),
        )

    def receive_vote(self, vote: CommitVote) -> None:
        if vote.epoch != self.epoch or not 0 <= vote.voter_id < self.n_agents:
            return
        digest = self.proposal_set_digest
        if digest is None or vote.proposal_set_digest != digest:
            return
        previous = self._votes.get(vote.voter_id)
        if previous is not None and previous.site_id != vote.site_id:
            self._vote_conflict = True
            return
        self._votes[vote.voter_id] = vote
        if self._vote_conflict or len(self._votes) != self.n_agents:
            return
        site_ids = {item.site_id for item in self._votes.values()}
        if len(site_ids) == 1:
            site_id = next(iter(site_ids))
            if self.committed_site_id is None:
                self.committed_site_id = site_id
            elif self.committed_site_id != site_id:
                self._vote_conflict = True

    def advance_epoch(self, *, new_epoch: int, release_voters: Iterable[int]) -> None:
        """Advance only with an all-rover release certificate."""

        if int(new_epoch) <= self.epoch:
            return
        if set(int(item) for item in release_voters) != set(range(self.n_agents)):
            raise RuntimeError("Advancing a committed epoch requires all-rover release.")
        self.epoch = int(new_epoch)
        self._batches.clear()
        self._votes.clear()
        self._batch_conflict = False
        self._vote_conflict = False
        self.committed_site_id = None


class SiteBeliefCache:
    """Finite, versioned proposal memory suitable for bounded forwarding.

    Records retain the original source id and observation step when forwarded.
    Duplicate and out-of-order messages are idempotent; a lower source version
    can never overwrite a newer observation of the same physical proposal.
    """

    def __init__(
        self,
        *,
        agent_id: int,
        n_agents: int,
        epoch: int = 0,
        max_entries_per_source: int = 4,
        ttl_steps: int = 480,
    ) -> None:
        if not 0 <= agent_id < n_agents:
            raise ValueError("agent_id must be within the team.")
        if max_entries_per_source <= 0 or ttl_steps <= 0:
            raise ValueError("belief capacity and ttl_steps must be positive.")
        self.agent_id = int(agent_id)
        self.n_agents = int(n_agents)
        self.epoch = int(epoch)
        self.max_entries_per_source = int(max_entries_per_source)
        self.ttl_steps = int(ttl_steps)
        self._records: dict[int, dict[str, SiteBeliefRecord]] = {
            source: {} for source in range(self.n_agents)
        }
        self._tombstones: dict[int, dict[str, SiteBeliefTombstone]] = {
            source: {} for source in range(self.n_agents)
        }

    def _trim_source(self, source_id: int) -> None:
        records = self._records[source_id]
        if len(records) <= self.max_entries_per_source:
            return
        ranked = sorted(
            records.values(),
            key=lambda item: (
                item.proposal.terrain_cost,
                -item.source_version,
                -item.observed_step,
                item.proposal.proposal_id,
            ),
        )[: self.max_entries_per_source]
        self._records[source_id] = {
            item.proposal.proposal_id: item for item in ranked
        }

    def observe_local(
        self,
        proposals: Sequence[SiteProposal],
        *,
        source_version: int,
        observed_step: int,
    ) -> None:
        for proposal in proposals:
            if proposal.epoch != self.epoch or proposal.source_id != self.agent_id:
                raise ValueError("Local proposals must match cache epoch and agent id.")
            self._merge_record(
                SiteBeliefRecord(
                    proposal=proposal,
                    source_version=int(source_version),
                    observed_step=int(observed_step),
                )
            )
        self._trim_source(self.agent_id)

    def _merge_record(self, record: SiteBeliefRecord) -> bool:
        proposal = record.proposal
        if proposal.epoch != self.epoch:
            return False
        if not 0 <= proposal.source_id < self.n_agents:
            return False
        records = self._records[proposal.source_id]
        tombstone = self._tombstones[proposal.source_id].get(proposal.proposal_id)
        incoming_key = (record.source_version, record.observed_step, 0)
        if tombstone is not None:
            tombstone_key = (
                tombstone.source_version,
                tombstone.observed_step,
                1,
            )
            if incoming_key <= tombstone_key:
                return False
            del self._tombstones[proposal.source_id][proposal.proposal_id]
        previous = records.get(proposal.proposal_id)
        previous_key = (
            previous.source_version,
            previous.observed_step,
        ) if previous is not None else (-1, -1)
        record_key = (record.source_version, record.observed_step)
        if record_key < previous_key:
            return False
        if record_key == previous_key and previous is not None:
            return False
        records[proposal.proposal_id] = record
        self._trim_source(proposal.source_id)
        return True

    def _merge_tombstone(self, tombstone: SiteBeliefTombstone) -> bool:
        if tombstone.epoch != self.epoch:
            return False
        if not 0 <= tombstone.source_id < self.n_agents:
            return False
        records = self._records[tombstone.source_id]
        previous_record = records.get(tombstone.proposal_id)
        previous_tombstone = self._tombstones[tombstone.source_id].get(
            tombstone.proposal_id
        )
        incoming_key = (tombstone.source_version, tombstone.observed_step, 1)
        previous_keys = [(-1, -1, -1)]
        if previous_record is not None:
            previous_keys.append(
                (previous_record.source_version, previous_record.observed_step, 0)
            )
        if previous_tombstone is not None:
            previous_keys.append(
                (
                    previous_tombstone.source_version,
                    previous_tombstone.observed_step,
                    1,
                )
            )
        if incoming_key <= max(previous_keys):
            return False
        records.pop(tombstone.proposal_id, None)
        self._tombstones[tombstone.source_id][tombstone.proposal_id] = tombstone
        return True

    @property
    def event_clock(self) -> dict[tuple[int, str], tuple[int, int, int]]:
        """Return the locally acknowledged version of every cached event."""

        clock: dict[tuple[int, str], tuple[int, int, int]] = {}
        for source_id, records in self._records.items():
            for proposal_id, record in records.items():
                clock[(source_id, proposal_id)] = (
                    record.source_version,
                    record.observed_step,
                    0,
                )
        for source_id, tombstones in self._tombstones.items():
            for proposal_id, tombstone in tombstones.items():
                key = (source_id, proposal_id)
                value = (
                    tombstone.source_version,
                    tombstone.observed_step,
                    1,
                )
                if value > clock.get(key, (-1, -1, -1)):
                    clock[key] = value
        return clock

    def delta_message(
        self,
        acknowledged_clock: dict[tuple[int, str], tuple[int, int, int]] | None = None,
    ) -> SiteBeliefDeltaMessage:
        """Create an idempotent event delta for one acknowledged link state."""

        acknowledged_clock = acknowledged_clock or {}
        records = tuple(
            sorted(
                (
                    record
                    for source_records in self._records.values()
                    for record in source_records.values()
                    if (
                        record.source_version,
                        record.observed_step,
                        0,
                    )
                    > acknowledged_clock.get(
                        (record.proposal.source_id, record.proposal.proposal_id),
                        (-1, -1, -1),
                    )
                ),
                key=lambda item: (
                    item.proposal.source_id,
                    item.proposal.proposal_id,
                ),
            )
        )
        tombstones = tuple(
            sorted(
                (
                    tombstone
                    for source_tombstones in self._tombstones.values()
                    for tombstone in source_tombstones.values()
                    if (
                        tombstone.source_version,
                        tombstone.observed_step,
                        1,
                    )
                    > acknowledged_clock.get(
                        (tombstone.source_id, tombstone.proposal_id),
                        (-1, -1, -1),
                    )
                ),
                key=lambda item: (item.source_id, item.proposal_id),
            )
        )
        return SiteBeliefDeltaMessage(
            epoch=self.epoch,
            sender_id=self.agent_id,
            records=records,
            tombstones=tombstones,
        )

    def merge_delta(self, message: SiteBeliefDeltaMessage) -> int:
        if message.epoch != self.epoch or not 0 <= message.sender_id < self.n_agents:
            return 0
        changed = 0
        for record in message.records:
            changed += int(self._merge_record(record))
        for tombstone in message.tombstones:
            changed += int(self._merge_tombstone(tombstone))
        return changed

    def message(self) -> SiteBeliefMessage:
        records = tuple(
            sorted(
                (
                    record
                    for source_records in self._records.values()
                    for record in source_records.values()
                ),
                key=lambda item: (
                    item.proposal.source_id,
                    item.proposal.proposal_id,
                ),
            )
        )
        return SiteBeliefMessage(
            epoch=self.epoch,
            sender_id=self.agent_id,
            records=records,
        )

    def merge_message(self, message: SiteBeliefMessage) -> int:
        if message.epoch != self.epoch or not 0 <= message.sender_id < self.n_agents:
            return 0
        changed = 0
        for record in message.records:
            changed += int(self._merge_record(record))
        return changed

    def expire(self, *, current_step: int) -> int:
        removed = 0
        for source_id, records in self._records.items():
            keep = {}
            expired: list[SiteBeliefRecord] = []
            for proposal_id, record in records.items():
                if int(current_step) - record.observed_step <= self.ttl_steps:
                    keep[proposal_id] = record
                    continue
                expired.append(record)
            removed += len(expired)
            self._records[source_id] = keep
            for record in expired:
                self._merge_tombstone(
                    SiteBeliefTombstone(
                        epoch=self.epoch,
                        source_id=source_id,
                        proposal_id=record.proposal.proposal_id,
                        source_version=record.source_version,
                        observed_step=record.observed_step,
                    )
                )
        return removed

    def proposals_by_source(self) -> list[list[SiteProposal]]:
        return [
            [
                record.proposal
                for record in sorted(
                    self._records[source].values(),
                    key=lambda item: item.proposal.proposal_id,
                )
            ]
            for source in range(self.n_agents)
        ]

    @property
    def record_count(self) -> int:
        return sum(len(records) for records in self._records.values())


def transform_points_se2(
    points: Sequence[tuple[float, float]],
    *,
    rotation_rad: float,
    translation_xy: tuple[float, float],
) -> list[tuple[float, float]]:
    cosine = math.cos(float(rotation_rad))
    sine = math.sin(float(rotation_rad))
    tx, ty = translation_xy
    return [
        (cosine * x - sine * y + tx, sine * x + cosine * y + ty)
        for x, y in points
    ]


__all__ = [
    "CommitVote",
    "ProposalBatchMessage",
    "SiteCertificate",
    "SiteBeliefCache",
    "SiteBeliefDeltaMessage",
    "SiteBeliefMessage",
    "SiteBeliefRecord",
    "SiteBeliefTombstone",
    "SiteCommitReplica",
    "SiteProposal",
    "SiteWitness",
    "build_site_witnesses",
    "combined_proposal_set_digest",
    "extract_local_site_proposals",
    "proposal_batch_digest",
    "robust_disc_intersection_witness",
    "select_site_certificate",
    "transform_points_se2",
    "transform_proposal",
]
