from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lunar_rover_tasks.tasks.multi_rover_gathering.site_commitment import (  # noqa: E402
    CommitVote,
    ProposalBatchMessage,
    SiteCommitReplica,
    SiteBeliefCache,
    SiteProposal,
    build_site_witnesses,
    combined_proposal_set_digest,
    proposal_batch_digest,
    select_site_certificate,
    transform_points_se2,
    transform_proposal,
)


def _proposal(
    source: int,
    proposal_id: str,
    center: tuple[float, float],
    *,
    terrain_cost: float = 0.1,
) -> SiteProposal:
    return SiteProposal(
        epoch=0,
        source_id=source,
        proposal_id=proposal_id,
        local_center_xy=center,
        center_xy=center,
        verification_radius_m=1.25,
        required_radius_m=0.75,
        terrain_cost=terrain_cost,
        height_range_m=0.02,
        max_slope=0.03,
    )


def test_contraction_rejects_overlap_created_only_by_pose_dilation() -> None:
    proposals = [
        [_proposal(0, "a", (0.0, 0.0))],
        [_proposal(1, "b", (0.85, 0.0))],
    ]
    witnesses = build_site_witnesses(
        proposals,
        pose_uncertainty_m=0.10,
        min_support_rovers=2,
    )
    # Nominal center margins are 0.5 m. Contracting each by 0.1 m gives
    # 0.4+0.4 < 0.85, so uncertainty cannot manufacture a shared site.
    assert witnesses == []


def test_candidate_permutation_and_se2_do_not_change_selected_site() -> None:
    proposals = [
        [_proposal(0, "a", (0.0, 0.0)), _proposal(0, "x", (3.0, 0.0), terrain_cost=0.8)],
        [_proposal(1, "b", (0.1, 0.0)), _proposal(1, "y", (3.1, 0.0), terrain_cost=0.8)],
        [_proposal(2, "c", (0.0, 0.1))],
        [_proposal(3, "d", (0.1, 0.1))],
    ]
    batches = [
        ProposalBatchMessage(0, source, proposal_batch_digest(group))
        for source, group in enumerate(proposals)
    ]
    digest = combined_proposal_set_digest(batches)
    positions = [(-2.0, 0.0), (2.0, 0.0), (0.0, -2.0), (0.0, 2.0)]
    witnesses = build_site_witnesses(proposals, pose_uncertainty_m=0.05)
    certificate = select_site_certificate(
        witnesses, positions, epoch=0, proposal_set_digest=digest
    )
    assert certificate is not None

    permuted = [list(reversed(group)) for group in proposals]
    permuted_witnesses = build_site_witnesses(permuted, pose_uncertainty_m=0.05)
    permuted_certificate = select_site_certificate(
        permuted_witnesses, positions, epoch=0, proposal_set_digest=digest
    )
    assert permuted_certificate is not None
    assert permuted_certificate.site_id == certificate.site_id
    assert permuted_certificate.center_xy == pytest.approx(certificate.center_xy)

    angle = 0.73
    translation = (1.2, -2.4)
    transformed = [
        [
            transform_proposal(item, rotation_rad=angle, translation_xy=translation)
            for item in group
        ]
        for group in proposals
    ]
    transformed_positions = transform_points_se2(
        positions, rotation_rad=angle, translation_xy=translation
    )
    transformed_certificate = select_site_certificate(
        build_site_witnesses(transformed, pose_uncertainty_m=0.05),
        transformed_positions,
        epoch=0,
        proposal_set_digest=digest,
    )
    assert transformed_certificate is not None
    expected_center = transform_points_se2(
        [certificate.center_xy], rotation_rad=angle, translation_xy=translation
    )[0]
    assert transformed_certificate.site_id == certificate.site_id
    assert transformed_certificate.center_xy == pytest.approx(expected_center, abs=1.0e-7)
    assert transformed_certificate.selection_score == pytest.approx(
        certificate.selection_score, abs=1.0e-7
    )


def test_all_rover_commit_is_order_independent_and_stale_safe() -> None:
    proposals = [[_proposal(agent, f"p{agent}", (0.05 * agent, 0.0))] for agent in range(4)]
    batches = [
        ProposalBatchMessage(0, source, proposal_batch_digest(group))
        for source, group in enumerate(proposals)
    ]
    replicas = [SiteCommitReplica(agent_id=agent, n_agents=4) for agent in range(4)]
    rng = random.Random(160)
    for replica in replicas:
        delivery = batches * 2
        rng.shuffle(delivery)
        for message in delivery:
            replica.receive_batch(message)
        assert replica.ready
    digest = replicas[0].proposal_set_digest
    assert digest is not None
    assert all(replica.proposal_set_digest == digest for replica in replicas)

    votes = [replica.make_vote("site-A") for replica in replicas]
    stale = CommitVote(1, 0, digest, "site-B")
    for replica in replicas:
        delivery = [*votes, *votes, stale]
        rng.shuffle(delivery)
        for vote in delivery:
            replica.receive_vote(vote)
    assert {replica.committed_site_id for replica in replicas} == {"site-A"}
    assert not any(replica.conflicted for replica in replicas)


def test_conflicting_votes_fail_closed_without_split_brain() -> None:
    batches = [ProposalBatchMessage(0, source, f"digest-{source}") for source in range(4)]
    replicas = [SiteCommitReplica(agent_id=agent, n_agents=4) for agent in range(4)]
    for replica in replicas:
        for batch in batches:
            replica.receive_batch(batch)
    digest = replicas[0].proposal_set_digest
    assert digest is not None
    votes = [
        CommitVote(0, 0, digest, "site-A"),
        CommitVote(0, 1, digest, "site-A"),
        CommitVote(0, 2, digest, "site-B"),
        CommitVote(0, 3, digest, "site-B"),
    ]
    for replica in replicas:
        for vote in votes:
            replica.receive_vote(vote)
    assert all(replica.committed_site_id is None for replica in replicas)


def test_epoch_advance_requires_all_rover_release() -> None:
    replica = SiteCommitReplica(agent_id=0, n_agents=4)
    with pytest.raises(RuntimeError, match="all-rover release"):
        replica.advance_epoch(new_epoch=1, release_voters=(0, 1, 2))
    replica.advance_epoch(new_epoch=1, release_voters=(0, 1, 2, 3))
    assert replica.epoch == 1
    assert not replica.ready
    assert replica.committed_site_id is None


def test_required_flat_radius_contains_collision_safe_four_rover_square() -> None:
    collision_distance = 0.28
    clearance = 0.04
    side = collision_distance + clearance
    circumradius = side / math.sqrt(2.0)
    dmax = side * math.sqrt(2.0)
    dispersion = 0.5 * side**2
    assert circumradius < 0.75
    assert dmax < 1.25
    assert dispersion < 0.30


def test_site_belief_cache_is_bounded_versioned_and_stale_safe() -> None:
    cache = SiteBeliefCache(
        agent_id=0,
        n_agents=4,
        max_entries_per_source=2,
        ttl_steps=10,
    )
    proposals = [
        _proposal(0, "a", (0.0, 0.0), terrain_cost=0.3),
        _proposal(0, "b", (1.0, 0.0), terrain_cost=0.1),
        _proposal(0, "c", (2.0, 0.0), terrain_cost=0.2),
    ]
    cache.observe_local(proposals, source_version=2, observed_step=5)
    assert cache.record_count == 2
    assert {item.proposal_id for item in cache.proposals_by_source()[0]} == {"b", "c"}

    receiver = SiteBeliefCache(agent_id=1, n_agents=4, max_entries_per_source=2, ttl_steps=10)
    assert receiver.merge_message(cache.message()) == 2
    # Replaying the exact message is idempotent.
    assert receiver.merge_message(cache.message()) == 0
    assert receiver.record_count == 2
    assert receiver.expire(current_step=16) == 2
    assert receiver.record_count == 0


def test_dynamic_source_frame_makes_proposal_identity_odometric_not_body_slot() -> None:
    first = _proposal(0, "stable", (1.0, 0.0))
    cache = SiteBeliefCache(agent_id=0, n_agents=4)
    cache.observe_local([first], source_version=0, observed_step=0)
    updated = SiteProposal(
        epoch=0,
        source_id=0,
        proposal_id="stable",
        local_center_xy=(1.0, 0.0),
        center_xy=(2.0, 0.0),
        verification_radius_m=1.25,
        required_radius_m=0.75,
        terrain_cost=0.05,
        height_range_m=0.01,
        max_slope=0.02,
    )
    cache.observe_local([updated], source_version=1, observed_step=2)
    records = cache.proposals_by_source()[0]
    assert len(records) == 1
    assert records[0].center_xy == (2.0, 0.0)


def test_delta_messages_are_idempotent_and_omit_acknowledged_records() -> None:
    sender = SiteBeliefCache(agent_id=0, n_agents=4)
    receiver = SiteBeliefCache(agent_id=1, n_agents=4)
    sender.observe_local(
        [_proposal(0, "a", (0.0, 0.0)), _proposal(0, "b", (1.0, 0.0))],
        source_version=1,
        observed_step=4,
    )
    first = sender.delta_message({})
    assert len(first.records) == 2
    assert receiver.merge_delta(first) == 2
    assert receiver.merge_delta(first) == 0
    acknowledged = receiver.event_clock
    assert sender.delta_message(acknowledged).records == ()

    sender.observe_local(
        [_proposal(0, "a", (0.1, 0.0))],
        source_version=2,
        observed_step=8,
    )
    update = sender.delta_message(acknowledged)
    assert [record.proposal.proposal_id for record in update.records] == ["a"]
    assert receiver.merge_delta(update) == 1
    assert receiver.proposals_by_source()[0][0].center_xy == (0.1, 0.0)


def test_delta_tombstone_blocks_stale_out_of_order_upsert() -> None:
    sender = SiteBeliefCache(agent_id=0, n_agents=4, ttl_steps=2)
    receiver = SiteBeliefCache(agent_id=1, n_agents=4, ttl_steps=2)
    sender.observe_local(
        [_proposal(0, "a", (0.0, 0.0))],
        source_version=1,
        observed_step=1,
    )
    stale = sender.delta_message({})
    assert sender.expire(current_step=4) == 1
    deletion = sender.delta_message({})
    assert len(deletion.tombstones) == 1
    assert receiver.merge_delta(deletion) == 1
    assert receiver.merge_delta(stale) == 0
    assert receiver.proposals_by_source()[0] == []
