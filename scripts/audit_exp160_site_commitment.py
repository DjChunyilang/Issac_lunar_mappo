#!/usr/bin/env python3
"""Frozen H0 audit for decentralized site proposals and all-rover commit.

This script performs no learning, creates no teacher actions and does not feed
the certificate to the Actor. It reuses the six fixed exp156 scenario cells to
test whether rover-local terrain evidence can produce conservative common-site
certificates and whether an immutable all-rover commit register remains
split-brain free under message reorder, duplication, loss and stale replay.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from datetime import datetime, timezone

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.site_commitment import (
    CommitVote,
    ProposalBatchMessage,
    SiteCertificate,
    SiteCommitReplica,
    SiteProposal,
    build_site_witnesses,
    combined_proposal_set_digest,
    extract_local_site_proposals,
    proposal_batch_digest,
    select_site_certificate,
    transform_points_se2,
    transform_proposal,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    evaluate_gather_point_flatness,
)


def _communication_reachability(positions: torch.Tensor, radius_m: float) -> tuple[torch.Tensor, torch.Tensor]:
    distance = torch.cdist(positions[..., :2], positions[..., :2])
    adjacency = distance <= float(radius_m)
    reach = adjacency.clone()
    for pivot in range(positions.shape[1]):
        reach = reach | (reach[:, :, pivot, None] & reach[:, None, pivot, :])
    return adjacency.all(dim=(1, 2)), reach.all(dim=(1, 2))


def _batch_messages(groups: list[list[SiteProposal]], epoch: int) -> list[ProposalBatchMessage]:
    return [
        ProposalBatchMessage(
            epoch=int(epoch),
            source_id=source,
            payload_digest=proposal_batch_digest(proposals),
        )
        for source, proposals in enumerate(groups)
    ]


def _selected_certificate(
    groups: list[list[SiteProposal]],
    positions: list[tuple[float, float]],
    *,
    epoch: int,
    pose_uncertainty_m: float,
    min_support: int,
    distance_weight: float,
    support_weight: float,
) -> SiteCertificate | None:
    batches = _batch_messages(groups, epoch)
    digest = combined_proposal_set_digest(batches)
    witnesses = build_site_witnesses(
        groups,
        pose_uncertainty_m=pose_uncertainty_m,
        min_support_rovers=min_support,
    )
    return select_site_certificate(
        witnesses,
        positions,
        epoch=epoch,
        proposal_set_digest=digest,
        distance_weight=distance_weight,
        support_weight=support_weight,
    )


def _protocol_checks(
    groups: list[list[SiteProposal]],
    certificate: SiteCertificate,
    *,
    seed: int,
) -> dict[str, bool]:
    n_agents = len(groups)
    batches = _batch_messages(groups, certificate.epoch)
    rng = random.Random(int(seed))

    replicas = [
        SiteCommitReplica(agent_id=agent, n_agents=n_agents, epoch=certificate.epoch)
        for agent in range(n_agents)
    ]
    for replica in replicas:
        delivery = [*batches, *batches]
        rng.shuffle(delivery)
        for message in delivery:
            replica.receive_batch(message)
    full_view = all(replica.ready for replica in replicas)
    same_digest = len({replica.proposal_set_digest for replica in replicas}) == 1
    votes = [replica.make_vote(certificate.site_id) for replica in replicas]
    for replica in replicas:
        delivery = [*votes, *votes]
        rng.shuffle(delivery)
        for vote in delivery:
            replica.receive_vote(vote)
        replica.receive_vote(
            CommitVote(
                epoch=certificate.epoch + 1,
                voter_id=0,
                proposal_set_digest=certificate.proposal_set_digest,
                site_id="stale-conflict",
            )
        )
    complete_agreement = {
        replica.committed_site_id for replica in replicas
    } == {certificate.site_id}

    # Loss is fail-closed: every replica misses a different immutable source
    # batch, so no replica may issue a vote or commit a site.
    lossy = [
        SiteCommitReplica(agent_id=agent, n_agents=n_agents, epoch=certificate.epoch)
        for agent in range(n_agents)
    ]
    for replica in lossy:
        for message in batches:
            if message.source_id != replica.agent_id:
                replica.receive_batch(message)
    loss_fail_closed = all(
        not replica.ready and replica.committed_site_id is None for replica in lossy
    )

    # Conflicting local preferences cannot each gather all-rover signatures.
    split = [
        SiteCommitReplica(agent_id=agent, n_agents=n_agents, epoch=certificate.epoch)
        for agent in range(n_agents)
    ]
    for replica in split:
        for message in batches:
            replica.receive_batch(message)
    digest = split[0].proposal_set_digest
    split_votes = [
        CommitVote(
            epoch=certificate.epoch,
            voter_id=agent,
            proposal_set_digest=str(digest),
            site_id="site-A" if agent < n_agents // 2 else "site-B",
        )
        for agent in range(n_agents)
    ]
    for replica in split:
        for vote in split_votes:
            replica.receive_vote(vote)
    conflict_fail_closed = all(replica.committed_site_id is None for replica in split)
    return {
        "full_view": full_view,
        "same_digest": same_digest,
        "complete_agreement": complete_agreement,
        "loss_fail_closed": loss_fail_closed,
        "conflict_fail_closed": conflict_fail_closed,
    }


def _support_containment(
    groups: list[list[SiteProposal]],
    certificate: SiteCertificate,
    pose_uncertainty_m: float,
) -> bool:
    lookup = {
        proposal.proposal_id: proposal
        for group in groups
        for proposal in group
    }
    for support_id in certificate.support_ids:
        proposal = lookup[support_id]
        consumed = (
            math.dist(certificate.center_xy, proposal.center_xy)
            + float(pose_uncertainty_m)
            + certificate.required_radius_m
        )
        if consumed > proposal.verification_radius_m + 1.0e-7:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiment/exp160_dstc_h0_certificate.yaml"
    )
    parser.add_argument(
        "--manifest",
        default=(
            "outputs/runs/exp156_differential_multiscale_ablation/_suite/"
            "scenario_manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/runs/exp160_dstc_site_commitment/h0_certificate_audit/"
            "metrics/h0_certificate_audit.json"
        ),
    )
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = load_yaml(args.config)
    settings = raw.get("site_commitment", {})
    required_keys = {
        "epoch",
        "max_candidates_per_rover",
        "verification_radius_m",
        "required_flat_radius_m",
        "pose_uncertainty_m",
        "flatness_rings",
        "flatness_samples_per_ring",
        "nms_distance_m",
        "id_quantization_m",
        "minimum_support_for_commit",
        "distance_score_weight",
        "support_score_weight",
        "terminal_clearance_m",
        "se2_rotation_rad",
        "se2_translation_xy",
    }
    missing = sorted(required_keys - settings.keys())
    if missing:
        raise ValueError(f"Missing site_commitment setting(s): {', '.join(missing)}")

    all_cell_reports = []
    aggregate_counts = {
        "episodes": 0,
        "rovers": 0,
        "rovers_with_candidate": 0,
        "episodes_with_any_candidate": 0,
        "certificate_support1": 0,
        "certificate_support2": 0,
        "certificate_support4": 0,
        "selected_support_sum": 0,
        "selected_support_count": 0,
        "central_flatness_pass": 0,
        "containment_pass": 0,
        "permutation_pass": 0,
        "se2_pass": 0,
        "protocol_complete_pass": 0,
        "protocol_adverse_pass": 0,
        "connected": 0,
        "clique": 0,
    }
    max_se2_error = 0.0

    for cell in manifest["cells"]:
        cell_raw = copy.deepcopy(raw)
        cell_raw.setdefault("experiment", {})["seed"] = int(cell["seed"])
        cell_raw["experiment"]["num_envs"] = int(manifest["episodes_per_cell"])
        cell_raw.setdefault("initial_state", {}).update(cell["initial_state_overrides"])
        cell_raw["initial_state"]["curriculum_enabled"] = False
        cell_raw.setdefault("terrain", {}).update(cell["terrain_overrides"])
        temporary = ROOT / "outputs/runs/exp160_dstc_site_commitment/_suite/tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        config_path = temporary / f"h0_{cell['cell']}.json"
        # cfg_from_experiment ignores the audit-only top-level section.
        config_path.write_text(json.dumps(cell_raw, indent=2), encoding="utf-8")
        cfg = cfg_from_experiment(config_path)
        cfg.simulation.device = "cpu"
        core = MultiRoverGatheringCore(cfg)
        proposals = extract_local_site_proposals(
            core.positions,
            core.yaws,
            cfg.terrain,
            core.terrain_runtime,
            epoch=int(settings["epoch"]),
            max_candidates_per_rover=int(settings["max_candidates_per_rover"]),
            verification_radius_m=float(settings["verification_radius_m"]),
            required_radius_m=float(settings["required_flat_radius_m"]),
            flatness_rings=int(settings["flatness_rings"]),
            flatness_samples_per_ring=int(settings["flatness_samples_per_ring"]),
            required_flatness_rings=int(cfg.gather_point.flatness_rings),
            required_flatness_samples_per_ring=int(
                cfg.gather_point.flatness_samples_per_ring
            ),
            max_height_range_m=float(cfg.gather_point.max_height_range),
            max_slope=float(cfg.gather_point.max_slope),
            nms_distance_m=float(settings["nms_distance_m"]),
            id_quantization_m=float(settings["id_quantization_m"]),
        )
        clique, connected = _communication_reachability(
            core.positions, float(cfg.observation.communication_radius)
        )

        selected: list[SiteCertificate | None] = []
        central_points = torch.zeros(core.num_envs, 2, dtype=core.positions.dtype)
        rover_candidate_count = 0
        any_candidate_count = 0
        support1 = support2 = support4 = 0
        containment_count = permutation_count = se2_count = 0
        protocol_complete_count = protocol_adverse_count = 0
        cell_max_se2 = 0.0
        for environment, groups in enumerate(proposals):
            rover_candidate_count += sum(bool(group) for group in groups)
            any_candidate_count += int(any(groups))
            rover_positions = [
                tuple(float(value) for value in point[:2])
                for point in core.positions[environment]
            ]
            certificates = {}
            for support in (1, 2, 4):
                certificates[support] = _selected_certificate(
                    groups,
                    rover_positions,
                    epoch=int(settings["epoch"]),
                    pose_uncertainty_m=float(settings["pose_uncertainty_m"]),
                    min_support=support,
                    distance_weight=float(settings["distance_score_weight"]),
                    support_weight=float(settings["support_score_weight"]),
                )
            support1 += int(certificates[1] is not None)
            support2 += int(certificates[2] is not None)
            support4 += int(certificates[4] is not None)
            certificate = certificates[int(settings["minimum_support_for_commit"])]
            selected.append(certificate)
            if certificate is None or not bool(connected[environment]):
                continue
            central_points[environment] = torch.tensor(certificate.center_xy)
            containment_count += int(
                _support_containment(
                    groups, certificate, float(settings["pose_uncertainty_m"])
                )
            )

            permuted = [list(reversed(group)) for group in groups]
            permuted_certificate = _selected_certificate(
                permuted,
                rover_positions,
                epoch=int(settings["epoch"]),
                pose_uncertainty_m=float(settings["pose_uncertainty_m"]),
                min_support=int(settings["minimum_support_for_commit"]),
                distance_weight=float(settings["distance_score_weight"]),
                support_weight=float(settings["support_score_weight"]),
            )
            permutation_count += int(
                permuted_certificate is not None
                and permuted_certificate.site_id == certificate.site_id
                and math.dist(permuted_certificate.center_xy, certificate.center_xy) <= 1.0e-7
            )

            angle = float(settings["se2_rotation_rad"])
            translation = tuple(float(value) for value in settings["se2_translation_xy"])
            transformed_groups = [
                [
                    transform_proposal(
                        proposal,
                        rotation_rad=angle,
                        translation_xy=translation,
                    )
                    for proposal in group
                ]
                for group in groups
            ]
            transformed_positions = transform_points_se2(
                rover_positions,
                rotation_rad=angle,
                translation_xy=translation,
            )
            transformed_certificate = _selected_certificate(
                transformed_groups,
                transformed_positions,
                epoch=int(settings["epoch"]),
                pose_uncertainty_m=float(settings["pose_uncertainty_m"]),
                min_support=int(settings["minimum_support_for_commit"]),
                distance_weight=float(settings["distance_score_weight"]),
                support_weight=float(settings["support_score_weight"]),
            )
            expected_center = transform_points_se2(
                [certificate.center_xy],
                rotation_rad=angle,
                translation_xy=translation,
            )[0]
            error = (
                math.dist(transformed_certificate.center_xy, expected_center)
                if transformed_certificate is not None
                else float("inf")
            )
            cell_max_se2 = max(cell_max_se2, error)
            se2_count += int(
                transformed_certificate is not None
                and transformed_certificate.site_id == certificate.site_id
                and error <= 1.0e-5
            )
            protocol = _protocol_checks(
                groups,
                certificate,
                seed=int(cell["seed"]) + environment,
            )
            protocol_complete_count += int(
                protocol["full_view"]
                and protocol["same_digest"]
                and protocol["complete_agreement"]
            )
            protocol_adverse_count += int(
                protocol["loss_fail_closed"] and protocol["conflict_fail_closed"]
            )

        selected_mask = torch.tensor(
            [item is not None and bool(connected[index]) for index, item in enumerate(selected)],
            dtype=torch.bool,
        )
        flatness = evaluate_gather_point_flatness(
            central_points,
            cfg.terrain,
            core.terrain_runtime,
            radius=float(settings["required_flat_radius_m"]),
            rings=int(cfg.gather_point.flatness_rings),
            samples_per_ring=int(cfg.gather_point.flatness_samples_per_ring),
            max_height_range=float(cfg.gather_point.max_height_range),
            max_slope=float(cfg.gather_point.max_slope),
        )
        central_flat_count = int((flatness.is_flat & selected_mask).sum())
        committed_count = int(selected_mask.sum())
        support_sum = sum(
            len(item.source_ids)
            for index, item in enumerate(selected)
            if item is not None and bool(connected[index])
        )
        report = {
            "cell": cell["cell"],
            "episodes": int(core.num_envs),
            "rover_candidate_rate": rover_candidate_count / float(core.num_envs * core.n_agents),
            "any_candidate_rate": any_candidate_count / float(core.num_envs),
            "communication_clique_rate": float(clique.float().mean()),
            "communication_connected_rate": float(connected.float().mean()),
            "certificate_rate_support1": support1 / float(core.num_envs),
            "certificate_rate_support2": support2 / float(core.num_envs),
            "certificate_rate_support4": support4 / float(core.num_envs),
            "committed_certificate_rate": committed_count / float(core.num_envs),
            "selected_support_mean": support_sum / float(max(committed_count, 1)),
            "central_flatness_pass_rate": central_flat_count / float(max(committed_count, 1)),
            "support_containment_pass_rate": containment_count / float(max(committed_count, 1)),
            "candidate_permutation_pass_rate": permutation_count / float(max(committed_count, 1)),
            "se2_pass_rate": se2_count / float(max(committed_count, 1)),
            "se2_max_error": cell_max_se2,
            "protocol_complete_delivery_pass_rate": protocol_complete_count / float(max(committed_count, 1)),
            "protocol_adverse_no_split_brain_rate": protocol_adverse_count / float(max(committed_count, 1)),
        }
        all_cell_reports.append(report)
        max_se2_error = max(max_se2_error, cell_max_se2)
        aggregate_counts["episodes"] += int(core.num_envs)
        aggregate_counts["rovers"] += int(core.num_envs * core.n_agents)
        aggregate_counts["rovers_with_candidate"] += rover_candidate_count
        aggregate_counts["episodes_with_any_candidate"] += any_candidate_count
        aggregate_counts["certificate_support1"] += support1
        aggregate_counts["certificate_support2"] += support2
        aggregate_counts["certificate_support4"] += support4
        aggregate_counts["selected_support_sum"] += support_sum
        aggregate_counts["selected_support_count"] += committed_count
        aggregate_counts["central_flatness_pass"] += central_flat_count
        aggregate_counts["containment_pass"] += containment_count
        aggregate_counts["permutation_pass"] += permutation_count
        aggregate_counts["se2_pass"] += se2_count
        aggregate_counts["protocol_complete_pass"] += protocol_complete_count
        aggregate_counts["protocol_adverse_pass"] += protocol_adverse_count
        aggregate_counts["connected"] += int(connected.sum())
        aggregate_counts["clique"] += int(clique.sum())

    episodes = aggregate_counts["episodes"]
    committed = aggregate_counts["selected_support_count"]
    aggregate = {
        "episodes": episodes,
        "rover_candidate_rate": aggregate_counts["rovers_with_candidate"] / float(aggregate_counts["rovers"]),
        "any_candidate_rate": aggregate_counts["episodes_with_any_candidate"] / float(episodes),
        "communication_clique_rate": aggregate_counts["clique"] / float(episodes),
        "communication_connected_rate": aggregate_counts["connected"] / float(episodes),
        "certificate_rate_support1": aggregate_counts["certificate_support1"] / float(episodes),
        "certificate_rate_support2": aggregate_counts["certificate_support2"] / float(episodes),
        "certificate_rate_support4": aggregate_counts["certificate_support4"] / float(episodes),
        "committed_certificate_rate": committed / float(episodes),
        "selected_support_mean": aggregate_counts["selected_support_sum"] / float(max(committed, 1)),
        "central_flatness_pass_rate": aggregate_counts["central_flatness_pass"] / float(max(committed, 1)),
        "support_containment_pass_rate": aggregate_counts["containment_pass"] / float(max(committed, 1)),
        "candidate_permutation_pass_rate": aggregate_counts["permutation_pass"] / float(max(committed, 1)),
        "se2_pass_rate": aggregate_counts["se2_pass"] / float(max(committed, 1)),
        "se2_max_error": max_se2_error,
        "protocol_complete_delivery_pass_rate": aggregate_counts["protocol_complete_pass"] / float(max(committed, 1)),
        "protocol_adverse_no_split_brain_rate": aggregate_counts["protocol_adverse_pass"] / float(max(committed, 1)),
    }

    spacing = max(
        float(cfg.safety.collision_distance) + float(settings["terminal_clearance_m"]),
        float(cfg.success_thresholds.min_pairwise_distance),
    )
    formation_circumradius = spacing / math.sqrt(2.0)
    formation_dmax = spacing * math.sqrt(2.0)
    formation_dispersion = 0.5 * spacing**2
    capacity_ok = (
        formation_circumradius <= float(settings["required_flat_radius_m"])
        and formation_dmax <= float(cfg.success_thresholds.dmax)
        and formation_dispersion <= float(cfg.success_thresholds.dispersion)
    )
    checks = {
        "nonzero_certificate_coverage": aggregate["committed_certificate_rate"] > 0.0,
        "central_flatness_no_false_certificate": aggregate["central_flatness_pass_rate"] == 1.0,
        "pose_uncertainty_is_conservative": aggregate["support_containment_pass_rate"] == 1.0,
        "candidate_order_invariant": aggregate["candidate_permutation_pass_rate"] == 1.0,
        "se2_equivariant": aggregate["se2_pass_rate"] == 1.0 and max_se2_error <= 1.0e-5,
        "complete_delivery_commits_one_site": aggregate["protocol_complete_delivery_pass_rate"] == 1.0,
        "loss_and_conflict_fail_closed": aggregate["protocol_adverse_no_split_brain_rate"] == 1.0,
        "terminal_geometry_has_capacity": capacity_ok,
    }
    report = {
        "experiment": "exp160_dstc_site_commitment",
        "phase": "H0_certificate_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "teacher_actions_generated": False,
        "actor_or_control_modified": False,
        "oracle_used_for_candidate_generation": False,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "config": args.config,
        "site_commitment": settings,
        "cell_reports": all_cell_reports,
        "aggregate": aggregate,
        "terminal_capacity": {
            "center_spacing_m": spacing,
            "circumradius_m": formation_circumradius,
            "dmax_m": formation_dmax,
            "dispersion_m2": formation_dispersion,
        },
        "checks": checks,
        "h0_certificate_core_passed": all(checks.values()),
        "online_actor_integration_ready": False,
        "online_actor_integration_blocker": (
            "The frozen H0 audit validates static local proposals, conservative association "
            "and commit safety only. Online candidate memory, bounded flooding over the "
            "existing tiered cache, commit dwell/abort timing and certificate-derived site "
            "potential have not yet been integrated into environment steps."
        ),
    }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    run_dir = output.parents[1]
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp160_dstc_site_commitment",
                "run_id": "h0_certificate_audit",
                "lifecycle_status": (
                    "completed_diagnostic" if report["h0_certificate_core_passed"] else "failed_gate"
                ),
                "eligible_for_training": False,
                "artifacts": {
                    "h0_certificate_audit": str(output.relative_to(ROOT)),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not report["h0_certificate_core_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
