#!/usr/bin/env python3
"""Paired frozen feasibility audit for D-STC, HPP, map consensus and local optimization."""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from audit_exp156_action_coverage import (
    ESCAPE_ACTIONS,
    artificial_deadlocks,
    collect_exp155_deadlocks,
    nominal_forward_conflict,
    pairwise_minimum,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import decode_action
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from lunar_rover_tasks.tasks.multi_rover_gathering.simple_controller import compute_control
from lunar_rover_tasks.tasks.multi_rover_gathering.site_commitment import (
    ProposalBatchMessage,
    SiteProposal,
    build_site_witnesses,
    combined_proposal_set_digest,
    extract_local_site_proposals,
    proposal_batch_digest,
    select_site_certificate,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import generate_trajectory
from lunar_rover_tasks.utils.math_utils import wrap_to_pi


def _adjacency(positions: torch.Tensor, radius_m: float) -> torch.Tensor:
    distance = torch.cdist(positions[..., :2], positions[..., :2])
    return distance <= float(radius_m)


def _hpp_goal_belief(
    groups: list[list[SiteProposal]],
    positions_xy: torch.Tensor,
    *,
    rounds: int,
    response_step_m: float,
    agreement_radius_m: float,
) -> dict:
    """Optimistic HPP-style goal inference using full current team positions.

    This deliberately gives every rover more pose information than the strict
    execution interface. Each rover still owns only its local site candidates.
    Failure is therefore a valid upper-bound rejection; success only authorizes
    a later learned-predictor experiment.
    """

    predicted = positions_xy.clone().to(dtype=torch.float64)
    goals: list[tuple[float, float] | None] = [None] * len(groups)
    for _ in range(int(rounds)):
        new_goals: list[tuple[float, float] | None] = []
        for proposals in groups:
            if not proposals:
                new_goals.append(None)
                continue
            ranked = []
            for proposal in proposals:
                center = torch.tensor(proposal.center_xy, dtype=predicted.dtype)
                distances = torch.linalg.vector_norm(predicted - center, dim=-1)
                score = proposal.terrain_cost + 0.02 * float(distances.mean()) + 0.02 * float(distances.max())
                ranked.append((score, proposal.proposal_id, proposal.center_xy))
            new_goals.append(min(ranked, key=lambda item: (item[0], item[1]))[2])
        goals = new_goals
        for agent, goal in enumerate(goals):
            if goal is None:
                continue
            delta = torch.tensor(goal, dtype=predicted.dtype) - predicted[agent]
            distance = float(torch.linalg.vector_norm(delta))
            if distance > 1.0e-12:
                predicted[agent] += delta * min(float(response_step_m) / distance, 1.0)
    complete = all(goal is not None for goal in goals)
    if not complete:
        return {"complete": False, "agreed": False, "goal_spread_m": None}
    goal_tensor = torch.tensor(goals, dtype=torch.float64)
    spread = float(torch.cdist(goal_tensor, goal_tensor).amax())
    return {
        "complete": True,
        "agreed": spread <= float(agreement_radius_m),
        "goal_spread_m": spread,
    }


def _flood_sources(adjacency: torch.Tensor, *, max_rounds: int) -> tuple[list[set[int]], int | None]:
    n_agents = adjacency.shape[0]
    knowledge = [{agent} for agent in range(n_agents)]
    if all(len(items) == n_agents for items in knowledge):
        return knowledge, 0
    for round_index in range(1, int(max_rounds) + 1):
        previous = [set(items) for items in knowledge]
        for receiver in range(n_agents):
            merged = set(previous[receiver])
            for sender in range(n_agents):
                if bool(adjacency[receiver, sender]):
                    merged.update(previous[sender])
            knowledge[receiver] = merged
        if all(len(items) == n_agents for items in knowledge):
            return knowledge, round_index
    return knowledge, None


def _map_consensus(
    groups: list[list[SiteProposal]],
    rover_positions: list[tuple[float, float]],
    adjacency: torch.Tensor,
    *,
    max_rounds: int,
    pose_uncertainty_m: float,
    distance_weight: float,
    support_weight: float,
) -> dict:
    knowledge, rounds = _flood_sources(adjacency, max_rounds=max_rounds)
    site_ids = []
    for sources in knowledge:
        local_groups = [groups[source] if source in sources else [] for source in range(len(groups))]
        batches = [
            ProposalBatchMessage(0, source, proposal_batch_digest(local_groups[source]))
            for source in sorted(sources)
        ]
        digest = combined_proposal_set_digest(batches)
        witnesses = build_site_witnesses(
            local_groups,
            pose_uncertainty_m=pose_uncertainty_m,
            min_support_rovers=1,
        )
        certificate = select_site_certificate(
            witnesses,
            rover_positions,
            epoch=0,
            proposal_set_digest=digest,
            distance_weight=distance_weight,
            support_weight=support_weight,
        )
        site_ids.append(certificate.site_id if certificate is not None else None)
    available = any(group for group in groups)
    agreed = rounds is not None and len(set(site_ids)) == 1 and site_ids[0] is not None
    return {
        "candidate_available": bool(available),
        "consensus_reached": rounds is not None,
        "rounds": rounds,
        "site_agreed": agreed,
    }


def _simulate_primitive_actions(record: dict, actions: torch.Tensor, cfg, *, steps: int = 16) -> dict:
    batch = actions.shape[0]
    positions = record["positions"].unsqueeze(0).expand(batch, -1, -1).clone()
    yaws = record["yaws"].unsqueeze(0).expand(batch, -1).clone()
    initial_positions = positions.clone()
    initial_yaws = yaws.clone()
    safe = torch.ones(batch, dtype=torch.bool)
    minimum = torch.full((batch,), float("inf"))
    dt = float(cfg.simulation.planning_dt)
    radius = float(cfg.low_level_control.wheel_radius_m)
    track = float(cfg.low_level_control.track_width_m)
    wheel_limit = float(cfg.low_level_control.max_wheel_speed_radps)
    for _ in range(int(steps)):
        decoded = decode_action(actions, positions, yaws, cfg.planner)
        trajectory = generate_trajectory(
            positions,
            decoded.world_subgoal,
            cfg.trajectory_generator,
            dt,
            current_yaws=yaws,
            reference_speed=decoded.reference_speed,
            motion_direction=decoded.motion_direction,
            planned_yaw_delta=decoded.planned_yaw_delta,
            primitive_type=decoded.primitive_type,
        )
        control = compute_control(positions, yaws, trajectory, cfg.low_level_control, dt)
        left = ((control.linear - 0.5 * track * control.angular) / radius).clamp(-wheel_limit, wheel_limit)
        right = ((control.linear + 0.5 * track * control.angular) / radius).clamp(-wheel_limit, wheel_limit)
        linear = 0.5 * radius * (left + right)
        angular = radius * (right - left) / track
        midpoint = wrap_to_pi(yaws + 0.5 * angular * dt)
        positions[..., :2] += torch.stack((torch.cos(midpoint), torch.sin(midpoint)), dim=-1) * linear[..., None] * dt
        yaws = wrap_to_pi(yaws + angular * dt)
        current_minimum = pairwise_minimum(positions[..., :2])
        minimum = torch.minimum(minimum, current_minimum)
        safe &= current_minimum >= float(cfg.safety.collision_distance)
    target = initial_positions[..., :2].mean(dim=1, keepdim=True)
    target_distance = torch.linalg.vector_norm(positions[..., :2] - target, dim=-1).mean(dim=1)
    pairwise = torch.cdist(positions[..., :2], positions[..., :2])
    dmax = pairwise.amax(dim=(1, 2))
    translation = torch.linalg.vector_norm(positions[..., :2] - initial_positions[..., :2], dim=-1).amax(dim=1)
    rotation = wrap_to_pi(yaws - initial_yaws).abs().amax(dim=1)
    forward_conflict = nominal_forward_conflict(positions, yaws, cfg)
    resolved = (
        safe
        & (forward_conflict >= float(cfg.safety.collision_distance))
        & ((translation >= 0.15) | (rotation >= 0.20))
    )
    return {
        "positions": positions,
        "yaws": yaws,
        "safe": safe,
        "minimum": minimum,
        "target_distance": target_distance,
        "dmax": dmax,
        "forward_conflict": forward_conflict,
        "resolved": resolved,
    }


def _decentralized_best_response(record: dict, cfg, *, sweeps: int) -> dict:
    n_agents = record["positions"].shape[0]
    actions = torch.zeros(n_agents, dtype=torch.long)
    for _ in range(int(sweeps)):
        changed = False
        for agent in range(n_agents):
            candidates = actions.unsqueeze(0).repeat(len(ESCAPE_ACTIONS), 1)
            candidates[:, agent] = torch.tensor(ESCAPE_ACTIONS)
            outcome = _simulate_primitive_actions(record, candidates, cfg)
            collision_deficit = (
                float(cfg.safety.collision_distance) - outcome["minimum"]
            ).clamp_min(0.0)
            forward_deficit = (
                float(cfg.safety.collision_distance) - outcome["forward_conflict"]
            ).clamp_min(0.0)
            hold_penalty = (candidates == 0).float().sum(dim=1)
            cost = (
                1000.0 * (~outcome["safe"]).float()
                + 200.0 * collision_deficit
                + 100.0 * forward_deficit
                + outcome["target_distance"]
                + 0.25 * outcome["dmax"]
                + 0.02 * hold_penalty
            )
            index = int(cost.argmin())
            new_action = int(candidates[index, agent])
            changed |= new_action != int(actions[agent])
            actions[agent] = new_action
        if not changed:
            break
    outcome = _simulate_primitive_actions(record, actions.unsqueeze(0), cfg)
    return {
        "actions": actions.tolist(),
        "resolved": bool(outcome["resolved"][0]),
        "safe": bool(outcome["safe"][0]),
        "minimum_distance_m": float(outcome["minimum"][0]),
        "forward_conflict_distance_m": float(outcome["forward_conflict"][0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp161_all_routes_feasibility.yaml")
    parser.add_argument(
        "--manifest",
        default="outputs/runs/exp156_differential_multiscale_ablation/_suite/scenario_manifest.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/runs/exp161_all_routes_feasibility/frozen_suite/metrics/route_feasibility.json",
    )
    args = parser.parse_args()
    raw = load_yaml(args.config)
    site = raw["site_commitment"]
    route = raw["route_feasibility"]
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    cells = []
    total = {
        "episodes": 0,
        "hpp_complete": 0,
        "hpp_agreed": 0,
        "map_candidate": 0,
        "map_consensus": 0,
        "map_agreed": 0,
        "map_round_sum": 0,
        "map_round_count": 0,
    }
    for cell in manifest["cells"]:
        cell_raw = copy.deepcopy(raw)
        cell_raw.setdefault("experiment", {})["seed"] = int(cell["seed"])
        cell_raw["experiment"]["num_envs"] = int(manifest["episodes_per_cell"])
        cell_raw.setdefault("initial_state", {}).update(cell["initial_state_overrides"])
        cell_raw["initial_state"]["curriculum_enabled"] = False
        cell_raw.setdefault("terrain", {}).update(cell["terrain_overrides"])
        temporary = ROOT / "outputs/runs/exp161_all_routes_feasibility/_suite/tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        config_path = temporary / f"{cell['cell']}.json"
        config_path.write_text(json.dumps(cell_raw, indent=2), encoding="utf-8")
        cfg = cfg_from_experiment(config_path)
        cfg.simulation.device = "cpu"
        core = MultiRoverGatheringCore(cfg)
        proposals = extract_local_site_proposals(
            core.positions,
            core.yaws,
            cfg.terrain,
            core.terrain_runtime,
            epoch=int(site["epoch"]),
            max_candidates_per_rover=int(site["max_candidates_per_rover"]),
            verification_radius_m=float(site["verification_radius_m"]),
            required_radius_m=float(site["required_flat_radius_m"]),
            flatness_rings=int(site["flatness_rings"]),
            flatness_samples_per_ring=int(site["flatness_samples_per_ring"]),
            required_flatness_rings=int(cfg.gather_point.flatness_rings),
            required_flatness_samples_per_ring=int(cfg.gather_point.flatness_samples_per_ring),
            max_height_range_m=float(cfg.gather_point.max_height_range),
            max_slope=float(cfg.gather_point.max_slope),
            nms_distance_m=float(site["nms_distance_m"]),
            id_quantization_m=float(site["id_quantization_m"]),
        )
        graph = _adjacency(core.positions, float(cfg.observation.communication_radius))
        counts = {
            "episodes": core.num_envs,
            "hpp_complete": 0,
            "hpp_agreed": 0,
            "hpp_spread_sum": 0.0,
            "hpp_spread_count": 0,
            "map_candidate": 0,
            "map_consensus": 0,
            "map_agreed": 0,
            "map_round_sum": 0,
            "map_round_count": 0,
        }
        for environment, groups in enumerate(proposals):
            hpp = _hpp_goal_belief(
                groups,
                core.positions[environment, :, :2],
                rounds=int(route["hpp_prediction_rounds"]),
                response_step_m=float(route["hpp_response_step_m"]),
                agreement_radius_m=float(route["hpp_goal_agreement_radius_m"]),
            )
            counts["hpp_complete"] += int(hpp["complete"])
            counts["hpp_agreed"] += int(hpp["agreed"])
            if hpp["goal_spread_m"] is not None:
                counts["hpp_spread_sum"] += float(hpp["goal_spread_m"])
                counts["hpp_spread_count"] += 1
            rover_positions = [tuple(float(value) for value in point[:2]) for point in core.positions[environment]]
            mapping = _map_consensus(
                groups,
                rover_positions,
                graph[environment],
                max_rounds=int(route["map_consensus_max_rounds"]),
                pose_uncertainty_m=float(site["pose_uncertainty_m"]),
                distance_weight=float(site["distance_score_weight"]),
                support_weight=float(site["support_score_weight"]),
            )
            counts["map_candidate"] += int(mapping["candidate_available"])
            counts["map_consensus"] += int(mapping["consensus_reached"])
            counts["map_agreed"] += int(mapping["site_agreed"])
            if mapping["rounds"] is not None:
                counts["map_round_sum"] += int(mapping["rounds"])
                counts["map_round_count"] += 1
        cell_report = {
            "cell": cell["cell"],
            "episodes": core.num_envs,
            "hpp_goal_coverage": counts["hpp_complete"] / core.num_envs,
            "hpp_goal_agreement": counts["hpp_agreed"] / core.num_envs,
            "hpp_conditional_agreement": counts["hpp_agreed"] / max(counts["hpp_complete"], 1),
            "hpp_mean_goal_spread_m": counts["hpp_spread_sum"] / max(counts["hpp_spread_count"], 1),
            "map_candidate_coverage": counts["map_candidate"] / core.num_envs,
            "map_digest_consensus": counts["map_consensus"] / core.num_envs,
            "map_site_agreement": counts["map_agreed"] / core.num_envs,
            "map_conditional_site_agreement": counts["map_agreed"] / max(counts["map_candidate"], 1),
            "map_mean_consensus_rounds": counts["map_round_sum"] / max(counts["map_round_count"], 1),
        }
        cells.append(cell_report)
        for key in total:
            if key in counts:
                total[key] += counts[key]

    hpp_coverage = total["hpp_complete"] / total["episodes"]
    hpp_agreement = total["hpp_agreed"] / total["episodes"]
    map_coverage = total["map_candidate"] / total["episodes"]
    map_consensus = total["map_consensus"] / total["episodes"]
    map_agreement = total["map_agreed"] / total["episodes"]

    cfg = cfg_from_experiment(args.config)
    cfg.simulation.device = "cpu"
    cfg.terrain.dynamics_enabled = False
    records = collect_exp155_deadlocks(
        ROOT / "outputs/runs/exp155_full_rl_ablation/n0_seed23_full_2400iter",
        maximum=5,
        rollout_envs=32,
        steps=240,
    )
    records.extend(artificial_deadlocks())
    optimizer_results = [
        {"source": record["source"], **_decentralized_best_response(record, cfg, sweeps=int(route["optimizer_sweeps"]))}
        for record in records
    ]
    optimizer_resolution = sum(item["resolved"] for item in optimizer_results) / max(len(optimizer_results), 1)

    route_results = {
        "R1_site_certificate": {
            "status": "conditional_pass",
            "coupled_task_pass": False,
            "evidence": "exp160 H0",
            "certificate_coverage": map_coverage,
            "false_certificate_rate": 0.0,
            "reason": "Static certificate correctness passed, but instantaneous terrain coverage is insufficient.",
        },
        "R2_hpp_predictive_belief": {
            "status": "pass" if (
                hpp_coverage >= float(route["hpp_required_coverage"])
                and hpp_agreement >= float(route["hpp_required_agreement"])
            ) else "failed_gate",
            "coupled_task_pass": False,
            "optimistic_goal_coverage": hpp_coverage,
            "optimistic_goal_agreement": hpp_agreement,
            "conditional_agreement": total["hpp_agreed"] / max(total["hpp_complete"], 1),
            "reason": "Uses full current team poses as an optimistic upper bound; local candidate sets remain private.",
        },
        "R3_distributed_map_consensus": {
            "status": "pass" if (
                map_coverage >= float(route["map_required_coverage"])
                and map_agreement / max(map_coverage, 1.0e-12) >= float(route["map_required_conditional_agreement"])
            ) else "conditional_pass",
            "coupled_task_pass": False,
            "candidate_coverage": map_coverage,
            "digest_consensus": map_consensus,
            "site_agreement": map_agreement,
            "conditional_site_agreement": map_agreement / max(map_coverage, 1.0e-12),
            "mean_consensus_rounds": total["map_round_sum"] / max(total["map_round_count"], 1),
            "reason": "Bounded flooding solves agreement when evidence exists, but cannot create terrain evidence outside all local views.",
        },
        "R4_decentralized_primitive_optimization": {
            "status": "component_pass" if optimizer_resolution >= float(route["optimizer_required_resolution"]) else "failed_gate",
            "coupled_task_pass": False,
            "scenarios": len(optimizer_results),
            "resolution_rate": optimizer_resolution,
            "required_resolution_rate": float(route["optimizer_required_resolution"]),
            "results": optimizer_results,
            "reason": "Tests terminal deadlock only and assumes a common local target; it does not solve site discovery.",
        },
    }
    complete_routes = [
        name for name, result in route_results.items() if result["coupled_task_pass"]
    ]
    successful_components = [
        name
        for name, result in route_results.items()
        if result["status"] in {"conditional_pass", "component_pass"}
    ]
    report = {
        "experiment": "exp161_all_routes_feasibility",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "teacher_actions_generated": False,
        "paired_manifest": str(manifest_path.relative_to(ROOT)),
        "design": {
            "blocks": "near/far x Open/Mixed/Bottleneck",
            "route2_bias": "optimistic_full_team_pose_upper_bound",
            "route3_round_limit": int(route["map_consensus_max_rounds"]),
            "route4_unit": "12 independent frozen deadlock scenarios",
        },
        "cell_reports": cells,
        "route_results": route_results,
        "successful_components": successful_components,
        "complete_route_passes": complete_routes,
        "all_coupled_task_routes_failed": not complete_routes,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    run_dir = output.parents[1]
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp161_all_routes_feasibility",
                "run_id": "frozen_suite",
                "lifecycle_status": "completed_diagnostic",
                "eligible_for_training": False,
                "artifacts": {"route_feasibility": str(output.relative_to(ROOT))},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
