#!/usr/bin/env python3
"""Training-free Active-DSTC H0.5 discovery and exchange audit."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import (
    SPATIOTEMPORAL_ENDPOINTS,
    decode_action,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from lunar_rover_tasks.tasks.multi_rover_gathering.simple_controller import compute_control
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
    evaluate_gather_point_flatness,
    sample_trajectory_terrain_risk,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import generate_trajectory
from lunar_rover_tasks.utils.math_utils import wrap_to_pi


ACTIVE_SCAN_OFFSETS = ((0.0, 0.0), *SPATIOTEMPORAL_ENDPOINTS)


def _adjacency(positions: torch.Tensor, radius_m: float) -> torch.Tensor:
    return torch.cdist(positions[..., :2], positions[..., :2]) <= float(radius_m)


def _exploration_actions(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    active: torch.Tensor,
    initial_team_center: torch.Tensor,
    anchor_yaw: torch.Tensor,
    *,
    step: int,
    cycle_steps: int,
    spin_steps: int,
    collision_avoidance_distance_m: float,
    world_limit_m: float,
    boundary_margin_m: float,
    frontier_segment_steps: int,
    frontier_radius_m: float,
    frontier_yaw_tolerance_rad: float,
) -> torch.Tensor:
    envs, agents = positions.shape[:2]
    agent_ids = torch.arange(agents, device=positions.device)[None].expand(envs, -1)
    del cycle_steps, spin_steps
    segment = step // int(frontier_segment_steps)
    # Agent 0's reset heading and the initial relative-pose centroid define a
    # shared odometric frame. They are legitimate locally exchanged quantities,
    # not a terrain goal or world-axis Oracle. Adjacent segments rotate the four
    # disjoint rays by 45 degrees to cover eight boundary sectors in 96 s.
    angle = (
        anchor_yaw[:, None]
        + agent_ids.to(dtype=yaws.dtype) * (2.0 * torch.pi / float(agents))
        + float(segment) * (torch.pi / 4.0)
    )
    target = initial_team_center[:, None, :] + float(frontier_radius_m) * torch.stack(
        (torch.cos(angle), torch.sin(angle)), dim=-1
    )
    desired_yaw = torch.atan2(
        target[..., 1] - positions[..., 1],
        target[..., 0] - positions[..., 0],
    )
    target_error = wrap_to_pi(desired_yaw - yaws)
    actions = torch.full((envs, agents), 32, dtype=torch.long, device=positions.device)
    turning = target_error.abs() > float(frontier_yaw_tolerance_rad)
    actions = torch.where(
        turning,
        torch.where(target_error >= 0.0, torch.full_like(actions, 43), torch.full_like(actions, 44)),
        actions,
    )

    # Boundary response uses only the rover's own odometry and the known proxy
    # operational boundary; it does not query a team centroid or Oracle site.
    boundary = positions[..., :2].abs().amax(dim=-1) >= float(world_limit_m - boundary_margin_m)
    inward = torch.atan2(-positions[..., 1], -positions[..., 0])
    yaw_error = wrap_to_pi(inward - yaws)
    boundary_spin = torch.where(yaw_error >= 0.0, torch.full_like(actions, 43), torch.full_like(actions, 44))
    actions = torch.where(boundary, boundary_spin, actions)

    pairwise = torch.cdist(positions[..., :2], positions[..., :2])
    eye = torch.eye(agents, dtype=torch.bool, device=positions.device)[None]
    nearest_distance, nearest_agent = pairwise.masked_fill(eye, float("inf")).min(dim=-1)
    close = nearest_distance < float(collision_avoidance_distance_m)
    lower_priority = agent_ids > nearest_agent
    escape = torch.where(
        lower_priority,
        torch.full_like(actions, 41),
        torch.where(agent_ids.remainder(2) == 0, torch.full_like(actions, 45), torch.full_like(actions, 46)),
    )
    actions = torch.where(close, escape, actions)
    return torch.where(active[:, None], actions, torch.zeros_like(actions))


def _advance(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    actions: torch.Tensor,
    cfg,
    terrain_runtime,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dt = float(cfg.simulation.planning_dt)
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
    risk = sample_trajectory_terrain_risk(
        trajectory.points,
        cfg.terrain,
        terrain_runtime,
    )["risk_mean"]
    control = compute_control(positions, yaws, trajectory, cfg.low_level_control, dt)
    radius = float(cfg.low_level_control.wheel_radius_m)
    track = float(cfg.low_level_control.track_width_m)
    wheel_limit = float(cfg.low_level_control.max_wheel_speed_radps)
    left = ((control.linear - 0.5 * track * control.angular) / radius).clamp(-wheel_limit, wheel_limit)
    right = ((control.linear + 0.5 * track * control.angular) / radius).clamp(-wheel_limit, wheel_limit)
    linear = 0.5 * radius * (left + right)
    angular = radius * (right - left) / track
    midpoint = wrap_to_pi(yaws + 0.5 * angular * dt)
    next_positions = positions.clone()
    next_positions[..., :2] += torch.stack((torch.cos(midpoint), torch.sin(midpoint)), dim=-1) * linear[..., None] * dt
    next_yaws = wrap_to_pi(yaws + angular * dt)
    return next_positions, next_yaws, risk


def _select_from_cache(cache: SiteBeliefCache, rover_positions) -> SiteCertificate | None:
    groups = cache.proposals_by_source()
    batches = [
        ProposalBatchMessage(0, source, proposal_batch_digest(group))
        for source, group in enumerate(groups)
    ]
    digest = combined_proposal_set_digest(batches)
    witnesses = build_site_witnesses(groups, pose_uncertainty_m=0.10, min_support_rovers=1)
    return select_site_certificate(
        witnesses,
        rover_positions,
        epoch=0,
        proposal_set_digest=digest,
        distance_weight=0.03,
        support_weight=0.02,
    )


def _scan_and_exchange(
    caches: list[list[SiteBeliefCache]],
    proposals,
    adjacency: torch.Tensor,
    positions: torch.Tensor,
    *,
    step: int,
    version: int,
    forwarding_rounds: int,
) -> tuple[list[SiteCertificate | None], int, int]:
    envs, agents = adjacency.shape[:2]
    for environment in range(envs):
        for source in range(agents):
            caches[environment][source].observe_local(
                proposals[environment][source],
                source_version=version,
                observed_step=step,
            )
            caches[environment][source].expire(current_step=step)
    transmitted_records = 0
    changed_records = 0
    for _ in range(int(forwarding_rounds)):
        messages = [
            [caches[environment][sender].message() for sender in range(agents)]
            for environment in range(envs)
        ]
        for environment in range(envs):
            for receiver in range(agents):
                for sender in range(agents):
                    if receiver == sender or not bool(adjacency[environment, receiver, sender]):
                        continue
                    transmitted_records += len(messages[environment][sender].records)
                    changed_records += caches[environment][receiver].merge_message(
                        messages[environment][sender]
                    )
    certificates: list[SiteCertificate | None] = []
    for environment in range(envs):
        rover_positions = [
            tuple(float(value) for value in point[:2]) for point in positions[environment]
        ]
        local = [
            _select_from_cache(caches[environment][agent], rover_positions)
            for agent in range(agents)
        ]
        site_ids = {item.site_id if item is not None else None for item in local}
        certificates.append(local[0] if len(site_ids) == 1 and None not in site_ids else None)
    return certificates, transmitted_records, changed_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp162_active_dstc_h05.yaml")
    parser.add_argument(
        "--manifest",
        default="outputs/runs/exp156_differential_multiscale_ablation/_suite/scenario_manifest.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/runs/exp162_active_dstc_h05/frozen_closed_loop/metrics/active_dstc_h05.json",
    )
    parser.add_argument("--episodes-per-cell", type=int, default=None)
    args = parser.parse_args()
    raw = load_yaml(args.config)
    experiment_id = str(raw.get("experiment", {}).get("name", "exp162_active_dstc_h05"))
    active_cfg = raw["active_dstc"]
    site_cfg = raw["site_commitment"]
    benchmark_repair = raw.get("benchmark_repair", {})
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes_per_cell = int(args.episodes_per_cell or manifest["episodes_per_cell"])
    cell_reports = []

    for cell in manifest["cells"]:
        cell_raw = copy.deepcopy(raw)
        cell_raw.setdefault("experiment", {})["seed"] = int(cell["seed"])
        cell_raw["experiment"]["num_envs"] = episodes_per_cell
        cell_raw.setdefault("initial_state", {}).update(cell["initial_state_overrides"])
        cell_raw["initial_state"]["curriculum_enabled"] = False
        cell_raw.setdefault("terrain", {}).update(cell["terrain_overrides"])
        if "bottleneck" in str(cell["cell"]) and "bottleneck_crater_count" in benchmark_repair:
            cell_raw["terrain"]["crater_count"] = int(
                benchmark_repair["bottleneck_crater_count"]
            )
        temporary = ROOT / "outputs/runs/exp162_active_dstc_h05/_suite/tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        config_path = temporary / f"{cell['cell']}_{episodes_per_cell}.json"
        config_path.write_text(json.dumps(cell_raw, indent=2), encoding="utf-8")
        cfg = cfg_from_experiment(config_path)
        cfg.simulation.device = "cpu"
        core = MultiRoverGatheringCore(cfg)
        positions = core.positions.clone()
        yaws = core.yaws.clone()
        initial_positions = positions.clone()
        initial_yaws = yaws.clone()
        initial_team_center = initial_positions[..., :2].mean(dim=1)
        anchor_yaw = initial_yaws[:, 0]
        caches = [
            [
                SiteBeliefCache(
                    agent_id=agent,
                    n_agents=core.n_agents,
                    max_entries_per_source=int(active_cfg["max_entries_per_source"]),
                    ttl_steps=int(active_cfg["belief_ttl_steps"]),
                )
                for agent in range(core.n_agents)
            ]
            for _ in range(core.num_envs)
        ]
        committed_step = torch.full((core.num_envs,), -1, dtype=torch.long)
        committed_centers = torch.zeros(core.num_envs, 2)
        collision = torch.zeros(core.num_envs, dtype=torch.bool)
        out_of_bounds = torch.zeros(core.num_envs, dtype=torch.bool)
        risk_sum = torch.zeros(core.num_envs)
        risk_steps = torch.zeros(core.num_envs)
        transmitted_records = 0
        changed_records = 0
        scan_count = 0
        travel_distance = torch.zeros(core.num_envs, core.n_agents)
        for step in range(int(active_cfg["episode_steps"])):
            active = (committed_step < 0) & ~collision & ~out_of_bounds
            if step % int(active_cfg["scan_interval_steps"]) == 0:
                proposals = extract_local_site_proposals(
                    positions,
                    yaws,
                    cfg.terrain,
                    core.terrain_runtime,
                    epoch=0,
                    max_candidates_per_rover=int(site_cfg["max_candidates_per_rover"]),
                    verification_radius_m=float(site_cfg["verification_radius_m"]),
                    required_radius_m=float(site_cfg["required_flat_radius_m"]),
                    flatness_rings=int(site_cfg["flatness_rings"]),
                    flatness_samples_per_ring=int(site_cfg["flatness_samples_per_ring"]),
                    required_flatness_rings=int(cfg.gather_point.flatness_rings),
                    required_flatness_samples_per_ring=int(cfg.gather_point.flatness_samples_per_ring),
                    max_height_range_m=float(cfg.gather_point.max_height_range),
                    max_slope=float(cfg.gather_point.max_slope),
                    nms_distance_m=float(site_cfg["nms_distance_m"]),
                    id_quantization_m=float(site_cfg["id_quantization_m"]),
                    source_frame_positions=initial_positions,
                    source_frame_yaws=initial_yaws,
                    candidate_offsets_body=ACTIVE_SCAN_OFFSETS,
                )
                certificates, sent, changed = _scan_and_exchange(
                    caches,
                    proposals,
                    _adjacency(positions, float(cfg.observation.communication_radius)),
                    positions,
                    step=step,
                    version=scan_count,
                    forwarding_rounds=int(active_cfg["forwarding_rounds"]),
                )
                transmitted_records += sent
                changed_records += changed
                for environment, certificate in enumerate(certificates):
                    if certificate is not None and committed_step[environment] < 0:
                        committed_step[environment] = step
                        committed_centers[environment] = torch.tensor(certificate.center_xy)
                scan_count += 1
                active = (committed_step < 0) & ~collision & ~out_of_bounds
            actions = _exploration_actions(
                positions,
                yaws,
                active,
                initial_team_center,
                anchor_yaw,
                step=step,
                cycle_steps=int(active_cfg["exploration_cycle_steps"]),
                spin_steps=int(active_cfg["exploration_spin_steps"]),
                collision_avoidance_distance_m=float(active_cfg["collision_avoidance_distance_m"]),
                world_limit_m=float(cfg.safety.world_xy_limit),
                boundary_margin_m=float(active_cfg["boundary_margin_m"]),
                frontier_segment_steps=int(active_cfg["frontier_segment_steps"]),
                frontier_radius_m=float(active_cfg["frontier_radius_m"]),
                frontier_yaw_tolerance_rad=float(active_cfg["frontier_yaw_tolerance_rad"]),
            )
            previous_xy = positions[..., :2].clone()
            positions, yaws, risk = _advance(
                positions, yaws, actions, cfg, core.terrain_runtime
            )
            travel_distance += torch.linalg.vector_norm(
                positions[..., :2] - previous_xy, dim=-1
            ) * active[:, None].float()
            risk_sum += risk.mean(dim=1) * active.float()
            risk_steps += active.float()
            distances = torch.cdist(positions[..., :2], positions[..., :2])
            eye = torch.eye(core.n_agents, dtype=torch.bool)[None]
            collision |= (
                distances.masked_fill(eye, float("inf")).amin(dim=(1, 2))
                < float(cfg.safety.collision_distance)
            ) & active
            out_of_bounds |= (
                positions[..., :2].abs().amax(dim=(1, 2)) > float(cfg.safety.world_xy_limit)
            ) & active

        committed = committed_step >= 0
        flatness = evaluate_gather_point_flatness(
            committed_centers,
            cfg.terrain,
            core.terrain_runtime,
            radius=float(site_cfg["required_flat_radius_m"]),
            rings=int(cfg.gather_point.flatness_rings),
            samples_per_ring=int(cfg.gather_point.flatness_samples_per_ring),
            max_height_range=float(cfg.gather_point.max_height_range),
            max_slope=float(cfg.gather_point.max_slope),
        )
        false_certificate = committed & ~flatness.is_flat
        report = {
            "cell": cell["cell"],
            "episodes": core.num_envs,
            "certificate_rate": float(committed.float().mean()),
            "false_certificate_rate": float(false_certificate.float().mean()),
            "collision_rate": float(collision.float().mean()),
            "out_of_bounds_rate": float(out_of_bounds.float().mean()),
            "timeout_rate": float((~committed & ~collision & ~out_of_bounds).float().mean()),
            "mean_certificate_step": float(committed_step[committed].float().mean()) if committed.any() else None,
            "mean_path_risk": float((risk_sum / risk_steps.clamp_min(1.0)).mean()),
            "mean_rover_travel_distance_m": float(travel_distance.mean()),
            "mean_final_displacement_m": float(
                torch.linalg.vector_norm(
                    positions[..., :2] - initial_positions[..., :2], dim=-1
                ).mean()
            ),
            "mean_transmitted_records_per_episode": transmitted_records / float(core.num_envs),
            "mean_changed_records_per_episode": changed_records / float(core.num_envs),
            "scan_count": scan_count,
        }
        cell_reports.append(report)

    checks = {
        "all_cells_certificate_ge_required": all(
            item["certificate_rate"] >= float(active_cfg["required_certificate_rate"])
            for item in cell_reports
        ),
        "false_certificate_zero": all(item["false_certificate_rate"] == 0.0 for item in cell_reports),
        "all_cells_collision_le_limit": all(
            item["collision_rate"] <= float(active_cfg["maximum_collision_rate"])
            for item in cell_reports
        ),
        "all_cells_timeout_below_limit": all(
            item["timeout_rate"] < float(active_cfg["maximum_timeout_rate"])
            for item in cell_reports
        ),
    }
    report = {
        "experiment": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "teacher_actions_generated": False,
        "oracle_used": False,
        "benchmark_repair": benchmark_repair,
        "episodes_per_cell": episodes_per_cell,
        "cell_reports": cell_reports,
        "checks": checks,
        "active_dstc_h05_passed": all(checks.values()),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    run_dir = output.parents[1]
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "run_id": "frozen_closed_loop",
                "lifecycle_status": "completed_diagnostic" if report["active_dstc_h05_passed"] else "failed_gate",
                "eligible_for_training": False,
                "artifacts": {"active_dstc_h05": str(output.relative_to(ROOT))},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not report["active_dstc_h05_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
