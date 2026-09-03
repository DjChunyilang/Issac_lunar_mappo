#!/usr/bin/env python3
"""Active-DSTC discovery/commit plus decentralized R4 gathering pilot."""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from audit_exp162_active_dstc import (
    ACTIVE_SCAN_OFFSETS,
    _adjacency,
    _advance,
    _exploration_actions,
    _select_from_cache,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.decentralized_primitive_optimizer import (
    select_decentralized_primitives,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.site_commitment import (
    SiteBeliefCache,
    SiteCertificate,
    extract_local_site_proposals,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    evaluate_gather_point_flatness,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.termination import (
    check_collision,
    check_out_of_bounds,
    compute_success_gates,
)


Clock = dict[tuple[int, str], tuple[int, int, int]]


def _cache_signature(cache: SiteBeliefCache) -> tuple:
    return tuple(sorted(cache.event_clock.items()))


def _local_certificates(
    caches: list[list[SiteBeliefCache]], positions: torch.Tensor
) -> list[list[SiteCertificate | None]]:
    result: list[list[SiteCertificate | None]] = []
    for environment in range(len(caches)):
        rover_positions = [
            tuple(float(value) for value in point[:2])
            for point in positions[environment]
        ]
        result.append(
            [
                _select_from_cache(caches[environment][agent], rover_positions)
                for agent in range(len(caches[environment]))
            ]
        )
    return result


def _agreed_certificate(
    local: list[SiteCertificate | None],
) -> SiteCertificate | None:
    if not local or any(item is None for item in local):
        return None
    site_ids = {item.site_id for item in local if item is not None}
    digests = {item.proposal_set_digest for item in local if item is not None}
    if len(site_ids) != 1 or len(digests) != 1:
        return None
    # This is the all-rover commit condition: every local replica signs the
    # same site id under the same proposal-set digest.
    return local[0]


def _exchange_delta(
    caches: list[list[SiteBeliefCache]],
    link_clocks: list[list[list[Clock]]],
    proposals,
    adjacency: torch.Tensor,
    positions: torch.Tensor,
    *,
    step: int,
    version: int,
    forwarding_rounds: int,
) -> tuple[list[SiteCertificate | None], dict[str, int | bool]]:
    envs, agents = adjacency.shape[:2]
    for environment in range(envs):
        for source in range(agents):
            caches[environment][source].observe_local(
                proposals[environment][source],
                source_version=version,
                observed_step=step,
            )
            caches[environment][source].expire(current_step=step)

    # Shadow the exact same local state and use legacy full-cache flooding as
    # a paired semantic oracle.  It is diagnostic only and never selects an
    # action or site in the delta execution path.
    full_caches = copy.deepcopy(caches)
    delta_records = 0
    full_records = 0
    changed_records = 0
    for _ in range(int(forwarding_rounds)):
        for environment in range(envs):
            delta_messages = [
                [
                    caches[environment][sender].delta_message(
                        (
                            link_clocks[environment][sender][receiver]
                            if _cache_signature(caches[environment][sender])
                            == _cache_signature(caches[environment][receiver])
                            else {}
                        )
                    )
                    for receiver in range(agents)
                ]
                for sender in range(agents)
            ]
            full_messages = [
                full_caches[environment][sender].message()
                for sender in range(agents)
            ]
            for receiver in range(agents):
                for sender in range(agents):
                    if receiver == sender or not bool(
                        adjacency[environment, receiver, sender]
                    ):
                        continue
                    delta = delta_messages[sender][receiver]
                    delta_records += len(delta.records) + len(delta.tombstones)
                    full_records += len(full_messages[sender].records)
                    changed_records += caches[environment][receiver].merge_delta(delta)
                    full_caches[environment][receiver].merge_message(
                        full_messages[sender]
                    )
            # ACK only after the synchronous round is fully applied. Earlier
            # per-sender ACKs could become stale when a later sender caused a
            # bounded-cache eviction in the same round.
            for receiver in range(agents):
                receiver_clock = dict(caches[environment][receiver].event_clock)
                for sender in range(agents):
                    if receiver != sender and bool(
                        adjacency[environment, receiver, sender]
                    ):
                        link_clocks[environment][sender][receiver] = receiver_clock

    local_delta = _local_certificates(caches, positions)
    local_full = _local_certificates(full_caches, positions)
    certificates = [_agreed_certificate(local) for local in local_delta]
    semantic_match = True
    for delta_local, full_local in zip(local_delta, local_full):
        delta_signature = [
            None
            if item is None
            else (item.proposal_set_digest, item.site_id)
            for item in delta_local
        ]
        full_signature = [
            None
            if item is None
            else (item.proposal_set_digest, item.site_id)
            for item in full_local
        ]
        semantic_match &= delta_signature == full_signature
    return certificates, {
        "delta_records": delta_records,
        "full_records": full_records,
        "changed_records": changed_records,
        "semantic_match": semantic_match,
    }


def _action_family_counts(actions: torch.Tensor, active: torch.Tensor) -> dict[str, int]:
    selected = actions[active[:, None].expand_as(actions)]
    return {
        "hold": int((selected == 0).sum()),
        "forward": int(((selected >= 1) & (selected <= 39)).sum()),
        "reverse": int(((selected >= 40) & (selected <= 42)).sum()),
        "spin": int(((selected >= 43) & (selected <= 44)).sum()),
        "yield": int(((selected >= 45) & (selected <= 46)).sum()),
    }


def _run_cell(raw: dict, cell: dict, *, episodes: int, device: str) -> dict:
    active_cfg = raw["active_dstc"]
    site_cfg = raw["site_commitment"]
    closed_cfg = raw["closed_loop"]
    benchmark_repair = raw.get("benchmark_repair", {})
    cell_raw = copy.deepcopy(raw)
    cell_raw.setdefault("experiment", {})["seed"] = int(cell["seed"])
    cell_raw["experiment"]["num_envs"] = int(episodes)
    cell_raw.setdefault("initial_state", {}).update(cell["initial_state_overrides"])
    cell_raw["initial_state"]["curriculum_enabled"] = False
    cell_raw.setdefault("terrain", {}).update(cell["terrain_overrides"])
    if "bottleneck" in str(cell["cell"]) and "bottleneck_crater_count" in benchmark_repair:
        cell_raw["terrain"]["crater_count"] = int(
            benchmark_repair["bottleneck_crater_count"]
        )
    temporary = ROOT / "outputs/runs/exp165_active_dstc_closed_loop/_suite/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    config_path = temporary / f"{cell['cell']}_{episodes}_{device}.json"
    config_path.write_text(json.dumps(cell_raw, indent=2), encoding="utf-8")
    cfg = cfg_from_experiment(config_path)
    cfg.simulation.device = device
    core = MultiRoverGatheringCore(cfg)
    positions = core.positions.clone()
    yaws = core.yaws.clone()
    initial_positions = positions.clone()
    initial_yaws = yaws.clone()
    initial_team_center = initial_positions[..., :2].mean(dim=1)
    anchor_yaw = initial_yaws[:, 0]
    zero_velocity = torch.zeros_like(positions[..., :2])
    initial_metrics = compute_team_metrics(positions, zero_velocity)

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
    link_clocks: list[list[list[Clock]]] = [
        [[{} for _ in range(core.n_agents)] for _ in range(core.n_agents)]
        for _ in range(core.num_envs)
    ]
    committed_step = torch.full(
        (core.num_envs,), -1, dtype=torch.long, device=positions.device
    )
    committed_centers = torch.zeros(
        core.num_envs, 2, dtype=positions.dtype, device=positions.device
    )
    success = torch.zeros(core.num_envs, dtype=torch.bool, device=positions.device)
    collision = torch.zeros_like(success)
    out_of_bounds = torch.zeros_like(success)
    hold_count = torch.zeros(core.num_envs, dtype=torch.long, device=positions.device)
    previous_actions = torch.zeros(
        core.num_envs, core.n_agents, dtype=torch.long, device=positions.device
    )
    velocities = zero_velocity
    risk_sum = torch.zeros(core.num_envs, device=positions.device)
    risk_steps = torch.zeros(core.num_envs, device=positions.device)
    travel_distance = torch.zeros(
        core.num_envs, core.n_agents, device=positions.device
    )
    action_counts = {key: 0 for key in ("hold", "forward", "reverse", "spin", "yield")}
    action_switches = torch.zeros(
        core.num_envs, core.n_agents, device=positions.device
    )
    action_decisions = torch.zeros_like(action_switches)
    total_delta = 0
    total_full = 0
    total_changed = 0
    all_semantic_match = True
    scan_count = 0
    dwell_steps = int(closed_cfg["primitive_commitment_steps"])
    start_time = time.monotonic()

    for step in range(int(active_cfg["episode_steps"])):
        done = success | collision | out_of_bounds
        discovering = (committed_step < 0) & ~done
        if step % int(active_cfg["scan_interval_steps"]) == 0 and bool(discovering.any()):
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
                required_flatness_samples_per_ring=int(
                    cfg.gather_point.flatness_samples_per_ring
                ),
                max_height_range_m=float(cfg.gather_point.max_height_range),
                max_slope=float(cfg.gather_point.max_slope),
                nms_distance_m=float(site_cfg["nms_distance_m"]),
                id_quantization_m=float(site_cfg["id_quantization_m"]),
                source_frame_positions=initial_positions,
                source_frame_yaws=initial_yaws,
                candidate_offsets_body=ACTIVE_SCAN_OFFSETS,
            )
            certificates, communication = _exchange_delta(
                caches,
                link_clocks,
                proposals,
                _adjacency(positions, float(cfg.observation.communication_radius)),
                positions,
                step=step,
                version=scan_count,
                forwarding_rounds=int(active_cfg["forwarding_rounds"]),
            )
            total_delta += int(communication["delta_records"])
            total_full += int(communication["full_records"])
            total_changed += int(communication["changed_records"])
            all_semantic_match &= bool(communication["semantic_match"])
            for environment, certificate in enumerate(certificates):
                if certificate is not None and int(committed_step[environment]) < 0:
                    committed_step[environment] = step
                    committed_centers[environment] = torch.tensor(
                        certificate.center_xy,
                        dtype=positions.dtype,
                        device=positions.device,
                    )
            scan_count += 1
            discovering = (committed_step < 0) & ~done

        exploration_actions = _exploration_actions(
            positions,
            yaws,
            discovering,
            initial_team_center,
            anchor_yaw,
            step=step,
            cycle_steps=int(active_cfg["exploration_cycle_steps"]),
            spin_steps=int(active_cfg["exploration_spin_steps"]),
            collision_avoidance_distance_m=float(
                active_cfg["collision_avoidance_distance_m"]
            ),
            world_limit_m=float(cfg.safety.world_xy_limit),
            boundary_margin_m=float(active_cfg["boundary_margin_m"]),
            frontier_segment_steps=int(active_cfg["frontier_segment_steps"]),
            frontier_radius_m=float(active_cfg["frontier_radius_m"]),
            frontier_yaw_tolerance_rad=float(
                active_cfg["frontier_yaw_tolerance_rad"]
            ),
        )

        gathering = (committed_step >= 0) & ~done
        metrics_before = compute_team_metrics(positions, velocities)
        centroid_flatness = evaluate_gather_point_flatness(
            metrics_before.centroid[..., :2],
            cfg.terrain,
            core.terrain_runtime,
            radius=float(site_cfg["required_flat_radius_m"]),
            rings=int(cfg.gather_point.flatness_rings),
            samples_per_ring=int(cfg.gather_point.flatness_samples_per_ring),
            max_height_range=float(cfg.gather_point.max_height_range),
            max_slope=float(cfg.gather_point.max_slope),
        )
        terminal_gates = compute_success_gates(
            metrics_before,
            torch.zeros_like(velocities),
            cfg.success_thresholds,
            flatness_ok=centroid_flatness.is_flat,
        )
        all_neighbours_visible = metrics_before.dmax <= float(
            cfg.observation.communication_radius
        )
        terminal_hold = gathering & terminal_gates.instant_success & all_neighbours_visible
        recompute = gathering & (
            ((step - committed_step).remainder(max(dwell_steps, 1))) == 0
        )
        if bool(recompute.any()):
            candidate_commitments = previous_actions
            result = None
            for _ in range(int(closed_cfg["primitive_negotiation_rounds"])):
                result = select_decentralized_primitives(
                    positions,
                    yaws,
                    committed_centers,
                    candidate_commitments,
                    gathering,
                    cfg,
                    core.terrain_runtime,
                    terminal_hold=terminal_hold,
                    communication_radius_m=float(cfg.observation.communication_radius),
                )
                candidate_commitments = torch.where(
                    recompute[:, None], result.actions, previous_actions
                )
            assert result is not None
            action_switches += (
                (candidate_commitments != previous_actions)
                & recompute[:, None]
            ).to(action_switches.dtype)
            action_decisions += recompute[:, None].to(action_decisions.dtype)
            previous_actions = candidate_commitments
        previous_actions = torch.where(
            terminal_hold[:, None], torch.zeros_like(previous_actions), previous_actions
        )
        combined_actions = torch.where(
            gathering[:, None], previous_actions, exploration_actions
        )
        combined_actions = torch.where(
            done[:, None], torch.zeros_like(combined_actions), combined_actions
        )
        for key, value in _action_family_counts(combined_actions, gathering).items():
            action_counts[key] += value

        previous_xy = positions[..., :2].clone()
        next_positions, next_yaws, risk = _advance(
            positions, yaws, combined_actions, cfg, core.terrain_runtime
        )
        dt = float(cfg.simulation.planning_dt)
        velocities = (next_positions[..., :2] - previous_xy) / dt
        positions, yaws = next_positions, next_yaws
        moved = ~done
        travel_distance += torch.linalg.vector_norm(
            positions[..., :2] - previous_xy, dim=-1
        ) * moved[:, None].to(positions.dtype)
        risk_sum += risk.mean(dim=1) * moved.to(risk.dtype)
        risk_steps += moved.to(risk.dtype)

        new_collision = check_collision(positions, cfg.safety) & ~done
        new_oob = check_out_of_bounds(positions, cfg.safety) & ~done
        collision |= new_collision
        out_of_bounds |= new_oob
        alive_committed = (committed_step >= 0) & ~collision & ~out_of_bounds & ~success
        metrics_after = compute_team_metrics(positions, velocities)
        flatness_after = evaluate_gather_point_flatness(
            metrics_after.centroid[..., :2],
            cfg.terrain,
            core.terrain_runtime,
            radius=float(site_cfg["required_flat_radius_m"]),
            rings=int(cfg.gather_point.flatness_rings),
            samples_per_ring=int(cfg.gather_point.flatness_samples_per_ring),
            max_height_range=float(cfg.gather_point.max_height_range),
            max_slope=float(cfg.gather_point.max_slope),
        )
        gates_after = compute_success_gates(
            metrics_after,
            velocities,
            cfg.success_thresholds,
            flatness_ok=flatness_after.is_flat,
        )
        hold_count = torch.where(
            alive_committed & gates_after.instant_success,
            hold_count + 1,
            torch.zeros_like(hold_count),
        )
        success |= hold_count >= int(cfg.success_thresholds.hold_steps)

    final_metrics = compute_team_metrics(positions, velocities)
    committed = committed_step >= 0
    timeout = ~success & ~collision & ~out_of_bounds
    center_flatness = evaluate_gather_point_flatness(
        final_metrics.centroid[..., :2],
        cfg.terrain,
        core.terrain_runtime,
        radius=float(site_cfg["required_flat_radius_m"]),
        rings=int(cfg.gather_point.flatness_rings),
        samples_per_ring=int(cfg.gather_point.flatness_samples_per_ring),
        max_height_range=float(cfg.gather_point.max_height_range),
        max_slope=float(cfg.gather_point.max_slope),
    )
    final_gates = compute_success_gates(
        final_metrics,
        velocities,
        cfg.success_thresholds,
        flatness_ok=center_flatness.is_flat,
    )
    final_site_distance = torch.linalg.vector_norm(
        positions[..., :2] - committed_centers[:, None, :], dim=-1
    ).mean(dim=-1)
    reduction = 1.0 - total_delta / max(float(total_full), 1.0)
    return {
        "cell": cell["cell"],
        "episodes": core.num_envs,
        "device": device,
        "elapsed_seconds": time.monotonic() - start_time,
        "certificate_rate": float(committed.float().mean().cpu()),
        "success_rate": float(success.float().mean().cpu()),
        "collision_rate": float(collision.float().mean().cpu()),
        "out_of_bounds_rate": float(out_of_bounds.float().mean().cpu()),
        "timeout_rate": float(timeout.float().mean().cpu()),
        "centroid_flatness_rate": float(center_flatness.is_flat.float().mean().cpu()),
        "mean_certificate_step": (
            float(committed_step[committed].float().mean().cpu()) if bool(committed.any()) else None
        ),
        "mean_final_dmax_m": float(final_metrics.dmax.mean().cpu()),
        "mean_final_dmax_ratio": float(
            (final_metrics.dmax / initial_metrics.dmax.clamp_min(1.0e-6)).mean().cpu()
        ),
        "mean_final_dispersion": float(final_metrics.dispersion.mean().cpu()),
        "mean_final_site_distance_m": float(
            final_site_distance[committed].mean().cpu()
        ) if bool(committed.any()) else None,
        "final_gate_rates": {
            "dmax": float(final_gates.dmax_ok.float().mean().cpu()),
            "dispersion": float(final_gates.dispersion_ok.float().mean().cpu()),
            "speed": float(final_gates.speed_ok.float().mean().cpu()),
            "min_pairwise": float(final_gates.min_pairwise_ok.float().mean().cpu()),
            "flatness": float(final_gates.flatness_ok.float().mean().cpu()),
            "instant_success": float(final_gates.instant_success.float().mean().cpu()),
        },
        "timeout_gate_failure_counts": {
            "dmax": int((timeout & ~final_gates.dmax_ok).sum().cpu()),
            "dispersion": int((timeout & ~final_gates.dispersion_ok).sum().cpu()),
            "speed": int((timeout & ~final_gates.speed_ok).sum().cpu()),
            "min_pairwise": int((timeout & ~final_gates.min_pairwise_ok).sum().cpu()),
            "flatness": int((timeout & ~final_gates.flatness_ok).sum().cpu()),
        },
        "mean_action_switch_rate": float(
            (action_switches / action_decisions.clamp_min(1.0)).mean().cpu()
        ),
        "mean_path_risk": float((risk_sum / risk_steps.clamp_min(1.0)).mean().cpu()),
        "mean_rover_travel_distance_m": float(travel_distance.mean().cpu()),
        "delta_records": total_delta,
        "full_flood_records": total_full,
        "delta_record_reduction": reduction,
        "changed_records": total_changed,
        "delta_full_semantic_match": all_semantic_match,
        "action_family_counts": action_counts,
        "scan_count": scan_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/exp165_active_dstc_closed_loop.yaml",
    )
    parser.add_argument(
        "--manifest",
        default=(
            "outputs/runs/exp156_differential_multiscale_ablation/"
            "_suite/scenario_manifest.json"
        ),
    )
    parser.add_argument("--episodes-per-cell", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--run-dir",
        default="outputs/runs/exp165_active_dstc_closed_loop/pilot_32env",
    )
    parser.add_argument("--cells", default="")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    selected = {item.strip() for item in args.cells.split(",") if item.strip()}
    cells = [
        cell for cell in manifest["cells"] if not selected or cell["cell"] in selected
    ]
    run_dir = ROOT / args.run_dir
    metrics_dir = run_dir / "metrics"
    config_dir = run_dir / "config"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "experiment.yaml").write_text(
        (ROOT / args.config).read_text(encoding="utf-8"), encoding="utf-8"
    )
    started = datetime.now(timezone.utc)
    manifest_payload = {
        "experiment_id": "exp165_active_dstc_closed_loop",
        "run_id": run_dir.name,
        "lifecycle_status": "running",
        "training_performed": False,
        "execution_semantics": "active_dstc_delta_commit_plus_r4",
        "started_at": started.isoformat(),
        "command": " ".join(__import__("sys").argv),
        "artifacts": {
            "summary": str((metrics_dir / "summary.json").relative_to(ROOT)),
            "log": str((run_dir / "run.log").relative_to(ROOT)),
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2), encoding="utf-8"
    )
    reports = []
    for index, cell in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] {cell['cell']}", flush=True)
        report = _run_cell(
            raw,
            cell,
            episodes=int(args.episodes_per_cell),
            device=args.device,
        )
        reports.append(report)
        (metrics_dir / f"{cell['cell']}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)

    closed_cfg = raw["closed_loop"]
    checks = {
        "all_cells_certificate_ge_required": all(
            item["certificate_rate"] >= float(closed_cfg["required_certificate_rate"])
            for item in reports
        ),
        "all_cells_success_ge_required": all(
            item["success_rate"] >= float(closed_cfg["required_success_rate"])
            for item in reports
        ),
        "all_cells_collision_le_limit": all(
            item["collision_rate"] <= float(closed_cfg["maximum_collision_rate"])
            for item in reports
        ),
        "all_cells_timeout_below_limit": all(
            item["timeout_rate"] < float(closed_cfg["maximum_timeout_rate"])
            for item in reports
        ),
        "all_cells_dmax_ratio_le_limit": all(
            item["mean_final_dmax_ratio"] <= float(closed_cfg["maximum_dmax_ratio"])
            for item in reports
        ),
        "delta_semantics_identical": all(
            bool(item["delta_full_semantic_match"]) for item in reports
        ),
        "delta_records_reduced": (
            1.0
            - sum(item["delta_records"] for item in reports)
            / max(float(sum(item["full_flood_records"] for item in reports)), 1.0)
        )
        >= float(closed_cfg["minimum_delta_record_reduction"]),
    }
    aggregate_delta_reduction = (
        1.0
        - sum(item["delta_records"] for item in reports)
        / max(float(sum(item["full_flood_records"] for item in reports)), 1.0)
    )
    summary = {
        "experiment": "exp165_active_dstc_closed_loop",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "teacher_actions_generated": False,
        "oracle_used": False,
        "episodes_per_cell": int(args.episodes_per_cell),
        "cell_reports": reports,
        "aggregate_delta_record_reduction": aggregate_delta_reduction,
        "checks": checks,
        "pilot_passed": all(checks.values()),
    }
    (metrics_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    manifest_payload["lifecycle_status"] = (
        "completed_pilot" if summary["pilot_passed"] else "failed_gate"
    )
    manifest_payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["pilot_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
