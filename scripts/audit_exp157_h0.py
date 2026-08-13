#!/usr/bin/env python3
"""H0 frozen audit for site observability and decentralized information limits.

The audit performs no learning and creates no action labels. It quantifies
whether the frozen exp156 scenarios expose the common feasible region in the
current local grids, whether the 12 m graph could propagate such evidence, and
whether the H1 spatial site channel obeys the intended SE(2) contract.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    MULTISCALE_TERRAIN_COARSE_X,
    MULTISCALE_TERRAIN_COARSE_Y,
    MULTISCALE_TERRAIN_FINE_X,
    MULTISCALE_TERRAIN_FINE_Y,
    MULTISCALE_TERRAIN_MEDIUM_X,
    MULTISCALE_TERRAIN_MEDIUM_Y,
    build_multiscale_site_belief_observation,
)


def _grid_offsets(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    chunks = []
    for x_values, y_values in (
        (MULTISCALE_TERRAIN_FINE_X, MULTISCALE_TERRAIN_FINE_Y),
        (MULTISCALE_TERRAIN_MEDIUM_X, MULTISCALE_TERRAIN_MEDIUM_Y),
        (MULTISCALE_TERRAIN_COARSE_X, MULTISCALE_TERRAIN_COARSE_Y),
    ):
        x = torch.tensor(x_values, device=device, dtype=dtype)
        y = torch.tensor(y_values, device=device, dtype=dtype)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        chunks.append(torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1))
    return torch.cat(chunks, dim=0)


def _site_in_local_grid(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    site_point: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    delta = site_point[:, None, :2] - positions[..., :2]
    cosine = torch.cos(yaws)
    sine = torch.sin(yaws)
    local = torch.stack(
        (cosine * delta[..., 0] + sine * delta[..., 1],
         -sine * delta[..., 0] + cosine * delta[..., 1]),
        dim=-1,
    )
    offsets = _grid_offsets(device=positions.device, dtype=positions.dtype)
    nearest = torch.linalg.vector_norm(
        local[..., None, :] - offsets[None, None, :, :], dim=-1
    ).amin(dim=-1)
    return nearest <= float(radius)


def _connected_graph(positions: torch.Tensor, radius: float) -> torch.Tensor:
    distance = torch.cdist(positions[..., :2], positions[..., :2])
    reach = distance <= float(radius)
    for pivot in range(positions.shape[1]):
        reach = reach | (reach[:, :, pivot, None] & reach[:, None, pivot, :])
    return reach.all(dim=(1, 2))


def _se2_site_channel_error(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    site_point: torch.Tensor,
    *,
    radius: float,
    sigma: float,
) -> float:
    angle = torch.tensor(0.731, device=positions.device, dtype=positions.dtype)
    translation = torch.tensor((1.7, -2.3), device=positions.device, dtype=positions.dtype)
    rotation = torch.stack(
        (torch.stack((torch.cos(angle), -torch.sin(angle))),
         torch.stack((torch.sin(angle), torch.cos(angle)))),
    )
    transformed_positions = positions.clone()
    transformed_positions[..., :2] = positions[..., :2] @ rotation.T + translation
    transformed_site = site_point.clone()
    transformed_site[..., :2] = site_point[..., :2] @ rotation.T + translation
    original = build_multiscale_site_belief_observation(
        positions, yaws, site_point, None, None,
        site_radius=radius, potential_sigma=sigma,
    ).reshape(*positions.shape[:-1], 112, 3)[..., 2]
    transformed = build_multiscale_site_belief_observation(
        transformed_positions, yaws + angle, transformed_site, None, None,
        site_radius=radius, potential_sigma=sigma,
    ).reshape(*positions.shape[:-1], 112, 3)[..., 2]
    return float((original - transformed).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiment/exp157_h1_site_belief_n1.yaml"
    )
    parser.add_argument(
        "--manifest",
        default=(
            "outputs/runs/exp156_differential_multiscale_ablation/_suite/"
            "scenario_manifest.json"
        ),
    )
    parser.add_argument(
        "--action-audit",
        default=(
            "outputs/runs/exp157_site_belief_diagnostic/h0_frozen_audit/metrics/"
            "action_coverage_audit.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/runs/exp157_site_belief_diagnostic/h0_frozen_audit/metrics/"
            "h0_site_information_audit.json"
        ),
    )
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_raw = load_yaml(args.config)
    cell_reports = []
    se2_errors = []
    for cell in manifest["cells"]:
        raw = copy.deepcopy(base_raw)
        raw.setdefault("experiment", {})["seed"] = int(cell["seed"])
        raw["experiment"]["num_envs"] = int(manifest["episodes_per_cell"])
        raw.setdefault("initial_state", {}).update(cell["initial_state_overrides"])
        raw["initial_state"]["curriculum_enabled"] = False
        raw.setdefault("terrain", {}).update(cell["terrain_overrides"])
        temporary = ROOT / "outputs/runs/exp157_site_belief_diagnostic/_suite/tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        config_path = temporary / f"h0_{cell['cell']}.json"
        config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        cfg = cfg_from_experiment(config_path)
        cfg.simulation.device = "cpu"
        core = MultiRoverGatheringCore(cfg)
        detectable = _site_in_local_grid(
            core.positions,
            core.yaws,
            core.oracle_point,
            float(cfg.observation.site_belief_radius),
        )
        connected = _connected_graph(
            core.positions,
            float(cfg.observation.communication_radius),
        )
        any_detectable = detectable.any(dim=1)
        all_detectable = detectable.all(dim=1)
        propagatable = any_detectable & connected
        se2_errors.append(
            _se2_site_channel_error(
                core.positions[:8],
                core.yaws[:8],
                core.oracle_point[:8],
                radius=float(cfg.observation.site_belief_radius),
                sigma=float(cfg.observation.site_belief_sigma),
            )
        )
        cell_reports.append(
            {
                "cell": cell["cell"],
                "episodes": int(core.num_envs),
                "any_rover_initially_detects_site_rate": float(any_detectable.float().mean()),
                "all_rovers_initially_detect_site_rate": float(all_detectable.float().mean()),
                "communication_graph_connected_rate": float(connected.float().mean()),
                "evidence_propagation_possible_rate": float(propagatable.float().mean()),
            }
        )

    action_audit_path = ROOT / args.action_audit
    action_audit = (
        json.loads(action_audit_path.read_text(encoding="utf-8"))
        if action_audit_path.is_file()
        else None
    )
    cfg = cfg_from_experiment(args.config)
    safe_distance = float(cfg.success_thresholds.min_pairwise_distance)
    square_dmax = (2.0**0.5) * safe_distance
    square_dispersion = 0.5 * safe_distance**2
    square_radius = safe_distance / (2.0**0.5)
    capacity_feasible = (
        square_dmax <= float(cfg.success_thresholds.dmax)
        and square_dispersion <= float(cfg.success_thresholds.dispersion)
        and square_radius <= float(cfg.observation.site_belief_radius)
    )
    weighted_episodes = sum(item["episodes"] for item in cell_reports)
    weighted_propagation = sum(
        item["episodes"] * item["evidence_propagation_possible_rate"]
        for item in cell_reports
    ) / max(weighted_episodes, 1)
    max_se2_error = max(se2_errors, default=float("inf"))
    report = {
        "experiment": "exp157_site_belief_diagnostic",
        "phase": "H0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "teacher_actions_generated": False,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "episodes": weighted_episodes,
        "cell_reports": cell_reports,
        "aggregate": {
            "evidence_propagation_possible_rate": weighted_propagation,
            "site_channel_se2_max_abs_error": max_se2_error,
            "four_rover_square_dmax": square_dmax,
            "four_rover_square_dispersion": square_dispersion,
            "four_rover_square_circumradius": square_radius,
        },
        "checks": {
            "site_channel_se2_invariant": max_se2_error <= 1.0e-5,
            "site_region_has_geometric_capacity": capacity_feasible,
            "action_coverage_audit_available": action_audit is not None,
            "action_coverage_audit_passed": bool(action_audit and action_audit.get("passed")),
        },
        "decentralized_site_selection_ready": False,
        "decentralized_readiness_reason": (
            "H0 quantifies initial evidence and connectivity only; candidate extraction, "
            "cross-rover data association and commit consensus are not yet implemented."
        ),
    }
    report["h1_low_level_diagnostic_launch_allowed"] = all(report["checks"].values())

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    run_dir = output.parents[1]
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp157_site_belief_diagnostic",
                "run_id": "h0_frozen_audit",
                "lifecycle_status": "completed_diagnostic",
                "config": args.config,
                "artifacts": {
                    "site_information_audit": str(output.relative_to(ROOT)),
                    "action_coverage_audit": str(action_audit_path.relative_to(ROOT)),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not report["h1_low_level_diagnostic_launch_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
