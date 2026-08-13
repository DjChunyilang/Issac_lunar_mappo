#!/usr/bin/env python3
"""Diagnose label identifiability and gate consistency for exp155.

This script does not train or select a policy. It audits the frozen offline
dataset after the predeclared network screen has stopped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn.functional as F

from _common import ROOT
from run_exp155_offline_network_ablation import _direction_groups, _spearman
from train_skrl_mappo import SKRLCategoricalPolicy


def _frequency(classes: torch.Tensor, count: int) -> list[float]:
    values = torch.bincount(classes, minlength=count).float() / classes.numel()
    return [float(value) for value in values]


@torch.no_grad()
def diagnose_split(split: dict[str, torch.Tensor]) -> dict[str, object]:
    terrain = split["terrain"]
    path_risk = split["path_risk"]
    far_risk = split["far_risk"]
    path_by_endpoint = path_risk.reshape(-1, 13, 3)

    groups = _direction_groups(torch.device("cpu"))
    local_direction_risk = torch.stack(
        tuple(path_risk[:, group].amin(dim=-1) for group in groups), dim=-1
    )
    local_direction = local_direction_risk.argmin(dim=-1)
    far_direction = far_risk.argmin(dim=-1)

    sorted_far = far_risk.sort(dim=-1).values
    exact_tie_count = (
        (far_risk - far_risk.amin(dim=-1, keepdim=True)).abs() < 1.0e-8
    ).sum(dim=-1)
    near_tie_count = (
        far_risk - far_risk.amin(dim=-1, keepdim=True) < 1.0e-3
    ).sum(dim=-1)

    policy = SKRLCategoricalPolicy(
        gym.spaces.Box(-float("inf"), float("inf"), shape=(291,)),
        gym.spaces.Discrete(40),
        "cpu",
        architecture="multiscale_n2_path_conditioned",
    )
    fine = terrain[:, :126].reshape(-1, 7, 9, 2).permute(0, 3, 1, 2)[:, 1:2]
    medium = terrain[:, 126:168].reshape(-1, 3, 7, 2).permute(0, 3, 1, 2)[:, 1:2]
    fine_grid = policy.fine_path_grid.expand(terrain.shape[0], -1, -1, -1)
    medium_grid = policy.medium_path_grid.expand(terrain.shape[0], -1, -1, -1)
    fine_samples = F.grid_sample(
        fine, fine_grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    medium_samples = F.grid_sample(
        medium,
        medium_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    fine_valid = (fine_grid.abs() <= 1.0).all(dim=-1).unsqueeze(1)
    medium_valid = (medium_grid.abs() <= 1.0).all(dim=-1).unsqueeze(1)
    selected = torch.where(fine_valid, fine_samples, medium_samples)
    valid = fine_valid | medium_valid
    interpolated = (
        (selected * valid).sum(dim=-1).squeeze(1)
        / valid.sum(dim=-1).squeeze(1).clamp_min(1)
    )
    endpoint_target = path_by_endpoint[:, :, 0]
    interpolation_error = (interpolated - endpoint_target).abs()

    equivalent_minima = (
        path_risk == path_risk.amin(dim=-1, keepdim=True)
    ).sum(dim=-1)
    speed_target_delta = (
        path_by_endpoint - path_by_endpoint[:, :, :1]
    ).abs().amax()
    local_far_agreement = (local_direction == far_direction).float().mean()

    return {
        "sample_count": int(terrain.shape[0]),
        "far_direction_frequency": _frequency(far_direction, 5),
        "far_majority_class_accuracy": max(_frequency(far_direction, 5)),
        "far_exact_tie_fraction": float((exact_tie_count > 1).float().mean()),
        "far_near_tie_fraction_1e-3": float((near_tie_count > 1).float().mean()),
        "far_best_second_gap_median": float(
            (sorted_far[:, 1] - sorted_far[:, 0]).median()
        ),
        "local_direction_frequency": _frequency(local_direction, 5),
        "local_to_far_direction_agreement": float(local_far_agreement),
        "speed_triplet_target_max_delta": float(speed_target_delta),
        "fraction_with_at_least_three_equivalent_minima": float(
            (equivalent_minima >= 3).float().mean()
        ),
        "raw_grid_interpolation": {
            "spearman": _spearman(interpolated, endpoint_target),
            "normalized_error_p95": float(
                torch.quantile(interpolation_error.flatten(), 0.95)
            ),
            "mean_absolute_error": float(interpolation_error.mean()),
        },
        "diagnostic_flags": {
            "far_gate_not_supervised_by_local_path_target": bool(
                local_far_agreement < 0.70
            ),
            "speed_cross_entropy_has_equivalent_label_conflict": bool(
                speed_target_delta == 0.0
                and (equivalent_minima >= 3).all()
            ),
            "path_risk_is_reconstructable_from_observed_grids": bool(
                _spearman(interpolated, endpoint_target) >= 0.85
                and torch.quantile(interpolation_error.flatten(), 0.95) <= 0.15
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=(
            "outputs/runs/exp155_multiscale_network_ablation/_suite/metrics/"
            "offline_dataset.pt"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/runs/exp155_multiscale_network_ablation/_suite/metrics/"
            "offline_task_diagnostic.json"
        ),
    )
    args = parser.parse_args()

    dataset = torch.load(ROOT / args.dataset, map_location="cpu")
    result = {
        "dataset": args.dataset,
        "splits": {
            name: diagnose_split(split) for name, split in dataset.items()
        },
        "training_allowed": False,
        "reason": (
            "The frozen screen stopped and the audit found inconsistent local-path, "
            "speed, and far-direction supervision."
        ),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
