#!/usr/bin/env python3
"""Fixed-budget offline representation screen for exp155 N0/N1/N2.

The script deliberately performs one predeclared three-way comparison. It
does not launch RL or mutate candidate definitions based on intermediate
results. Dataset splits are separated by deterministic terrain seeds.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn.functional as F

from _common import ROOT, cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import (
    SPATIOTEMPORAL_ENDPOINTS,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    build_multiscale_local_terrain_observation,
    make_terrain_runtime,
    query_height,
    query_terrain_features,
    randomize_terrain_runtime,
    sample_trajectory_terrain_risk,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import (
    generate_quintic_path,
)
from train_skrl_mappo import SKRLCategoricalPolicy, SKRLPolicy


CANDIDATES = (
    "multiscale_n0_mlp",
    "multiscale_n1_cnn",
    "multiscale_n2_path_conditioned",
)
FAR_ANGLES = torch.tensor((-math.pi / 4, -math.pi / 8, 0.0, math.pi / 8, math.pi / 4))


def _rotate(local_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    return torch.stack(
        (
            cos_yaw * local_xy[..., 0] - sin_yaw * local_xy[..., 1],
            sin_yaw * local_xy[..., 0] + cos_yaw * local_xy[..., 1],
        ),
        dim=-1,
    )


@torch.no_grad()
def generate_split(
    cfg,
    *,
    samples: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    terrain_rows: list[torch.Tensor] = []
    path_rows: list[torch.Tensor] = []
    far_rows: list[torch.Tensor] = []
    trajectory_cfg = copy.deepcopy(cfg.trajectory_generator)
    trajectory_cfg.n_trajectory_points = max(
        32,
        int(math.ceil(1.5 / 0.05)) + 1,
    )
    endpoints = torch.tensor(SPATIOTEMPORAL_ENDPOINTS, device=device)
    far_angles = FAR_ANGLES.to(device)
    for offset in range(0, samples, batch_size):
        count = min(batch_size, samples - offset)
        runtime = make_terrain_runtime(count, device=device)
        cfg.terrain.randomize_per_reset = True
        randomize_terrain_runtime(
            runtime,
            torch.arange(count, device=device),
            cfg.terrain,
            generator=generator,
        )
        positions = torch.empty(count, 1, 3, device=device)
        positions[..., :2].uniform_(-5.0, 5.0, generator=generator)
        positions[..., 2] = query_height(
            positions[..., :2], cfg.terrain, runtime
        ).squeeze(-1)
        yaws = torch.empty(count, 1, device=device).uniform_(
            -math.pi, math.pi, generator=generator
        )
        terrain = build_multiscale_local_terrain_observation(
            positions, yaws, cfg.terrain, runtime
        )[:, 0]

        path_risks = []
        for endpoint in endpoints:
            local = endpoint.view(1, 1, 2).expand(count, 1, -1)
            world_xy = positions[..., :2] + _rotate(local, yaws)
            subgoal = torch.cat((world_xy, positions[..., 2:3]), dim=-1)
            trajectory = generate_quintic_path(
                positions,
                subgoal,
                trajectory_cfg,
                cfg.simulation.planning_dt,
                current_yaws=yaws,
            )
            path_risks.append(
                sample_trajectory_terrain_risk(
                    trajectory.points, cfg.terrain, runtime
                )["risk_mean"][:, 0]
            )
        endpoint_risk = torch.stack(path_risks, dim=-1)
        path_target = endpoint_risk.repeat_interleave(3, dim=-1)

        fractions = torch.linspace(0.0, 1.0, 81, device=device)
        far_local = 4.0 * torch.stack(
            (torch.cos(far_angles), torch.sin(far_angles)), dim=-1
        )
        far_world = positions[:, :, None, None, :2] + _rotate(
            far_local[None, None, :, None, :] * fractions[None, None, None, :, None],
            yaws[:, :, None, None],
        )
        far_features = query_terrain_features(
            far_world[:, 0], cfg.terrain, runtime
        )
        far_target = (1.0 - far_features[..., 4]).mean(dim=-1)
        terrain_rows.append(terrain.cpu())
        path_rows.append(path_target.cpu())
        far_rows.append(far_target.cpu())
    return {
        "terrain": torch.cat(terrain_rows),
        "path_risk": torch.cat(path_rows),
        "far_risk": torch.cat(far_rows),
    }


def _observation(terrain: torch.Tensor, device: torch.device) -> torch.Tensor:
    observation = torch.zeros(terrain.shape[0], 291, device=device)
    observation[:, 62:286] = terrain.to(device)
    return observation


def _direction_groups(device: torch.device) -> tuple[torch.Tensor, ...]:
    endpoints = torch.tensor(SPATIOTEMPORAL_ENDPOINTS, device=device)
    endpoint_angle = torch.atan2(endpoints[:, 1], endpoints[:, 0])
    bins = torch.argmin((endpoint_angle[:, None] - FAR_ANGLES.to(device)[None]).abs(), dim=-1)
    action_bins = bins.repeat_interleave(3)
    return tuple(torch.nonzero(action_bins == index, as_tuple=False).flatten() for index in range(5))


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    left_rank = torch.argsort(torch.argsort(left.flatten())).float()
    right_rank = torch.argsort(torch.argsort(right.flatten())).float()
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    value = (left_rank * right_rank).sum() / (
        torch.linalg.vector_norm(left_rank) * torch.linalg.vector_norm(right_rank)
    ).clamp_min(1.0e-8)
    return float(value)


@torch.no_grad()
def evaluate(
    policy: SKRLCategoricalPolicy,
    split: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    policy.eval()
    predictions = []
    batch_size = 4096
    for start in range(0, split["terrain"].shape[0], batch_size):
        logits, _ = policy.compute(
            {"observations": _observation(split["terrain"][start:start + batch_size], device)},
            role="policy",
        )
        predictions.append(torch.sigmoid(-logits[:, 1:]).cpu())
    predicted = torch.cat(predictions)
    target = split["path_risk"]
    error = (predicted - target).abs()
    selected = predicted.argmin(dim=-1)
    selected_risk = target.gather(1, selected[:, None]).squeeze(1)
    regret = selected_risk - target.amin(dim=-1)
    groups = _direction_groups(torch.device("cpu"))
    predicted_far = torch.stack(
        tuple(predicted[:, group].amin(dim=-1) for group in groups), dim=-1
    )
    far_accuracy = (
        predicted_far.argmin(dim=-1) == split["far_risk"].argmin(dim=-1)
    ).float().mean()
    spearman = _spearman(predicted, target)
    p95 = float(torch.quantile(error.flatten(), 0.95))
    regret_mean = float(regret.mean())
    far_error = float(1.0 - far_accuracy)
    return {
        "spearman": spearman,
        "normalized_error_p95": p95,
        "mean_relative_regret": regret_mean,
        "far_top1_accuracy": float(far_accuracy),
        "offline_score": 0.5 * p95 + 0.25 * (1.0 - spearman) + 0.25 * far_error,
    }


def train_candidate(
    architecture: str,
    train: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
    *,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[SKRLCategoricalPolicy, dict[str, float]]:
    torch.manual_seed(seed)
    policy = SKRLCategoricalPolicy(
        gym.spaces.Box(-float("inf"), float("inf"), shape=(291,)),
        gym.spaces.Discrete(40),
        device,
        architecture=architecture,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    batch_size = 1024
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(train["terrain"].shape[0], generator=generator)
        policy.train()
        for start in range(0, order.numel(), batch_size):
            index = order[start:start + batch_size]
            target = train["path_risk"][index].to(device)
            logits, _ = policy.compute(
                {"observations": _observation(train["terrain"][index], device)},
                role="policy",
            )
            predicted = torch.sigmoid(-logits[:, 1:])
            best = target.argmin(dim=-1)
            loss = F.mse_loss(predicted, target) + 0.2 * F.cross_entropy(
                logits[:, 1:], best
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return policy, evaluate(policy, validation, device)


@torch.no_grad()
def throughput(policy, observation_dim: int, device: torch.device) -> float:
    batch = torch.zeros(1024, observation_dim, device=device)
    for _ in range(20):
        policy.compute({"observations": batch}, role="policy")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(100):
        policy.compute({"observations": batch}, role="policy")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 102_400 / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp155_multiscale_network_ablation.yaml")
    parser.add_argument("--output", default="outputs/runs/exp155_multiscale_network_ablation/_suite/metrics/offline_network_ablation.json")
    parser.add_argument("--dataset", default="outputs/runs/exp155_multiscale_network_ablation/_suite/metrics/offline_dataset.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    raw = load_yaml(args.config)
    settings = raw["network_ablation"]
    total = int(args.samples or settings["dataset_samples"])
    train_count = int(total * 2 / 3) if args.samples else int(settings["train_samples"])
    validation_count = int(total / 6) if args.samples else int(settings["validation_samples"])
    test_count = total - train_count - validation_count
    epochs = int(args.epochs or settings["epochs"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")
    cfg = cfg_from_experiment(args.config)
    dataset_path = ROOT / args.dataset
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.is_file():
        dataset = torch.load(dataset_path, map_location="cpu")
    else:
        dataset = {
            "train": generate_split(cfg, samples=train_count, seed=23, batch_size=2048, device=device),
            "validation": generate_split(cfg, samples=validation_count, seed=31, batch_size=2048, device=device),
            "test": generate_split(cfg, samples=test_count, seed=47, batch_size=2048, device=device),
        }
        torch.save(dataset, dataset_path)

    baseline = SKRLPolicy(
        gym.spaces.Box(-float("inf"), float("inf"), shape=(101,)),
        gym.spaces.Box(-1.0, 1.0, shape=(2,)),
        device,
        architecture="branched_v5",
    ).to(device)
    baseline_throughput = throughput(baseline, 101, device)
    records = []
    for architecture in CANDIDATES:
        seed_records = []
        best_policy = None
        best_score = float("inf")
        for seed in settings["terrain_seeds"]:
            policy, validation = train_candidate(
                architecture,
                dataset["train"],
                dataset["validation"],
                seed=int(seed),
                epochs=epochs,
                device=device,
            )
            test_metrics = evaluate(policy, dataset["test"], device)
            seed_records.append({"seed": int(seed), "validation": validation, "test": test_metrics})
            if test_metrics["offline_score"] < best_score:
                best_policy = policy
                best_score = test_metrics["offline_score"]
        candidate_throughput = throughput(best_policy, 291, device)
        peak_memory_mb = 0.0
        finite_backward = True
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        probe = _observation(dataset["validation"]["terrain"][:1024], device)
        best_policy.zero_grad(set_to_none=True)
        probe_logits, _ = best_policy.compute({"observations": probe}, role="policy")
        probe_logits.square().mean().backward()
        finite_backward = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in best_policy.parameters()
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        mean_metrics = {
            key: sum(row["test"][key] for row in seed_records) / len(seed_records)
            for key in seed_records[0]["test"]
        }
        parameter_count = sum(parameter.numel() for parameter in best_policy.parameters())
        gates = {
            "spearman": mean_metrics["spearman"] >= 0.85,
            "normalized_error_p95": mean_metrics["normalized_error_p95"] <= 0.15,
            "relative_regret": mean_metrics["mean_relative_regret"] <= 0.10,
            "far_top1_accuracy": mean_metrics["far_top1_accuracy"] >= 0.70,
            "parameter_count": parameter_count <= int(settings["max_actor_parameters"]),
            "relative_throughput": candidate_throughput / baseline_throughput >= float(settings["minimum_relative_throughput"]),
            "finite_backward": finite_backward,
            "peak_memory": peak_memory_mb <= float(settings["max_peak_memory_gb"]) * 1024.0,
        }
        records.append(
            {
                "architecture": architecture,
                "parameter_count": parameter_count,
                "throughput_samples_per_s": candidate_throughput,
                "relative_throughput": candidate_throughput / baseline_throughput,
                "peak_memory_mb": peak_memory_mb,
                "metrics": mean_metrics,
                "gates": gates,
                "passed": all(gates.values()),
                "seeds": seed_records,
            }
        )
    eligible = sorted(
        (row for row in records if row["passed"]),
        key=lambda row: (row["metrics"]["offline_score"], row["parameter_count"]),
    )
    result = {
        "config": args.config,
        "dataset": args.dataset,
        "split_sizes": {key: value["terrain"].shape[0] for key, value in dataset.items()},
        "epochs": epochs,
        "baseline_throughput_samples_per_s": baseline_throughput,
        "candidates": records,
        "selected_rl_finalists": [row["architecture"] for row in eligible[:2]],
        "passed": len(eligible) >= 2,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
