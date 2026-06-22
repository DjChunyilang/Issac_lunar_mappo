#!/usr/bin/env python
"""Run the exp014 CUDA probe and validate the terrain-observation training contract."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from _common import ROOT, cfg_from_experiment
from _skrl_metadata import validate_checkpoint_compatibility
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore


DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "exp014_terrain_grid_observation_probe.yaml"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_acceptance(metrics: dict[str, Any], *, terrain_observation_max_abs: float) -> dict[str, Any]:
    action_std = metrics.get("post_training_action_std")
    checks = {
        "finite_training": not bool(metrics.get("nan_flag")),
        "policy_parameters_updated": float(metrics.get("policy_parameter_delta_l2") or 0.0) > 0.0,
        "terrain_input_weights_updated": (
            float(metrics.get("terrain_input_weight_delta_l2") or 0.0) > 0.0
        ),
        "actions_non_degenerate": (
            isinstance(action_std, (int, float))
            and math.isfinite(float(action_std))
            and float(action_std) > 1.0e-4
        ),
        "terrain_observation_nonzero": terrain_observation_max_abs > 1.0e-6,
    }
    return {"passed": all(checks.values()), "checks": checks}


def run_validation(
    config: str | Path = DEFAULT_CONFIG,
    *,
    device: str = "cuda",
    timesteps: int = 5000,
) -> dict[str, Any]:
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = cfg_from_experiment(config_path)
    cfg.simulation.device = device
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the exp014 terrain observation validation.")

    metrics_path = ROOT / "outputs" / "runs" / config_path.stem / "metrics.jsonl"
    previous_count = len(_read_jsonl(metrics_path))
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_skrl_mappo.py"),
        "--config",
        str(config_path),
        "--device",
        device,
        "--timesteps",
        str(timesteps),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"exp014 training failed with exit code {completed.returncode}.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    new_rows = _read_jsonl(metrics_path)[previous_count:]
    final_rows = [row for row in new_rows if row.get("phase") == "final"]
    if not final_rows:
        raise RuntimeError(f"exp014 did not append final telemetry: {metrics_path}")
    metrics = final_rows[-1]

    checkpoint_path = Path(str(metrics.get("checkpoint_path", "")))
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise RuntimeError(f"exp014 checkpoint was not generated: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = validate_checkpoint_compatibility(checkpoint, cfg)

    sample_cfg = cfg_from_experiment(config_path)
    sample_cfg.simulation.device = device
    sample_cfg.simulation.num_envs = min(32, sample_cfg.simulation.num_envs)
    env = MultiRoverGatheringCore(sample_cfg)
    actor_obs, _ = env.get_observations()
    terrain_start = (
        sample_cfg.observation.ego_dim
        + sample_cfg.observation.max_neighbors * sample_cfg.observation.neighbor_dim
    )
    terrain_end = terrain_start + sample_cfg.observation.terrain_dim
    terrain_observation_max_abs = float(
        actor_obs[..., terrain_start:terrain_end].abs().amax().detach().cpu()
    )

    acceptance = build_acceptance(
        metrics,
        terrain_observation_max_abs=terrain_observation_max_abs,
    )
    result = {
        "status": "passed" if acceptance["passed"] else "failed",
        "config": str(config_path),
        "device": device,
        "timesteps": timesteps,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "observation_schema_version": metadata["observation_schema_version"],
        "actor_obs_dim": metadata["actor_obs_dim"],
        "critic_state_dim": metadata["critic_state_dim"],
        "policy_parameter_delta_l2": metrics.get("policy_parameter_delta_l2"),
        "terrain_input_weight_delta_l2": metrics.get("terrain_input_weight_delta_l2"),
        "post_training_action_std": metrics.get("post_training_action_std"),
        "terrain_observation_max_abs": terrain_observation_max_abs,
        "acceptance": acceptance,
    }
    output = metrics_path.parent / "terrain_observation_validation_summary.json"
    result["artifact"] = str(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not acceptance["passed"]:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timesteps", type=int, default=5000)
    args = parser.parse_args()
    print(
        json.dumps(
            run_validation(args.config, device=args.device, timesteps=args.timesteps),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
