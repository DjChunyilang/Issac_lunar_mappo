#!/usr/bin/env python
"""Measure whether a decentralized policy uses terrain observations.

The counterfactual action is evaluated at exactly the same rover state and
communication snapshot as the normal action. Only selected terrain scales are
zeroed. Counterfactuals are never executed, so this diagnostic cannot alter
the policy rollout or communication cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import (
    decode_action,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    sample_trajectory_terrain_risk,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import (
    generate_trajectory,
)
from play import _load_policy_players


TERRAIN_SLICES = {
    "ego_v8_decentralized_tiered": {"all": slice(46, 96)},
    "ego_v9_multiscale_intent": {
        "all": slice(62, 286),
        "fine": slice(62, 188),
        "medium": slice(188, 230),
        "coarse": slice(230, 286),
    },
}


def evaluate_terrain_contrast(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    num_envs: int = 512,
    steps: int = 120,
    seed: int = 12023,
    initial_state_progress: int | None = None,
    output: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict:
    raw_cfg = load_yaml(config)
    cfg = cfg_from_experiment(config)
    cfg.simulation.device = device
    cfg.simulation.num_envs = num_envs
    cfg.seed = seed
    if cfg.observation.schema_version not in TERRAIN_SLICES:
        raise ValueError("Terrain contrast requires a tiered decentralized schema.")

    checkpoint_data = torch.load(checkpoint, map_location=torch.device(device))
    metadata = checkpoint_data.get("metadata", {})
    if cfg.initial_state.curriculum_enabled:
        cfg.initial_state.progress_timestep_override = (
            int(metadata.get("timesteps", 0))
            if initial_state_progress is None
            else int(initial_state_progress)
        )

    env = MultiRoverGatheringCore(cfg)
    act, backend = _load_policy_players(
        checkpoint_data,
        cfg,
        env.device,
        raw_cfg=raw_cfg,
    )
    actor_obs, _ = env.get_observations()
    terrain_input_gradient_norms: dict[str, float] = {}
    if cfg.observation.schema_version == "ego_v9_multiscale_intent" and hasattr(act, "logits"):
        differentiable_obs = actor_obs.detach().clone().requires_grad_(True)
        logits = act.logits(differentiable_obs)
        action_weights = torch.linspace(
            -1.0,
            1.0,
            logits.shape[-1],
            device=env.device,
            dtype=logits.dtype,
        )
        sensitivity_objective = (
            torch.softmax(logits, dim=-1) * action_weights
        ).sum(dim=-1).mean()
        input_gradient = torch.autograd.grad(
            sensitivity_objective,
            differentiable_obs,
            retain_graph=False,
            create_graph=False,
        )[0]
        for name, terrain_slice in TERRAIN_SLICES[cfg.observation.schema_version].items():
            if name != "all":
                terrain_input_gradient_norms[name] = float(
                    input_gradient[..., terrain_slice].norm().detach().cpu()
                )

    action_mse_sum = torch.zeros((), device=env.device)
    normal_risk_sum = torch.zeros((), device=env.device)
    variant_risk_sums = {
        name: torch.zeros((), device=env.device)
        for name in TERRAIN_SLICES[cfg.observation.schema_version]
    }
    js_sum = torch.zeros((), device=env.device)
    sample_count = 0
    communication_sums: dict[str, float] = {}
    conflict_sums: dict[str, float] = {}

    for _ in range(steps):
        variants = {}
        for name, terrain_slice in TERRAIN_SLICES[cfg.observation.schema_version].items():
            counterfactual = actor_obs.clone()
            counterfactual[..., terrain_slice] = 0.0
            variants[name] = counterfactual
        with torch.no_grad():
            normal_action = act(actor_obs)
            variant_actions = {name: act(value) for name, value in variants.items()}
            all_zero_action = variant_actions["all"]
            if cfg.observation.schema_version != "ego_v9_multiscale_intent":
                action_mse_sum += (
                    normal_action.float() - all_zero_action.float()
                ).square().mean()
            if hasattr(act, "probabilities"):
                normal_probability = act.probabilities(actor_obs)
                zero_probability = act.probabilities(variants["all"])
                mixture = 0.5 * (normal_probability + zero_probability)
                js_sum += 0.5 * (
                    (normal_probability * (normal_probability.clamp_min(1.0e-8).log() - mixture.clamp_min(1.0e-8).log())).sum(dim=-1)
                    + (zero_probability * (zero_probability.clamp_min(1.0e-8).log() - mixture.clamp_min(1.0e-8).log())).sum(dim=-1)
                ).mean()

            normal_decoded = decode_action(
                normal_action,
                env.positions,
                env.yaws,
                env.cfg.planner,
            )
            normal_trajectory = generate_trajectory(
                env.positions,
                normal_decoded.world_subgoal,
                env.cfg.trajectory_generator,
                env.cfg.simulation.planning_dt,
                current_yaws=env.yaws,
                reference_speed=normal_decoded.reference_speed,
            )
            normal_risk = sample_trajectory_terrain_risk(
                normal_trajectory.points,
                env.cfg.terrain,
                env.terrain_runtime,
            )["risk_mean"]
            normal_risk_sum += normal_risk.mean()
            for name, variant_action in variant_actions.items():
                decoded = decode_action(
                    variant_action, env.positions, env.yaws, env.cfg.planner
                )
                trajectory = generate_trajectory(
                    env.positions,
                    decoded.world_subgoal,
                    env.cfg.trajectory_generator,
                    env.cfg.simulation.planning_dt,
                    current_yaws=env.yaws,
                    reference_speed=decoded.reference_speed,
                )
                variant_risk_sums[name] += sample_trajectory_terrain_risk(
                    trajectory.points, env.cfg.terrain, env.terrain_runtime
                )["risk_mean"].mean()

        step_output = env.step(normal_action)
        actor_obs = step_output.actor_obs
        sample_count += 1

        for key, value in (step_output.info.get("communication") or {}).items():
            if isinstance(value, torch.Tensor):
                communication_sums[key] = communication_sums.get(key, 0.0) + float(
                    value.float().mean().cpu()
                )
        for key, value in (step_output.info.get("trajectory_conflicts") or {}).items():
            if isinstance(value, torch.Tensor) and value.ndim <= 1:
                conflict_sums[key] = conflict_sums.get(key, 0.0) + float(
                    value.float().mean().cpu()
                )

    divisor = max(sample_count, 1)
    action_mse = (
        None
        if cfg.observation.schema_version == "ego_v9_multiscale_intent"
        else float((action_mse_sum / divisor).cpu())
    )
    normal_path_risk = float((normal_risk_sum / divisor).cpu())
    variant_path_risk = {
        name: float((value / divisor).cpu())
        for name, value in variant_risk_sums.items()
    }
    zero_path_risk = variant_path_risk["all"]
    path_risk_reduction = (zero_path_risk - normal_path_risk) / max(
        abs(zero_path_risk),
        1.0e-8,
    )
    scale_risk_reduction = {
        name: (value - normal_path_risk) / max(abs(value), 1.0e-8)
        for name, value in variant_path_risk.items()
        if name != "all"
    }
    terrain_parameter_delta = None
    if run_dir is not None:
        summary_path = Path(run_dir) / "metrics" / "summary.json"
        if not summary_path.is_absolute():
            summary_path = ROOT / summary_path
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            terrain_parameter_delta = summary.get("training_diagnostics", {}).get(
                "terrain_encoder_parameter_delta_l2"
            )
    result = {
        "status": "ok",
        "backend": backend,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "device": str(env.device),
        "num_envs": num_envs,
        "steps": steps,
        "seed": seed,
        "initial_state_progress_timestep": int(
            cfg.initial_state.progress_timestep_override
        ),
        "action_mse_normal_vs_zero_terrain": action_mse,
        "policy_js_normal_vs_zero_terrain": float((js_sum / divisor).cpu()),
        "normal_path_risk_mean": normal_path_risk,
        "zero_terrain_path_risk_mean": zero_path_risk,
        "path_risk_reduction_fraction": path_risk_reduction,
        "scale_zero_path_risk_mean": variant_path_risk,
        "scale_zero_path_risk_reduction_fraction": scale_risk_reduction,
        "terrain_input_gradient_norms": terrain_input_gradient_norms,
        "terrain_encoder_parameter_delta_l2": terrain_parameter_delta,
        "checks": {
            "normal_path_risk_reduced_5pct": path_risk_reduction >= 0.05,
            "policy_js_ge_0_05": float((js_sum / divisor).cpu()) >= 0.05,
            "all_scales_have_nonzero_input_gradient": (
                bool(terrain_input_gradient_norms)
                and all(value > 0.0 for value in terrain_input_gradient_norms.values())
            ),
            "terrain_encoder_parameters_updated": (
                terrain_parameter_delta is not None
                and float(terrain_parameter_delta) > 0.0
            ),
            "coarse_context_reduces_path_risk_5pct": (
                scale_risk_reduction.get("coarse", float("-inf")) >= 0.05
            ),
        },
        "communication": {
            key: value / divisor for key, value in communication_sums.items()
        },
        "mapf_conflicts": {
            key: value / divisor for key, value in conflict_sums.items()
        },
    }

    if output is None and run_dir is not None:
        output = Path(run_dir) / "metrics" / "terrain_contrast.json"
    if output is not None:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["artifact"] = str(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=12023)
    parser.add_argument("--initial-state-progress", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = evaluate_terrain_contrast(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        initial_state_progress=args.initial_state_progress,
        output=args.output,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
