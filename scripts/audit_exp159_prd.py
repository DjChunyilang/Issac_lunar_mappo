#!/usr/bin/env python3
"""Frozen invariance and policy-gradient variance audit for exp159 ALO-PRD."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from analyze_exp125_credit_assignment import _load_critic, _value
from audit_exp158_dae import (
    DEFAULT_MANIFEST,
    _cfg_for_manifest_cell,
    capture_core_state,
    restore_core_state,
)
from dae_credit import compute_raw_gae
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from play import _load_policy_players
from prd_credit import compute_analytical_prd_advantages
from train_skrl_mappo import SKRLCategoricalPolicy


EXPERIMENT_ID = "exp159_analytical_prd"
CONFIGS = {
    "h1": ROOT / "configs/experiment/exp159_h1_prd.yaml",
    "strict": ROOT / "configs/experiment/exp159_strict_prd.yaml",
}
CHECKPOINTS = {
    "h1": ROOT
    / "outputs/runs/exp157_h1_site_belief_n1/"
    "h1_n1_seed23_full_2400iter/checkpoints/ppo_timestep_134400.pt",
    "strict": ROOT
    / "outputs/runs/exp156_differential_multiscale_ablation/"
    "n1_seed23_full_2400iter/checkpoints/ppo_timestep_153600.pt",
}


@dataclass(slots=True)
class PRDRolloutDataset:
    observations: torch.Tensor
    actions: torch.Tensor
    team_raw_advantages: torch.Tensor
    loo_baseline: torch.Tensor
    source_reconstruction_error_max: float
    team_reward_preservation_error_max: float
    collision_specificity_error_max: float


def _load_shared_policy(
    checkpoint: dict[str, Any], cfg: Any, device: torch.device
) -> SKRLCategoricalPolicy:
    metadata = checkpoint.get("metadata") or {}
    observation_space = gym.spaces.Box(
        low=-float("inf"),
        high=float("inf"),
        shape=(cfg.actor_obs_dim,),
        dtype=float,
    )
    action_space = gym.spaces.Discrete(47)
    policy = SKRLCategoricalPolicy(
        observation_space,
        action_space,
        device,
        architecture=str(metadata.get("actor_architecture", "multiscale_n1_cnn")),
    ).to(device)
    policy.load_state_dict(checkpoint["rover_0"]["policy"])
    policy.eval()
    return policy


def collect_prd_rollout(
    *,
    config: Path,
    checkpoint: dict[str, Any],
    device: str,
    num_envs: int,
    steps: int,
    seed: int,
    rollout_length: int = 64,
) -> PRDRolloutDataset:
    raw_cfg = load_yaml(config)
    cfg = cfg_from_experiment(config)
    cfg.simulation.device = device
    cfg.simulation.num_envs = num_envs
    cfg.seed = seed
    cfg.task.analytical_prd_enabled = True
    cfg.initial_state.progress_timestep_override = int(
        (checkpoint.get("metadata") or {}).get("timesteps", 0)
    )
    core = MultiRoverGatheringCore(cfg)
    act, _ = _load_policy_players(checkpoint, cfg, core.device, raw_cfg=raw_cfg)
    critic = _load_critic(checkpoint, cfg, core.device)
    actor_obs, critic_state = core.get_observations()
    generator = torch.Generator(device=core.device)
    generator.manual_seed(seed + 159_000)
    records: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "observations",
            "actions",
            "rewards",
            "terminated",
            "truncated",
            "values",
            "next_values",
            "baseline",
        )
    }
    source_error = 0.0
    reward_error = 0.0
    collision_specificity_error = 0.0
    for _ in range(steps):
        with torch.no_grad():
            probabilities = act.probabilities(actor_obs)
            actions = torch.multinomial(
                probabilities.reshape(-1, 47),
                1,
                generator=generator,
            ).reshape(num_envs, cfg.task.n_agents)
            values = _value(critic, critic_state)
        records["observations"].append(actor_obs.detach().cpu())
        records["actions"].append(actions.detach().cpu())
        records["values"].append(values.detach().cpu())
        output = core.step(actions)
        info = output.info["analytical_prd"]
        baseline = info["loo_baseline"]
        records["baseline"].append(baseline.detach().cpu())
        records["rewards"].append(output.rewards[:, :1].detach().cpu())
        records["terminated"].append(output.terminated[:, None].detach().cpu())
        records["truncated"].append(output.truncated[:, None].detach().cpu())
        records["next_values"].append(
            _value(critic, output.critic_state).detach().cpu()[:, None]
        )
        source_error = max(
            source_error,
            float(info["source_reconstruction_error"].abs().amax().cpu()),
        )
        reward_error = max(
            reward_error,
            float(info["team_reward_preservation_error"].abs().amax().cpu()),
        )
        participants = info["actual_collision_participants"]
        participant_count = participants.sum(dim=-1)
        single_pair = participant_count == 2
        if single_pair.any():
            collision = info["collision_other"]
            participant_values = collision[single_pair][participants[single_pair]]
            collision_specificity_error = max(
                collision_specificity_error,
                float(participant_values.abs().amax().cpu()),
            )
        actor_obs = output.actor_obs
        critic_state = output.critic_state

    stacked = {key: torch.stack(value) for key, value in records.items()}
    raw_segments = []
    for start in range(0, steps, rollout_length):
        end = min(start + rollout_length, steps)
        _, raw = compute_raw_gae(
            rewards=stacked["rewards"][start:end],
            terminated=stacked["terminated"][start:end],
            truncated=stacked["truncated"][start:end],
            values=stacked["values"][start:end, :, None],
            last_values=stacked["next_values"][end - 1],
            discount_factor=float(raw_cfg["algorithm"]["gamma"]),
            lambda_coefficient=float(raw_cfg["algorithm"]["gae_lambda"]),
            time_limit_bootstrap=False,
        )
        raw_segments.append(raw)
    raw_advantages = torch.cat(raw_segments, dim=0)
    return PRDRolloutDataset(
        observations=stacked["observations"],
        actions=stacked["actions"],
        team_raw_advantages=raw_advantages,
        loo_baseline=stacked["baseline"],
        source_reconstruction_error_max=source_error,
        team_reward_preservation_error_max=reward_error,
        collision_specificity_error_max=collision_specificity_error,
    )


def _flat_actor_samples(
    dataset: PRDRolloutDataset,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    time_steps, num_envs, n_agents, obs_dim = dataset.observations.shape
    observations = dataset.observations.reshape(time_steps * num_envs * n_agents, obs_dim)
    actions = dataset.actions.reshape(-1)
    team = dataset.team_raw_advantages[:, :, None, :].expand(
        -1, -1, n_agents, -1
    ).reshape(-1)
    prd = (
        dataset.team_raw_advantages[:, :, None, 0] - dataset.loo_baseline
    ).reshape(-1)
    baseline = dataset.loo_baseline.reshape(-1)
    return observations, actions, team, prd, baseline


def _gradient_vector(
    policy: SKRLCategoricalPolicy,
    observations: torch.Tensor,
    actions: torch.Tensor,
    weights: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    parameters = tuple(policy.parameters())
    accumulators = [torch.zeros_like(parameter) for parameter in parameters]
    total = observations.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        logits, _ = policy.compute(
            {"observations": observations[start:end].to(device)}, role="policy"
        )
        log_prob = torch.log_softmax(logits, dim=-1).gather(
            1, actions[start:end].to(device).long()[:, None]
        )[:, 0]
        loss = -(log_prob * weights[start:end].to(device)).sum() / float(total)
        gradients = torch.autograd.grad(
            loss,
            parameters,
            allow_unused=True,
        )
        for accumulator, gradient in zip(accumulators, gradients, strict=True):
            if gradient is not None:
                accumulator.add_(gradient.detach())
    return torch.cat([value.flatten() for value in accumulators]).cpu()


def _gradient_metrics(
    policy: SKRLCategoricalPolicy,
    dataset: PRDRolloutDataset,
    *,
    device: torch.device,
    bootstrap_minibatches: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float]:
    observations, actions, team, prd, baseline = _flat_actor_samples(dataset)
    team_gradient = _gradient_vector(
        policy, observations, actions, team, device=device
    )
    prd_gradient = _gradient_vector(
        policy, observations, actions, prd, device=device
    )
    baseline_gradient = _gradient_vector(
        policy, observations, actions, baseline, device=device
    )
    team_norm = torch.linalg.vector_norm(team_gradient)
    prd_norm = torch.linalg.vector_norm(prd_gradient)
    cosine = float(
        torch.dot(team_gradient, prd_gradient)
        / (team_norm * prd_norm).clamp_min(1.0e-12)
    )
    baseline_ratio = float(
        torch.linalg.vector_norm(baseline_gradient) / team_norm.clamp_min(1.0e-12)
    )
    norm_difference = float((prd_norm - team_norm).abs() / team_norm.clamp_min(1.0e-12))

    generator = torch.Generator().manual_seed(seed + 159_159)
    team_bootstrap = []
    prd_bootstrap = []
    for _ in range(bootstrap_minibatches):
        index = torch.randint(
            0,
            observations.shape[0],
            (min(bootstrap_samples, observations.shape[0]),),
            generator=generator,
        )
        team_bootstrap.append(
            _gradient_vector(
                policy,
                observations[index],
                actions[index],
                team[index],
                device=device,
            )
        )
        prd_bootstrap.append(
            _gradient_vector(
                policy,
                observations[index],
                actions[index],
                prd[index],
                device=device,
            )
        )
    team_stack = torch.stack(team_bootstrap)
    prd_stack = torch.stack(prd_bootstrap)
    team_variance = float(
        (team_stack - team_stack.mean(dim=0)).square().sum(dim=-1).mean()
    )
    prd_variance = float(
        (prd_stack - prd_stack.mean(dim=0)).square().sum(dim=-1).mean()
    )
    variance_reduction = (team_variance - prd_variance) / max(team_variance, 1.0e-12)
    prd_advantages = compute_analytical_prd_advantages(
        team_raw_advantages=dataset.team_raw_advantages,
        loo_baseline=dataset.loo_baseline,
    )
    return {
        "baseline_gradient_norm_ratio": baseline_ratio,
        "team_prd_gradient_cosine": cosine,
        "team_prd_gradient_norm_relative_difference": norm_difference,
        "team_gradient_variance": team_variance,
        "prd_gradient_variance": prd_variance,
        "gradient_variance_reduction": variance_reduction,
        "prd_advantage_agent_std": float(
            prd_advantages.squeeze(-1).std(dim=0).mean()
        ),
        "baseline_std": float(dataset.loo_baseline.std()),
        "baseline_nonzero_rate": float(
            (dataset.loo_baseline.abs() > 1.0e-8).float().mean()
        ),
        "baseline_team_advantage_abs_ratio": float(
            dataset.loo_baseline.abs().mean()
            / dataset.team_raw_advantages.abs().mean().clamp_min(1.0e-8)
        ),
    }


def _enumeration_invariance(
    *,
    config: Path,
    checkpoint: dict[str, Any],
    manifest_path: Path,
    device: str,
    num_envs_per_cell: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = load_yaml(config)
    maximum_invariance_error = 0.0
    maximum_reward_error = 0.0
    maximum_source_error = 0.0
    states = 0
    for cell in manifest["cells"]:
        cfg, raw = _cfg_for_manifest_cell(
            base,
            cell,
            device=device,
            num_envs=num_envs_per_cell,
        )
        cfg.task.analytical_prd_enabled = True
        core = MultiRoverGatheringCore(cfg)
        act, _ = _load_policy_players(checkpoint, cfg, core.device, raw_cfg=raw)
        actor_obs, _ = core.get_observations()
        for _ in range(32):
            with torch.no_grad():
                actions = act(actor_obs)
            output = core.step(actions)
            actor_obs = output.actor_obs
        snapshot = capture_core_state(core)
        with torch.no_grad():
            base_actions = act(actor_obs)
        for agent in range(core.n_agents):
            baseline_rows = []
            for action in range(47):
                restore_core_state(core, snapshot)
                candidate = base_actions.clone()
                candidate[:, agent] = action
                output = core.step(candidate)
                info = output.info["analytical_prd"]
                baseline_rows.append(info["loo_baseline"][:, agent].cpu())
                maximum_reward_error = max(
                    maximum_reward_error,
                    float(info["team_reward_preservation_error"].abs().amax().cpu()),
                )
                maximum_source_error = max(
                    maximum_source_error,
                    float(info["source_reconstruction_error"].abs().amax().cpu()),
                )
            values = torch.stack(baseline_rows)
            maximum_invariance_error = max(
                maximum_invariance_error,
                float((values - values[:1]).abs().amax()),
            )
        states += core.num_envs
    return {
        "states": states,
        "action_branches": states * 4 * 47,
        "own_action_invariance_max": maximum_invariance_error,
        "team_reward_preservation_error_max": maximum_reward_error,
        "source_reconstruction_error_max": maximum_source_error,
    }


def run_audit(
    *,
    mode: str,
    config: Path,
    checkpoint_path: Path,
    manifest: Path,
    run_dir: Path,
    device: str,
    train_num_envs: int,
    validation_num_envs: int,
    steps: int,
    counterfactual_num_envs: int,
    bootstrap_minibatches: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    train = collect_prd_rollout(
        config=config,
        checkpoint=checkpoint,
        device=device,
        num_envs=train_num_envs,
        steps=steps,
        seed=14023,
    )
    policy_cfg = cfg_from_experiment(config)
    policy = _load_shared_policy(checkpoint, policy_cfg, torch_device)
    validation = {}
    for seed in (15023, 16023):
        dataset = collect_prd_rollout(
            config=config,
            checkpoint=checkpoint,
            device=device,
            num_envs=validation_num_envs,
            steps=steps,
            seed=seed,
        )
        validation[str(seed)] = {
            "samples": int(dataset.observations.shape[0] * dataset.observations.shape[1]),
            "source_reconstruction_error_max": dataset.source_reconstruction_error_max,
            "team_reward_preservation_error_max": dataset.team_reward_preservation_error_max,
            "collision_specificity_error_max": dataset.collision_specificity_error_max,
            **_gradient_metrics(
                policy,
                dataset,
                device=torch_device,
                bootstrap_minibatches=bootstrap_minibatches,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            ),
        }
    enumeration = _enumeration_invariance(
        config=config,
        checkpoint=checkpoint,
        manifest_path=manifest,
        device=device,
        num_envs_per_cell=counterfactual_num_envs,
    )
    checks = {
        "team_reward_preserved": max(
            train.team_reward_preservation_error_max,
            *(item["team_reward_preservation_error_max"] for item in validation.values()),
            enumeration["team_reward_preservation_error_max"],
        )
        == 0.0,
        "source_reconstruction_error_le_1e_6": max(
            train.source_reconstruction_error_max,
            *(item["source_reconstruction_error_max"] for item in validation.values()),
            enumeration["source_reconstruction_error_max"],
        )
        <= 1.0e-6,
        "own_action_invariance_le_1e_6": enumeration["own_action_invariance_max"]
        <= 1.0e-6,
        "collision_specificity_error_zero": max(
            train.collision_specificity_error_max,
            *(item["collision_specificity_error_max"] for item in validation.values()),
        )
        == 0.0,
        "baseline_std_gt_1e_4": all(
            item["baseline_std"] > 1.0e-4 for item in validation.values()
        ),
        "baseline_nonzero_rate_ge_0_10": all(
            item["baseline_nonzero_rate"] >= 0.10 for item in validation.values()
        ),
        "baseline_team_advantage_ratio_ge_0_10": all(
            item["baseline_team_advantage_abs_ratio"] >= 0.10
            for item in validation.values()
        ),
        "baseline_gradient_norm_ratio_le_0_10": all(
            item["baseline_gradient_norm_ratio"] <= 0.10
            for item in validation.values()
        ),
        "team_prd_gradient_cosine_ge_0_95": all(
            item["team_prd_gradient_cosine"] >= 0.95
            for item in validation.values()
        ),
        "gradient_norm_difference_le_0_10": all(
            item["team_prd_gradient_norm_relative_difference"] <= 0.10
            for item in validation.values()
        ),
        "gradient_variance_reduction_ge_0_15": all(
            item["gradient_variance_reduction"] >= 0.15
            for item in validation.values()
        ),
        "prd_advantage_agent_std_gt_1e_4": all(
            item["prd_advantage_agent_std"] > 1.0e-4
            for item in validation.values()
        ),
    }
    passed = all(checks.values())
    result = {
        "material_passport": {
            "origin_skill": "academic-research-suite",
            "origin_mode": "validate",
            "origin_date": datetime.now(timezone.utc).date().isoformat(),
            "verification_status": "ANALYZED",
            "version_label": "exp159_offline_audit_v1",
        },
        "experiment": EXPERIMENT_ID,
        "mode": mode,
        "status": "prd_offline_gate_passed" if passed else "prd_offline_gate_failed",
        "passed": passed,
        "config": str(config),
        "checkpoint": str(checkpoint_path),
        "collection": {
            "train_environments": train_num_envs,
            "validation_environments_per_seed": validation_num_envs,
            "steps": steps,
            "counterfactual_states": enumeration["states"],
            "counterfactual_action_branches": enumeration["action_branches"],
            "bootstrap_minibatches": bootstrap_minibatches,
            "bootstrap_samples": bootstrap_samples,
        },
        "validation": validation,
        "enumeration": enumeration,
        "checks": checks,
        "decision": (
            "allow_h1_seed23_training"
            if passed and mode == "h1"
            else "allow_strict_only_after_h1"
            if passed
            else "stop_prd_training_for_this_mode"
        ),
    }
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output = metrics_dir / "offline_gate.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest_payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "run": f"offline_{mode}_audit",
        "producer": "scripts/audit_exp159_prd.py",
        "summary": {"status": result["status"], "passed": passed},
        "artifacts": {"metrics": str(output.relative_to(ROOT))},
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("h1", "strict"), required=True)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST.relative_to(ROOT)))
    parser.add_argument("--run-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-num-envs", type=int, default=128)
    parser.add_argument("--validation-num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--counterfactual-num-envs", type=int, default=64)
    parser.add_argument("--bootstrap-minibatches", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=4096)
    args = parser.parse_args()

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    config = resolve(args.config) if args.config else CONFIGS[args.mode]
    checkpoint = resolve(args.checkpoint) if args.checkpoint else CHECKPOINTS[args.mode]
    manifest = resolve(args.manifest)
    run_dir = (
        resolve(args.run_dir)
        if args.run_dir
        else ROOT / f"outputs/runs/{EXPERIMENT_ID}/offline_{args.mode}_audit"
    )
    for required in (config, checkpoint, manifest):
        if not required.is_file():
            raise SystemExit(f"Required exp159 audit input is missing: {required}")
    result = run_audit(
        mode=args.mode,
        config=config,
        checkpoint_path=checkpoint,
        manifest=manifest,
        run_dir=run_dir,
        device=args.device,
        train_num_envs=args.train_num_envs,
        validation_num_envs=args.validation_num_envs,
        steps=args.steps,
        counterfactual_num_envs=args.counterfactual_num_envs,
        bootstrap_minibatches=args.bootstrap_minibatches,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps({"status": result["status"], "passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
