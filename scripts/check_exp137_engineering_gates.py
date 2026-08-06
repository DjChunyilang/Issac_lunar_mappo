#!/usr/bin/env python
"""Write machine-readable engineering gates for the exp137 B2 screen."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from _common import ROOT, cfg_from_experiment
from _skrl_metadata import CheckpointCompatibilityError, validate_checkpoint_compatibility
from train_skrl_mappo import build_skrl_mappo_models
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringSKRLEnv,
)


EXPERIMENT_ID = "exp137_decentralized_b2_graph_attention"
CONFIG = ROOT / "configs" / "experiment" / "exp137_decentralized_b2_graph_attention.yaml"
DEFAULT_CPU_RUN = "smoke_cpu_seed23_8env_64steps"
DEFAULT_CUDA_RUN = "smoke_cuda_seed23_256env_64steps"


def _load_summary(run_name: str) -> dict:
    path = ROOT / "outputs" / "runs" / EXPERIMENT_ID / run_name / "metrics" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _smoke_checks(summary: dict, *, expected_device: str) -> dict[str, bool]:
    diagnostics = summary.get("training_diagnostics") or {}
    return {
        f"{expected_device}_status_ok": summary.get("status") == "ok",
        f"{expected_device}_device": str(summary.get("device", "")).startswith(expected_device),
        f"{expected_device}_policy_finite": diagnostics.get("policy_parameters_finite") is True,
        f"{expected_device}_neighbor_encoder_updated": float(
            diagnostics.get("neighbor_encoder_parameter_delta_l2", 0.0)
        )
        > 0.0,
        f"{expected_device}_action_non_degenerate": float(
            diagnostics.get("post_training_action_std", 0.0)
        )
        > 1.0e-4,
        f"{expected_device}_pure_rl": int(diagnostics.get("bc_updates", -1)) == 0
        and float(diagnostics.get("bc_parameter_delta_l2", -1.0)) == 0.0,
    }


def _architecture_checks() -> tuple[dict[str, bool], dict[str, float]]:
    torch.manual_seed(137)
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 8
    env = MultiRoverGatheringSKRLEnv(cfg)
    models = build_skrl_mappo_models(
        env,
        shared_actor=True,
        centralized_critic=True,
        actor_architecture="branched_v6_graph_attention",
        critic_architecture="structured_v1",
    )
    policy = models[env.possible_agents[0]]["policy"]
    observations, _ = env.core.get_observations()
    observations = observations.reshape(-1, cfg.actor_obs_dim)

    def mean(obs: torch.Tensor) -> torch.Tensor:
        return policy.compute({"observations": obs}, role="policy")[0]

    reference = mean(observations)
    neighbor_slots = observations[..., 10:46].reshape(-1, 3, 12)
    permutation_errors = []
    for order in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        permuted = observations.clone()
        permuted[..., 10:46] = neighbor_slots[:, order, :].reshape(-1, 36)
        permutation_errors.append(
            float((mean(permuted) - reference).abs().amax().detach())
        )

    invalid_reference_obs = observations.clone()
    invalid_neighbors = invalid_reference_obs[..., 10:46].reshape(-1, 3, 12)
    invalid_neighbors[:, 1, 11] = 0.0
    invalid_reference = mean(invalid_reference_obs)
    invalid_perturbed = invalid_reference_obs.clone()
    invalid_perturbed_neighbors = invalid_perturbed[..., 10:46].reshape(-1, 3, 12)
    invalid_perturbed_neighbors[:, 1, :11] = torch.linspace(-100.0, 100.0, 11)
    invalid_leak_error = float(
        (mean(invalid_perturbed) - invalid_reference).abs().amax().detach()
    )

    empty = observations.clone()
    empty[..., 10:46] = 0.0
    ego_encoded = policy.ego_encoder(empty[..., :10])
    empty_neighbor_encoding = policy.neighbor_encoder(empty[..., 10:46], ego_encoded)
    empty_encoding_max = float(empty_neighbor_encoding.abs().amax().detach())
    empty_policy_finite = bool(torch.isfinite(mean(empty)).all())

    incompatible_checkpoint = {
        "metadata": {
            "observation_schema_version": cfg.observation.schema_version,
            "actor_obs_dim": cfg.actor_obs_dim,
            "critic_state_dim": cfg.critic_state_dim,
            "actor_architecture": "branched_v5",
            "critic_architecture": "structured_v1",
        }
    }
    checkpoint_rejected = False
    try:
        validate_checkpoint_compatibility(
            incompatible_checkpoint,
            cfg,
            expected_actor_architecture="branched_v6_graph_attention",
            expected_critic_architecture="structured_v1",
        )
    except CheckpointCompatibilityError:
        checkpoint_rejected = True

    evidence = {
        "permutation_max_abs_error": max(permutation_errors),
        "invalid_neighbor_leak_max_abs_error": invalid_leak_error,
        "empty_neighbor_encoding_max_abs": empty_encoding_max,
    }
    checks = {
        "neighbor_permutation_invariant": evidence["permutation_max_abs_error"] <= 1.0e-6,
        "invalid_neighbor_cannot_leak": invalid_leak_error == 0.0,
        "empty_neighbor_encoding_zero": empty_encoding_max == 0.0,
        "empty_neighbor_policy_finite": empty_policy_finite,
        "b0_checkpoint_architecture_rejected": checkpoint_rejected,
    }
    return checks, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-run", default=DEFAULT_CPU_RUN)
    parser.add_argument("--cuda-run", default=DEFAULT_CUDA_RUN)
    args = parser.parse_args()

    cpu_summary = _load_summary(args.cpu_run)
    cuda_summary = _load_summary(args.cuda_run)
    architecture_checks, architecture_evidence = _architecture_checks()
    checks = {
        **architecture_checks,
        **_smoke_checks(cpu_summary, expected_device="cpu"),
        **_smoke_checks(cuda_summary, expected_device="cuda"),
    }
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "status": "engineering_gate_passed" if all(checks.values()) else "engineering_gate_failed",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "permutation_max_abs_error": 1.0e-6,
            "action_std": 1.0e-4,
        },
        "evidence": {
            **architecture_evidence,
            "cpu_smoke": cpu_summary.get("training_diagnostics"),
            "cuda_smoke": cuda_summary.get("training_diagnostics"),
        },
        "runs": {"cpu": args.cpu_run, "cuda": args.cuda_run},
    }
    output = (
        ROOT
        / "outputs"
        / "runs"
        / EXPERIMENT_ID
        / "_suite"
        / "metrics"
        / "engineering_gate.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
