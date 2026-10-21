"""T11 preparation: probability/GRU contracts only, no environment or optimizer."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from planner_evidence import EvidenceRun, file_sha256, json_write, source_snapshot
from upstream_evidence import require_contract_tests
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import (
    REPRESENTATIONS, build_ablation_policies, clipped_actor_objective,
    decode_payload, replay_action_sequence,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.baselines import primitive_fingerprint

DEFAULT_CONFIG = ROOT / "configs/ablation/t11_action_contracts.yaml"
TEST_FILES = ("tests/test_action_ablation.py", "tests/test_t11_action_diagnostics.py")


def contract_test_sources():
    return {name: file_sha256(ROOT / name) for name in TEST_FILES}


def require_unchanged_test_sources(expected):
    if contract_test_sources() != expected:
        raise ValueError("Contract test source changed during diagnostic")


def load_preparation(path):
    path = Path(path).resolve()
    value = load_yaml(path)
    expected = {"schema", "experiment_id", "base_config", "training_paused", "environment_enabled",
                "representations", "seeds", "synthetic", "screening_budgets_reserved"}
    if (set(value) != expected or value["schema"] != "t11_action_preparation_v1"
            or value["experiment_id"] != "t11_action_contracts"
            or value["training_paused"] is not True or value["environment_enabled"] is not False
            or value["representations"] != list(REPRESENTATIONS)
            or value["seeds"] != [23, 31, 47]
            or value["synthetic"] != {"num_envs": 2, "agents_per_env": 4, "actor_dim": 411,
                                      "critic_dim": 991, "terrain_channels": 2, "hidden_dim": 128,
                                      "sequence_length": 8}
            or value["screening_budgets_reserved"] != {"signal_check": 100000, "multi_seed": 500000,
                                                       "finalist": 2000000, "multi_seed_count": 3}):
        raise ValueError("Invalid or unpaused T11 preparation configuration")
    if not isinstance(value["base_config"], str):
        raise ValueError("Expected independent upstream base config")
    base = (path.parent / value["base_config"]).resolve()
    if not base.is_relative_to(ROOT / "configs/experiment"):
        raise ValueError("T11 base config must be a repository experiment")
    cfg = cfg_from_experiment(base)
    if (getattr(cfg, "runtime_family", None) != "upstream_scale_v1"
            or cfg.metadata.get("training_paused") is not True or cfg.planner_device != "cpu"):
        raise ValueError("T11 preparation requires paused upstream identity")
    return value, base, cfg


def synthetic_rollout(policy, observations, initial_hidden, valid, ended):
    hidden, samples = initial_hidden.clone(), []
    for obs, is_valid, is_ended in zip(observations, valid, ended):
        # The test fixture creates invalid transitions only AFTER original sampling.
        sample = policy.sample(obs, hidden)
        samples.append(sample)
        hidden = torch.where((is_ended | ~is_valid)[:, None, None], 0.0, sample.hidden)
    return samples


def probe_matrix(cfg, seeds):
    records, payloads = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        policies = build_ablation_policies(991, cfg.spec)
        observations = torch.randn(8, 2, 4, 411) * .05
        initial = torch.randn(2, 4, 128) * .1
        valid, ended = torch.ones(8, 2, dtype=torch.bool), torch.zeros(8, 2, dtype=torch.bool)
        valid[4, 1] = False
        ended[2, 0] = ended[5, 1] = True
        for rep, policy in policies.items():
            torch.manual_seed(seed)
            with torch.no_grad():
                original = synthetic_rollout(policy, observations, initial, valid, ended)
            samples = list(original)
            replay_obs = observations.clone()
            replay_obs[~valid] = float("nan")
            for t, sample in enumerate(samples):
                bad = ~valid[t]
                action = sample.action
                physical, latent, log = action.continuous.clone(), sample.latent.clone(), sample.log_prob.clone()
                physical[bad], latent[bad], log[bad] = float("nan"), float("nan"), float("nan")
                samples[t] = replace(sample, action=replace(action, continuous=physical), latent=latent, log_prob=log)
            replay = replay_action_sequence(policy, replay_obs, initial, samples, valid, ended)
            advantage = torch.linspace(-1, 1, 16).reshape(8, 2)
            advantage[~valid] = float("nan")
            loss = clipped_actor_objective(replay["ratio"], advantage, valid)
            before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
            loss.backward()  # Inspect gradients; never construct or step an optimizer.
            gradients = [param.grad for param in policy.parameters() if param.grad is not None]
            gradients_finite = bool(gradients) and all(torch.isfinite(g).all().item() for g in gradients)
            entropy_finite = bool(torch.isfinite(replay["entropy"][valid]).all())
            gradient_norm = float(torch.stack([g.detach().square().sum() for g in gradients]).sum().sqrt())
            unchanged = all(torch.equal(value, policy.state_dict()[name]) for name, value in before.items())
            first = original[0]
            requests = decode_payload(first.action, policy.contract, np.zeros((2, 4, 3)), np.zeros(2))
            state = policy.export_diagnostic_state()
            clone = type(policy)(991, policy.contract)
            clone.load_diagnostic_state(state)
            with torch.no_grad():
                restored = clone.sample(observations[0], initial, deterministic=True)
                expected = policy.sample(observations[0], initial, deterministic=True)
            restore_exact = (torch.equal(restored.action.continuous, expected.action.continuous)
                             and torch.equal(restored.action.primitive_id, expected.action.primitive_id)
                             and torch.equal(restored.action.hold, expected.action.hold)
                             and torch.equal(restored.log_prob, expected.log_prob)
                             and torch.equal(restored.hidden, expected.hidden))
            ratio_error = float((replay["ratio"][valid] - 1).abs().max().detach())
            record = {"representation": rep, "seed": seed, "metadata": policy.diagnostic_metadata,
                      "actual_policy_device": "cpu", "policy_dtype": "float32",
                      "sampled_decisions": 64, "learning_valid_decisions": int(valid.sum()) * 4,
                      "max_original_ratio_error": ratio_error,
                      "finite_gradients": gradients_finite, "weights_unchanged": unchanged,
                      "finite_entropy": entropy_finite, "gradient_norm": gradient_norm,
                      "restore_exact": restore_exact, "hold_samples": int(first.action.hold.sum()),
                      "passed": ratio_error <= 1e-5 and gradients_finite and entropy_finite and unchanged and restore_exact}
            records.append(record)
            payloads.append({"representation": rep, "seed": seed,
                             "requests": [asdict(item) for item in requests],
                             "continuous": first.action.continuous.tolist(), "hold": first.action.hold.tolist(),
                             "primitive_id": first.action.primitive_id.tolist(), "latent": first.latent.tolist(),
                             "log_prob": first.log_prob.tolist(), "executor_feedback": "not_connected",
                             "sequence": {"continuous": torch.stack([s.action.continuous for s in original]).tolist(),
                                          "hold": torch.stack([s.action.hold for s in original]).tolist(),
                                          "primitive_id": torch.stack([s.action.primitive_id for s in original]).tolist(),
                                          "latent": torch.stack([s.latent for s in original]).tolist(),
                                          "log_prob": torch.stack([s.log_prob for s in original]).tolist(),
                                          "learning_valid": valid.tolist(), "episode_ended": ended.tolist()},
                             "recorded_decisions": 64, "sample_record_coverage": 1.0})
    return records, payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in args.run_id):
        raise ValueError("Use a lowercase run_id without path separators")
    torch.set_num_threads(1)
    value, base, cfg = load_preparation(args.config)
    preparation_sha = file_sha256(args.config)
    library_sha = primitive_fingerprint()
    test_sources = contract_test_sources()
    run = EvidenceRun(ROOT / "outputs/runs/t11_action_contracts" / args.run_id, base,
                      settings={"stage": "t11_preparation", "configuration": value,
                                "configuration_sha256": preparation_sha,
                                "primitive_fingerprint": library_sha, "actual_policy_device": "cpu",
                                "contract_test_sources": test_sources,
                                "torch_threads": 1})
    json_write(run.path / "config/action_preparation.json", value)
    junit = run.path / "metrics/contracts.xml"
    command = [sys.executable, "-m", "pytest", "-o", "addopts=", "-q", *TEST_FILES,
               f"--junitxml={junit}"]
    json_write(run.path / "logs/commands.json", {"contract_tests": command, "diagnostic": sys.argv})
    with (run.path / "logs/contracts.log").open("w") as stream:
        process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    require_unchanged_test_sources(test_sources)
    json_write(junit.with_suffix(".source.json"),
               {"source_snapshot": run.identity["source_snapshot"],
                "test_sources": test_sources})
    if process.returncode:
        json_write(run.path / "metrics/preparation_acceptance.json",
                   {"preparation_passed": False, "reason": "contract_tests_failed",
                    "t11_accepted": False, "gate_c_passed": False, "formal_training": "paused"})
        raise RuntimeError(f"T11 contract tests failed; see {run.path / 'logs/contracts.log'}")
    test_record = require_contract_tests(junit, run, TEST_FILES)
    records, payloads = probe_matrix(cfg, value["seeds"])
    require_unchanged_test_sources(test_sources)
    if (source_snapshot() != run.identity["source_snapshot"]
            or file_sha256(args.config) != preparation_sha
            or load_yaml(base) != run.identity["resolved_config"]
            or cfg_from_experiment(base).metadata != run.identity["contracts"]
            or primitive_fingerprint() != library_sha):
        raise ValueError("Producer source/config/library changed; preserve run and retry separately")
    json_write(run.path / "samples/actions.json", payloads)
    json_write(run.path / "metrics/probability_matrix.json", records)
    passed = len(records) == 12 and all(record["passed"] for record in records)
    result = {"preparation_passed": passed, "contract_tests": test_record, "matrix_entries": len(records),
              "sampled_decisions": sum(r["sampled_decisions"] for r in records),
              "training_interactions": 0, "optimizer_updates": 0, "environment_connected": False,
              "actual_policy_device": "cpu", "policy_dtype": "float32",
              "new_scale_gate_a_passed": False, "new_scale_gate_b_passed": False,
              "new_scale_t10_passed": False, "t11_accepted": False, "gate_c_passed": False,
              "formal_training": "paused", "pending": ["S02–S05 unified version and full evidence",
              "executor bridges and execution feedback", "frozen train/eval scenarios and target hardware",
              "explicitly enter gated 100k/500k/2M learning stages"]}
    json_write(run.path / "metrics/preparation_acceptance.json", result)
    print(f"T11 probability preparation: {passed}; 12 panels; {test_record['tests']} contracts; "
          f"no training; evidence={run.path}")
    if not passed:
        raise RuntimeError("T11 probability matrix rejected")


if __name__ == "__main__":
    main()
