#!/usr/bin/env python3
"""Validate a trained DAE reward model against exact frozen interventions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from _common import ROOT
from audit_exp158_dae import (
    DEFAULT_MANIFEST,
    _counterfactual_model_metrics,
    collect_counterfactual_dataset,
)
from dae_credit import CounterfactualRewardModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST.relative_to(ROOT)))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs-per-cell", type=int, default=64)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    config = run_dir / "config/experiment.yaml"
    checkpoint_path = run_dir / "checkpoints/best.pt"
    if not config.is_file() or not checkpoint_path.is_file() or not manifest.is_file():
        raise SystemExit("DAE validation requires run config, best checkpoint and manifest")
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    dae_training = checkpoint.get("dae_training")
    if not isinstance(dae_training, dict) or dae_training.get("deployable") is not False:
        raise SystemExit("Checkpoint has no training-only DAE reward model")
    model = CounterfactualRewardModel().to(device)
    model.load_state_dict(dae_training["reward_model"])
    model.eval()
    counterfactual, collection = collect_counterfactual_dataset(
        config=config,
        checkpoint_data=checkpoint,
        manifest_path=manifest,
        device=args.device,
        num_envs_per_cell=args.num_envs_per_cell,
        skip_long_horizon=True,
    )
    metrics = _counterfactual_model_metrics([model], counterfactual, device=device)
    checks = {
        "prediction_std_gt_1e_4": metrics["minimum_prediction_std"] > 1.0e-4,
        "action_spearman_ge_0_30_each_seed": metrics[
            "minimum_seed_action_spearman"
        ]
        >= 0.30,
        "policy_weighted_expectation_error_le_0_25_std": metrics[
            "policy_weighted_expectation_error_std_fraction"
        ]
        <= 0.25,
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "manifest": str(manifest.relative_to(ROOT)),
        "dae_update_count": int(dae_training["update_count"]),
        "dae_beta": float(dae_training["beta"]),
        "counterfactual_states": int(counterfactual.states.shape[0]),
        "collection": collection,
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = run_dir / "metrics/dae_counterfactual_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": report["passed"]}, indent=2))


if __name__ == "__main__":
    main()
