#!/usr/bin/env python
"""Run and decide the exp125 B0 seed23 4M screening stage."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT, load_yaml
from evaluate_terrain_contrast import evaluate_terrain_contrast


EXPERIMENT_ID = "exp125_decentralized_tiered_b0_pure_rl"
SUPPORTED_EXPERIMENT_IDS = {
    EXPERIMENT_ID,
    "exp126_decentralized_b0_centered_terrain_credit",
    "exp131_decentralized_b0_primary_projected_terrain_credit",
    "exp148_trajectory_time_consistency_fix",
}
RUN_NAME = "b0_screen_seed23_4m_near"
TIMESTEPS = 2048
NUM_ENVS = 2048
ROLLOUT_STEPS = 64
CHECKPOINT_INTERVAL = 1024


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cuda_free_memory_mb() -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return min(int(line.strip()) for line in completed.stdout.splitlines() if line.strip())


def _quarter_dmax(records: list[dict]) -> tuple[float | None, float | None]:
    values = [
        float(record["dmax_mean"])
        for record in records
        if record.get("phase") == "train"
        and isinstance(record.get("dmax_mean"), (int, float))
        and math.isfinite(float(record["dmax_mean"]))
    ]
    if not values:
        return None, None
    width = max(1, math.ceil(len(values) / 4))
    return sum(values[:width]) / width, sum(values[-width:]) / width


def screen_acceptance(
    summary: dict,
    telemetry: list[dict],
    terrain_contrast: dict,
) -> dict:
    diagnostics = summary.get("training_diagnostics") or {}
    final_eval = summary.get("final_eval") or {}
    first_dmax, last_dmax = _quarter_dmax(telemetry)
    dmax_reduction = (
        (first_dmax - last_dmax) / first_dmax
        if first_dmax is not None and last_dmax is not None and first_dmax > 0.0
        else float("-inf")
    )
    success_count = max(
        (
            int(record.get("success_done", record.get("success_count", 0)))
            for record in telemetry
        ),
        default=0,
    )
    checks = {
        "no_nan_or_inf": summary.get("status") == "ok"
        and diagnostics.get("policy_parameters_finite") is True
        and bool(telemetry)
        and all(record.get("nan_flag") is not True for record in telemetry),
        "actor_updated": diagnostics.get("policy_parameter_delta_l2", 0.0) > 0.0,
        "neighbor_encoder_updated": diagnostics.get(
            "neighbor_encoder_parameter_delta_l2", 0.0
        )
        > 0.0,
        "terrain_encoder_updated": diagnostics.get(
            "terrain_encoder_parameter_delta_l2", 0.0
        )
        > 0.0,
        "action_non_degenerate": diagnostics.get("post_training_action_std", 0.0)
        > 1.0e-4,
        "training_dmax_reduced_30pct": dmax_reduction >= 0.30,
        "nonzero_success_episode": success_count > 0
        or float(final_eval.get("success_rate", 0.0)) > 0.0,
        "collision_not_over_10pct": float(
            final_eval.get("collision_rate", float("inf"))
        )
        <= 0.10,
        "terrain_action_mse_gt_0_02": float(
            terrain_contrast.get("action_mse_normal_vs_zero_terrain", 0.0)
        )
        > 0.02,
        "terrain_path_risk_reduced_5pct": float(
            terrain_contrast.get("path_risk_reduction_fraction", float("-inf"))
        )
        >= 0.05,
    }
    if diagnostics.get("actor_credit_assignment") not in (None, "none"):
        preservation_errors = [
            float(record.get("actor_credit_team_reward_preservation_error", 0.0))
            for record in telemetry
            if record.get("phase") == "train"
        ]
        checks.update(
            {
                "actor_credit_trace_nonzero": diagnostics.get(
                    "last_actor_credit_std", 0.0
                )
                > 1.0e-4,
                "actor_credit_preserves_team_reward": max(
                    preservation_errors, default=float("inf")
                )
                <= 1.0e-6,
            }
        )
    if diagnostics.get("actor_credit_gradient_mode") == "primary_projected_norm_cap":
        checks.update(
            {
                "gradient_projection_active": diagnostics.get(
                    "last_actor_gradient_conflict_fraction", 0.0
                )
                >= 0.20,
                "gradient_projection_primary_dot_nonnegative": diagnostics.get(
                    "last_actor_gradient_projected_dot_min", float("-inf")
                )
                >= -1.0e-8,
                "gradient_projection_primary_alignment": diagnostics.get(
                    "last_actor_gradient_combined_cosine_min", 0.0
                )
                >= 0.970,
            }
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "training_dmax_reduction": 0.30,
            "action_std": 1.0e-4,
            "collision_rate": 0.10,
            "terrain_action_mse": 0.02,
            "terrain_path_risk_reduction": 0.05,
        },
        "evidence": {
            "first_quarter_dmax_mean": first_dmax,
            "last_quarter_dmax_mean": last_dmax,
            "training_dmax_reduction": dmax_reduction,
            "training_success_count": success_count,
            "final_eval": {
                key: final_eval.get(key)
                for key in (
                    "dmax_reduction_ratio",
                    "success_rate",
                    "collision_rate",
                    "timeout_rate",
                )
            },
            "training_diagnostics": diagnostics,
            "terrain_contrast": terrain_contrast,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/exp125_decentralized_tiered_b0_pure_rl.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--minimum-free-gpu-mb", type=int, default=8192)
    parser.add_argument("--contrast-num-envs", type=int, default=512)
    parser.add_argument("--contrast-steps", type=int, default=120)
    args = parser.parse_args()

    config = Path(args.config)
    if not config.is_absolute():
        config = ROOT / config
    raw = load_yaml(config)
    experiment_id = str(raw.get("experiment", {}).get("name", ""))
    if experiment_id not in SUPPORTED_EXPERIMENT_IDS:
        raise SystemExit(
            "Expected one of experiment.name="
            f"{sorted(SUPPORTED_EXPERIMENT_IDS)}."
        )
    if raw.get("algorithm", {}).get("bc_updates") != 0:
        raise SystemExit("The B0 screen requires bc_updates=0.")

    free_mb = cuda_free_memory_mb() if args.device.startswith("cuda") else None
    if free_mb is not None and free_mb < args.minimum_free_gpu_mb:
        raise SystemExit(
            f"B0 screen requires {args.minimum_free_gpu_mb} MB free GPU memory; "
            f"found {free_mb} MB."
        )

    run_name = str(args.run_name)
    run_dir = ROOT / "outputs" / "runs" / experiment_id / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty run directory: {run_dir}")

    command = [
        str(ROOT / ".venv_isaaclab" / "bin" / "python"),
        str(ROOT / "scripts" / "train_skrl_mappo.py"),
        "--config",
        str(config),
        "--device",
        args.device,
        "--timesteps",
        str(TIMESTEPS),
        "--seed",
        "23",
        "--num-envs",
        str(NUM_ENVS),
        "--output-layout",
        "run",
        "--run-name",
        run_name,
        "--rollout-steps",
        str(ROLLOUT_STEPS),
        "--checkpoint-interval",
        str(CHECKPOINT_INTERVAL),
        "--eval-num-envs",
        "1024",
        "--eval-steps",
        "480",
        "--eval-seed-offset",
        "1000",
        "--bc-updates",
        "0",
        "--selection-gate",
        "screen",
    ]
    print(f"running: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    metrics_dir = run_dir / "metrics"
    summary = json.loads((metrics_dir / "summary.json").read_text(encoding="utf-8"))
    telemetry = [
        json.loads(line)
        for line in (metrics_dir / "train_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    terrain_contrast = evaluate_terrain_contrast(
        config=run_dir / "config" / "experiment.yaml",
        checkpoint=run_dir / "checkpoints" / "best.pt",
        device=args.device,
        num_envs=args.contrast_num_envs,
        steps=args.contrast_steps,
        seed=12023,
        initial_state_progress=TIMESTEPS,
        run_dir=run_dir,
    )
    gate = screen_acceptance(summary, telemetry, terrain_contrast)
    _write_json(metrics_dir / "screen_gate.json", gate)

    suite_dir = ROOT / "outputs" / "runs" / experiment_id / "_suite"
    suite_metrics = suite_dir / "metrics"
    suite_summary = {
        "status": "screen_passed" if gate["passed"] else "screen_failed",
        "experiment": experiment_id,
        "run": run_name,
        "seed": 23,
        "timesteps": TIMESTEPS,
        "env_steps": TIMESTEPS * NUM_ENVS,
        "free_gpu_memory_mb_at_start": free_mb,
        "screen_gate": gate,
        "next_stage": "b0_seed23_40m" if gate["passed"] else "diagnose_b0_failure",
    }
    # Preserve the latest pointer for compatibility, but also retain one summary
    # per run so diagnostic variants do not overwrite each other's evidence.
    _write_json(suite_metrics / "b0_screen_summary.json", suite_summary)
    _write_json(suite_metrics / f"{run_name}_screen_summary.json", suite_summary)
    manifest_path = suite_dir / "run_manifest.json"
    existing_artifacts: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            existing_artifacts = json.loads(
                manifest_path.read_text(encoding="utf-8")
            ).get("artifacts", {})
        except (json.JSONDecodeError, OSError):
            existing_artifacts = {}
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment": experiment_id,
            "producer": "scripts/run_exp125_b0_screen.py",
            "command": " ".join(sys.argv),
            "status": suite_summary["status"],
            "artifacts": {
                **existing_artifacts,
                "screen_summary": str(
                    (suite_metrics / "b0_screen_summary.json").relative_to(ROOT)
                ),
                "run_specific_screen_summary": str(
                    (suite_metrics / f"{run_name}_screen_summary.json").relative_to(ROOT)
                ),
                "run_screen_gate": str(
                    (metrics_dir / "screen_gate.json").relative_to(ROOT)
                ),
                "terrain_contrast": str(
                    (metrics_dir / "terrain_contrast.json").relative_to(ROOT)
                ),
            },
        },
    )
    print(json.dumps(suite_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
