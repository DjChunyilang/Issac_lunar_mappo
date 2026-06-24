#!/usr/bin/env python
"""Run the uninterrupted exp017 pure-RL 20M training and summarize milestones."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT, load_yaml
from train_skrl_mappo import pure_rl_long_acceptance


EXPERIMENT_ID = "exp017_shared_mappo_pure_rl_comm12"
RUN_NAME = "pure_rl_seed23_20m_medium_soft_comm12"
CHECKPOINT_INTERVAL = 1024
ROLLOUT_STEPS = 32
MILESTONES = {
    1024: "2m",
    4096: "8m",
    10240: "20m",
}


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


def milestone_records(
    evaluations: list[dict],
    *,
    num_envs: int,
    timesteps: int,
) -> list[dict]:
    by_timestep = {
        int(item["candidate_timestep"]): item
        for item in evaluations
        if "candidate_timestep" in item
    }
    records = []
    for timestep, label in MILESTONES.items():
        if timestep > timesteps:
            continue
        if timestep not in by_timestep:
            raise RuntimeError(f"Missing evaluated milestone checkpoint at timestep {timestep}.")
        evaluation = by_timestep[timestep]
        records.append(
            {
                "label": label,
                "timestep": timestep,
                "env_steps": timestep * num_envs,
                "checkpoint": evaluation["checkpoint"],
                "metrics": {
                    name: evaluation.get(name)
                    for name in (
                        "dmax_reduction_ratio",
                        "success_rate",
                        "collision_rate",
                        "timeout_rate",
                    )
                },
                "trend_acceptance": pure_rl_long_acceptance(evaluation),
            }
        )
    return records


def engineering_acceptance(
    summary: dict,
    milestones: list[dict],
    *,
    timesteps: int,
    train_metrics_path: Path,
) -> dict:
    diagnostics = summary.get("training_diagnostics") or {}
    expected_updates = timesteps // ROLLOUT_STEPS
    expected_candidates = math.ceil(timesteps / CHECKPOINT_INTERVAL)
    telemetry = [
        json.loads(line)
        for line in train_metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checks = {
        "status": summary.get("status") == "ok",
        "pure_rl": summary.get("bc", {}).get("updates") == 0,
        "shared_joint": summary.get("update_mode") == "shared_joint",
        "optimizer_count": diagnostics.get("optimizer_count") == 1,
        "joint_update_count": diagnostics.get("joint_update_count") == expected_updates,
        "critic_update_count": diagnostics.get("critic_update_count") == expected_updates,
        "policy_parameter_updated": diagnostics.get("policy_parameter_delta_l2", 0.0) > 0.0,
        "terrain_input_weights_updated": diagnostics.get(
            "terrain_input_weight_delta_l2", 0.0
        )
        > 0.0,
        "action_non_degenerate": diagnostics.get("post_training_action_std", 0.0)
        > 1.0e-4,
        "candidate_count": summary.get("candidate_count") == expected_candidates,
        "milestones_present": len(milestones)
        == sum(step <= timesteps for step in MILESTONES),
        "no_nan_flag": bool(telemetry)
        and all(record.get("nan_flag") is not True for record in telemetry),
        "observation_schema": summary.get("observation_schema_version")
        == "ego_v3_local_terrain_grid",
        "actor_obs_dim": summary.get("actor_obs_dim") == 86,
        "critic_state_dim": summary.get("critic_state_dim") == 54,
        "communication_radius": summary.get("communication_radius") == 12.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_joint_updates": expected_updates,
        "expected_candidates": expected_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/exp017_shared_mappo_pure_rl_comm12.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timesteps", type=int, default=10240)
    parser.add_argument("--minimum-free-gpu-mb", type=int, default=6144)
    args = parser.parse_args()

    config = Path(args.config)
    if not config.is_absolute():
        config = ROOT / config
    raw = load_yaml(config)
    experiment = raw.get("experiment", {})
    algorithm = raw.get("algorithm", {})
    if experiment.get("name") != EXPERIMENT_ID:
        raise SystemExit(f"Expected experiment.name={EXPERIMENT_ID}")
    if algorithm.get("mode") != "pure_rl" or int(algorithm.get("bc_updates", -1)) != 0:
        raise SystemExit("exp017 requires algorithm.mode=pure_rl and bc_updates=0.")
    if args.timesteps != 10240:
        raise SystemExit("The exp017 long-run contract requires exactly 10240 timesteps.")
    if args.device.startswith("cuda"):
        free_mb = cuda_free_memory_mb()
        if free_mb < args.minimum_free_gpu_mb:
            raise SystemExit(
                f"exp017 requires at least {args.minimum_free_gpu_mb} MB free GPU memory; "
                f"found {free_mb} MB."
            )
    else:
        free_mb = None

    run_dir = ROOT / "outputs" / "runs" / EXPERIMENT_ID / RUN_NAME
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
        str(args.timesteps),
        "--seed",
        "23",
        "--num-envs",
        "2048",
        "--output-layout",
        "run",
        "--run-name",
        RUN_NAME,
        "--rollout-steps",
        str(ROLLOUT_STEPS),
        "--checkpoint-interval",
        str(CHECKPOINT_INTERVAL),
        "--eval-num-envs",
        "1024",
        "--eval-steps",
        "220",
        "--eval-seed-offset",
        "1000",
        "--bc-updates",
        "0",
        "--selection-gate",
        "pure_rl_long",
    ]
    print(f"running: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    metrics_dir = run_dir / "metrics"
    summary = json.loads((metrics_dir / "summary.json").read_text(encoding="utf-8"))
    eval_metrics = json.loads(
        (metrics_dir / "eval_metrics.json").read_text(encoding="utf-8")
    )
    milestones = milestone_records(
        eval_metrics["evaluations"],
        num_envs=2048,
        timesteps=args.timesteps,
    )
    engineering = engineering_acceptance(
        summary,
        milestones,
        timesteps=args.timesteps,
        train_metrics_path=metrics_dir / "train_metrics.jsonl",
    )

    suite_dir = ROOT / "outputs" / "runs" / EXPERIMENT_ID / "_suite"
    suite_metrics = suite_dir / "metrics"
    strict = summary.get("strict_acceptance") or {"passed": False}
    suite = {
        "status": (
            "single_seed_candidate"
            if strict.get("passed")
            else "completed_engineering_pass"
            if engineering["passed"]
            else "engineering_failed"
        ),
        "experiment": EXPERIMENT_ID,
        "run": RUN_NAME,
        "seed": 23,
        "timesteps": args.timesteps,
        "env_steps": args.timesteps * 2048,
        "free_gpu_memory_mb_at_start": free_mb,
        "best_candidate": summary.get("best_candidate"),
        "final_eval": summary.get("final_eval"),
        "engineering_acceptance": engineering,
        "strict_passed": bool(strict.get("passed")),
        "single_seed_only": True,
    }
    _write_json(
        suite_metrics / "milestones.json",
        {
            "experiment": EXPERIMENT_ID,
            "run": RUN_NAME,
            "candidate_eval_seed": 1023,
            "final_eval_seed": 11023,
            "timeout_used_for_ranking": False,
            "milestones": milestones,
        },
    )
    _write_json(suite_metrics / "suite_summary.json", suite)
    _write_json(
        suite_metrics / "strict_acceptance.json",
        {
            **strict,
            "single_seed_only": True,
            "seed": 23,
            "run": RUN_NAME,
        },
    )
    _write_json(suite_metrics / "final_eval_best.json", summary.get("final_eval") or {})

    best = run_dir / "checkpoints" / "best.pt"
    if best.exists():
        target = suite_dir / "checkpoints" / "seed_23_best.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, target)
    _write_json(
        suite_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment": EXPERIMENT_ID,
            "producer": "scripts/run_exp017_pure_rl_long.py",
            "command": " ".join(sys.argv),
            "status": suite["status"],
            "artifacts": {
                "suite_summary": str(
                    (suite_metrics / "suite_summary.json").relative_to(ROOT)
                ),
                "milestones": str((suite_metrics / "milestones.json").relative_to(ROOT)),
                "strict_acceptance": str(
                    (suite_metrics / "strict_acceptance.json").relative_to(ROOT)
                ),
                "final_eval_best": str(
                    (suite_metrics / "final_eval_best.json").relative_to(ROOT)
                ),
            },
        },
    )
    print(json.dumps(suite, indent=2), flush=True)


if __name__ == "__main__":
    main()
