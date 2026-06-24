#!/usr/bin/env python
"""Run exp016 shared-update, BC-only, screen and formal stages."""

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


EXPERIMENT_ID = "exp016_shared_mappo_comm12"
PROBE_RUN = "shared_update_probe_seed23_512k"
BC_RUN = "local_teacher_bc100_seed23"
SCREEN_RUN = "screen_seed23_2m"
FORMAL_RUN = "formal_seed23_8m"
TREND_THRESHOLDS = {
    "dmax_reduction_ratio": 0.40,
    "success_rate": 0.50,
    "collision_rate": 0.03,
    "timeout_rate": 0.50,
}
SCREEN_THRESHOLDS = {
    "dmax_reduction_ratio": 0.30,
    "success_rate": 0.50,
    "collision_rate": 0.03,
    "timeout_rate": 0.50,
}
STRICT_THRESHOLDS = {
    "dmax_reduction_ratio": 0.20,
    "success_rate": 0.90,
    "collision_rate": 0.02,
    "timeout_rate": 0.0,
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


def stage_acceptance(summary: dict, thresholds: dict) -> dict:
    metrics = summary.get("final_eval") or {}
    diagnostics = summary.get("training_diagnostics") or {}
    names = ("dmax_reduction_ratio", "success_rate", "collision_rate", "timeout_rate")
    checks = {
        "finite_metrics": all(
            isinstance(metrics.get(name), (int, float))
            and math.isfinite(float(metrics[name]))
            for name in names
        ),
        "dmax_reduction_ratio": metrics.get("dmax_reduction_ratio", float("inf"))
        <= thresholds["dmax_reduction_ratio"],
        "success_rate": metrics.get("success_rate", 0.0) >= thresholds["success_rate"],
        "collision_rate": metrics.get("collision_rate", float("inf"))
        <= thresholds["collision_rate"],
        "timeout_rate": metrics.get("timeout_rate", float("inf"))
        <= thresholds["timeout_rate"],
        "policy_parameter_updated": diagnostics.get("policy_parameter_delta_l2", 0.0) > 0.0,
        "terrain_input_weights_updated": diagnostics.get("terrain_input_weight_delta_l2", 0.0)
        > 0.0,
        "action_non_degenerate": diagnostics.get("post_training_action_std", 0.0) > 1.0e-4,
        "observation_schema": summary.get("observation_schema_version")
        == "ego_v3_local_terrain_grid",
        "actor_obs_dim": summary.get("actor_obs_dim") == 86,
        "critic_state_dim": summary.get("critic_state_dim") == 54,
        "communication_radius": summary.get("communication_radius") == 12.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "metrics": {name: metrics.get(name) for name in names},
    }


def probe_acceptance(summary: dict) -> dict:
    diagnostics = summary.get("training_diagnostics") or {}
    checks = {
        "status": summary.get("status") == "ok",
        "update_mode": summary.get("update_mode") == "shared_joint",
        "optimizer_count": diagnostics.get("optimizer_count") == 1,
        "joint_update_count": diagnostics.get("joint_update_count") == 2,
        "critic_update_count": diagnostics.get("critic_update_count") == 2,
        "policy_parameter_updated": diagnostics.get("policy_parameter_delta_l2", 0.0) > 0.0,
        "terrain_input_weights_updated": diagnostics.get("terrain_input_weight_delta_l2", 0.0)
        > 0.0,
        "action_non_degenerate": diagnostics.get("post_training_action_std", 0.0) > 1.0e-4,
    }
    return {"passed": all(checks.values()), "checks": checks}


def run_stage(
    config: Path,
    *,
    run_name: str,
    timesteps: int,
    rollout_steps: int,
    checkpoint_interval: int,
    eval_num_envs: int,
    eval_steps: int,
    bc_updates: int | None = None,
    bc_only: bool = False,
    selection_gate: str = "screen",
) -> dict:
    run_dir = ROOT / "outputs" / "runs" / EXPERIMENT_ID / run_name
    summary_path = run_dir / "metrics" / "summary.json"
    if summary_path.exists():
        raise RuntimeError(f"Refusing to overwrite completed run: {run_dir}")
    command = [
        str(ROOT / ".venv_isaaclab" / "bin" / "python"),
        str(ROOT / "scripts" / "train_skrl_mappo.py"),
        "--config",
        str(config),
        "--device",
        "cuda",
        "--timesteps",
        str(timesteps),
        "--seed",
        "23",
        "--num-envs",
        "2048",
        "--output-layout",
        "run",
        "--run-name",
        run_name,
        "--rollout-steps",
        str(rollout_steps),
        "--checkpoint-interval",
        str(checkpoint_interval),
        "--eval-num-envs",
        str(eval_num_envs),
        "--eval-steps",
        str(eval_steps),
        "--eval-seed-offset",
        "1000",
        "--selection-gate",
        selection_gate,
    ]
    if bc_updates is not None:
        command.extend(["--bc-updates", str(bc_updates)])
    if bc_only:
        command.append("--bc-only")
    print(f"running: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{run_name} failed with exit code {completed.returncode}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/exp016_shared_mappo_comm12.yaml",
    )
    parser.add_argument("--minimum-free-gpu-mb", type=int, default=6144)
    parser.add_argument("--formal-timesteps", type=int, default=4096)
    args = parser.parse_args()

    config = Path(args.config)
    if not config.is_absolute():
        config = ROOT / config
    raw = load_yaml(config)
    if raw.get("experiment", {}).get("name") != EXPERIMENT_ID:
        raise SystemExit(f"Expected experiment.name={EXPERIMENT_ID}")
    free_mb = cuda_free_memory_mb()
    if free_mb < args.minimum_free_gpu_mb:
        raise SystemExit(
            f"exp016 requires at least {args.minimum_free_gpu_mb} MB free GPU memory; "
            f"found {free_mb} MB."
        )

    suite_dir = ROOT / "outputs" / "runs" / EXPERIMENT_ID / "_suite"
    suite_metrics = suite_dir / "metrics"
    stages: dict[str, dict] = {}

    probe = run_stage(
        config,
        run_name=PROBE_RUN,
        timesteps=256,
        rollout_steps=128,
        checkpoint_interval=256,
        eval_num_envs=256,
        eval_steps=60,
        bc_updates=0,
    )
    probe_gate = probe_acceptance(probe)
    stages["shared_update_probe"] = {"summary": probe, "gate": probe_gate}
    _write_json(
        ROOT / "outputs" / "runs" / EXPERIMENT_ID / PROBE_RUN / "metrics" / "probe_gate.json",
        probe_gate,
    )
    if not probe_gate["passed"]:
        status = "shared_update_probe_failed"
    else:
        bc = run_stage(
            config,
            run_name=BC_RUN,
            timesteps=0,
            rollout_steps=64,
            checkpoint_interval=0,
            eval_num_envs=1024,
            eval_steps=220,
            bc_updates=100,
            bc_only=True,
        )
        bc_gate = stage_acceptance(bc, TREND_THRESHOLDS)
        stages["bc_only"] = {"summary": bc, "gate": bc_gate}
        _write_json(
            ROOT / "outputs" / "runs" / EXPERIMENT_ID / BC_RUN / "metrics" / "bc_gate.json",
            bc_gate,
        )
        if not bc_gate["passed"]:
            status = "bc_probe_failed"
        else:
            screen = run_stage(
                config,
                run_name=SCREEN_RUN,
                timesteps=1024,
                rollout_steps=64,
                checkpoint_interval=256,
                eval_num_envs=1024,
                eval_steps=220,
                bc_updates=100,
            )
            screen_gate = stage_acceptance(screen, SCREEN_THRESHOLDS)
            stages["screen"] = {"summary": screen, "gate": screen_gate}
            _write_json(
                ROOT
                / "outputs"
                / "runs"
                / EXPERIMENT_ID
                / SCREEN_RUN
                / "metrics"
                / "screen_gate.json",
                screen_gate,
            )
            if not screen_gate["passed"]:
                status = "screen_failed"
            else:
                formal = run_stage(
                    config,
                    run_name=FORMAL_RUN,
                    timesteps=args.formal_timesteps,
                    rollout_steps=64,
                    checkpoint_interval=256,
                    eval_num_envs=1024,
                    eval_steps=220,
                    bc_updates=100,
                    selection_gate="strict",
                )
                formal_gate = stage_acceptance(formal, STRICT_THRESHOLDS)
                stages["formal"] = {"summary": formal, "gate": formal_gate}
                status = "single_seed_candidate" if formal_gate["passed"] else "formal_failed"

    strict_passed = bool(
        stages.get("formal", {}).get("gate", {}).get("passed", False)
    )
    suite = {
        "status": status,
        "experiment": EXPERIMENT_ID,
        "seed": 23,
        "free_gpu_memory_mb_at_start": free_mb,
        "stages": {
            name: {
                "run": {
                    "shared_update_probe": PROBE_RUN,
                    "bc_only": BC_RUN,
                    "screen": SCREEN_RUN,
                    "formal": FORMAL_RUN,
                }[name],
                "gate": item["gate"],
            }
            for name, item in stages.items()
        },
        "strict_passed": strict_passed,
        "single_seed_only": True,
    }
    _write_json(suite_metrics / "suite_summary.json", suite)
    _write_json(
        suite_metrics / "strict_acceptance.json",
        {
            "passed": strict_passed,
            "single_seed_only": True,
            "seed": 23,
            "formal_gate": stages.get("formal", {}).get("gate"),
        },
    )
    selected_run = (
        FORMAL_RUN
        if "formal" in stages
        else SCREEN_RUN
        if "screen" in stages
        else BC_RUN
        if "bc_only" in stages
        else PROBE_RUN
    )
    best = ROOT / "outputs" / "runs" / EXPERIMENT_ID / selected_run / "checkpoints" / "best.pt"
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
            "producer": "scripts/run_exp016_shared_mappo_training.py",
            "command": " ".join(sys.argv),
            "status": status,
            "artifacts": {
                "suite_summary": str((suite_metrics / "suite_summary.json").relative_to(ROOT)),
                "strict_acceptance": str(
                    (suite_metrics / "strict_acceptance.json").relative_to(ROOT)
                ),
            },
        },
    )
    print(json.dumps(suite, indent=2), flush=True)


if __name__ == "__main__":
    main()
