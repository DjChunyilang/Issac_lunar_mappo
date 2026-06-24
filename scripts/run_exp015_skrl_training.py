#!/usr/bin/env python
"""Queue and run the exp015 screen/formal SKRL MAPPO training stages."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT, load_yaml


EXPERIMENT_ID = "exp015_skrl_medium_soft_terrain_grid"
SCREEN_RUN = "screen_seed23_2m"
FORMAL_RUN = "formal_seed23_8m"
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


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid(pid: int | None, *, poll_seconds: float = 60.0) -> None:
    if pid is None:
        return
    while process_exists(pid):
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] waiting for PID {pid}; "
            "it will not be interrupted",
            flush=True,
        )
        time.sleep(poll_seconds)
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] PID {pid} has exited; starting exp015",
        flush=True,
    )


def _finite_metrics(metrics: dict, names: tuple[str, ...]) -> bool:
    return all(
        isinstance(metrics.get(name), (int, float))
        and math.isfinite(float(metrics[name]))
        for name in names
    )


def stage_acceptance(
    summary: dict,
    *,
    thresholds: dict,
) -> dict:
    metrics = summary.get("final_eval") or {}
    diagnostics = summary.get("training_diagnostics") or {}
    metric_names = (
        "dmax_reduction_ratio",
        "success_rate",
        "collision_rate",
        "timeout_rate",
    )
    checks = {
        "finite_metrics": _finite_metrics(metrics, metric_names),
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
        "bc_updated_policy": diagnostics.get("bc_parameter_delta_l2", 0.0) > 0.0,
        "observation_schema": summary.get("observation_schema_version")
        == "ego_v3_local_terrain_grid",
        "actor_obs_dim": summary.get("actor_obs_dim") == 86,
        "critic_state_dim": summary.get("critic_state_dim") == 54,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "metrics": {name: metrics.get(name) for name in metric_names},
    }


def _stage_paths(run_name: str) -> tuple[Path, Path]:
    run_dir = ROOT / "outputs" / "runs" / EXPERIMENT_ID / run_name
    return run_dir, run_dir / "metrics" / "summary.json"


def run_stage(
    *,
    config: Path,
    device: str,
    run_name: str,
    timesteps: int,
) -> dict:
    run_dir, summary_path = _stage_paths(run_name)
    if summary_path.exists():
        raise RuntimeError(f"Refusing to overwrite completed run: {run_dir}")
    command = [
        str(ROOT / ".venv_isaaclab" / "bin" / "python"),
        str(ROOT / "scripts" / "train_skrl_mappo.py"),
        "--config",
        str(config),
        "--device",
        device,
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
        "--checkpoint-interval",
        "512",
        "--eval-num-envs",
        "1024",
        "--eval-steps",
        "220",
        "--eval-seed-offset",
        "1000",
        "--selection-gate",
        "screen" if run_name == SCREEN_RUN else "strict",
    ]
    print(f"running: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{run_name} failed with exit code {completed.returncode}")
    if not summary_path.exists():
        raise RuntimeError(f"{run_name} did not produce {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _suite_payload(
    *,
    screen_summary: dict,
    screen_gate: dict,
    formal_summary: dict | None,
    formal_gate: dict | None,
) -> dict:
    formal_passed = bool(formal_gate and formal_gate["passed"])
    return {
        "status": (
            "single_seed_candidate"
            if formal_passed
            else "formal_failed"
            if formal_summary is not None
            else "screen_failed"
        ),
        "experiment": EXPERIMENT_ID,
        "seed": 23,
        "screen": {
            "run": SCREEN_RUN,
            "timesteps": screen_summary.get("timesteps"),
            "env_steps": screen_summary.get("env_steps"),
            "gate": screen_gate,
        },
        "formal": (
            {
                "run": FORMAL_RUN,
                "timesteps": formal_summary.get("timesteps"),
                "env_steps": formal_summary.get("env_steps"),
                "gate": formal_gate,
            }
            if formal_summary is not None
            else None
        ),
        "strict_passed": formal_passed,
        "conclusion": (
            "seed23 passed the strict proxy gate; this is a single-seed candidate only"
            if formal_passed
            else "exp015 has not passed the requested gate"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/exp015_skrl_weak_warmup_medium_soft.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wait-pid", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--screen-timesteps", type=int, default=1024)
    parser.add_argument("--formal-timesteps", type=int, default=4096)
    args = parser.parse_args()

    config = Path(args.config)
    if not config.is_absolute():
        config = ROOT / config
    raw_cfg = load_yaml(config)
    if raw_cfg.get("experiment", {}).get("name") != EXPERIMENT_ID:
        raise SystemExit(f"Expected experiment.name={EXPERIMENT_ID}")

    wait_for_pid(args.wait_pid, poll_seconds=args.poll_seconds)
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required for exp015.")

    suite_dir = ROOT / "outputs" / "runs" / EXPERIMENT_ID / "_suite"
    metrics_dir = suite_dir / "metrics"
    screen_summary = run_stage(
        config=config,
        device=args.device,
        run_name=SCREEN_RUN,
        timesteps=args.screen_timesteps,
    )
    screen_gate = stage_acceptance(screen_summary, thresholds=SCREEN_THRESHOLDS)
    screen_gate["stage"] = "screen_seed23_2m"
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

    formal_summary = None
    formal_gate = None
    if screen_gate["passed"]:
        formal_summary = run_stage(
            config=config,
            device=args.device,
            run_name=FORMAL_RUN,
            timesteps=args.formal_timesteps,
        )
        formal_gate = stage_acceptance(formal_summary, thresholds=STRICT_THRESHOLDS)
        formal_gate["stage"] = "formal_seed23_8m"
        _write_json(
            ROOT
            / "outputs"
            / "runs"
            / EXPERIMENT_ID
            / FORMAL_RUN
            / "metrics"
            / "strict_acceptance.json",
            formal_gate,
        )

    suite = _suite_payload(
        screen_summary=screen_summary,
        screen_gate=screen_gate,
        formal_summary=formal_summary,
        formal_gate=formal_gate,
    )
    _write_json(metrics_dir / "suite_summary.json", suite)
    _write_json(
        metrics_dir / "strict_acceptance.json",
        {
            "passed": suite["strict_passed"],
            "single_seed_only": True,
            "seed": 23,
            "formal_gate": formal_gate,
        },
    )
    best_source = (
        ROOT / "outputs" / "runs" / EXPERIMENT_ID / FORMAL_RUN / "checkpoints" / "best.pt"
        if formal_summary is not None
        else ROOT / "outputs" / "runs" / EXPERIMENT_ID / SCREEN_RUN / "checkpoints" / "best.pt"
    )
    if best_source.exists():
        suite_checkpoint = suite_dir / "checkpoints" / "seed_23_best.pt"
        suite_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_source, suite_checkpoint)
    _write_json(
        suite_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment": EXPERIMENT_ID,
            "producer": "scripts/run_exp015_skrl_training.py",
            "command": " ".join(sys.argv),
            "status": suite["status"],
            "artifacts": {
                "suite_summary": str((metrics_dir / "suite_summary.json").relative_to(ROOT)),
                "strict_acceptance": str(
                    (metrics_dir / "strict_acceptance.json").relative_to(ROOT)
                ),
                "seed_23_best": str(
                    (suite_dir / "checkpoints" / "seed_23_best.pt").relative_to(ROOT)
                ),
            },
        },
    )
    print(json.dumps(suite, indent=2), flush=True)
    raise SystemExit(0 if suite["strict_passed"] else 2)


if __name__ == "__main__":
    main()
