#!/usr/bin/env python
"""Run the pre-registered exp137 B2 seed23 4M exception screen."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT, load_yaml
from evaluate_terrain_contrast import evaluate_terrain_contrast
from run_exp125_b0_screen import cuda_free_memory_mb, screen_acceptance


EXPERIMENT_ID = "exp137_decentralized_b2_graph_attention"
RUN_NAME = "b2_screen_seed23_4m_relative_quintic"
TIMESTEPS = 2048
NUM_ENVS = 2048
ROLLOUT_STEPS = 64
CHECKPOINT_INTERVAL = 1024


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/exp137_decentralized_b2_graph_attention.yaml",
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
    experiment = raw.get("experiment") or {}
    algorithm = raw.get("algorithm") or {}
    if experiment.get("name") != EXPERIMENT_ID:
        raise SystemExit(f"Expected experiment.name={EXPERIMENT_ID!r}.")
    if algorithm.get("actor_architecture") != "branched_v6_graph_attention":
        raise SystemExit("exp137 requires actor_architecture=branched_v6_graph_attention.")
    if algorithm.get("bc_updates") != 0 or algorithm.get("init_checkpoint") is not None:
        raise SystemExit("exp137 requires random-initialized Pure RL.")

    suite_dir = ROOT / "outputs" / "runs" / EXPERIMENT_ID / "_suite"
    engineering_path = suite_dir / "metrics" / "engineering_gate.json"
    if not engineering_path.is_file():
        raise SystemExit("Missing exp137 engineering gate; run check_exp137_engineering_gates.py first.")
    engineering = json.loads(engineering_path.read_text(encoding="utf-8"))
    if engineering.get("passed") is not True:
        raise SystemExit("exp137 engineering gate did not pass; refusing to start 4M.")

    free_mb = cuda_free_memory_mb() if args.device.startswith("cuda") else None
    if free_mb is not None and free_mb < args.minimum_free_gpu_mb:
        raise SystemExit(
            f"exp137 requires {args.minimum_free_gpu_mb} MB free GPU memory; found {free_mb} MB."
        )

    run_name = str(args.run_name)
    run_dir = ROOT / "outputs" / "runs" / EXPERIMENT_ID / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty run directory: {run_dir}")

    command = [
        str(ROOT / ".venv_isaaclab" / "bin" / "python"),
        str(ROOT / "scripts" / "train_skrl_mappo.py"),
        "--config", str(config),
        "--device", args.device,
        "--timesteps", str(TIMESTEPS),
        "--seed", "23",
        "--num-envs", str(NUM_ENVS),
        "--output-layout", "run",
        "--run-name", run_name,
        "--rollout-steps", str(ROLLOUT_STEPS),
        "--checkpoint-interval", str(CHECKPOINT_INTERVAL),
        "--eval-num-envs", "1024",
        "--eval-steps", "480",
        "--eval-seed-offset", "1000",
        "--bc-updates", "0",
        "--selection-gate", "screen",
    ]
    print(f"running: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    metrics_dir = run_dir / "metrics"
    summary = json.loads((metrics_dir / "summary.json").read_text(encoding="utf-8"))
    telemetry = [
        json.loads(line)
        for line in (metrics_dir / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()
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
    base_gate = screen_acceptance(summary, telemetry, terrain_contrast)
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "run": run_name,
        "seed": 23,
        "timesteps": TIMESTEPS,
        "env_steps": TIMESTEPS * NUM_ENVS,
        "free_gpu_memory_mb_at_start": free_mb,
        "engineering_gate": engineering,
        "base_convergence_gate": base_gate,
        "status": "base_gate_passed_pending_b2_comparison" if base_gate["passed"] else "stopped_at_base_gate",
        "next_stage": "run_b2_candidate_comparison" if base_gate["passed"] else "stop_b2_exception",
    }
    _write_json(metrics_dir / "b2_screen_gate.json", result)
    _write_json(suite_dir / "metrics" / "b2_screen_summary.json", result)
    _write_json(
        suite_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment": EXPERIMENT_ID,
            "producer": "scripts/run_exp137_b2_screen.py",
            "command": " ".join(sys.argv),
            "status": result["status"],
            "artifacts": {
                "engineering_gate": str(engineering_path.relative_to(ROOT)),
                "screen_summary": str((suite_dir / "metrics" / "b2_screen_summary.json").relative_to(ROOT)),
                "run_screen_gate": str((metrics_dir / "b2_screen_gate.json").relative_to(ROOT)),
                "terrain_contrast": str((metrics_dir / "terrain_contrast.json").relative_to(ROOT)),
            },
        },
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
