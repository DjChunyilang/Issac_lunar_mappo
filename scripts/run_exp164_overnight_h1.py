#!/usr/bin/env python3
"""Run one bounded overnight H1 standard-MAPPO diagnostic."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT


EXPERIMENT = "exp164_overnight_h1_repaired"
RUN_NAME = "n1_seed23_full_4800iter"
SUITE = ROOT / "outputs/runs" / EXPERIMENT / "_suite"
RUN_DIR = ROOT / "outputs/runs" / EXPERIMENT / RUN_NAME


def _write_status(phase: str, **extra: object) -> None:
    SUITE.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": EXPERIMENT,
        "run": RUN_NAME,
        "phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    (SUITE / "suite_status.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _run(command: list[str], phase: str) -> None:
    _write_status(phase, command=command)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )
    _write_status(phase, command=command, pid=process.pid)
    returncode = process.wait()
    if returncode:
        _write_status("failed", failed_phase=phase, returncode=returncode)
        raise SystemExit(returncode)


def main() -> None:
    if RUN_DIR.exists():
        raise SystemExit(f"Refusing to overwrite existing overnight run: {RUN_DIR}")
    python = str(ROOT / ".venv_isaaclab/bin/python3.12")
    _run(
        [
            python,
            "scripts/train_skrl_mappo.py",
            "--config",
            "configs/experiment/exp164_overnight_h1_smoke.yaml",
            "--device",
            "cuda:0",
            "--num-envs",
            "256",
            "--rollout-steps",
            "64",
            "--timesteps",
            "64",
            "--output-layout",
            "run",
            "--run-name",
            "cuda_256env_64step",
            "--selection-gate",
            "final",
        ],
        "cuda_smoke",
    )
    smoke_summary = (
        ROOT
        / "outputs/runs/exp164_overnight_h1_smoke/cuda_256env_64step/metrics/summary.json"
    )
    if not smoke_summary.is_file():
        raise SystemExit("exp164 smoke completed without metrics/summary.json")
    smoke = json.loads(smoke_summary.read_text(encoding="utf-8"))
    diagnostics = smoke.get("training_diagnostics", {})
    if not diagnostics.get("policy_parameters_finite", False):
        raise SystemExit("exp164 smoke policy parameters are non-finite")
    _run(
        [
            python,
            "scripts/train_skrl_mappo.py",
            "--config",
            "configs/experiment/exp164_overnight_h1_repaired_n1.yaml",
            "--device",
            "cuda:0",
            "--num-envs",
            "256",
            "--rollout-steps",
            "64",
            "--timesteps",
            "307200",
            "--seed",
            "23",
            "--actor-architecture",
            "multiscale_n1_cnn",
            "--output-layout",
            "run",
            "--run-name",
            RUN_NAME,
            "--selection-gate",
            "final",
        ],
        "overnight_training",
    )
    summary_path = RUN_DIR / "metrics/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _write_status(
        "completed",
        summary=str(summary_path.relative_to(ROOT)),
        wall_time_s=summary.get("wall_time_s"),
        timesteps=summary.get("timesteps"),
        env_steps=summary.get("env_steps"),
    )


if __name__ == "__main__":
    main()
