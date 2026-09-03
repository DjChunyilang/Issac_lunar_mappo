#!/usr/bin/env python3
"""Run engineering gates then the bounded exp166 seed23 MAPPO training."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone

from _common import ROOT


EXPERIMENT = "exp166_active_dstc_mappo"
RUN_NAME = "n1_seed23_full_4800iter"
SMOKE_EXPERIMENT = "exp166_active_dstc_mappo_smoke"
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
        env={
            **os.environ,
            "PYTHONPATH": "scripts",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
    )
    _write_status(phase, command=command, pid=process.pid)
    returncode = process.wait()
    if returncode:
        _write_status("failed", failed_phase=phase, returncode=returncode)
        raise SystemExit(returncode)


def _assert_smoke(run_name: str) -> None:
    summary_path = (
        ROOT
        / "outputs/runs"
        / SMOKE_EXPERIMENT
        / run_name
        / "metrics/summary.json"
    )
    if not summary_path.is_file():
        raise SystemExit(f"Smoke completed without {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    diagnostics = summary.get("training_diagnostics", {})
    if not diagnostics.get("policy_parameters_finite", False):
        raise SystemExit(f"{run_name} policy parameters are non-finite")


def main() -> None:
    if RUN_DIR.exists():
        raise SystemExit(f"Refusing to overwrite existing exp166 run: {RUN_DIR}")
    python = str(ROOT / ".venv_isaaclab/bin/python3.12")
    _run(
        [
            python,
            "-m",
            "pytest",
            "-q",
            "-ra",
            "tests/test_exp166_active_dstc_mappo.py",
            "tests/test_exp160_site_commitment.py",
        ],
        "engineering_tests",
    )
    _run(
        [
            python,
            "scripts/train_skrl_mappo.py",
            "--config",
            "configs/experiment/exp166_active_dstc_mappo_smoke.yaml",
            "--device",
            "cpu",
            "--num-envs",
            "2",
            "--rollout-steps",
            "64",
            "--timesteps",
            "64",
            "--output-layout",
            "run",
            "--run-name",
            "cpu_2env_64step",
            "--selection-gate",
            "final",
        ],
        "cpu_smoke",
    )
    _assert_smoke("cpu_2env_64step")
    _run(
        [
            python,
            "scripts/train_skrl_mappo.py",
            "--config",
            "configs/experiment/exp166_active_dstc_mappo_smoke.yaml",
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
    _assert_smoke("cuda_256env_64step")
    _run(
        [
            python,
            "scripts/train_skrl_mappo.py",
            "--config",
            "configs/experiment/exp166_active_dstc_mappo_n1.yaml",
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
        "full_training",
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
