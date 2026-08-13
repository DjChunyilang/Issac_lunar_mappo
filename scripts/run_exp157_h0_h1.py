#!/usr/bin/env python3
"""Run the exp157 H0 frozen audit and H1 N1 Pure-RL diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    python = str(ROOT / ".venv_isaaclab/bin/python")
    suite = ROOT / "outputs/runs/exp157_site_belief_diagnostic/_suite"
    suite.mkdir(parents=True, exist_ok=True)
    log_path = suite / "logs/launcher.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = suite / "suite_status.json"

    def status(phase: str, **extra: object) -> None:
        status_path.write_text(
            json.dumps(
                {
                    "experiment": "exp157_site_belief_diagnostic",
                    "phase": phase,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    **extra,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def run(command: list[str]) -> None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n"
            )
            stream.flush()
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            status("failed", command=command, returncode=result.returncode)
            raise SystemExit(result.returncode)

    if not args.skip_tests:
        status("engineering_tests")
        run(
            [
                python,
                "-m",
                "pytest",
                "-q",
                "-ra",
                "tests/test_exp157_site_belief.py",
            ]
        )

    status("h0_action_coverage")
    run(
        [
            python,
            "scripts/audit_exp156_action_coverage.py",
            "--config",
            "configs/experiment/exp157_h1_site_belief_n1.yaml",
            "--output",
            (
                "outputs/runs/exp157_site_belief_diagnostic/h0_frozen_audit/"
                "metrics/action_coverage_audit.json"
            ),
        ]
    )
    status("h0_site_information")
    run([python, "scripts/audit_exp157_h0.py"])

    if not args.skip_smoke:
        status("h1_cpu_smoke")
        run(
            [
                python,
                "scripts/train_skrl_mappo.py",
                "--config",
                "configs/experiment/exp157_h1_smoke.yaml",
                "--device",
                "cpu",
                "--num-envs",
                "2",
                "--rollout-steps",
                "8",
                "--timesteps",
                "16",
                "--output-layout",
                "run",
                "--run-name",
                "h1_cpu_n1_2env_16step",
            ]
        )
        status("h1_cuda_smoke")
        run(
            [
                python,
                "scripts/train_skrl_mappo.py",
                "--config",
                "configs/experiment/exp157_h1_smoke.yaml",
                "--device",
                args.device,
                "--num-envs",
                "256",
                "--rollout-steps",
                "16",
                "--timesteps",
                "32",
                "--output-layout",
                "run",
                "--run-name",
                "h1_cuda_n1_256env_32step",
            ]
        )

    run_name = "h1_n1_seed23_full_2400iter"
    status("h1_training", run=run_name, policy_iterations=2400)
    run(
        [
            python,
            "scripts/train_skrl_mappo.py",
            "--config",
            "configs/experiment/exp157_h1_site_belief_n1.yaml",
            "--device",
            args.device,
            "--num-envs",
            "256",
            "--rollout-steps",
            "64",
            "--timesteps",
            "153600",
            "--seed",
            "23",
            "--actor-architecture",
            "multiscale_n1_cnn",
            "--output-layout",
            "run",
            "--run-name",
            run_name,
            "--selection-gate",
            "strict",
        ]
    )
    run_dir = Path("outputs/runs/exp157_h1_site_belief_n1") / run_name
    status("h1_paired_evaluation", run=run_name)
    run(
        [
            python,
            "scripts/evaluate_exp156_paired.py",
            "--run-dir",
            str(run_dir),
            "--device",
            args.device,
        ]
    )
    status("completed", run=run_name)


if __name__ == "__main__":
    main()
