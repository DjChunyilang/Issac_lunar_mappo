#!/usr/bin/env python3
"""Preflight and launch the paired equal-budget exp156 N0/N1/N2 comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT
from exp156_statistics import paired_bootstrap_difference


FULL_TRAIN_CANDIDATES = (
    "multiscale_n0_mlp",
    "multiscale_n1_cnn",
)
SMOKE_CANDIDATES = (
    *FULL_TRAIN_CANDIDATES,
    "multiscale_n2_path_conditioned",
)
SHORT_NAMES = {
    "multiscale_n0_mlp": "n0",
    "multiscale_n1_cnn": "n1",
    "multiscale_n2_path_conditioned": "n2",
}
PARAMETER_COUNTS = {
    "multiscale_n0_mlp": 83_343,
    "multiscale_n1_cnn": 106_591,
    "multiscale_n2_path_conditioned": 31_752,
}


def run(command: list[str], *, log: Path | None = None) -> int:
    if log is None:
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
        stream.flush()
        return subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def cell_episode_values(report: dict, key: str) -> list[float]:
    values = []
    for cell in report["cells"]:
        for episode in cell["metrics"]["episode_metrics"]:
            if key == "success":
                values.append(float(bool(episode["success"])))
            elif key == "collision":
                values.append(float(bool(episode["collision"])))
            elif key == "timeout":
                values.append(float(bool(episode["timeout"])))
            else:
                values.append(float(episode["dmax_ratio"]))
    return values


def aggregate(report: dict) -> dict[str, float]:
    cells = report["cells"]
    return {
        key: sum(
            float(cell["acceptance"]["point_estimates"][key]) for cell in cells
        ) / len(cells)
        for key in ("success", "collision", "timeout", "dmax_ratio")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/exp156_differential_multiscale_ablation.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-cpu-smoke", action="store_true")
    parser.add_argument("--skip-cuda-smoke", action="store_true")
    args = parser.parse_args()

    python = str(ROOT / ".venv_isaaclab/bin/python3.12")
    suite_dir = ROOT / "outputs/runs/exp156_differential_multiscale_ablation/_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = suite_dir / "logs/launcher.log"
    status_path = suite_dir / "suite_status.json"

    def status(phase: str, **extra) -> None:
        status_path.write_text(
            json.dumps(
                {
                    "experiment": "exp156_differential_multiscale_ablation",
                    "phase": phase,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    **extra,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if not args.skip_preflight:
        status("preflight")
        commands = [
            [python, "-m", "pytest", "-q", "tests/test_exp156_differential.py"],
            [python, "scripts/generate_exp156_scenario_manifest.py"],
            [python, "scripts/audit_exp156_action_coverage.py"],
        ]
        for command in commands:
            if run(command, log=launcher_log) != 0:
                status("preflight_failed", command=command)
                raise SystemExit(2)

    if not args.skip_cpu_smoke:
        status("cpu_smoke")
        command = [
            python,
            "scripts/train_skrl_mappo.py",
            "--config", "configs/experiment/exp156_smoke.yaml",
            "--device", "cpu",
            "--num-envs", "2",
            "--rollout-steps", "16",
            "--timesteps", "32",
            "--actor-architecture", "multiscale_n0_mlp",
            "--output-layout", "run",
            "--run-name", "cpu_n0_2env_32step",
        ]
        if run(command, log=launcher_log) != 0:
            status("cpu_smoke_failed")
            raise SystemExit(2)

    if not args.skip_cuda_smoke:
        status("cuda_smoke")
        for architecture in SMOKE_CANDIDATES:
            short = SHORT_NAMES[architecture]
            command = [
                python,
                "scripts/train_skrl_mappo.py",
                "--config", "configs/experiment/exp156_smoke.yaml",
                "--device", args.device,
                "--num-envs", "256",
                "--rollout-steps", "16",
                "--timesteps", "32",
                "--actor-architecture", architecture,
                "--output-layout", "run",
                "--run-name", f"cuda_{short}_256env_32step",
            ]
            if run(command, log=launcher_log) != 0:
                status("cuda_smoke_failed", architecture=architecture)
                raise SystemExit(2)

    records = []
    # N2 remains in the CUDA engineering smoke above, but its full training was
    # cancelled after N0/N1 both exhibited a success-rate floor. Keeping it out
    # of this loop prevents an accidental 39.3M-interaction relaunch.
    for architecture in FULL_TRAIN_CANDIDATES:
        short = SHORT_NAMES[architecture]
        run_name = f"{short}_seed23_full_2400iter"
        run_dir = ROOT / "outputs/runs/exp156_differential_multiscale_ablation" / run_name
        status("training", architecture=architecture, run=run_name, completed=records)
        train = [
            python,
            "scripts/train_skrl_mappo.py",
            "--config", args.config,
            "--device", args.device,
            "--num-envs", "256",
            "--rollout-steps", "64",
            "--timesteps", "153600",
            "--seed", "23",
            "--actor-architecture", architecture,
            "--output-layout", "run",
            "--run-name", run_name,
            "--selection-gate", "strict",
        ]
        returncode = run(train, log=launcher_log)
        record = {
            "architecture": architecture,
            "run": run_name,
            "train_returncode": returncode,
            "parameter_count": PARAMETER_COUNTS[architecture],
        }
        records.append(record)
        if returncode != 0:
            record["status"] = "training_error"
            status("training_failed", record=record, completed=records)
            raise SystemExit(2)

        status("paired_evaluation", architecture=architecture, completed=records)
        evaluate = [
            python,
            "scripts/evaluate_exp156_paired.py",
            "--run-dir", str(run_dir.relative_to(ROOT)),
            "--device", args.device,
        ]
        record["evaluation_returncode"] = run(evaluate, log=launcher_log)
        report_path = run_dir / "metrics/paired_strict_acceptance.json"
        if record["evaluation_returncode"] != 0 or not report_path.is_file():
            record["status"] = "evaluation_error"
            status("evaluation_failed", record=record, completed=records)
            raise SystemExit(2)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        record.update(
            status="evaluated",
            strict_cells_passed=sum(bool(cell["passed"]) for cell in report["cells"]),
            all_cells_passed=bool(report["passed"]),
            aggregate=aggregate(report),
            report=str(report_path.relative_to(ROOT)),
        )

    def score(record: dict) -> float:
        values = record["aggregate"]
        return (
            values["success"]
            - values["collision"]
            - 0.5 * values["timeout"]
            - 0.5 * values["dmax_ratio"]
        )

    ranked = sorted(
        records,
        key=lambda item: (
            -item["strict_cells_passed"],
            -score(item),
            item["parameter_count"],
        ),
    )
    if (
        len(ranked) > 1
        and ranked[0]["strict_cells_passed"] == ranked[1]["strict_cells_passed"]
        and abs(score(ranked[0]) - score(ranked[1])) < 0.02
    ):
        comparable = [
            item
            for item in ranked
            if item["strict_cells_passed"] == ranked[0]["strict_cells_passed"]
            and abs(score(item) - score(ranked[0])) < 0.02
        ]
        winner = min(comparable, key=lambda item: item["parameter_count"])
        reason = "score difference below 0.02; selected fewer parameters"
    else:
        winner = ranked[0]
        reason = "strict cells, then preregistered aggregate score"

    winner_report = json.loads((ROOT / winner["report"]).read_text(encoding="utf-8"))
    paired = {}
    for record in records:
        if record is winner:
            continue
        other_report = json.loads((ROOT / record["report"]).read_text(encoding="utf-8"))
        paired[record["architecture"]] = {
            metric: paired_bootstrap_difference(
                cell_episode_values(winner_report, metric),
                cell_episode_values(other_report, metric),
                seed=156,
            )
            for metric in ("success", "collision", "timeout", "dmax_ratio")
        }

    result = {
        "status": "completed",
        "scenario_manifest": (
            "outputs/runs/exp156_differential_multiscale_ablation/_suite/"
            "scenario_manifest.json"
        ),
        "records": records,
        "winner": winner["architecture"],
        "selection_reason": reason,
        "paired_bootstrap_winner_minus_candidate": paired,
    }
    result_path = suite_dir / "metrics/architecture_comparison.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    status("completed", result=str(result_path.relative_to(ROOT)), winner=result["winner"])


if __name__ == "__main__":
    main()
