#!/usr/bin/env python3
"""Gate and launch the serial exp159 analytical PRD workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT
from summarize_exp159_three_seed import summarize_phase


EXPERIMENT = "exp159_analytical_prd"
SUITE = ROOT / "outputs/runs" / EXPERIMENT / "_suite"


def _status(phase: str, **extra: object) -> None:
    SUITE.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "experiment": EXPERIMENT,
        "phase": phase,
        "updated_at": timestamp,
        **extra,
    }
    (SUITE / "suite_status.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (SUITE / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": timestamp,
                "experiment": EXPERIMENT,
                "run": "_suite",
                "producer": "scripts/run_exp159_prd_validation.py",
                "summary": payload,
                "artifacts": {
                    "suite_status": str((SUITE / "suite_status.json").relative_to(ROOT)),
                    "engineering_smoke": str(
                        (SUITE / "metrics/engineering_smoke.json").relative_to(ROOT)
                    ),
                    "h1_offline_gate": (
                        "outputs/runs/exp159_analytical_prd/offline_h1_audit/"
                        "metrics/offline_gate.json"
                    ),
                    "strict_offline_gate": (
                        "outputs/runs/exp159_analytical_prd/offline_strict_audit/"
                        "metrics/offline_gate.json"
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _run(command: list[str], *, label: str, timeout_s: int = 48 * 3600) -> None:
    log = SUITE / "logs/launcher.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    _status(label, command=command)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        )
        started = time.monotonic()
        heartbeat = started
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > timeout_s:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    _status("hard_timeout", label=label, elapsed_s=elapsed)
                    raise SystemExit(124)
                if time.monotonic() - heartbeat >= 60.0:
                    _status(label, command=command, pid=process.pid, elapsed_s=elapsed)
                    heartbeat = time.monotonic()
                time.sleep(5.0)
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            _status("interrupted", label=label, pid=process.pid)
            raise
        returncode = int(process.returncode or 0)
    if returncode:
        _status("failed", label=label, returncode=returncode)
        raise SystemExit(returncode)


def _load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Required exp159 gate is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(mode: str) -> Path:
    return ROOT / f"outputs/runs/{EXPERIMENT}/offline_{mode}_audit/metrics/offline_gate.json"


def _pair(phase: str, seed: int) -> Path:
    return SUITE / "metrics" / f"{phase}_seed{seed}_pair.json"


def _ensure_smoke(python: str, device: str) -> dict:
    specs = (
        (
            "cpu_prd",
            "configs/experiment/exp159_prd_smoke.yaml",
            "cpu",
            "2",
            "4",
            "8",
            "exp159_prd_smoke",
            "cpu_prd_2env_8step",
        ),
        (
            "cuda_gae",
            "configs/experiment/exp159_gae_smoke.yaml",
            device,
            "256",
            "64",
            "64",
            "exp159_gae_smoke",
            "cuda_gae_256env_64step",
        ),
        (
            "cuda_prd",
            "configs/experiment/exp159_prd_smoke.yaml",
            device,
            "256",
            "64",
            "64",
            "exp159_prd_smoke",
            "cuda_prd_256env_64step",
        ),
    )
    summaries = {}
    for label, config, run_device, envs, rollout, steps, experiment, run in specs:
        summary = ROOT / f"outputs/runs/{experiment}/{run}/metrics/summary.json"
        if not summary.is_file():
            _run(
                [
                    python,
                    "scripts/train_skrl_mappo.py",
                    "--config",
                    config,
                    "--device",
                    run_device,
                    "--num-envs",
                    envs,
                    "--rollout-steps",
                    rollout,
                    "--timesteps",
                    steps,
                    "--output-layout",
                    "run",
                    "--run-name",
                    run,
                    "--selection-gate",
                    "final",
                ],
                label=f"{label}_smoke",
            )
        summaries[label] = _load(summary)
    last_line = (
        ROOT
        / "outputs/runs/exp159_prd_smoke/cuda_prd_256env_64step/"
        "metrics/train_metrics.jsonl"
    ).read_text(encoding="utf-8").splitlines()[-1]
    metrics = json.loads(last_line)
    gae_diag = summaries["cuda_gae"]["training_diagnostics"]
    prd_diag = summaries["cuda_prd"]["training_diagnostics"]
    throughput = float(summaries["cuda_gae"]["wall_time_s"]) / max(
        float(summaries["cuda_prd"]["wall_time_s"]), 1.0e-8
    )
    checks = {
        "cpu_prd_finite": bool(
            summaries["cpu_prd"]["training_diagnostics"]["policy_parameters_finite"]
        ),
        "cuda_prd_finite": bool(prd_diag["policy_parameters_finite"]),
        "initial_policy_hash_match": gae_diag["initial_policy_sha256"]
        == prd_diag["initial_policy_sha256"],
        "initial_critic_hash_match": gae_diag["initial_critic_sha256"]
        == prd_diag["initial_critic_sha256"],
        "peak_cuda_memory_le_8192_mb": float(metrics["peak_cuda_memory_mb"]) <= 8192.0,
        "throughput_ratio_ge_0_90": throughput >= 0.90,
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "throughput_ratio": throughput,
        "peak_cuda_memory_mb": float(metrics["peak_cuda_memory_mb"]),
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = SUITE / "metrics/engineering_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _train_pair(phase: str, seed: int, device: str, python: str) -> None:
    configs = (
        ("gae", f"configs/experiment/exp159_{phase}_gae.yaml"),
        ("prd", f"configs/experiment/exp159_{phase}_prd.yaml"),
    )
    runs = {arm: f"{phase}_{arm}_seed{seed}_full_2400iter" for arm, _ in configs}
    for arm, config in configs:
        run_dir = ROOT / "outputs/runs" / EXPERIMENT / runs[arm]
        if run_dir.exists():
            raise SystemExit(f"Refusing to overwrite or retry existing run: {run_dir}")
        _run(
            [
                python,
                "scripts/train_skrl_mappo.py",
                "--config",
                config,
                "--device",
                device,
                "--num-envs",
                "256",
                "--rollout-steps",
                "64",
                "--timesteps",
                "153600",
                "--seed",
                str(seed),
                "--actor-architecture",
                "multiscale_n1_cnn",
                "--output-layout",
                "run",
                "--run-name",
                runs[arm],
                "--selection-gate",
                "final",
            ],
            label=f"{phase}_{arm}_seed{seed}_training",
        )
        _run(
            [
                python,
                "scripts/evaluate_exp156_paired.py",
                "--run-dir",
                str(run_dir.relative_to(ROOT)),
                "--device",
                device,
            ],
            label=f"{phase}_{arm}_seed{seed}_evaluation",
        )
        if arm == "prd":
            _run(
                [
                    python,
                    "scripts/audit_exp159_prd.py",
                    "--mode",
                    phase,
                    "--config",
                    str((run_dir / "config/experiment.yaml").relative_to(ROOT)),
                    "--checkpoint",
                    str((run_dir / "checkpoints/best.pt").relative_to(ROOT)),
                    "--run-dir",
                    str((run_dir / "prd_checkpoint_validation").relative_to(ROOT)),
                    "--device",
                    device,
                ],
                label=f"{phase}_prd_seed{seed}_credit_validation",
            )
    _run(
        [
            python,
            "scripts/compare_exp159_pair.py",
            "--gae-run",
            f"outputs/runs/{EXPERIMENT}/{runs['gae']}",
            "--prd-run",
            f"outputs/runs/{EXPERIMENT}/{runs['prd']}",
            "--phase",
            phase,
            "--seed",
            str(seed),
        ],
        label=f"{phase}_seed{seed}_comparison",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("offline", "h1", "strict"), default="offline")
    parser.add_argument("--seed", type=int, choices=(23, 31, 47), default=23)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    python = str(ROOT / ".venv_isaaclab/bin/python")
    if not args.skip_tests:
        _run(
            [python, "-m", "pytest", "-q", "tests/test_exp159_prd.py"],
            label="engineering_tests",
        )
    if args.phase == "offline":
        if not args.skip_smoke:
            smoke = _ensure_smoke(python, args.device)
            if not smoke["passed"]:
                _status("engineering_smoke_failed", report=smoke)
                return
        _run(
            [python, "scripts/audit_exp159_prd.py", "--mode", "h1", "--device", args.device],
            label="offline_h1_audit",
        )
        if not bool(_load(_gate("h1"))["passed"]):
            _status("offline_h1_gate_failed", next_phase=None)
            return
        _run(
            [python, "scripts/audit_exp159_prd.py", "--mode", "strict", "--device", args.device],
            label="offline_strict_audit",
        )
        _status(
            "offline_completed",
            h1_passed=True,
            strict_passed=bool(_load(_gate("strict"))["passed"]),
            next_phase="h1_seed23",
        )
        return

    if not bool(_load(_gate("h1"))["passed"]):
        raise SystemExit("H1 offline PRD gate failed; training is forbidden")
    if args.phase == "h1" and args.seed != 23:
        if not bool(_load(_pair("h1", 23))["passed"]):
            raise SystemExit("H1 seed23 pair did not pass")
        if args.seed == 47 and not _pair("h1", 31).is_file():
            raise SystemExit("H1 seed31 must complete before seed47")
    if args.phase == "strict":
        if not bool(_load(_gate("strict"))["passed"]):
            raise SystemExit("Strict offline PRD gate failed")
        summary_path = SUITE / "metrics/h1_three_seed_summary.json"
        if not summary_path.is_file():
            summary = summarize_phase("h1")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if not bool(_load(summary_path)["passed"]):
            raise SystemExit("H1 three-seed gate did not pass")
        if args.seed != 23 and not bool(_load(_pair("strict", 23))["passed"]):
            raise SystemExit("Strict seed23 pair did not pass")
        if args.seed == 47 and not _pair("strict", 31).is_file():
            raise SystemExit("Strict seed31 must complete before seed47")

    _train_pair(args.phase, args.seed, args.device, python)
    report = _load(_pair(args.phase, args.seed))
    if args.seed == 23 and not bool(report["passed"]):
        _status(f"{args.phase}_seed23_gate_failed", next_phase=None)
        return
    if args.seed == 47 and _pair(args.phase, 31).is_file():
        summary = summarize_phase(args.phase)
        output = SUITE / "metrics" / f"{args.phase}_three_seed_summary.json"
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _status(f"{args.phase}_three_seed_completed", passed=summary["passed"])
    else:
        _status(f"{args.phase}_seed{args.seed}_completed", passed=report["passed"])


if __name__ == "__main__":
    main()
