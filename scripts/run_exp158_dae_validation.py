#!/usr/bin/env python3
"""Gate and launch the serial exp158 GAE/DAE validation workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT
from summarize_exp158_three_seed import summarize_phase as _hierarchical_summary


EXPERIMENT = "exp158_dae_validation"
SUITE = ROOT / "outputs/runs" / EXPERIMENT / "_suite"


def _write_status(phase: str, **extra: object) -> None:
    SUITE.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    status = {
        "experiment": EXPERIMENT,
        "phase": phase,
        "updated_at": timestamp,
        **extra,
    }
    (SUITE / "suite_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    (SUITE / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": timestamp,
                "experiment": EXPERIMENT,
                "run": "_suite",
                "producer": "scripts/run_exp158_dae_validation.py",
                "summary": status,
                "artifacts": {
                    "suite_status": str((SUITE / "suite_status.json").relative_to(ROOT)),
                    "engineering_smoke": str(
                        (SUITE / "metrics/engineering_smoke.json").relative_to(ROOT)
                    ),
                    "offline_gate": (
                        "outputs/runs/exp158_dae_validation/offline_credit_audit/"
                        "metrics/offline_gate.json"
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _run(command: list[str], *, label: str, hard_timeout_s: int = 48 * 3600) -> None:
    log = SUITE / "logs" / "launcher.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    _write_status(label, command=command)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n"
        )
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        )
        started = time.monotonic()
        last_heartbeat = started
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > hard_timeout_s:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    _write_status(
                        "hard_timeout",
                        label=label,
                        command=command,
                        pid=process.pid,
                        elapsed_s=elapsed,
                        hard_timeout_s=hard_timeout_s,
                    )
                    raise SystemExit(124)
                if time.monotonic() - last_heartbeat >= 60.0:
                    _write_status(
                        label,
                        command=command,
                        pid=process.pid,
                        elapsed_s=elapsed,
                        hard_timeout_s=hard_timeout_s,
                    )
                    last_heartbeat = time.monotonic()
                time.sleep(5.0)
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            _write_status(
                "interrupted",
                label=label,
                command=command,
                pid=process.pid,
            )
            raise
        returncode = int(process.returncode or 0)
    if returncode != 0:
        _write_status("failed", label=label, command=command, returncode=returncode)
        raise SystemExit(returncode)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Required exp158 gate is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _offline_gate_passed() -> bool:
    path = ROOT / "outputs/runs/exp158_dae_validation/offline_credit_audit/metrics/offline_gate.json"
    return bool(_load_json(path).get("passed", False))


def _ensure_engineering_smoke(*, python: str, device: str) -> dict:
    smoke_specs = (
        (
            "cpu_dae",
            "configs/experiment/exp158_dae_smoke.yaml",
            "cpu",
            "2",
            "4",
            "8",
            "exp158_dae_smoke",
            "cpu_dae_2env_8step_cached",
        ),
        (
            "cuda_gae",
            "configs/experiment/exp158_gae_smoke.yaml",
            device,
            "256",
            "64",
            "64",
            "exp158_gae_smoke",
            "cuda_gae_256env_64step",
        ),
        (
            "cuda_dae",
            "configs/experiment/exp158_dae_smoke.yaml",
            device,
            "256",
            "64",
            "64",
            "exp158_dae_smoke",
            "cuda_dae_256env_64step",
        ),
    )
    summaries = {}
    for label, config, run_device, num_envs, rollout, timesteps, experiment, run in smoke_specs:
        summary_path = ROOT / "outputs/runs" / experiment / run / "metrics/summary.json"
        if not summary_path.is_file():
            _run(
                [
                    python,
                    "scripts/train_skrl_mappo.py",
                    "--config",
                    config,
                    "--device",
                    run_device,
                    "--num-envs",
                    num_envs,
                    "--rollout-steps",
                    rollout,
                    "--timesteps",
                    timesteps,
                    "--output-layout",
                    "run",
                    "--run-name",
                    run,
                    "--selection-gate",
                    "strict",
                ],
                label=f"{label}_smoke",
            )
        summaries[label] = _load_json(summary_path)
    dae_metrics_path = (
        ROOT
        / "outputs/runs/exp158_dae_smoke/cuda_dae_256env_64step/"
        "metrics/train_metrics.jsonl"
    )
    last_metrics = json.loads(
        dae_metrics_path.read_text(encoding="utf-8").splitlines()[-1]
    )
    gae_diag = summaries["cuda_gae"]["training_diagnostics"]
    dae_diag = summaries["cuda_dae"]["training_diagnostics"]
    throughput_ratio = float(summaries["cuda_gae"]["wall_time_s"]) / max(
        float(summaries["cuda_dae"]["wall_time_s"]), 1.0e-8
    )
    checks = {
        "cpu_dae_finite": bool(
            summaries["cpu_dae"]["training_diagnostics"]["policy_parameters_finite"]
        )
        and bool(
            summaries["cpu_dae"]["training_diagnostics"][
                "reward_model_parameters_finite"
            ]
        ),
        "cuda_dae_finite": bool(dae_diag["policy_parameters_finite"])
        and bool(dae_diag["reward_model_parameters_finite"]),
        "initial_policy_hash_match": gae_diag["initial_policy_sha256"]
        == dae_diag["initial_policy_sha256"],
        "initial_critic_hash_match": gae_diag["initial_critic_sha256"]
        == dae_diag["initial_critic_sha256"],
        "peak_cuda_memory_le_8192_mb": float(last_metrics["peak_cuda_memory_mb"])
        <= 8192.0,
        "throughput_ratio_ge_0_60": throughput_ratio >= 0.60,
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "throughput_ratio_dae_overhead_adjusted": throughput_ratio,
        "peak_cuda_memory_mb": float(last_metrics["peak_cuda_memory_mb"]),
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = SUITE / "metrics/engineering_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _pair_path(phase: str, seed: int) -> Path:
    return SUITE / "metrics" / f"{phase}_seed{seed}_pair.json"


def _summarize_phase(phase: str) -> dict:
    summary = _hierarchical_summary(phase, samples=10_000, seed=158)
    output = SUITE / "metrics" / f"{phase}_three_seed_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _train_pair(*, phase: str, seed: int, device: str, python: str) -> None:
    if phase == "h1":
        gae_config = "configs/experiment/exp158_h1_gae.yaml"
        dae_config = "configs/experiment/exp158_h1_dae.yaml"
    else:
        gae_config = "configs/experiment/exp158_strict_gae.yaml"
        dae_config = "configs/experiment/exp158_strict_dae.yaml"
    run_names = {
        "gae": f"{phase}_gae_seed{seed}_full_2400iter",
        "dae": f"{phase}_dae_seed{seed}_full_2400iter",
    }
    for arm, config in (("gae", gae_config), ("dae", dae_config)):
        run_dir = ROOT / "outputs/runs" / EXPERIMENT / run_names[arm]
        if run_dir.exists():
            raise SystemExit(
                f"Refusing to overwrite or silently retry existing exp158 run: {run_dir}"
            )
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
                run_names[arm],
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
            label=f"{phase}_{arm}_seed{seed}_paired_evaluation",
        )
        if arm == "dae":
            _run(
                [
                    python,
                    "scripts/validate_exp158_reward_model.py",
                    "--run-dir",
                    str(run_dir.relative_to(ROOT)),
                    "--device",
                    device,
                ],
                label=f"{phase}_dae_seed{seed}_reward_validation",
            )
    _run(
        [
            python,
            "scripts/compare_exp158_pair.py",
            "--gae-run",
            f"outputs/runs/{EXPERIMENT}/{run_names['gae']}",
            "--dae-run",
            f"outputs/runs/{EXPERIMENT}/{run_names['dae']}",
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
            [
                python,
                "-m",
                "pytest",
                "-q",
                "tests/test_exp158_dae.py",
            ],
            label="engineering_tests",
        )
    if args.phase == "offline":
        if not args.skip_smoke:
            smoke = _ensure_engineering_smoke(python=python, device=args.device)
            if not smoke["passed"]:
                _write_status("engineering_smoke_failed", report=smoke)
                return
        _run(
            [python, "scripts/audit_exp158_dae.py", "--device", args.device],
            label="offline_credit_audit",
        )
        passed = _offline_gate_passed()
        _write_status(
            "offline_gate_passed" if passed else "offline_gate_failed",
            next_phase="h1_seed23" if passed else None,
        )
        return

    if not _offline_gate_passed():
        raise SystemExit("Offline DAE gate did not pass; refusing to launch training")
    if args.phase == "h1" and args.seed != 23:
        if not bool(_load_json(_pair_path("h1", 23)).get("passed", False)):
            raise SystemExit("H1 seed23 pair did not pass; later seeds are gated")
        if args.seed == 47 and not _pair_path("h1", 31).is_file():
            raise SystemExit("H1 seed31 must complete before seed47")
    if args.phase == "strict":
        h1_summary_path = SUITE / "metrics/h1_three_seed_summary.json"
        if not h1_summary_path.is_file():
            _summarize_phase("h1")
        if not bool(_load_json(h1_summary_path).get("passed", False)):
            raise SystemExit("H1 three-seed gate did not pass; strict phase is gated")
        if args.seed != 23 and not bool(
            _load_json(_pair_path("strict", 23)).get("passed", False)
        ):
            raise SystemExit("Strict seed23 pair did not pass; later seeds are gated")
        if args.seed == 47 and not _pair_path("strict", 31).is_file():
            raise SystemExit("Strict seed31 must complete before seed47")

    _train_pair(phase=args.phase, seed=args.seed, device=args.device, python=python)
    pair = _load_json(_pair_path(args.phase, args.seed))
    if args.seed == 23 and not bool(pair.get("passed", False)):
        _write_status(
            f"{args.phase}_seed23_gate_failed",
            pair=str(_pair_path(args.phase, args.seed).relative_to(ROOT)),
        )
        return
    if args.seed == 47 and _pair_path(args.phase, 31).is_file():
        summary = _summarize_phase(args.phase)
        _write_status(
            f"{args.phase}_three_seed_completed",
            passed=summary["passed"],
            summary=str(
                (SUITE / "metrics" / f"{args.phase}_three_seed_summary.json").relative_to(ROOT)
            ),
        )
    else:
        _write_status(
            f"{args.phase}_seed{args.seed}_completed",
            passed=bool(pair.get("passed", False)),
        )


if __name__ == "__main__":
    main()
