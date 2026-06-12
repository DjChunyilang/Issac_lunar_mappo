#!/usr/bin/env python
"""Run CUDA-only SKRL MAPPO validation jobs and summarize telemetry."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment" / "exp_cuda_contract.yaml"
SUMMARY_PATH = ROOT / "outputs" / "runs" / "cuda_training_validation_summary.json"
REQUIRED_METADATA = {
    "training_semantics",
    "shared_actor",
    "centralized_critic",
    "shared_value",
    "observation_schema_version",
    "device",
}
RUNS = (
    ("cuda_contract", 32),
    ("cuda_short", 512),
    ("cuda_signal", 5000),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Checkpoint was not generated: {path}")
    payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata", {})
    missing = sorted(key for key in REQUIRED_METADATA if key not in metadata)
    if missing:
        raise RuntimeError(f"Checkpoint metadata missing required field(s): {', '.join(missing)}")
    return metadata


def _metric_value(rows: list[dict[str, Any]], key: str, *, first: bool) -> Any:
    if not rows:
        return None
    row = rows[0] if first else rows[-1]
    return row.get(key)


def _summarize_run(
    *,
    run_name: str,
    timesteps: int,
    process_wall_time_s: float,
    metrics_path: Path,
    checkpoint_path: Path,
    new_metrics: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    nan_detected = any(bool(row.get("nan_flag")) for row in new_metrics)
    return {
        "run_name": run_name,
        "device": metadata.get("device"),
        "timesteps": timesteps,
        "status": "ok",
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "wall_time_s": process_wall_time_s,
        "peak_cuda_memory_mb": None,
        "mean_pairwise_distance_start": _metric_value(
            new_metrics, "mean_pairwise_distance", first=True
        ),
        "mean_pairwise_distance_end": _metric_value(
            new_metrics, "mean_pairwise_distance", first=False
        ),
        "mean_oracle_distance_start": _metric_value(new_metrics, "mean_oracle_distance", first=True),
        "mean_oracle_distance_end": _metric_value(new_metrics, "mean_oracle_distance", first=False),
        "success_rate_final": _metric_value(new_metrics, "success_rate", first=False),
        "nan_detected": nan_detected,
        "error_message": None,
    }


def _run_training(run_name: str, timesteps: int) -> dict[str, Any]:
    metrics_path = ROOT / "outputs" / "runs" / "exp_cuda_contract" / "metrics.jsonl"
    previous_metric_count = len(_read_jsonl(metrics_path))
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_skrl_mappo.py"),
        "--config",
        str(CONFIG),
        "--device",
        "cuda",
        "--timesteps",
        str(timesteps),
    ]

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    wall_time_s = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"{run_name} failed with exit code {completed.returncode}.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    metrics = _read_jsonl(metrics_path)
    new_metrics = metrics[previous_metric_count:]
    if not metrics_path.exists() or not new_metrics:
        raise RuntimeError(f"{run_name} did not append telemetry metrics: {metrics_path}")

    checkpoint_path = Path(new_metrics[-1].get("checkpoint_path", "")).resolve()
    metadata = _load_checkpoint_metadata(checkpoint_path)
    if any(bool(row.get("nan_flag")) for row in new_metrics):
        raise RuntimeError(f"{run_name} reported nan_flag=true in {metrics_path}")

    return _summarize_run(
        run_name=run_name,
        timesteps=timesteps,
        process_wall_time_s=wall_time_s,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        new_metrics=new_metrics,
        metadata=metadata,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this validation stage.")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    for run_name, timesteps in RUNS:
        summaries.append(_run_training(run_name, timesteps))
    payload = {
        "status": "ok",
        "config": str(CONFIG),
        "runs": summaries,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
