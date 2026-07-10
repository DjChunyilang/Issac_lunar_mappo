#!/usr/bin/env python
"""Evaluate proxy checkpoints across multiple deterministic eval seeds."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from _common import ROOT
from evaluate_proxy_policy import evaluate_checkpoint, proxy_acceptance


MetricEvaluator = Callable[..., dict]


def _resolve(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {value}") from exc


def _resolve_checkpoint(run_dir: Path, checkpoint: str | Path) -> Path:
    candidate = Path(checkpoint)
    if candidate.is_absolute():
        return candidate
    root_relative = ROOT / candidate
    if root_relative.exists():
        return root_relative
    return run_dir / "checkpoints" / candidate


def _clear_device_cache() -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _stats(rows: list[dict], key: str) -> dict:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {f"{key}_mean": None, f"{key}_min": None, f"{key}_max": None}
    return {
        f"{key}_mean": statistics.fmean(values),
        f"{key}_min": min(values),
        f"{key}_max": max(values),
    }


def summarize_rows(rows: list[dict], checkpoints: list[str]) -> list[dict]:
    summary = []
    for checkpoint in checkpoints:
        subset = [row for row in rows if row["checkpoint"] == checkpoint]
        item = {"checkpoint": checkpoint, "n": len(subset)}
        for key in (
            "dmax_reduction_ratio",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "mean_done_step",
            "filter_applied_fraction",
            "filter_collision_override_fraction",
            "control_safety_applied_fraction",
        ):
            item.update(_stats(subset, key))
        timeout_counts = [
            int(row["timeout_count"])
            for row in subset
            if row.get("timeout_count") is not None
        ]
        item["timeout_count_mean"] = statistics.fmean(timeout_counts) if timeout_counts else None
        item["timeout_count_min"] = min(timeout_counts) if timeout_counts else None
        item["timeout_count_max"] = max(timeout_counts) if timeout_counts else None
        item["strict_pass_count"] = sum(1 for row in subset if row.get("passed"))
        item["timeout_zero_count"] = sum(1 for row in subset if row.get("timeout_rate") == 0.0)
        summary.append(item)
    return summary


def run_seed_sweep(
    *,
    config: str | Path,
    run_dir: str | Path,
    checkpoints: list[str],
    seeds: list[int],
    device: str | None = None,
    num_envs: int = 1024,
    steps: int = 320,
    output_dir: str | Path | None = None,
    evaluator: MetricEvaluator = evaluate_checkpoint,
) -> dict:
    config_path = _resolve(config)
    run_path = _resolve(run_dir)
    out_path = _resolve(output_dir) if output_dir is not None else run_path / "metrics" / "checkpoint_seed_sweep"
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []
    checkpoint_paths = [_resolve_checkpoint(run_path, checkpoint) for checkpoint in checkpoints]
    checkpoint_labels = [checkpoint.name for checkpoint in checkpoint_paths]
    for checkpoint_path, checkpoint_label in zip(checkpoint_paths, checkpoint_labels, strict=True):
        for seed in seeds:
            eval_output = out_path / f"{checkpoint_path.stem}_seed{seed}_eval.json"
            print(f"evaluating {checkpoint_label} seed={seed}", flush=True)
            metrics = evaluator(
                config_path,
                checkpoint_path,
                device=device,
                num_envs=num_envs,
                steps=steps,
                seed=seed,
                output=eval_output,
            )
            if not eval_output.exists():
                eval_output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
            gate = proxy_acceptance(metrics)
            rows.append(
                {
                    "checkpoint": checkpoint_label,
                    "checkpoint_path": _relative(checkpoint_path),
                    "seed": seed,
                    "artifact": _relative(eval_output),
                    "dmax_reduction_ratio": metrics["dmax_reduction_ratio"],
                    "success_rate": metrics["success_rate"],
                    "collision_rate": metrics["collision_rate"],
                    "timeout_rate": metrics["timeout_rate"],
                    "timeout_count": metrics.get("timeout_episode_metrics", {}).get("count"),
                    "mean_done_step": metrics.get("mean_done_step"),
                    "filter_applied_fraction": metrics.get("filter_applied_fraction"),
                    "filter_collision_override_fraction": metrics.get("filter_collision_override_fraction"),
                    "control_safety_applied_fraction": metrics.get("control_safety_applied_fraction"),
                    "passed": gate["passed"],
                    "checks": gate["checks"],
                }
            )
            _clear_device_cache()

    summary_path = out_path / "summary.json"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": _relative(run_path),
        "config": _relative(config_path),
        "artifact": _relative(summary_path),
        "num_envs": num_envs,
        "steps": steps,
        "device": device,
        "seeds": seeds,
        "checkpoints": checkpoint_labels,
        "rows": rows,
        "summary": summarize_rows(rows, checkpoint_labels),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint path or filename under <run-dir>/checkpoints. May be repeated.",
    )
    parser.add_argument("--seeds", type=parse_int_list, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_seed_sweep(
        config=args.config,
        run_dir=args.run_dir,
        checkpoints=args.checkpoint,
        seeds=args.seeds,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"summary_path {result['artifact']}")


if __name__ == "__main__":
    main()
