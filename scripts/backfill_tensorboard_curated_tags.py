#!/usr/bin/env python
"""Backfill curated TensorBoard scalar groups from saved proxy training logs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


def _load_summary_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - depends on optional tensorboard package
        raise SystemExit(f"torch.utils.tensorboard is unavailable: {exc}") from exc
    return SummaryWriter


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _add_scalar(writer, tag: str, value: object, step: int) -> int:
    if not _is_number(value):
        return 0
    writer.add_scalar(tag, float(value), step)
    return 1


def _write_custom_scalar_layout(writer) -> None:
    layout = {
        "00_overview": {
            "Reward": ["Multiline", ["00_overview/eval_reward", "00_overview/rollout_reward"]],
            "Task": ["Multiline", ["00_overview/success_rate", "00_overview/dmax_ratio"]],
            "Safety": ["Multiline", ["00_overview/collision_rate", "00_overview/timeout_rate"]],
        },
        "01_ppo_health": {
            "Optimization": ["Multiline", ["01_ppo_health/policy_loss", "01_ppo_health/value_loss"]],
            "Trust Region": ["Multiline", ["01_ppo_health/approx_kl", "01_ppo_health/clip_fraction"]],
            "Exploration": ["Multiline", ["01_ppo_health/entropy", "01_ppo_health/reference_policy_loss"]],
            "Value Fit": ["Multiline", ["01_ppo_health/explained_variance"]],
        },
        "02_task_detail": {
            "Gathering": ["Multiline", ["02_task_detail/final_dmax", "02_task_detail/final_dispersion"]],
            "Spacing": [
                "Multiline",
                [
                    "02_task_detail/mean_nearest_distance",
                    "02_task_detail/min_nearest_distance",
                    "02_task_detail/near_violation_rate",
                ],
            ],
        },
    }
    writer.add_custom_scalars(layout)


def _load_train_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _run_dirs(log_root: Path, include_smoke: bool) -> list[Path]:
    run_dirs = []
    for path in log_root.glob("*/eval_metrics.json"):
        run_dir = path.parent
        if not (run_dir / "train_metrics.jsonl").exists():
            continue
        if not include_smoke and "smoke" in run_dir.name:
            continue
        run_dirs.append(run_dir)
    return sorted(run_dirs)


def _prepare_output_dir(run_dir: Path, output_subdir: str, overwrite: bool) -> Path:
    output_dir = run_dir / output_subdir
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _backfill_run(run_dir: Path, output_subdir: str, overwrite: bool) -> dict:
    eval_path = run_dir / "eval_metrics.json"
    train_path = run_dir / "train_metrics.jsonl"
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    evaluations = payload.get("evaluations", [])
    train_rows = _load_train_rows(train_path)
    ppo_rows = [row for row in train_rows if row.get("phase") == "ppo"]

    summary_writer = _load_summary_writer()
    output_dir = _prepare_output_dir(run_dir, output_subdir, overwrite)
    writer = summary_writer(str(output_dir))
    _write_custom_scalar_layout(writer)

    counts: dict[str, int] = {}

    def add(tag: str, value: object, step: int) -> None:
        counts[tag] = counts.get(tag, 0) + _add_scalar(writer, tag, value, step)

    for index, metrics in enumerate(evaluations):
        step = int(metrics.get("update", index))
        add("00_overview/eval_reward", metrics.get("mean_reward"), step)
        add("00_overview/success_rate", metrics.get("success_rate"), step)
        add("00_overview/dmax_ratio", metrics.get("dmax_reduction_ratio"), step)
        add("00_overview/collision_rate", metrics.get("collision_rate"), step)
        add("00_overview/timeout_rate", metrics.get("timeout_rate"), step)
        add("02_task_detail/final_dmax", metrics.get("final_dmax"), step)
        add("02_task_detail/final_dispersion", metrics.get("final_dispersion"), step)
        add("02_task_detail/mean_nearest_distance", metrics.get("mean_nearest_distance"), step)
        add("02_task_detail/min_nearest_distance", metrics.get("min_nearest_distance"), step)
        add("02_task_detail/near_violation_rate", metrics.get("near_violation_rate"), step)

    for row in ppo_rows:
        step = int(row["update"])
        add("00_overview/rollout_reward", row.get("rollout_reward"), step)
        for key in ("policy_loss", "value_loss", "entropy", "reference_policy_loss"):
            add(f"01_ppo_health/{key}", row.get(key), step)

    writer.flush()
    writer.close()

    unavailable = ["approx_kl", "clip_fraction", "explained_variance"]
    return {
        "run": run_dir.name,
        "output_dir": str(output_dir),
        "evaluations": len(evaluations),
        "ppo_updates": len(ppo_rows),
        "scalar_counts": dict(sorted(counts.items())),
        "unavailable_ppo_health": unavailable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="outputs/logs/exp_006_ppo_selected")
    parser.add_argument("--output-subdir", default="tensorboard_curated")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-smoke", action="store_true", help="Also backfill short smoke-test runs.")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    runs = _run_dirs(log_root, include_smoke=args.include_smoke)
    if not runs:
        raise SystemExit(f"No run directories with eval_metrics.json and train_metrics.jsonl found under {log_root}")

    summaries = [_backfill_run(run_dir, args.output_subdir, args.overwrite) for run_dir in runs]
    print(json.dumps({"status": "ok", "log_root": str(log_root), "runs": summaries}, indent=2))


if __name__ == "__main__":
    main()
