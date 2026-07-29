#!/usr/bin/env python
"""Plot SKRL MAPPO training and candidate-eval curves for run directories."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_mpl_config_dir = Path(os.environ.get("MPLCONFIGDIR", f"/tmp/isaac_mappo_matplotlib_{os.getuid()}"))
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_mpl_config_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value


def _artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _update_run_manifest_artifact(run_dir: Path, key: str, path: Path) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    existing_artifacts = manifest.get("artifacts", {})
    if not isinstance(existing_artifacts, dict):
        return False

    artifacts = dict(existing_artifacts)
    artifacts[key] = _artifact_path(path)
    manifest["artifacts"] = artifacts
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()

    temp_path: Path | None = None
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            dir=manifest_path.parent,
        )
        temp_path = Path(temp_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temp_path, manifest_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return True


def _path_belongs_to_run(path: Path, run_dir: Path) -> bool:
    try:
        path.resolve().relative_to(run_dir.resolve())
    except ValueError:
        return False
    return True


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _read_candidate_evals(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    evaluations = payload.get("evaluations", []) if isinstance(payload, dict) else []
    return [item for item in evaluations if isinstance(item, dict)]


def _candidate_timestep_from_artifact_name(path: Path) -> int | None:
    match = re.search(r"(?:^|_)timestep_(\d+)_eval$", path.stem)
    return int(match.group(1)) if match is not None else None


def _read_run_candidate_evals(run_dir: Path) -> list[dict[str, Any]]:
    """Prefer independently re-evaluable candidate artifacts over a stale summary."""
    candidate_paths = sorted((run_dir / "metrics").glob("candidate_*_eval.json"))
    records: list[dict[str, Any]] = []
    for path in candidate_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        record = dict(payload)
        timestep = _candidate_timestep_from_artifact_name(path)
        if timestep is not None:
            record.setdefault("candidate_timestep", timestep)
        records.append(record)
    if records:
        return sorted(
            records,
            key=lambda record: int(record.get("candidate_timestep", 0)),
        )
    return _read_candidate_evals(run_dir / "metrics" / "eval_metrics.json")


def _record_value(record: dict[str, Any], key: str) -> Any:
    value: Any = record
    for field in key.split("."):
        if not isinstance(value, dict) or field not in value:
            return None
        value = value[field]
    return value


def _series(records: list[dict[str, Any]], key: str) -> list[float | None]:
    values: list[float | None] = []
    for record in records:
        value = _record_value(record, key)
        values.append(float(value) if isinstance(value, (int, float)) else None)
    return values


def _x_values(records: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for index, record in enumerate(records):
        value = record.get("timesteps", record.get("candidate_timestep", index + 1))
        values.append(float(value) if isinstance(value, (int, float)) else float(index + 1))
    return values


def _plot_if_present(
    ax,
    x: list[float],
    records: list[dict[str, Any]],
    keys: list[str],
    *,
    ylabel: str,
    title: str,
) -> None:
    plotted = False
    for key in keys:
        y = _series(records, key)
        if not y or all(value is None for value in y):
            continue
        ax.plot(x, y, marker="o" if len(x) <= 24 else None, linewidth=1.5, label=key)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("timesteps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)


def plot_training_curves(run_dir: Path, output: Path | None = None) -> Path | None:
    records = _read_jsonl(run_dir / "metrics" / "train_metrics.jsonl")
    if not records:
        return None
    output = output or run_dir / "figures" / "training_curves.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    x = _x_values(records)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    _plot_if_present(
        axes[0, 0],
        x,
        records,
        [
            "mean_reward",
            "reward_weighted_total",
            "progress_reward",
            "reward_contribution_flatness",
        ],
        ylabel="reward",
        title="Reward / progress",
    )
    _plot_if_present(
        axes[0, 1],
        x,
        records,
        ["success_rate", "safe_success_rate", "collision_done", "timeout_done"],
        ylabel="rate / count",
        title="Training rollout outcomes",
    )
    _plot_if_present(
        axes[1, 0],
        x,
        records,
        ["final_dmax", "final_dispersion", "final_nearest_neighbor_distance"],
        ylabel="m",
        title="Terminal geometry",
    )
    _plot_if_present(
        axes[1, 1],
        x,
        records,
        ["path_terrain_risk_mean", "action_std", "filter_applied_fraction", "control_safety_applied_fraction"],
        ylabel="value",
        title="Terrain / action / safety signals",
    )
    _plot_if_present(
        axes[2, 0],
        x,
        records,
        [
            "flatness_ok_rate",
            "gather_point_height_range_mean",
            "gather_point_max_slope_mean",
            "centroid_flatness_cost_mean",
            "centroid_flatness_activation_mean",
        ],
        ylabel="rate / m / gradient",
        title="Gather-site flatness / terrain",
    )
    _plot_if_present(
        axes[2, 1],
        x,
        records,
        ["oracle_search_feasible_rate", "oracle_search_objective_mean"],
        ylabel="rate / objective",
        title="Oracle search",
    )
    fig.suptitle(run_dir.name, fontsize=12)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    _update_run_manifest_artifact(run_dir, "training_curves", output)
    return output


def plot_candidate_eval_curves(run_dir: Path, output: Path | None = None) -> Path | None:
    records = _read_run_candidate_evals(run_dir)
    if not records:
        return None
    output = output or run_dir / "figures" / "candidate_eval_curves.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    x = _x_values(records)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    _plot_if_present(
        axes[0, 0],
        x,
        records,
        ["dmax_reduction_ratio", "success_rate", "collision_rate", "timeout_rate"],
        ylabel="rate",
        title="Strict-gate metrics",
    )
    axes[0, 0].axhline(0.2, color="#2563eb", linestyle="--", linewidth=0.9, alpha=0.6)
    axes[0, 0].axhline(0.9, color="#16a34a", linestyle="--", linewidth=0.9, alpha=0.6)
    axes[0, 0].axhline(0.02, color="#dc2626", linestyle="--", linewidth=0.9, alpha=0.6)
    _plot_if_present(
        axes[0, 1],
        x,
        records,
        ["final_dmax", "final_dispersion", "final_nearest_neighbor_distance"],
        ylabel="m",
        title="Final geometry",
    )
    _plot_if_present(
        axes[1, 0],
        x,
        records,
        ["max_success_hold_count_mean", "final_success_hold_count_mean", "mean_done_step"],
        ylabel="steps",
        title="Hold / finish diagnostics",
    )
    _plot_if_present(
        axes[1, 1],
        x,
        records,
        ["path_terrain_risk_mean", "filter_applied_fraction", "control_safety_applied_fraction"],
        ylabel="value",
        title="Terrain / filter / control diagnostics",
    )
    _plot_if_present(
        axes[2, 0],
        x,
        records,
        [
            "final_flatness_ok_rate",
            "final_gather_point_is_flat_rate",
            "final_gather_point_height_range_mean",
            "final_gather_point_max_slope_mean",
        ],
        ylabel="rate / m / gradient",
        title="Final gather-site flatness / terrain",
    )
    _plot_if_present(
        axes[2, 1],
        x,
        records,
        ["oracle_search.feasible_rate", "oracle_search.objective_mean"],
        ylabel="rate / objective",
        title="Oracle search",
    )
    fig.suptitle(run_dir.name, fontsize=12)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    _update_run_manifest_artifact(run_dir, "candidate_eval_curves", output)
    return output


def plot_comparison(run_dirs: list[Path], labels: list[str], output: Path) -> Path | None:
    rows: list[tuple[str, list[dict[str, Any]]]] = []
    for label, run_dir in zip(labels, run_dirs, strict=True):
        records = _read_run_candidate_evals(run_dir)
        if records:
            rows.append((label, records))
    if not rows:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    fields = [
        ("dmax_reduction_ratio", "dmax ratio", 0.2),
        ("success_rate", "success", 0.9),
        ("collision_rate", "collision", 0.02),
        ("timeout_rate", "timeout", 0.0),
        ("final_flatness_ok_rate", "final flatness", 0.9),
        ("oracle_search.feasible_rate", "oracle feasible", 1.0),
    ]
    for ax, (field, title, threshold) in zip(axes.ravel(), fields, strict=True):
        plotted = False
        for label, records in rows:
            x = _x_values(records)
            y = _series(records, field)
            if y and not all(value is None for value in y):
                ax.plot(x, y, marker="o", linewidth=1.5, label=label)
                plotted = True
        ax.axhline(threshold, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("timesteps")
        ax.set_ylabel("rate")
        ax.grid(True, alpha=0.25)
        if plotted:
            ax.legend(fontsize=8)
    fig.suptitle("Candidate eval comparison", fontsize=12)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    for run_dir in run_dirs:
        if _path_belongs_to_run(output, run_dir):
            _update_run_manifest_artifact(run_dir, "comparison_curves", output)
            break
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory. May be repeated.")
    parser.add_argument("--label", action="append", default=None, help="Label for comparison plot. May be repeated.")
    parser.add_argument("--comparison-output", default=None, help="Optional multi-run comparison PNG path.")
    args = parser.parse_args()

    run_dirs = [_resolve(path) for path in args.run_dir]
    labels = args.label or [path.name for path in run_dirs]
    if len(labels) != len(run_dirs):
        raise SystemExit("--label count must match --run-dir count")

    outputs: dict[str, Any] = {"runs": []}
    for run_dir in run_dirs:
        training = plot_training_curves(run_dir)
        candidate = plot_candidate_eval_curves(run_dir)
        outputs["runs"].append(
            {
                "run_dir": str(run_dir),
                "training_curves": str(training) if training is not None else None,
                "candidate_eval_curves": str(candidate) if candidate is not None else None,
            }
        )

    if args.comparison_output:
        comparison = plot_comparison(run_dirs, labels, _resolve(args.comparison_output))
        outputs["comparison_curves"] = str(comparison) if comparison is not None else None

    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
