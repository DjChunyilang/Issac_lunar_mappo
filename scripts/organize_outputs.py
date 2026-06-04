#!/usr/bin/env python
"""Organize generated artifacts into the canonical run-oriented output layout.

The tool is intentionally non-destructive by default: it creates symlinks or copies
under ``outputs/runs/<experiment>/<run>/`` and leaves legacy paths in place.
Use ``--dry-run`` to inspect planned operations before creating links.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from _common import ROOT


RUN_FILE_MAP = {
    "train_metrics.jsonl": "metrics/train_metrics.jsonl",
    "eval_metrics.json": "metrics/eval_metrics.json",
    "summary.json": "metrics/summary.json",
    "convergence_curves.png": "figures/convergence_curves.png",
    "safety_diagnostics.png": "figures/safety_diagnostics.png",
    "eval_rollout.gif": "videos/proxy_eval_rollout.gif",
    "tensorboard": "tensorboard",
    "tensorboard_curated": "tensorboard_curated",
}

SUITE_FILE_MAP = {
    "suite_summary.json": "metrics/suite_summary.json",
    "strict_acceptance.json": "metrics/strict_acceptance.json",
    "final_eval_best.json": "metrics/final_eval_best.json",
    "comparison_curves.png": "figures/comparison_curves.png",
    "safety_diagnostics.png": "figures/safety_diagnostics.png",
}

EXP007_PHASE_C_ARTIFACTS = {
    "config/experiment.yaml": "configs/experiment/exp_007_phase_c_weak_warmstart.yaml",
    "checkpoints/best.pt": "outputs/checkpoints/exp_007_phase_c_best.pt",
    "checkpoints/ppo_update_007.pt": "outputs/checkpoints/exp_007_phase_c_weak50_lr3e3_teacher_2m.pt",
    "metrics/train_metrics.jsonl": "outputs/logs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/train_metrics.jsonl",
    "metrics/eval_metrics.json": "outputs/logs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/eval_metrics.json",
    "metrics/final_eval_proxy.json": "outputs/logs/exp_007_phase_c/final_eval_proxy_weak50_lr3e3_teacher_2m.json",
    "figures/convergence_curves.png": "outputs/logs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/convergence_curves.png",
    "figures/safety_diagnostics.png": "outputs/logs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/safety_diagnostics.png",
    "videos/proxy_eval_rollout.gif": "outputs/logs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/eval_rollout.gif",
    "tensorboard": "outputs/logs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/tensorboard",
    "physx/metrics/lunar_crater_headless.json": "outputs/logs/physx_four_jetbots/evaluation_exp007_lunar_crater_headless.json",
    "physx/metrics/lunar_crater_render.json": "outputs/logs/physx_four_jetbots/evaluation_exp007_lunar_crater_render.json",
    "physx/figures/lunar_crater_scene.png": "outputs/figures/physx_four_jetbots/evaluation_exp007_lunar_crater_scene.png",
    "physx/videos/lunar_crater_rollout.gif": "outputs/videos/physx_four_jetbots/evaluation_exp007_lunar_crater_rollout.gif",
}


@dataclass(slots=True)
class OrganizeResult:
    experiment: str
    run: str
    run_dir: Path
    manifest_path: Path
    records: list[dict]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _replace_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool, dry_run: bool) -> dict:
    record = {
        "target": _relative(dst),
        "source": _relative(src),
        "exists": src.exists(),
        "kind": "directory" if src.is_dir() else "file",
        "mode": mode,
    }
    if not src.exists():
        record["status"] = "missing_source"
        return record
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            record["status"] = "already_exists"
            return record
        record["overwrites"] = True
    if dry_run:
        record["status"] = "would_create"
        return record
    if dst.exists() or dst.is_symlink():
        _replace_existing(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src)
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    record["status"] = "created"
    return record


def _load_json(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _infer_config_path(experiment: str, run: str, explicit_config: str | Path | None = None) -> Path | None:
    if explicit_config is not None:
        return _resolve(explicit_config)
    candidates: list[Path] = []
    if run.startswith("bc_ppo"):
        candidates.append(ROOT / "configs" / "experiment" / f"{experiment}_bc_ppo.yaml")
    if run.startswith("pure_rl"):
        candidates.append(ROOT / "configs" / "experiment" / f"{experiment}_pure_rl.yaml")
    if run.startswith("weak_warmstart") or "weak" in run:
        candidates.append(ROOT / "configs" / "experiment" / f"{experiment}_weak_warmstart.yaml")
    if run.startswith("bc_only"):
        candidates.append(ROOT / "configs" / "experiment" / f"{experiment}_bc_only.yaml")
    candidates.extend(
        [
            ROOT / "configs" / "experiment" / f"{experiment}.yaml",
            ROOT / "configs" / "experiment" / f"{experiment}_weak_warmstart.yaml",
            ROOT / "configs" / "experiment" / f"{experiment}_bc_ppo.yaml",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _checkpoint_candidates(experiment: str, run: str) -> Iterable[Path]:
    checkpoints = ROOT / "outputs" / "checkpoints"
    suffixes = [run]
    for prefix in ("phase_c_", f"{experiment}_"):
        if run.startswith(prefix):
            suffixes.append(run.removeprefix(prefix))
    seen: set[str] = set()
    for suffix in suffixes:
        name = f"{experiment}_{suffix}.pt"
        if name not in seen:
            seen.add(name)
            yield checkpoints / name
    yield checkpoints / f"{experiment}_best.pt"


def _load_run_summary(legacy_run_dir: Path) -> dict:
    eval_data = _load_json(legacy_run_dir / "eval_metrics.json") or {}
    summary = eval_data.get("summary") if isinstance(eval_data, dict) else None
    if isinstance(summary, dict):
        return {
            key: summary.get(key)
            for key in (
                "status",
                "mode",
                "seed",
                "device",
                "bc_steps",
                "updates",
                "checkpoint_path",
            )
            if key in summary
        } | {
            "best_metrics": summary.get("best_metrics"),
            "strict_acceptance": summary.get("strict_acceptance"),
        }
    return {}


def _write_manifest(
    run_dir: Path,
    experiment: str,
    run: str,
    producer: str,
    records: list[dict],
    summary: dict | None,
    dry_run: bool,
) -> Path:
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "run": run,
        "layout": "outputs/runs/<experiment>/<run>/<artifact_group>/...",
        "producer": producer,
        "summary": summary or {},
        "records": records,
    }
    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def organize_legacy_run(
    experiment: str,
    run: str,
    legacy_run_dir: Path,
    *,
    mode: str,
    overwrite: bool,
    dry_run: bool,
    config: str | Path | None = None,
) -> OrganizeResult:
    run_dir = ROOT / "outputs" / "runs" / experiment / run
    records: list[dict] = []
    for source_name, target_name in RUN_FILE_MAP.items():
        records.append(
            _link_or_copy(legacy_run_dir / source_name, run_dir / target_name, mode, overwrite, dry_run)
        )
    strict_path = legacy_run_dir.parent / f"{run}_strict.json"
    records.append(_link_or_copy(strict_path, run_dir / "metrics" / "strict_acceptance.json", mode, overwrite, dry_run))

    config_path = _infer_config_path(experiment, run, config)
    if config_path is not None:
        records.append(_link_or_copy(config_path, run_dir / "config" / "experiment.yaml", mode, overwrite, dry_run))

    checkpoint_path = next((candidate for candidate in _checkpoint_candidates(experiment, run) if candidate.exists()), None)
    if checkpoint_path is not None:
        records.append(_link_or_copy(checkpoint_path, run_dir / "checkpoints" / "best.pt", mode, overwrite, dry_run))

    summary = _load_run_summary(legacy_run_dir)
    manifest_path = _write_manifest(
        run_dir,
        experiment,
        run,
        "scripts/organize_outputs.py:legacy_run",
        records,
        summary,
        dry_run,
    )
    return OrganizeResult(experiment, run, run_dir, manifest_path, records)


def organize_suite_artifacts(
    experiment: str,
    legacy_exp_dir: Path,
    *,
    mode: str,
    overwrite: bool,
    dry_run: bool,
) -> OrganizeResult | None:
    run = "_suite"
    run_dir = ROOT / "outputs" / "runs" / experiment / run
    records: list[dict] = []
    for source_name, target_name in SUITE_FILE_MAP.items():
        records.append(_link_or_copy(legacy_exp_dir / source_name, run_dir / target_name, mode, overwrite, dry_run))
    for final_eval in sorted(legacy_exp_dir.glob("final_eval*.json")):
        records.append(_link_or_copy(final_eval, run_dir / "metrics" / final_eval.name, mode, overwrite, dry_run))
    if not any(record["exists"] for record in records):
        return None
    manifest_path = _write_manifest(
        run_dir,
        experiment,
        run,
        "scripts/organize_outputs.py:suite",
        records,
        {},
        dry_run,
    )
    return OrganizeResult(experiment, run, run_dir, manifest_path, records)


def _is_legacy_run_dir(path: Path) -> bool:
    return path.is_dir() and any((path / name).exists() for name in RUN_FILE_MAP)


def organize_legacy_experiment(
    experiment: str,
    *,
    legacy_root: Path,
    mode: str,
    overwrite: bool,
    dry_run: bool,
    config: str | Path | None = None,
) -> list[OrganizeResult]:
    legacy_exp_dir = legacy_root / experiment
    if not legacy_exp_dir.exists():
        return []
    results = []
    for child in sorted(legacy_exp_dir.iterdir()):
        if _is_legacy_run_dir(child):
            results.append(
                organize_legacy_run(
                    experiment,
                    child.name,
                    child,
                    mode=mode,
                    overwrite=overwrite,
                    dry_run=dry_run,
                    config=config,
                )
            )
    suite = organize_suite_artifacts(
        experiment,
        legacy_exp_dir,
        mode=mode,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    if suite is not None:
        results.append(suite)
    return results


def organize_exp007_preset(mode: str, overwrite: bool, dry_run: bool) -> OrganizeResult:
    experiment = "exp_007_phase_c"
    run = "phase_c_weak50_lr3e3_teacher_2m"
    run_dir = ROOT / "outputs" / "runs" / experiment / run
    records = [
        _link_or_copy(_resolve(source), run_dir / target, mode, overwrite, dry_run)
        for target, source in EXP007_PHASE_C_ARTIFACTS.items()
    ]
    summary = _load_run_summary(ROOT / "outputs" / "logs" / experiment / run)
    manifest_path = _write_manifest(
        run_dir,
        experiment,
        run,
        "scripts/organize_outputs.py:exp007_phase_c",
        records,
        summary,
        dry_run,
    )
    return OrganizeResult(experiment, run, run_dir, manifest_path, records)


def _discover_experiments(legacy_root: Path) -> list[str]:
    if not legacy_root.exists():
        return []
    return sorted(path.name for path in legacy_root.iterdir() if path.is_dir() and path.name.startswith("exp_"))


def build_outputs_index(dry_run: bool = False) -> Path:
    runs_root = ROOT / "outputs" / "runs"
    index_path = runs_root / "_index.json"
    runs = []
    for manifest_path in sorted(runs_root.glob("*/*/run_manifest.json")):
        data = _load_json(manifest_path) or {}
        run_dir = manifest_path.parent
        records = data.get("records", [])
        artifact_count = sum(1 for record in records if record.get("status") in {"created", "already_exists"})
        runs.append(
            {
                "experiment": data.get("experiment", run_dir.parent.name),
                "run": data.get("run", run_dir.name),
                "run_dir": _relative(run_dir),
                "manifest": _relative(manifest_path),
                "producer": data.get("producer"),
                "summary": data.get("summary", {}),
                "artifact_count": artifact_count,
            }
        )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "layout": "outputs/runs/<experiment>/<run>/...",
        "run_count": len(runs),
        "runs": runs,
    }
    if not dry_run:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return index_path


def _print_result(result: OrganizeResult) -> None:
    created = sum(1 for record in result.records if record.get("status") == "created")
    existing = sum(1 for record in result.records if record.get("status") == "already_exists")
    missing = sum(1 for record in result.records if record.get("status") == "missing_source")
    would_create = sum(1 for record in result.records if record.get("status") == "would_create")
    print(
        json.dumps(
            {
                "experiment": result.experiment,
                "run": result.run,
                "run_dir": _relative(result.run_dir),
                "manifest": _relative(result.manifest_path),
                "created": created,
                "already_exists": existing,
                "missing_source": missing,
                "would_create": would_create,
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("exp007_phase_c",), default=None)
    parser.add_argument("--experiment", default=None, help="Legacy experiment id under outputs/logs/.")
    parser.add_argument("--all-known", action="store_true", help="Organize every exp_* directory under outputs/logs/.")
    parser.add_argument("--legacy-root", default="outputs/logs")
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", default=None, help="Optional config snapshot source for --experiment runs.")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    legacy_root = _resolve(args.legacy_root)
    results: list[OrganizeResult] = []
    if args.preset == "exp007_phase_c":
        results.append(organize_exp007_preset(args.mode, args.overwrite, args.dry_run))
    if args.experiment:
        results.extend(
            organize_legacy_experiment(
                args.experiment,
                legacy_root=legacy_root,
                mode=args.mode,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                config=args.config,
            )
        )
    if args.all_known:
        for experiment in _discover_experiments(legacy_root):
            results.extend(
                organize_legacy_experiment(
                    experiment,
                    legacy_root=legacy_root,
                    mode=args.mode,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )
            )
    if not results and not (args.preset or args.experiment or args.all_known):
        results.append(organize_exp007_preset(args.mode, args.overwrite, args.dry_run))

    for result in results:
        _print_result(result)
    if not args.skip_index:
        index_path = build_outputs_index(dry_run=args.dry_run)
        status = "would_write" if args.dry_run else "written"
        print(json.dumps({"index": _relative(index_path), "status": status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
