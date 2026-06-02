#!/usr/bin/env python
"""Run proxy convergence comparison experiments and strict acceptance checks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

for cache_env, cache_dir in (
    ("MPLCONFIGDIR", "/tmp/isaac_mappo_matplotlib"),
    ("XDG_CACHE_HOME", "/tmp/isaac_mappo_cache"),
):
    os.environ.setdefault(cache_env, cache_dir)
    Path(os.environ[cache_env]).mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import ROOT
from train_proxy_convergence import STRICT_THRESHOLDS, strict_acceptance


MODES = ("pure_rl", "bc_only", "bc_ppo")
DEFAULT_SEEDS = (23, 31, 47)


def _resolve(path: str | Path) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _run_one(
    config: Path,
    mode: str,
    seed: int,
    device: str,
    run_name: str,
    checkpoint_path: Path,
    overrides: list[str] | None = None,
) -> dict:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_proxy_convergence.py"),
        "--config",
        str(config),
        "--device",
        device,
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--run-name",
        run_name,
        "--checkpoint-path",
        str(checkpoint_path),
    ]
    if overrides:
        command.extend(overrides)
    print("RUN", " ".join(command), flush=True)
    started = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    run_dir = ROOT / "outputs" / "logs" / "exp_005_safety_tuned" / run_name
    eval_path = run_dir / "eval_metrics.json"
    result = {
        "mode": mode,
        "seed": seed,
        "run_name": run_name,
        "checkpoint_path": str(checkpoint_path),
        "eval_metrics_path": str(eval_path),
        "returncode": started.returncode,
        "stdout": started.stdout[-4000:],
        "stderr": started.stderr[-4000:],
    }
    if started.returncode == 0 and eval_path.exists():
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        result["summary"] = summary
        result["best_metrics"] = summary["best_metrics"]
        result["strict_acceptance"] = strict_acceptance(summary["best_metrics"])
    else:
        result["strict_acceptance"] = {"passed": False, "checks": {}, "thresholds": STRICT_THRESHOLDS}
    return result


def build_strict_acceptance(results: list[dict], final_mode: str = "bc_ppo") -> dict:
    final_results = [item for item in results if item["mode"] == final_mode and "retry_of" not in item]
    retry_results = {item["retry_of"]: item for item in results if item["mode"] == final_mode and "retry_of" in item}
    seed_results = []
    for item in final_results:
        selected = retry_results.get(item["seed"], item)
        seed_results.append(
            {
                "seed": selected["seed"],
                "run_name": selected["run_name"],
                "checkpoint_path": selected["checkpoint_path"],
                "metrics": selected.get("best_metrics"),
                "strict_acceptance": selected["strict_acceptance"],
                "used_retry": "retry_of" in selected,
            }
        )
    return {
        "passed": bool(seed_results) and all(item["strict_acceptance"]["passed"] for item in seed_results),
        "final_mode": final_mode,
        "thresholds": STRICT_THRESHOLDS,
        "seeds": seed_results,
    }


def _save_comparison_curves(results: list[dict], path: Path) -> None:
    rows = [item for item in results if "best_metrics" in item and "retry_of" not in item]
    if not rows:
        return
    modes = list(MODES)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    metric_names = [
        ("dmax_reduction_ratio", "dmax ratio"),
        ("success_rate", "success rate"),
        ("collision_rate", "collision rate"),
        ("timeout_rate", "timeout rate"),
    ]
    for ax, (metric, title) in zip(axes.flatten(), metric_names):
        means = []
        stds = []
        for mode in modes:
            values = [item["best_metrics"][metric] for item in rows if item["mode"] == mode]
            means.append(float(np.mean(values)) if values else 0.0)
            stds.append(float(np.std(values)) if values else 0.0)
        ax.bar(modes, means, yerr=stds, capsize=4)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_safety_diagnostics(results: list[dict], path: Path) -> None:
    rows = [item for item in results if item["mode"] == "bc_ppo" and "best_metrics" in item]
    if not rows:
        return
    seeds = [str(item["seed"]) + ("r" if "retry_of" in item else "") for item in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].bar(seeds, [item["best_metrics"].get("min_nearest_distance") or 0.0 for item in rows])
    axes[0, 0].set_title("Min nearest distance")
    axes[0, 1].bar(seeds, [item["best_metrics"].get("near_violation_rate") or 0.0 for item in rows])
    axes[0, 1].set_title("Near violation rate")
    axes[1, 0].bar(seeds, [item["best_metrics"]["collision_rate"] for item in rows])
    axes[1, 0].axhline(STRICT_THRESHOLDS["collision_rate"], color="tab:red", linestyle="--")
    axes[1, 0].set_title("Collision rate")
    axes[1, 1].scatter(
        [item["best_metrics"]["collision_rate"] for item in rows],
        [item["best_metrics"]["success_rate"] for item in rows],
    )
    axes[1, 1].set_xlabel("collision rate")
    axes[1, 1].set_ylabel("success rate")
    axes[1, 1].set_title("Success / collision")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _copy_best_artifacts(strict: dict, results: list[dict], output_root: Path) -> dict:
    candidates = [
        item
        for item in results
        if item["mode"] == strict["final_mode"] and "best_metrics" in item and item["strict_acceptance"]["passed"]
    ]
    if not candidates:
        candidates = [item for item in results if item["mode"] == strict["final_mode"] and "best_metrics" in item]
    if not candidates:
        return {}
    best = min(candidates, key=lambda item: item["best_metrics"]["dmax_reduction_ratio"])
    best_checkpoint = _resolve("outputs/checkpoints/exp_005_safety_tuned_best.pt")
    shutil.copy2(best["checkpoint_path"], best_checkpoint)
    run_dir = ROOT / "outputs" / "logs" / "exp_005_safety_tuned" / best["run_name"]
    video_dir = ROOT / "outputs" / "videos" / "exp_005_safety_tuned"
    video_dir.mkdir(parents=True, exist_ok=True)
    gif_src = run_dir / "eval_rollout.gif"
    gif_dst = video_dir / "best_proxy_rollout.gif"
    if gif_src.exists():
        shutil.copy2(gif_src, gif_dst)
    return {
        "best_run_name": best["run_name"],
        "best_checkpoint": str(best_checkpoint),
        "best_proxy_rollout_gif": str(gif_dst) if gif_dst.exists() else None,
        "best_metrics": best["best_metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-root", default="configs/experiment")
    parser.add_argument("--suite", default="exp_005_safety_tuned")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="23,31,47")
    args = parser.parse_args()

    config_root = ROOT / args.config_root
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    suite_root = ROOT / "outputs" / "logs" / args.suite
    suite_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for mode in MODES:
        config = config_root / f"{args.suite}_{mode}.yaml"
        for seed in seeds:
            run_name = f"{mode}_seed_{seed}"
            checkpoint = ROOT / "outputs" / "checkpoints" / f"{args.suite}_{mode}_seed_{seed}.pt"
            results.append(_run_one(config, mode, seed, args.device, run_name, checkpoint))

    strict = build_strict_acceptance(results)
    retry_overrides = [
        "--bc-steps",
        "600",
        "--total-env-steps",
        "1500000",
        "--teacher-stop-radius",
        "0.55",
        "--safety-near-distance",
        "1.0",
        "--near-penalty-coef",
        "5.0",
        "--collision-penalty-coef",
        "60.0",
    ]
    for seed_result in strict["seeds"]:
        if seed_result["strict_acceptance"]["passed"]:
            continue
        seed = int(seed_result["seed"])
        run_name = f"bc_ppo_retry_seed_{seed}"
        checkpoint = ROOT / "outputs" / "checkpoints" / f"{args.suite}_bc_ppo_retry_seed_{seed}.pt"
        result = _run_one(
            config_root / f"{args.suite}_bc_ppo.yaml",
            "bc_ppo",
            seed,
            args.device,
            run_name,
            checkpoint,
            retry_overrides,
        )
        result["retry_of"] = seed
        results.append(result)
    strict = build_strict_acceptance(results)

    comparison_path = suite_root / "comparison_curves.png"
    safety_path = suite_root / "safety_diagnostics.png"
    _save_comparison_curves(results, comparison_path)
    _save_safety_diagnostics(results, safety_path)
    best = _copy_best_artifacts(strict, results, suite_root)
    summary = {
        "status": "ok",
        "suite": args.suite,
        "device": args.device,
        "seeds": list(seeds),
        "modes": list(MODES),
        "strict_acceptance": strict,
        "best": best,
        "artifacts": {
            "suite_summary": str(suite_root / "suite_summary.json"),
            "strict_acceptance": str(suite_root / "strict_acceptance.json"),
            "comparison_curves": str(comparison_path),
            "safety_diagnostics": str(safety_path),
        },
        "results": results,
    }
    (suite_root / "suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (suite_root / "strict_acceptance.json").write_text(json.dumps(strict, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not strict["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
