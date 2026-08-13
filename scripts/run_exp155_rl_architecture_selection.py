#!/usr/bin/env python3
"""Run the equal-budget full Pure-RL exp155 architecture ablation.

The offline dataset is diagnostic only. N0, N1 and N2 each receive the same
seed, environment count, rollout length and fixed 800/800/800-stage schedule.
No performance gate may stop or extend an architecture during this comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from _common import ROOT


CANDIDATES = (
    "multiscale_n0_mlp",
    "multiscale_n1_cnn",
    "multiscale_n2_path_conditioned",
)
SHORT_NAMES = {
    "multiscale_n0_mlp": "n0",
    "multiscale_n1_cnn": "n1",
    "multiscale_n2_path_conditioned": "n2",
}
PARAMETER_COUNTS = {
    "multiscale_n0_mlp": 82_264,
    "multiscale_n1_cnn": 105_512,
    "multiscale_n2_path_conditioned": 22_936,
}


def _mean_cell_metrics(stratified: dict) -> dict[str, float]:
    cells = list(stratified["cells"])
    keys = (
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "dmax_reduction_ratio",
        "path_terrain_risk_mean",
    )
    return {
        key: sum(float(cell["metrics"].get(key, 0.0)) for cell in cells) / len(cells)
        for key in keys
    }


def _selection_score(metrics: dict[str, float]) -> float:
    """Fixed score used only after the number of strict cells is compared."""

    return (
        float(metrics["success_rate"])
        - float(metrics["collision_rate"])
        - 0.5 * float(metrics["timeout_rate"])
        - 0.5 * float(metrics["dmax_reduction_ratio"])
        - 0.25 * float(metrics["path_terrain_risk_mean"])
    )


def _run(command: list[str], *, execute: bool) -> int | None:
    if not execute:
        return None
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiment/exp155_full_rl_ablation.yaml"
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/runs/exp155_full_rl_ablation/_suite/metrics/"
            "full_rl_architecture_ablation.json"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    python = str(ROOT / ".venv_isaaclab/bin/python3.12")
    records: list[dict] = []
    for index, architecture in enumerate(CANDIDATES):
        short = SHORT_NAMES[architecture]
        run_name = f"{short}_seed23_full_2400iter"
        run_dir = ROOT / "outputs/runs/exp155_full_rl_ablation" / run_name
        train_command = [
            python,
            str(ROOT / "scripts/train_skrl_mappo.py"),
            "--config", str(ROOT / args.config),
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
        record = {
            "architecture": architecture,
            "parameter_count": PARAMETER_COUNTS[architecture],
            "seed": 23,
            "policy_iterations": 2400,
            "environment_interactions": 39_321_600,
            "run_dir": str(run_dir.relative_to(ROOT)),
            "commands": {"train": train_command},
        }
        records.append(record)
        train_returncode = _run(train_command, execute=args.execute)
        record["train_returncode"] = train_returncode
        if not args.execute:
            continue

        summary_path = run_dir / "metrics/summary.json"
        if train_returncode != 0 or not summary_path.is_file():
            record["status"] = "training_error"
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        fixed_schedule_complete = (
            summary.get("bounded_curriculum", {}).get("status") == "completed"
            and summary.get("bounded_curriculum", {}).get("transition_mode")
            == "fixed_schedule"
            and int(summary.get("timesteps", -1)) == 153_600
        )
        record["fixed_schedule_complete"] = fixed_schedule_complete
        record["wall_time_s"] = float(summary.get("wall_time_s", 0.0))
        if not fixed_schedule_complete:
            record["status"] = "incomplete_fixed_schedule"
            continue

        stratified_command = [
            python,
            str(ROOT / "scripts/evaluate_exp155_stratified.py"),
            "--run-dir", str(run_dir.relative_to(ROOT)),
            "--device", args.device,
            "--episodes-per-cell", "64",
            "--steps", "480",
            "--seed", str(23_000 + index * 1_000),
        ]
        record["commands"]["stratified_evaluation"] = stratified_command
        record["stratified_returncode"] = _run(
            stratified_command, execute=args.execute
        )
        stratified_path = run_dir / "metrics/stratified_strict_acceptance.json"
        if not stratified_path.is_file():
            record["status"] = "missing_stratified_evaluation"
            continue
        stratified = json.loads(stratified_path.read_text(encoding="utf-8"))

        terrain_command = [
            python,
            str(ROOT / "scripts/evaluate_terrain_contrast.py"),
            "--config", str(run_dir / "config/experiment.yaml"),
            "--checkpoint", str(run_dir / "checkpoints/best.pt"),
            "--device", args.device,
            "--num-envs", "512",
            "--steps", "120",
            "--seed", str(26_000 + index * 1_000),
            "--run-dir", str(run_dir.relative_to(ROOT)),
        ]
        record["commands"]["terrain_contrast"] = terrain_command
        record["terrain_returncode"] = _run(terrain_command, execute=args.execute)
        terrain_path = run_dir / "metrics/terrain_contrast.json"
        terrain = (
            json.loads(terrain_path.read_text(encoding="utf-8"))
            if terrain_path.is_file()
            else None
        )

        means = _mean_cell_metrics(stratified)
        strict_cells_passed = sum(bool(cell["passed"]) for cell in stratified["cells"])
        terrain_checks_passed = bool(terrain) and all(terrain["checks"].values())
        record.update(
            status="evaluated",
            strict_cells_passed=strict_cells_passed,
            all_six_strata_passed=bool(stratified["passed"]),
            terrain_checks_passed=terrain_checks_passed,
            formal_passed=bool(stratified["passed"]) and terrain_checks_passed,
            mean_metrics=means,
            terrain_metrics=(
                {
                    "policy_js_normal_vs_zero_terrain": terrain[
                        "policy_js_normal_vs_zero_terrain"
                    ],
                    "path_risk_reduction_fraction": terrain[
                        "path_risk_reduction_fraction"
                    ],
                    "checks": terrain["checks"],
                }
                if terrain
                else None
            ),
            score=_selection_score(means),
        )

    eligible = [record for record in records if record.get("status") == "evaluated"]
    winner = None
    selection_reason = "No architecture completed training and evaluation."
    if eligible:
        eligible.sort(
            key=lambda record: (
                -int(record["strict_cells_passed"]),
                -float(record["score"]),
                int(record["parameter_count"]),
            )
        )
        winner_record = eligible[0]
        winner = winner_record["architecture"]
        selection_reason = (
            "Maximum strict cells, then the predeclared aggregate score; parameter "
            "count breaks a score difference below 0.02."
        )
        if (
            len(eligible) >= 2
            and eligible[0]["strict_cells_passed"]
            == eligible[1]["strict_cells_passed"]
            and abs(float(eligible[0]["score"]) - float(eligible[1]["score"])) < 0.02
        ):
            tied = [
                record
                for record in eligible
                if record["strict_cells_passed"] == eligible[0]["strict_cells_passed"]
                and abs(float(record["score"]) - float(eligible[0]["score"])) < 0.02
            ]
            winner_record = min(
                tied,
                key=lambda record: (
                    int(record["parameter_count"]),
                    float(record.get("wall_time_s", float("inf"))),
                ),
            )
            winner = winner_record["architecture"]
            selection_reason = (
                "Strict-cell count tied and aggregate scores differed by less than "
                "0.02; the smaller model, then lower training wall time, won."
            )

    result = {
        "offline_screen_role": "diagnostic_only",
        "offline_screen_used_for_admission": False,
        "config": args.config,
        "seed": 23,
        "candidates": list(CANDIDATES),
        "fixed_schedule": {
            "stage_a_iterations": 800,
            "stage_b_iterations": 800,
            "stage_c_iterations": 800,
            "total_policy_iterations": 2400,
            "timesteps": 153_600,
            "environment_interactions_per_candidate": 39_321_600,
            "environment_interactions_total": 117_964_800,
            "early_stop": "engineering failures only",
            "extension_iterations": 0,
        },
        "selection_score": (
            "mean_success - mean_collision - 0.5*mean_timeout - "
            "0.5*mean_dmax_ratio - 0.25*mean_path_risk"
        ),
        "runs": records,
        "winner": winner,
        "winner_selection_reason": selection_reason,
        "winner_formal_passed": bool(
            winner
            and next(
                record for record in records if record["architecture"] == winner
            ).get("formal_passed", False)
        ),
        "completed": len(eligible) == len(CANDIDATES),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.execute and not result["completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
