from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import plot_skrl_run_curves as curves  # noqa: E402


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def _write_candidate_evals(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"evaluations": records}), encoding="utf-8")


def test_series_reads_nested_oracle_search_metrics_and_missing_values() -> None:
    records = [
        {
            "oracle_search": {
                "feasible_rate": 0.75,
                "objective_mean": 1.25,
            }
        },
        {},
    ]

    assert curves._series(records, "oracle_search.feasible_rate") == [0.75, None]
    assert curves._series(records, "oracle_search.objective_mean") == [1.25, None]


def test_candidate_artifacts_override_stale_aggregate_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "exp" / "run"
    _write_candidate_evals(
        run_dir / "metrics" / "eval_metrics.json",
        [{"candidate_timestep": 32, "final_flatness_ok_rate": 0.1}],
    )
    candidate_path = run_dir / "metrics" / "candidate_ppo_timestep_000032_eval.json"
    candidate_path.write_text(
        json.dumps({"final_flatness_ok_rate": 0.9}),
        encoding="utf-8",
    )

    records = curves._read_run_candidate_evals(run_dir)

    assert records == [{"candidate_timestep": 32, "final_flatness_ok_rate": 0.9}]


def test_single_run_curves_include_flatness_and_oracle_panels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "exp" / "run"
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "producer": "trainer",
                "command": "train --preserve-me",
                "artifacts": {"checkpoint_best": "checkpoints/best.pt"},
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        run_dir / "metrics" / "train_metrics.jsonl",
        [
            {
                "timesteps": 64,
                "flatness_ok_rate": 0.8,
                "gather_point_height_range_mean": 0.1,
                "gather_point_max_slope_mean": 0.2,
                "centroid_flatness_cost_mean": 0.8,
                "centroid_flatness_activation_mean": 1.0,
                "reward_contribution_flatness": 0.05,
                "oracle_search_feasible_rate": 1.0,
                "oracle_search_objective_mean": 3.0,
            }
        ],
    )
    _write_candidate_evals(
        run_dir / "metrics" / "eval_metrics.json",
        [
            {
                "candidate_timestep": 64,
                "final_flatness_ok_rate": 0.9,
                "final_gather_point_is_flat_rate": 0.9,
                "final_gather_point_height_range_mean": 0.08,
                "final_gather_point_max_slope_mean": 0.18,
                "oracle_search": {
                    "feasible_rate": 1.0,
                    "objective_mean": 2.5,
                },
            }
        ],
    )

    plotted_panels: list[tuple[str, tuple[str, ...]]] = []
    original_plot = curves._plot_if_present

    def capture_panel(ax, x, records, keys, *, ylabel, title) -> None:
        plotted_panels.append((title, tuple(keys)))
        original_plot(
            ax,
            x,
            records,
            keys,
            ylabel=ylabel,
            title=title,
        )

    monkeypatch.setattr(curves, "_plot_if_present", capture_panel)
    training_output = curves.plot_training_curves(run_dir)
    candidate_output = curves.plot_candidate_eval_curves(run_dir)

    assert training_output == run_dir / "figures" / "training_curves.png"
    assert candidate_output == run_dir / "figures" / "candidate_eval_curves.png"
    assert training_output.exists()
    assert candidate_output.exists()
    assert (
        "Gather-site flatness / terrain",
        (
            "flatness_ok_rate",
            "gather_point_height_range_mean",
            "gather_point_max_slope_mean",
            "centroid_flatness_cost_mean",
            "centroid_flatness_activation_mean",
        ),
    ) in plotted_panels
    assert (
        "Final gather-site flatness / terrain",
        (
            "final_flatness_ok_rate",
            "final_gather_point_is_flat_rate",
            "final_gather_point_height_range_mean",
            "final_gather_point_max_slope_mean",
        ),
    ) in plotted_panels
    assert (
        "Oracle search",
        ("oracle_search_feasible_rate", "oracle_search_objective_mean"),
    ) in plotted_panels
    assert (
        "Oracle search",
        ("oracle_search.feasible_rate", "oracle_search.objective_mean"),
    ) in plotted_panels
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["producer"] == "trainer"
    assert manifest["command"] == "train --preserve-me"
    assert manifest["artifacts"]["checkpoint_best"] == "checkpoints/best.pt"
    assert manifest["artifacts"]["training_curves"] == str(training_output)
    assert manifest["artifacts"]["candidate_eval_curves"] == str(candidate_output)
    assert manifest["generated_at"] != "2026-01-01T00:00:00+00:00"


def test_legacy_single_run_metrics_without_new_fields_still_plot(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "legacy_exp" / "legacy_run"
    _write_jsonl(
        run_dir / "metrics" / "train_metrics.jsonl",
        [{"timesteps": 32, "mean_reward": 1.0, "success_rate": 0.25}],
    )
    _write_candidate_evals(
        run_dir / "metrics" / "eval_metrics.json",
        [{"candidate_timestep": 32, "success_rate": 0.25, "timeout_rate": 0.75}],
    )

    training_output = curves.plot_training_curves(run_dir)
    candidate_output = curves.plot_candidate_eval_curves(run_dir)

    assert training_output is not None and training_output.exists()
    assert candidate_output is not None and candidate_output.exists()
    assert not (run_dir / "run_manifest.json").exists()


def test_comparison_manifest_registration_only_for_output_inside_run(tmp_path: Path) -> None:
    run_dirs = [
        tmp_path / "outputs" / "runs" / "exp" / "run_a",
        tmp_path / "outputs" / "runs" / "exp" / "run_b",
    ]
    for index, run_dir in enumerate(run_dirs):
        _write_candidate_evals(
            run_dir / "metrics" / "eval_metrics.json",
            [
                {
                    "candidate_timestep": 32,
                    "success_rate": 0.5 + 0.1 * index,
                    "timeout_rate": 0.5 - 0.1 * index,
                }
            ],
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "generated_at": "2026-01-01T00:00:00+00:00",
                    "producer": f"trainer_{index}",
                    "artifacts": {},
                }
            ),
            encoding="utf-8",
        )

    independent_output = (
        tmp_path / "outputs" / "runs" / "_comparisons" / "comparison" / "figures" / "curves.png"
    )
    result = curves.plot_comparison(run_dirs, ["a", "b"], independent_output)

    assert result == independent_output
    assert result.exists()
    for index, run_dir in enumerate(run_dirs):
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest == {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "producer": f"trainer_{index}",
            "artifacts": {},
        }

    run_output = run_dirs[0] / "figures" / "comparison_curves.png"
    result = curves.plot_comparison(run_dirs, ["a", "b"], run_output)

    assert result == run_output
    first_manifest = json.loads(
        (run_dirs[0] / "run_manifest.json").read_text(encoding="utf-8")
    )
    second_manifest = json.loads(
        (run_dirs[1] / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert first_manifest["producer"] == "trainer_0"
    assert first_manifest["artifacts"]["comparison_curves"] == str(run_output)
    assert first_manifest["generated_at"] != "2026-01-01T00:00:00+00:00"
    assert second_manifest == {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "producer": "trainer_1",
        "artifacts": {},
    }
