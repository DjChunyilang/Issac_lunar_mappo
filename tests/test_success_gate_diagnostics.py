from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from diagnose_proxy_success_gates import (  # noqa: E402
    _register_diagnostic_artifact,
    summarize_episode_rows,
)


def test_success_gate_diagnostics_summarizes_timeout_gate_failures() -> None:
    rows = [
        {
            "done_reason": "success",
            "final_dmax": 1.0,
            "final_dispersion": 0.2,
            "final_mean_speed": 0.01,
            "final_min_pairwise": 0.5,
            "final_success_hold_count": 8,
            "max_success_hold_count": 8,
            "final_terrain_speed_scale": 0.6,
            "final_dmax_ok": True,
            "final_dispersion_ok": True,
            "final_speed_ok": True,
            "final_min_pairwise_ok": True,
            "final_instant_success": True,
        },
        {
            "done_reason": "timeout",
            "final_dmax": 0.9,
            "final_dispersion": 0.16,
            "final_mean_speed": 0.02,
            "final_min_pairwise": 0.35,
            "final_success_hold_count": 0,
            "max_success_hold_count": 1,
            "final_terrain_speed_scale": 0.5,
            "final_dmax_ok": True,
            "final_dispersion_ok": True,
            "final_speed_ok": True,
            "final_min_pairwise_ok": False,
            "final_instant_success": False,
        },
        {
            "done_reason": "timeout",
            "final_dmax": 1.4,
            "final_dispersion": 0.35,
            "final_mean_speed": 0.03,
            "final_min_pairwise": 0.6,
            "final_success_hold_count": 0,
            "max_success_hold_count": 0,
            "final_terrain_speed_scale": 0.7,
            "final_dmax_ok": False,
            "final_dispersion_ok": False,
            "final_speed_ok": True,
            "final_min_pairwise_ok": True,
            "final_instant_success": False,
        },
    ]

    summary = summarize_episode_rows(rows)

    assert summary["num_envs"] == 3
    assert summary["counts_by_reason"]["success"] == 1
    assert summary["counts_by_reason"]["timeout"] == 2
    assert summary["by_reason"]["timeout"]["final_min_pairwise_mean"] == 0.475
    assert summary["by_reason"]["timeout"]["dmax_ok_rate"] == 0.5
    assert summary["by_reason"]["timeout"]["min_pairwise_ok_rate"] == 0.5
    assert summary["timeout_final_gate_failure_counts"] == {
        "dmax": 1,
        "dispersion": 1,
        "speed": 0,
        "min_pairwise": 1,
        "instant_success": 2,
    }


def test_mixed_legacy_rows_do_not_count_missing_flatness_as_failure() -> None:
    common = {
        "done_reason": "timeout",
        "final_dmax": 1.0,
        "final_dispersion": 0.2,
        "final_mean_speed": 0.01,
        "final_min_pairwise": 0.5,
        "final_success_hold_count": 0,
        "max_success_hold_count": 0,
        "final_terrain_speed_scale": 1.0,
        "final_dmax_ok": True,
        "final_dispersion_ok": True,
        "final_speed_ok": True,
        "final_min_pairwise_ok": True,
        "final_instant_success": False,
    }
    rows = [
        common,
        {**common, "final_flatness_ok": True},
        {**common, "final_flatness_ok": False},
    ]

    summary = summarize_episode_rows(rows)

    assert summary["by_reason"]["timeout"]["flatness_ok_rate"] == 0.5
    assert summary["timeout_final_gate_failure_counts"]["flatness"] == 1


def test_diagnostic_registers_existing_run_manifest_atomically(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text('{"artifacts": {"metrics_train": "train.jsonl"}}', encoding="utf-8")
    output_path = run_dir / "metrics" / "success_gate_diagnostics.json"
    output_path.parent.mkdir()
    output_path.write_text("{}", encoding="utf-8")

    _register_diagnostic_artifact(run_dir, output_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["metrics_train"] == "train.jsonl"
    assert manifest["artifacts"]["metrics_success_gate_diagnostic"] == str(output_path)
