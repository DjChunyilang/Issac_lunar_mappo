from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from diagnose_proxy_success_gates import summarize_episode_rows  # noqa: E402


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
