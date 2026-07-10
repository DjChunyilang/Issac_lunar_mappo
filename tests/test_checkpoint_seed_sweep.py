from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_proxy_checkpoint_seed_sweep import run_seed_sweep, summarize_rows  # noqa: E402


def test_summarize_rows_tracks_timeout_and_strict_counts() -> None:
    rows = [
        {
            "checkpoint": "a.pt",
            "dmax_reduction_ratio": 0.18,
            "success_rate": 0.95,
            "collision_rate": 0.0,
            "timeout_rate": 0.0,
            "timeout_count": 0,
            "passed": True,
        },
        {
            "checkpoint": "a.pt",
            "dmax_reduction_ratio": 0.19,
            "success_rate": 0.94,
            "collision_rate": 0.01,
            "timeout_rate": 0.02,
            "timeout_count": 2,
            "passed": False,
        },
        {
            "checkpoint": "b.pt",
            "dmax_reduction_ratio": 0.21,
            "success_rate": 0.80,
            "collision_rate": 0.03,
            "timeout_rate": 0.20,
            "timeout_count": 20,
            "passed": False,
        },
    ]

    summary = summarize_rows(rows, ["a.pt", "b.pt"])

    assert summary[0]["checkpoint"] == "a.pt"
    assert summary[0]["n"] == 2
    assert summary[0]["timeout_rate_mean"] == 0.01
    assert summary[0]["timeout_count_max"] == 2
    assert summary[0]["strict_pass_count"] == 1
    assert summary[0]["timeout_zero_count"] == 1
    assert summary[1]["checkpoint"] == "b.pt"
    assert summary[1]["strict_pass_count"] == 0


def test_run_seed_sweep_writes_per_seed_outputs_and_summary(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text("experiment:\n  name: sweep_test\n", encoding="utf-8")
    run_dir = tmp_path / "outputs" / "runs" / "sweep_test" / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for name in ("a.pt", "b.pt"):
        (checkpoint_dir / name).write_bytes(b"mock")

    def evaluator(config_path, checkpoint_path, device=None, num_envs=0, steps=0, seed=None, output=None):
        return {
            "dmax_reduction_ratio": 0.18 if Path(checkpoint_path).name == "a.pt" else 0.21,
            "success_rate": 0.95,
            "collision_rate": 0.0,
            "timeout_rate": 0.0 if seed == 1 else 0.01,
            "timeout_episode_metrics": {"count": 0 if seed == 1 else 1},
            "mean_done_step": 100.0,
            "filter_applied_fraction": 0.2,
            "filter_collision_override_fraction": 0.1,
            "control_safety_applied_fraction": 0.05,
        }

    result = run_seed_sweep(
        config=config,
        run_dir=run_dir,
        checkpoints=["a.pt", "b.pt"],
        seeds=[1, 2],
        device="cpu",
        num_envs=4,
        steps=8,
        evaluator=evaluator,
    )

    summary_path = run_dir / "metrics" / "checkpoint_seed_sweep" / "summary.json"
    assert summary_path.exists()
    assert (run_dir / "metrics" / "checkpoint_seed_sweep" / "a_seed1_eval.json").exists()
    assert result["num_envs"] == 4
    assert result["steps"] == 8
    assert result["summary"][0]["checkpoint"] == "a.pt"
    assert result["summary"][0]["timeout_zero_count"] == 1
