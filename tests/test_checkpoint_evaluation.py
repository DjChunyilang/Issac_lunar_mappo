from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_checkpoint_evaluation import (  # noqa: E402
    CHECKPOINT_STATES,
    parse_evaluation_config,
    run_checkpoint_evaluation,
)


def _write_config(tmp_path: Path, evaluation: dict | None = None) -> Path:
    payload = {"experiment": {"name": "eval_test", "seed": 7}}
    if evaluation is not None:
        payload["evaluation"] = evaluation
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"mock")
    return checkpoint


def _proxy_runner(metrics: dict):
    def runner(config, checkpoint, device=None, num_envs=0, steps=0, seed=None, run_dir=None):
        assert num_envs > 0
        assert steps > 0
        assert seed is not None
        result = {
            "status": "ok",
            "artifact": str(Path(run_dir) / "metrics" / "final_eval_proxy.json"),
            **metrics,
        }
        path = Path(run_dir) / "metrics" / "final_eval_proxy.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")
        return result

    return runner


def _physx_runner(metrics: dict):
    def runner(*, config, checkpoint, run_dir, high_cfg):
        result = {
            "status": "ok",
            "artifact": str(Path(run_dir) / "physx" / "metrics" / f"{high_cfg.terrain}_tracking_summary.json"),
            **metrics,
        }
        path = Path(result["artifact"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")
        return result

    return runner


def test_evaluation_config_defaults() -> None:
    cfg = parse_evaluation_config({"experiment": {"name": "default_eval"}})

    assert cfg.proxy_eval.enabled is True
    assert cfg.proxy_eval.num_envs == 1024
    assert cfg.proxy_eval.steps == 220
    assert cfg.high_fidelity_eval.backend == "physx_jackal"
    assert cfg.high_fidelity_eval.trigger == "proxy_passed"
    assert cfg.high_fidelity_eval.terrain == "strong_lunar_crater"


def test_evaluation_config_unknown_key_fails_fast() -> None:
    with pytest.raises(ValueError, match=r"evaluation\.proxy_eval\.bad_key"):
        parse_evaluation_config({"evaluation": {"proxy_eval": {"bad_key": 1}}})


def test_evaluation_config_trigger_enum_is_checked() -> None:
    with pytest.raises(ValueError, match="Unsupported high-fidelity trigger"):
        parse_evaluation_config({"evaluation": {"high_fidelity_eval": {"trigger": "sometimes"}}})


def test_checkpoint_status_proxy_fail_skips_physx(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    run_dir = tmp_path / "outputs" / "runs" / "eval_test" / "run_fail"

    status = run_checkpoint_evaluation(
        config=config,
        checkpoint=checkpoint,
        run_dir=run_dir,
        device="cpu",
        proxy_runner=_proxy_runner(
            {
                "dmax_reduction_ratio": 0.35,
                "success_rate": 0.0,
                "collision_rate": 0.0,
                "timeout_rate": 1.0,
            }
        ),
        physx_runner=_physx_runner({"passed": True}),
    )

    assert status["state"] == "candidate"
    assert status["states"] == list(CHECKPOINT_STATES)
    assert status["high_fidelity_eval"]["skip_reason"] == "proxy_not_passed"
    assert (run_dir / "metrics" / "checkpoint_status.json").exists()
    assert (run_dir / "run_manifest.json").exists()


def test_checkpoint_status_physx_pass_can_be_final_selected(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    run_dir = tmp_path / "outputs" / "runs" / "eval_test" / "run_pass"

    status = run_checkpoint_evaluation(
        config=config,
        checkpoint=checkpoint,
        run_dir=run_dir,
        device="cpu",
        mark_final_selected=True,
        proxy_runner=_proxy_runner(
            {
                "dmax_reduction_ratio": 0.1,
                "success_rate": 0.95,
                "collision_rate": 0.0,
                "timeout_rate": 0.0,
            }
        ),
        physx_runner=_physx_runner(
            {
                "passed": True,
                "aggregate": {
                    "mean_rmse_cross_track_m": 0.12,
                    "max_cross_track_m": 0.35,
                    "min_path_completion_ratio": 0.93,
                    "max_tilt_deg": 3.0,
                },
            }
        ),
    )

    assert status["state"] == "final_selected"
    assert status["proxy_eval"]["gate"]["passed"]
    assert status["high_fidelity_eval"]["gate"]["passed"]
    assert status["high_fidelity_eval"]["diagnostics"]["mean_rmse_cross_track_m"] == 0.12
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_evaluation"]["state"] == "final_selected"


def test_checkpoint_status_manual_trigger_keeps_proxy_passed(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        {
            "high_fidelity_eval": {
                "trigger": "manual",
            }
        },
    )
    checkpoint = _checkpoint(tmp_path)
    run_dir = tmp_path / "outputs" / "runs" / "eval_test" / "run_manual"

    status = run_checkpoint_evaluation(
        config=config,
        checkpoint=checkpoint,
        run_dir=run_dir,
        device="cpu",
        proxy_runner=_proxy_runner(
            {
                "dmax_reduction_ratio": 0.1,
                "success_rate": 0.95,
                "collision_rate": 0.0,
                "timeout_rate": 0.0,
            }
        ),
        physx_runner=_physx_runner({"passed": True}),
    )

    assert status["state"] == "proxy_passed"
    assert status["high_fidelity_eval"]["skip_reason"] == "manual_trigger"
