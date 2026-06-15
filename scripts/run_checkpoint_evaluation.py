#!/usr/bin/env python
"""Run checkpoint-level proxy and high-fidelity evaluation for one run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from _common import ROOT, load_yaml
from evaluate_physx_jackal_tracking import physx_acceptance, physx_diagnostics
from evaluate_proxy_policy import proxy_acceptance
from evaluate_proxy_policy import evaluate_checkpoint as evaluate_proxy_checkpoint


CHECKPOINT_STATES = (
    "candidate",
    "proxy_passed",
    "physx_evaluated",
    "physx_passed",
    "final_selected",
)
HIGH_FIDELITY_TRIGGERS = ("always", "proxy_passed", "manual")
HIGH_FIDELITY_BACKENDS = ("physx_jackal",)


@dataclass(slots=True)
class ProxyEvalCfg:
    enabled: bool = True
    num_envs: int = 1024
    steps: int = 220
    seed_offset: int = 1000


@dataclass(slots=True)
class HighFidelityEvalCfg:
    enabled: bool = True
    backend: str = "physx_jackal"
    trigger: str = "proxy_passed"
    terrain: str = "strong_lunar_crater"
    episodes: int = 3
    steps: int = 100
    sim_steps_per_control: int = 8
    render: bool = False


@dataclass(slots=True)
class EvaluationCfg:
    proxy_eval: ProxyEvalCfg
    high_fidelity_eval: HighFidelityEvalCfg


def _require_mapping(section: str, value: Any) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")
    return value


def _reject_unknown(section: str, values: dict, supported: set[str]) -> None:
    unknown = sorted(key for key in values if key not in supported)
    if unknown:
        unknown_keys = ", ".join(f"{section}.{key}" for key in unknown)
        raise ValueError(f"Unsupported config key(s): {unknown_keys}.")


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def parse_evaluation_config(raw_cfg: dict) -> EvaluationCfg:
    evaluation = _require_mapping("evaluation", raw_cfg.get("evaluation", {}))
    _reject_unknown("evaluation", evaluation, {"proxy_eval", "high_fidelity_eval"})

    proxy_values = _require_mapping("evaluation.proxy_eval", evaluation.get("proxy_eval", {}))
    _reject_unknown("evaluation.proxy_eval", proxy_values, {"enabled", "num_envs", "steps", "seed_offset"})
    proxy_cfg = ProxyEvalCfg(
        enabled=_bool(proxy_values.get("enabled", True)),
        num_envs=int(proxy_values.get("num_envs", 1024)),
        steps=int(proxy_values.get("steps", 220)),
        seed_offset=int(proxy_values.get("seed_offset", 1000)),
    )

    high_values = _require_mapping("evaluation.high_fidelity_eval", evaluation.get("high_fidelity_eval", {}))
    _reject_unknown(
        "evaluation.high_fidelity_eval",
        high_values,
        {
            "enabled",
            "backend",
            "trigger",
            "terrain",
            "episodes",
            "steps",
            "sim_steps_per_control",
            "render",
        },
    )
    high_cfg = HighFidelityEvalCfg(
        enabled=_bool(high_values.get("enabled", True)),
        backend=str(high_values.get("backend", "physx_jackal")),
        trigger=str(high_values.get("trigger", "proxy_passed")),
        terrain=str(high_values.get("terrain", "strong_lunar_crater")),
        episodes=int(high_values.get("episodes", 3)),
        steps=int(high_values.get("steps", 100)),
        sim_steps_per_control=int(high_values.get("sim_steps_per_control", 8)),
        render=_bool(high_values.get("render", False)),
    )
    if high_cfg.backend not in HIGH_FIDELITY_BACKENDS:
        raise ValueError(f"Unsupported high-fidelity backend: {high_cfg.backend}")
    if high_cfg.trigger not in HIGH_FIDELITY_TRIGGERS:
        raise ValueError(f"Unsupported high-fidelity trigger: {high_cfg.trigger}")
    return EvaluationCfg(proxy_eval=proxy_cfg, high_fidelity_eval=high_cfg)


def _resolve(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _experiment_seed(raw_cfg: dict) -> int:
    experiment = raw_cfg.get("experiment", {})
    if isinstance(experiment, dict):
        return int(experiment.get("seed", 0))
    return 0


def _physx_output_path(run_dir: Path, high_cfg: HighFidelityEvalCfg) -> Path:
    suffix = f"{high_cfg.terrain}_tracking_summary"
    return run_dir / "physx" / "metrics" / f"{suffix}.json"


def run_physx_evaluation(
    *,
    config: Path,
    checkpoint: Path,
    run_dir: Path,
    high_cfg: HighFidelityEvalCfg,
) -> dict:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_physx_jackal_tracking.py"),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--terrain",
        high_cfg.terrain,
        "--profile",
        "all",
        "--steps",
        str(high_cfg.steps),
        "--sim-steps-per-control",
        str(high_cfg.sim_steps_per_control),
        "--run-dir",
        str(run_dir),
        "--output",
        str(_physx_output_path(run_dir, high_cfg)),
    ]
    if high_cfg.render:
        command.append("--render")
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    output_path = _physx_output_path(run_dir, high_cfg)
    if completed.returncode != 0:
        raise RuntimeError(
            "PhysX evaluation failed with exit code "
            f"{completed.returncode}.\nSTDOUT:\n{completed.stdout[-4000:]}\nSTDERR:\n{completed.stderr[-4000:]}"
        )
    if not output_path.exists():
        raise RuntimeError(f"PhysX evaluation did not write expected metrics: {output_path}")
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result["artifact"] = str(output_path)
    result["command"] = command
    return result


def _should_run_physx(
    high_cfg: HighFidelityEvalCfg,
    *,
    skip_physx: bool,
    proxy_passed: bool,
) -> tuple[bool, str | None]:
    if skip_physx:
        return False, "disabled_by_cli"
    if not high_cfg.enabled:
        return False, "disabled_by_config"
    if high_cfg.trigger == "manual":
        return False, "manual_trigger"
    if high_cfg.trigger == "proxy_passed" and not proxy_passed:
        return False, "proxy_not_passed"
    return True, None


def build_checkpoint_status(
    *,
    proxy_result: dict | None,
    proxy_gate: dict | None,
    physx_result: dict | None,
    physx_gate: dict | None,
    physx_skip_reason: str | None,
    mark_final_selected: bool,
    evaluation_cfg: EvaluationCfg,
) -> dict:
    state = "candidate"
    if proxy_gate and proxy_gate.get("passed"):
        state = "proxy_passed"
    if physx_result is not None:
        state = "physx_evaluated"
    if physx_gate and physx_gate.get("passed"):
        state = "physx_passed"
    if mark_final_selected and state in {"proxy_passed", "physx_passed"}:
        state = "final_selected"
    if state not in CHECKPOINT_STATES:
        raise ValueError(f"Invalid checkpoint state: {state}")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "states": list(CHECKPOINT_STATES),
        "proxy_eval": {
            "enabled": evaluation_cfg.proxy_eval.enabled,
            "gate": proxy_gate,
            "metrics": proxy_result,
        },
        "high_fidelity_eval": {
            "enabled": evaluation_cfg.high_fidelity_eval.enabled,
            "backend": evaluation_cfg.high_fidelity_eval.backend,
            "trigger": evaluation_cfg.high_fidelity_eval.trigger,
            "skip_reason": physx_skip_reason,
            "gate": physx_gate,
            "diagnostics": physx_diagnostics(physx_result) if physx_result is not None else None,
            "metrics": physx_result,
        },
    }


def update_run_manifest(run_dir: Path, status: dict, *, config: Path, checkpoint: Path) -> Path:
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment": run_dir.parent.name,
            "run": run_dir.name,
            "layout": "outputs/runs/<experiment>/<run>/<artifact_group>/...",
        }
    status_path = run_dir / "metrics" / "checkpoint_status.json"
    proxy_path = run_dir / "metrics" / "final_eval_proxy.json"
    physx_metrics = status.get("high_fidelity_eval", {}).get("metrics") or {}
    physx_artifact = physx_metrics.get("artifact")
    paths = {
        "config": _relative(config),
        "checkpoint": _relative(checkpoint),
        "checkpoint_status": _relative(status_path),
        "final_eval_proxy": _relative(proxy_path),
    }
    if physx_artifact:
        paths["high_fidelity_eval"] = _relative(_resolve(physx_artifact))
    manifest["checkpoint_evaluation"] = {
        "state": status["state"],
        "updated_at": status["generated_at"],
        "paths": paths,
    }
    manifest.setdefault("artifacts", {})
    if isinstance(manifest["artifacts"], dict):
        manifest["artifacts"].update(paths)
    _write_json(manifest_path, manifest)
    return manifest_path


def run_checkpoint_evaluation(
    *,
    config: str | Path,
    checkpoint: str | Path,
    run_dir: str | Path,
    device: str | None,
    mark_final_selected: bool = False,
    skip_physx: bool = False,
    render_physx: bool = False,
    proxy_runner: Callable[..., dict] = evaluate_proxy_checkpoint,
    physx_runner: Callable[..., dict] = run_physx_evaluation,
) -> dict:
    config_path = _resolve(config)
    checkpoint_path = _resolve(checkpoint)
    run_path = _resolve(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    raw_cfg = load_yaml(config_path)
    evaluation_cfg = parse_evaluation_config(raw_cfg)
    if render_physx:
        evaluation_cfg.high_fidelity_eval.render = True

    proxy_result = None
    proxy_gate = None
    if evaluation_cfg.proxy_eval.enabled:
        proxy_seed = _experiment_seed(raw_cfg) + evaluation_cfg.proxy_eval.seed_offset
        proxy_result = proxy_runner(
            config_path,
            checkpoint_path,
            device=device,
            num_envs=evaluation_cfg.proxy_eval.num_envs,
            steps=evaluation_cfg.proxy_eval.steps,
            seed=proxy_seed,
            run_dir=run_path,
        )
        proxy_gate = proxy_acceptance(proxy_result)

    proxy_passed = bool(proxy_gate and proxy_gate.get("passed"))
    should_run_physx, physx_skip_reason = _should_run_physx(
        evaluation_cfg.high_fidelity_eval,
        skip_physx=skip_physx,
        proxy_passed=proxy_passed,
    )
    physx_result = None
    physx_gate = None
    if should_run_physx:
        physx_result = physx_runner(
            config=config_path,
            checkpoint=checkpoint_path,
            run_dir=run_path,
            high_cfg=evaluation_cfg.high_fidelity_eval,
        )
        physx_gate = physx_acceptance(physx_result)

    status = build_checkpoint_status(
        proxy_result=proxy_result,
        proxy_gate=proxy_gate,
        physx_result=physx_result,
        physx_gate=physx_gate,
        physx_skip_reason=physx_skip_reason,
        mark_final_selected=mark_final_selected,
        evaluation_cfg=evaluation_cfg,
    )
    status["config"] = {
        "config_path": _relative(config_path),
        "checkpoint_path": _relative(checkpoint_path),
        "run_dir": _relative(run_path),
        "evaluation": {
            "proxy_eval": asdict(evaluation_cfg.proxy_eval),
            "high_fidelity_eval": asdict(evaluation_cfg.high_fidelity_eval),
        },
    }
    status_path = run_path / "metrics" / "checkpoint_status.json"
    _write_json(status_path, status)
    manifest_path = update_run_manifest(run_path, status, config=config_path, checkpoint=checkpoint_path)
    status["artifacts"] = {
        "checkpoint_status": _relative(status_path),
        "run_manifest": _relative(manifest_path),
    }
    _write_json(status_path, status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mark-final-selected", action="store_true")
    parser.add_argument("--skip-physx", action="store_true")
    parser.add_argument("--render-physx", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_checkpoint_evaluation(
        config=args.config,
        checkpoint=args.checkpoint,
        run_dir=args.run_dir,
        device=args.device,
        mark_final_selected=args.mark_final_selected,
        skip_physx=args.skip_physx,
        render_physx=args.render_physx,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
