#!/usr/bin/env python
"""Aggregate all completed exp125 B0 screen runs without changing training state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT


EXPERIMENT_ID = "exp125_decentralized_tiered_b0_pure_rl"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_screen_records(experiment_dir: Path) -> list[dict]:
    records: list[dict] = []
    for gate_path in sorted(experiment_dir.glob("*/metrics/screen_gate.json")):
        run_dir = gate_path.parents[1]
        gate = _read_json(gate_path)
        evidence = gate.get("evidence") or {}
        final_eval = evidence.get("final_eval") or {}
        terrain = evidence.get("terrain_contrast") or {}
        records.append(
            {
                "run": run_dir.name,
                "passed": bool(gate.get("passed")),
                "training_dmax_reduction": evidence.get("training_dmax_reduction"),
                "eval_dmax_reduction_ratio": final_eval.get("dmax_reduction_ratio"),
                "eval_success_rate": final_eval.get("success_rate"),
                "eval_collision_rate": final_eval.get("collision_rate"),
                "eval_timeout_rate": final_eval.get("timeout_rate"),
                "terrain_action_mse": terrain.get(
                    "action_mse_normal_vs_zero_terrain"
                ),
                "terrain_path_risk_reduction_fraction": terrain.get(
                    "path_risk_reduction_fraction"
                ),
                "failed_checks": [
                    name
                    for name, passed in (gate.get("checks") or {}).items()
                    if not passed
                ],
                "screen_gate": str(gate_path.relative_to(ROOT)),
            }
        )
    return records


def summarize(records: list[dict]) -> dict:
    def best(metric: str, *, minimize: bool = False) -> dict | None:
        candidates = [record for record in records if record.get(metric) is not None]
        if not candidates:
            return None
        selected = min(candidates, key=lambda record: record[metric]) if minimize else max(
            candidates, key=lambda record: record[metric]
        )
        return {"run": selected["run"], "value": selected[metric]}

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "completed_screen_count": len(records),
        "passed_screen_count": sum(bool(record["passed"]) for record in records),
        "any_passed": any(bool(record["passed"]) for record in records),
        "best": {
            "training_dmax_reduction": best("training_dmax_reduction"),
            "eval_dmax_reduction_ratio": best(
                "eval_dmax_reduction_ratio", minimize=True
            ),
            "eval_success_rate": best("eval_success_rate"),
            "terrain_path_risk_reduction_fraction": best(
                "terrain_path_risk_reduction_fraction"
            ),
        },
        "decision": "stop_before_40m" if not any(r["passed"] for r in records) else "review_passed_run",
        "records": records,
    }


def main() -> None:
    experiment_dir = ROOT / "outputs" / "runs" / EXPERIMENT_ID
    records = collect_screen_records(experiment_dir)
    if not records:
        raise SystemExit(f"No screen gates found under {experiment_dir}")
    payload = summarize(records)
    output = experiment_dir / "_suite" / "metrics" / "b0_screen_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
