#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv_isaaclab/bin/python"
EXPERIMENT_ID="exp013_action_scale_ablation"
RUN_ROOT="$ROOT/outputs/runs/$EXPERIMENT_ID"
SUITE_DIR="$RUN_ROOT/_suite"
SUITE_LOG_DIR="$SUITE_DIR/logs"

RENDER_PROXY_GIF="${RENDER_PROXY_GIF:-1}"
RENDER_STEPS="${RENDER_STEPS:-120}"
RUN_CONSERVATIVE_LONG="${RUN_CONSERVATIVE_LONG:-0}"

mkdir -p \
  "$SUITE_DIR/metrics" \
  "$SUITE_DIR/checkpoints" \
  "$SUITE_DIR/figures" \
  "$SUITE_DIR/videos" \
  "$SUITE_LOG_DIR"

"$PY" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this validation stage.")
print("cuda_available:", torch.cuda.is_available())
print("cuda_device:", torch.cuda.get_device_name(0))
PY

if [[ "${SKIP_CORE_TESTS:-0}" != "1" ]]; then
  "$PY" -m pytest -q \
    tests/test_config_wiring.py \
    tests/test_reward.py \
    tests/test_observation.py \
    tests/test_four_rover_observation_space.py \
    tests/test_convergence_tools.py \
    tests/test_skrl_import.py \
    tests/test_skrl_mappo_semantics.py \
    2>&1 | tee "$SUITE_LOG_DIR/00_core_tests.log"
fi

config_value() {
  local config="$1"
  local dotted_key="$2"
  "$PY" - "$config" "$dotted_key" <<'PY'
import sys
import yaml

config_path, dotted_key = sys.argv[1], sys.argv[2]
with open(config_path, "r", encoding="utf-8") as stream:
    data = yaml.safe_load(stream) or {}
value = data
for part in dotted_key.split("."):
    value = value[part]
print(value)
PY
}

copy_latest_metrics() {
  local source_metrics="$1"
  local target_metrics="$2"
  local diagnosis_path="$3"
  local summary_path="$4"
  local managed_run_id="$5"
  "$PY" - "$source_metrics" "$target_metrics" "$diagnosis_path" "$summary_path" "$managed_run_id" <<'PY'
import json
import sys
from pathlib import Path

source_metrics = Path(sys.argv[1])
target_metrics = Path(sys.argv[2])
diagnosis_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
managed_run_id = sys.argv[5]

diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
source_run_id = diagnosis.get("run_id")
rows = [
    json.loads(line)
    for line in source_metrics.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if source_run_id:
    rows = [row for row in rows if row.get("run_id") == source_run_id]
target_metrics.parent.mkdir(parents=True, exist_ok=True)
target_metrics.write_text(
    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    encoding="utf-8",
)
latest = rows[-1] if rows else {}
summary = {
    "managed_run_id": managed_run_id,
    "source_run_id": source_run_id,
    "row_count": len(rows),
    "timesteps": latest.get("timesteps"),
    "device": latest.get("device"),
    "checkpoint_path": latest.get("checkpoint_path"),
    "metrics_path": str(target_metrics),
    "judgement": diagnosis.get("judgement"),
    "next_experiment_focus": diagnosis.get("next_experiment_focus"),
    "mean_pairwise_distance": diagnosis.get("mean_pairwise_distance"),
    "mean_oracle_distance": diagnosis.get("mean_oracle_distance"),
    "success_rate": diagnosis.get("success_rate"),
    "action_scale_summary": diagnosis.get("action_scale_summary"),
    "done_reason_summary": diagnosis.get("done_reason_summary"),
    "reward_component_summary": diagnosis.get("reward_component_summary"),
    "post_training_eval": diagnosis.get("post_training_eval"),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
PY
}

write_run_manifest() {
  local run_dir="$1"
  local managed_run_id="$2"
  local config="$3"
  local steps="$4"
  local train_log="$5"
  "$PY" - "$run_dir" "$managed_run_id" "$config" "$steps" "$train_log" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
managed_run_id = sys.argv[2]
config = sys.argv[3]
steps = int(sys.argv[4])
train_log = sys.argv[5]

paths = {
    "config": str(run_dir / "config" / "experiment.yaml"),
    "checkpoint": str(run_dir / "checkpoints" / "best.pt"),
    "train_metrics": str(run_dir / "metrics" / "train_metrics.jsonl"),
    "summary": str(run_dir / "metrics" / "summary.json"),
    "diagnosis": str(run_dir / "metrics" / "diagnosis.json"),
    "final_eval_proxy": str(run_dir / "metrics" / "final_eval_proxy.json"),
    "checkpoint_status": str(run_dir / "metrics" / "checkpoint_status.json"),
    "proxy_gif": str(run_dir / "videos" / "proxy_eval_rollout.gif"),
    "proxy_gif_metrics": str(run_dir / "metrics" / "proxy_rollout_render.json"),
    "train_log": train_log,
    "checkpoint_evaluation_stdout": str(run_dir / "metrics" / "checkpoint_evaluation_stdout.json"),
}
manifest = {
    "experiment_id": "exp013_action_scale_ablation",
    "run_id": managed_run_id,
    "timesteps": steps,
    "device": "cuda",
    "config_source": config,
    "paths": paths,
    "commands": {
        "train": f".venv_isaaclab/bin/python scripts/train_skrl_mappo.py --config {config} --device cuda --timesteps {steps}",
        "diagnose": f".venv_isaaclab/bin/python scripts/diagnose_cuda_training_signal.py --metrics {paths['train_metrics']}",
        "checkpoint_evaluation": f".venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py --config {config} --checkpoint {paths['checkpoint']} --device cuda --run-dir {run_dir}",
        "render_proxy": f".venv_isaaclab/bin/python scripts/render_skrl_proxy_rollout.py --config {config} --checkpoint {paths['checkpoint']} --device cpu --steps 120 --run-dir {run_dir}",
        "render_physx": f".venv_isaaclab/bin/python scripts/evaluate_physx_jackal_tracking.py --config {config} --checkpoint {paths['checkpoint']} --terrain flat --profile all --steps 80 --render --run-dir {run_dir}",
    },
}
run_dir.joinpath("run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
PY
}

run_training_case() {
  local case_id="$1"
  local config="$2"
  local label="$3"
  local steps="$4"
  local managed_run_id="${case_id}_seed7_${label}_${steps}"
  local run_dir="$RUN_ROOT/$managed_run_id"
  local train_log="$SUITE_LOG_DIR/${managed_run_id}.log"
  local experiment_name
  local checkpoint_name
  local source_metrics
  local source_checkpoint

  experiment_name="$(config_value "$config" "experiment.name")"
  checkpoint_name="$(config_value "$config" "experiment.checkpoint_name")"
  source_metrics="$ROOT/outputs/runs/$experiment_name/metrics.jsonl"
  source_checkpoint="$ROOT/outputs/checkpoints/$checkpoint_name"

  mkdir -p \
    "$run_dir/config" \
    "$run_dir/checkpoints" \
    "$run_dir/metrics" \
    "$run_dir/figures" \
    "$run_dir/videos" \
    "$run_dir/tensorboard" \
    "$run_dir/tensorboard_curated" \
    "$run_dir/physx/metrics" \
    "$run_dir/physx/figures" \
    "$run_dir/physx/videos"
  cp "$config" "$run_dir/config/experiment.yaml"

  echo "=== $managed_run_id ===" | tee "$train_log"
  "$PY" scripts/train_skrl_mappo.py \
    --config "$config" \
    --device cuda \
    --timesteps "$steps" \
    2>&1 | tee -a "$train_log"

  cp "$source_checkpoint" "$run_dir/checkpoints/best.pt"
  cp "$source_checkpoint" "$SUITE_DIR/checkpoints/${managed_run_id}.pt"

  "$PY" scripts/diagnose_cuda_training_signal.py \
    --metrics "$source_metrics" \
    | tee "$run_dir/metrics/diagnosis.json"

  copy_latest_metrics \
    "$source_metrics" \
    "$run_dir/metrics/train_metrics.jsonl" \
    "$run_dir/metrics/diagnosis.json" \
    "$run_dir/metrics/summary.json" \
    "$managed_run_id"

  "$PY" scripts/run_checkpoint_evaluation.py \
    --config "$config" \
    --checkpoint "$run_dir/checkpoints/best.pt" \
    --device cuda \
    --run-dir "$run_dir" \
    > "$run_dir/metrics/checkpoint_evaluation_stdout.json"

  if [[ "$RENDER_PROXY_GIF" != "0" ]]; then
    "$PY" scripts/render_skrl_proxy_rollout.py \
      --config "$config" \
      --checkpoint "$run_dir/checkpoints/best.pt" \
      --device cpu \
      --steps "$RENDER_STEPS" \
      --run-dir "$run_dir" \
      > "$run_dir/metrics/proxy_rollout_render_stdout.json"
  fi

  write_run_manifest "$run_dir" "$managed_run_id" "$config" "$steps" "$train_log"
  echo "run_dir: $run_dir" | tee -a "$train_log"
}

run_training_case "rho06_beta45" "configs/experiment/exp013_action_scale_rho06_beta45.yaml" "smoke" 32
run_training_case "rho05_beta30" "configs/experiment/exp013_action_scale_rho05_beta30.yaml" "smoke" 32
run_training_case "rho06_beta45" "configs/experiment/exp013_action_scale_rho06_beta45.yaml" "probe" 20000
run_training_case "rho05_beta30" "configs/experiment/exp013_action_scale_rho05_beta30.yaml" "probe" 20000
run_training_case "rho06_beta45" "configs/experiment/exp013_action_scale_rho06_beta45.yaml" "long" 120000

if [[ "$RUN_CONSERVATIVE_LONG" == "1" ]]; then
  run_training_case "rho05_beta30" "configs/experiment/exp013_action_scale_rho05_beta30.yaml" "long" 120000
fi

"$PY" - "$RUN_ROOT" "$SUITE_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
suite_dir = Path(sys.argv[2])
items = []
for manifest_path in sorted(run_root.glob("*/run_manifest.json")):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = manifest_path.parent
    summary_path = run_dir / "metrics" / "summary.json"
    final_eval_path = run_dir / "metrics" / "final_eval_proxy.json"
    status_path = run_dir / "metrics" / "checkpoint_status.json"
    render_path = run_dir / "metrics" / "proxy_rollout_render.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    final_eval = json.loads(final_eval_path.read_text(encoding="utf-8")) if final_eval_path.exists() else {}
    checkpoint_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    render = json.loads(render_path.read_text(encoding="utf-8")) if render_path.exists() else {}
    action_summary = summary.get("action_scale_summary") or {}
    items.append(
        {
            "run_id": manifest.get("run_id"),
            "timesteps": manifest.get("timesteps"),
            "checkpoint": manifest.get("paths", {}).get("checkpoint"),
            "train_metrics": manifest.get("paths", {}).get("train_metrics"),
            "diagnosis": manifest.get("paths", {}).get("diagnosis"),
            "proxy_gif": render.get("gif_path") or manifest.get("paths", {}).get("proxy_gif"),
            "judgement": summary.get("judgement"),
            "next_experiment_focus": summary.get("next_experiment_focus"),
            "success_rate": summary.get("success_rate"),
            "mean_pairwise_distance": summary.get("mean_pairwise_distance"),
            "mean_oracle_distance": summary.get("mean_oracle_distance"),
            "action_scale_flags": action_summary.get("flags"),
            "action_saturation_fraction": action_summary.get("action_saturation_fraction"),
            "final_eval_success_rate": final_eval.get("success_rate"),
            "final_eval_mean_reward": final_eval.get("mean_reward"),
            "checkpoint_state": checkpoint_status.get("state"),
            "proxy_gate": (checkpoint_status.get("proxy_eval") or {}).get("gate"),
            "high_fidelity_skip_reason": (checkpoint_status.get("high_fidelity_eval") or {}).get("skip_reason"),
        }
    )
suite_summary = {
    "experiment_id": "exp013_action_scale_ablation",
    "status": "complete",
    "runs": items,
    "selection_hint": "Prefer a long run only if action saturation drops while distance and eval success do not regress.",
}
suite_metrics = suite_dir / "metrics" / "suite_summary.json"
suite_metrics.write_text(json.dumps(suite_summary, indent=2, sort_keys=True), encoding="utf-8")
suite_manifest = {
    "experiment_id": "exp013_action_scale_ablation",
    "paths": {
        "suite_summary": str(suite_metrics),
        "logs": str(suite_dir / "logs"),
        "checkpoints": str(suite_dir / "checkpoints"),
    },
    "runs": [item["run_id"] for item in items],
}
(suite_dir / "run_manifest.json").write_text(json.dumps(suite_manifest, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(suite_summary, indent=2, sort_keys=True))
PY

echo "=== exp013 suite complete ==="
echo "suite_summary: $SUITE_DIR/metrics/suite_summary.json"
echo "suite_logs: $SUITE_LOG_DIR"
