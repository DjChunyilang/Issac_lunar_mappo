#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv_isaaclab/bin/python"
CONFIG="configs/experiment/exp012_action_scale_warmup_probe.yaml"
EXPERIMENT="exp012_action_scale_warmup_probe"
RUN_DIR="$ROOT/outputs/runs/$EXPERIMENT"
CHECKPOINT="$ROOT/outputs/checkpoints/${EXPERIMENT}.pt"
LOG_DIR="$RUN_DIR/suite_logs"

mkdir -p "$LOG_DIR" "$ROOT/outputs/checkpoints"

"$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this validation stage.")
print("cuda_available:", torch.cuda.is_available())
print("cuda_device:", torch.cuda.get_device_name(0))
PY

run_tests() {
    "$PY" -m pytest -q \
        tests/test_config_wiring.py \
        tests/test_reward.py \
        tests/test_observation.py \
        tests/test_four_rover_observation_space.py \
        tests/test_convergence_tools.py \
        tests/test_skrl_import.py \
        tests/test_skrl_mappo_semantics.py \
        2>&1 | tee "$LOG_DIR/00_core_tests.log"
}

run_training() {
    local label="$1"
    local steps="$2"
    local log_path="$LOG_DIR/${label}_${steps}.log"
    local diagnosis_path="$RUN_DIR/diagnosis_${label}_${steps}.json"
    local checkpoint_copy="$ROOT/outputs/checkpoints/${EXPERIMENT}_${label}_${steps}.pt"

    echo "=== ${label}: ${steps} timesteps ===" | tee "$log_path"
    "$PY" scripts/train_skrl_mappo.py \
        --config "$CONFIG" \
        --device cuda \
        --timesteps "$steps" \
        2>&1 | tee -a "$log_path"

    cp "$CHECKPOINT" "$checkpoint_copy"

    "$PY" scripts/diagnose_cuda_training_signal.py \
        --metrics "$RUN_DIR/metrics.jsonl" \
        > "$diagnosis_path"

    echo "diagnosis: $diagnosis_path" | tee -a "$log_path"
    echo "checkpoint_copy: $checkpoint_copy" | tee -a "$log_path"
}

run_tests
run_training "smoke" 32
run_training "probe" 20000
run_training "long_5h" 500000

echo "=== exp012 suite complete ==="
echo "metrics: $RUN_DIR/metrics.jsonl"
echo "logs: $LOG_DIR"
echo "diagnoses: $RUN_DIR/diagnosis_*.json"
