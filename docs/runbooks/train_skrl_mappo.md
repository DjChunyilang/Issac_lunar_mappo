# SKRL-MAPPO 训练诊断操作手册

本 runbook 用于 SKRL-MAPPO CUDA contract、动作尺度诊断和短训练信号检查。这里的结果只证明工程链路或诊断方向，不作为 strict pass。

## 前置检查

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_config_wiring.py \
  tests/test_reward.py \
  tests/test_observation.py \
  tests/test_four_rover_observation_space.py \
  tests/test_convergence_tools.py \
  tests/test_skrl_import.py \
  tests/test_skrl_mappo_semantics.py
```

如果要跑 CUDA stage，先确认 CUDA 可见：

```bash
.venv_isaaclab/bin/python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this validation stage.")
print(torch.cuda.get_device_name(0))
PY
```

## CUDA Contract

```bash
.venv_isaaclab/bin/python scripts/run_cuda_training_validation.py
```

该命令使用 `configs/experiment/exp_cuda_contract.yaml`，依次运行 `32 / 512 / 5000` timesteps，并检查：

- `scripts/train_skrl_mappo.py --device cuda` 必须在 CUDA 不可用时失败。
- 每段训练都追加 telemetry JSONL。
- checkpoint metadata 必须包含 `training_semantics`、`shared_actor`、`centralized_critic`、`shared_value`、`observation_schema_version` 和 `device`。
- `nan_flag` 必须为 false。

产物路径：

```text
outputs/runs/cuda_training_validation_summary.json
outputs/runs/exp_cuda_contract/metrics.jsonl
outputs/checkpoints/exp_cuda_contract.pt
```

`success_rate_final` 可以为 0；这个 contract 不要求策略收敛。

## exp012 Action-Scale Suite

```bash
bash scripts/run_exp012_action_scale_suite.sh
```

该脚本使用 `configs/experiment/exp012_action_scale_warmup_probe.yaml`，先跑核心测试，再运行 `32`、`20000` 和 `500000` timesteps 三段 CUDA 探针。`500000` 是长预算，不应在未生成完整诊断 JSON 前写成实验结论。

产物路径：

```text
outputs/runs/exp012_action_scale_warmup_probe/metrics.jsonl
outputs/runs/exp012_action_scale_warmup_probe/diagnosis_<label>_<timesteps>.json
outputs/runs/exp012_action_scale_warmup_probe/suite_logs/
outputs/checkpoints/exp012_action_scale_warmup_probe_<label>_<timesteps>.pt
```

## 单次手动训练

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp012_action_scale_warmup_probe.yaml \
  --device cuda \
  --timesteps 20000
```

训练会输出：

- action telemetry：normalized action、`rho/beta` 物理尺度、near-zero 和 saturation 比例。
- reward telemetry：各 reward component 的 raw、weight、contribution 和占比。
- done reason：success、timeout、collision、safety 和 other 计数。
- random baseline 和 post-training deterministic eval。
- checkpoint metadata，包括 observation schema 和 SKRL-MAPPO 语义字段。

## 诊断 JSON

```bash
.venv_isaaclab/bin/python scripts/diagnose_cuda_training_signal.py \
  --metrics outputs/runs/exp012_action_scale_warmup_probe/metrics.jsonl \
  > outputs/runs/exp012_action_scale_warmup_probe/diagnosis_probe_20000.json
```

重点读取：

- `judgement`
- `success_rate`
- `mean_pairwise_distance`
- `mean_oracle_distance`
- `action_scale_summary.flags`
- `reward_component_summary`
- `done_reason_summary`
- `next_experiment_focus`

如果出现 `normalized_action_saturation`、`forward_high_saturation`、`forward_low_saturation`、`turn_saturation` 或物理尺度饱和，下一步优先做 action scale 消融；如果 `success_rate.max` 仍为 0，继续做 success gate reachability 诊断。不要只根据 reward 上升或距离短暂改善宣布通过。

## Git 规则

不要提交 `outputs/` 下的 checkpoint、JSONL、诊断 JSON 或日志。需要保留结论时，更新对应实验文档和 `docs/current_status.md`。
