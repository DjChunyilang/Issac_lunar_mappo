# SKRL-MAPPO 训练诊断操作手册

本 runbook 用于 SKRL-MAPPO proxy CUDA contract、动作尺度诊断和短训练信号检查。这里的训练结果只证明 proxy 工程链路或诊断方向，不作为 Isaac / PhysX 物理训练结果。

## 前置检查

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_config_wiring.py \
  tests/test_reward.py \
  tests/test_observation.py \
  tests/test_four_rover_observation_space.py \
  tests/test_convergence_tools.py \
  tests/test_checkpoint_evaluation.py \
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
- checkpoint metadata 还必须包含 `actor_obs_dim` 和 `critic_state_dim`；旧 schema 或维度不匹配 checkpoint 会明确拒绝。
- `nan_flag` 必须为 false。

产物路径：

```text
outputs/runs/cuda_training_validation_summary.json
outputs/runs/exp_cuda_contract/metrics.jsonl
outputs/checkpoints/exp_cuda_contract.pt
```

`success_rate_final` 可以为 0；这个 contract 不要求策略收敛。

## exp014 Terrain-Grid Observation Probe

先运行完整测试：

```bash
.venv_isaaclab/bin/python -m pytest -q -ra
```

再运行弱 lunar crater CUDA 探针：

```bash
.venv_isaaclab/bin/python scripts/run_terrain_observation_validation.py \
  --device cuda \
  --timesteps 5000
```

该脚本使用 `configs/experiment/exp014_terrain_grid_observation_probe.yaml`，检查：

- observation schema 为 `ego_v3_local_terrain_grid`；
- Actor / Critic 维度为 `86 / 54`；
- 训练无 NaN/Inf；
- policy 参数和第一层 terrain 输入列权重发生更新；
- post-training 动作非退化；
- 弱月面局部地形观测不是全零。

产物路径：

```text
outputs/runs/exp014_terrain_grid_observation_probe/metrics.jsonl
outputs/runs/exp014_terrain_grid_observation_probe/terrain_observation_validation_summary.json
outputs/checkpoints/exp014_terrain_grid_observation_probe.pt
```

该探针只验证工程与训练信号，不要求 strict convergence。

## exp012 Action-Scale Suite

```bash
bash scripts/run_exp012_action_scale_suite.sh
```

该脚本使用 `configs/experiment/exp012_action_scale_warmup_probe.yaml`，先跑核心测试，再运行 `32`、`20000` 和 `500000` timesteps 三段 CUDA 探针。`diagnosis_long_5h_500000.json` 可用于训练信号分析，但它不是 strict acceptance；不要只根据 reward 或 distance 曲线写成通过。

产物路径：

```text
outputs/runs/exp012_action_scale_warmup_probe/metrics.jsonl
outputs/runs/exp012_action_scale_warmup_probe/diagnosis_<label>_<timesteps>.json
outputs/runs/exp012_action_scale_warmup_probe/suite_logs/
outputs/checkpoints/exp012_action_scale_warmup_probe_<label>_<timesteps>.pt
```

## exp013 Action-Scale Ablation Suite

```bash
bash scripts/run_exp013_action_scale_ablation_suite.sh
```

该脚本使用：

```text
configs/experiment/exp013_action_scale_rho06_beta45.yaml
configs/experiment/exp013_action_scale_rho05_beta30.yaml
```

默认运行 `rho06_beta45` 的 `32 / 20000 / 120000` timesteps，以及 `rho05_beta30` 的 `32 / 20000` timesteps。它会把每个 run 的配置快照、checkpoint、训练 JSONL、diagnosis、final proxy eval、checkpoint status 和 proxy GIF 写入标准 run 目录。

产物路径：

```text
outputs/runs/exp013_action_scale_ablation/_suite/metrics/suite_summary.json
outputs/runs/exp013_action_scale_ablation/<run_id>/config/experiment.yaml
outputs/runs/exp013_action_scale_ablation/<run_id>/checkpoints/best.pt
outputs/runs/exp013_action_scale_ablation/<run_id>/metrics/train_metrics.jsonl
outputs/runs/exp013_action_scale_ablation/<run_id>/metrics/summary.json
outputs/runs/exp013_action_scale_ablation/<run_id>/metrics/diagnosis.json
outputs/runs/exp013_action_scale_ablation/<run_id>/metrics/final_eval_proxy.json
outputs/runs/exp013_action_scale_ablation/<run_id>/metrics/checkpoint_status.json
outputs/runs/exp013_action_scale_ablation/<run_id>/videos/proxy_eval_rollout.gif
```

最新结论：`rho06_beta45_seed7_probe_20000` 是本轮最佳短探针，但 final eval success_rate 仍为 `0.0`；`rho06_beta45_seed7_long_120000` 没有带来 success 改善，动作饱和加重。因此不要继续直接拉长相同配置训练预算，下一步应做 success gate reachability 和动作饱和机制诊断。

补充 teacher reachability sanity 的最新结论：

```text
outputs/runs/exp013_action_scale_ablation/_suite/metrics/teacher_reachability_summary.json
```

当前 `rho=0.6, beta=pi/4, 100 steps` 对 scripted teacher 也几乎不可达；恢复到 `220` steps 后 teacher success_rate 为 `1.0`。因此下一轮应先构造 teacher-reachable 配置，再继续 SKRL-MAPPO 训练诊断。

查看 suite 汇总：

```bash
.venv_isaaclab/bin/python -m json.tool \
  outputs/runs/exp013_action_scale_ablation/_suite/metrics/suite_summary.json
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

## Checkpoint 统一评估

训练完成后运行：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/exp013_action_scale_rho06_beta45.yaml \
  --checkpoint outputs/runs/exp013_action_scale_ablation/<run_id>/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/exp013_action_scale_ablation/<run_id>
```

该命令根据配置中的 `evaluation:` block 先执行 proxy final eval，再写入 checkpoint 状态。若 proxy strict gate 未通过，PhysX 会被标记为 `proxy_not_passed` 并跳过；若通过，则按配置执行低频 high-fidelity closed-loop eval。

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
