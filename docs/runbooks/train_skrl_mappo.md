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

## exp016 Shared-Joint 分阶段验证

运行前要求至少保留 6 GB 空闲显存：

```bash
.venv_isaaclab/bin/python scripts/run_exp016_shared_mappo_training.py \
  --config configs/experiment/exp016_shared_mappo_comm12.yaml \
  --minimum-free-gpu-mb 6144
```

该 runner 依次执行：

1. `shared_update_probe_seed23_512k`：验证单 optimizer 和两次 joint update。
2. `local_teacher_bc100_seed23`：保存并独立评估 BC-only checkpoint。
3. 只有 BC probe 通过才执行 `screen_seed23_2m`。
4. 只有 screen 通过才从随机初始化执行 `formal_seed23_8m`。

机器可读总结果：

```text
outputs/runs/exp016_shared_mappo_comm12/_suite/metrics/suite_summary.json
outputs/runs/exp016_shared_mappo_comm12/_suite/metrics/strict_acceptance.json
```

`communication_radius=12 m` 只用于当前诊断，不应直接作为最终通信系统设定。

## exp017 Pure RL 连续 20M

exp017 不使用 BC，也不会因为中间 gate 失败而停止。一次训练连续运行 10240 timesteps，并在 2M、8M、20M environment steps 保存和评估里程碑。

启动前先运行：

```bash
.venv_isaaclab/bin/python -m pytest -q -ra
```

后台启动：

```bash
mkdir -p outputs/runs/exp017_shared_mappo_pure_rl_comm12/_launcher

nohup .venv_isaaclab/bin/python scripts/run_exp017_pure_rl_long.py \
  --config configs/experiment/exp017_shared_mappo_pure_rl_comm12.yaml \
  --device cuda \
  --timesteps 10240 \
  --minimum-free-gpu-mb 6144 \
  > outputs/runs/exp017_shared_mappo_pure_rl_comm12/_launcher/train.log 2>&1 &
```

监控：

```bash
tail -f outputs/runs/exp017_shared_mappo_pure_rl_comm12/_launcher/train.log
```

当前由用户级 transient service 托管时，也可以检查：

```bash
systemctl --user status exp017-pure-rl.service --no-pager
```

训练完成后优先读取：

```text
outputs/runs/exp017_shared_mappo_pure_rl_comm12/_suite/metrics/milestones.json
outputs/runs/exp017_shared_mappo_pure_rl_comm12/_suite/metrics/suite_summary.json
outputs/runs/exp017_shared_mappo_pure_rl_comm12/_suite/metrics/strict_acceptance.json
```

`pure_rl_long` 候选排序忽略 timeout，但 timeout 仍写入所有评估结果。正式 strict gate 仍要求 `timeout_rate=0`，单 seed 通过也只能记为 candidate。

## exp018 随机增强地形

exp018 在每个并行环境 reset 时独立重采样地图，episode 内地图保持固定。先运行专项与完整测试：

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_terrain_features.py \
  tests/test_reward.py \
  tests/test_exp018_training.py

.venv_isaaclab/bin/python -m pytest -q -ra
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp018_randomized_terrain_pure_rl.yaml \
  --device cpu \
  --num-envs 8 \
  --rollout-steps 4 \
  --timesteps 8 \
  --checkpoint-interval 4 \
  --run-name smoke_cpu_exp018
```

CUDA shared-update smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp018_randomized_terrain_pure_rl.yaml \
  --device cuda \
  --num-envs 256 \
  --rollout-steps 32 \
  --timesteps 64 \
  --checkpoint-interval 32 \
  --run-name smoke_cuda_exp018
```

验收至少检查：

- 不同并行环境的 terrain phase 和 transform 不完全相同；
- reset 子集不会改变其他环境的地图；
- Actor / Critic 仍为 `86 / 54`；
- 一个 optimizer、每个 rollout 一次 joint update；
- policy 参数和 terrain 输入列权重更新；
- reward、observation 和 action 无 NaN/Inf；
- GIF 的 height map 与 rollout 使用同一份 terrain runtime。

标准 smoke 产物：

```text
outputs/runs/exp018_randomized_terrain_pure_rl/smoke_cpu_exp018/
outputs/runs/exp018_randomized_terrain_pure_rl/smoke_cuda_exp018/
```

seed23 连续 20M 已完成。若以后重新启动同类长跑，可用用户级 transient service 或 `nohup` 托管；历史 service 名为：

```bash
systemctl --user status exp018-randomized-terrain-20m.service --no-pager
```

读取训练末尾 telemetry：

```bash
tail -n 1 \
  outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/metrics/train_metrics.jsonl
```

checkpoint 每 1024 timesteps 写入，20M run 共有 10 个候选 checkpoint：

```text
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/checkpoints/
```

快速查看候选里程碑：

```bash
jq -r '.evaluations[] | [
  .candidate_timestep,
  (.candidate_timestep * 2048),
  .dmax_reduction_ratio,
  .success_rate,
  .collision_rate,
  .timeout_rate
] | @tsv' \
  outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/metrics/eval_metrics.json
```

最终结论以这些文件为准：

```text
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/metrics/final_eval_proxy.json
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/metrics/strict_acceptance.json
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/metrics/checkpoint_status.json
```

当前结果：20M best checkpoint 的 final eval dmax ratio `0.1417`、success `0.9609` 已达标，但 collision `0.0352` 和 timeout `0.0088` 未过 strict gate。因此 exp018 是随机地形 candidate / 安全失败分析，不是 strict pass。

## exp019 安全成功门控 + 路径级地形风险

exp019 基于 exp018，新增 `success_thresholds.min_pairwise_distance=0.42` 和路径级 terrain risk。先运行专项测试：

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_termination.py \
  tests/test_reward.py \
  tests/test_terrain_features.py \
  tests/test_exp019_training.py

.venv_isaaclab/bin/python -m pytest -q -ra
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp019_randomized_terrain_safe_path_risk.yaml \
  --device cpu \
  --num-envs 8 \
  --rollout-steps 4 \
  --timesteps 8 \
  --checkpoint-interval 4 \
  --run-name smoke_cpu_exp019 \
  --output-layout run
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp019_randomized_terrain_safe_path_risk.yaml \
  --device cuda \
  --num-envs 256 \
  --rollout-steps 32 \
  --timesteps 64 \
  --checkpoint-interval 32 \
  --run-name smoke_cuda_exp019 \
  --output-layout run
```

长训练使用用户级 transient service：

```bash
ROOT=$(pwd)
mkdir -p outputs/runs/exp019_randomized_terrain_safe_path_risk/_launcher

systemd-run --user --unit exp019-safe-path-risk-20m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${ROOT}/outputs/runs/exp019_randomized_terrain_safe_path_risk/_launcher/train.log \
  --property=StandardError=append:${ROOT}/outputs/runs/exp019_randomized_terrain_safe_path_risk/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp019_randomized_terrain_safe_path_risk.yaml \
    --device cuda \
    --timesteps 10240 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_20m_safe_path_risk \
    --rollout-steps 32 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 220 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate pure_rl_long
```

`systemd-run --property=StandardOutput=append:` 要使用绝对路径；相对路径可能被 systemd 拒绝。

监控：

```bash
systemctl --user status exp019-safe-path-risk-20m.service --no-pager

tail -n 1 \
  outputs/runs/exp019_randomized_terrain_safe_path_risk/pure_rl_seed23_20m_safe_path_risk/metrics/train_metrics.jsonl | jq '{
    timesteps,
    success_rate,
    safe_success_rate,
    collision_done,
    timeout_done,
    final_nearest_neighbor_distance,
    min_pairwise_ok_rate,
    path_terrain_risk_mean,
    path_terrain_risk_max
  }'
```

训练后重点读取：

```text
outputs/runs/exp019_randomized_terrain_safe_path_risk/pure_rl_seed23_20m_safe_path_risk/metrics/final_eval_proxy.json
outputs/runs/exp019_randomized_terrain_safe_path_risk/pure_rl_seed23_20m_safe_path_risk/metrics/strict_acceptance.json
outputs/runs/exp019_randomized_terrain_safe_path_risk/pure_rl_seed23_20m_safe_path_risk/metrics/eval_metrics.json
```

本轮 exp019 seed23 20M 已完成，但 strict 未通过。关键结论：

- 训练工程正常：`joint_update_count=320`、`optimizer_count=1`、无 NaN，terrain 输入权重更新。
- `ppo_timestep_010240.pt` 有集合趋势：dmax ratio `0.1552`、success `0.6201`，但 collision `0.1279`、timeout `0.2627`。
- 当前 `best.pt` 由 collision-aware 排序选中 `ppo_timestep_001024.pt`，5 seed 复验均值为 success `0.0143`、collision `0.0801`、timeout `0.9082`。
- path risk telemetry 稳定非零，但软惩罚没有诱导出可靠绕障。

复验/GIF 输出：

```text
outputs/runs/exp019_randomized_terrain_safe_path_risk/pure_rl_seed23_20m_safe_path_risk/metrics/multi_eval_20260624_115351/
outputs/runs/exp019_randomized_terrain_safe_path_risk/pure_rl_seed23_20m_safe_path_risk/videos/multi_eval_20260624_115351/
outputs/runs/exp019_randomized_terrain_safe_path_risk/_suite/metrics/
```

## exp020 地形/安全感知子目标过滤器

exp020 基于 exp019，在 actor 输出 `[rho, beta]` 后、轨迹生成前启用 `planner.subgoal_filter`。专项测试：

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_subgoal_filter.py \
  tests/test_exp020_training.py \
  tests/test_exp019_training.py \
  tests/test_terrain_features.py \
  tests/test_reward.py \
  tests/test_termination.py

.venv_isaaclab/bin/python -m pytest -q -ra
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp020_randomized_terrain_subgoal_filter.yaml \
  --device cpu \
  --num-envs 8 \
  --rollout-steps 4 \
  --timesteps 8 \
  --checkpoint-interval 4 \
  --run-name smoke_cpu_exp020 \
  --output-layout run \
  --selection-gate safe_progress_long
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp020_randomized_terrain_subgoal_filter.yaml \
  --device cuda \
  --num-envs 256 \
  --rollout-steps 32 \
  --timesteps 64 \
  --checkpoint-interval 32 \
  --eval-num-envs 256 \
  --eval-steps 220 \
  --run-name smoke_cuda_exp020 \
  --output-layout run \
  --selection-gate safe_progress_long
```

长训练命令：

```bash
ROOT=$(pwd)
mkdir -p outputs/runs/exp020_randomized_terrain_subgoal_filter/_launcher

systemd-run --user --unit exp020-subgoal-filter-20m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${ROOT}/outputs/runs/exp020_randomized_terrain_subgoal_filter/_launcher/train.log \
  --property=StandardError=append:${ROOT}/outputs/runs/exp020_randomized_terrain_subgoal_filter/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp020_randomized_terrain_subgoal_filter.yaml \
    --device cuda \
    --timesteps 10240 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_20m_subgoal_filter \
    --rollout-steps 32 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 220 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate safe_progress_long
```

训练曲线和候选评估曲线：

```text
outputs/runs/exp020_randomized_terrain_subgoal_filter/pure_rl_seed23_20m_subgoal_filter/figures/training_curves.png
outputs/runs/exp020_randomized_terrain_subgoal_filter/pure_rl_seed23_20m_subgoal_filter/figures/candidate_eval_curves.png
outputs/runs/exp020_randomized_terrain_subgoal_filter/_suite/figures/
```

本轮 exp020 seed23 20M 已完成，但 strict 未通过。关键结论：

- 工程链路正常：`joint_update_count=320`、`optimizer_count=1`、无 NaN，terrain 输入权重更新。
- filter 有效降低地形路径风险：5 seed raw path risk mean `0.3815`，filtered path risk mean `0.3187`。
- 策略失败：5 seed success `0.0`、collision `0.0498`、timeout `0.9506`。
- 结论是 hard filter 过强，导致安全绕行/徘徊而非集合；下一轮应做课程化或软约束。

复验/GIF 输出：

```text
outputs/runs/exp020_randomized_terrain_subgoal_filter/pure_rl_seed23_20m_subgoal_filter/metrics/multi_eval_20260625_101520/
outputs/runs/exp020_randomized_terrain_subgoal_filter/pure_rl_seed23_20m_subgoal_filter/videos/multi_eval_20260625_101520/
outputs/runs/exp020_randomized_terrain_subgoal_filter/_suite/metrics/
```

## exp021：课程化/软化子目标过滤器

exp021 基于 exp020，但不再从一开始强制执行最低风险候选。前 `2048` timesteps 保留 raw action，只记录 filter telemetry 和 raw-risk / deviation 辅助惩罚；之后 `4096` timesteps 内逐步提高 filter 介入概率和 score 权重。

专项测试：

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_subgoal_filter.py \
  tests/test_exp019_training.py \
  tests/test_exp020_training.py \
  tests/test_exp021_training.py \
  tests/test_reward.py \
  tests/test_termination.py

.venv_isaaclab/bin/python -m pytest -q -ra
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp021_randomized_terrain_filter_curriculum.yaml \
  --device cpu \
  --num-envs 8 \
  --rollout-steps 4 \
  --timesteps 8 \
  --checkpoint-interval 4 \
  --run-name smoke_cpu_exp021 \
  --output-layout run \
  --selection-gate balanced_progress_long
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp021_randomized_terrain_filter_curriculum.yaml \
  --device cuda \
  --num-envs 256 \
  --rollout-steps 32 \
  --timesteps 64 \
  --checkpoint-interval 32 \
  --eval-num-envs 256 \
  --eval-steps 220 \
  --run-name smoke_cuda_exp021 \
  --output-layout run \
  --selection-gate balanced_progress_long
```

正式 20M 长训练：

```bash
ROOT=$(pwd)
mkdir -p outputs/runs/exp021_randomized_terrain_filter_curriculum/_launcher

systemd-run --user --unit exp021-filter-curriculum-20m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${ROOT}/outputs/runs/exp021_randomized_terrain_filter_curriculum/_launcher/train.log \
  --property=StandardError=append:${ROOT}/outputs/runs/exp021_randomized_terrain_filter_curriculum/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp021_randomized_terrain_filter_curriculum.yaml \
    --device cuda \
    --timesteps 10240 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_20m_filter_curriculum \
    --rollout-steps 32 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 220 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate balanced_progress_long
```

监控：

```bash
tail -f outputs/runs/exp021_randomized_terrain_filter_curriculum/_launcher/train.log
```

本轮 exp021 seed23 20M 已完成，但 strict 未通过。关键结论：

- 课程化 filter 恢复集合进度：5 seed mean success `0.6361`，dmax ratio `0.1460`。
- timeout 明显低于 exp020：`0.1967` vs exp020 `0.9506`。
- 碰撞成为主失败项：collision `0.1746`，远高于 strict `0.02`，也高于 exp020 `0.0498`。

## exp022 endpoint/path safety constrained curriculum filter

目的：保留 exp021 的集合趋势，同时在 Actor 输出后、轨迹生成前加入更直接的 endpoint/path safety constraint，针对 exp021 的高 collision 失败模式。

关键配置：

```text
config: configs/experiment/exp022_randomized_terrain_endpoint_safety_filter.yaml
experiment: exp022_randomized_terrain_endpoint_safety_filter
run: pure_rl_seed23_20m_endpoint_safety
filter mode: terrain_safe_candidate_constrained_curriculum
candidate_count: 28
selection gate: safe_progress_long
```

专项测试：

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_subgoal_filter.py \
  tests/test_exp019_training.py \
  tests/test_exp020_training.py \
  tests/test_exp021_training.py \
  tests/test_exp022_training.py

.venv_isaaclab/bin/python -m pytest -q -ra
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp022_randomized_terrain_endpoint_safety_filter.yaml \
  --device cpu \
  --num-envs 8 \
  --rollout-steps 4 \
  --timesteps 8 \
  --checkpoint-interval 4 \
  --run-name smoke_cpu_exp022 \
  --output-layout run \
  --selection-gate safe_progress_long
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp022_randomized_terrain_endpoint_safety_filter.yaml \
  --device cuda \
  --num-envs 256 \
  --rollout-steps 32 \
  --timesteps 64 \
  --checkpoint-interval 32 \
  --eval-num-envs 256 \
  --eval-steps 220 \
  --run-name smoke_cuda_exp022 \
  --output-layout run \
  --selection-gate safe_progress_long
```

正式 20M 长训练：

```bash
ROOT=$(pwd)
mkdir -p outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_launcher

systemd-run --user --unit exp022-endpoint-safety-20m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${ROOT}/outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_launcher/train.log \
  --property=StandardError=append:${ROOT}/outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp022_randomized_terrain_endpoint_safety_filter.yaml \
    --device cuda \
    --timesteps 10240 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_20m_endpoint_safety \
    --rollout-steps 32 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 220 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate safe_progress_long
```

监控：

```bash
tail -f outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_launcher/train.log
```

判读：

- strict gate 不变：`dmax <= 0.20`、`success >= 0.90`、`collision <= 0.02`、`timeout = 0`。
- 若未 strict，重点比较 exp021：collision 是否低于 `0.1746`，success 是否仍显著高于 exp020 的 `0.0`。
- 重点 telemetry：`filter_safety_override_fraction`、`filter_feasible_fraction`、`filter_raw_endpoint_near_violation_mean`、`filter_raw_path_collision_violation_mean`、`filter_path_collision_violation_mean`。

本轮 exp022 seed23 20M 已完成，但 strict 未通过。关键结论：

- collision 明显改善并通过 strict：5 seed mean `0.0170`，低于阈值 `0.02`，也远低于 exp021 的 `0.1746`。
- 集合进度严重退化：5 seed mean success `0.0139`，dmax ratio `0.4719`，timeout `0.9699`。
- filter 很强：raw path risk `0.3379` 降到 filtered path risk `0.2737`，risk reduction `0.0642`，applied fraction `0.6165`。
- 结论是 constrained post-processing 能压碰撞，但会重新变成“安全但不集合”；下一轮应做安全/集合联合 action representation、planner projection 或 success geometry，而不是继续强化 filter 权重。

复验/GIF 和曲线输出：

```text
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/pure_rl_seed23_20m_endpoint_safety/metrics/multi_eval_20260626_153606/
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/pure_rl_seed23_20m_endpoint_safety/videos/multi_eval_20260626_153606/
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/pure_rl_seed23_20m_endpoint_safety/figures/training_curves.png
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/pure_rl_seed23_20m_endpoint_safety/figures/candidate_eval_curves.png
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_suite/metrics/
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_suite/figures/
```

不要提交 `outputs/` 下的 checkpoint、JSON、PNG、GIF 或 TensorBoard。

## exp023 soft progress-preserving subgoal filter

目的：针对 exp022 的低成功率/高 timeout，把 filter 从 hard safety shield 改回软进度保护。exp023 保留 exp021 的集合底座，只在 raw action 预测真实 collision 时允许 override，并在 score 中加入 visible-neighbor center / center-progress 软惩罚。

关键配置：

```text
config: configs/experiment/exp023_randomized_terrain_soft_progress_filter.yaml
experiment: exp023_randomized_terrain_soft_progress_filter
run: pure_rl_seed23_20m_soft_progress_filter
filter mode: terrain_safe_candidate_soft_progress_curriculum
candidate_count: 28
selection gate: progress_preserving_long
```

专项测试：

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_subgoal_filter.py \
  tests/test_exp021_training.py \
  tests/test_exp022_training.py \
  tests/test_exp023_training.py

.venv_isaaclab/bin/python -m pytest -q -ra
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp023_randomized_terrain_soft_progress_filter.yaml \
  --device cpu \
  --num-envs 8 \
  --rollout-steps 4 \
  --timesteps 8 \
  --checkpoint-interval 4 \
  --eval-num-envs 8 \
  --eval-steps 16 \
  --run-name smoke_cpu_exp023 \
  --output-layout run \
  --selection-gate progress_preserving_long
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp023_randomized_terrain_soft_progress_filter.yaml \
  --device cuda \
  --num-envs 256 \
  --rollout-steps 32 \
  --timesteps 64 \
  --checkpoint-interval 32 \
  --eval-num-envs 256 \
  --eval-steps 220 \
  --run-name smoke_cuda_exp023 \
  --output-layout run \
  --selection-gate progress_preserving_long
```

正式 20M 长训练：

```bash
ROOT=$(pwd)
mkdir -p outputs/runs/exp023_randomized_terrain_soft_progress_filter/_launcher

systemd-run --user --unit exp023-soft-progress-filter-20m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${ROOT}/outputs/runs/exp023_randomized_terrain_soft_progress_filter/_launcher/train.log \
  --property=StandardError=append:${ROOT}/outputs/runs/exp023_randomized_terrain_soft_progress_filter/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp023_randomized_terrain_soft_progress_filter.yaml \
    --device cuda \
    --timesteps 10240 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_20m_soft_progress_filter \
    --rollout-steps 32 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 220 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate progress_preserving_long
```

监控：

```bash
tail -f outputs/runs/exp023_randomized_terrain_soft_progress_filter/_launcher/train.log
```

判读：

- strict gate 不变：`dmax <= 0.20`、`success >= 0.90`、`collision <= 0.02`、`timeout = 0`。
- 若未 strict，重点比较 exp022：success 是否显著高于 `0.0139`、timeout 是否低于 `0.9699`。
- 同时比较 exp021：collision 是否低于 `0.1746`。
- 重点 telemetry：`filter_applied_fraction`、`filter_collision_override_fraction`、`filter_raw_visible_center_cost_mean`、`filter_filtered_visible_center_cost_mean`、`filter_center_progress_regression_mean`。

不要提交 `outputs/` 下的 checkpoint、JSON、PNG、GIF 或 TensorBoard。

## exp024 mutual path safety subgoal filter

目的：针对 exp023 的 late-stage collision，把可见邻居 raw subgoal path 纳入候选打分，避免 static endpoint/path filter 漏掉多车同步相向运动。

关键配置：

```text
config: configs/experiment/exp024_randomized_terrain_mutual_path_filter.yaml
experiment: exp024_randomized_terrain_mutual_path_filter
run: pure_rl_seed23_20m_mutual_path_filter
filter mode: terrain_safe_candidate_mutual_progress_curriculum
candidate_count: 28
selection gate: progress_preserving_long
```

专项测试：

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_subgoal_filter.py \
  tests/test_exp023_training.py \
  tests/test_exp024_training.py

.venv_isaaclab/bin/python -m pytest -q -ra
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp024_randomized_terrain_mutual_path_filter.yaml \
  --device cpu \
  --num-envs 8 \
  --rollout-steps 4 \
  --timesteps 8 \
  --checkpoint-interval 4 \
  --eval-num-envs 8 \
  --eval-steps 16 \
  --run-name smoke_cpu_exp024 \
  --output-layout run \
  --selection-gate progress_preserving_long
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp024_randomized_terrain_mutual_path_filter.yaml \
  --device cuda \
  --num-envs 256 \
  --rollout-steps 32 \
  --timesteps 64 \
  --checkpoint-interval 32 \
  --eval-num-envs 256 \
  --eval-steps 220 \
  --run-name smoke_cuda_exp024 \
  --output-layout run \
  --selection-gate progress_preserving_long
```

正式 20M 长训练：

```bash
ROOT=$(pwd)
mkdir -p outputs/runs/exp024_randomized_terrain_mutual_path_filter/_launcher

systemd-run --user --unit exp024-mutual-path-filter-20m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${ROOT}/outputs/runs/exp024_randomized_terrain_mutual_path_filter/_launcher/train.log \
  --property=StandardError=append:${ROOT}/outputs/runs/exp024_randomized_terrain_mutual_path_filter/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp024_randomized_terrain_mutual_path_filter.yaml \
    --device cuda \
    --timesteps 10240 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_20m_mutual_path_filter \
    --rollout-steps 32 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 220 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate progress_preserving_long
```

监控：

```bash
tail -f outputs/runs/exp024_randomized_terrain_mutual_path_filter/_launcher/train.log
```

判读：

- strict gate 不变：`dmax <= 0.20`、`success >= 0.90`、`collision <= 0.02`、`timeout = 0`。
- 若未 strict，重点比较 exp023：collision 是否低于 `0.2295`，success 是否仍高于 exp022 的 `0.0139`。
- 重点 telemetry：`filter_raw_mutual_path_collision_violation_mean`、`filter_mutual_path_collision_violation_mean`、`filter_applied_fraction`、`filter_collision_override_fraction`。

不要提交 `outputs/` 下的 checkpoint、JSON、PNG、GIF 或 TensorBoard。

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
