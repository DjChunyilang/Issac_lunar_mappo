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

## 兼容 checkpoint warm-start

当新配置只改执行期控制或平整度相关门控、且 Actor/Critic 架构与观测 schema 不变时，可用已有 checkpoint 初始化模型参数：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/<warmstart_config>.yaml \
  --init-checkpoint outputs/runs/<source_experiment>/<source_run>/checkpoints/best.pt \
  --device cuda \
  --run-name warmstart_<source>_seed23_screen \
  --output-layout run \
  --timesteps 1024
```

这是**重新训练**而不是 `--resume-checkpoint`：不会恢复 optimizer、rollout memory 或 source timestep。若不显式传 `--bc-updates`，训练器会禁用 BC，并将原始策略作为 `ppo_timestep_000000.pt` 候选一并评估。只在 `metrics/final_eval_proxy.json` 和 `metrics/strict_acceptance.json` 通过后，才可把更新后的 checkpoint 作为候选；0-step 选中只能表明 PPO 没有改善 source policy。

## 执行时域消融

若要检验 timeout 是否由执行时间不足导致，派生 YAML 使用 `extends:` 继承基线，并把以下三处同步修改：`simulation.episode_length_s`、`experiment.eval_steps`、`evaluation.proxy_eval.steps`（高保真评估若启用，也同步 `evaluation.high_fidelity_eval.steps`）。当前本配置族采用 `96 s` 对应 control steps=`480`。这改变的是策略完成任务的时域，不改变 strict gate；仍必须满足 `timeout_rate == 0`。时域对照应固定 checkpoint、eval seed 和环境数，并把后验评测写为单独 JSON，不能覆盖正式 run 的 `final_eval_proxy.json`。

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
selection gate: success_progress_long
```

exp024 训练后新增 `success_progress_long`，用于同类 mutual path 实验的后续默认选择：strict 优先；未 strict 时优先保留 success/dmax 已有明显进展的 checkpoint，再比较 collision / timeout，避免再次选中早期低成功 checkpoint。

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
  --selection-gate success_progress_long
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
  --selection-gate success_progress_long
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
    --selection-gate success_progress_long
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
  --selection-gate success_progress_long
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
  --selection-gate success_progress_long
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
    --selection-gate success_progress_long
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

## exp025 dense mutual path safety filter

目的：基于 exp024 的最小步调参，继续压低 late-stage collision / timeout。exp025 不改 Actor/Critic 接口、不使用 BC、不回退到 exp022 的 hard constraint；它只把 mutual/path safety 采样从 5 点加密到 9 点，并适度提高 path/mutual collision 权重。

关键配置：

```text
config: configs/experiment/exp025_randomized_terrain_dense_mutual_filter.yaml
experiment: exp025_randomized_terrain_dense_mutual_filter
run: pure_rl_seed23_20m_dense_mutual_filter
filter mode: terrain_safe_candidate_mutual_progress_curriculum
candidate_count: 28
path_samples: 9
selection gate: success_progress_long
```

专项测试：

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_subgoal_filter.py \
  tests/test_exp024_training.py \
  tests/test_exp025_training.py

.venv_isaaclab/bin/python -m pytest -q -ra
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp025_randomized_terrain_dense_mutual_filter.yaml \
  --device cpu \
  --num-envs 8 \
  --rollout-steps 4 \
  --timesteps 8 \
  --checkpoint-interval 4 \
  --eval-num-envs 8 \
  --eval-steps 16 \
  --run-name smoke_cpu_exp025 \
  --output-layout run \
  --selection-gate success_progress_long
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp025_randomized_terrain_dense_mutual_filter.yaml \
  --device cuda \
  --num-envs 256 \
  --rollout-steps 32 \
  --timesteps 64 \
  --checkpoint-interval 32 \
  --eval-num-envs 256 \
  --eval-steps 220 \
  --run-name smoke_cuda_exp025 \
  --output-layout run \
  --selection-gate success_progress_long
```

正式 20M 长训练：

```bash
ROOT=$(pwd)
mkdir -p outputs/runs/exp025_randomized_terrain_dense_mutual_filter/_launcher

systemd-run --user --unit exp025-dense-mutual-filter-20m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${ROOT}/outputs/runs/exp025_randomized_terrain_dense_mutual_filter/_launcher/train.log \
  --property=StandardError=append:${ROOT}/outputs/runs/exp025_randomized_terrain_dense_mutual_filter/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp025_randomized_terrain_dense_mutual_filter.yaml \
    --device cuda \
    --timesteps 10240 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_20m_dense_mutual_filter \
    --rollout-steps 32 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 220 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate success_progress_long
```

监控：

```bash
tail -f outputs/runs/exp025_randomized_terrain_dense_mutual_filter/_launcher/train.log
```

判读：

- strict gate 不变：`dmax <= 0.20`、`success >= 0.90`、`collision <= 0.02`、`timeout = 0`。
- 若未 strict，重点比较 exp024：collision 是否低于 `0.0674`，success 是否保持接近或高于 `0.8398`，timeout 是否低于 `0.0947`。
- 重点 telemetry：`filter_raw_mutual_path_collision_violation_mean`、`filter_mutual_path_collision_violation_mean`、`filter_path_collision_violation_mean`、`filter_applied_fraction`、`filter_collision_override_fraction`。

当前结果：

- seed23 20M 已完成，best 为 `ppo_timestep_009216.pt`。
- final eval：dmax ratio `0.1434`、success `0.8525`、collision `0.0449`、timeout `0.1035`，strict 未通过。
- 相比 exp024，collision 从 `0.0674` 降低到 `0.0449`，但 timeout 从 `0.0947` 小幅升到 `0.1035`；下一轮应改末段 hold / success-zone 稳定，而不是继续单纯加大 filter 权重。

不要提交 `outputs/` 下的 checkpoint、JSON、PNG、GIF 或 TensorBoard。

## exp026 hold-stable subgoal filter

目的：针对 exp025 的末段 hold 不稳定，不再继续单纯增加 path/mutual collision 权重；在接近 success gate 时启用 hold-zone rho / spacing cost，偏好短步长和更大的 endpoint spacing buffer。

关键配置：

```text
config: configs/experiment/exp026_randomized_terrain_hold_stable_filter.yaml
experiment: exp026_randomized_terrain_hold_stable_filter
run: pure_rl_seed23_20m_hold_stable_filter
filter mode: terrain_safe_candidate_hold_progress_curriculum
candidate_count: 28
path_samples: 9
selection gate: success_progress_long
```

专项测试：

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_subgoal_filter.py \
  tests/test_exp026_training.py

.venv_isaaclab/bin/python -m pytest -q -ra
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp026_randomized_terrain_hold_stable_filter.yaml \
  --device cpu \
  --num-envs 8 \
  --rollout-steps 4 \
  --timesteps 8 \
  --checkpoint-interval 4 \
  --eval-num-envs 8 \
  --eval-steps 16 \
  --run-name smoke_cpu_exp026 \
  --output-layout run \
  --selection-gate success_progress_long
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp026_randomized_terrain_hold_stable_filter.yaml \
  --device cuda \
  --num-envs 256 \
  --rollout-steps 32 \
  --timesteps 64 \
  --checkpoint-interval 32 \
  --eval-num-envs 256 \
  --eval-steps 220 \
  --run-name smoke_cuda_exp026 \
  --output-layout run \
  --selection-gate success_progress_long
```

正式 20M 长训练：

```bash
ROOT=$(pwd)
mkdir -p outputs/runs/exp026_randomized_terrain_hold_stable_filter/_launcher

systemd-run --user --unit exp026-hold-stable-filter-20m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${ROOT}/outputs/runs/exp026_randomized_terrain_hold_stable_filter/_launcher/train.log \
  --property=StandardError=append:${ROOT}/outputs/runs/exp026_randomized_terrain_hold_stable_filter/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp026_randomized_terrain_hold_stable_filter.yaml \
    --device cuda \
    --timesteps 10240 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_20m_hold_stable_filter \
    --rollout-steps 32 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 220 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate success_progress_long
```

判读：

- strict gate 不变：`dmax <= 0.20`、`success >= 0.90`、`collision <= 0.02`、`timeout = 0`。
- 若未 strict，重点比较 exp025：success 是否高于 `0.8525`、collision 是否低于 `0.0449`、timeout 是否低于 `0.1035`。
- 重点 telemetry：`filter_hold_zone_activation_mean`、`filter_hold_zone_rho_cost_mean`、`filter_hold_zone_spacing_violation_mean`、`max_success_hold_count_mean`、`first_collision_step_mean`。

当前结果：

- seed23 20M 已完成，final eval：dmax ratio `0.1474`、success `0.7529`、collision `0.0615`、timeout `0.1865`。
- `filter_hold_zone_activation_mean=0.1731`，说明 hold-zone 确实介入，但介入过早/过宽，压制了集合进度。
- exp026 不能作为当前主结果；不要继续沿这个宽触发 hold-zone 配置加权。

## exp027–exp029 后续随机地形诊断

三轮均沿用 pure RL、shared-joint MAPPO、随机增强地形、`12 m` 通信半径和 `success_progress_long` checkpoint selection。

| exp | config | run | final success | collision | timeout | 结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| exp027 | `configs/experiment/exp027_randomized_terrain_strict_hold_filter.yaml` | `pure_rl_seed23_20m_strict_hold_filter` | 0.8418 | 0.0498 | 0.1123 | 严格 hold-zone trigger 避免 exp026 退化，但未优于 exp025。 |
| exp028 | `configs/experiment/exp028_randomized_terrain_hold_reward.yaml` | `pure_rl_seed23_20m_hold_reward` | 0.8691 | 0.0469 | 0.0889 | 强化 hold reward 有效，是 exp026–029 中最好结果，但仍未 strict。 |
| exp029 | `configs/experiment/exp029_randomized_terrain_hold_reward_safe.yaml` | `pure_rl_seed23_20m_hold_reward_safe` | 0.8262 | 0.0557 | 0.1221 | 继续加强 safety reward/filter 权重导致退化。 |

启动这些实验时使用同一模板替换 config、unit、experiment/run 名：

```bash
ROOT=$(pwd)
mkdir -p outputs/runs/<experiment>/_launcher

systemd-run --user --unit <unit-name> \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${ROOT}/outputs/runs/<experiment>/_launcher/train.log \
  --property=StandardError=append:${ROOT}/outputs/runs/<experiment>/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/<config>.yaml \
    --device cuda \
    --timesteps 10240 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name <run> \
    --rollout-steps 32 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 220 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate success_progress_long
```

历史下一步曾以 exp028 为最佳随机地形 candidate，随后 exp030–exp038 已完成控制层和 success-zone 稳定诊断。当前不要再回到单纯加大静态 path/mutual collision 权重；最新路线见下方 exp032–exp041 小节。

不要提交 `outputs/` 下的 checkpoint、JSON、PNG、GIF 或 TensorBoard。

## exp030 control safety projection

目的：回到 exp028 的 dense mutual filter + hold reward，不再继续加重静态 safety 权重；在低层控制链路中加入默认关闭的 one-step collision anticipation 和 success-zone velocity damping。

关键配置：

```text
config: configs/experiment/exp030_randomized_terrain_control_safety.yaml
experiment: exp030_randomized_terrain_control_safety
run: pure_rl_seed23_20m_control_safety
selection gate: success_progress_long
```

专项测试：

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_control_safety.py \
  tests/test_exp030_training.py \
  tests/test_exp029_training.py \
  tests/test_exp028_training.py \
  tests/test_subgoal_filter.py

.venv_isaaclab/bin/python -m pytest -q -ra
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp030_randomized_terrain_control_safety.yaml \
  --device cpu \
  --num-envs 8 \
  --rollout-steps 4 \
  --timesteps 8 \
  --checkpoint-interval 4 \
  --eval-num-envs 8 \
  --eval-steps 16 \
  --run-name smoke_cpu_exp030 \
  --output-layout run \
  --selection-gate success_progress_long
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp030_randomized_terrain_control_safety.yaml \
  --device cuda \
  --num-envs 256 \
  --rollout-steps 32 \
  --timesteps 64 \
  --checkpoint-interval 32 \
  --eval-num-envs 256 \
  --eval-steps 220 \
  --run-name smoke_cuda_exp030 \
  --output-layout run \
  --selection-gate success_progress_long
```

正式 20M 长训练：

```bash
ROOT=$(pwd)
mkdir -p outputs/runs/exp030_randomized_terrain_control_safety/_launcher

systemd-run --user --unit exp030-control-safety-20m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${ROOT}/outputs/runs/exp030_randomized_terrain_control_safety/_launcher/train.log \
  --property=StandardError=append:${ROOT}/outputs/runs/exp030_randomized_terrain_control_safety/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp030_randomized_terrain_control_safety.yaml \
    --device cuda \
    --timesteps 10240 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_20m_control_safety \
    --rollout-steps 32 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 220 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate success_progress_long
```

重点 telemetry：

- `control_safety_applied_fraction`
- `control_safety_linear_scale_mean`
- `control_safety_linear_scale_min`
- `control_safety_pairwise_risk_mean`
- `control_safety_success_zone_fraction`
- `first_collision_step_mean`
- `max_success_hold_count_mean`

当前结果：

- seed23 20M 已完成，best `ppo_timestep_010240.pt`，final eval：dmax ratio `0.1528`、success `0.8330`、collision `0.0313`、timeout `0.1357`。
- 相对 exp028，collision 从 `0.0469` 降低，但 success 和 timeout 退化。
- `control_safety_applied_fraction=0.1610`、`control_safety_linear_scale_mean=0.9304`、`control_safety_linear_scale_min=0.25`，说明投影起作用但偏强。
- 下一轮建议降低触发范围和强度：`projection_activation_distance 0.62 -> 0.52`、`projection_strength 0.80 -> 0.55`、`projection_min_linear_scale 0.25 -> 0.45`，并关闭或推迟 success-zone damping。

## exp031 narrow control safety projection

目的：调弱 exp030 的低层控制投影，保留降碰撞方向，同时减少 success/timeout 退化。

关键配置：

```text
config: configs/experiment/exp031_randomized_terrain_narrow_control_safety.yaml
experiment: exp031_randomized_terrain_narrow_control_safety
run: pure_rl_seed23_20m_narrow_control_safety
selection gate: success_progress_long
```

正式 20M 长训练：

```bash
ROOT=$(pwd)
mkdir -p outputs/runs/exp031_randomized_terrain_narrow_control_safety/_launcher

systemd-run --user --unit exp031-narrow-control-safety-20m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${ROOT}/outputs/runs/exp031_randomized_terrain_narrow_control_safety/_launcher/train.log \
  --property=StandardError=append:${ROOT}/outputs/runs/exp031_randomized_terrain_narrow_control_safety/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp031_randomized_terrain_narrow_control_safety.yaml \
    --device cuda \
    --timesteps 10240 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_20m_narrow_control_safety \
    --rollout-steps 32 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 220 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate success_progress_long
```

## exp032–exp041 control/success-zone 稳定诊断速查

这些实验都沿用随机增强 lunar crater proxy、shared-joint MAPPO pure RL、`12 m` 通信半径和 `86/54` 接口；除特别说明外，seed23 20M、2048 env、rollout 32、checkpoint interval 1024。

| exp | config | run / eval | success | collision | timeout | 结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| exp032 | `configs/experiment/exp032_randomized_terrain_closing_control_safety.yaml` | `pure_rl_seed23_20m_closing_control_safety` | 0.8379 | 0.0361 | 0.1279 | closing-only 略优于 exp031，但未达标。 |
| exp033 | `configs/experiment/exp033_randomized_terrain_directional_control_safety.yaml` | `pure_rl_seed23_20m_directional_control_safety` | 0.8154 | 0.0488 | 0.1387 | directional scale 没有安全收益。 |
| exp034 | `configs/experiment/exp034_randomized_terrain_directional_mask_control_safety.yaml` | `pure_rl_seed23_20m_directional_mask_control_safety` | 0.8828 | 0.0361 | 0.0840 | directional mask 恢复部分 success/timeout。 |
| exp035 | `configs/experiment/exp035_randomized_terrain_directional_mask_buffer.yaml` | `pure_rl_seed23_20m_directional_mask_buffer` | 0.9072 | 0.0127 | 0.0811 | success/collision 首次同时达标，timeout 成主瓶颈。 |
| exp036 | `configs/experiment/exp036_randomized_terrain_directional_mask_timeout_hold.yaml` | `pure_rl_seed23_20m_directional_mask_timeout_hold` | 0.9336 | 0.0088 | 0.0586 | stronger hold/timeout shaping 继续改善 timeout。 |
| exp037 | `configs/experiment/exp037_randomized_terrain_directional_mask_timeout260.yaml` | `pure_rl_seed23_20m_directional_mask_timeout260` | 0.9238 | 0.0352 | 0.0410 | 延长到 260 steps 降 timeout，但 collision 反弹。 |
| exp038 | `configs/experiment/exp038_randomized_terrain_success_zone_stabilizer.yaml` | `pure_rl_seed23_20m_success_zone_stabilizer_timeout320` | 0.9756 | 0.0137 | 0.0107 | 当前最佳；strict 只剩 timeout 未过。 |
| exp039 | `configs/experiment/exp039_randomized_terrain_hard_near_stabilizer.yaml` | exp038 best 诊断复评 | 0.9424 | 0.0254 | 0.0322 | hard near 退化，不建议长训。 |
| exp040 | `configs/experiment/exp040_randomized_terrain_soft_hold_stabilizer.yaml` | exp038 best 诊断复评 | 0.9658 | 0.0186 | 0.0166 | soft hold 仍差于 exp038，不建议长训。 |
| exp041 | `configs/experiment/exp041_randomized_terrain_hold_zone_override.yaml` | exp038 best 诊断复评 + smoke | 0.9795 | 0.0107 | 0.0098 | 略优于 exp038，但当前暂停长训。 |

当前暂停继续长训；下面命令只作为历史候选参考，不是当前推荐启动命令。恢复训练研究时应新建 exp043 或后续实验，并重新确认是否沿用 exp042 的环境改造：

```bash
ROOT=$(pwd)
mkdir -p outputs/runs/exp041_randomized_terrain_hold_zone_override/_launcher

systemd-run --user --unit exp041-hold-zone-override-20m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${ROOT}/outputs/runs/exp041_randomized_terrain_hold_zone_override/_launcher/train.log \
  --property=StandardError=append:${ROOT}/outputs/runs/exp041_randomized_terrain_hold_zone_override/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp041_randomized_terrain_hold_zone_override.yaml \
    --device cuda \
    --timesteps 10240 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_20m_hold_zone_override_timeout320 \
    --rollout-steps 32 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 320 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate strict
```

判读重点：

- 以 `metrics/final_eval_proxy.json` 和 `metrics/strict_acceptance.json` 为准。
- exp038 的剩余 timeout 主要卡在 `0.28–0.42 m` 最近邻灰区；若 exp041 仍未 strict，应继续做末端 pairwise spacing controller，而不是全局加硬 near/hold filter。
- exp039/exp040 只是诊断配置，除非重新设计，否则不要直接长训。

## exp042 结构化网络 / bicycle / quintic smoke

exp042 只做训练环境工程验证，不启动 20M 长训。配置：

```text
configs/experiment/exp042_structured_actor_bicycle_quintic_probe.yaml
```

当前 exp042 配置额外验证两个环境尺度调整：

- `safety.world_xy_limit=12.5`、`terrain.crater_field_size=25.0`，对应 `25 m × 25 m` 训练区域。
- `observation.communication_radius=0.0`，表示临时取消通信距离限制，所有非自身 rover 可见。

Actor 的局部地形网格仍是 `5×5×2=50` 维；本轮不扩大地形感知窗口，也不改变 Actor/Critic `86/54` 接口。

聚焦测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaaclab/bin/python -m pytest -q \
  tests/test_observation.py \
  tests/test_trajectory_generator.py \
  tests/test_proxy_rover_model.py \
  tests/test_config_wiring.py \
  tests/test_skrl_mappo_semantics.py
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp042_structured_actor_bicycle_quintic_probe.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_structured_bicycle_quintic_comm0_map25 \
  --rollout-steps 4 \
  --checkpoint-interval 4 \
  --eval-num-envs 8 \
  --eval-steps 16 \
  --bc-updates 0 \
  --selection-gate strict
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp042_structured_actor_bicycle_quintic_probe.yaml \
  --device cuda \
  --timesteps 64 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_structured_bicycle_quintic_comm0_map25 \
  --rollout-steps 32 \
  --checkpoint-interval 32 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate strict
```

验收重点：

- summary 中 `actor_architecture=branched_v1`、`critic_architecture=structured_v1`、`kinematic_model=bicycle`、`trajectory_geometry_method=quintic`。
- summary/config 中 `communication_radius=0.0`、`world_xy_limit=12.5`、`crater_field_size=25.0`。
- `optimizer_count=1`，`joint_update_count=2`。
- `terrain_input_weight_delta_l2 > 0`。
- `post_training_action_std > 0`，动作非退化。
- training telemetry 中有 `steering_angle_abs_mean`、`actual_yaw_rate_abs_mean` 和 `turning_radius_*`。

产物保留在 `outputs/runs/exp042_structured_actor_bicycle_quintic_probe/`，不要提交 checkpoint、JSONL、TensorBoard 或日志。

## exp043 structured / bicycle / quintic / map25 长训

exp043 是恢复长训后的第一轮新环境栈收敛实验，已完成但未通过 strict。配置：

```text
configs/experiment/exp043_structured_bicycle_quintic_map25_long.yaml
```

关键差异：

- `branched_v1 / structured_v1`，`bicycle`，`quintic`。
- `world_xy_limit=12.5`、`crater_field_size=25.0`、`communication_radius=0.0`。
- `initial_state.spawn_radius_min/max=4.5/6.5`、`center_xy_range=3.0`。
- `crater_count=48`，避免 25m 地图上地形密度被过度稀释。
- 继承 exp041 的 hold-zone override。

历史启动命令：

```bash
mkdir -p outputs/runs/exp043_structured_bicycle_quintic_map25_long/_launcher

systemd-run --user --unit exp043-structured-bicycle-quintic-map25-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp043_structured_bicycle_quintic_map25_long/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp043_structured_bicycle_quintic_map25_long/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp043_structured_bicycle_quintic_map25_long.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25 \
    --rollout-steps 64 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 320 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate success_progress_long
```

监控：

```bash
tail -f outputs/runs/exp043_structured_bicycle_quintic_map25_long/_launcher/train.log
systemctl --user status exp043-structured-bicycle-quintic-map25-40m
```

训练完成后优先读取：

```text
outputs/runs/exp043_structured_bicycle_quintic_map25_long/pure_rl_seed23_40m_structured_bicycle_quintic_map25/metrics/summary.json
outputs/runs/exp043_structured_bicycle_quintic_map25_long/pure_rl_seed23_40m_structured_bicycle_quintic_map25/metrics/final_eval_proxy.json
outputs/runs/exp043_structured_bicycle_quintic_map25_long/pure_rl_seed23_40m_structured_bicycle_quintic_map25/metrics/strict_acceptance.json
```

严格结论仍以 `dmax<=0.20`、`success>=0.90`、`collision<=0.02`、`timeout=0` 为准。

当前结果：

```text
best_candidate: ppo_timestep_020480.pt
dmax_reduction_ratio: 0.8596
success_rate: 0.0
collision_rate: 0.0
timeout_rate: 1.0
```

判读：不是工程链路故障，而是在新环境栈 + 较大 initial-state 分布下 pure RL 冷启动没有学出集合进度。当前不要继续重复启动 exp043；下一轮使用 exp044 的 initial-state curriculum。

## exp044 structured / bicycle / quintic / map25 initial-state curriculum

exp044 保留 exp043 的新网络、bicycle、quintic、25m 地图和 `communication_radius=0.0`，但训练时从较近队形开始并逐步 ramp 到目标 reset 分布。配置：

```text
configs/experiment/exp044_structured_bicycle_quintic_map25_curriculum.yaml
```

关键差异：

- 目标 reset 分布：`spawn_radius_min/max=3.8/5.2`、`center_xy_range=2.0`、`jitter_std=0.40`。
- curriculum 起点：`spawn_radius_min/max=3.0/4.0`、`center_xy_range=1.0`、`jitter_std=0.35`。
- `curriculum_warmup_timesteps=4096`，`curriculum_ramp_timesteps=8192`。
- 候选/最终 eval 不设置 progress override，因此仍在目标 reset 分布上判定。
- `crater_count=36`，避免 exp043 的地形密度和大初始分布同时过强。

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp044_structured_bicycle_quintic_map25_curriculum.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_curriculum \
  --rollout-steps 4 \
  --checkpoint-interval 4 \
  --eval-num-envs 8 \
  --eval-steps 16 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp044_structured_bicycle_quintic_map25_curriculum.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_curriculum \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

长训启动：

```bash
mkdir -p outputs/runs/exp044_structured_bicycle_quintic_map25_curriculum/_launcher

systemd-run --user --unit exp044-structured-bicycle-quintic-map25-curriculum-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp044_structured_bicycle_quintic_map25_curriculum/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp044_structured_bicycle_quintic_map25_curriculum/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp044_structured_bicycle_quintic_map25_curriculum.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_curriculum \
    --rollout-steps 64 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 320 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate success_progress_long
```

监控：

```bash
tail -f outputs/runs/exp044_structured_bicycle_quintic_map25_curriculum/_launcher/train.log
systemctl --user status exp044-structured-bicycle-quintic-map25-curriculum-40m
```

训练完成后优先读取：

```text
outputs/runs/exp044_structured_bicycle_quintic_map25_curriculum/pure_rl_seed23_40m_structured_bicycle_quintic_map25_curriculum/metrics/summary.json
outputs/runs/exp044_structured_bicycle_quintic_map25_curriculum/pure_rl_seed23_40m_structured_bicycle_quintic_map25_curriculum/metrics/final_eval_proxy.json
outputs/runs/exp044_structured_bicycle_quintic_map25_curriculum/pure_rl_seed23_40m_structured_bicycle_quintic_map25_curriculum/metrics/strict_acceptance.json
```

当前结果：exp044 未通过 strict，final eval dmax ratio `0.4796`、success `0.0`、collision `0.00195`、timeout `0.9980`。下一轮使用 exp045 的 local-success bootstrap，不要重复启动 exp044。

## exp045 structured / bicycle / quintic / map25 local success bootstrap

exp045 先缩小 reset 目标分布，验证新网络/bicycle/quintic/25m 地图下是否能恢复 success。配置：

```text
configs/experiment/exp045_structured_bicycle_quintic_map25_local_success_bootstrap.yaml
```

关键差异：

- 目标 reset：`spawn_radius_min/max=2.4/3.4`、`center_xy_range=1.0`。
- curriculum 起点：`spawn_radius_min/max=1.6/2.4`、`center_xy_range=0.5`。
- 动作/低层：`rho_max=1.6`、`beta_max=60°`、`max_steer_angle≈45°`、`reference_speed=0.9`。
- reward 更偏中距离集合：`dmax_progress=5.5`、`dispersion_progress=2.4`，terrain weight `0.20`。
- filter 仍保留，但 `apply_probability_end=0.35`、`score_scale_end=0.50`。

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp045_structured_bicycle_quintic_map25_local_success_bootstrap.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_local_success \
  --rollout-steps 4 \
  --checkpoint-interval 4 \
  --eval-num-envs 8 \
  --eval-steps 16 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp045_structured_bicycle_quintic_map25_local_success_bootstrap.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_local_success \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

长训启动：

```bash
mkdir -p outputs/runs/exp045_structured_bicycle_quintic_map25_local_success_bootstrap/_launcher

systemd-run --user --unit exp045-structured-bicycle-quintic-map25-local-success-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp045_structured_bicycle_quintic_map25_local_success_bootstrap/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp045_structured_bicycle_quintic_map25_local_success_bootstrap/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp045_structured_bicycle_quintic_map25_local_success_bootstrap.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_success \
    --rollout-steps 64 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 320 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate success_progress_long
```

监控：

```bash
tail -f outputs/runs/exp045_structured_bicycle_quintic_map25_local_success_bootstrap/_launcher/train.log
systemctl --user status exp045-structured-bicycle-quintic-map25-local-success-40m
```

训练完成后优先读取：

```text
outputs/runs/exp045_structured_bicycle_quintic_map25_local_success_bootstrap/pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_success/metrics/summary.json
outputs/runs/exp045_structured_bicycle_quintic_map25_local_success_bootstrap/pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_success/metrics/final_eval_proxy.json
outputs/runs/exp045_structured_bicycle_quintic_map25_local_success_bootstrap/pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_success/metrics/strict_acceptance.json
```

当前结果：exp045 未通过 strict，final eval dmax ratio `0.2734`、success `0.1846`、collision `0.0`、timeout `0.8174`。下一轮使用 exp046 释放末端 filter/control-safety 阻尼，并增强 dmax/dispersion/hold。

## exp046 structured / bicycle / quintic / map25 local hold release

exp046 沿用 exp045 的 local reset 分布，但降低 filter/control-safety 末端阻尼并增强末端收缩。配置：

```text
configs/experiment/exp046_structured_bicycle_quintic_map25_local_hold_release.yaml
```

关键差异：

- filter：`apply_probability_end=0.22`、`score_scale_end=0.35`。
- control safety：`projection_activation_distance=0.68`、`projection_strength=0.70`、`projection_min_linear_scale=0.40`。
- 低层：`reference_speed=1.0`、`max_linear_speed=1.2`、success-zone damping scale `0.65`。
- reward：`dmax_progress=7.0`、`dispersion_progress=3.2`、`success_bonus=85`、`timeout_penalty=45`、terrain weight `0.15`。

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp046_structured_bicycle_quintic_map25_local_hold_release.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_local_hold_release \
  --rollout-steps 4 \
  --checkpoint-interval 4 \
  --eval-num-envs 8 \
  --eval-steps 16 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp046_structured_bicycle_quintic_map25_local_hold_release.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_local_hold_release \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

长训启动：

```bash
mkdir -p outputs/runs/exp046_structured_bicycle_quintic_map25_local_hold_release/_launcher

systemd-run --user --unit exp046-structured-bicycle-quintic-map25-local-hold-release-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp046_structured_bicycle_quintic_map25_local_hold_release/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp046_structured_bicycle_quintic_map25_local_hold_release/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp046_structured_bicycle_quintic_map25_local_hold_release.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_hold_release \
    --rollout-steps 64 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 320 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate success_progress_long
```

监控：

```bash
tail -f outputs/runs/exp046_structured_bicycle_quintic_map25_local_hold_release/_launcher/train.log
systemctl --user status exp046-structured-bicycle-quintic-map25-local-hold-release-40m
```

训练完成后优先读取：

```text
outputs/runs/exp046_structured_bicycle_quintic_map25_local_hold_release/pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_hold_release/metrics/summary.json
outputs/runs/exp046_structured_bicycle_quintic_map25_local_hold_release/pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_hold_release/metrics/final_eval_proxy.json
 outputs/runs/exp046_structured_bicycle_quintic_map25_local_hold_release/pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_hold_release/metrics/strict_acceptance.json
```

当前结果：exp046 未通过 strict，但 local success 从 exp045 的 `0.1846` 提升到 `0.6123`，collision 保持 `0.0`。失败项为 dmax ratio `0.2424`、success `0.6123` 和 timeout `0.3877`。下一轮使用 exp047 继续释放 terminal safety/filter/control damping，并增强 dmax/dispersion/timeout shaping。

## exp047 structured / bicycle / quintic / map25 terminal convergence

exp047 沿用 exp046 的 local reset 分布，但更聚焦 terminal convergence。配置：

```text
configs/experiment/exp047_structured_bicycle_quintic_map25_terminal_convergence.yaml
```

关键差异：

- filter：`apply_probability_end=0.16`、`score_scale_end=0.28`、`hold_zone_pairwise_distance=0.48`。
- control safety：`projection_activation_distance=0.62`、`projection_strength=0.50`、`projection_min_linear_scale=0.55`。
- 低层：`reference_speed=1.05`、`max_linear_speed=1.25`、success-zone damping scale `0.80`。
- reward：`dmax_progress=9.0`、`dispersion_progress=4.5`、`success_bonus=115`、`timeout_penalty=65`、terrain weight `0.12`。

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp047_structured_bicycle_quintic_map25_terminal_convergence.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_terminal_convergence \
  --rollout-steps 4 \
  --checkpoint-interval 4 \
  --eval-num-envs 8 \
  --eval-steps 16 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp047_structured_bicycle_quintic_map25_terminal_convergence.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_terminal_convergence \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

长训启动：

```bash
mkdir -p outputs/runs/exp047_structured_bicycle_quintic_map25_terminal_convergence/_launcher

systemd-run --user --unit exp047-structured-bicycle-quintic-map25-terminal-convergence-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp047_structured_bicycle_quintic_map25_terminal_convergence/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp047_structured_bicycle_quintic_map25_terminal_convergence/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp047_structured_bicycle_quintic_map25_terminal_convergence.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_convergence \
    --rollout-steps 64 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 320 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate success_progress_long
```

监控：

```bash
tail -f outputs/runs/exp047_structured_bicycle_quintic_map25_terminal_convergence/_launcher/train.log
systemctl --user status exp047-structured-bicycle-quintic-map25-terminal-convergence-40m
```

训练完成后优先读取：

```text
outputs/runs/exp047_structured_bicycle_quintic_map25_terminal_convergence/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_convergence/metrics/summary.json
outputs/runs/exp047_structured_bicycle_quintic_map25_terminal_convergence/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_convergence/metrics/final_eval_proxy.json
outputs/runs/exp047_structured_bicycle_quintic_map25_terminal_convergence/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_convergence/metrics/strict_acceptance.json
```

当前结果：exp047 未通过 strict，但 final eval 提升到 success `0.7188`、collision `0.0059`、dmax ratio `0.2132`、timeout `0.2764`。下一轮使用 exp048，在 exp047 附近提高 terminal drive 和 dispersion 收缩。

## exp048 structured / bicycle / quintic / map25 terminal drive

exp048 沿用 exp047 local reset 分布，但提高末端推进速度和 dispersion 收缩。配置：

```text
configs/experiment/exp048_structured_bicycle_quintic_map25_terminal_drive.yaml
```

关键差异：

- filter：`apply_probability_end=0.18`、`score_scale_end=0.30`、`hold_zone_pairwise_distance=0.46`。
- control safety：`projection_strength=0.45`、`projection_min_linear_scale=0.65`。
- 低层：`reference_speed=1.15`、`max_linear_speed=1.35`、success-zone damping scale `0.95`。
- reward：`dmax_progress=9.5`、`dispersion_progress=6.0`、`success_bonus=130`、`timeout_penalty=80`、terrain weight `0.10`。

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp048_structured_bicycle_quintic_map25_terminal_drive.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_terminal_drive \
  --rollout-steps 4 \
  --checkpoint-interval 4 \
  --eval-num-envs 8 \
  --eval-steps 16 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp048_structured_bicycle_quintic_map25_terminal_drive.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_terminal_drive \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

长训启动：

```bash
mkdir -p outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/_launcher

systemd-run --user --unit exp048-structured-bicycle-quintic-map25-terminal-drive-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp048_structured_bicycle_quintic_map25_terminal_drive.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_drive \
    --rollout-steps 64 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 320 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate success_progress_long
```

监控：

```bash
tail -f outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/_launcher/train.log
systemctl --user status exp048-structured-bicycle-quintic-map25-terminal-drive-40m
```

当前结果：exp048 未通过 strict，但已经通过 dmax/success/collision gates：dmax ratio `0.1866`、success `0.9844`、collision `0.0020`，唯一失败为 timeout `0.0137`。剩余 timeout episode 已满足 dmax/dispersion，但最近邻距离均值约 `0.393 m`，低于 success 安全间距 `0.42 m`。

## exp049 structured / bicycle / quintic / map25 terminal spacing

exp049 沿用 exp048 local reset 分布，针对最后的最近邻安全间距灰区 timeout 做轻量 terminal spacing 修正。配置：

```text
configs/experiment/exp049_structured_bicycle_quintic_map25_terminal_spacing.yaml
```

关键差异：

- filter spacing：`hold_zone_pairwise_distance=0.52`、`hold_zone_spacing_weight=4.60`、`endpoint_safe_distance=0.44`。
- filter schedule：`apply_probability_end=0.20`、`score_scale_end=0.32`。
- control safety：`projection_activation_distance=0.64`、`projection_strength=0.55`、`projection_min_linear_scale=0.58`。
- reward：`near_distance=3.4`、`dispersion_progress=6.2`、`success_bonus=135`、`timeout_penalty=90`。

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp049_structured_bicycle_quintic_map25_terminal_spacing.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_terminal_spacing \
  --rollout-steps 4 \
  --checkpoint-interval 4 \
  --eval-num-envs 8 \
  --eval-steps 16 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp049_structured_bicycle_quintic_map25_terminal_spacing.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_terminal_spacing \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

长训启动：

```bash
mkdir -p outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/_launcher

systemd-run --user --unit exp049-structured-bicycle-quintic-map25-terminal-spacing-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp049_structured_bicycle_quintic_map25_terminal_spacing.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_spacing \
    --rollout-steps 64 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 320 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate success_progress_long
```

监控：

```bash
tail -f outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/_launcher/train.log
systemctl --user status exp049-structured-bicycle-quintic-map25-terminal-spacing-40m
```

当前结果：exp049 未通过 strict，且不优于 exp048。final eval 为 dmax ratio `0.1884`、success `0.8926`、collision `0.0010`、timeout `0.1064`。相比 exp048，碰撞略低但 success 跌破 `0.90`，timeout 从 `0.0137` 升到 `0.1064`；后续不要按原样继续增强全局 terminal spacing。

可视化复现：

```bash
.venv_isaaclab/bin/python scripts/render_skrl_proxy_rollout.py \
  --config outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_spacing/config/experiment.yaml \
  --checkpoint outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_spacing/checkpoints/best.pt \
  --device cuda \
  --steps 320 \
  --seed 11023 \
  --run-dir outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_spacing \
  --capture-interval 4 \
  --max-frames 90
```

曲线输出：

```bash
.venv_isaaclab/bin/python scripts/plot_skrl_run_curves.py \
  --run-dir outputs/runs/exp038_randomized_terrain_success_zone_stabilizer/pure_rl_seed23_20m_success_zone_stabilizer_timeout320 \
  --label exp038 \
  --run-dir outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_drive \
  --label exp048 \
  --run-dir outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_spacing \
  --label exp049 \
  --comparison-output outputs/runs/_comparisons/exp038_exp048_exp049_20260707/figures/candidate_eval_comparison.png
```

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

## exp158 DAE两级验证

先运行正式离线门限：

```bash
.venv_isaaclab/bin/python scripts/run_exp158_dae_validation.py \
  --phase offline \
  --device cuda:0
```

launcher会先复核专项测试、CPU/CUDA smoke、初始化hash、8 GB显存和60%吞吐门限，再执行冻结审计。只有以下文件的 `passed=true` 才允许训练：

```text
outputs/runs/exp158_dae_validation/offline_credit_audit/metrics/offline_gate.json
```

H1 seed23完整配对：

```bash
.venv_isaaclab/bin/python scripts/run_exp158_dae_validation.py \
  --phase h1 \
  --seed 23 \
  --device cuda:0
```

后续seed必须分别显式启动，launcher不会自动重试失败run：

```bash
.venv_isaaclab/bin/python scripts/run_exp158_dae_validation.py --phase h1 --seed 31 --device cuda:0
.venv_isaaclab/bin/python scripts/run_exp158_dae_validation.py --phase h1 --seed 47 --device cuda:0
```

strict阶段要求H1三seed汇总通过；命令相同，只将 `--phase` 改为 `strict`。任何已有run目录都会触发拒绝覆盖，必须先检查该run状态，不得删除产物后静默重试。

## exp159解析式ALO-PRD

先运行双离线门控；A-H1失败时launcher不会运行A-strict：

```bash
.venv_isaaclab/bin/python scripts/run_exp159_prd_validation.py \
  --phase offline \
  --device cuda:0
```

正式门限事实源：

```text
outputs/runs/exp159_analytical_prd/offline_h1_audit/metrics/offline_gate.json
outputs/runs/exp159_analytical_prd/offline_strict_audit/metrics/offline_gate.json
```

A-H1通过后才允许：

```bash
.venv_isaaclab/bin/python scripts/run_exp159_prd_validation.py \
  --phase h1 --seed 23 --device cuda:0
```

seed31、47及strict阶段必须分别显式启动。ALO-PRD不允许与DAE、历史Actor credit或collision constraint同时启用，现有run目录也不会被覆盖或自动重试。
