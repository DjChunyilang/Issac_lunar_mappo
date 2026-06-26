# exp023 soft progress-preserving subgoal filter 随机地形实验

## 目的

exp021 会集合但碰撞高；exp022 把 collision 压进 strict gate，却把 success 压到 `0.0139`、timeout 推到 `0.9699`。exp023 的目标不是继续加大安全权重，而是验证一个更温和的假设：filter 只能作为进度保持的软投影，不能在训练后期接管大部分动作。

本轮仍是 proxy 训练，不引入 PhysX、真实车体尺寸、轮地接触、硬性陷车终止或 BC。

## 配置

```text
config: configs/experiment/exp023_randomized_terrain_soft_progress_filter.yaml
experiment_id: exp023_randomized_terrain_soft_progress_filter
run_id: pure_rl_seed23_20m_soft_progress_filter
```

关键设置：

- shared-joint MAPPO pure RL，`bc_updates=0`。
- 随机增强 lunar crater proxy，`randomize_per_reset=true`。
- 通信半径 `12 m`，Actor / Critic 接口仍为 `86 / 54`。
- 2048 CUDA envs，rollout 32，连续 `10240` timesteps，即约 20.97M env steps。
- checkpoint interval `1024` timesteps。
- 候选选择使用 `progress_preserving_long`，避免优先选择 success 接近 0 的安全 standoff checkpoint。

soft progress filter：

```text
mode: terrain_safe_candidate_soft_progress_curriculum
rho_scales: [0.75, 0.90, 1.0, 1.08]
beta_offsets_deg: [-35, -20, -10, 0, 10, 20, 35]
candidate_count: 28
warmup_timesteps: 2048
ramp_timesteps: 4096
apply_probability: 0.0 -> 0.35
score_scale: 0.20 -> 0.55
endpoint_safe_distance: 0.42
path_safe_distance: 0.32
hard_endpoint_near_filter: false
hard_path_collision_filter: false
hard_center_progress_filter: false
safety_override_after_warmup: false
collision_override_after_warmup: true
```

score 中的主要权重：

```text
intent_deviation_weight: 0.40
path_terrain_mean_weight: 0.40
path_terrain_max_weight: 0.25
path_height_change_weight: 0.10
subgoal_terrain_weight: 0.15
endpoint_near_weight: 1.20
endpoint_collision_weight: 500.0
path_near_weight: 0.80
path_collision_weight: 350.0
visible_neighbor_center_weight: 1.00
center_progress_weight: 1.50
```

reward 相对 exp022 回退强安全设置：

```text
safety.near_distance: 0.85
near_distance: 6.0
inter_agent_collision: 90.0
failure_penalty: 50.0
filter_raw_path_risk_cost: 0.20
filter_deviation_cost: 0.05
```

## 验收标准

proxy strict gate 不变：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

如果 strict 未通过，本轮诊断重点看：

- success 是否明显高于 exp022 的 `0.0139`，避免再次安全 standoff。
- collision 是否低于 exp021 的 `0.1746`，证明 collision-only override 和软安全项有价值。
- filter applied fraction 是否显著低于 exp022 的 `0.6165`。
- `filter_collision_override_fraction` 是否远低于总 applied fraction，说明 filter 主要是软建议而非强接管。
- `filter_filtered_visible_center_cost_mean` 是否不高于 raw 太多，避免离集合中心越来越远。

## 工程验证

已完成：

- soft progress warmup 期不替换 raw action，但输出 center-progress telemetry。
- center-progress 软项会偏好不远离可见邻居中心的候选。
- collision override 只对 raw endpoint/path collision 生效，不对 near-distance violation 生效。
- exp023 配置 contract、候选数、reward 回退、selection gate 和 telemetry shape。
- 完整 `.venv_isaaclab/bin/python -m pytest -q -ra` 通过。
- CPU smoke 和 CUDA smoke 通过。
- CUDA smoke 确认 `optimizer_count=1`、`joint_update_count=2`、terrain 输入权重更新、动作非退化。

专项测试命令：

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_subgoal_filter.py \
  tests/test_exp021_training.py \
  tests/test_exp022_training.py \
  tests/test_exp023_training.py
```

完整测试：

```bash
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

## 长训练命令

seed23 20M 已作为用户级 systemd transient service 启动：

```text
unit: exp023-soft-progress-filter-20m.service
run_id: pure_rl_seed23_20m_soft_progress_filter
```

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

## 结果表

seed23 20M 训练、候选评估和 final eval 已完成。结果以以下机器可读文件为准：

```text
outputs/runs/exp023_randomized_terrain_soft_progress_filter/pure_rl_seed23_20m_soft_progress_filter/metrics/strict_acceptance.json
outputs/runs/exp023_randomized_terrain_soft_progress_filter/pure_rl_seed23_20m_soft_progress_filter/metrics/summary.json
outputs/runs/exp023_randomized_terrain_soft_progress_filter/pure_rl_seed23_20m_soft_progress_filter/metrics/final_eval_proxy.json
outputs/runs/exp023_randomized_terrain_soft_progress_filter/pure_rl_seed23_20m_soft_progress_filter/metrics/eval_metrics.json
```

| eval | checkpoint | dmax ratio | success | collision | timeout | filter applied | strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| best candidate eval seed1023 | `ppo_timestep_010240.pt` | 0.1772 | 0.3164 | 0.2246 | 0.4658 | 0.0798 | 未通过 |
| final eval seed1023 | `best.pt` | 0.1789 | 0.3027 | 0.2295 | 0.4717 | 0.0805 | 未通过 |

关键 filter telemetry：

```text
filter_raw_path_terrain_risk_mean: 0.4024
filter_filtered_path_terrain_risk_mean: 0.4006
filter_collision_override_fraction: 0.0335
filter_raw_endpoint_collision_violation_mean: 0.00462
filter_endpoint_collision_violation_mean: 0.000005
filter_raw_path_collision_violation_mean: 0.01086
filter_path_collision_violation_mean: 0.000356
filter_raw_visible_center_cost_mean: 1.3194
filter_filtered_visible_center_cost_mean: 1.3111
first_collision_step_mean: 183.2 / 220
```

## 失败分析

strict gate 未通过：

- `dmax_reduction_ratio=0.1789`，通过 `<=0.20`。
- `success_rate=0.3027`，未达到 `>=0.90`。
- `collision_rate=0.2295`，远高于 `<=0.02`。
- `timeout_rate=0.4717`，远高于 `0`。

和 exp022 对比：

- success 从 `0.0139` 回升到 `0.3027`，说明 soft progress filter 缓解了 exp022 的安全 standoff。
- filter applied fraction 从 exp022 的 `0.6165` 降到 `0.0805`，不再大规模接管 actor。
- collision 从 exp022 的 `0.0170` 升到 `0.2295`，安全失败严重。

当前判断：

exp023 没有复现 exp022 的强接管问题，但它暴露了 filter 的另一个盲点：endpoint/path safety 只把候选路径与“邻居当前位置”比较，未预测可见邻居也同时移动。final eval 中 endpoint/path collision violation 已接近 0，但真实 rollout 的 `first_collision_step_mean≈183`，说明很多碰撞发生在接近集合/hold 的动态相向运动阶段。

下一轮 exp024 应加入 mutual path safety：用可见邻居的 raw subgoal path 作为动态障碍，对候选路径进行同时间采样比较，仍不使用 oracle，也不恢复 exp022 的 hard constraint。

## 产物路径

```text
outputs/runs/exp023_randomized_terrain_soft_progress_filter/pure_rl_seed23_20m_soft_progress_filter/
outputs/runs/exp023_randomized_terrain_soft_progress_filter/_suite/metrics/
outputs/runs/exp023_randomized_terrain_soft_progress_filter/_suite/figures/
outputs/runs/exp023_randomized_terrain_soft_progress_filter/_launcher/train.log
```

生成的 checkpoint、JSON、TensorBoard、PNG、GIF 均保留在 `outputs/`，不提交到 git。
