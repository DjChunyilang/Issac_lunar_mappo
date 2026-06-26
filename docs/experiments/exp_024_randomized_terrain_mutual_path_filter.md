# exp024 mutual path safety subgoal filter 随机地形实验

## 目的

exp023 恢复了一部分集合进度，但 collision 仍高达 `0.2295`。关键矛盾是：filter telemetry 中 endpoint/path collision violation 已被压到接近 0，但真实 rollout 仍在后期碰撞，`first_collision_step_mean≈183/220`。

当前判断是 exp023 的 path safety 只把本车候选路径与“邻居当前位置”比较，没有预测可见邻居也会同时移动。exp024 的目标是补上这个盲点：把可见邻居的 raw subgoal path 当作动态障碍，按相同时间采样比较本车候选路径与邻居 raw path，降低接近集合/hold 阶段的相向运动碰撞。

本轮仍是 proxy 训练，不引入 PhysX、真实车体尺寸、轮地接触、硬性陷车终止或 BC。

## 配置

```text
config: configs/experiment/exp024_randomized_terrain_mutual_path_filter.yaml
experiment_id: exp024_randomized_terrain_mutual_path_filter
run_id: pure_rl_seed23_20m_mutual_path_filter
```

关键设置：

- shared-joint MAPPO pure RL，`bc_updates=0`。
- 随机增强 lunar crater proxy，`randomize_per_reset=true`。
- 通信半径 `12 m`，Actor / Critic 接口仍为 `86 / 54`。
- 2048 CUDA envs，rollout 32，连续 `10240` timesteps，即约 20.97M env steps。
- checkpoint interval `1024` timesteps。
- 长训时原始候选选择使用 `progress_preserving_long`；训练后发现该 gate 过度偏好早期低 collision checkpoint，因此新增并 post-hoc 使用 `success_progress_long` 重选 best。

mutual path filter：

```text
mode: terrain_safe_candidate_mutual_progress_curriculum
rho_scales: [0.65, 0.85, 1.0, 1.08]
beta_offsets_deg: [-40, -25, -12.5, 0, 12.5, 25, 40]
candidate_count: 28
warmup_timesteps: 2048
ramp_timesteps: 4096
apply_probability: 0.0 -> 0.45
score_scale: 0.20 -> 0.60
endpoint_safe_distance: 0.42
path_safe_distance: 0.32
hard_endpoint_near_filter: false
hard_path_collision_filter: false
hard_center_progress_filter: false
safety_override_after_warmup: false
collision_override_after_warmup: true
```

score 相对 exp023 新增：

```text
mutual_path_near_weight: 1.50
mutual_path_collision_weight: 900.0
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

- collision 是否低于 exp023 的 `0.2295`，尤其是否低于 exp021 的 `0.1746`。
- success 是否保持明显高于 exp022 的 `0.0139`，避免退回安全 standoff。
- `filter_raw_mutual_path_collision_violation_mean` 是否被 `filter_mutual_path_collision_violation_mean` 明显降低。
- filter applied fraction 是否保持低于 exp022 的 `0.6165`。

## 工程验证

已完成：

- mutual path filter 可检测“两车当前位置安全但 raw path 迎面交叉”的动态路径冲突。
- 不可见 rover 的 raw path 不影响单 rover mutual filter 输出。
- exp024 配置 contract、候选数、mutual path 权重和 telemetry shape。
- 完整 `.venv_isaaclab/bin/python -m pytest -q -ra` 通过。
- CPU smoke 和 CUDA smoke 通过。
- CUDA smoke 确认 `optimizer_count=1`、`joint_update_count=2`、terrain 输入权重更新、动作非退化。

专项测试：

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_subgoal_filter.py \
  tests/test_exp023_training.py \
  tests/test_exp024_training.py
```

完整测试：

```bash
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

## 长训练命令

seed23 20M 已作为用户级 systemd transient service 启动：

```text
unit: exp024-mutual-path-filter-20m.service
run_id: pure_rl_seed23_20m_mutual_path_filter
```

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

## 结果表

seed23 20M 训练、候选评估、post-hoc checkpoint reselection 和 final eval 已完成。结果以以下机器可读文件为准：

```text
outputs/runs/exp024_randomized_terrain_mutual_path_filter/pure_rl_seed23_20m_mutual_path_filter/metrics/strict_acceptance.json
outputs/runs/exp024_randomized_terrain_mutual_path_filter/pure_rl_seed23_20m_mutual_path_filter/metrics/summary.json
outputs/runs/exp024_randomized_terrain_mutual_path_filter/pure_rl_seed23_20m_mutual_path_filter/metrics/final_eval_proxy.json
outputs/runs/exp024_randomized_terrain_mutual_path_filter/pure_rl_seed23_20m_mutual_path_filter/metrics/eval_metrics.json
```

| eval | checkpoint | dmax ratio | success | collision | timeout | filter applied | strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| original best by `progress_preserving_long` | `ppo_timestep_002048.pt` | 0.2116 | 0.2402 | 0.0381 | 0.7266 | 0.1130 | 未通过 |
| best candidate by `success_progress_long` | `ppo_timestep_010240.pt` | 0.1397 | 0.8398 | 0.0674 | 0.0947 | 0.2605 | 未通过 |
| final eval seed1023 | `best.pt` = `ppo_timestep_010240.pt` | 0.1397 | 0.8398 | 0.0674 | 0.0947 | 0.2605 | 未通过 |

关键 telemetry：

```text
filter_raw_mutual_path_collision_violation_mean: 0.0341
filter_mutual_path_collision_violation_mean: 0.000879
filter_raw_path_collision_violation_mean: 0.0294
filter_path_collision_violation_mean: 0.00172
filter_collision_override_fraction: 0.1803
first_collision_step_mean: 174.6 / 220
```

## 失败分析

strict gate 未通过：

- `dmax_reduction_ratio=0.1397`，通过 `<=0.20`。
- `success_rate=0.8398`，低于 `>=0.90`。
- `collision_rate=0.0674`，高于 `<=0.02`。
- `timeout_rate=0.0947`，高于 `0`。

和 exp023 对比：

- success 从 `0.3027` 提升到 `0.8398`。
- collision 从 `0.2295` 降到 `0.0674`。
- timeout 从 `0.4717` 降到 `0.0947`。
- mutual path violation 被明显压低，说明动态路径建模有效。

当前判断：

exp024 是最近几轮最接近 strict 的随机地形结果，但仍不是 pass。剩余失败主要是 late-stage collision 和少量 timeout；此时不宜恢复 exp022 hard constraint，因为 exp024 已证明软 mutual path 能保留集合进度。下一轮应在 exp024 基础上做末段速度/hold 稳定或 success-zone safety shaping，目标是把 collision 从 `0.0674` 压到 `<=0.02`，同时保住 success。

## 产物路径

```text
outputs/runs/exp024_randomized_terrain_mutual_path_filter/pure_rl_seed23_20m_mutual_path_filter/
outputs/runs/exp024_randomized_terrain_mutual_path_filter/_suite/metrics/
outputs/runs/exp024_randomized_terrain_mutual_path_filter/_suite/figures/
outputs/runs/exp024_randomized_terrain_mutual_path_filter/_launcher/train.log
```

生成的 checkpoint、JSON、TensorBoard、PNG、GIF 均保留在 `outputs/`，不提交到 git。
