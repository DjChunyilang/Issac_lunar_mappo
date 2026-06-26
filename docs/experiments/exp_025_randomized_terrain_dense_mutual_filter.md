# exp025 dense mutual path safety filter 随机地形实验

## 目的

exp024 把 success 提升到 `0.8398`，collision 降到 `0.0674`，说明 mutual path safety 是有效方向，但仍未通过 strict。exp025 不改接口和算法，只做最小步调参：把 path safety 采样从 5 个点加密到 9 个点，并适度提高 path/mutual collision 权重，目标是进一步降低 late-stage collision，同时用 `success_progress_long` 避免选择早期低成功 checkpoint。

本轮仍是 proxy 训练，不引入 PhysX、真实车体尺寸、轮地接触、硬性陷车终止或 BC。

## 配置

```text
config: configs/experiment/exp025_randomized_terrain_dense_mutual_filter.yaml
experiment_id: exp025_randomized_terrain_dense_mutual_filter
run_id: pure_rl_seed23_20m_dense_mutual_filter
```

关键设置：

- shared-joint MAPPO pure RL，`bc_updates=0`。
- 随机增强 lunar crater proxy，`randomize_per_reset=true`。
- 通信半径 `12 m`，Actor / Critic 接口仍为 `86 / 54`。
- 2048 CUDA envs，rollout 32，连续 `10240` timesteps。
- checkpoint interval `1024` timesteps。
- 候选选择使用 `success_progress_long`。

dense mutual filter：

```text
mode: terrain_safe_candidate_mutual_progress_curriculum
path_samples: 9
rho_scales: [0.60, 0.80, 1.0, 1.08]
beta_offsets_deg: [-40, -25, -12.5, 0, 12.5, 25, 40]
candidate_count: 28
apply_probability: 0.0 -> 0.50
score_scale: 0.20 -> 0.65
path_safe_distance: 0.34
path_collision_weight: 450.0
mutual_path_near_weight: 1.80
mutual_path_collision_weight: 1200.0
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

- collision 是否低于 exp024 的 `0.0674`。
- success 是否保持接近或高于 exp024 的 `0.8398`。
- timeout 是否低于 exp024 的 `0.0947`。
- `filter_mutual_path_collision_violation_mean` 是否继续明显低于 raw。

## 结果

seed23 20M 已完成，候选选择使用 `success_progress_long`，当前 best 为：

```text
outputs/runs/exp025_randomized_terrain_dense_mutual_filter/pure_rl_seed23_20m_dense_mutual_filter/checkpoints/ppo_timestep_009216.pt
```

最终独立 eval：

| checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | ---: | ---: | ---: | ---: | --- |
| `ppo_timestep_009216.pt` | `0.1434` | `0.8525` | `0.0449` | `0.1035` | 未通过 |

候选评估摘要：

| timestep | dmax ratio | success | collision | timeout |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.3756 | 0.0088 | 0.0693 | 0.9248 |
| 2048 | 0.2338 | 0.2158 | 0.0098 | 0.7773 |
| 3072 | 0.1974 | 0.4199 | 0.0449 | 0.5371 |
| 4096 | 0.1565 | 0.6846 | 0.0898 | 0.2295 |
| 5120 | 0.1460 | 0.7959 | 0.0811 | 0.1279 |
| 6144 | 0.1453 | 0.8301 | 0.0723 | 0.0986 |
| 7168 | 0.1429 | 0.8281 | 0.0684 | 0.1084 |
| 8192 | 0.1425 | 0.8320 | 0.0674 | 0.1035 |
| 9216 | 0.1441 | 0.8506 | 0.0410 | 0.1133 |
| 10240 | 0.1431 | 0.8330 | 0.0449 | 0.1299 |

关键 telemetry：

```text
filter_raw_mutual_path_collision_violation_mean: 0.0363
filter_mutual_path_collision_violation_mean: 0.0010
filter_raw_path_collision_violation_mean: 0.0318
filter_path_collision_violation_mean: 0.0014
filter_applied_fraction: 0.2785
filter_collision_override_fraction: 0.1866
max_success_hold_count_mean: 7.2246 / 8
final_success_hold_count_mean: 6.8838 / 8
first_collision_step_mean: 175.7 / 220
```

和 exp024 对比：

- success 从 `0.8398` 小幅升到 `0.8525`。
- collision 从 `0.0674` 降到 `0.0449`，说明 dense mutual path safety 有正向作用。
- timeout 从 `0.0947` 小幅升到 `0.1035`，未解决末段 hold / timeout 稳定性。
- strict 仍失败：success 未到 `0.90`，collision 仍高于 `0.02`，timeout 仍高于 `0`。

## 失败分析

exp025 是有效但不充分的小步改进。失败不再是“完全不集合”：dmax ratio、dispersion 和 final nearest distance 都接近可接受区间，且平均最大 success hold count 已达到 `7.22/8`。主要问题是集合末段仍不稳：

- 多数未通过 episode 已经接近成功区，但不能稳定保持满 `8` 个 hold steps。
- collision 仍发生在后段，`first_collision_step_mean≈175/220`，说明密集 mutual path 仍不能完全覆盖执行期的末段相互挤压。
- filter 对路径冲突的打分生效，但 `filter_applied_fraction≈0.28`，说明不能只继续放大后处理；否则容易回到 exp022 的 standoff。

下一轮应优先做末段 hold / success-zone 稳定，而不是继续单纯增加 path/mutual collision 权重。候选方向：

- 在接近 success zone 时降低子目标步长或速度，减少穿越/相向冲入集合区。
- 在 success hold 阶段加入轻量的 pairwise spacing / velocity damping shaping。
- 让 filter score 显式看“保持成功区”和“不要破坏 hold count”，而不只是当前路径风险和互相接近。

## 工程验证计划

专项测试：

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_subgoal_filter.py \
  tests/test_exp024_training.py \
  tests/test_exp025_training.py
```

完整测试：

```bash
.venv_isaaclab/bin/python -m pytest -q -ra
```

当前工程验证状态：

- exp025 专项测试已通过：`tests/test_subgoal_filter.py tests/test_exp024_training.py tests/test_exp025_training.py`。
- 完整 `.venv_isaaclab/bin/python -m pytest -q -ra` 已通过。
- CPU smoke 已通过：`smoke_cpu_exp025`，确认 run-oriented 输出、一个 optimizer、两次 joint update、terrain 输入权重更新和 warmup 期不替换 action。
- CUDA smoke 已通过：`smoke_cuda_exp025`，确认 256 env / 64 timesteps / rollout 32 链路正常，无 NaN，动作非退化。

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

## 长训练命令

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

## 产物路径

```text
outputs/runs/exp025_randomized_terrain_dense_mutual_filter/pure_rl_seed23_20m_dense_mutual_filter/
outputs/runs/exp025_randomized_terrain_dense_mutual_filter/_suite/metrics/
outputs/runs/exp025_randomized_terrain_dense_mutual_filter/_suite/figures/
outputs/runs/exp025_randomized_terrain_dense_mutual_filter/_launcher/train.log
```

生成的 checkpoint、JSON、TensorBoard、PNG、GIF 均保留在 `outputs/`，不提交到 git。

本轮实际生成的机器可读结果：

```text
outputs/runs/exp025_randomized_terrain_dense_mutual_filter/pure_rl_seed23_20m_dense_mutual_filter/metrics/summary.json
outputs/runs/exp025_randomized_terrain_dense_mutual_filter/pure_rl_seed23_20m_dense_mutual_filter/metrics/eval_metrics.json
outputs/runs/exp025_randomized_terrain_dense_mutual_filter/pure_rl_seed23_20m_dense_mutual_filter/metrics/final_eval_proxy.json
outputs/runs/exp025_randomized_terrain_dense_mutual_filter/pure_rl_seed23_20m_dense_mutual_filter/metrics/strict_acceptance.json
outputs/runs/exp025_randomized_terrain_dense_mutual_filter/pure_rl_seed23_20m_dense_mutual_filter/metrics/checkpoint_status.json
```
