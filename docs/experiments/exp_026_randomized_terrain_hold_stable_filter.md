# exp026 hold-stable subgoal filter 随机地形实验

## 目的

exp025 证明 dense mutual path safety 能把 collision 从 exp024 的 `0.0674` 降到 `0.0449`，但 strict 仍失败，且 timeout 没有改善。关键诊断信号是 `max_success_hold_count_mean=7.2246/8`、`first_collision_step_mean≈175.7/220`：策略常常已经接近成功区，却不能稳定保持 8 个 hold steps，并在末段出现碰撞或 timeout。

exp026 不继续单纯加大 path/mutual collision 权重，而是在子目标过滤器中加入默认关闭的 hold-zone cost：当当前队形已经接近 success gate 时，候选 score 会额外偏好更短的 rho 和更大的 endpoint pairwise buffer，目标是减少末段过冲和相向冲入。

本轮仍是 proxy 训练，不引入 PhysX、真实车体尺寸、轮地接触、硬性陷车终止或 BC。

## 配置

```text
config: configs/experiment/exp026_randomized_terrain_hold_stable_filter.yaml
experiment_id: exp026_randomized_terrain_hold_stable_filter
run_id: pure_rl_seed23_20m_hold_stable_filter
```

关键设置：

- shared-joint MAPPO pure RL，`bc_updates=0`。
- 随机增强 lunar crater proxy，`randomize_per_reset=true`。
- 通信半径 `12 m`，Actor / Critic 接口仍为 `86 / 54`。
- 2048 CUDA envs，rollout 32，连续 `10240` timesteps。
- checkpoint interval `1024` timesteps。
- 候选选择使用 `success_progress_long`。

hold-stable filter：

```text
mode: terrain_safe_candidate_hold_progress_curriculum
path_samples: 9
rho_scales: [0.45, 0.65, 0.85, 1.0]
beta_offsets_deg: [-40, -25, -12.5, 0, 12.5, 25, 40]
candidate_count: 28
apply_probability: 0.0 -> 0.50
score_scale: 0.20 -> 0.65
endpoint_safe_distance: 0.46
path_safe_distance: 0.36
hold_zone_dmax_multiplier: 1.20
hold_zone_dispersion_multiplier: 1.60
hold_zone_rho_weight: 0.65
hold_zone_spacing_weight: 2.20
hold_zone_pairwise_distance: 0.48
```

hold-zone 只在当前 dmax 和 dispersion 已接近 success gate 时激活。默认配置中这些字段为 0，因此 exp025 及更早实验行为不变。

## 验收标准

proxy strict gate 不变：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

如果 strict 未通过，本轮诊断重点看：

- success 是否高于 exp025 的 `0.8525`。
- collision 是否低于 exp025 的 `0.0449`。
- timeout 是否低于 exp025 的 `0.1035`。
- `max_success_hold_count_mean` 是否从 exp025 的 `7.2246/8` 提高到更接近 8。
- `filter_hold_zone_activation_mean`、`filter_hold_zone_rho_cost_mean` 和 `filter_hold_zone_spacing_violation_mean` 是否有限且非零，证明 hold-zone cost 确实参与 eval。

## 工程验证计划

专项测试：

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_subgoal_filter.py \
  tests/test_exp026_training.py
```

完整测试：

```bash
.venv_isaaclab/bin/python -m pytest -q -ra
```

当前工程验证状态：

- exp026 专项测试已通过：`tests/test_subgoal_filter.py tests/test_exp026_training.py`。
- 完整 `.venv_isaaclab/bin/python -m pytest -q -ra` 已通过。
- CPU smoke 已通过：`smoke_cpu_exp026`，确认 run-oriented 输出、一个 optimizer、两次 joint update、terrain 输入权重更新和 hold-zone telemetry 字段。
- CUDA smoke 已通过：`smoke_cuda_exp026`，确认 256 env / 64 timesteps / rollout 32 链路正常，无 NaN，动作非退化。

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

## 长训练命令

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

监控：

```bash
tail -f outputs/runs/exp026_randomized_terrain_hold_stable_filter/_launcher/train.log
```

## 产物路径

```text
outputs/runs/exp026_randomized_terrain_hold_stable_filter/pure_rl_seed23_20m_hold_stable_filter/
outputs/runs/exp026_randomized_terrain_hold_stable_filter/_suite/metrics/
outputs/runs/exp026_randomized_terrain_hold_stable_filter/_suite/figures/
outputs/runs/exp026_randomized_terrain_hold_stable_filter/_launcher/train.log
```

生成的 checkpoint、JSON、TensorBoard、PNG、GIF 均保留在 `outputs/`，不提交到 git。

## 结果

| run | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `pure_rl_seed23_20m_hold_stable_filter` | `best.pt` | 0.1474 | 0.7529 | 0.0615 | 0.1865 | 未通过 |

关键 telemetry：

```text
max_success_hold_count_mean: 6.5781 / 8
final_success_hold_count_mean: 6.1064 / 8
filter_hold_zone_activation_mean: 0.1731
filter_hold_zone_rho_cost_mean: 0.0933
filter_hold_zone_spacing_violation_mean: 0.0061
first_collision_step_mean: 181.4 / 220
```

## 失败分析

exp026 未改善 exp025。hold-zone activation 太早太宽：`hold_zone_dmax_multiplier=1.20`、`hold_zone_dispersion_multiplier=1.60` 会在尚未真正稳定进入 success gate 时就偏好短 rho 和 spacing buffer。结果是集合进度被压制，success 从 exp025 的 `0.8525` 降到 `0.7529`，timeout 从 `0.1035` 升到 `0.1865`。

## 结论

不能作为当前主结果。下一轮应把 hold-zone 触发条件收窄，或改用 reward-side hold shaping，避免在接近区过早阻碍集合。
