# exp044 structured bicycle quintic map25 initial-state curriculum

## 目的

exp043 证明新环境栈可以稳定长跑，但直接在 `25 m × 25 m` 地图和较大初始队形分布上 pure RL 冷启动失败：final eval success 为 `0.0`、timeout 为 `1.0`。exp044 保留结构化网络、bicycle proxy、quintic 轨迹和无限通信语义，只改变学习课程与难度：

- 训练时先从较近初始队形开始，避免一开始就面对约 `11 m` initial dmax；
- warmup 后逐步 ramp 到目标 reset 分布；
- 独立评估仍使用目标 reset 分布，避免把课程难度本身当成成绩。

本轮从随机初始化开始，不续训 exp038/exp041/exp043 checkpoint。

## 配置

```text
configs/experiment/exp044_structured_bicycle_quintic_map25_curriculum.yaml
```

关键设置：

- Actor/Critic：`branched_v1 / structured_v1`，接口保持 `86 / 54`。
- 动力学/轨迹：`low_level_control.kinematic_model=bicycle`，`trajectory_generator.geometry_method=quintic`，`n_trajectory_points=12`。
- 地图/通信：`world_xy_limit=12.5`、`crater_field_size=25.0`、`communication_radius=0.0`。
- 目标 reset 分布：spawn radius `3.8–5.2 m`，center range `±2.0 m`，jitter `0.40 m`。
- initial-state curriculum：前 `4096` timesteps 使用 spawn radius `3.0–4.0 m`、center range `±1.0 m`、jitter `0.35 m`；随后 `8192` timesteps 线性 ramp 到目标分布。
- 地形：`crater_count=36`、`randomize_per_reset=true`、`random_translation_m=5.0`。
- 末端稳定：继承 exp041/exp043 的 hold-zone override。
- MAPPO：2048 env、rollout 64、LR `1.2e-4`、entropy `0.0015 -> 0.0003` over 8192 timesteps。
- 预算：`20480` timesteps，约 `41,943,040` env steps，checkpoint interval `1024`。

## 严格标准

仍使用标准 proxy strict gate：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

## 工程验证

专项测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaaclab/bin/python -m pytest -q \
  tests/test_config_wiring.py \
  tests/test_observation.py \
  tests/test_skrl_mappo_semantics.py
```

完整测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaaclab/bin/python -m pytest -q -ra
```

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

## 长训命令

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

## 结果表

| seed | run_id | checkpoint | final_eval | strict |
| --- | --- | --- | --- | --- |
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_curriculum` | smoke only | optimizer `1`、joint update `2`、terrain delta `0.1056`、action std `0.1339` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_curriculum` | smoke only | optimizer `1`、joint update `2`、terrain delta `0.1189`、action std `0.1581` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_curriculum` | `ppo_timestep_020480.pt` | dmax `0.4796` / success `0.0` / collision `0.00195` / timeout `0.9980` | 未通过 |

smoke telemetry 已确认：

```text
CPU smoke final progress: 8
CUDA smoke final progress: 128
effective spawn radius: 3.0–4.0 m
effective center range: 1.0 m
```

这说明短 smoke 仍处在 initial-state curriculum warmup 起点；长训中应在 `4096–12288` timesteps 之间 ramp 到目标 reset 分布。

## 判读重点

训练已完成，initial-state curriculum 确实改善了 exp043 的“几乎不靠拢”问题，但仍没有进入 success basin：

```text
timesteps: 20480
env_steps: 41943040
best_candidate: ppo_timestep_020480.pt
policy_parameter_delta_l2: 12.3614
terrain_input_weight_delta_l2: 2.5407
post_training_action_std: 0.4419
optimizer_count: 1
joint_update_count: 320
```

final eval：

```text
initial_dmax: 9.3545
final_dmax: 4.4865
dmax_reduction_ratio: 0.4796
success_rate: 0.0
safe_success_rate: 0.0
collision_rate: 0.00195
timeout_rate: 0.9980
final_nearest_neighbor_distance: 1.9176
final_mean_speed: 0.0518
```

候选评估显示全部 checkpoint 的 `success_rate=0.0`；后期 best 的 dmax ratio 稳定在约 `0.48`，没有继续推进到 `0.20` 附近。

失败分析：

- 相比 exp043，exp044 把 final dmax 从约 `9.8 m` 改善到约 `4.5 m`，说明课程化 reset 是有效方向。
- collision 已达 strict，说明当前主要问题不是安全约束。
- 动作并不退化：post-training action std `0.4419`，rho/beta 在 eval 中有明显高饱和；问题不是“策略不动”，而是完整目标 reset 分布仍太难，策略只学到中距离靠拢。
- filter 在 final eval 中 applied fraction 约 `0.40`，降低 path risk 但没有把队形推入 success zone；下一步应先做 local-success bootstrap，而不是继续加安全/地形过滤。

## 产物路径

```text
outputs/runs/exp044_structured_bicycle_quintic_map25_curriculum/
  pure_rl_seed23_40m_structured_bicycle_quintic_map25_curriculum/
  _launcher/train.log
```

机器可读事实来源：

```text
outputs/runs/exp044_structured_bicycle_quintic_map25_curriculum/pure_rl_seed23_40m_structured_bicycle_quintic_map25_curriculum/metrics/summary.json
outputs/runs/exp044_structured_bicycle_quintic_map25_curriculum/pure_rl_seed23_40m_structured_bicycle_quintic_map25_curriculum/metrics/final_eval_proxy.json
outputs/runs/exp044_structured_bicycle_quintic_map25_curriculum/pure_rl_seed23_40m_structured_bicycle_quintic_map25_curriculum/metrics/strict_acceptance.json
```

## 结论

exp044 未通过 strict gate，不能作为收敛结果。它保留为“initial-state curriculum 有效但不足”的诊断基线。

## 下一步

新建 exp045：保持新网络、bicycle、quintic、25m 地图和无限通信，但进一步缩小 target reset 分布、放大动作 reach、增强中距离 gather progress，并减轻 terrain/filter 对早期集合的压制。目标是先让 success 从 `0` 起跳；若 exp045 成功，再逐步扩展回 exp044 的目标分布。
