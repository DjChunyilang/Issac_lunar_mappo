# exp043 structured bicycle quintic map25 长训

## 目的

在 exp042 已验证的新训练环境栈上恢复长训，检验结构化 Actor/Critic、bicycle proxy、quintic 轨迹、`25 m × 25 m` 地图和无限通信语义能否在随机增强地形上重新达到或超过 exp038/exp041 的收敛趋势。

本轮不是从旧 checkpoint 续训，而是 seed23 从随机初始化开始。

## 配置

```text
configs/experiment/exp043_structured_bicycle_quintic_map25_long.yaml
```

关键设置：

- Actor/Critic：`branched_v1 / structured_v1`，接口保持 `86 / 54`。
- 动力学/轨迹：`low_level_control.kinematic_model=bicycle`，`trajectory_generator.geometry_method=quintic`，`n_trajectory_points=12`。
- 地图/通信：`world_xy_limit=12.5`、`crater_field_size=25.0`、`communication_radius=0.0`。
- reset 分布：spawn radius `4.5–6.5 m`，center range `±3.0 m`，jitter `0.45 m`。
- 地形：`crater_count=48`、`randomize_per_reset=true`、`random_translation_m=5.0`。
- 末端稳定：继承 exp041 的 hold-zone override，`hold_zone_spacing_weight=8.0`、`hold_zone_pairwise_distance=0.58`。
- MAPPO：2048 env、rollout 64、LR `1.2e-4`、entropy `0.003 -> 0.0005` over 8192 timesteps。
- 预算：`20480` timesteps，约 `41,943,040` env steps，checkpoint interval `1024`。

## 严格标准

仍使用标准 proxy strict gate：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

## 结果表

| seed | run_id | checkpoint | final_eval | strict |
| --- | --- | --- | --- | --- |
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25` | smoke only | smoke 通过 | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25` | smoke only | smoke 通过 | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25` | `ppo_timestep_020480.pt` | dmax `0.8596` / success `0.0` / collision `0.0` / timeout `1.0` | 未通过 |

CUDA smoke 摘要：

```text
num_envs: 256
timesteps: 128
rollout_steps: 64
optimizer_count: 1
joint_update_count: 2
terrain_input_weight_delta_l2: 0.1082
post_training_action_std: 0.2978
initial_dmax: 11.3953
path_terrain_risk_mean: 0.3702
```

## 失败分析

训练已完成，机器可读结果显示工程链路正常但策略没有形成有效集合行为：

```text
timesteps: 20480
env_steps: 41943040
best_candidate: ppo_timestep_020480.pt
policy_parameter_delta_l2: 16.9715
terrain_input_weight_delta_l2: 2.2086
post_training_action_std: 0.7611
optimizer_count: 1
joint_update_count: 320
```

final eval：

```text
dmax_reduction_ratio: 0.8596
success_rate: 0.0
safe_success_rate: 0.0
collision_rate: 0.0
timeout_rate: 1.0
initial_dmax: 11.4048
final_dmax: 9.8041
final_dispersion: 22.3341
final_nearest_neighbor_distance: 5.7386
final_mean_speed: 0.0056
path_terrain_risk_mean: 0.4159
```

判读：

- 不是 optimizer / shared update / checkpoint 链路故障：参数、terrain branch、动作方差和 320 次 joint update 都正常。
- 不是碰撞主导失败：collision 为 `0.0`。
- 主要失败是集合进度不足：final dmax 仍接近 `9.8 m`，success 为 `0.0`，timeout 为 `1.0`。
- 直接把 25m 地图、较大 reset 初始分布、bicycle 运动学、quintic 轨迹和随机地形一起交给 pure RL，冷启动难度过大。
- 下一轮应先降低初始队形难度并课程化恢复目标分布，而不是继续简单拉长相同配置预算。

## 产物路径

```text
outputs/runs/exp043_structured_bicycle_quintic_map25_long/
  pure_rl_seed23_40m_structured_bicycle_quintic_map25/
  _launcher/train.log
```

结果事实来源：

```text
outputs/runs/exp043_structured_bicycle_quintic_map25_long/pure_rl_seed23_40m_structured_bicycle_quintic_map25/metrics/summary.json
outputs/runs/exp043_structured_bicycle_quintic_map25_long/pure_rl_seed23_40m_structured_bicycle_quintic_map25/metrics/final_eval_proxy.json
outputs/runs/exp043_structured_bicycle_quintic_map25_long/pure_rl_seed23_40m_structured_bicycle_quintic_map25/metrics/strict_acceptance.json
```

## 结论

exp043 未通过 strict gate。它保留为“新环境栈直接长训失败基线”：证明工程链路可长跑，但直接大初始分布下没有恢复集合学习，不能写成 strict pass。

## 下一步

已新建 exp044：保留新环境栈，但加入 initial-state curriculum，并把目标 reset 分布适度收回到 `3.8–5.2 m` spawn radius；训练时 warmup/ramp，评估仍使用最终目标分布。
