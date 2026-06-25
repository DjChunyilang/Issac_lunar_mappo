# exp020 地形/安全感知子目标过滤器

## 目的

exp019 证明路径级 terrain risk 已经进入训练信号，但软惩罚没有稳定诱导绕障；后期 checkpoint 会集合但 collision 高，早期 best 又安全但几乎不成功。exp020 把 terrain risk 前移到 actor action 后处理：在 `[rho, beta]` 解码后、轨迹生成前，从固定候选子目标中选择路径风险更低、endpoint safety 更好的子目标。

## 配置

```text
configs/experiment/exp020_randomized_terrain_subgoal_filter.yaml
```

沿用 exp019 的 shared-joint MAPPO、pure RL、随机增强 lunar crater、`12 m` 通信半径、`2048` CUDA env、rollout `32`、20M 预算和 strict gate。关键新增：

- `planner.subgoal_filter.enabled: true`
- 候选集合：`rho_scales=[0.65, 1.0]` × `beta_offsets_deg=[-45, -22.5, 0, 22.5, 45]`
- 每个候选路径采样 `5` 点。
- score 固定为：

```text
0.35 * intent_deviation
+ 0.70 * path_terrain_mean
+ 0.50 * path_terrain_max
+ 0.20 * path_height_change
+ 0.30 * subgoal_risk
+ 2.00 * endpoint_near_violation
+ 1000.0 * endpoint_collision_violation
```

filter 只考虑通信半径内可见邻居；不改变 Actor 输入/输出维度，也不改变 Critic 维度。

## 严格标准

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

新增诊断指标：

```text
filter_applied_fraction
filter_raw_path_terrain_risk_mean
filter_filtered_path_terrain_risk_mean
filter_path_terrain_risk_reduction_mean
filter_endpoint_collision_violation_fraction
```

## 结果表

seed23 20M 训练正常完成：

```text
joint_update_count: 320
optimizer_count: 1
policy_parameter_delta_l2: 5.8930
terrain_input_weight_delta_l2: 1.7635
post_training_action_std: 0.4563
```

final eval：

```text
dmax_reduction_ratio: 0.3765
success_rate: 0.0010
safe_success_rate: 0.0010
collision_rate: 0.0439
timeout_rate: 0.9570
path_terrain_risk_mean: 0.3178
filter_applied_fraction: 0.4455
filter_raw_path_terrain_risk_mean: 0.3812
filter_filtered_path_terrain_risk_mean: 0.3178
filter_path_terrain_risk_reduction_mean: 0.0634
```

5 轮独立 eval，seeds `12023–12027`：

```text
dmax_reduction_ratio: 0.3752
success_rate: 0.0000
safe_success_rate: 0.0000
collision_rate: 0.0498
timeout_rate: 0.9506
final_nearest_neighbor_distance: 1.1017 m
path_terrain_risk_mean: 0.3187
filter_applied_fraction: 0.4433
filter_raw_path_terrain_risk_mean: 0.3815
filter_filtered_path_terrain_risk_mean: 0.3187
filter_path_terrain_risk_reduction_mean: 0.0628
```

候选 checkpoint 中 `2048` 和 `3072` timesteps 有少量 success，但 collision 很高；`10240` 被 `safe_progress_long` 选为 best，仍未通过 strict。

| checkpoint timestep | env steps | dmax ratio | success | collision | timeout | path risk mean | risk reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 2,097,152 | 0.4599 | 0.0010 | 0.0547 | 0.9463 | 0.3223 | 0.0380 |
| 2048 | 4,194,304 | 0.2942 | 0.0547 | 0.2988 | 0.6504 | 0.3258 | 0.0456 |
| 3072 | 6,291,456 | 0.2620 | 0.0557 | 0.2451 | 0.7080 | 0.3239 | 0.0526 |
| 5120 | 10,485,760 | 0.3357 | 0.0146 | 0.1436 | 0.8438 | 0.3308 | 0.0634 |
| 10240 | 20,971,520 | 0.3724 | 0.0000 | 0.0420 | 0.9600 | 0.3181 | 0.0631 |

## 失败分析

exp020 的工程目标成立：filter 在训练、candidate eval 和 5 seed eval 中稳定降低路径风险，5 seed raw risk `0.3815` 降到 filtered risk `0.3187`。但策略层面失败更严重：

- success 被压到 `0`，说明 hard filter 抑制了集合进度和探索。
- collision 仍为 `0.0498`，虽然低于 exp019 后期高 collision checkpoint，但仍高于 `0.02` strict gate。
- timeout 约 `0.95`，说明多数 episode 安全地绕行/徘徊而非完成集合。

因此本轮不能作为成功策略，只能作为“硬过滤器降低 path risk 但牺牲任务进度”的 failure analysis。

## 产物路径

```text
outputs/runs/exp020_randomized_terrain_subgoal_filter/pure_rl_seed23_20m_subgoal_filter/metrics/final_eval_proxy.json
outputs/runs/exp020_randomized_terrain_subgoal_filter/pure_rl_seed23_20m_subgoal_filter/metrics/eval_metrics.json
outputs/runs/exp020_randomized_terrain_subgoal_filter/pure_rl_seed23_20m_subgoal_filter/metrics/multi_eval_20260625_101520/
outputs/runs/exp020_randomized_terrain_subgoal_filter/pure_rl_seed23_20m_subgoal_filter/videos/multi_eval_20260625_101520/
outputs/runs/exp020_randomized_terrain_subgoal_filter/pure_rl_seed23_20m_subgoal_filter/figures/training_curves.png
outputs/runs/exp020_randomized_terrain_subgoal_filter/pure_rl_seed23_20m_subgoal_filter/figures/candidate_eval_curves.png
outputs/runs/exp020_randomized_terrain_subgoal_filter/_suite/metrics/
outputs/runs/exp020_randomized_terrain_subgoal_filter/_suite/figures/
```

每个 eval seed 都有：

```text
videos/multi_eval_20260625_101520/seed<seed>/proxy_eval_rollout.gif
videos/multi_eval_20260625_101520/seed<seed>/terrain_height_map.png
```

## 结论

exp020 未通过 strict gate，不能作为随机地形成功结果。它说明“硬性 terrain/safety 子目标替换”确实能降低路径风险，但会破坏集合任务学习；下一轮不应继续增加 filter 强度。

## 下一步

建议把 filter 改成课程化或软约束：

- 训练前期关闭或低概率启用 filter，先保留集合探索；随后逐步提高 applied probability 或 score 权重。
- 或把 filter score 作为 auxiliary penalty / critic feature，而不是直接替换 action。
- 若继续做 hard filter，需要加入集合中心/邻居中心 intent 项，避免所有动作被推向安全但不集合的候选。
