# exp021 课程化/软化子目标过滤器随机地形实验

## 目的

exp020 证明 hard subgoal filter 可以降低路径地形风险，但它介入过早、过强，导致策略几乎不再学习集合。exp021 的目标是验证：如果训练前期保留 Actor raw action，后期再以低概率、低强度逐步启用 filter，同时把 raw path risk 和 filter deviation 作为辅助惩罚，是否能恢复集合学习并保留地形/安全收益。

本轮仍是 proxy 训练，不引入 PhysX、真实车体尺寸、轮地接触、硬性陷车终止或 BC。

## 配置

```text
config: configs/experiment/exp021_randomized_terrain_filter_curriculum.yaml
experiment_id: exp021_randomized_terrain_filter_curriculum
run_id: pure_rl_seed23_20m_filter_curriculum
```

关键设置：

- shared-joint MAPPO pure RL，`bc_updates=0`。
- 随机增强 lunar crater proxy，`randomize_per_reset=true`。
- 通信半径 `12 m`，Actor / Critic 接口仍为 `86 / 54`。
- 2048 CUDA envs，rollout 32，连续 `10240` timesteps，即约 20.97M env steps。
- checkpoint interval `1024` timesteps。
- 候选选择使用 `balanced_progress_long`。

课程化 filter：

```text
mode: terrain_safe_candidate_curriculum
rho_scales: [0.85, 1.0]
beta_offsets_deg: [-30, -15, 0, 15, 30]
warmup_timesteps: 2048
ramp_timesteps: 4096
apply_probability: 0.0 -> 0.60
score_scale: 0.15 -> 0.75
deterministic_improvement_margin: 0.02
visible_neighbor_center_weight: 0.35
```

reward 新增辅助项：

```text
filter_raw_path_risk_cost: 0.30
filter_deviation_cost: 0.10
```

默认配置中这两个系数为 0，因此 exp020 及更早实验语义不变。

## 严格标准

proxy strict gate 不变：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

如果 strict 未通过，也要与 exp020 比较：

- success 必须高于 exp020 的 `0.0`。
- collision 目标低于 exp020 的 `0.0498`。
- timeout 目标低于 exp020 的 `0.9506`。
- filtered path risk 允许略高于 exp020，但应低于 exp019 5-seed 均值约 `0.387`。

## 结果表

训练、候选评估、5 轮独立 eval、GIF、height map 和曲线已完成。结果以以下机器可读文件为准：

```text
outputs/runs/exp021_randomized_terrain_filter_curriculum/_suite/metrics/strict_acceptance.json
outputs/runs/exp021_randomized_terrain_filter_curriculum/_suite/metrics/suite_summary.json
outputs/runs/exp021_randomized_terrain_filter_curriculum/pure_rl_seed23_20m_filter_curriculum/metrics/final_eval_proxy.json
outputs/runs/exp021_randomized_terrain_filter_curriculum/pure_rl_seed23_20m_filter_curriculum/metrics/eval_metrics.json
outputs/runs/exp021_randomized_terrain_filter_curriculum/pure_rl_seed23_20m_filter_curriculum/metrics/multi_eval_20260626_120357/multi_eval_summary.json
```

| eval | checkpoint | dmax ratio | success | collision | timeout | path risk mean | strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| candidate eval seed1023 | `ppo_timestep_010240.pt` | 0.1444 | 0.6621 | 0.1602 | 0.1826 | 0.3626 | 未通过 |
| final eval seed1023 | `best.pt` | 0.1434 | 0.6396 | 0.1885 | 0.1797 | 0.3615 | 未通过 |
| 5 seed mean `12023–12027` | `best.pt` | 0.1460 | 0.6361 | 0.1746 | 0.1967 | 0.3638 | 未通过 |

5 seed filter 统计：

```text
filter_raw_path_terrain_risk_mean: 0.3871
filter_filtered_path_terrain_risk_mean: 0.3638
filter_path_terrain_risk_reduction_mean: 0.0233
filter_applied_fraction: 0.2185
final_nearest_neighbor_distance: 0.5106
min_nearest_distance: 0.2244
```

## 失败分析

工程验证：

- exp021 专项测试通过。
- 完整 `.venv_isaaclab/bin/python -m pytest -q -ra` 通过。
- CPU smoke 和 CUDA smoke 通过。
- CUDA smoke 确认 `optimizer_count=1`、`joint_update_count=2`、terrain 输入权重更新、动作非退化。
- warmup checkpoint 的 deterministic eval 不替换 action，`filter_applied_fraction=0`。

严格标准未通过：

- `dmax_reduction_ratio=0.1460` 通过。
- `success_rate=0.6361` 未达到 `0.90`。
- `collision_rate=0.1746` 远高于 `0.02`。
- `timeout_rate=0.1967` 未达到 `0`。

和 exp020 对比：

- success 从 `0.0` 恢复到 `0.6361`，说明课程化/软化 filter 确实恢复了集合学习。
- timeout 从 `0.9506` 降到 `0.1967`，不再是纯安全徘徊。
- collision 从 `0.0498` 升到 `0.1746`，成为最主要失败项。
- filtered path risk `0.3638` 低于 exp019 的约 `0.387`，但高于 exp020 的 `0.3187`；这符合“弱化 filter 换回集合进度”的预期。

当前判断：

exp021 解决了 exp020 的“安全但不集合”问题，但重新暴露出 exp019 后期的“会集合但撞车”问题。也就是说，post-processing filter 的强弱本身不是最终答案：强 filter 会牺牲集合，弱/课程化 filter 会恢复集合但不能控制多车近距离碰撞。

## 产物路径

```text
outputs/runs/exp021_randomized_terrain_filter_curriculum/pure_rl_seed23_20m_filter_curriculum/
outputs/runs/exp021_randomized_terrain_filter_curriculum/_suite/metrics/
outputs/runs/exp021_randomized_terrain_filter_curriculum/_launcher/train.log
```

复验/GIF、height map 和曲线：

```text
outputs/runs/exp021_randomized_terrain_filter_curriculum/pure_rl_seed23_20m_filter_curriculum/metrics/multi_eval_20260626_120357/
outputs/runs/exp021_randomized_terrain_filter_curriculum/pure_rl_seed23_20m_filter_curriculum/videos/multi_eval_20260626_120357/
outputs/runs/exp021_randomized_terrain_filter_curriculum/pure_rl_seed23_20m_filter_curriculum/figures/training_curves.png
outputs/runs/exp021_randomized_terrain_filter_curriculum/pure_rl_seed23_20m_filter_curriculum/figures/candidate_eval_curves.png
outputs/runs/exp021_randomized_terrain_filter_curriculum/_suite/figures/training_curves.png
outputs/runs/exp021_randomized_terrain_filter_curriculum/_suite/figures/candidate_eval_curves.png
```

## 结论

exp021 不能作为当前主结果，也不能写成随机地形收敛。它是一个重要诊断：课程化 filter 成功恢复了集合进度，但碰撞率显著过高。下一轮不应继续只调 filter 强度或 terrain reward，而应把“集合成功动作”和“多车安全间距”更强地耦合到 action representation、局部 planner 或终止/约束设计中。

## 下一步

下一轮优先做安全/集合联合 action representation 或 endpoint conflict-aware policy loss。具体方向：

- 在 action / planner 层显式预测 endpoint pairwise distance，而不是只在 reward 后验惩罚碰撞。
- 对候选子目标加入“集合进度下限 + endpoint 安全”联合约束，避免为了集合直接穿越邻车。
- 考虑把多车相对位姿的安全边界做成 differentiable auxiliary loss 或 shield，而不是只用 episode 终止惩罚。
