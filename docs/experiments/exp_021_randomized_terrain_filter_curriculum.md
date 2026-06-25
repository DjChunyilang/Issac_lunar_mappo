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

训练正在运行，尚未完成。结果以以下机器可读文件为准：

```text
outputs/runs/exp021_randomized_terrain_filter_curriculum/_suite/metrics/strict_acceptance.json
outputs/runs/exp021_randomized_terrain_filter_curriculum/_suite/metrics/suite_summary.json
outputs/runs/exp021_randomized_terrain_filter_curriculum/pure_rl_seed23_20m_filter_curriculum/metrics/final_eval_proxy.json
outputs/runs/exp021_randomized_terrain_filter_curriculum/pure_rl_seed23_20m_filter_curriculum/metrics/eval_metrics.json
```

## 失败分析

待训练和独立复验完成后填写。当前已完成工程验证：

- exp021 专项测试通过。
- 完整 `.venv_isaaclab/bin/python -m pytest -q -ra` 通过。
- CPU smoke 和 CUDA smoke 通过。
- CUDA smoke 确认 `optimizer_count=1`、`joint_update_count=2`、terrain 输入权重更新、动作非退化。
- warmup checkpoint 的 deterministic eval 不替换 action，`filter_applied_fraction=0`。

后续重点对比：

- `success_rate` 是否从 exp020 的 0 恢复；
- `collision_rate` 是否低于 exp020 后期高碰撞 checkpoint；
- `timeout_rate` 是否低于 exp020 的安全徘徊状态；
- `filter_applied_fraction` 是否符合课程预期；
- `filter_raw_path_terrain_risk_mean`、`filter_filtered_path_terrain_risk_mean` 和 `filter_path_terrain_risk_reduction_mean` 是否仍显示地形风险收益。

## 产物路径

```text
outputs/runs/exp021_randomized_terrain_filter_curriculum/pure_rl_seed23_20m_filter_curriculum/
outputs/runs/exp021_randomized_terrain_filter_curriculum/_suite/metrics/
outputs/runs/exp021_randomized_terrain_filter_curriculum/_launcher/train.log
```

训练完成后复验/GIF 计划写入：

```text
outputs/runs/exp021_randomized_terrain_filter_curriculum/pure_rl_seed23_20m_filter_curriculum/metrics/multi_eval_<timestamp>/
outputs/runs/exp021_randomized_terrain_filter_curriculum/pure_rl_seed23_20m_filter_curriculum/videos/multi_eval_<timestamp>/
outputs/runs/exp021_randomized_terrain_filter_curriculum/pure_rl_seed23_20m_filter_curriculum/figures/training_curves.png
outputs/runs/exp021_randomized_terrain_filter_curriculum/pure_rl_seed23_20m_filter_curriculum/figures/candidate_eval_curves.png
```

## 结论

待训练完成后填写。当前只能表述为“exp021 是针对 exp020 hard filter 过强问题的课程化验证实验”。

## 下一步

若 exp021 恢复 success 但 collision 仍高，下一轮优先做安全/集合联合 action representation 或 endpoint conflict-aware policy loss；若 exp021 仍 success 很低，说明 post-processing 类 filter 可能不适合作为主修复路径，应回到 action 表达、episode curriculum 或局部目标生成器结构。
