# exp022 endpoint safety constrained curriculum filter 随机地形实验

## 目的

exp021 证明课程化 filter 可以恢复集合学习，但也把失败模式推回“会集合但碰撞高”：5 seed mean success `0.6361`，collision `0.1746`。exp022 的目标是只针对这一点迭代：在保留 exp021 课程节奏和集合意图的前提下，把 endpoint / path safety constraint 显式加入子目标后处理。

本轮仍是 proxy 训练，不引入 PhysX、真实车体尺寸、轮地接触、硬性陷车终止或 BC。

## 配置

```text
config: configs/experiment/exp022_randomized_terrain_endpoint_safety_filter.yaml
experiment_id: exp022_randomized_terrain_endpoint_safety_filter
run_id: pure_rl_seed23_20m_endpoint_safety
```

关键设置：

- shared-joint MAPPO pure RL，`bc_updates=0`。
- 随机增强 lunar crater proxy，`randomize_per_reset=true`。
- 通信半径 `12 m`，Actor / Critic 接口仍为 `86 / 54`。
- 2048 CUDA envs，rollout 32，连续 `10240` timesteps，即约 20.97M env steps。
- checkpoint interval `1024` timesteps。
- 候选选择使用 `safe_progress_long`，避免优先选择 success 很低但稍安全的 checkpoint。

constrained curriculum filter：

```text
mode: terrain_safe_candidate_constrained_curriculum
rho_scales: [0.45, 0.70, 0.90, 1.0]
beta_offsets_deg: [-45, -30, -15, 0, 15, 30, 45]
candidate_count: 28
warmup_timesteps: 2048
ramp_timesteps: 4096
apply_probability: 0.0 -> 0.75
score_scale: 0.15 -> 0.75
endpoint_safe_distance: 0.50
path_safe_distance: 0.42
hard_endpoint_near_filter: true
hard_path_collision_filter: true
hard_center_progress_filter: true
center_progress_slack: 0.35
safety_override_after_warmup: true
```

score 相比 exp021 增强的项：

```text
endpoint_near_weight: 8.0
endpoint_collision_weight: 2000.0
path_near_weight: 8.0
path_collision_weight: 2000.0
visible_neighbor_center_weight: 0.50
```

reward 安全项也轻微提高：

```text
safety.near_distance: 0.95
near_distance: 8.0
inter_agent_collision: 120.0
failure_penalty: 60.0
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

- collision 是否明显低于 exp021 的 `0.1746`。
- success 是否仍显著高于 exp020 的 `0.0`，避免再次退化为安全但不集合。
- timeout 是否不回到 exp020 的 `0.9506`。
- filter telemetry 中 raw endpoint/path violation 是否被 filtered action 降低。

## 工程验证

已完成：

- constrained curriculum warmup 期不覆盖 raw action。
- warmup 后 safety override 可在 raw endpoint unsafe 时强制选择可行候选。
- 不可见 rover 不影响单 rover safety constraint / filter 输出。
- exp022 配置 contract、候选数、约束开关、安全 reward 和 telemetry shape。
- 完整 `.venv_isaaclab/bin/python -m pytest -q -ra` 通过。
- CPU smoke 和 CUDA smoke 通过。
- CUDA smoke 确认 `optimizer_count=1`、`joint_update_count=2`、terrain 输入权重更新、动作非退化。

关键兼容修复：

- `scripts/render_skrl_proxy_rollout.py` 已支持 `terrain_safe_candidate_constrained_curriculum` 的 checkpoint timestep metadata；否则 exp022 GIF 会用错 filter schedule progress。

## 结果表

训练、候选评估、5 轮独立 eval、GIF、height map 和曲线已完成。结果以以下机器可读文件为准：

```text
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_suite/metrics/strict_acceptance.json
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_suite/metrics/suite_summary.json
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_suite/metrics/multi_eval_summary.json
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/pure_rl_seed23_20m_endpoint_safety/metrics/final_eval_proxy.json
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/pure_rl_seed23_20m_endpoint_safety/metrics/eval_metrics.json
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/pure_rl_seed23_20m_endpoint_safety/metrics/multi_eval_20260626_153606/multi_eval_summary.json
```

| eval | checkpoint | dmax ratio | success | collision | timeout | path risk mean | strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| candidate eval seed1023 | `ppo_timestep_007168.pt` | 0.4722 | 0.0137 | 0.0176 | 0.9697 | 0.2732 | 未通过 |
| final eval seed1023 | `best.pt` | 0.4768 | 0.0244 | 0.0146 | 0.9609 | 0.2726 | 未通过 |
| 5 seed mean `12023–12027` | `best.pt` | 0.4719 | 0.0139 | 0.0170 | 0.9699 | 0.2737 | 未通过 |

5 seed filter 统计：

```text
filter_raw_path_terrain_risk_mean: 0.3379
filter_filtered_path_terrain_risk_mean: 0.2737
filter_path_terrain_risk_reduction_mean: 0.0642
filter_applied_fraction: 0.6165
filter_safety_override_fraction: 0.0247
filter_feasible_fraction: 0.8380
filter_path_collision_violation_fraction: 0.00029
final_nearest_neighbor_distance: 1.3977
min_nearest_distance: 0.2567
```

## 失败分析

严格标准未通过：

- `dmax_reduction_ratio=0.4719`，未达到 `<=0.20`。
- `success_rate=0.0139`，远低于 `>=0.90`。
- `collision_rate=0.0170`，通过 `<=0.02`。
- `timeout_rate=0.9699`，远高于 `0`。

和 exp021 对比：

- collision 从 `0.1746` 降到 `0.0170`，说明 endpoint/path hard constraint 和 safety override 确实把碰撞压住了。
- success 从 `0.6361` 降到 `0.0139`，timeout 从 `0.1967` 升到 `0.9699`，说明约束过强，策略退化为“安全但不完成集合”。
- filtered path risk 从 exp021 的 `0.3638` 降到 `0.2737`，风险过滤非常强，但这并没有转化为可完成任务的集合行为。
- best checkpoint 被 `safe_progress_long` 选为 `7168` timesteps；后续 checkpoint collision 继续下降，但 success 也进一步接近 0。

当前判断：

exp022 解决了 exp021 的碰撞问题，但重新暴露 exp020 的核心失败模式：post-processing filter / shield 一旦足够强，就会牺牲集合进度。继续只堆 filter 权重或 hard constraint 不太可能得到 strict pass。

下一轮更应该把“保持集合进度”和“避免碰撞”放到同一个 action / planner 表达里，而不是把安全作为 actor 输出后的强替换。可考虑：

- 分离径向集合速度和切向避障速度，允许 actor 同时表达“朝中心收缩”和“绕开邻居”。
- 对 endpoint safety 使用软 barrier / CBF-like projection，并保留最小集合进度约束，而不是大范围替换为低风险候选。
- 降低 filter 的 endpoint safe distance 或把 success zone 设计为环带队形，避免所有车被约束推到过大的最近邻距离。

## 长训练命令

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

## 产物路径

```text
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/pure_rl_seed23_20m_endpoint_safety/
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_suite/metrics/
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_suite/figures/
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_launcher/train.log
```

复验/GIF、height map 和曲线：

```text
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/pure_rl_seed23_20m_endpoint_safety/metrics/multi_eval_20260626_153606/
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/pure_rl_seed23_20m_endpoint_safety/videos/multi_eval_20260626_153606/
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/pure_rl_seed23_20m_endpoint_safety/figures/training_curves.png
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/pure_rl_seed23_20m_endpoint_safety/figures/candidate_eval_curves.png
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_suite/figures/training_curves.png
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_suite/figures/candidate_eval_curves.png
```

每个 seed 的可视化：

```text
videos/multi_eval_20260626_153606/seed<seed>/proxy_eval_rollout.gif
videos/multi_eval_20260626_153606/seed<seed>/terrain_height_map.png
```

## 结论

exp022 不能作为当前主结果，也不能写成随机地形安全策略收敛。它是一个清晰诊断：安全 constrained filter 能把 collision 压进 strict gate，但过强地牺牲集合，导致 success / timeout 严重失败。下一轮不建议继续强化 post-processing filter，应改 action representation、planner projection 或 success geometry，让安全和集合成为同一个可优化目标。
