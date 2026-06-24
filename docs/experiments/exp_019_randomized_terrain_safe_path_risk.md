# exp019 安全成功门控与路径级地形风险

## 目的

exp018 在随机增强地形上已经学会集合，但 5 轮独立 eval 的 collision 稳定高于 2% strict gate，且 GIF 显示轨迹仍近似直穿地形。exp019 只针对两个问题做最小改造：

- success gate 显式要求集合成功时全队最近邻距离大于碰撞区；
- terrain reward 从脚下/落点扩展到当前点到子目标的直线路径风险。

## 配置

```text
configs/experiment/exp019_randomized_terrain_safe_path_risk.yaml
```

沿用 exp018 的 shared-joint MAPPO、pure RL、随机增强 lunar crater、`12 m` 通信半径、`2048` CUDA env、rollout `32` 和 20M 预算。关键差异：

- `success_thresholds.min_pairwise_distance: 0.42 m`。
- `safety.collision_distance: 0.28 m`，满足 `0.28 < 0.42 < dmax 1.25`。
- `safety.near_distance: 0.85 m`。
- `reward.coefficients.near_distance: 6.0`。
- `reward.coefficients.inter_agent_collision: 80.0`。
- `reward.coefficients.failure_penalty: 45.0`。
- 路径级地形项：
  - `path_terrain_mean_cost: 0.60`
  - `path_terrain_max_cost: 0.40`
  - `path_height_change_cost: 0.20`
- `reward.weights.terrain` 保持 `0.30`，不继续整体放大。

## 严格标准

最终 strict gate 保持标准 proxy gate：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

新增诊断指标：

```text
safe_success_rate
final_nearest_neighbor_distance
min_pairwise_ok_rate
path_terrain_risk_mean
path_terrain_risk_max
path_height_change_mean
```

## 验证记录

- success gate 单元测试覆盖：满足 dmax/dispersion/speed 但最近邻不足 `0.42 m` 时不能成功。
- 路径级地形风险单元测试覆盖：平地风险为零，穿 crater 的直线路径风险高于绕开路径。
- 完整 `pytest -q -ra` 已通过。
- CPU smoke `smoke_cpu_exp019` 已通过，final eval 正常写出新增安全和路径风险指标。
- CUDA smoke `smoke_cuda_exp019` 已通过：一个 optimizer、两次 joint update、无 NaN，policy 参数和 terrain 输入权重均更新，path-risk telemetry 非零。
- seed23 连续 20M 长训练已完成：`10240` timesteps / `20,971,520` env steps，10 个候选 checkpoint 均完成评估。

## 20M 结果

训练本身正常结束，无 NaN/Inf；`summary.json` 中记录：

```text
joint_update_count: 320
optimizer_count: 1
policy_parameter_delta_l2: 4.5316
terrain_input_weight_delta_l2: 1.2213
post_training_action_std: 0.5254
```

候选里程碑显示：越往后成功率提高，但 collision 也明显升高。当前 `pure_rl_long` 选择逻辑优先规避 collision，因此最终 `best.pt` 选中了 `ppo_timestep_001024.pt`，它较安全但几乎没有学会集合。

| checkpoint timestep | env steps | dmax ratio | success | collision | timeout | final nearest m | path risk mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 2,097,152 | 0.4293 | 0.0195 | 0.0771 | 0.9053 | 0.8104 | 0.3864 |
| 4096 | 8,388,608 | 0.1564 | 0.3965 | 0.3896 | 0.2178 | 0.4200 | 0.3964 |
| 6144 | 12,582,912 | 0.1351 | 0.5967 | 0.3223 | 0.0840 | 0.4335 | 0.4009 |
| 10240 | 20,971,520 | 0.1552 | 0.6201 | 0.1279 | 0.2627 | 0.5750 | 0.3945 |

最终 `best.pt` 独立 eval：

```text
dmax_reduction_ratio: 0.4174
success_rate: 0.0195
safe_success_rate: 0.0195
collision_rate: 0.0791
timeout_rate: 0.9023
final_nearest_neighbor_distance: 0.8073 m
min_pairwise_ok_rate: 0.9871
path_terrain_risk_mean: 0.3845
path_terrain_risk_max: 0.8647
```

strict gate 未通过，四项均失败：

```text
dmax_reduction_ratio <= 0.20: false
success_rate >= 0.90: false
collision_rate <= 0.02: false
timeout_rate == 0: false
```

## 5 轮独立 eval 与 GIF

对 `best.pt` 运行 seeds `12023–12027`，每轮 `1024 env × 220 steps`，并输出每个 seed 的 GIF 和 terrain height map。

5 seed 均值：

```text
dmax_reduction_ratio: 0.4186
success_rate: 0.0143
safe_success_rate: 0.0143
collision_rate: 0.0801
timeout_rate: 0.9082
final_nearest_neighbor_distance: 0.8031 m
min_pairwise_ok_rate: 0.9872
path_terrain_risk_mean: 0.3872
path_terrain_risk_max: 0.8653
path_height_change_mean: 0.0555
```

复验产物：

```text
outputs/runs/exp019_randomized_terrain_safe_path_risk/pure_rl_seed23_20m_safe_path_risk/metrics/multi_eval_20260624_115351/
outputs/runs/exp019_randomized_terrain_safe_path_risk/pure_rl_seed23_20m_safe_path_risk/videos/multi_eval_20260624_115351/
outputs/runs/exp019_randomized_terrain_safe_path_risk/_suite/metrics/
```

每个 seed 都有：

```text
videos/multi_eval_20260624_115351/seed<seed>/proxy_eval_rollout.gif
videos/multi_eval_20260624_115351/seed<seed>/terrain_height_map.png
```

## 产物路径

正式长跑目标路径：

```text
outputs/runs/exp019_randomized_terrain_safe_path_risk/pure_rl_seed23_20m_safe_path_risk/
outputs/runs/exp019_randomized_terrain_safe_path_risk/_launcher/train.log
```

smoke 产物：

```text
outputs/runs/exp019_randomized_terrain_safe_path_risk/smoke_cpu_exp019/
outputs/runs/exp019_randomized_terrain_safe_path_risk/smoke_cuda_exp019/
```

## 结论

exp019 工程目标已完成：成功区安全间距、路径级地形风险、训练 telemetry、候选评估、5 seed 复验和 GIF 输出均已打通。但策略没有收敛到可接受行为，不能写成 strict pass。

本轮最重要的诊断结论是：

- path risk 已经进入 reward 和 telemetry，但只靠软惩罚没有让 policy 明确绕开高风险路径；5 seed path risk mean 稳定在约 `0.39`。
- 训练后期可以获得集合趋势，但 collision 显著超标；选择较低 collision 的早期 checkpoint 又会退化为大 timeout、低 success。
- 新增 success 最近邻 gate 没有破坏兼容性，但当前 reward / termination 组合没有提供足够干净的安全收敛信号。

## 下一步

下一轮不建议继续简单放大 terrain weight。更值得尝试的是把地形风险从“结果惩罚”前移为动作可达性约束，例如对子目标候选做 path-risk filtering / local planner score，或者把高风险路径作为截断/不可达区域单独处理；同时需要重新协调 success 半径、安全半径、episode 长度和 collision penalty，避免再次出现“会集合但撞车”和“安全但超时”的两极化。
