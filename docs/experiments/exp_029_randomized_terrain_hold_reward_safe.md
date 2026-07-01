# exp029 hold reward + stronger safety 随机地形实验

## 目的

exp028 提高了 success/hold，但 collision 仍高。exp029 在 exp028 基础上轻微加强 path/mutual collision filter 权重和终端碰撞惩罚，验证是否能降低真实碰撞而不破坏成功率。

## 配置

config: `configs/experiment/exp029_randomized_terrain_hold_reward_safe.yaml`

- pure RL，shared-joint MAPPO，seed23，2048 env，rollout 32，10240 timesteps。
- `success_hold_step=4.0` 保持 exp028。
- `endpoint_collision_weight=650`、`path_collision_weight=520`、`mutual_path_collision_weight=1450`。
- `inter_agent_collision=120`，`failure_penalty=60`。

## 严格标准

`dmax_reduction_ratio <= 0.20`、`success_rate >= 0.90`、`collision_rate <= 0.02`、`timeout_rate == 0`。

## 结果表

| run | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `pure_rl_seed23_20m_hold_reward_safe` | `best.pt` | 0.1439 | 0.8262 | 0.0557 | 0.1221 | 未通过 |

## 失败分析

加强 safety 权重没有降低真实碰撞，反而让 success 和 timeout 都退化。filter telemetry 仍显示 raw mutual path collision violation 被大幅压低，但 rollout collision 仍在后段出现，说明当前候选评分对执行期动态挤压的建模仍不足。

## 产物路径

```text
outputs/runs/exp029_randomized_terrain_hold_reward_safe/pure_rl_seed23_20m_hold_reward_safe/
```

## 结论

不能作为当前主结果。exp029 说明“继续加 safety reward/filter 权重”不是下一步主方向。

## 下一步

回到 exp028 作为当前最佳 candidate，下一轮应引入速度/相对速度或低层执行轨迹约束，而不是继续堆静态 path/mutual collision 权重。
