# exp028 hold reward 随机地形实验

## 目的

exp025/exp027 都接近 8-step hold 但未稳定通过。exp028 回退到 exp025 的 dense mutual filter 主体，不再加入 hold-zone filter，只强化 success hold 奖励，验证 policy 是否能自己学会“进圈后别乱动”。

## 配置

config: `configs/experiment/exp028_randomized_terrain_hold_reward.yaml`

- pure RL，shared-joint MAPPO，seed23，2048 env，rollout 32，10240 timesteps。
- dense mutual filter 与 exp025 相同。
- `success_hold_step=4.0`，`success_bonus=45.0`，`timeout_penalty=18.0`。
- collision / near 安全系数保持 exp025 主体。

## 严格标准

`dmax_reduction_ratio <= 0.20`、`success_rate >= 0.90`、`collision_rate <= 0.02`、`timeout_rate == 0`。

## 结果表

| run | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `pure_rl_seed23_20m_hold_reward` | `best.pt` | 0.1415 | 0.8691 | 0.0469 | 0.0889 | 未通过 |

关键诊断：

```text
max_success_hold_count_mean: 7.3633 / 8
final_success_hold_count_mean: 7.0273 / 8
first_collision_step_mean: 179.5 / 220
```

## 失败分析

强化 hold reward 是有效方向：相对 exp025，success 从 `0.8525` 提高到 `0.8691`，timeout 从 `0.1035` 降到 `0.0889`。但 collision 仍为 `0.0469`，远高于 strict `0.02`。

一次 post-hoc 将 deterministic filter margin 从 `0.025` 降为 `0.0` 并未改善：collision 升到 `0.0576`。说明问题不是 eval filter 过保守，而是真实执行期末段碰撞仍未被当前 filter/reward 捕获。

## 产物路径

```text
outputs/runs/exp028_randomized_terrain_hold_reward/pure_rl_seed23_20m_hold_reward/
outputs/runs/exp028_randomized_terrain_hold_reward/pure_rl_seed23_20m_hold_reward/metrics/posthoc_margin0_eval_seed1023.json
```

## 结论

exp028 是 exp026–029 中最好的随机地形结果，但仍不是 strict pass。它证明 hold reward 有用，也证明仅靠现有 filter margin 调整不够。

## 下一步

不要继续只增强 hold；需要处理末段真实碰撞，优先考虑速度/相对速度感知的 collision anticipation 或低层 planner projection。
