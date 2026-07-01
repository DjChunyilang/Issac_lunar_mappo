# exp035 directional mask buffer 随机地形实验

## 目的

exp034 的 directional mask 恢复 success/timeout，但 collision 仍高。exp035 在保留 mask 的基础上扩大控制安全 buffer，验证是否能把 collision 压到 strict，同时不显著牺牲集合。

## 配置

config: `configs/experiment/exp035_randomized_terrain_directional_mask_buffer.yaml`

- pure RL，shared-joint MAPPO，seed23，2048 env，rollout 32，10240 timesteps。
- `projection_directional_agent_scale_mode=mask`。
- 相比 exp034：
  - `projection_activation_distance: 0.62 -> 0.68`
  - `projection_stop_distance: 0.36 -> 0.40`
  - `projection_strength: 0.80 -> 0.85`
  - `success_hold_step: 4.0`

## 严格标准

`dmax_reduction_ratio <= 0.20`、`success_rate >= 0.90`、`collision_rate <= 0.02`、`timeout_rate == 0`。

## 结果表

| run | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `pure_rl_seed23_20m_directional_mask_buffer` | `ppo_timestep_010240.pt` | 0.1519 | 0.9072 | 0.0127 | 0.0811 | 未通过 |

## 失败分析

exp035 首次让 success 和 collision 同时达标：success `0.9072`，collision `0.0127`。strict 失败只剩 timeout `0.0811`。

这说明扩大 directional mask buffer 是有效安全方向；剩余问题转为末段 hold / timeout，不再是主碰撞问题。

## 产物路径

```text
outputs/runs/exp035_randomized_terrain_directional_mask_buffer/pure_rl_seed23_20m_directional_mask_buffer/
outputs/runs/exp035_randomized_terrain_directional_mask_buffer/_launcher/train.log
```

## 结论

不能作为 strict 结果，但它是 exp030 之后第一个把 success 和 collision 同时推过门槛的关键拐点。

