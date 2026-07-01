# exp037 directional mask timeout260 随机地形实验

## 目的

exp036 剩余主要失败项是 timeout。exp037 在保持 exp036 主体的基础上，把 episode/eval 从 220 steps 延长到 260 steps，验证 timeout 是否只是时间预算不足。

## 配置

config: `configs/experiment/exp037_randomized_terrain_directional_mask_timeout260.yaml`

- pure RL，shared-joint MAPPO，seed23，2048 env，rollout 32，10240 timesteps。
- 仍使用 directional mask buffer、stronger hold/timeout shaping。
- `episode_length_s=52.0`，对应 260 control steps。
- eval steps 使用 260。

## 严格标准

`dmax_reduction_ratio <= 0.20`、`success_rate >= 0.90`、`collision_rate <= 0.02`、`timeout_rate == 0`。

## 结果表

| run | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `pure_rl_seed23_20m_directional_mask_timeout260` | `ppo_timestep_008192.pt` | 0.1517 | 0.9238 | 0.0352 | 0.0410 | 未通过 |

## 失败分析

延长 episode 确实降低 timeout：从 exp036 的 `0.0586` 降到 `0.0410`。但 collision 从 `0.0088` 反弹到 `0.0352`，超过 strict。

这说明 timeout 不只是时间不足。更长 episode 让更多 episode 有机会进入末段高风险区，如果成功区稳定器不足，collision 会回升。

## 产物路径

```text
outputs/runs/exp037_randomized_terrain_directional_mask_timeout260/pure_rl_seed23_20m_directional_mask_timeout260/
outputs/runs/exp037_randomized_terrain_directional_mask_timeout260/_launcher/train.log
```

## 结论

不能作为当前主结果。下一轮应对成功区末段几何做稳定，而不是单纯继续延长 episode。

