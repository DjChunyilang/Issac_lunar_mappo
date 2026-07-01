# exp036 directional mask timeout/hold 随机地形实验

## 目的

exp035 的主要失败项变成 timeout。exp036 保留 directional mask buffer，并强化 hold/timeout shaping，验证能否减少“已接近成功区但没有完成 hold”的尾部 episode。

## 配置

config: `configs/experiment/exp036_randomized_terrain_directional_mask_timeout_hold.yaml`

- pure RL，shared-joint MAPPO，seed23，2048 env，rollout 32，10240 timesteps。
- 保持 exp035 的 directional mask projection 参数。
- `success_hold_step=6.0`，提高 hold 奖励和 timeout 惩罚。

## 严格标准

`dmax_reduction_ratio <= 0.20`、`success_rate >= 0.90`、`collision_rate <= 0.02`、`timeout_rate == 0`。

## 结果表

| run | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `pure_rl_seed23_20m_directional_mask_timeout_hold` | `ppo_timestep_008192.pt` | 0.1523 | 0.9336 | 0.0088 | 0.0586 | 未通过 |

## 失败分析

相对 exp035，success 从 `0.9072` 升到 `0.9336`，collision 从 `0.0127` 降到 `0.0088`，timeout 从 `0.0811` 降到 `0.0586`。方向正确，但 timeout 仍不为 0。

候选曲线显示 collision 与 timeout 存在 trade-off：后期 checkpoint collision 更低，但 timeout 没有彻底消失。

## 产物路径

```text
outputs/runs/exp036_randomized_terrain_directional_mask_timeout_hold/pure_rl_seed23_20m_directional_mask_timeout_hold/
outputs/runs/exp036_randomized_terrain_directional_mask_timeout_hold/_launcher/train.log
```

## 结论

不能作为 strict 结果，但它是 exp035 之后更强的随机地形候选。下一步应区分 timeout 是否来自 episode 长度不足，还是成功区几何/hold gate 抖动。

