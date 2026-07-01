# exp034 directional mask control safety 随机地形实验

## 目的

exp033 的方向性连续缩放没有改善安全/成功平衡。exp034 把方向性 agent-scale 改为 `mask` 模式，目标是更明确地抑制会缩短 pairwise distance 的风险方向，同时保留非风险方向的集合运动。

## 配置

config: `configs/experiment/exp034_randomized_terrain_directional_mask_control_safety.yaml`

- pure RL，shared-joint MAPPO，seed23，2048 env，rollout 32，10240 timesteps。
- 延续 exp032/033 的随机增强地形和 mutual path filter。
- `projection_directional_agent_scale=true`
- `projection_directional_agent_scale_mode=mask`
- `projection_activation_distance=0.62`、`projection_stop_distance=0.36`、`projection_strength=0.80`。

## 严格标准

`dmax_reduction_ratio <= 0.20`、`success_rate >= 0.90`、`collision_rate <= 0.02`、`timeout_rate == 0`。

## 结果表

| run | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `pure_rl_seed23_20m_directional_mask_control_safety` | `ppo_timestep_006144.pt` | 0.1491 | 0.8828 | 0.0361 | 0.0840 | 未通过 |

## 失败分析

mask 版本恢复了部分集合进度：success 从 exp033 的 `0.8154` 升到 `0.8828`，timeout 从 `0.1387` 降到 `0.0840`。但 collision 仍为 `0.0361`，高于 strict `0.02`。

候选曲线显示后期 collision 可下降，但 timeout 会升高；这提示安全 buffer/episode 长度/hold 奖励仍需联动，而不是只改 mask。

## 产物路径

```text
outputs/runs/exp034_randomized_terrain_directional_mask_control_safety/pure_rl_seed23_20m_directional_mask_control_safety/
outputs/runs/exp034_randomized_terrain_directional_mask_control_safety/_launcher/train.log
```

## 结论

不能作为当前主结果，但 directional mask 是 exp030–034 中更有希望的控制层方向。

