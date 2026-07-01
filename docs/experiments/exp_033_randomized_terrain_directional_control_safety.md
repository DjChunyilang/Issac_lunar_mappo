# exp033 directional control safety 随机地形实验

## 目的

exp032 说明 closing-only 投影比简单调弱更合理，但仍未解决 collision/timeout。exp033 在低层控制安全投影中加入方向性 agent-scale，目标是只压低会加剧 pairwise 风险的运动分量，减少对集合进度的副作用。

## 配置

config: `configs/experiment/exp033_randomized_terrain_directional_control_safety.yaml`

- pure RL，shared-joint MAPPO，seed23，2048 env，rollout 32，10240 timesteps。
- 随机增强 lunar crater proxy、`12 m` 通信半径、Actor/Critic `86/54`。
- 延续 exp032 的 closing-only 投影，不使用 success-zone damping。
- 新增 `projection_directional_agent_scale=true`。

## 严格标准

`dmax_reduction_ratio <= 0.20`、`success_rate >= 0.90`、`collision_rate <= 0.02`、`timeout_rate == 0`。

## 结果表

| run | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `pure_rl_seed23_20m_directional_control_safety` | `ppo_timestep_009216.pt` | 0.1436 | 0.8154 | 0.0488 | 0.1387 | 未通过 |

## 失败分析

方向性缩放没有带来预期收益：success 低于 exp032，collision 高于 exp032，timeout 也没有改善。说明当前方向性 scale 仍然过粗，或者没有准确区分“安全退让”和“阻碍集合”的控制分量。

## 产物路径

```text
outputs/runs/exp033_randomized_terrain_directional_control_safety/pure_rl_seed23_20m_directional_control_safety/
outputs/runs/exp033_randomized_terrain_directional_control_safety/_launcher/train.log
```

## 结论

不能作为当前主结果。下一步应把方向性缩放改成更硬的 directional mask，只屏蔽风险方向，而不是连续缩放所有相关 agent。

