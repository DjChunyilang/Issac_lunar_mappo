# exp031 narrow control safety 随机地形实验

## 目的

exp030 的低层 control safety projection 能降低 collision，但投影触发过强，导致 success/timeout 退化。exp031 保留 exp028 的 reward/filter 主体和 exp030 的 telemetry，只把控制投影改成窄触发、弱投影，并关闭 success-zone damping。

目标是保留 exp030 的降碰撞收益，同时把 success/timeout 拉回 exp028 附近。

## 配置

config: `configs/experiment/exp031_randomized_terrain_narrow_control_safety.yaml`

- pure RL，shared-joint MAPPO，seed23，2048 env，rollout 32，10240 timesteps。
- 地形、reward、dense mutual subgoal filter 与 exp028 / exp030 保持一致。
- 不使用 BC，不续训旧 checkpoint。
- 相比 exp030 的控制安全变化：
  - `projection_activation_distance: 0.62 -> 0.52`
  - `projection_stop_distance: 0.36 -> 0.34`
  - `projection_horizon_s: 0.60 -> 0.45`
  - `projection_strength: 0.80 -> 0.55`
  - `projection_min_linear_scale: 0.25 -> 0.45`
  - `success_zone_damping_enabled: true -> false`

## 严格标准

`dmax_reduction_ratio <= 0.20`、`success_rate >= 0.90`、`collision_rate <= 0.02`、`timeout_rate == 0`。

## 结果表

| run | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `pure_rl_seed23_20m_narrow_control_safety` | `ppo_timestep_009216.pt` | 0.1448 | 0.8105 | 0.0449 | 0.1455 | 未通过 |

## 失败分析

exp031 的“简单调弱”没有恢复 exp028 的 success/timeout，也丢失了 exp030 的安全收益：

- control safety 触发率从 exp030 的 `0.1610` 降到 `0.1052`，`linear_scale_mean` 从 `0.9304` 提高到 `0.9690`，说明弱化确实生效。
- collision 从 exp030 的 `0.0313` 回升到 `0.0449`，接近 exp028 的 `0.0469`。
- success 从 exp030 的 `0.8330` 进一步降到 `0.8105`，timeout 从 `0.1357` 升到 `0.1455`。
- `max_success_hold_count_mean=6.9023/8`，仍低于 exp028 的 `7.3633/8`。

结论：问题不只是投影强度过大，而是投影条件本身过粗。当前实现会对 `distance < activation` 的非相向/非 closing pair 也降速，容易在末段造成不必要的迟滞。

## 产物路径

```text
outputs/runs/exp031_randomized_terrain_narrow_control_safety/pure_rl_seed23_20m_narrow_control_safety/
outputs/runs/exp031_randomized_terrain_narrow_control_safety/_launcher/train.log
```

## 结论

不能作为当前主结果。它说明单纯缩小/减弱 projection 参数不够。

## 下一步

下一轮应改控制投影逻辑：只对正在 closing 的 pair 或已经低于 stop distance 的 pair 触发；对虽然近但正在分开的 pair 不再降速。为保持 exp030/031 可复现，该行为应使用新配置开关启用。
