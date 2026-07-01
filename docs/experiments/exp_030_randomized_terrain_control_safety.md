# exp030 control safety 随机地形实验

## 目的

exp028 是当前随机增强地形下最好的 candidate，但仍在末段出现 collision / timeout。exp029 说明继续加大静态 safety reward 或 filter 权重会牺牲 success，且不能压低真实执行碰撞。

exp030 回到 exp028 配置，只新增默认关闭、该实验启用的低层控制安全投影：在 `compute_control()` 后、proxy `_integrate()` 前，根据当前相对位置和一小段预测相对速度缩放线速度，并在成功区附近做 velocity damping。

## 配置

config: `configs/experiment/exp030_randomized_terrain_control_safety.yaml`

- pure RL，shared-joint MAPPO，seed23，2048 env，rollout 32，10240 timesteps。
- 地形、reward、dense mutual subgoal filter 与 exp028 保持一致。
- 不使用 BC，不续训旧 checkpoint。
- 新增低层控制安全：
  - `safety_projection_enabled=true`
  - `projection_activation_distance=0.62`
  - `projection_stop_distance=0.36`
  - `projection_horizon_s=0.60`
  - `projection_strength=0.80`
  - `projection_min_linear_scale=0.25`
  - `success_zone_damping_enabled=true`
  - `success_zone_linear_scale=0.65`

## 严格标准

`dmax_reduction_ratio <= 0.20`、`success_rate >= 0.90`、`collision_rate <= 0.02`、`timeout_rate == 0`。

## 结果表

| run | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `pure_rl_seed23_20m_control_safety` | `ppo_timestep_010240.pt` | 0.1528 | 0.8330 | 0.0313 | 0.1357 | 未通过 |

## 失败分析

exp030 验证了低层控制投影的方向部分有效，但当前参数过强：

- 相对 exp028，collision 从 `0.0469` 降到 `0.0313`，最优候选 eval 的 `ppo_timestep_010240.pt` collision 为 `0.0234`，已经接近 strict `0.02`。
- success 从 exp028 的 `0.8691` 降到 `0.8330`，timeout 从 `0.0889` 升到 `0.1357`，说明控制投影和 success-zone damping 拖慢了末段集合/hold。
- final eval 中 `control_safety_applied_fraction=0.1610`、`control_safety_linear_scale_mean=0.9304`、`control_safety_linear_scale_min=0.25`，投影触发足够多，且经常打到最低缩放。
- `control_safety_success_zone_fraction=0.0372`，success-zone damping 触发不算高，主要影响来自 pairwise projection。
- `max_success_hold_count_mean=6.9023/8`，低于 exp028 的 `7.3633/8`，说明速度投影确实削弱了 hold 完成度。

## 产物路径

```text
outputs/runs/exp030_randomized_terrain_control_safety/pure_rl_seed23_20m_control_safety/
outputs/runs/exp030_randomized_terrain_control_safety/_launcher/train.log
```

## 结论

不能作为当前主结果。它证明“低层动态执行约束”比 exp029 的静态 safety 加权更接近目标：collision 明显下降；但当前投影过早/过强，牺牲了 success 和 timeout。

## 下一步

下一轮应保留 exp028 主体和 control safety telemetry，但收窄投影触发：

- 降低触发范围，例如 `projection_activation_distance 0.62 -> 0.52`。
- 提高最低速度缩放，例如 `projection_min_linear_scale 0.25 -> 0.45`。
- 降低投影强度，例如 `projection_strength 0.80 -> 0.55`。
- 关闭或进一步收窄 success-zone damping，避免进入集合区后过早拖慢。
