# exp027 strict hold-zone filter 随机地形实验

## 目的

exp026 的 hold-zone filter 过早介入，导致 success/timeout 明显退化。exp027 保留 exp025 的 dense mutual filter 主体，只把 hold-zone activation 收窄到真正 success dmax/dispersion 附近，验证“严格晚介入”是否能稳定 8-step hold。

## 配置

config: `configs/experiment/exp027_randomized_terrain_strict_hold_filter.yaml`

- pure RL，shared-joint MAPPO，seed23，2048 env，rollout 32，10240 timesteps。
- `rho_scales=[0.60, 0.80, 1.0, 1.08]`，`path_samples=9`。
- `hold_zone_dmax_multiplier=1.00`，`hold_zone_dispersion_multiplier=1.00`。
- `hold_zone_rho_weight=0.35`，`hold_zone_spacing_weight=1.40`，`hold_zone_pairwise_distance=0.44`。

## 严格标准

`dmax_reduction_ratio <= 0.20`、`success_rate >= 0.90`、`collision_rate <= 0.02`、`timeout_rate == 0`。

## 结果表

| run | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `pure_rl_seed23_20m_strict_hold_filter` | `best.pt` | 0.1464 | 0.8418 | 0.0498 | 0.1123 | 未通过 |

## 失败分析

严格 hold-zone 触发避免了 exp026 的明显退化，但没有改善 exp025 的末段稳定问题。`max_success_hold_count_mean=7.168/8`，仍卡在接近成功但未稳定 hold；collision 仍发生在后段，`first_collision_step_mean≈168.7/220`。

## 产物路径

```text
outputs/runs/exp027_randomized_terrain_strict_hold_filter/pure_rl_seed23_20m_strict_hold_filter/
```

## 结论

不能作为当前主结果。单独收窄 hold-zone filter 不足以解决随机地形下的末段碰撞和 timeout。

## 下一步

回到 exp025/exp027 中共同暴露的 hold reward 信号，尝试强化 success hold 的稠密奖励。
