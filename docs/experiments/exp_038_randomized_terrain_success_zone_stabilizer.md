# exp038 success-zone stabilizer 随机地形实验

## 目的

exp037 说明单纯延长 episode 会降低 timeout，但会让末段 collision 反弹。exp038 引入 success-zone stabilizer：在真正接近 success gate 时强化 spacing buffer、降低线速度、延长到 320 steps，目标是在成功区内稳定保持最近邻安全间距。

## 配置

config: `configs/experiment/exp038_randomized_terrain_success_zone_stabilizer.yaml`

- pure RL，shared-joint MAPPO，seed23，2048 env，rollout 32，10240 timesteps。
- `episode_length_s=64.0`，eval steps `320`。
- filter mode: `terrain_safe_candidate_hold_progress_curriculum`。
- hold-zone 仅在 success gate 附近激活：
  - `hold_zone_dmax_multiplier=1.00`
  - `hold_zone_dispersion_multiplier=1.00`
  - `hold_zone_rho_weight=1.20`
  - `hold_zone_spacing_weight=5.00`
  - `hold_zone_pairwise_distance=0.54`
- control safety:
  - `projection_activation_distance=0.82`
  - `projection_stop_distance=0.46`
  - `projection_strength=0.95`
  - `projection_directional_agent_scale_mode=mask`
  - `success_zone_damping_enabled=true`
  - `success_zone_linear_scale=0.45`

## 严格标准

`dmax_reduction_ratio <= 0.20`、`success_rate >= 0.90`、`collision_rate <= 0.02`、`timeout_rate == 0`。

## 结果表

| run | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `pure_rl_seed23_20m_success_zone_stabilizer_timeout320` | `best.pt` = `ppo_timestep_009216.pt` | 0.1590 | 0.9756 | 0.0137 | 0.0107 | 未通过 |

说明：训练结束时 `eval_metrics.json` 的旧 `best_candidate` 仍指向 `ppo_timestep_010240.pt`；后续修正 strict ranking 后，`checkpoints/best.pt` 已手动重选为 `ppo_timestep_009216.pt` 并重新运行 final eval。以 `final_eval_proxy.json` 和 `strict_acceptance.json` 为准。

## 失败分析

exp038 是当前随机增强地形下的最佳综合结果：

- dmax、success、collision 均通过 strict。
- timeout `0.0107` 未通过 `timeout_rate == 0`。
- `max_success_hold_count_mean=7.8281/8`，说明绝大多数 episode 已接近完成 hold。
- timeout 子集的 final dmax / dispersion / speed 基本满足成功区要求，主要问题是最近邻安全间距：timeout episode 的 `final_nearest_neighbor_distance_mean≈0.400`，低于 `min_pairwise_distance=0.42`。

也就是说，当前失败已经不是“不会集合”，而是约 1% episode 卡在 `collision_distance=0.28` 与 `success min_pairwise=0.42` 之间的灰区，无法稳定计入 safe success。

## 产物路径

```text
outputs/runs/exp038_randomized_terrain_success_zone_stabilizer/pure_rl_seed23_20m_success_zone_stabilizer_timeout320/
outputs/runs/exp038_randomized_terrain_success_zone_stabilizer/_launcher/train.log
```

## 结论

不能写成 strict pass，但它是当前随机地形最佳候选。下一步应只针对末端 `0.28–0.42 m` 灰区做局部修正，避免破坏已经达到的 high success / low collision。

