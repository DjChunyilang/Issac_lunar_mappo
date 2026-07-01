# exp032 closing-only control safety 随机地形实验

## 目的

exp030 证明低层动态安全投影能降低 collision，但过强导致 success/timeout 退化。exp031 简单调弱后既没有恢复 success，也丢失了降碰撞收益。

exp032 修正控制投影条件：默认行为仍保持 exp030/031 可复现；本实验设置 `projection_damp_nonclosing_near=false`，只对正在 closing 的 pair 或已经低于 stop distance 的 pair 触发投影。目标是避免对“近但正在分开”的 rover 无谓降速。

## 配置

config: `configs/experiment/exp032_randomized_terrain_closing_control_safety.yaml`

- pure RL，shared-joint MAPPO，seed23，2048 env，rollout 32，10240 timesteps。
- 地形、reward、dense mutual subgoal filter 与 exp028 / exp030 保持一致。
- 不使用 BC，不续训旧 checkpoint。
- 相比 exp030：
  - `projection_damp_nonclosing_near=false`
  - `success_zone_damping_enabled=false`
  - 其他 projection 范围和强度回到 exp030。

## 严格标准

`dmax_reduction_ratio <= 0.20`、`success_rate >= 0.90`、`collision_rate <= 0.02`、`timeout_rate == 0`。

## 结果表

| run | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `pure_rl_seed23_20m_closing_control_safety` | `ppo_timestep_008192.pt` | 0.1495 | 0.8379 | 0.0361 | 0.1279 | 未通过 |

## 失败分析

exp032 相对 exp031 有小幅改善，但仍没有回到 exp028 的综合水平：

- success 从 exp031 的 `0.8105` 回升到 `0.8379`，timeout 从 `0.1455` 降到 `0.1279`。
- collision 从 exp031 的 `0.0449` 降到 `0.0361`，但仍高于 strict `0.02`。
- `max_success_hold_count_mean=6.9346/8`，说明末段 hold 仍不稳定。

结论：closing-only 投影条件比简单调弱更合理，但还不足以同时压低碰撞和维持集合成功。

## 产物路径

```text
outputs/runs/exp032_randomized_terrain_closing_control_safety/pure_rl_seed23_20m_closing_control_safety/
outputs/runs/exp032_randomized_terrain_closing_control_safety/_launcher/train.log
```

## 结论

不能作为当前主结果。它证明“只看 closing”是必要但不充分的修正。

## 下一步

下一轮应继续沿控制层方向，但要让 projection 只影响真正会造成碰撞的 agent/方向，而不是整车线速度统一缩放。
