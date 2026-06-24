# exp016 共享 MAPPO、通信半径 12 m 与安全课程

## 目的

修正 exp015 中 shared Actor/Critic 被四个独立 optimizer 顺序更新的问题，并用 `12 m` 临时通信半径和可观测 local teacher 验证 BC 知识转移。只有 shared-update probe 和 BC-only probe 通过，才允许进入 2M screen 与 8M formal。

## 配置

```text
configs/experiment/exp016_shared_mappo_comm12.yaml
```

关键设置：

- `algorithm.update_mode: shared_joint`：一个共享 Actor、一个共享 Critic、一个 Adam optimizer。
- Actor 样本合并四个 rover memory；Critic 每个 rollout 只使用一份 centralized team state。
- communication radius 临时设为 `12 m`，observation schema 和维度仍为 `ego_v3_local_terrain_grid / 86 / 54`。
- local teacher 只使用可见邻居，`stop_radius=0.54 m`、`max_rho=0.8 m`，BC100。
- `near_distance=0.75 m`、`success_hold_step=1.0`、`timeout_penalty=15.0`。
- 正式 MAPPO 使用 rollout 64、entropy `0.002 → 0.0005`。

## 晋级标准

- shared-update probe：一个 optimizer、两次 joint update、两次 critic update、参数与 terrain 权重更新、动作非退化。
- BC-only：dmax ratio ≤ `0.40`、success ≥ `0.50`、collision ≤ `0.03`、timeout ≤ `0.50`。
- 2M screen：dmax ratio ≤ `0.30`、success ≥ `0.50`、collision ≤ `0.03`、timeout ≤ `0.50`。
- 8M formal：dmax ratio ≤ `0.20`、success ≥ `0.90`、collision ≤ `0.02`、timeout = `0`。

## 结果表

| stage | run_id | 状态 | 关键结果 |
| --- | --- | --- | --- |
| shared update | `shared_update_probe_seed23_512k` | 通过 | optimizer `1`；joint/critic updates `2/2`；Actor/Critic samples `1,048,576 / 262,144`。 |
| BC-only | `local_teacher_bc100_seed23` | 未通过 | dmax ratio `0.438`、success `0`、collision `0.0088`、timeout `0.991`。 |
| 2M screen | `screen_seed23_2m` | 未启动 | BC-only 未达到晋级门槛。 |
| 8M formal | `formal_seed23_8m` | 未启动 | screen 未运行。 |

## 失败分析

共享更新修正已经通过 CUDA 验证：只有一个 optimizer，每个 rollout 只执行一次联合更新，Critic 没有再被四倍重复训练；参数、terrain 输入权重和动作分布均正常更新。

BC100 的 loss 从 `0.156` 降至 `0.0396`，collision 从 exp015 的高碰撞状态降至 `0.0088`，说明通信半径扩展、朝向约束和安全状态过滤有效。但 BC-only 最终 `dmax=3.194 m`、`dispersion=2.280`，几乎全部 episode timeout，策略仍未进入成功区。当前瓶颈已经从“过激接近和碰撞”转为“teacher/蒸馏过于保守，集合速度不足”。

## 产物路径

```text
outputs/runs/exp016_shared_mappo_comm12/shared_update_probe_seed23_512k/
outputs/runs/exp016_shared_mappo_comm12/local_teacher_bc100_seed23/
outputs/runs/exp016_shared_mappo_comm12/_suite/metrics/
```

## 结论

exp016 完成了 shared-joint MAPPO 的工程修正，但 BC-only probe 未通过，因此按设计停止，没有运行 2M/8M。本实验不能作为新 schema 收敛结果。

## 下一步

保持 shared-joint MAPPO 和 `12 m` 通信半径不变，单独重新设计 teacher 速度课程：减少候选落点 traversability 对 rho 的重复缩放，或在前半程提高 teacher `max_rho`，同时保留末段 stop radius 与安全过滤。先让 teacher/BC-only 达到晋级门槛，再重新进入 MAPPO screen。
