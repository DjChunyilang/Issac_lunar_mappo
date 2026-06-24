# exp017 Shared MAPPO Pure RL 连续长跑

## 目的

在不使用 BC teacher 的条件下，验证修正后的 shared-joint MAPPO 和 86 维局部地形网格能否通过纯强化学习形成有效集合策略。单次训练连续运行到约 20M environment steps，并保留 2M、8M、20M 里程碑。

## 配置

```text
configs/experiment/exp017_shared_mappo_pure_rl_comm12.yaml
```

关键设置：

- seed `23`，2048 个 CUDA environments，episode 220 steps。
- `shared_joint`：一个共享 Actor、一个共享 Critic和一个 optimizer。
- `pure_rl`、`bc_updates=0`，checkpoint 中 teacher metadata 为 `null`。
- communication radius 暂时保持 `12 m`，地形、reward 和安全参数沿用 exp016。
- rollout `32`、PPO epochs `4`、mini-batches `16`、LR `1.2e-4`。
- entropy 在前 4096 timesteps 从 `0.002` 降至 `0.0005`，之后保持不变。
- 总预算 10240 timesteps，即 20,971,520 environment steps。
- 每 1024 timesteps 保存并评估 checkpoint；2M、8M、20M 对应 timestep `1024/4096/10240`。

## 严格标准

正式 strict gate 保持：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

长跑候选选择使用诊断趋势标准：dmax ratio ≤ `0.45`、success ≥ `0.05`、collision ≤ `0.03`。timeout 继续记录，但不参与候选通过判定和排序，且不会提前终止训练。

## 结果表

| seed | run_id | 预算 | 状态 | 结果 |
| --- | --- | ---: | --- | --- |
| 23 | `pure_rl_seed23_20m_medium_soft_comm12` | 20.97M env steps | single-seed strict 通过 | 独立 final eval：dmax ratio `0.1318`、success `0.9990`、collision `0.00098`、timeout `0`。 |

## 训练趋势

| milestone | dmax ratio | success | collision | timeout |
| --- | ---: | ---: | ---: | ---: |
| 2M | 0.2030 | 0.2158 | 0.4766 | 0.3174 |
| 8M | 0.1471 | 0.9668 | 0.0332 | 0.00098 |
| 20M | 0.1315 | 0.9961 | 0.00488 | 0 |

训练前段主要问题是碰撞；8M 后集合与超时指标已经明显改善，20M milestone 通过 strict gate。最终候选不是最后一个 checkpoint，而是 timestep `9216`，并使用独立 seed `11023` 完成 final eval。

## 产物路径

```text
outputs/runs/exp017_shared_mappo_pure_rl_comm12/pure_rl_seed23_20m_medium_soft_comm12/
outputs/runs/exp017_shared_mappo_pure_rl_comm12/_suite/metrics/milestones.json
outputs/runs/exp017_shared_mappo_pure_rl_comm12/_suite/metrics/suite_summary.json
outputs/runs/exp017_shared_mappo_pure_rl_comm12/_suite/metrics/strict_acceptance.json
outputs/runs/exp017_shared_mappo_pure_rl_comm12/_launcher/train.log
```

## 结论

exp017 证明修正后的 shared-joint MAPPO 可以在固定偏弱中档 proxy 地图上从随机初始化 pure RL 达到单 seed strict gate。该结果仍不能外推为多 seed 收敛或随机地图泛化，也不是 PhysX 物理训练结果。

## 下一步

以 exp017 作为固定地图 pure RL baseline，使用 exp018 检查更强、按 episode 重采样地形下的泛化和轨迹地形敏感性；在随机地图结果明确前暂不把 exp017 写成多 seed 正式结论。
