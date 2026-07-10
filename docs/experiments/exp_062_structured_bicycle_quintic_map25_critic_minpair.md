# exp062 structured bicycle quintic map25 critic minpair

## 目的

exp061 说明 terminal gate 特征直接进入 Actor 会扰动策略。exp062 改成更克制的 critic-only 诊断：保持 exp051 的 Actor observation、`branched_v1`、action 输出、reward、filter 和 control 不变，只让 centralized critic state 显式包含 terminal `min_pairwise`，检查 value 侧是否能更好识别尾部 timeout。

## 配置

```text
configs/experiment/exp062_structured_bicycle_quintic_map25_critic_minpair.yaml
```

相对 exp051 的变量：

- `state.include_terminal_min_pairwise: true`
- Critic state dim `54 -> 55`
- `algorithm.critic_architecture: structured_v1 -> structured_v2`

保持不变：

- Actor observation 仍为 `ego_v3_local_terrain_grid`，输入维度仍为 `86`
- Actor architecture 仍为 `branched_v1`
- Actor 输出仍是单点 `[rho, beta]`
- 不新增多点采样
- reward、filter、control safety、reset、terrain、trajectory、PPO 超参全部保持 exp051

## 严格标准

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

## 结果表

| seed | run_id | checkpoint | final_eval | strict |
| --- | --- | --- | --- | --- |
| 23 | `smoke_seed23_128_cpu_critic_minpair` | smoke only | CPU 工程通过；`actor_obs_dim=86`、`critic_state_dim=55`、`branched_v1/structured_v2`、一个 optimizer、8 次 joint update | 非收敛验证 |
| 23 | `smoke_seed23_128_cuda_critic_minpair` | smoke only | CUDA 工程通过；terrain branch 更新 `0.1596`，action std `0.0654` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_critic_minpair` | `ppo_timestep_016384.pt` / `best.pt` | dmax `0.1832`、success `0.9736`、collision `0.0059`、timeout `0.0205` | 未通过；timeout 失败 |

## Checkpoint Seed Sweep

对 `015360/016384/017408` 做 `1023/2023/3023/4023` 四个 eval seed 复验：

| checkpoint | dmax mean | success mean | collision mean | timeout mean | strict |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ppo_timestep_015360.pt` | `0.1826` | `0.9663` | `0.0120` | `0.0217` | `0/4` |
| `ppo_timestep_016384.pt` | `0.1823` | `0.9780` | `0.0059` | `0.0161` | `0/4` |
| `ppo_timestep_017408.pt` | `0.1807` | `0.9719` | `0.0085` | `0.0195` | `0/4` |

`016384` 仍是附近最好点，但 timeout 均值 `0.0161` 不优于 exp051 的 `0.0134`。

## 失败分析

exp062 保留了 dmax/success/collision 达标，但 timeout 没有清零，且单 seed final eval timeout `0.0205` 差于 exp051。

success-gate recheck 写入：

```text
outputs/runs/exp062_structured_bicycle_quintic_map25_critic_minpair/pure_rl_seed23_40m_critic_minpair/metrics/success_gate_diagnostics.json
```

诊断结果：success `1005`、collision `6`、timeout `13`。timeout 中 `min_pairwise` gate 失败 `8/13`，dmax gate 失败 `5/13`，dispersion gate 失败 `1/13`，speed gate `0/13`。这说明 critic-only `min_pairwise` state 没有把尾部失败压到 0，并且部分 timeout 已混入 dmax 余量不足。

## 产物路径

```text
outputs/runs/exp062_structured_bicycle_quintic_map25_critic_minpair/pure_rl_seed23_40m_critic_minpair/metrics/summary.json
outputs/runs/exp062_structured_bicycle_quintic_map25_critic_minpair/pure_rl_seed23_40m_critic_minpair/metrics/final_eval_proxy.json
outputs/runs/exp062_structured_bicycle_quintic_map25_critic_minpair/pure_rl_seed23_40m_critic_minpair/metrics/success_gate_diagnostics.json
outputs/runs/exp062_structured_bicycle_quintic_map25_critic_minpair/pure_rl_seed23_40m_critic_minpair/metrics/checkpoint_seed_sweep/summary.json
outputs/runs/exp062_structured_bicycle_quintic_map25_critic_minpair/pure_rl_seed23_40m_critic_minpair/checkpoints/best.pt
```

## 结论

exp062 不是当前最好结果。critic-only `min_pairwise` state 比 exp061 克制，也保住了 dmax/success/collision，但 timeout 仍明显失败，multi-seed 复验也不优于 exp051。

## 下一步

回到 exp051 作为当前最好候选。不要继续扩大 gate 特征或 filter/control 权限；下一轮若继续优化，应优先做更保守的训练稳定性/seed 稳健性诊断，或非常窄的末端 hold 学习信号对照。
