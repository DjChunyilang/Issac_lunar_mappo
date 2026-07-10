# exp061 structured bicycle quintic map25 terminal gate observation

## 目的

exp051 的剩余 timeout 主要卡在 terminal `min_pairwise` gate。exp061 检查一个 RL 侧可观测性假设：在不改 action 输出、不新增多点采样、不改 reward/filter/control 的前提下，把 terminal success gate margin 显式加入 Actor/Critic，是否能让 policy 学会末端安全间距与 hold。

## 配置

```text
configs/experiment/exp061_structured_bicycle_quintic_map25_terminal_gate_obs.yaml
```

相对 exp051 的变量：

- `observation.schema_version: ego_v3_local_terrain_grid -> ego_v4_terminal_gate`
- Actor obs dim `86 -> 91`
- Critic state dim `54 -> 55`
- `algorithm.actor_architecture: branched_v1 -> branched_v2`
- `algorithm.critic_architecture: structured_v1 -> structured_v2`

保持不变：单点 `[rho, beta]` action、reward、filter、control safety、reset、terrain、trajectory、PPO 超参和 strict gate。

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
| 23 | `smoke_seed23_128_cpu_terminal_gate_obs` | smoke only | CPU 工程通过；`actor_obs_dim=91`、`critic_state_dim=55`、`branched_v2/structured_v2`、一个 optimizer、8 次 joint update | 非收敛验证 |
| 23 | `smoke_seed23_128_cuda_terminal_gate_obs` | smoke only | CUDA 工程通过；terrain branch 更新 `0.1370`，action std `0.2153` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_terminal_gate_obs` | `ppo_timestep_020480.pt` / `best.pt` | dmax `0.1890`、success `0.8506`、collision `0.0205`、timeout `0.1289` | 未通过 |

## 失败分析

exp061 明显差于 exp051。虽然 dmax 达标，但 success 低于 `0.90`，collision 略高于 `0.02`，timeout 大幅升至 `0.1289`。

success-gate recheck 写入：

```text
outputs/runs/exp061_structured_bicycle_quintic_map25_terminal_gate_obs/pure_rl_seed23_40m_terminal_gate_obs/metrics/success_gate_diagnostics.json
```

诊断结果：`1024` 个 episode 中 success `900`、collision `10`、timeout `114`；timeout 中 `min_pairwise` gate 失败 `105/114`，dmax gate 失败 `14/114`，speed gate `0/114`。同时 post-training eval action saturation 达到 `0.4308`，forward high saturation `0.8540`，说明 gate 特征直接进入 Actor 后策略更激进，反而破坏末端稳定。

## 产物路径

```text
outputs/runs/exp061_structured_bicycle_quintic_map25_terminal_gate_obs/pure_rl_seed23_40m_terminal_gate_obs/metrics/summary.json
outputs/runs/exp061_structured_bicycle_quintic_map25_terminal_gate_obs/pure_rl_seed23_40m_terminal_gate_obs/metrics/final_eval_proxy.json
outputs/runs/exp061_structured_bicycle_quintic_map25_terminal_gate_obs/pure_rl_seed23_40m_terminal_gate_obs/metrics/success_gate_diagnostics.json
outputs/runs/exp061_structured_bicycle_quintic_map25_terminal_gate_obs/pure_rl_seed23_40m_terminal_gate_obs/checkpoints/best.pt
```

## 结论

exp061 不是有效方向。直接把 terminal gate margin 暴露给 Actor/Critic，没有解决 `min_pairwise` timeout，反而让动作更饱和，导致 success、collision 和 timeout 同时退化。

## 下一步

不要沿 exp061 继续加 Actor gate 特征。若继续探索可观测性，应更克制地只改 centralized critic 或训练信号；当前主结果仍回到 exp051。
