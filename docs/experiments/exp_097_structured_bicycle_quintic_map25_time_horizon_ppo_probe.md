# exp097：执行时域扫描与 128 秒 PPO 微调探针

## 目的

验证 exp092 的 timeout 是否主要由 `64 s` 执行上限造成；严格成功定义不变，即实际团队质心的平整圆盘、几何/速度/间距和 8-step hold 必须同时满足。随后在最好的时间时域上做短 PPO 微调，并保留未更新的策略作为可回退候选。

## 配置

基线 checkpoint 为 `exp092` 的 `bc_reanchor_exp073_seed23_32_flatness_early_radius35/checkpoints/best.pt`。exp093–096 逐层继承配置，唯一有效环境变化是将 `simulation.episode_length_s` 及全部评测 steps 同步为 `80/400`、`96/480`、`112/560`、`128/640`。exp097 继承 128 秒配置，checkpoint interval 为 `256`；从上述 BC32 checkpoint 初始化，执行 seed `23`、2048 training env、512 PPO timesteps，并以 1024 环境、640 steps 评估 `t=0/256/512` 候选。

严格 proxy gate 保持：dmax ratio `<=0.2`、success `>=0.9`、collision `<=0.02`、timeout `=0`。

## 结果

### 时域对照

相同 checkpoint、1024 环境、seed `11023`：

| 执行时域 / steps | dmax ratio | success | collision | timeout | 实际集合点平整率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 s / 320 | `0.1910` | `0.7002` | `0.0000` | `0.2998` | `0.7510` |
| 80 s / 400 | `0.1824` | `0.8135` | `0.0000` | `0.1865` | `0.8740` |
| 96 s / 480 | `0.1837` | `0.8594` | `0.0000` | `0.1406` | `0.9092` |
| 112 s / 560 | `0.1824` | `0.8779` | `0.0000` | `0.1221` | `0.9121` |
| 128 s / 640 | `0.1801` | `0.8994` | `0.0000` | `0.1006` | `0.9199` |

增加执行时域显著提升了收敛机会，128 秒的 success 仅差 1/1024 到达 `0.9`，但 timeout 仍为 103/1024，故严格验收失败。

### PPO 候选筛选

训练期候选采用独立 seed `1023`，仅用于比较更新方向；最终正式终评仍使用 seed `11023`。

| checkpoint | PPO timestep | dmax ratio | success | collision | timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ppo_timestep_000000.pt` | 0 | `0.1803` | `0.8887` | `0.0000` | `0.1113` |
| `ppo_timestep_000256.pt` | 256 | `0.1822` | `0.8691` | `0.0000` | `0.1309` |
| `ppo_timestep_000512.pt` | 512 | `0.1823` | `0.8418` | `0.0000` | `0.1582` |

筛选器正确选择 t=0。其正式 1024 环境复验为 `0.1801/0.8994/0/0.1006`，没有将 PPO 退化误报为改进。

## 失败分析

128 秒的 1024-episode gate 诊断：921 success、0 collision、103 timeout。timeout 的最终失败项为 flatness `82`、dmax `58`、dispersion `50`、min-pairwise `0`（失败项可以重叠）；102 个 timeout 在结束时尚未达到瞬时 success，另 1 个只累积了 1/8 hold。故较长时域可以处理一部分慢收敛 episode，但剩余样本并非只差等待时间；应对其末端地形迁移和共同收紧施加直接训练/控制信号。

## 产物路径

- 正式 run：`outputs/runs/exp097_structured_bicycle_quintic_map25_flatness_center_early_radius35_time128_ppo_probe/ppo_from_exp092_bc32_seed23_512_time128/`
- 终评与 strict：`metrics/final_eval_proxy.json`、`metrics/strict_acceptance.json`
- 候选复评：`metrics/eval_metrics.json` 与 `metrics/candidate_ppo_timestep_*_eval.json`
- gate 诊断：`metrics/success_gate_diagnostics.json`
- 曲线：`figures/training_curves.png`、`figures/candidate_eval_curves.png`
- 成功展示：`videos/proxy_eval_rollout.gif`；seed `11023` 在第 244 step 成功，final dmax=`1.0496 m`，实际集合点 height range=`0.1639 m`、max slope=`0.2481`。
- 80–128 秒后验对照：`outputs/runs/exp093.../counterfactual_exp092_bc32_episode400_eval_1024.json` 至 `outputs/runs/exp096.../counterfactual_exp092_bc32_episode640_eval_1024.json`

## 结论与下一步

原 `64 s` 时间预算偏紧；128 秒证明更长时域仍能继续减少 timeout，但按当前实验决策，后续本配置族统一使用 `96 s/480`（`exp094`）作为执行/评测时域，以控制 episode 成本。这不是放宽 strict timeout 验收。短 PPO 微调会退化，当前保留 t=0 的 BC32 checkpoint。下一步聚焦两类 timeout：几何达标但实际集合点不平整的迁移，以及地点平整但 dmax/dispersion 尚未收紧的共同收紧；先做受控末端策略或 reward/teacher 消融，再决定是否进行长训。
