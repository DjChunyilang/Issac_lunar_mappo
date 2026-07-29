# exp087：兼容 checkpoint warm-start 的实际平整度门控校正

## 目的

exp086 的 scratch 训练未复现执行期收益。本实验从 exp073 best checkpoint 初始化共享 Actor/Critic，保持真实地形最优搜索、实际质心平整度 success gate 和 `0.39 m` 对称槽位不变。初始化不是 resume：optimizer、rollout memory 和训练步数均重新开始；BC 更新关闭，并把未更新的 source 保存为候选 `ppo_timestep_000000.pt`，使 PPO 退化可以被直接发现。

## 配置

配置：`configs/experiment/exp087_structured_bicycle_quintic_map25_flatness_gated_warmstart.yaml`。关键差异是 `algorithm.init_checkpoint` 指向 exp073 `best.pt`、学习率 `3e-5`、`bc_updates=0`、checkpoint interval=`512`。训练为 seed `23`、2048 env、1024 updates（2,097,152 env steps）。

严格 proxy gate 保持 dmax ratio `<=0.2`、success `>=0.9`、collision `<=0.02`、timeout `=0`。

## 结果

候选筛选选中 `ppo_timestep_000000.pt`，不是 `512/1024` 次 PPO 更新后的 checkpoint。训练时的候选筛选（seed `1023`、512 env、320 steps）与修复后的默认独立复验（seed `11023`、1024 env、320 steps）分别为：

| 评测口径 | dmax ratio | success | collision | timeout | 实际集合点平整率 | strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 训练时 source 候选 | `0.2052` | `0.6465` | `0.0000` | `0.3535` | `0.7383` | 未通过 |
| 独立正式复验 | `0.2034` | `0.6289` | `0.0000` | `0.7168` | 未通过 |

两种口径均未 strict；512 环境候选筛选和 1024 环境独立复验不能直接互相替代。两者都表明该表现来自兼容 source 在新的执行配置下的 `t=0` 状态，**不能描述为 PPO 改进**。

## Gate 诊断

按默认独立复验的 1024 个 episode：644 success、0 collision、380 timeout。每个成功 episode 的实际集合点均平整；timeout 的最终失败计数为 flatness `290`、dmax `225`、dispersion `221`、min-pairwise `0`。因此剩余失败不是槽位间距或碰撞，而是平整地点与几何收紧未能在同一 hold 窗口同时达成。

成功示例 GIF（seed `11023`）第 277 步完成：final dmax=`0.9985 m`、height range=`0.0845 m`、max slope=`0.2481`，满足实际平整度约束。

## 产物路径

- run：`outputs/runs/exp087_structured_bicycle_quintic_map25_flatness_gated_warmstart/warmstart_exp073_seed23_2m_flatness_gated/`
- 终评与 strict：`metrics/final_eval_proxy.json`、`metrics/strict_acceptance.json`
- gate 诊断：`metrics/success_gate_diagnostics.json`
- 曲线：`figures/training_curves.png`、`figures/candidate_eval_curves.png`、`figures/exp073_vs_exp087_training_curves.png`
- 展示：`videos/proxy_eval_rollout.gif`、`figures/terrain_height_map.png`

## 结论与下一步

warm-start 链路与 `t=0` 候选保护均工作正常，但 PPO 更新没有超过 source；该 checkpoint 仍为 `candidate`，不触发 PhysX。下一轮应针对诊断中两类 timeout 设计训练信号：在平整但 dmax/dispersion 未达标时强化共同收紧；在几何达标但实际点不平时强化迁移到可行平整盆地，且保持真实平整度门不放宽。
