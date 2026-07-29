# exp101：96 秒局部平整集合点搜索与保守 PPO 探针

## 目的

在固定 `96 s/480` control-step 时域、真实 terrain-aware 最优集合点和实际质心平整度 gate 的前提下，验证三种末段干预，以及策略是否能通过小学习率适应局部平整度中心搜索：

- exp098：严格逐槽位捕获；
- exp099：增大共同中心校正上限；
- exp100：在当前实际质心附近搜索满足同一 37 点平整度判据的真实候选中心；
- exp101：从 exp092 的 BC32 checkpoint warm-start，以较小学习率进行 4M environment-step PPO 探针。

成功仍只由实际团队质心、几何/速度/安全条件和连续 hold 共同决定；oracle 集合点或局部搜索点均不能直接代理成功。

## 配置

- exp098：`configs/experiment/exp098_structured_bicycle_quintic_map25_time96_strict_slot_capture.yaml`
- exp099：`configs/experiment/exp099_structured_bicycle_quintic_map25_time96_full_center_correction.yaml`
- exp100：`configs/experiment/exp100_structured_bicycle_quintic_map25_time96_local_flatness_center.yaml`
- exp101：`configs/experiment/exp101_structured_bicycle_quintic_map25_time96_local_flatness_ppo.yaml`

exp100 的 `formation_center_local_flatness_search` 只在末段 dmax/dispersion 已达到 success 阈值、上一状态实际质心不平整且普通共同中心校正已经启用时工作。它枚举当前质心与半径 `0.25 m` 的 8 个环形候选，对每个候选运行和 success gate 完全相同的平整圆盘评估；仅在找到真实平整候选时，以该候选替换共同中心校正目标。它不修改各车专属槽位、success 门槛或碰撞约束。

exp101 从 exp092 `BC32` 初始化，以 `learning_rate=1e-5`、`2048 × 2048 = 4,194,304` environment steps 训练；每 `512` 个训练 timestep 保存一个候选，且把 `t=0` 初始化 checkpoint 一起参加筛选。

## 严格标准

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

## 结果表

所有后验控制对照使用相同的 exp092 BC32 source checkpoint、1024 环境和 480 steps。

| 变体 | dmax ratio | success | collision | timeout | 最终实际质心平整率 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| exp094 基线共同中心校正 | `0.1837` | `0.8594` | `0.0000` | `0.1406` | `0.9092` | 未通过 |
| exp098 严格逐槽位捕获 | `0.1831` | `0.8555` | `0.0000` | `0.1445` | — | 退化，关闭 |
| exp099 增大共同中心校正 | `0.1843` | `0.8643` | `0.0000` | `0.1357` | `0.9102` | 该组中最好，但未 strict |
| exp100 真实局部平整中心搜索 | `0.1845` | `0.8604` | `0.0000` | `0.1396` | `0.9150` | 平整率提高，未转化为更多 success |

exp101 的候选筛选（同一 1024 环境、480 steps 协议）如下。`t=0` 即 exp100 的未更新 BC32 候选，并被自动保留为最优；所有 PPO 更新均退化。

| PPO timestep | dmax ratio | success | collision | timeout |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `0.1870` | `0.8604` | `0.0000` | `0.1396` |
| 512 | `0.1885` | `0.8154` | `0.0000` | `0.1846` |
| 1024 | `0.1910` | `0.7539` | `0.0000` | `0.2461` |
| 1536 | `0.1966` | `0.7109` | `0.0000` | `0.2891` |
| 2048 | `0.1988` | `0.6621` | `0.0000` | `0.3379` |

自动最终复评选择 `t=0`：dmax ratio `0.1845`、success `0.8604`、collision `0.0000`、timeout `0.1396`，严格验收失败。PPO 更新后的 checkpoint 不作为候选，也不触发 PhysX。

## 失败分析

对 `t=0` 的 1024 环境、480-step success-gate 逐 episode 诊断得到 143 个 timeout：

| 最终失败模式 | timeout 数 |
| --- | ---: |
| 仅实际质心平整度失败 | 45 |
| 仅 dmax/dispersion 几何失败 | 58 |
| 平整度与 dmax/dispersion 同时失败 | 40 |

所有 timeout 的速度和最小两两间距均通过；因此瓶颈已不是刹停或安全间距。局部搜索实际激活时间占 `12.56%`，使最终平整率从 exp099 的 `0.9102` 上升到 `0.9150`，却没有提高总 success。这表明当前统一的、上限 `0.35 m` 的共同中心校正没有同时解决两种不同尾部状态：平整度单独失败需要短距离重定位，而几何失败需要保持平整约束下的队形收紧。

小学习率 PPO 仍从第一个 `512` timestep 开始持续损失 success，说明在没有显式的末段模式或相应训练信号时，继续微调会破坏 BC32 的现有收敛行为；不能将其当作解决 timeout 的通用手段。

## 产物路径

- exp098 对照：`outputs/runs/exp098_structured_bicycle_quintic_map25_time96_strict_slot_capture/counterfactual_exp092_bc32_strict_slot_capture_eval_1024.json`
- exp099 对照：`outputs/runs/exp099_structured_bicycle_quintic_map25_time96_full_center_correction/counterfactual_exp092_bc32_full_center_correction_eval_1024.json`
- exp100 对照：`outputs/runs/exp100_structured_bicycle_quintic_map25_time96_local_flatness_center/counterfactual_exp092_bc32_local_flatness_center_eval_1024.json`
- exp101 run：`outputs/runs/exp101_structured_bicycle_quintic_map25_time96_local_flatness_ppo/ppo_from_exp092_bc32_seed23_4m_time96_localflat/`
- 候选筛选：`.../metrics/eval_metrics.json`
- 最终复评：`.../metrics/final_eval_proxy.json`
- 严格验收：`.../metrics/strict_acceptance.json`
- gate 诊断：`.../metrics/success_gate_diagnostics.json`

## 结论与下一步

96 秒仍是固定时域，strict timeout gate 不放宽。保留真实局部平整候选搜索的实现及其测试，但不把它作为当前默认候选；同协议下 exp099 的增大共同中心校正在 96 秒控制对照中更好。下一轮应实施显式的条件末段控制：对“几何已过、仅不平整”的状态只做平整候选重定位；对“地点已平整、几何未过”的状态只收紧固定槽位/共同中心，而不是用同一个校正信号兼顾两者。先做后验控制对照，只有有稳定增益时才重新训练。
