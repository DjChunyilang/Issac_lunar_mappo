# exp112--exp117：96 秒 on-policy 尾部 BC 与线速度/共同中心控制扫描

## 目的

在固定 `96 s/480` control steps、terrain-aware 真正最优集合点和实际团队质心 37 点平整度 gate 的前提下，先检验 exp099 的超时是否主要来自 BC 随机快照与实际闭环末段状态的分布不匹配；若否，则在不改策略、奖励或 success predicate 的条件下，扫描低层轨迹跟踪线速度增益，并只对剩余平整度失败测试更早的共同中心校正。

## 配置

- `exp112`：新增 `bc_on_policy_*` 选项。冻结 exp092 `BC32`，在 480 步闭环 rollout 中采集同时接近 dmax/dispersion 门限的 Actor observation，并以既有 `oracle_slots` 教师标注；BC8 每个 batch 的 50% 使用这些尾部样本，50% 保留随机接近状态。
- `exp113`--`exp116`：均从 exp099 和同一 BC32 checkpoint 直接执行；只将 `low_level_control.k_linear` 由 `2.20` 依次提高至 `2.60/2.80/3.00/3.20`。安全投影、固定槽位、真实平整度与所有终止 gate 不变。
- `exp117`：以 `k_linear=3.00` 为基础，仅把共同中心校正的 dmax/dispersion 激活倍数从 `1.50` 提前至 `2.00`。

所有后验评估均为 `seed=1023`、1024 环境、480 steps；BC 使用 `seed=23`，最终比较仍只采用独立的 `seed=1023` 评估文件。

## 严格标准

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

## 结果表

| 变体 | dmax ratio | success | collision | timeout | 最终实际平整率 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| exp099 BC32 控制对照 | `0.1843` | `0.8643` | `0.0000` | `0.1357` | `0.9102` | 原对照 |
| exp112 on-policy 尾部 BC8 | `0.1875` | `0.8457` | `0.0000` | `0.1543` | `0.8965` | reject |
| exp113 `k_linear=2.60` | `0.1839` | `0.8682` | `0.0000` | `0.1318` | `0.8984` | 改善 |
| exp114 `k_linear=2.80` | `0.1812` | `0.8809` | `0.0000` | `0.1191` | `0.9053` | 改善 |
| exp115 `k_linear=3.00` | `0.1798` | `0.8906` | `0.0000` | `0.1094` | `0.9150` | 改善 |
| exp116 `k_linear=3.20` | `0.1802` | `0.8916` | `0.0000` | `0.1084` | `0.9111` | 当前最优，未 strict |
| exp117 提前共同中心校正 | `0.1822` | `0.8838` | `0.0000` | `0.1162` | `0.9082` | reject |

## 失败分析

exp112 的 rollout 共采到约 103 万个近末段 rover-state 样本，说明不是样本不足；但 BC8 仍比 exp099 少约 1.86 个百分点 success。因而“随机 reset 快照与末段闭环状态的分布差”不是当前失败的主要单一原因，继续以同一固定槽位教师进行 BC 重锚定没有依据。

提高 `k_linear` 在不产生碰撞的情况下，将成功数从 exp099 的 885/1024 提升至 exp116 的 913/1024，且平均完成步从约 301 降至 233。`3.00 -> 3.20` 只增加 1 个成功 episode，同时安全投影激活率升至 `29.36%`，已经进入收益平台；不再继续单独提高该增益。

对 exp116 的逐时间门控诊断显示 111 个 timeout 中，最终仍有 57 个 dmax、52 个 dispersion、91 个 flatness 失败（可重叠）；仅 2 个 episode 最终已满足所有瞬时 gate 但未完成 8-step hold。因此 timeout 不是把 hold 门限卡在临界值造成的。轨迹上有 29 个 timeout 从未进入平整 footprint，82 个曾进入但其中 62 个又离开，只有 11 个曾出现瞬时成功。提前共同中心校正使 active fraction 从约 `17.2%` 升至 `20.4%`，却使 success 回落到 `0.8838`，表明在几何收紧之前共同平移会干扰既有闭环轨迹。

## 产物路径

- exp112：`outputs/runs/exp112_structured_bicycle_quintic_map25_time96_on_policy_tail_bc/bc8_on_policy_tail_seed23/metrics/counterfactual_seed1023_eval_1024.json`
- exp113--116：`outputs/runs/exp11{3,4,5,6}_structured_bicycle_quintic_map25_time96_*/counterfactual_exp092_bc32_eval_1024.json`
- exp116 时序 gate：`outputs/runs/exp116_structured_bicycle_quintic_map25_time96_linear_gain32/success_gate_trajectory_diagnostics_seed1023.json`
- exp117：`outputs/runs/exp117_structured_bicycle_quintic_map25_time96_early_center_gain30/counterfactual_exp092_bc32_eval_1024.json`

## 结论

当前 96 秒最佳执行设置为同一 exp092 `BC32` checkpoint 配合 `exp116` 的 `k_linear=3.20`，但它仍有 `10.84%` timeout，严格验收失败；不能触发 PhysX 或标记为通过。保留 on-policy 尾部采样与更细粒度 gate 诊断能力，默认配置不启用尾部 BC。

## 下一步

控制增益和更早共同平移均已出现平台或退化。下一次训练迭代应改变末段连续决策本身：以 timeout 轨迹中“未进入平整 footprint”与“进入后离开”的状态分别构造可执行的多步监督或课程，而不是继续用同一固定槽位单步 BC 标签、放宽 timeout，或以 oracle 点代理真实成功。
