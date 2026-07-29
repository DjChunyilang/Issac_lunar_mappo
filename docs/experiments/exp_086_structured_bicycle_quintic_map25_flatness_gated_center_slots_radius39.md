# exp086：实际平整度门控的共同质心校正 + 0.39 m 槽位

## 目的

exp080 的共同质心校正在几何接近终端区时也会触发。exp086 将其收紧为：只有上一状态的**实际质心 37 点平整圆盘不合格**时才可校正；同时将执行槽位半径降至 `0.39 m`，相邻间距约 `0.552 m`，仍大于 `0.42 m` 的安全下限。目标是在不以 oracle 或几何中点替代 success 的前提下，减少“几何已接近但实际集合点不平”的 timeout。

## 配置与严格标准

配置：`configs/experiment/exp086_structured_bicycle_quintic_map25_flatness_gated_center_slots_radius39.yaml`。

严格 proxy gate：dmax ratio `<=0.2`、success `>=0.9`、collision `<=0.02`、timeout `=0`。实际集合点仍要求半径 `0.75 m`、37 个采样点的 `height_range <=0.18 m` 且 `max_slope <=0.25`。

## 结果

| 设置 | dmax ratio | success | collision | timeout | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| exp073 best 后验，seed `11023`、512 env、320 steps | `0.1993` | `0.6406` | `0.0000` | `0.3594` | 仅执行期对照 |
| 从零训练 `screen_seed23_4m_flatness_gated_radius39`，512 env、320 steps | `0.2088` | `0.6055` | `0.0000` | `0.3945` | reject |

后验中的校正 active fraction 为 `0.1164`，mean/max offset 为 `0.0126/0.1925 m`。从零训练没有继承该后验收益，dmax 也退回 strict 上限外。

## 失败分析

该门控确保成功只在实际点平整时成立，但并不能教会从零策略在末段同时选择平地并收紧队形。结果表明仅改变执行期控制不够；直接把后验改进写成新策略收敛会混淆实验口径。

## 产物路径

- 从零训练：`outputs/runs/exp086_structured_bicycle_quintic_map25_flatness_gated_center_slots_radius39/screen_seed23_4m_flatness_gated_radius39/`
- 机器可读终评：同目录 `metrics/final_eval_proxy.json`、`metrics/strict_acceptance.json`

## 结论与下一步

保留实际平整度门控实现，拒绝从零 scratch checkpoint。下一步改用兼容 checkpoint warm-start，并将初始化策略和 PPO 更新候选显式分开评估，检验更新是否真的优于 source。
