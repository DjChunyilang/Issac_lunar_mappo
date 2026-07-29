# exp109：96 秒条件末段控制、鲁棒集合点与槽位半径筛选

## 目的

在 exp101 的 gate 诊断表明 timeout 分为“仅实际质心不平整”和“仅 dmax/dispersion 未收紧”两类后，验证三组不改变 strict gate 的执行期对照：

- 条件末段控制：不平整时搜索真实局部平整候选；已平整但几何未过时，将各车子目标向以当前实际质心为中心的对称槽位轻微收缩；
- 鲁棒 terrain-aware 集合点：要求候选在小范围实际质心偏移下仍通过相同完整平整圆盘；
- 槽位半径：在保持最小两两距离门槛的前提下缩小执行槽位。

所有变体都使用 exp092 的 BC32 checkpoint、1024 环境、seed `1023`、96 秒/480 control steps；成功仍由实际团队质心的平整度、dmax、dispersion、速度、最小两两距离和 8-step hold 独立判定。

## 配置

- 条件控制：`exp102`（扩大真实局部平整搜索）、`exp103`（已平整时的原地几何收紧）、`exp104`（组合）、`exp105`（动态槽位重匹配）。
- 鲁棒集合点：`exp106`（`robustness_radius=0.05 m`）、`exp107`（`0.075 m`）。
- 槽位半径：`exp108`（`0.33 m`）、`exp109`（`0.34 m`）。

`flat_geometry_capture` 仅在上一状态实际质心已通过完整平整 gate、仍处于近末段区域且 dmax 或 dispersion 尚未通过时激活。它并不改变 Actor 的固定槽位观测，也不使用 oracle 代理成功；仅把执行子目标向“当前质心 + 对称槽位偏移”插值。`exp105` 额外按当前 rover 位置重新匹配这些局部槽位，以避免复用 reset 时的固定分配。

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
| exp099 共同中心校正对照 | `0.1843` | `0.8643` | `0.0000` | `0.1357` | `0.9102` | 本轮对照最好，未 strict |
| exp102 宽局部平整搜索 (`0.50 m/16`) | `0.1892` | `0.8359` | `0.0000` | `0.1641` | `0.9033` | reject |
| exp103 平整状态原地几何收紧 | `0.1856` | `0.8535` | `0.0000` | `0.1465` | `0.9082` | reject |
| exp104 条件分支组合 | `0.1878` | `0.8438` | `0.0000` | `0.1563` | `0.9111` | reject |
| exp105 动态重匹配几何收紧 | `0.1856` | `0.8535` | `0.0000` | `0.1465` | `0.9092` | 与 exp103 等价，reject |
| exp106 `0.05 m` 鲁棒集合点 | `0.1961` | `0.8535` | `0.0000` | `0.1465` | `0.9443` | 平整率提高但到达性退化，reject |
| exp107 `0.075 m` 鲁棒集合点 | `0.1969` | `0.8457` | `0.0000` | `0.1543` | `0.9551` | 更强同向退化，reject |
| exp108 槽位半径 `0.33 m` | `0.1859` | `0.8379` | `0.0000` | `0.1621` | `0.8906` | reject |
| exp109 槽位半径 `0.34 m` | `0.1864` | `0.8467` | `0.0000` | `0.1533` | `0.8965` | reject |

## 失败分析

exp103 的独立 gate 诊断有 150 个 timeout。相较 BC32 的原始诊断，几何通过率略有提高（timeout 内 dmax 通过率 `36.0%`、dispersion 通过率 `42.7%`），但 flatness 通过率降至 `37.3%`，并产生更多平整度与几何同时失败的 episode。动态槽位重匹配没有改变结果，说明退化不由 reset-time 槽位身份陈旧主导。

exp106/107 证明更强的真实平整盆地搜索可以将最终平整率提升到 `94.4%/95.5%`，但搜索目标的平均行程从对照约 `3.25 m` 提升到 `3.42 m/3.49 m`，导致 dmax/dispersion 收敛变慢。缩小固定执行槽位同样未能弥补这种末段耦合。所有实验 collision 均为零，因此不能通过放松安全约束来解释或修复该问题。

这组结果否定了“在未重新训练策略的情况下，以额外执行期控制同时修复两类 timeout”的假设。尤其是，平整度和紧凑度不是可独立叠加的末段后处理目标；策略需要学习到在接近地形边界时保持兼顾二者的轨迹，而不是在最后时刻被强制改写子目标。

## 产物路径

- `outputs/runs/exp102_structured_bicycle_quintic_map25_time96_wide_local_flatness_center/counterfactual_exp092_bc32_eval_1024.json`
- `outputs/runs/exp103_structured_bicycle_quintic_map25_time96_flat_geometry_capture/counterfactual_exp092_bc32_eval_1024.json`
- `outputs/runs/exp103_structured_bicycle_quintic_map25_time96_flat_geometry_capture/success_gate_diagnostics.json`
- `outputs/runs/exp104_structured_bicycle_quintic_map25_time96_conditional_terminal_branches/counterfactual_exp092_bc32_eval_1024.json`
- `outputs/runs/exp105_structured_bicycle_quintic_map25_time96_dynamic_flat_geometry_capture/counterfactual_exp092_bc32_eval_1024.json`
- `outputs/runs/exp106_structured_bicycle_quintic_map25_time96_robust_flat_oracle05/counterfactual_exp092_bc32_eval_1024.json`
- `outputs/runs/exp107_structured_bicycle_quintic_map25_time96_robust_flat_oracle075/counterfactual_exp092_bc32_eval_1024.json`
- `outputs/runs/exp108_structured_bicycle_quintic_map25_time96_slots_radius33/counterfactual_exp092_bc32_eval_1024.json`
- `outputs/runs/exp109_structured_bicycle_quintic_map25_time96_slots_radius34/counterfactual_exp092_bc32_eval_1024.json`

## 结论与下一步

保持 exp099 作为当前 96 秒执行期对照；exp102–exp109 全部拒绝，不启动 PPO，也不触发 PhysX。保留默认关闭的 `flat_geometry_capture` 实现和单元测试，作为可复现的条件控制基线。

下一轮训练改为目标/教师层面的迭代，而非继续堆叠后处理：教师需要在接近末段时显式输出兼顾完整平整 footprint 与紧凑度的连续目标，且必须与 Actor 的固定槽位观测契约同步更新。先进行短 BC screen 并将未更新 checkpoint 一并候选筛选；只有超过 exp099 的 96 秒后验成功率才进入 PPO。
