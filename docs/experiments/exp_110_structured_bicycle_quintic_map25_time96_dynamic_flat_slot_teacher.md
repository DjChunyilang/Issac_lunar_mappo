# exp110：96 秒动态真实平整槽位目标与短 BC 筛选

## 目的

exp102–109 说明，直接在执行层叠加末段后处理不能同时修复实际质心完整平整 footprint 和几何收紧。exp110 改为目标/教师层：当团队进入近末段、实际质心未通过同一 37 点平整度 gate 时，搜索当前质心附近的真实平整候选，并把以候选为中心、按当前行程最小化分配的对称槽位作为 Actor 的下一步局部目标。

这个目标仍以 `ego_v6_gather_slot_goal` 的车体系相对向量与距离表达，Actor 输入维度保持 `89`；不暴露全局坐标、搜索 score 或 oracle 成功代理。固定 reset 槽位继续用于 dense reward，成功仍独立检查实际团队质心的平整度、dmax、dispersion、速度、最小两两距离和 8-step hold。

## 配置

- `configs/experiment/exp110_structured_bicycle_quintic_map25_time96_dynamic_flat_slot_teacher.yaml`
  - 近末段门槛：`1.25 × dmax/dispersion`；仅实际平整度失败时启用；以当前质心为中心搜索 `0.25 m/8` 个真实候选。
  - 基于 exp099 保留固定集合点共同中心校正；因此先验证现有 BC32 在新观测契约下的后验行为，再用 `8` 次、`3e-5` 学习率的 BC 更新筛选。
  - BC batch 的 `50%` 为 `0.35–0.60 m` 半径、`0.04 m` jitter 的近末段状态，其余保留原随机接近状态。
- `configs/experiment/exp111_structured_bicycle_quintic_map25_time96_dynamic_flat_slot_no_static_correction.yaml`
  - 只关闭 exp099 的固定中心校正，验证它是否与动态局部平整槽位指向不同中心而造成退化；不做 BC 或 PPO。

所有比较使用 exp092 的 BC32 checkpoint、`seed=1023`、`1024` 并行环境、`96 s/480` control steps。BC run 本身使用 `seed=23`；最终对照仍以同一 `seed=1023` 独立评测文件为准。

## 严格标准

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

## 结果表

| 变体 | dmax ratio | success | collision | timeout | 最终实际平整率 | 动态目标激活步占比 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp099 BC32 固定中心校正对照 | `0.1843` | `0.8643` | `0.0000` | `0.1357` | `0.9102` | `0` | 当前 96 秒最好，未 strict |
| exp110 动态槽位、未更新 BC32 | `0.1860` | `0.8574` | `0.0000` | `0.1426` | `0.9072` | `0.1036` | reject |
| exp110 动态槽位 + BC8 | `0.1858` | `0.8516` | `0.0000` | `0.1484` | `0.8945` | `0.1074` | reject |
| exp111 动态槽位、关闭固定中心校正 | `0.1767` | `0.8027` | `0.0000` | `0.1973` | `0.8291` | `0.1534` | reject |

## 失败分析

动态目标确实在评测中生效（约 `10.4%–15.3%` 的有效步），但没有改善成功率。保留 exp099 中心校正时，未更新 BC32 已比 exp099 少 `0.0068` success；BC8 又减少到 `0.8516`，并把 safety projection 的激活率从未更新变体的 `22.84%` 推至 `24.98%`。随机快照 BC 不能保证新目标切换后的闭环状态分布与标签一致，且更多末段动作触发安全投影，未能形成所需的平整、紧凑和 8-step hold 轨迹。

关闭固定中心校正确实使几何指标更紧（dmax ratio `0.1767`），但 timeout 上升到 `19.73%`，最终实际平整率降至 `82.91%`。因此退化不只是“固定中心校正和动态槽位互相抢控制权”：固定全局 terrain-aware 中心校正对维持完整平整 footprint 仍有贡献；当前以局部平整候选替换 Actor 目标会把团队带到更难持续满足完整圆盘 gate 的区域。

三组结果 collision 均为零，却同时违反 success 和 timeout gate；不能通过放松安全或 timeout 标准解释为成功。

## 产物路径

- `outputs/runs/exp110_structured_bicycle_quintic_map25_time96_dynamic_flat_slot_teacher/counterfactual_exp092_bc32_eval_1024.json`
- `outputs/runs/exp110_structured_bicycle_quintic_map25_time96_dynamic_flat_slot_teacher/bc8_terminal_flat_seed23/metrics/summary.json`
- `outputs/runs/exp110_structured_bicycle_quintic_map25_time96_dynamic_flat_slot_teacher/bc8_terminal_flat_seed23/metrics/counterfactual_seed1023_eval_1024.json`
- `outputs/runs/exp111_structured_bicycle_quintic_map25_time96_dynamic_flat_slot_no_static_correction/counterfactual_exp092_bc32_eval_1024.json`

## 结论

exp110/111 均未通过 strict，也未超过 exp099；不启动 PPO 或 PhysX。实现保留为默认关闭的、可测试的 Actor/teacher 契约基线，输出 checkpoint 仍只是 `candidate`，不作为推荐 checkpoint。

## 下一步

停止继续叠加单步局部平整目标或随机快照 BC。下一轮先对 exp099 的 timeout trajectory 做时序 gate 诊断：区分“进入平整 footprint 后再次离开”和“从未在平整 footprint 内完成 hold”，再决定是否需要基于 on-policy rollout 状态而不是独立随机快照构造教师数据。96 秒时域、实际 37 点平整度 gate 和 strict timeout gate 均不放宽。
