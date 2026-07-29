# exp073：平整余量搜索 + 0.42 m 对称槽位

## 目的

exp072 已提升实际质心平整率和 success，但 dmax 略超限。exp073 仅把其 `execution_slot_radius` 从 `0.45 m` 改为 `0.42 m`，在保持更平集合点偏好的同时回收紧凑度余量。

## 结果

seed `23`、4,194,304 env steps 的终评（seed `11023`、512 环境、320 步）如下：

| dmax ratio | success | collision | timeout | actual flatness | strict |
| ---: | ---: | ---: | ---: | ---: | --- |
| `0.1997` | `0.6113` | `0.0000` | `0.3887` | `0.7090` | 未通过 |

final dmax/dispersion/nearest 为 `1.2066/0.3056/0.5410 m`；oracle feasible rate 为 `1.0`。相对 exp069，success 从 `0.5664` 升至 `0.6113`，timeout 从 `0.4336` 降至 `0.3887`，实际平整率从 `0.6504` 升至 `0.7090`，且 dmax 与 collision 均通过。

## Gate 诊断与可视化

逐 episode recheck 得到 `313` success、`0` collision、`199` timeout。timeout 的最终 gate 失败计数为 flatness `149`、dmax `99`、dispersion `96`、min-pairwise `0`（计数可重叠）。因此 `0.42 m` 低层安全保护仍完整守住间距，当前瓶颈为实际质心平整与末段紧凑度的联合失败。

- 对比曲线：`figures/exp072_vs_exp073_training_curves.png`
- 成功示例 GIF：`videos/proxy_eval_rollout.gif`（seed `11023`，第 `297` 步成功；最终质心 max slope `0.2487`）
- 机器可读诊断：`metrics/success_gate_diagnostics.json`

## 结论

这是当前“真实地形最优搜索 + 实际质心平整 gate”路线中最均衡的 4M screen，但 success/timeout 仍远未达到 strict；不启动 formal long run 或 PhysX。下一步必须在不放宽平整度和最小间距的前提下减少末段 dmax/dispersion 超时。
