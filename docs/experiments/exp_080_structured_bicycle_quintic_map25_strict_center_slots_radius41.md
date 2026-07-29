# exp080：严格共同质心校正 + 0.41 m 槽位（当前最佳执行期对照）

## 目的

exp078 的严格共同质心校正几乎通过 dmax，但仍略超限。exp080 只将对称槽位半径从 `0.42 m` 缩至 `0.41 m`；相邻槽位间距约 `0.580 m`，仍高于 `0.42 m` 的 success 安全下限。共同质心校正只在 dmax 和 dispersion 都已经通过真实门时触发。

## 配置与严格标准

配置为 `configs/experiment/exp080_structured_bicycle_quintic_map25_strict_center_slots_radius41.yaml`。严格 proxy gate 为 dmax ratio `<=0.2`、success `>=0.9`、collision `<=0.02`、timeout `=0`。

本实验是对 exp073 best checkpoint 的**执行期后验评测**，不是该配置训练出的新 policy：checkpoint metadata 的 filter progress 为 `512`，因此子目标 filter 尚未介入。

## 结果

seed `11023`、512 环境、320 步的机器可读结果为：

| dmax ratio | success | collision | timeout | actual flatness | strict |
| ---: | ---: | ---: | ---: | ---: | --- |
| `0.1995` | `0.6250` | `0.0000` | `0.3750` | `0.7109` | 未通过 |

它相对 exp073 的同一 checkpoint/后验语义提高 success 并降低 timeout，同时保住 dmax 和 collision；但 success/timeout 仍远离 strict，不能作为训练成功声明。

## Gate 诊断与可视化

512 episode 诊断有 `320` success、`0` collision、`192` timeout；timeout 最终失败计数为 flatness `148`、dmax `101`、dispersion `95`、min-pairwise `0`。成功示例 GIF（seed `11023`）在第 `283` 步完成，最终 dmax=`1.0019 m`、height range=`0.0834 m`、max slope=`0.2477`。

## 产物路径

- 评测：`outputs/runs/exp080_structured_bicycle_quintic_map25_strict_center_slots_radius41/counterfactual_exp073_checkpoint_eval.json`
- gate 诊断：`outputs/runs/exp080_structured_bicycle_quintic_map25_strict_center_slots_radius41/metrics/success_gate_diagnostics.json`
- GIF 与高度图：同目录 `videos/proxy_eval_rollout.gif`、`figures/terrain_height_map.png`

## 结论

这是当前最好的执行期对照，推荐作为下一次“训练时同样启用严格共同校正”设计的比较基线；当前本身不触发 formal long run 或 PhysX。
