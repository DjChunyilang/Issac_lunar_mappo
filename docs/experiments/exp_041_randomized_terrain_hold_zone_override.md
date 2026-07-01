# exp041 hold-zone override 诊断

## 目的

exp038 已经非常接近 strict，失败集中在少量 episode 的 `0.28–0.42 m` 最近邻灰区。exp041 不再全局加硬 filter，而是在 warmup 后仅当 raw action 会破坏 hold-zone spacing、且候选 action 明确改善 spacing 时才允许 override。

## 配置

config: `configs/experiment/exp041_randomized_terrain_hold_zone_override.yaml`

相对 exp038：

- `hold_zone_spacing_weight=8.00`
- `hold_zone_pairwise_distance=0.58`
- `hold_zone_override_after_warmup=true`
- 其他主要训练、地形、control safety 和 success-zone damping 保持 exp038。

代码侧新增默认关闭字段：

```text
planner.subgoal_filter.hold_zone_override_after_warmup
```

## 严格标准

长训时仍使用 proxy strict gate：`dmax_reduction_ratio <= 0.20`、`success_rate >= 0.90`、`collision_rate <= 0.02`、`timeout_rate == 0`。

## 结果表

| eval | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| exp038 best with exp041 config, seed1023 | exp038 `best.pt` | 0.1595 | 0.9795 | 0.0107 | 0.0098 | 未通过 |

工程 smoke：

- CPU smoke：`outputs/runs/exp041_randomized_terrain_hold_zone_override/smoke_cpu_exp041/`
- CUDA smoke：`outputs/runs/exp041_randomized_terrain_hold_zone_override/smoke_cuda_exp041/`

## 失败分析

exp041 诊断略优于 exp038：success 更高、collision 更低、timeout 从 `0.0107` 降到 `0.0098`。但它仍未消除 timeout，且目前只是把 exp038 checkpoint 放在 exp041 规则下复评，不是从头长训结果。

timeout 子集仍显示最近邻安全间距是尾部瓶颈：`final_nearest_neighbor_distance_mean≈0.410`，低于 `0.42`。

## 产物路径

```text
outputs/runs/exp041_randomized_terrain_hold_zone_override/_diagnostics/
outputs/runs/exp041_randomized_terrain_hold_zone_override/smoke_cpu_exp041/
outputs/runs/exp041_randomized_terrain_hold_zone_override/smoke_cuda_exp041/
```

## 结论

exp041 是下一轮长训练候选，但尚不能写成训练通过。若长训后仍只剩少量 timeout，应继续做更细的末端 pairwise spacing controller，而不是扩大全局安全惩罚。

