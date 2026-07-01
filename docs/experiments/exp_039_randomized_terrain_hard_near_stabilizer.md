# exp039 hard near stabilizer 诊断

## 目的

基于 exp038 的剩余 timeout 尾部，尝试更强的 near/hold stabilizer，检查“全局更硬 endpoint-near/hold 约束”是否能消除 timeout。

## 配置

config: `configs/experiment/exp039_randomized_terrain_hard_near_stabilizer.yaml`

相对 exp038：

- `hold_zone_spacing_weight=6.50`
- `hold_zone_pairwise_distance=0.56`
- `endpoint_safe_distance=0.48`
- `path_safe_distance=0.38`
- `apply_probability_end=0.65`
- `deterministic_improvement_margin=0.015`

## 严格标准

诊断复评仍参考 proxy strict gate：`dmax_reduction_ratio <= 0.20`、`success_rate >= 0.90`、`collision_rate <= 0.02`、`timeout_rate == 0`。

## 结果表

| eval | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| exp038 best with exp039 config, seed1023 | exp038 `best.pt` | 0.1606 | 0.9424 | 0.0254 | 0.0322 | 未通过 |

## 失败分析

hard near stabilizer 在 exp038 best 上直接退化：success 降低，collision 和 timeout 均升高。timeout 子集也显示 dmax / dispersion 变差，说明更硬的全局 near/hold 约束会干扰集合几何。

## 产物路径

```text
outputs/runs/exp039_randomized_terrain_hard_near_stabilizer/_diagnostics/
```

## 结论

不建议按原样启动长训练。下一步不应继续全局加硬 near/hold filter。

