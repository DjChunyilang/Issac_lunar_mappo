# exp040 soft hold stabilizer 诊断

## 目的

exp039 的 hard near stabilizer 退化明显。exp040 改为 soft hold stabilizer：不扩大 endpoint/path hard 安全距离，只加强 hold-zone spacing score 和降低 deterministic replacement margin，诊断是否能温和减少 exp038 的 timeout 尾部。

## 配置

config: `configs/experiment/exp040_randomized_terrain_soft_hold_stabilizer.yaml`

相对 exp038：

- `hold_zone_rho_weight=1.30`
- `hold_zone_spacing_weight=8.00`
- `hold_zone_pairwise_distance=0.58`
- `apply_probability_end=0.65`
- `score_scale_end=0.75`
- `deterministic_improvement_margin=0.005`

## 严格标准

诊断复评仍参考 proxy strict gate：`dmax_reduction_ratio <= 0.20`、`success_rate >= 0.90`、`collision_rate <= 0.02`、`timeout_rate == 0`。

## 结果表

| eval | checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| exp038 best with exp040 config, seed1023 | exp038 `best.pt` | 0.1603 | 0.9658 | 0.0186 | 0.0166 | 未通过 |

## 失败分析

soft hold stabilizer 保持 collision 在 strict 内，但 timeout `0.0166` 仍高于 exp038 的 `0.0107`。timeout 子集的最近邻距离均值约 `0.429`，仍围绕 `0.42` 安全成功间距抖动。

## 产物路径

```text
outputs/runs/exp040_randomized_terrain_soft_hold_stabilizer/_diagnostics/
```

## 结论

不建议按原样长训。它说明单纯增加 soft hold score 仍可能扰动 success/timeout，下一步应只在 raw action 会破坏 hold-zone spacing 时局部 override。

