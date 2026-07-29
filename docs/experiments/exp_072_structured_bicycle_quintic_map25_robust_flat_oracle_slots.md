# exp072：加大平整度余量偏好

## 目的

搜索点虽满足单中心平整度约束，但实际质心存在约 `0.1 m` 的执行偏移，原 `flatness_weight=0.25` 留出的坡度余量很小。exp072 仅将搜索目标中的软平整度权重升至 `1.50`，偏好更平的可行盆地；真实质心 37 点 hard gate 不变。

## 结果

seed `23`、4,194,304 env steps 的 screen 在 seed `11023`、512 环境、320 步上得到：

| dmax ratio | success | collision | timeout | actual flatness |
| ---: | ---: | ---: | ---: | ---: |
| `0.2038` | `0.6289` | `0.0000` | `0.3711` | `0.7305` |

搜索可行率为 `1.0`，搜索点平均 max slope 从 exp069 的约 `0.2276` 降为 `0.2092`。success、timeout 与实际质心平整率均改善，但 dmax ratio 高出 strict 上限 `0.0038`，未通过。

## 结论

更平的真实搜索点确实改善了执行结果，证明不应退回几何中点代理；代价是更长的初始行程。下一组只微调槽位半径回收 dmax/dispersion 余量。

产物：`outputs/runs/exp072_structured_bicycle_quintic_map25_robust_flat_oracle_slots/screen_seed23_4m_robust_flat_oracle/`，含训练与候选曲线。
