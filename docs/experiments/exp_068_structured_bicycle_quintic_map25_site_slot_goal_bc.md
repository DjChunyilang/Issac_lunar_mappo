# exp068：共同中心 + 专属槽位双目标（BC reject）

## 假设

exp067 的超时主要卡在 actual-centroid flatness，因此该诊断在每辆 rover 的槽位目标外，再追加到共同 terrain-aware 搜索点的车体系相对向量。`ego_v7_gather_site_and_slot_goal` 将 Actor 从 `89` 维扩至 `92` 维，使用 `branched_v4`；Critic 仍为 `54` 维，所有 hard gate 不变。

## 结果

```text
run: bc_smoke_seed23_site_slot_goal
budget: 128 BC updates; 512 env / 320 steps
BC loss: 0.8890 -> 0.0681
success: 0.3203
collision: 0.5234
timeout: 0.1563
final dmax: 1.1971 m
final flatness: 0.4844
final nearest-neighbour: 0.4089 m
```

相对于 exp067 相同的 radius-0.45、128-step BC 对照（success `0.3809`、collision `0.3770`、flatness `0.5664`），双目标同时降低了成功/平整度，并显著增加碰撞。因此不启动 4M screen，不生成代表性 GIF，也不进行 high-fidelity evaluation。

## 结论

公共中心向量与专属槽位向量的简单拼接会让共享策略在末段过度收缩；这不是解决 collision/flatness trade-off 的有效方向。后续应保留 exp067 的对称槽位语义，将改造聚焦在可观测的末段避碰与安全控制，而不放宽真实质心平整度 gate。
