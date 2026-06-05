# 实验索引

除非实验文档另有说明，严格 gate 为：

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

| 实验 | 地形 | 方法 | Seeds | 严格结果 | 文档 |
| --- | --- | --- | --- | --- | --- |
| exp006 | 平地 proxy | BC + PPO，PPO 阶段选 checkpoint | 23, 31, 47 | 通过 | [exp_006_ppo_selected.md](exp_006_ppo_selected.md) |
| exp007 | lunar crater 展示 / Phase C | 弱 warm-start + PPO | selected run | selected checkpoint 通过 | [exp_007_phase_c.md](exp_007_phase_c.md) |
| exp008 | 弱 lunar crater 3D proxy | 弱 warm-start + PPO | 23, 31, 47 | 通过 | [exp_008_terrain3d.md](exp_008_terrain3d.md) |
| exp009 | 强 lunar crater 3D proxy | 弱 warm-start + PPO | 23, 31；47 未运行 | 未通过 | [exp_009_terrain3d_strong.md](exp_009_terrain3d_strong.md) |

新增实验时，在这里加一行，并在本目录创建独立的 `exp_###_*.md` 文档。日期流水账放入 `docs/archive/`，不要继续堆到当前实验文档里。

