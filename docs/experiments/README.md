# 实验索引

除非实验文档另有说明，proxy strict gate 为：

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

PhysX / Jetbot 结果是 high-fidelity closed-loop evaluation，不等于 Isaac Lab 物理训练结果。使用 checkpoint 前优先检查对应 run 的 `metrics/checkpoint_status.json`。

| 实验 | 地形 | 方法 | Seeds | Proxy strict | 当前结论 | 文档 |
| --- | --- | --- | --- | --- | --- | --- |
| exp006 | 平地 proxy | BC + PPO，PPO 阶段选 checkpoint | 23, 31, 47 | 通过 | 平地 proxy baseline；不是 pure RL 从零收敛。 | [exp_006_ppo_selected.md](exp_006_ppo_selected.md) |
| exp007 | lunar crater proxy + PhysX Jetbot eval | 弱 warm-start + PPO | selected run | selected checkpoint 通过 | 证明 checkpoint 可接入 PhysX / Jetbot 闭环评估；不代表物理训练。 | [exp_007_phase_c.md](exp_007_phase_c.md) |
| exp008 | 弱 lunar crater 3D proxy | 弱 warm-start + PPO | 23, 31, 47 | 通过 | 当前最完整 terrain-aware proxy baseline。 | [exp_008_terrain3d.md](exp_008_terrain3d.md) |
| exp009 | 强 lunar crater 3D proxy | 弱 warm-start + PPO | 23, 31；47 未运行 | 未通过 | strong terrain 诊断；近期不继续默认堆 long-budget PPO。 | [exp_009_terrain3d_strong.md](exp_009_terrain3d_strong.md) |
| exp010 | 强 lunar crater 3D proxy | 成功 gate 诊断 + hold/safety 短程修复 | seed23 eval；seed31 continuation；seed47 未启动 | 未通过 | success 可改善，但 collision/timeout gate 仍失败。 | [exp_010_strong_success_diagnostics.md](exp_010_strong_success_diagnostics.md) |
| exp012 | proxy SKRL-MAPPO CUDA 诊断 | action scale warmup probe | seed7 | 未通过 | 工程链路和动作尺度诊断，不作为主结果。 | [exp_012_action_scale_warmup_probe.md](exp_012_action_scale_warmup_probe.md) |
| exp013 | proxy SKRL-MAPPO CUDA 诊断 | action scale ablation + teacher reachability | seed7 | 未通过 | 当前 100-step 小动作配置本身几乎不可达；统一评估会写 checkpoint status。 | [exp_013_action_scale_ablation.md](exp_013_action_scale_ablation.md) |

新增实验时，在这里加一行，并在本目录创建独立的 `exp_###_*.md` 文档。日期流水账放入 `docs/archive/`，不要继续堆到当前实验文档里。
