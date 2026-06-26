# 实验索引

除非实验文档另有说明，proxy strict gate 为：

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

PhysX / Jackal tracking 结果是 high-fidelity validation，不等于 Isaac Lab 物理训练结果。使用 checkpoint 前优先检查对应 run 的 `metrics/checkpoint_status.json`。

| 实验 | 地形 | 方法 | Seeds | Proxy strict | 当前结论 | 文档 |
| --- | --- | --- | --- | --- | --- | --- |
| exp006 | 平地 proxy | BC + PPO，PPO 阶段选 checkpoint | 23, 31, 47 | 通过 | 平地 proxy baseline；不是 pure RL 从零收敛。 | [exp_006_ppo_selected.md](exp_006_ppo_selected.md) |
| exp007 | lunar crater proxy + 历史 PhysX sanity | 弱 warm-start + PPO | selected run | selected checkpoint 通过 | 历史高保真 sanity；当前活跃 PhysX 验证已切换为 Jackal tracking。 | [exp_007_phase_c.md](exp_007_phase_c.md) |
| exp008 | 弱 lunar crater 3D proxy | 弱 warm-start + PPO | 23, 31, 47 | 通过 | 当前最完整 terrain-aware proxy baseline。 | [exp_008_terrain3d.md](exp_008_terrain3d.md) |
| exp009 | 强 lunar crater 3D proxy | 弱 warm-start + PPO | 23, 31；47 未运行 | 未通过 | strong terrain 诊断；近期不继续默认堆 long-budget PPO。 | [exp_009_terrain3d_strong.md](exp_009_terrain3d_strong.md) |
| exp010 | 强 lunar crater 3D proxy | 成功 gate 诊断 + hold/safety 短程修复 | seed23 eval；seed31 continuation；seed47 未启动 | 未通过 | success 可改善，但 collision/timeout gate 仍失败。 | [exp_010_strong_success_diagnostics.md](exp_010_strong_success_diagnostics.md) |
| exp012 | proxy SKRL-MAPPO CUDA 诊断 | action scale warmup probe | seed7 | 未通过 | 工程链路和动作尺度诊断，不作为主结果。 | [exp_012_action_scale_warmup_probe.md](exp_012_action_scale_warmup_probe.md) |
| exp013 | proxy SKRL-MAPPO CUDA 诊断 | action scale ablation + teacher reachability | seed7 | 未通过 | 当前 100-step 小动作配置本身几乎不可达；统一评估会写 checkpoint status。 | [exp_013_action_scale_ablation.md](exp_013_action_scale_ablation.md) |
| exp014 | 弱 lunar crater proxy | 5×5 局部地形网格 SKRL-MAPPO probe | seed23 | 未运行 strict | CUDA 工程验收通过；只证明新观测与训练链路有效。 | [exp_014_terrain_grid_observation_probe.md](exp_014_terrain_grid_observation_probe.md) |
| exp015 | 偏弱中档 lunar crater proxy | SKRL MAPPO + BC20，86 维地形网格 | seed23 screen | 未通过 | 工程信号正常，但 2M screen 的 dmax/success/collision/timeout 全部失败；未启动 8M。 | [exp_015_skrl_medium_soft_terrain_grid.md](exp_015_skrl_medium_soft_terrain_grid.md) |
| exp016 | 偏弱中档 lunar crater proxy | shared-joint MAPPO + local BC100 + comm12 | seed23 staged probe | 未通过 | shared-update 工程探针通过；BC-only 安全但过于保守，未进入 2M screen。 | [exp_016_shared_mappo_comm12.md](exp_016_shared_mappo_comm12.md) |
| exp017 | 固定偏弱中档 lunar crater proxy | shared-joint MAPPO pure RL + comm12 | seed23 continuous 20M | 单 seed 通过 | final eval：dmax ratio 0.1318、success 0.9990、collision 0.00098、timeout 0；尚未证明多 seed 或随机地图泛化。 | [exp_017_shared_mappo_pure_rl_comm12.md](exp_017_shared_mappo_pure_rl_comm12.md) |
| exp018 | 按 episode 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + comm12 + terrain-aware reward | seed23 continuous 20M | 未通过 | dmax 和 success 达标，但 final eval collision 0.0352、timeout 0.0088 未过 strict；作为随机地形 candidate / 安全失败分析保留。 | [exp_018_randomized_terrain_pure_rl.md](exp_018_randomized_terrain_pure_rl.md) |
| exp019 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + 安全成功门控 + 路径级地形风险 | seed23 20M + 5 eval seeds | 未通过 | 工程链路完成；10240 checkpoint 有集合趋势但 collision 高，best checkpoint 安全但 success/timeout 很差；5 seed 均值 success 0.0143、collision 0.0801、timeout 0.9082。 | [exp_019_randomized_terrain_safe_path_risk.md](exp_019_randomized_terrain_safe_path_risk.md) |
| exp020 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + terrain/safety 子目标过滤 | seed23 20M + 5 eval seeds | 未通过 | 子目标过滤器把 5 seed path risk mean 从 raw 0.3815 降到 0.3187，但集合进度塌缩；success 0、collision 0.0498、timeout 0.9506。 | [exp_020_randomized_terrain_subgoal_filter.md](exp_020_randomized_terrain_subgoal_filter.md) |
| exp021 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + 课程化 terrain/safety 子目标过滤 | seed23 20M + 5 eval seeds | 未通过 | 课程化 filter 恢复集合趋势，5 seed success 0.6361、dmax ratio 0.1460、timeout 0.1967、filtered path risk 0.3638；但 collision 0.1746，安全 gate 明显失败。 | [exp_021_randomized_terrain_filter_curriculum.md](exp_021_randomized_terrain_filter_curriculum.md) |
| exp022 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + endpoint/path safety constrained curriculum filter | seed23 20M + 5 eval seeds | 未通过 | constrained filter 把 collision 压到 0.0170，但 success 只有 0.0139、timeout 0.9699；说明安全后处理过强，集合进度塌缩。 | [exp_022_randomized_terrain_endpoint_safety_filter.md](exp_022_randomized_terrain_endpoint_safety_filter.md) |
| exp023 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + soft progress-preserving subgoal filter | seed23 20M | 未通过 | success 从 exp022 的 0.0139 回升到 0.3027，但 collision 0.2295、timeout 0.4717；static endpoint/path filter 未预测邻居同步运动。 | [exp_023_randomized_terrain_soft_progress_filter.md](exp_023_randomized_terrain_soft_progress_filter.md) |
| exp024 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + mutual path safety subgoal filter | seed23 20M | 未通过 | post-hoc best `10240`：success 0.8398、collision 0.0674、timeout 0.0947；明显优于 exp023，但 strict 安全/timeout 仍未达标。 | [exp_024_randomized_terrain_mutual_path_filter.md](exp_024_randomized_terrain_mutual_path_filter.md) |

新增实验时，在这里加一行，并在本目录创建独立的 `exp_###_*.md` 文档。日期流水账放入 `docs/archive/`，不要继续堆到当前实验文档里。
