# 实验计划

## 当前主线

当前实验主线是“proxy 训练 + checkpoint 级 high-fidelity closed-loop evaluation”：

- `exp008` 保持为当前已验证的 3-seed terrain-aware proxy baseline。
- `exp009` / `exp010` 作为 strong terrain 诊断记录，近期不继续默认扩展。
- `exp012` / `exp013` 作为 SKRL-MAPPO proxy CUDA 与 action-scale 诊断，不作为 strict success。
- `exp017` 是固定地图、seed23、shared-joint MAPPO pure RL 的当前 schema strict pass 单 seed baseline。
- `exp018` 是随机增强地形的 20M 单 seed candidate；dmax/success 达标，但 collision/timeout 未过 strict gate。
- `exp019` 已完成随机增强地形安全/路径风险诊断；工程链路正常，但 strict 未通过，说明当前软路径风险惩罚和安全 gate 组合仍不足。
- `exp020` 已完成 terrain/safety-aware 子目标过滤器诊断；路径风险明显降低，但集合进度被抑制，strict 未通过。
- `exp021` 已完成课程化/软化 filter 诊断；集合进度恢复，但 collision 显著升高，strict 未通过。
- `exp022` 已完成 endpoint/path safety constrained curriculum filter 诊断；collision 被压到 strict 内，但 success/timeout 明显失败，说明 hard constrained post-processing 过强。
- `exp023` 已完成 soft progress-preserving filter 诊断；恢复部分集合进度，但 collision / timeout 仍失败，说明 static endpoint/path safety 不足。
- `exp024` 已完成 mutual path safety filter 诊断；success 提升到 `0.8398`、collision 降到 `0.0674`，但 strict 安全和 timeout 仍未达标。
- `exp025` 已完成 dense mutual path safety filter 诊断；collision 相对 exp024 降低，但 success/timeout/安全 strict 仍未过。
- `exp026` / `exp027` 已完成 hold-zone filter 诊断；过早介入会压制集合，严格晚介入也未优于 exp025。
- `exp028` 已完成 hold reward 诊断；success/timeout 改善，是 exp026–029 中最好结果，但 collision 仍未过 strict。
- `exp029` 已完成 stronger safety 诊断；继续加 safety reward/filter 权重导致 success/timeout/collision 全面退化。
- `exp030` 已完成低层 control safety projection 诊断；collision 降低但 success/timeout 退化。
- `exp031` / `exp032` 已完成投影强度和 closing-only 条件诊断；简单调弱不够，closing-only 略改善但仍未达标。
- `exp033` / `exp034` 已完成 directional scale / mask 诊断；directional mask 是更有效方向，但 collision 仍超 strict。
- `exp035` / `exp036` 已完成 directional mask buffer 与 hold/timeout shaping；success/collision 已同时达标，剩余瓶颈转为 timeout。
- `exp037` 已完成 260-step episode/eval 诊断；timeout 降低但 collision 反弹，说明问题不是单纯时间预算。
- `exp038` 已完成 success-zone stabilizer + 320-step episode/eval；当前随机地形最佳候选，success/collision 已过 strict，仅 timeout `0.0107` 未过。
- `exp039` / `exp040` 已完成 exp038 checkpoint 诊断复评；hard near 与 stronger soft hold 都不建议长训。
- `exp041` 已完成 hold-zone override 诊断和 CPU/CUDA smoke；在 exp038 best 上略优，但当前暂停继续长训。
- `exp042` 已完成结构化 Actor/Critic、bicycle proxy 和 quintic trajectory 工程探针；当前配置进一步扩大到 `25 m × 25 m` 地图，并用 `communication_radius=0.0` 临时取消通信距离限制；只验证训练环境链路，不作为收敛实验。
- `exp043` 是新环境栈下的直接长跑：从随机初始化开始，迁移 exp041 hold-zone override，并扩大 reset 初始队形分布；40M 已完成但 success `0.0`、timeout `1.0`，说明直接上 25m 大地图/大初始分布没有恢复集合学习。
- `exp044` 已完成：保留新网络、bicycle、quintic、25m 地图和无限通信，并新增 initial-state curriculum；final eval dmax ratio 改善到 `0.4796`，但 success 仍为 `0.0`、timeout `0.9980`，说明课程有效但不足。
- `exp045` 已完成：保留新环境栈和 25m 地图，但改成 local-success bootstrap；final eval success 提升到 `0.1846`，collision `0.0`，但 dmax/dispersion/timeout 仍未通过。
- `exp046` 已完成：local success 提升到 `0.6123`、collision `0.0`，但 dmax `0.2424` 和 timeout `0.3877` 未过 strict。
- `exp047` 已完成：success `0.7188`、collision `0.0059`、dmax ratio `0.2132`，但 timeout `0.2764` 仍未 strict；这是新环境栈 local reset 当前最好结果。
- `exp048` 已完成：dmax `0.1866`、success `0.9844`、collision `0.0020` 均通过，唯一失败为 timeout `0.0137`。
- `exp049` 已完成：针对 exp048 剩余最近邻安全间距灰区增强 terminal spacing，但 final eval success `0.8926`、timeout `0.1064`，明显差于 exp048，说明全局 spacing/filter/control safety 修正过强。
- PhysX / Isaac Sim 作为 checkpoint 级高保真闭环评估和展示层，不进入当前主训练 loop。
- 新训练和评估默认写入 `outputs/runs/<experiment>/<run>/`。

严格结论以机器可读结果为准，优先读取：

```text
outputs/runs/<experiment>/_suite/metrics/strict_acceptance.json
outputs/runs/<experiment>/_suite/metrics/suite_summary.json
outputs/runs/<experiment>/<run>/metrics/final_eval_proxy.json
outputs/runs/<experiment>/<run>/metrics/checkpoint_status.json
```

不要从 GIF、截图、单个 checkpoint 文件名或 TensorBoard 曲线推断 strict pass。

## 已确认基线

`exp008` 是当前推荐的完整 terrain-aware proxy 基线：

```text
outputs/runs/exp_008_terrain3d/_suite/
```

结论：

- 弱 lunar crater 3D proxy。
- 弱 warm-start + PPO。
- seeds `23, 31, 47` 均通过 proxy strict gate。
- 后续改动应避免破坏该 baseline，并在需要时用 exp008 suite 做回归对照。

当前 86 维局部地形网格 schema 下，`exp017` 可作为固定地图 single-seed baseline：

```text
outputs/runs/exp017_shared_mappo_pure_rl_comm12/
```

结论：

- 固定偏弱中档 lunar crater proxy。
- shared-joint MAPPO pure RL，不使用 BC。
- seed23 final eval 通过 strict gate。
- 不能替代 exp008 的 3-seed baseline，也不能证明随机地图泛化。

随机地图主诊断是 `exp018`：

```text
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/
```

结论：

- 每环境、每 episode reset 独立随机地形。
- 20M 后 dmax 和 success 达标。
- collision `0.0352`、timeout `0.0088` 未通过 strict gate。
- 下一轮优先改安全和路径级风险，而不是只放大 terrain penalty。

已完成的安全/路径风险诊断是 `exp019`：

```text
outputs/runs/exp019_randomized_terrain_safe_path_risk/
```

结果：

- success gate 已要求最近邻距离 `>= 0.42 m`，显式避开 `collision_distance=0.28 m`。
- terrain reward 已新增当前点到子目标直线路径的 mean/max risk 和高度变化。
- seed23 20M 训练、候选评估、5 轮独立 eval 和 GIF 均已完成。
- 当前 best checkpoint 5 seed 均值：dmax ratio `0.4186`、success `0.0143`、collision `0.0801`、timeout `0.9082`，strict 未通过。
- 10240 checkpoint 的 success 可到 `0.6201`，但 collision `0.1279`，说明后期学习出了集合趋势但安全不达标。

下一轮优先方向：

- 不再只放大 terrain weight；改为把 path risk 前移到子目标候选过滤、局部 planner score 或不可达路径约束。
- 重新成套协调 success 半径、安全半径、episode 长度和 collision penalty，避免“会集合但撞车”和“安全但超时”的两极化。

已完成的子目标过滤诊断是 `exp020`：

```text
outputs/runs/exp020_randomized_terrain_subgoal_filter/
```

结果：

- actor 输出后、轨迹生成前加入 10 候选子目标过滤器，按 path risk、subgoal risk 和 endpoint safety 选择候选。
- seed23 20M 训练、候选评估、5 轮独立 eval、GIF 和训练曲线均已完成。
- 过滤器有效：5 seed 均值 raw path risk `0.3815`，filtered path risk `0.3187`，risk reduction `0.0628`。
- 策略失败：5 seed 均值 dmax ratio `0.3752`、success `0.0`、collision `0.0498`、timeout `0.9506`，strict 未通过。
- 当前结论是 hard filter 过强，牺牲了集合进度；下一轮应做课程化/软化，而不是继续增加过滤强度。

已完成的课程化 filter 诊断是 `exp021`：

```text
outputs/runs/exp021_randomized_terrain_filter_curriculum/
```

结果：

- 保持 exp020 的 pure RL、shared-joint MAPPO、随机增强地形和 `12 m` 通信半径。
- 课程化 filter 前 `2048` timesteps 不替换 action，后续 `4096` timesteps 内把 `apply_probability` 从 `0.0` 线性升到 `0.60`，把 `score_scale` 从 `0.15` 升到 `0.75`。
- seed23 20M 训练、候选评估、5 轮独立 eval、GIF 和训练曲线已完成。
- 5 seed mean：dmax ratio `0.1460`、success `0.6361`、collision `0.1746`、timeout `0.1967`。
- filter 仍降低路径风险：raw path risk `0.3871`，filtered path risk `0.3638`，risk reduction `0.0233`。

判读：

- exp021 相比 exp020 恢复了集合和降低了 timeout，但 collision 明显恶化。
- 下一步不宜继续只调 filter 强度或 terrain reward；应让 endpoint safety 更直接地参与 action/planner 或 policy loss。

已完成的 endpoint/path safety constrained filter 诊断是 `exp022`：

```text
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/
```

结果：

- 保持 exp021 的 pure RL、shared-joint MAPPO、随机增强地形和 `12 m` 通信半径。
- 子目标 filter 改为 `terrain_safe_candidate_constrained_curriculum`。
- 候选从 10 个扩展到 28 个：`rho_scales=[0.45, 0.70, 0.90, 1.0]`，`beta_offsets_deg=[-45, -30, -15, 0, 15, 30, 45]`。
- warmup 前仍不替换 action；warmup 后对 endpoint/path unsafe raw action 允许 safety override。
- 加入 endpoint safety、path collision 和 visible-neighbor center progress hard constraint，避免只绕远但不集合。
- seed23 20M 训练、候选评估、5 轮独立 eval、GIF 和训练曲线已完成。
- 5 seed mean：dmax ratio `0.4719`、success `0.0139`、collision `0.0170`、timeout `0.9699`。
- filter 很强：raw path risk `0.3379`，filtered path risk `0.2737`，risk reduction `0.0642`，applied fraction `0.6165`。

判读：

- strict gate 未通过。
- collision 从 exp021 的 `0.1746` 降到 `0.0170`，说明 endpoint/path safety constraint 有效。
- success 从 exp021 的 `0.6361` 降到 `0.0139`，timeout 从 `0.1967` 升到 `0.9699`，说明 constrained filter 仍过强。
- 下一轮应转向 policy/action representation、planner projection 或 success geometry，而不是继续放大后处理权重。

`exp023_randomized_terrain_soft_progress_filter` 已完成：

- dmax ratio `0.1789` 通过，但 success `0.3027`、collision `0.2295`、timeout `0.4717` 未通过 strict。
- 和 exp022 相比，soft filter 恢复部分集合进度；和 exp021 相比，collision 更高。
- 失败原因是 static endpoint/path safety 只看邻居当前位置，未预测可见邻居 raw path 的同步运动。

`exp024_randomized_terrain_mutual_path_filter` 已完成：

- post-hoc 使用 `success_progress_long` 重选 10240 checkpoint。
- seed1023 final eval：dmax ratio `0.1397`、success `0.8398`、collision `0.0674`、timeout `0.0947`。
- mutual path filter 明显改善 exp023 的同步路径冲突：`filter_raw_mutual_path_collision_violation_mean=0.0341`，filtered 后为 `0.000879`。
- strict gate 仍未通过，剩余问题主要是 late-stage collision 和少量 timeout。

`exp025_randomized_terrain_dense_mutual_filter` 已完成：

- 仍使用 exp024 的 pure RL、shared-joint MAPPO、随机增强地形、`12 m` 通信半径和 20M budget。
- filter mode 仍为 `terrain_safe_candidate_mutual_progress_curriculum`，不启用 exp022 式 hard endpoint/path safety constraint。
- 把 `path_samples` 从 5 加密到 9，并把候选集合改为 `rho_scales=[0.60, 0.80, 1.0, 1.08]` × `beta_offsets_deg=[-40, -25, -12.5, 0, 12.5, 25, 40]`。
- 适度提高 path/mutual collision 权重：`path_collision_weight=450.0`、`mutual_path_collision_weight=1200.0`。
- checkpoint 选择使用 `success_progress_long`，避免再次选中低 collision 但低 success 的早期 checkpoint。
- best 为 `ppo_timestep_009216.pt`；final eval：dmax ratio `0.1434`、success `0.8525`、collision `0.0449`、timeout `0.1035`。
- 相对 exp024，success 小幅提高、collision 降低，但 timeout 未改善，strict 仍未通过。

`exp026_randomized_terrain_hold_stable_filter` 已完成：

- final eval：dmax ratio `0.1474`、success `0.7529`、collision `0.0615`、timeout `0.1865`。
- `filter_hold_zone_activation_mean=0.1731`，说明 hold-zone cost 参与了决策，但介入过早/过宽，反而压制集合。

`exp027_randomized_terrain_strict_hold_filter` 已完成：

- final eval：dmax ratio `0.1464`、success `0.8418`、collision `0.0498`、timeout `0.1123`。
- 严格 hold-zone trigger 避免 exp026 的明显退化，但未优于 exp025。

`exp028_randomized_terrain_hold_reward` 已完成：

- final eval：dmax ratio `0.1415`、success `0.8691`、collision `0.0469`、timeout `0.0889`。
- `max_success_hold_count_mean=7.3633/8`，是 exp026–029 中最好结果。
- post-hoc 把 deterministic filter margin 降到 `0.0` 后 collision 升到 `0.0576`，说明 eval filter margin 不是主要瓶颈。

`exp029_randomized_terrain_hold_reward_safe` 已完成：

- final eval：dmax ratio `0.1439`、success `0.8262`、collision `0.0557`、timeout `0.1221`。
- 加强 safety reward/filter 权重没有压低真实碰撞，反而牺牲 success/timeout。

`exp030_randomized_terrain_control_safety` 已完成：

- 回到 exp028 主体，在低层控制中加入相对速度 safety projection 和 success-zone damping。
- final eval：dmax ratio `0.1528`、success `0.8330`、collision `0.0313`、timeout `0.1357`。
- 候选 `ppo_timestep_010240.pt` eval collision `0.0234`，接近 strict `0.02`，说明动态控制投影方向有效。
- 但 success/timeout 明显退化，`control_safety_applied_fraction=0.1610`、`linear_scale_min=0.25`，说明当前投影触发过早/过强。

`exp031_randomized_terrain_narrow_control_safety` 已完成：

- final eval：dmax ratio `0.1448`、success `0.8105`、collision `0.0449`、timeout `0.1455`。
- 简单缩小投影范围和降低强度没有恢复 success/timeout，也丢失 exp030 的安全收益。

`exp032_randomized_terrain_closing_control_safety` 已完成：

- final eval：dmax ratio `0.1495`、success `0.8379`、collision `0.0361`、timeout `0.1279`。
- closing-only 投影条件比 exp031 略好，但仍没有回到 exp028/exp030 的综合水平。

`exp033_randomized_terrain_directional_control_safety` 已完成：

- final eval：dmax ratio `0.1436`、success `0.8154`、collision `0.0488`、timeout `0.1387`。
- directional agent-scale 没有带来安全收益，说明连续缩放仍然过粗。

`exp034_randomized_terrain_directional_mask_control_safety` 已完成：

- final eval：dmax ratio `0.1491`、success `0.8828`、collision `0.0361`、timeout `0.0840`。
- directional mask 明显恢复 success/timeout，但 collision 仍未达 strict。

`exp035_randomized_terrain_directional_mask_buffer` 已完成：

- final eval：dmax ratio `0.1519`、success `0.9072`、collision `0.0127`、timeout `0.0811`。
- success 和 collision 首次同时通过 strict 门槛，剩余瓶颈转为 timeout/hold。

`exp036_randomized_terrain_directional_mask_timeout_hold` 已完成：

- final eval：dmax ratio `0.1523`、success `0.9336`、collision `0.0088`、timeout `0.0586`。
- stronger hold/timeout shaping 继续改善 timeout，但不能把 timeout 降到 0。

`exp037_randomized_terrain_directional_mask_timeout260` 已完成：

- final eval：dmax ratio `0.1517`、success `0.9238`、collision `0.0352`、timeout `0.0410`。
- 延长 episode/eval 到 260 steps 能降低 timeout，但也给末段碰撞更多暴露机会，collision 反弹。

`exp038_randomized_terrain_success_zone_stabilizer` 已完成：

- 修正 strict rank 后重选 `ppo_timestep_009216.pt` 为 `best.pt`。
- final eval：dmax ratio `0.1590`、success `0.9756`、collision `0.0137`、timeout `0.0107`。
- dmax/success/collision 均达 strict，唯一失败项是 timeout。
- timeout 子集主要卡在最近邻安全间距：final nearest mean 约 `0.400 m`，低于 `success_thresholds.min_pairwise_distance=0.42 m`，不是整体 dmax/dispersion 不可达。

`exp039_randomized_terrain_hard_near_stabilizer` 已完成诊断复评：

- 在 exp038 best 上使用 exp039 配置 eval：success `0.9424`、collision `0.0254`、timeout `0.0322`。
- hard near stabilizer 差于 exp038，不建议按原样长训。

`exp040_randomized_terrain_soft_hold_stabilizer` 已完成诊断复评：

- 在 exp038 best 上使用 exp040 配置 eval：success `0.9658`、collision `0.0186`、timeout `0.0166`。
- collision 仍可过 strict，但 timeout 差于 exp038，不建议按原样长训。

`exp041_randomized_terrain_hold_zone_override` 已完成诊断复评和 smoke：

- 在 exp038 best 上使用 exp041 配置 eval：success `0.9795`、collision `0.0107`、timeout `0.0098`。
- CPU/CUDA smoke 已通过。
- 结果略优于 exp038，但仍只是 checkpoint 复评；当前暂停继续长训，不启动 exp041。

`exp042_structured_actor_bicycle_quintic_probe` 已完成环境工程探针：

- 配置：`configs/experiment/exp042_structured_actor_bicycle_quintic_probe.yaml`。
- Actor 使用 `branched_v1`：ego `10->32`、neighbor `21->48`、terrain `50->64`、aggregation `5->16`，concat `160` 后接 `128->128->2` 共享主干。
- Critic 使用 `structured_v1`：agent states `4x8` 经共享 encoder 后 mean+max 聚合，team stats、terrain summary 和 oracle state 分支后接 value trunk。
- Proxy 动力学使用 `low_level_control.kinematic_model=bicycle`，轨迹生成使用 `trajectory_generator.geometry_method=quintic`。
- 当前地图设置：`safety.world_xy_limit=12.5`、`terrain.crater_field_size=25.0`，对应 `25 m × 25 m` 训练区域。
- 当前通信设置：`observation.communication_radius=0.0`，表示所有非自身 rover 可见。Actor 地形网格仍保持 `5×5×2=50` 维，暂不扩大感知面积。
- CPU smoke：`8 env / 8 timesteps` 通过。
- CUDA smoke：`256 env / 64 timesteps / rollout 32` 通过；一个 optimizer、两次 joint update、terrain branch 权重更新 `0.1263`、动作非退化。

`exp043_structured_bicycle_quintic_map25_long` 已完成直接长训：

- 配置：`configs/experiment/exp043_structured_bicycle_quintic_map25_long.yaml`。
- 保持 `branched_v1` Actor、`structured_v1` Critic、`bicycle` proxy、`quintic` 轨迹、Actor/Critic `86/54` 接口和 `communication_radius=0.0`。
- 继承 exp041 的 hold-zone override：`hold_zone_spacing_weight=8.0`、`hold_zone_pairwise_distance=0.58`、`hold_zone_override_after_warmup=true`。
- 训练分布扩大：`initial_state.spawn_radius_min/max=4.5/6.5`、`center_xy_range=3.0`、`jitter_std=0.45`。
- 地形密度相对 exp042 probe 提高：`crater_count=48`、`crater_field_size=25.0`、`random_translation_m=5.0`。
- MAPPO：2048 env、rollout 64、`20480` timesteps（约 `41,943,040` env steps）、checkpoint interval 1024、`success_progress_long` selection。
- 结果：best 为 `ppo_timestep_020480.pt`，final eval dmax ratio `0.8596`、success `0.0`、collision `0.0`、timeout `1.0`。
- 判读：工程链路正常、参数和 terrain branch 均更新，但扩大后的初始分布让 pure RL 几乎没有学到集合进度；下一步优先做 initial-state curriculum，而不是单纯继续加训练预算。

`exp044_structured_bicycle_quintic_map25_curriculum` 已完成：

- 配置：`configs/experiment/exp044_structured_bicycle_quintic_map25_curriculum.yaml`。
- 保持 `branched_v1` Actor、`structured_v1` Critic、`bicycle` proxy、`quintic` 轨迹、Actor/Critic `86/54` 接口、`25 m × 25 m` 地图和 `communication_radius=0.0`。
- reset 目标难度改为 `spawn_radius_min/max=3.8/5.2`、`center_xy_range=2.0`、`jitter_std=0.40`。
- initial-state curriculum：训练前 `4096` timesteps 使用 `3.0/4.0` spawn radius、`center_xy_range=1.0`、`jitter_std=0.35`，随后 `8192` timesteps 线性 ramp 到目标分布；独立 eval 不使用课程 override，仍在目标难度上判定。
- 地形密度从 exp043 的 `crater_count=48` 回调到 `36`，避免在 25m 地图与 bicycle/quintic 叠加时地形阻力过早主导。
- 探索参数较 exp043 收敛一点：`initial_log_std=-1.1`、entropy `0.0015 -> 0.0003` over `8192` timesteps。
- MAPPO：2048 env、rollout 64、`20480` timesteps（约 `41,943,040` env steps）、checkpoint interval 1024、`success_progress_long` selection。
- 结果：best 为 `ppo_timestep_020480.pt`，final eval dmax ratio `0.4796`、success `0.0`、collision `0.00195`、timeout `0.9980`。
- 判读：相对 exp043 明显改善靠拢距离，但全部候选 checkpoint 的 success 仍为 `0`；下一步应先做 local-success bootstrap。

`exp045_structured_bicycle_quintic_map25_local_success_bootstrap` 已完成：

- 配置：`configs/experiment/exp045_structured_bicycle_quintic_map25_local_success_bootstrap.yaml`。
- 保持 `branched_v1` Actor、`structured_v1` Critic、`bicycle` proxy、`quintic` 轨迹、Actor/Critic `86/54` 接口、`25 m × 25 m` 地图和 `communication_radius=0.0`。
- reset 目标难度缩小到 `spawn_radius_min/max=2.4/3.4`、`center_xy_range=1.0`、`jitter_std=0.25`；课程起点为 `1.6/2.4`、`center_xy_range=0.5`。
- 动作/低层略放大：`rho_max=1.6`、`beta_max=60°`、`max_steer_angle≈45°`、`reference_speed=0.9`。
- reward 临时偏向中距离集合：`dmax_progress=5.5`、`dispersion_progress=2.4`、`oracle_mean_distance_progress=3.0`，terrain weight 降到 `0.20`。
- filter 仍保留 safety/path 作用，但 apply probability 和 score scale 降低，避免早期过度替换集合意图。
- MAPPO：2048 env、rollout 64、`20480` timesteps（约 `41,943,040` env steps）、checkpoint interval 1024、`success_progress_long` selection。
- 结果：best 为 `ppo_timestep_020480.pt`，final eval dmax ratio `0.2734`、success `0.1846`、collision `0.0`、timeout `0.8174`。
- 判读：local bootstrap 有效，但多数 episode 停在 success 区外侧；下一步需要末端释放与 dmax/dispersion 收缩。

`exp046_structured_bicycle_quintic_map25_local_hold_release` 已完成：

- 配置：`configs/experiment/exp046_structured_bicycle_quintic_map25_local_hold_release.yaml`。
- 保持 `branched_v1` Actor、`structured_v1` Critic、`bicycle` proxy、`quintic` 轨迹、Actor/Critic `86/54` 接口、`25 m × 25 m` 地图和 `communication_radius=0.0`。
- reset 分布沿用 exp045：目标 `2.4–3.4 m`，课程起点 `1.6–2.4 m`。
- 降低 filter 介入：`apply_probability_end=0.22`、`score_scale_end=0.35`。
- 降低 control safety 阻尼：activation distance `0.68`、projection strength `0.70`、min linear scale `0.40`。
- 增强末端集合：`dmax_progress=7.0`、`dispersion_progress=3.2`、`success_bonus=85`、`timeout_penalty=45`，terrain weight 降到 `0.15`。
- MAPPO：2048 env、rollout 64、`20480` timesteps（约 `41,943,040` env steps）、checkpoint interval 1024、`success_progress_long` selection。
- 结果：best 为 `ppo_timestep_015360.pt`，final eval dmax ratio `0.2424`、success `0.6123`、collision `0.0`、timeout `0.3877`。
- 判读：local terminal release 有效，但未 strict；失败样本仍停在 success 区外，下一轮应继续释放末端阻尼并加强收缩/timeout。

`exp047_structured_bicycle_quintic_map25_terminal_convergence` 已完成：

- 配置：`configs/experiment/exp047_structured_bicycle_quintic_map25_terminal_convergence.yaml`。
- reset 分布保持 exp046：目标 `2.4–3.4 m`，课程起点 `1.6–2.4 m`。
- filter 继续弱化：`apply_probability_end=0.16`、`score_scale_end=0.28`，并把 hold-zone 安全距离降到 `0.48 m` 附近。
- control safety 继续释放：activation distance `0.62`、projection strength `0.50`、min linear scale `0.55`、success-zone linear scale `0.80`。
- reward 更偏 terminal convergence：`dmax_progress=9.0`、`dispersion_progress=4.5`、`success_bonus=115`、`timeout_penalty=65`，terrain weight 降到 `0.12`。
- MAPPO：2048 env、rollout 64、`20480` timesteps（约 `41,943,040` env steps）、checkpoint interval 1024、`success_progress_long` selection。
- 结果：best 为 `ppo_timestep_015360.pt`，final eval dmax ratio `0.2132`、success `0.7188`、collision `0.0059`、timeout `0.2764`。
- 判读：接近 dmax strict，但 timeout episode 仍停在成功区外且速度低；下一轮应提高 terminal drive 与 dispersion 收缩，而不是只延长 episode。

`exp048_structured_bicycle_quintic_map25_terminal_drive` 已完成：

- 配置：`configs/experiment/exp048_structured_bicycle_quintic_map25_terminal_drive.yaml`。
- reset 分布保持 exp047：目标 `2.4–3.4 m`，课程起点 `1.6–2.4 m`。
- filter 小步调整：`apply_probability_end=0.18`、`score_scale_end=0.30`、`hold_zone_pairwise_distance=0.46`，保留 collision override。
- terminal drive：`reference_speed=1.15`、`max_linear_speed=1.35`、`projection_min_linear_scale=0.65`、`success_zone_linear_scale=0.95`。
- reward：`dmax_progress=9.5`、`dispersion_progress=6.0`、`success_bonus=130`、`timeout_penalty=80`，terrain weight 降到 `0.10`。
- MAPPO：2048 env、rollout 64、`20480` timesteps（约 `41,943,040` env steps）、checkpoint interval 1024、`success_progress_long` selection。
- 结果：best 为 `ppo_timestep_008192.pt`，final eval dmax ratio `0.1866`、success `0.9844`、collision `0.0020`、timeout `0.0137`。
- 判读：dmax/success/collision 已全部通过 strict；剩余 timeout episode 几何已经合格，但最近邻间距约 `0.393 m`，低于 `0.42 m` 成功安全阈值。

`exp049_structured_bicycle_quintic_map25_terminal_spacing` 已完成：

- 配置：`configs/experiment/exp049_structured_bicycle_quintic_map25_terminal_spacing.yaml`。
- reset 分布保持 exp048：目标 `2.4–3.4 m`，课程起点 `1.6–2.4 m`。
- terminal spacing：`hold_zone_pairwise_distance=0.52`、`hold_zone_spacing_weight=4.60`、`endpoint_safe_distance=0.44`、`path_safe_distance=0.32`。
- control safety 轻量回收：`projection_activation_distance=0.64`、`projection_strength=0.55`、`projection_min_linear_scale=0.58`、success-zone scale `0.88`。
- reward：`near_distance=3.4`、`dispersion_progress=6.2`、`timeout_penalty=90`、`success_bonus=135`。
- MAPPO：2048 env、rollout 64、`20480` timesteps（约 `41,943,040` env steps）、checkpoint interval 1024、`success_progress_long` selection。
- 结果：best 为 `ppo_timestep_010240.pt`，final eval dmax ratio `0.1884`、success `0.8926`、collision `0.0010`、timeout `0.1064`。
- 判读：collision 更低但 success/timeout 明显退化；timeout episode 的最近邻安全性已大多满足，但 dmax/dispersion 反而变差，说明全局 spacing 修正扰动了末段几何收缩。

当前下一轮方向：

- 以 exp048 作为当前新环境栈 local reset 最佳 candidate。
- 不再全局加硬 near/hold/spacing filter；exp039/exp040 和 exp049 均说明会扰动 success 或 timeout。
- 下一轮应回退 exp048 主体，只加入窄触发 terminal spacing：仅在 dmax/dispersion 已接近成功、且最近邻落入 `0.28–0.42 m` 灰区时生效。

## Checkpoint 复评计划

候选 checkpoint 使用统一入口：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/<experiment>/<run_id>
```

目标产物：

```text
metrics/final_eval_proxy.json
metrics/checkpoint_status.json
physx/metrics/<terrain>_headless.json
run_manifest.json
```

其中 PhysX 评估只在配置允许且 proxy gate 满足时触发。

## 暂缓的诊断线

`exp009` 只作为 strong terrain 诊断实验：

```text
outputs/runs/exp_009_terrain3d_strong/_suite/
```

已知结论：

- seed23 通过 strict gate。
- seed31 未通过 success/timeout gate。
- seed47 未运行，因为 3-seed strict acceptance 已经不可能成立。

seed31 当前失败模式：

```text
dmax_reduction_ratio: 0.1819  # 通过
success_rate: 0.8740          # 未通过
collision_rate: 0.0049        # 通过
timeout_rate: 0.1250          # 未通过
```

目前先不继续 seed31 失败 episode 回放、动作表示原型或 terrain curriculum 短 run。后续恢复训练研究时，再优先区分 timeout 来自地形速度缩放、dispersion 不稳定、speed hold 不稳定或 `[rho, beta]` 表达能力不足。

## 工程验收

近期验收包含：

1. `.venv_isaaclab/bin/python -m pytest -q -ra`。
2. `scripts/validate_first_stage.py --config configs/experiment/exp_001_minimal.yaml --device cpu --steps 32`。
3. `scripts/run_checkpoint_evaluation.py` 在 mock 或真实 run 上生成 `checkpoint_status.json`。
4. `scripts/evaluate_physx_jackal_tracking.py` 的 headless/render sanity 路径可手动运行，并记录产物路径。

如果新增工程 smoke run，应优先写入 `outputs/runs/<experiment>/<run_id>/` 并更新 runbook；不要把 partial train、GIF 或 TensorBoard 曲线写成 strict success。
