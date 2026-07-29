# 接口规范

## Actor 观测

形状：`(num_envs, 4, obs_dim)`。

默认 Actor observation 包含自车状态、邻居共享状态、地形手工特征和局部聚合特征，不包含 `p*`、oracle 距离或 oracle 距离缩减量。显式执行目标 schema 可追加不含世界坐标的车体系相对目标特征。

当前 observation schema：

```text
schema_version: ego_v3_local_terrain_grid
communication_radius: cfg.observation.communication_radius
ego_dim: 10
neighbor_dim: 7
terrain_dim: 50
aggregation_dim: 5
actor_obs_dim: 86
```

`observation.communication_radius` 是当前唯一允许从 experiment YAML 覆盖的 observation 字段。取值 `>0` 时表示有限通信半径；取值 `<=0` 时表示临时取消通信距离限制，所有非自身 rover 均视为可见。`max_neighbors`、`ego_dim`、`neighbor_dim`、`terrain_dim` 和 `aggregation_dim` 会改变模型输入接口，本轮不开放配置覆盖。

`ego_v5_gather_site_goal`（89 维）在基础 86 维后追加到公共搜索点的 `[local_dx, local_dy, normalized_distance]`。`ego_v6_gather_slot_goal`（同为 89 维）追加到 rover 专属对称槽位的同一三元组；默认槽位由环境在 reset 时围绕 `oracle_point` 等角生成，并枚举最小总初始行驶距离的分配。v6 每个 rover 的目标不同，但全部槽位的平均位置严格等于真实搜索点。默认关闭的 `task.dynamic_terminal_slot_goal_*` 可在近末段实际质心未通过完整平整度 gate 时，把 Actor 的下一步 v6/v7 槽位特征换为当前质心附近真实平整候选的最小行程分配；维度、Critic、固定 reward 槽位和 success predicate 都不变。`ego_v7_gather_site_and_slot_goal`（92 维）按顺序拼接公共点三元组和槽位三元组，是与 `branched_v4` 配对的诊断 schema。三种 schema 都必须与 `task.explicit_goal_in_execution=true` 成对出现；`task.execution_slot_reward_target=true` 只允许 v6/v7，并使 oracle-progress reward 接收 `[env, agent, 3]` 槽位目标而非共享 `[env, 3]` 搜索点。Critic 保持 54 维，字段均不包含全球 XY、搜索 score、可行性或平整度诊断；实际质心平整度和 success gate 不受该 reward 开关影响。

ego 10 维顺序：

```text
x, y, z, cos(yaw), sin(yaw), vx, vy, angular_velocity, speed_xy, abs_angular_velocity
```

ego 最后两维曾经是固定零占位；现在是运动派生特征。当前整体 Actor 输入已因地形网格从 41 维升级为 86 维，checkpoint metadata 会记录 schema 和输入维度。

地形 50 维来自车体坐标系下的固定 `5×5` 网格：

```text
forward_x_m: [-0.4, 0.0, 0.4, 0.8, 1.2]
lateral_y_m: [-0.8, -0.4, 0.0, 0.4, 0.8]
channels: [relative_height, risk]
flatten_order: x -> y -> channel
```

- `relative_height = height(sample) - height(rover)`。
- `risk = 1 - traversability`，范围为 `[0, 1]`。
- 平地网格全部为 0。
- 网格随 rover yaw 旋转到世界坐标采样，因此策略看到的是稳定的车体系局部地形。
- 当前网格布局固定在代码中，不开放 YAML 修改。
- 地图面积与局部地形观测窗口是两层设置：`safety.world_xy_limit=12.5` / `terrain.crater_field_size=25.0` 可把训练区域扩大为 `25 m × 25 m`，但不会改变 Actor 的 `5×5×2=50` 维局部地形输入；如需扩大地形感知范围，需要另开 observation schema 和网络切片改造。

## 地形 episode 随机化

`terrain.randomize_per_reset=true` 时，每个并行环境在 episode reset 时独立重采样：

```text
translation_xy
yaw
phase
amplitude_scale
crater_radius_scale
crater_depth_scale
```

- 地图在同一个 episode 内保持固定，不在每个 control step 跳变。
- 只 reset 部分环境时，仅更新对应环境的 terrain runtime。
- 地形高度、traversability、速度缩放、Actor 局部网格、Critic 摘要和可视化共用同一份 runtime。
- 历史配置默认 `randomize_per_reset=false`，继续使用固定地图。
- 随机化不改变 observation schema，Actor / Critic 维度仍为 `86 / 54`。

## Episode 固定的最优集合点搜索

训练期 oracle 集合点不再使用几何中点或几何中位数直接代理。每个环境在 reset 时依次完成地形随机化、初始位置采样和一次 `terrain_aware_multiresolution` 搜索；搜索得到的 `oracle_point` 与该环境的 terrain runtime 在整个 episode 内保持固定，只有对应环境再次 reset 时才重算。默认该点只供 centralized critic、oracle reward 和评估使用；`ego_v5/v6` 显式契约只将其导出的车体系局部目标三元组提供给 Actor。

默认搜索先在“初始 rover 包围盒 + `search_margin`”内构造 `9×9` 粗网格，并加入几何中位数与质心作为种子；随后围绕当前最优点执行两层 `5×5` 局部细化。局部搜索边界会预留完整平整度圆盘，并限制在 `safety.world_xy_limit` 内。

若局部搜索没有找到可行候选，且 `global_fallback_enabled=true`，搜索不会立即接受局部退化点，而是扩展到预留完整圆盘后的全部世界边界。全局阶段对 `33×33` 粗网格逐点执行完整圆盘、距离、路径风险和路径高差评估，按“可行候选优先、目标函数最小”保留 32 个候选组成 beam；随后围绕 beam 执行两层 `5×5` 多分辨率细化。全局回退只处理局部失败的环境，并按 `global_max_envs_per_batch=8` 分批，以限制峰值显存。

每个候选点 \(q\) 使用世界系对称圆盘评估地形，默认定义为：

```text
flatness_radius: 0.75 m
flatness_rings: 3
flatness_samples_per_ring: 12
sample_count: 1 + 3 * 12 = 37
max_height_range: 0.18 m
max_slope: 0.25
```

37 个采样点包括圆心和 3 个等间距同心圆。`max_slope` 表示高度梯度模长
\(\max\sqrt{(\partial h/\partial x)^2+(\partial h/\partial y)^2}\)，是无量纲 rise/run；`0.25` 对应约 \(14.0^\circ\) 坡角。候选仅在完整圆盘同时满足 `height_range <= 0.18 m` 和 `max_slope <= 0.25` 时可行。

`robustness_radius=0`（默认）只检查候选中心。正值则把搜索可行性提升为“可执行平整盆地”：对中心及 `robustness_samples` 个等角偏移后的完整 37 点圆盘逐一评估，所有偏移均通过才视为可行；`height_range/max_slope` telemetry 取该包络中的最坏值。该设置只让 reset 时的 oracle 搜索更保守，不替代也不放宽运行时对**实际团队质心**的同一 37 点 hard gate；正半径要求至少 4 个偏移样本。

在可行候选中最小化：

\[
J(q)=
w_{\mathrm{mean}}\,\overline d(q)
+w_{\mathrm{max}}\,d_{\max}(q)
+w_{\mathrm{risk}}\,C_{\mathrm{path\_risk}}(q)
+w_{\Delta h}\,C_{\mathrm{path\_height}}(q)
+w_{\mathrm{flat}}\,C_{\mathrm{flat}}(q).
\]

其中前两项分别是全部 rover 到候选点的平均距离和最远距离；路径风险与路径高差沿各 rover 到候选点的直线采样；平整度代价由归一化高度范围和最大梯度组成。可行候选始终优先于不可行候选。若全局回退完成后仍没有可行点（或显式关闭全局回退且局部搜索无解），则返回“目标函数 + 平整度违反量惩罚”最小的有限退化点并设置 `feasible=false`。该退化点只保留接口与诊断连续性，不得解释为满足地形约束的最优集合点；对应环境的 oracle 距离进度 shaping 关闭，即 oracle reward 项为 0。

默认示例配置为：

```yaml
gather_point:
  search_method: terrain_aware_multiresolution
  coarse_grid_size: 9
  refinement_grid_size: 5
  refinement_levels: 2
  search_margin: 1.5
  global_fallback_enabled: true
  global_grid_size: 33
  global_beam_width: 32
  global_refinement_levels: 2
  global_max_envs_per_batch: 8
  flatness_radius: 0.75
  flatness_rings: 3
  flatness_samples_per_ring: 12
  max_height_range: 0.18
  max_slope: 0.25
  robustness_radius: 0.0  # > 0: all sampled centroid offsets must be flat
  robustness_samples: 8
  mean_distance_weight: 1.0
  max_distance_weight: 0.25
  path_risk_weight: 0.75
  path_height_change_weight: 0.25
  flatness_weight: 0.25
  path_samples: 5
  infeasible_penalty: 1000.0
  max_envs_per_batch: 64
  require_flat_for_success: true
  execution_slot_radius: 0.35
```

`oracle_search` telemetry 记录 `method/objective/feasible/mean_distance/max_distance/path_risk/path_height_change/height_range/max_slope`。其中 `feasible=false` 同时表示该 episode 不使用 oracle 距离进度 shaping。搜索改变的是 oracle 的数值语义和任务成功语义；默认接口为 `86 / 54`，而显式 v5/v6 执行契约为 `89 / 54`。

## 初始队形分布

训练 reset 的初始队形可通过 `initial_state` 配置覆盖，默认保持旧行为：

```text
spawn_radius_min: 3.0
spawn_radius_max: 4.0
center_xy_range: 1.0
jitter_std: 0.35
```

该配置只改变 episode 初始状态采样，不改变 Actor observation、Critic state、checkpoint schema 或动作接口。`exp043` 使用更大的初始半径和中心采样范围，以匹配 `25 m × 25 m` 地图。

`initial_state.curriculum_enabled=true` 时，训练可从 `curriculum_start_*` 起始分布按 `curriculum_warmup_timesteps / curriculum_ramp_timesteps` 线性过渡到目标分布。课程只在训练脚本显式设置 `progress_timestep_override` 时生效；独立 eval 默认使用目标分布，避免用课程早期简单分布判定 strict。

## Critic 状态

形状：`(num_envs, state_dim)`。

Critic state 包含全部 rover 真值状态、队形几何信息、地形摘要和仅训练使用的 oracle 特征。

Critic 总维度仍为 54。5 维地形摘要依次为：

```text
mean_abs_relative_height
max_rise
max_descent
mean_risk
max_risk
```

## Checkpoint 兼容性

Checkpoint metadata 必须包含：

```text
observation_schema_version
actor_obs_dim
critic_state_dim
actor_architecture
critic_architecture
observation_slices
kinematic_model
trajectory_geometry_method
```

加载时 schema 和维度必须与当前配置完全一致。若当前配置显式要求 `branched_v1` / `structured_v1`，checkpoint 中的架构 metadata 也必须匹配；缺少架构 metadata 的当前 schema checkpoint 按旧 `mlp_v1` 解释，可继续用旧 MLP 配置播放和评估。旧 `ego_v2_speed_angular` checkpoint、缺少 schema metadata 的 checkpoint 和错误输入维度 checkpoint 均明确拒绝，不做补零或自动迁移。

地形约束搜索和 centroid 平整度成功门不改变 `86 / 54` 张量接口，因此旧 checkpoint 可能仍可结构化加载；但 oracle 特征值、reward shaping 和 success/timeout 语义已经变化。采用旧纯几何成功门得到的历史 `success_rate`、`timeout_rate` 或 strict 结论，不能与新语义结果直接横向比较。需要比较时，必须在同一 terrain runtime、同一搜索配置和同一平整度 gate 下重新评估所有 checkpoint。

## 网络架构

`algorithm.actor_architecture` 支持：

```text
mlp_v1
branched_v1
```

`mlp_v1` 是旧兼容路径：`86 -> 128 -> 128 -> 2`。`branched_v1` 保持 Actor 输入 86 和输出 `[rho, beta]` 不变，但按固定切片编码：

```text
ego:          [0:10]   10 -> 32
neighbors:   [10:31]  21 -> 48
terrain:     [31:81]  50 -> 64
aggregation: [81:86]   5 -> 16
concat: 160 -> shared trunk 128 -> 128 -> 2 -> tanh
```

`algorithm.critic_architecture` 支持：

```text
mlp_v1
structured_v1
```

`mlp_v1` 是旧兼容路径：`54 -> 128 -> 128 -> 1`。`structured_v1` 保持 Critic state 54 不变，固定拆分为：

```text
agent states:  [0:32]  4x8，经共享 agent encoder 后 mean + max 聚合
team stats:    [32:40] 8 维分支
terrain:       [40:45] 5 维分支
oracle:        [45:54] 9 维分支
concat -> value trunk 128 -> 128 -> 1
```

## 动作

形状：`(num_envs, 4, 2)`。

归一化 action 被映射为：

- `rho in [0, rho_max]`
- `beta in [-beta_max, beta_max]`

`planner.subgoal_filter` 是可选 proxy planner 后处理，默认关闭。启用时，它在 Actor 输出 `[rho, beta]` 之后、轨迹生成之前，从固定候选子目标中按地形路径风险和 endpoint safety 选择 filtered subgoal。该机制不改变 Actor 输出维度、Critic 状态维度或 checkpoint schema；checkpoint metadata 会记录 filter 配置摘要。

当前支持六种 mode：

- `terrain_safe_candidate`：exp020 使用的 hard filter，每步执行 score 最低的候选。
- `terrain_safe_candidate_curriculum`：exp021 使用的课程化 filter，训练时按 `warmup_timesteps` / `ramp_timesteps` 逐步提高 `apply_probability` 和 `score_scale`；评估时读取 checkpoint metadata 中的 `timesteps` 固定课程进度，并使用 deterministic rule，只有 filtered score 比 raw score 至少好 `deterministic_improvement_margin` 时才替换。
- `terrain_safe_candidate_constrained_curriculum`：exp022 使用的课程化安全约束 filter，在 exp021 课程 schedule 基础上加入 endpoint/path safety constraint、visible-neighbor center progress constraint 和 warmup 后 safety override；仍只使用通信半径内可见邻居，不使用 oracle。
- `terrain_safe_candidate_soft_progress_curriculum`：exp023 使用的软进度保护 filter，保留课程 schedule，但移除 hard safety constraint 和 near-distance override；score 中加入 visible-neighbor center / center-progress 软惩罚，只在 raw endpoint/path 预测碰撞且候选可降低碰撞违反时允许 collision override。
- `terrain_safe_candidate_mutual_progress_curriculum`：exp024 使用的 mutual path safety filter，在 exp023 基础上把可见邻居 raw subgoal path 作为动态障碍，按相同时间采样比较候选路径与邻居 raw path；仍不使用不可见 rover 或 oracle。
- `terrain_safe_candidate_hold_progress_curriculum`：exp026 之后使用的 hold-stable filter，在 mutual path safety 基础上增加默认关闭的 hold-zone cost；当当前队形已经接近 success gate 时，score 额外偏好较短 rho 和更大的 endpoint pairwise buffer，减少末段过冲和相向冲入。exp041 额外启用默认关闭的 `hold_zone_override_after_warmup`，仅当 raw action 会破坏 hold-zone spacing 且候选 action 明确改善 spacing 时才覆盖 raw action。该模式仍只使用当前可见邻居和队形几何，不向 Actor 输入 oracle，也不改变 `86 / 54` 接口。

`action_filter` telemetry 至少包含：

```text
raw_path_terrain_risk_mean
filtered_path_terrain_risk_mean
path_terrain_risk_reduction
subgoal_deviation
suggested_subgoal_deviation
raw_score
filtered_score
score_margin
applied
deterministic_applied
endpoint_near_violation
endpoint_collision_violation
path_near_violation
path_collision_violation
mutual_path_near_violation
mutual_path_collision_violation
raw_endpoint_near_violation
raw_endpoint_collision_violation
raw_path_near_violation
raw_path_collision_violation
raw_mutual_path_near_violation
raw_mutual_path_collision_violation
candidate_feasible
feasible_fraction
safety_override
safety_override_fraction
collision_override
collision_override_fraction
raw_visible_center_cost
filtered_visible_center_cost
suggested_visible_center_cost
center_progress_regression
hold_zone_activation
hold_zone_rho_cost
hold_zone_spacing_violation
raw_hold_zone_rho_cost
raw_hold_zone_spacing_violation
hold_zone_override
hold_zone_override_fraction
candidate_index
suggested_candidate_index
schedule_progress_step
apply_probability
score_scale
```

## 第一阶段动力学

当前 rover 仍是 torch-vectorized proxy 状态模型，不是 Isaac Sim articulation。`low_level_control.kinematic_model` 支持 `unicycle` 和 `bicycle`；旧配置默认 `unicycle`，新工程探针 `exp042` 显式使用 `bicycle`。

`bicycle` v1 仍使用现有控制器输出的期望 `linear` 和期望 yaw-rate，但在 `_integrate()` 中将 yaw-rate demand 转成转角：

```text
steer = atan(wheelbase_m * omega_cmd / max(v_cmd, eps))
steer = clamp(steer, +/- max_steer_angle_rad)
yaw_rate_actual = v_eff / wheelbase_m * tan(steer)
```

其中 `v_eff` 是经 terrain speed scaling 后的前向速度。位置/yaw 使用半隐式 midpoint heading 更新；`angular_velocities` 记录实际 bicycle yaw-rate。新增 `kinematics` telemetry 包含：

```text
kinematic_model
steering_angle
actual_yaw_rate
turning_radius
```

只有在 rover 资产和控制接口明确后，才应替换为真实 Isaac Sim articulation。

## 轨迹生成

`trajectory_generator.geometry_method` 支持：

```text
line
quintic
```

`line` 是旧兼容路径。`quintic` 使用 2D quintic Hermite 曲线：起点为 rover 当前 `xy`，起点切向为当前 yaw，终点为 decoded / filtered subgoal `xy`，终点切向默认指向 subgoal 方向，起终点二阶导为 0。`Trajectory.points/headings/timestamps/reference_speed` 接口不变。

## Control safety

`low_level_control` 支持默认关闭的 control safety projection。该机制位于 `compute_control()` 之后、proxy `_integrate()` 之前，只缩放本步线速度，不改变 Actor 输出 `[rho, beta]`、子目标、轨迹生成接口、Actor 86 维 observation、Critic 54 维 state 或 checkpoint schema。exp030 之后部分实验启用的主要字段为：

```text
safety_projection_enabled
projection_activation_distance
projection_stop_distance
projection_horizon_s
projection_strength
projection_min_linear_scale
projection_damp_nonclosing_near
projection_directional_agent_scale
projection_directional_agent_scale_mode
success_zone_damping_enabled
success_zone_dmax_multiplier
success_zone_dispersion_multiplier
success_zone_linear_scale
```

`control_safety` telemetry 包含：

```text
enabled
linear_scale
raw_linear
projected_linear
applied
pairwise_risk
predicted_nearest_distance
success_zone_active
control_safety_applied_fraction
control_safety_linear_scale_mean
control_safety_linear_scale_min
control_safety_pairwise_risk_mean
control_safety_predicted_nearest_mean
control_safety_success_zone_fraction
```

## 末端共同质心校正与槽位捕获

`low_level_control.formation_center_correction_*` 默认关闭。它位于子目标过滤之后、轨迹生成与 control safety projection 之前；当上一状态同时满足给定的 dmax 和 dispersion 触发倍数时，以已分配固定槽位的均值（严格等于 terrain-aware 搜索点）减去实际团队质心，裁剪到 `formation_center_correction_max_offset` 后乘以 gain，并把同一个世界系二维偏移加到所有 rover 子目标。故相对槽位、两两间距、Actor `[rho,beta]`、Actor/Critic 输入和真实成功门均不变；它只纠正整个队形相对真实搜索点的共同漂移。`formation_center_correction_require_flatness_failure=true` 时，校正还必须满足上一状态的**实际质心圆盘**未通过平整度判定；这使该控制只服务于真实地形平整度失败，而不会把 oracle 搜索点或几何中点当作成功代理。

```text
formation_center_correction_enabled
formation_center_activation_dmax_multiplier >= 1
formation_center_activation_dispersion_multiplier >= 1
formation_center_correction_max_offset >= 0
formation_center_correction_gain in [0, 1]
formation_center_correction_require_flatness_failure: bool
```

`terminal_slot_capture_*` 也是默认关闭的独立实验开关。它在同样的末端几何触发下，将每个 rover 的子目标按 `blend` 线性拉向**该 rover 自己的**固定槽位，不会把所有目标替换为几何中点；后续 control safety 仍会执行。其字段为 `terminal_slot_capture_enabled`、两个 activation multiplier 与 `terminal_slot_capture_blend in [0,1]`。exp081/083 的后验对照显示该机制当前不优于单独的共同质心校正，故不是推荐默认。

`StepOutput.info` 记录 `formation_center_correction.active/offset_xy`、`terminal_slot_capture.active`、`flat_geometry_capture.active` 与 `dynamic_terminal_slot_goal.active`；训练与独立评测汇总为相应的 active fraction，前者额外记录 offset mean/max。

## 兼容 checkpoint 的 warm-start

`scripts/train_skrl_mappo.py --init-checkpoint <path>` 可从架构、schema 与观测维度均兼容的 checkpoint 初始化共享 Actor/Critic 参数。这是 warm-start，不是 resume：optimizer state、rollout memory 与训练步数全部重新开始；若未显式指定 `--bc-updates`，会自动设为 `0`，避免 BC 覆盖初始化策略。训练器会额外保存 `ppo_timestep_000000.pt` 并参与同一候选筛选，因此 PPO 更新退化时会如实保留初始化策略，而不会把它伪装成已微调的最优 checkpoint。

## Success / safety gate

每一步状态推进后，以实际团队质心而不是 oracle 点作为已发生集合位置：

```text
actual_gather_point_xy = metrics.centroid[..., :2]
```

环境在该质心周围使用与 oracle 候选相同的世界系圆盘定义进行 37 点平整度评估：半径 `0.75 m`、3 圈、每圈 12 点。实际集合区域必须同时满足 `height_range <= 0.18 m` 和 `max_slope <= 0.25`。

标准 instant success gate 为：

```text
dmax <= success_thresholds.dmax
dispersion <= success_thresholds.dispersion
all rover speeds <= success_thresholds.speed
minimum pairwise distance gate, when configured
centroid footprint flatness == true
```

`success_thresholds.min_pairwise_distance` 是可选安全成功门控，默认 `0.0`，保持旧实验兼容。设置为正数时，instant success 还要求全队最近邻距离不小于该值。使用该字段的配置必须满足：

```text
safety.collision_distance < success_thresholds.min_pairwise_distance < success_thresholds.dmax
```

`gather_point.require_flat_for_success` 默认是 `true`。只有全部 instant gate 连续满足 `success_thresholds.hold_steps` 步，`success_hold_count` 才达到终止成功条件；任一步质心圆盘不平整都会将 hold count 清零。`success_gates.flatness_ok` 与 `gather_point_flatness.height_range/max_slope/mean_slope/is_flat` 用于诊断该门控。

该变更把成功定义从“几何聚集并低速保持”升级为“在合格地形上几何聚集并低速保持”。因此，所有在旧成功门下报告的历史成功率都只能作为旧语义基线，不能直接宣称优于或劣于新结果。

## Actual-centroid flatness reward / telemetry

平整度 shaping 使用状态推进后实际团队质心处的圆盘地形，而不是 episode 固定的 `oracle_point`。设圆盘高度范围为 \(\Delta h_t\)，最大坡度为 \(s_t\)，则与 hard gate 对齐的归一化代价为：

\[
C_t=\operatorname{clip}\left(
\max\left(
\frac{\Delta h_t}{\texttt{gather\_point.max\_height\_range}},
\frac{s_t}{\texttt{gather\_point.max\_slope}}
\right),
0,3
\right).
\]

接口保证 `C_t <= 1` 当且仅当高度范围和最大坡度两项 flatness gate 都通过。设 \(d_g=\texttt{success\_thresholds.dmax}\)，\(m=\texttt{reward.coefficients.centroid\_flatness\_dmax\_multiplier}>1\)，则：

\[
a_t=\operatorname{clip}\left(
\frac{m d_g-d_{\max,t}}{(m-1)d_g},
0,1
\right),
\qquad
P_t=a_tC_t,
\]

\[
r_t^{\mathrm{flat}}=
\texttt{centroid\_flatness\_progress}(P_{t-1}-P_t)
-\texttt{centroid\_flatness\_excess}\,
a_t\operatorname{ReLU}(C_t-1).
\]

`a_t=0` 对应 `dmax >= m * d_g`，中间区间线性 ramp，`a_t=1` 对应 `dmax <= d_g`。进展接口返回 gated potential 差 `previous_activation * previous_cost - current_activation * current_cost`，而不是裸 cost 差。对任意往返轨迹：

\[
\sum_{t=1}^{T}(P_{t-1}-P_t)=P_0-P_T.
\]

回到相同 cost/activation 状态时 \(P_T=P_0\)，进展贡献严格抵消；非正的 excess penalty 只会降低循环回报。这避免旧式“用当前 activation 乘裸 cost 差”在跨边界时弱化负向撤销量，从而产生可重复的正奖励循环。

exp064 设置 `d_g=1.25 m`、`m=2.0`，因此激活区间为 `2.50 m -> 1.25 m`；`centroid_flatness_progress=2.0`、`centroid_flatness_excess=0.02`、`reward.weights.flatness=1.0`。

默认接口配置为：

```yaml
reward:
  weights:
    flatness: 0.0
  coefficients:
    centroid_flatness_progress: 0.0
    centroid_flatness_excess: 0.0
    centroid_flatness_dmax_multiplier: 2.0
```

因此未显式启用的历史配置在行为上保持关闭。环境 `StepOutput.info` 增加：

```text
centroid_flatness_reward.cost
centroid_flatness_reward.progress
centroid_flatness_reward.activation
```

其中 `centroid_flatness_reward.progress` 的逐环境张量语义是 \(P_{t-1}-P_t\)；cost 与 activation 分别仍为当前步的 \(C_t\) 与 \(a_t\)。

训练 telemetry 展平为：

```text
centroid_flatness_cost_mean
centroid_flatness_progress_mean
centroid_flatness_activation_mean
reward_raw_flatness
reward_weight_flatness
reward_contribution_flatness
```

其中 reward contribution 按 `reward.weights.flatness * reward_raw_flatness` 计算。训练曲线可显示 cost 与 activation，用于区分“尚未进入激活区间”和“进入末段但平整度未改善”。

该计算只读取实际 `metrics.centroid`、`metrics.dmax` 与当前 terrain runtime，不读取 `oracle_point`、oracle 距离或 `oracle_search` 输出。新增内容仅属于 reward 与 telemetry，不增加 Actor observation、Critic state、action 或 checkpoint schema 字段；Actor/Critic 维度保持 `86 / 54`，执行期不存在由该 shaping 引入的 oracle 泄漏。

## Terrain reward

terrain reward 可组合以下代价：

```text
underfoot roughness
underfoot non-traversability
decoded subgoal risk
actual terrain speed loss
absolute terrain height change
straight path mean terrain risk
straight path max terrain risk
straight path mean height change
raw-action path risk auxiliary cost
filter-deviation auxiliary cost
```

候选子目标风险由当前 action 解码后的世界坐标落点计算；速度损失使用本步实际 terrain speed scale；高度变化使用积分前后地形高度差。路径级风险沿 rover 当前点到 decoded subgoal 的直线采样 5 个点，统计 mean risk、max risk 和 mean absolute height change。filter auxiliary cost 使用 raw action 的 path risk 和 filter 建议相对 raw intent 的 deviation。新增项在默认配置中系数均为 0，因此不会改变 exp020 及更早实验的历史语义。
