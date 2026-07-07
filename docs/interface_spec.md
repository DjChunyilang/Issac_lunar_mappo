# 接口规范

## Actor 观测

形状：`(num_envs, 4, obs_dim)`。

Actor observation 包含自车状态、邻居共享状态、地形手工特征和局部聚合特征。不包含 `p*`、oracle 距离或 oracle 距离缩减量。

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

## Success / safety gate

标准成功 gate 为：

```text
dmax <= success_thresholds.dmax
dispersion <= success_thresholds.dispersion
all rover speeds <= success_thresholds.speed
success_hold_count >= success_thresholds.hold_steps
```

`success_thresholds.min_pairwise_distance` 是可选安全成功门控，默认 `0.0`，保持旧实验兼容。设置为正数时，instant success 还要求全队最近邻距离不小于该值。使用该字段的配置必须满足：

```text
safety.collision_distance < success_thresholds.min_pairwise_distance < success_thresholds.dmax
```

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
