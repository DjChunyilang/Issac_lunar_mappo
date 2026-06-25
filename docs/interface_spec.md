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

`observation.communication_radius` 是当前唯一允许从 experiment YAML 覆盖的 observation 字段。`max_neighbors`、`ego_dim`、`neighbor_dim`、`terrain_dim` 和 `aggregation_dim` 会改变模型输入接口，本轮不开放配置覆盖。

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
```

加载时必须与当前配置完全一致。旧 `ego_v2_speed_angular` checkpoint、缺少 schema metadata 的 checkpoint 和错误输入维度 checkpoint 均明确拒绝，不做补零或自动迁移。

## 动作

形状：`(num_envs, 4, 2)`。

归一化 action 被映射为：

- `rho in [0, rho_max]`
- `beta in [-beta_max, beta_max]`

`planner.subgoal_filter` 是可选 proxy planner 后处理，默认关闭。启用时，它在 Actor 输出 `[rho, beta]` 之后、轨迹生成之前，从固定候选子目标中按地形路径风险和 endpoint safety 选择 filtered subgoal。该机制不改变 Actor 输出维度、Critic 状态维度或 checkpoint schema；checkpoint metadata 会记录 filter 配置摘要。

当前支持两种 mode：

- `terrain_safe_candidate`：exp020 使用的 hard filter，每步执行 score 最低的候选。
- `terrain_safe_candidate_curriculum`：exp021 使用的课程化 filter，训练时按 `warmup_timesteps` / `ramp_timesteps` 逐步提高 `apply_probability` 和 `score_scale`；评估时读取 checkpoint metadata 中的 `timesteps` 固定课程进度，并使用 deterministic rule，只有 filtered score 比 raw score 至少好 `deterministic_improvement_margin` 时才替换。

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
candidate_index
suggested_candidate_index
schedule_progress_step
apply_probability
score_scale
```

## 第一阶段动力学

当前 rover 是 proxy unicycle 状态模型。只有在 rover 资产和控制接口明确后，才应替换为真实 Isaac Sim articulation。

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
