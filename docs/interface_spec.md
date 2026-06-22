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

## 第一阶段动力学

当前 rover 是 proxy unicycle 状态模型。只有在 rover 资产和控制接口明确后，才应替换为真实 Isaac Sim articulation。
