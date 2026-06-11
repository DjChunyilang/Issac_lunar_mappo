# 接口规范

## Actor 观测

形状：`(num_envs, 4, obs_dim)`。

Actor observation 包含自车状态、邻居共享状态、地形手工特征和局部聚合特征。不包含 `p*`、oracle 距离或 oracle 距离缩减量。

当前 observation schema：

```text
schema_version: ego_v2_speed_angular
communication_radius: cfg.observation.communication_radius
ego_dim: 10
neighbor_dim: 7
terrain_dim: 5
aggregation_dim: 5
```

`observation.communication_radius` 是当前唯一允许从 experiment YAML 覆盖的 observation 字段。`max_neighbors`、`ego_dim`、`neighbor_dim`、`terrain_dim` 和 `aggregation_dim` 会改变模型输入接口，本轮不开放配置覆盖。

ego 10 维顺序：

```text
x, y, z, cos(yaw), sin(yaw), vx, vy, angular_velocity, speed_xy, abs_angular_velocity
```

最后两维曾经是固定零占位；现在是运动派生特征。tensor shape 保持不变，但 checkpoint 行为语义已经改变，SKRL checkpoint metadata 会写入 `observation_schema_version`。

## Critic 状态

形状：`(num_envs, state_dim)`。

Critic state 包含全部 rover 真值状态、队形几何信息、地形摘要和仅训练使用的 oracle 特征。

## 动作

形状：`(num_envs, 4, 2)`。

归一化 action 被映射为：

- `rho in [0, rho_max]`
- `beta in [-beta_max, beta_max]`

## 第一阶段动力学

当前 rover 是 proxy unicycle 状态模型。只有在 rover 资产和控制接口明确后，才应替换为真实 Isaac Sim articulation。
