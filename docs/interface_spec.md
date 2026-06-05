# 接口规范

## Actor 观测

形状：`(num_envs, 4, obs_dim)`。

Actor observation 包含自车状态、邻居共享状态、地形手工特征和局部聚合特征。不包含 `p*`、oracle 距离或 oracle 距离缩减量。

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
