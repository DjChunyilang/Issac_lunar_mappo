# 技术设计

本文是短版技术摘要，只定义任务建模、信息边界、观测/状态/action、轨迹控制链、reward、网络接口和评估判据。长期技术路径管理见主目录 `多月球车自组织集合局部参考轨迹规划技术文档.md`，工程目录见 [scaffold.md](scaffold.md)，当前实施计划见 [implementation_plan.md](implementation_plan.md)。

## 任务与路线

任务目标是让 4 个同构 rover 在有限通信和局部感知条件下自组织集合到隐含目标区域。当前训练主路径是 torch-vectorized proxy 环境；Isaac Sim / Isaac Lab / PhysX 当前用于 checkpoint 级 high-fidelity closed-loop evaluation，不参与每次 PPO/MAPPO 采样更新。

核心闭环：

```text
actor observation
-> policy 输出归一化 action
-> [rho, beta] 局部子目标
-> 局部参考轨迹
-> 简化速度控制
-> proxy unicycle 状态更新
-> reward / termination / metrics
```

## 信息边界

- Actor 执行期只使用自车、邻居、地形手工特征和局部聚合特征。
- Actor observation 不包含 `p*`、oracle 距离或 oracle 距离下降量。
- Centralized critic state 和 reward shaping 可以使用训练期 oracle 信息。
- 视觉观测当前不进入 policy input；地形以车体系局部结构化网格进入策略。

## Actor Observation

当前 schema 为 `ego_v3_local_terrain_grid`，形状为：

```text
(num_envs, 4, obs_dim)
```

字段组：

```text
ego_dim: 10
neighbor_dim: 7
terrain_dim: 50
aggregation_dim: 5
communication_radius: cfg.observation.communication_radius
```

Actor 总输入维度为 86。地形观测使用固定车体系 `5×5` 网格：

```text
x = [-0.4, 0.0, 0.4, 0.8, 1.2] m
y = [-0.8, -0.4, 0.0, 0.4, 0.8] m
channels = [relative_height, risk]
flatten = x -> y -> channel
```

`relative_height` 相对 rover 脚下高度计算，`risk=1-traversability`。平地输出全零。原脚下 5 维 `height/slope_x/slope_y/roughness/traversability` 仍用于 proxy 动力学和 terrain reward，不再直接作为 actor 地形观测。

`observation.communication_radius` 是当前唯一允许从 experiment YAML 覆盖的 observation 字段。`max_neighbors`、`ego_dim`、`neighbor_dim`、`terrain_dim` 和 `aggregation_dim` 会改变模型输入接口，本轮不开放配置覆盖。

## Critic State

Critic state 形状为：

```text
(num_envs, state_dim)
```

它包含全部 rover 真值状态、队形几何信息、地形摘要和仅训练使用的 oracle 特征。该信息只服务 centralized critic、reward shaping 和评估指标，不进入 actor 执行期输入。Critic 总维度保持 54；地形 5 维摘要改为平均绝对高差、最大上升、最大下降、平均风险和最大风险。

## Action 与轨迹控制

Policy 输出形状为：

```text
(num_envs, 4, 2)
```

归一化 action 被映射为：

```text
rho in [0, rho_max]
beta in [-beta_max, beta_max]
```

控制链路：

```text
action_interpreter.py
-> trajectory_generator.py
-> simple_controller.py
-> gathering_env.py::_integrate()
```

当前 proxy 动力学是 2D/2.5D kinematic unicycle 风格状态更新。地形开启时会查询 procedural heightfield / crater proxy 特征并施加速度缩放，但不包含质量、惯量、轮地接触、打滑、沉陷、悬挂或 PhysX contact。

## Reward 与终止

Reward 由以下部分组成：

- 自组织集合 reward：鼓励队形收缩和距离目标集合区域更近。
- Oracle 辅助 reward：仅训练期使用，用于距离进展 shaping。
- 能耗代理 reward：约束过大的速度和角速度命令。
- 安全惩罚：约束 rover 间最小距离和碰撞。
- 运动质量 reward：鼓励平滑、有效的运动。
- 一致性 reward：鼓励局部协同。
- 终端 reward：根据成功、碰撞、超时等终止原因给出 episode 级反馈。

终止条件包括集合成功、碰撞、安全边界失败和 episode timeout。不要只用训练 reward 判断成功，严格结论以机器可读评估结果为准。

## 网络与训练接口

- Actor 是同构多智能体策略，输入去中心化 observation，输出每车 `[rho, beta]`。
- Critic 使用 centralized state，服务 MAPPO/PPO 训练。
- SKRL-MAPPO 训练通过 `MultiRoverGatheringSKRLEnv` 和 `isaaclab-multi-agent` wrapper 接入；该 wrapper 是接口层，不代表 PhysX 训练。
- Checkpoint metadata 必须记录 `observation_schema_version`、`actor_obs_dim` 和 `critic_state_dim`。旧 schema 或缺少 metadata 的 checkpoint 明确拒绝，不自动迁移。

## 评估判据

Proxy strict gate 默认写成：

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

High-fidelity PhysX / Jackal tracking 当前报告：

```text
rmse_cross_track_m
max_cross_track_m
path_completion_ratio
max_tilt_deg
timeseries.csv
tracking.png
```

结果表述必须区分：

- proxy training：策略在 proxy 环境中训练。
- proxy strict evaluation：checkpoint 通过独立 deterministic proxy gate。
- Isaac/PhysX high-fidelity closed-loop evaluation：checkpoint 或参考轨迹在 PhysX 场景中做低频闭环验证。

exp006 / exp008 是 proxy strict pass，不是 Isaac 物理训练 pass。Jackal tracking 是高保真评估 / sanity check，不是真实月球车训练结果。
