# exp042 structured actor / bicycle / quintic 工程探针

## 目的

本轮暂停继续 20M 长训练，先完善训练环境三条核心链路：

- Actor 从旧 `mlp_v1` 可切换到 `branched_v1`，但输入仍为 86、输出仍为 `[rho,beta]`。
- Critic 从旧 `mlp_v1` 可切换到 `structured_v1`，但 centralized state 仍为 54。
- Proxy 运动学可切换到 `bicycle`，轨迹生成可切换到 `quintic`。
- 地图面积从旧实验常用的 `24 m × 24 m` 边界附近扩大为 `25 m × 25 m`，并临时取消通信距离限制。

该实验只证明工程链路可训练、可保存、可评估，不是收敛实验。

## 配置

```text
configs/experiment/exp042_structured_actor_bicycle_quintic_probe.yaml
```

关键字段：

```text
algorithm.actor_architecture: branched_v1
algorithm.critic_architecture: structured_v1
low_level_control.kinematic_model: bicycle
low_level_control.wheelbase_m: 0.65
low_level_control.max_steer_angle_rad: 0.610865
trajectory_generator.geometry_method: quintic
trajectory_generator.quintic_tangent_scale: 0.5
safety.world_xy_limit: 12.5
terrain.crater_field_size: 25.0
observation.communication_radius: 0.0
```

`communication_radius=0.0` 在当前代码中表示所有非自身 rover 可见；不是 0 米通信，而是临时取消通信半径限制。Actor 的局部地形网格仍为 `5×5×2=50` 维，保持 `86 / 54` 接口不变；扩大地形感知窗口需要另开 observation schema。

## 工程验证

聚焦单测：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaaclab/bin/python -m pytest -q \
  tests/test_trajectory_generator.py \
  tests/test_proxy_rover_model.py \
  tests/test_skrl_mappo_semantics.py
```

结果：通过。

CPU smoke：

```text
outputs/runs/exp042_structured_actor_bicycle_quintic_probe/smoke_cpu_structured_bicycle_quintic_comm0_map25/
```

CUDA smoke：

```text
outputs/runs/exp042_structured_actor_bicycle_quintic_probe/smoke_cuda_structured_bicycle_quintic_comm0_map25/
```

CUDA smoke 摘要：

```text
num_envs: 256
timesteps: 64
rollout_steps: 32
optimizer_count: 1
joint_update_count: 2
critic_update_count: 2
terrain_input_weight_delta_l2: 0.1263
post_training_action_std: 0.1459
communication_radius: 0.0
kinematic_model: bicycle
trajectory_geometry_method: quintic
```

## 判读

- `branched_v1/structured_v1` 的 shape、切片和 checkpoint metadata 已被测试覆盖。
- `bicycle` 中 `steering_angle`、`actual_yaw_rate`、`turning_radius` 已进入 step info / training telemetry。
- `quintic` 保持 `Trajectory.points/headings/timestamps/reference_speed` 接口不变。
- `communication_radius<=0` 的无限可见邻居语义用于 actor neighbor/aggregation、visible-local teacher 和子目标过滤器，不改变 observation 维度。
- 旧 `mlp_v1`、`unicycle`、`line` 仍是兼容路径；架构不匹配 checkpoint 会明确报错。

## 结论

exp042 可作为下一轮环境质量验证的起点，但不能写成策略收敛或 strict pass。若后续恢复长训，应新建 exp043 或后续实验，从随机初始化开始，并显式说明是否采用 exp042 的三项环境改造。
