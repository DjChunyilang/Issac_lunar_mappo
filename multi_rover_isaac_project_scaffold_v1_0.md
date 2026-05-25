# 多月球车自组织集合项目脚手架文档
## 基于 Isaac Sim / Isaac Lab 与 SKRL-MAPPO 的项目文件架构（V1.0）

---

## 1. 文档目的

本文档用于定义多月球车自组织集合项目在 Isaac Sim / Isaac Lab 环境下的工程脚手架结构。文档重点包括项目目录组织、模块职责、文件命名、数据流关系、配置文件结构、开发顺序与阶段性实现范围。

本文档不重复算法原理，仅服务于项目实现。对应技术路线为：

$$
\text{Isaac Sim / Isaac Lab}
+
\text{SKRL-MAPPO}
+
\text{低维局部子目标动作}
+
\text{确定性轨迹生成器}
+
\text{简化速度跟踪控制器}.
$$

当前阶段冻结的规划动作定义为：

$$
a_i(t)=\left[\rho_i(t),\beta_i(t)\right].
$$

其中，$\rho_i(t)$ 为车体系下局部子目标距离，$\beta_i(t)$ 为车体系下局部子目标方位角。

---

## 2. 项目总体目录结构

建议项目根目录命名为：

```text
multi_rover_gathering_isaac/
```

推荐目录结构如下：

```text
multi_rover_gathering_isaac/
├── README.md
├── pyproject.toml
├── setup.py
├── source/
│   └── lunar_rover_tasks/
│       ├── setup.py
│       └── lunar_rover_tasks/
│           ├── __init__.py
│           ├── tasks/
│           │   └── multi_rover_gathering/
│           │       ├── __init__.py
│           │       ├── gathering_env.py
│           │       ├── gathering_env_cfg.py
│           │       ├── observation.py
│           │       ├── state.py
│           │       ├── reward.py
│           │       ├── termination.py
│           │       ├── action_interpreter.py
│           │       ├── trajectory_generator.py
│           │       ├── simple_controller.py
│           │       ├── terrain_features.py
│           │       ├── communication.py
│           │       ├── oracle.py
│           │       └── metrics.py
│           ├── assets/
│           │   └── rover/
│           │       ├── README.md
│           │       ├── usd/
│           │       ├── urdf/
│           │       └── cfg/
│           └── utils/
│               ├── math_utils.py
│               ├── tensor_utils.py
│               ├── geometry_utils.py
│               └── visualization_utils.py
├── configs/
│   ├── env/
│   │   ├── base_env.yaml
│   │   ├── terrain_flat.yaml
│   │   └── terrain_lunar_simple.yaml
│   ├── agent/
│   │   ├── skrl_mappo_base.yaml
│   │   └── skrl_mappo_debug.yaml
│   ├── reward/
│   │   ├── reward_base.yaml
│   │   └── reward_ablation_no_oracle.yaml
│   ├── experiment/
│   │   ├── exp_001_minimal.yaml
│   │   ├── exp_002_oracle_ablation.yaml
│   │   └── exp_003_terrain_ablation.yaml
│   └── task/
│       └── multi_rover_gathering.yaml
├── scripts/
│   ├── train.py
│   ├── play.py
│   ├── eval.py
│   ├── debug_env.py
│   ├── debug_observation.py
│   ├── debug_reward.py
│   └── export_policy.py
├── tools/
│   ├── check_asset.py
│   ├── generate_terrain.py
│   ├── visualize_trajectory.py
│   └── plot_logs.py
├── tests/
│   ├── test_observation.py
│   ├── test_reward.py
│   ├── test_termination.py
│   ├── test_action_interpreter.py
│   └── test_trajectory_generator.py
├── docs/
│   ├── technical_design.md
│   ├── scaffold.md
│   ├── interface_spec.md
│   ├── experiment_plan.md
│   └── meeting_notes/
├── outputs/
│   ├── runs/
│   ├── checkpoints/
│   ├── videos/
│   ├── logs/
│   └── figures/
└── notebooks/
    ├── reward_analysis.ipynb
    ├── trajectory_visualization.ipynb
    └── experiment_result_analysis.ipynb
```

---

## 3. 顶层目录职责

| 目录/文件 | 职责 |
|---|---|
| `README.md` | 项目说明、安装方式、运行入口、当前阶段目标 |
| `pyproject.toml` | Python 工程配置、代码格式化工具配置 |
| `setup.py` | 项目安装入口 |
| `source/` | Isaac Lab 扩展任务源码 |
| `configs/` | 环境、算法、奖励、实验配置 |
| `scripts/` | 训练、评估、调试、策略导出脚本 |
| `tools/` | 资产检查、地形生成、日志绘图等辅助工具 |
| `tests/` | 关键模块单元测试 |
| `docs/` | 技术文档、接口文档、实验计划 |
| `outputs/` | 训练结果、模型权重、日志、视频与图表 |
| `notebooks/` | 奖励分析、轨迹可视化、结果处理 |

---

## 4. Isaac Lab 任务源码结构

核心源码位于：

```text
source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/
```

该目录是项目实现的主体，包含环境、观测、状态、奖励、终止、动作解释、轨迹生成和简化控制模块。

### 4.1 `gathering_env.py`

该文件定义多月球车自组织集合任务环境，是 Isaac Lab 环境的核心入口。

主要职责：

1. 管理环境 reset 与 step；
2. 调用观测构造模块；
3. 调用动作解释模块；
4. 调用轨迹生成模块；
5. 调用简化控制器；
6. 调用奖励函数；
7. 调用终止判据；
8. 向 SKRL-MAPPO 暴露环境接口。

建议包含的核心类：

```python
class MultiRoverGatheringEnv:
    pass
```

主要方法：

```text
__init__()
reset()
step()
_get_observations()
_apply_actions()
_get_rewards()
_get_dones()
```

### 4.2 `gathering_env_cfg.py`

该文件定义 Isaac Lab 任务配置。

主要职责：

1. 配置并行环境数量；
2. 配置仿真步长；
3. 配置 episode 长度；
4. 配置 rover 资产路径；
5. 配置地形类型；
6. 配置通信半径；
7. 配置动作范围；
8. 配置奖励权重；
9. 配置终止阈值。

关键配置项：

```text
num_envs
physics_dt
control_decimation
episode_length_s
rover_asset_path
terrain_type
communication_radius
rho_max
beta_max
reward_weights
success_thresholds
```

### 4.3 `observation.py`

该文件负责构造 actor 的局部观测：

$$
o_i(t)=
\left[
o_i^{\text{ego}}(t),
o_i^{\text{nbr}}(t),
o_i^{\text{ter}}(t),
o_i^{\text{agg}}(t)
\right].
$$

主要职责：

1. 读取自车状态；
2. 读取邻居相对状态；
3. 读取局部地形手工特征；
4. 计算局部聚集态势；
5. 拼接 actor 观测张量；
6. 保证 oracle 信息不进入 actor。

建议函数：

```text
build_actor_observation()
build_ego_features()
build_neighbor_features()
build_terrain_features()
build_aggregation_features()
```

### 4.4 `state.py`

该文件负责构造 centralized critic 的全局状态：

$$
s(t)=
\left[
s^{\text{agent}}(t),
s^{\text{team}}(t),
s^{\text{ter}}(t),
s^{\text{oracle}}(t)
\right].
$$

主要职责：

1. 拼接全部 rover 真值状态；
2. 计算团队几何统计量；
3. 构造全局地形摘要；
4. 加入 oracle 辅助信息；
5. 输出 critic state。

建议函数：

```text
build_critic_state()
build_agent_global_state()
build_team_state()
build_oracle_state()
```

### 4.5 `reward.py`

该文件负责奖励函数计算。

总奖励为：

$$
r_t =
w_g r_{\text{gather}}(t)
+
w_o r_{\text{oracle}}(t)
+
w_e r_{\text{energy}}(t)
+
w_s r_{\text{safety}}(t)
+
w_m r_{\text{motion}}(t)
+
w_c r_{\text{consistency}}(t)
+
w_T r_{\text{terminal}}(t).
$$

主要职责：

1. 计算自组织聚集奖励；
2. 计算 oracle 平均距离下降量奖励；
3. 计算能耗代理项；
4. 计算安全惩罚；
5. 计算低维动作下的运动质量项；
6. 计算一致性惩罚；
7. 计算终端奖励；
8. 输出总奖励及日志分项。

建议函数：

```text
compute_reward()
compute_gather_reward()
compute_oracle_reward()
compute_energy_reward()
compute_safety_reward()
compute_motion_reward()
compute_consistency_reward()
compute_terminal_reward()
```

### 4.6 `termination.py`

该文件负责成功/失败判据。

集合成功条件为：

$$
D_{\max}(t)\le \varepsilon_D,
\qquad
\sigma_p^2(t)\le \varepsilon_\sigma,
\qquad
\|v_i(t)\|\le \varepsilon_v.
$$

主要职责：

1. 判断集合成功；
2. 判断碰撞失败；
3. 判断越界失败；
4. 判断超时失败；
5. 维护连续成功步数 $n_{\text{hold}}$。

建议函数：

```text
check_success()
check_collision()
check_out_of_bounds()
check_timeout()
compute_done()
```

### 4.7 `action_interpreter.py`

该文件负责将 MAPPO actor 输出的低维动作解释为局部子目标。

输入：

$$
a_i(t)=
\left[
\rho_i(t),\beta_i(t)
\right].
$$

输出：

$$
p_{i,\text{sub}}^{b}(t)=
\begin{bmatrix}
\rho_i(t)\cos\beta_i(t)\\
\rho_i(t)\sin\beta_i(t)
\end{bmatrix}.
$$

主要职责：

1. 对动作进行范围裁剪；
2. 将归一化动作映射到物理范围；
3. 计算车体系局部子目标；
4. 转换到世界坐标系；
5. 查询地形高度补全 $z$。

建议函数：

```text
decode_action()
clip_action()
scale_action()
polar_to_local_subgoal()
local_to_world_subgoal()
query_subgoal_height()
```

### 4.8 `trajectory_generator.py`

该文件负责从局部子目标生成局部参考轨迹。

输入：

$$
p_{i,\text{sub}}^{b}(t)
$$

输出：

$$
\mathcal{T}_i(t)=
\{(x_{i,k},y_{i,k},z_{i,k},\psi_{i,k},t_{i,k},v_{i,k})\}_{k=1}^{K}.
$$

主要职责：

1. 生成直线、圆弧或 Bézier 局部路径；
2. 查询地形高度；
3. 计算轨迹航向；
4. 分配时间戳；
5. 分配参考速度；
6. 输出固定长度轨迹。

第一阶段建议优先采用直线或圆弧轨迹，不优先实现复杂 spline。

建议函数：

```text
generate_trajectory()
generate_line_path()
generate_arc_path()
assign_height()
assign_heading()
assign_timestamps()
assign_reference_speed()
```

### 4.9 `simple_controller.py`

该文件负责将局部参考轨迹转换为简化控制命令。

输入：

$$
\mathcal{T}_i(t)
$$

输出：

$$
u_i(t)=
[v_i^{\text{cmd}}(t),\omega_i^{\text{cmd}}(t)].
$$

主要职责：

1. 选择跟踪点；
2. 计算前向速度命令；
3. 计算角速度命令；
4. 适配 Isaac Sim 底层控制接口；
5. 在底层接口未冻结前提供简化控制输出。

建议函数：

```text
compute_control()
select_tracking_point()
compute_v_cmd()
compute_omega_cmd()
adapt_to_articulation_command()
```

### 4.10 `terrain_features.py`

该文件负责地形手工特征提取。

主要职责：

1. 读取或查询地形高度；
2. 计算局部坡度；
3. 计算局部粗糙度；
4. 计算局部高差；
5. 计算障碍密度；
6. 计算可通行宽度。

建议函数：

```text
query_height()
compute_slope_features()
compute_roughness_features()
compute_height_diff_features()
compute_obstacle_density()
compute_traversable_width()
```

### 4.11 `communication.py`

该文件负责邻居状态共享。

主要职责：

1. 根据通信半径计算邻居集合；
2. 构造邻居可见性 mask；
3. 计算邻居相对状态；
4. 输出固定槽位邻居特征。

建议函数：

```text
compute_neighbor_set()
compute_visibility_mask()
build_neighbor_state()
pad_neighbor_slots()
```

### 4.12 `oracle.py`

该文件负责训练期 oracle 辅助信息。

主要职责：

1. 接收或计算最优集合点 $p^{*}(t)$；
2. 计算 $d_i^{*}(t)$；
3. 计算 $\bar d^{*}(t)$；
4. 输出 critic state 所需 oracle 特征；
5. 输出 oracle 奖励所需距离下降量。

建议函数：

```text
get_optimal_gathering_point()
compute_oracle_distances()
compute_mean_oracle_distance()
build_oracle_features()
```

### 4.13 `metrics.py`

该文件负责评估指标计算。

主要职责：

1. 计算集合成功率；
2. 计算平均完成时间；
3. 计算最大 pairwise 距离；
4. 计算团队分散度；
5. 计算 oracle 最优性差距；
6. 计算轨迹质量指标；
7. 计算闭环执行指标。

---

## 5. 配置文件结构

配置文件位于：

```text
configs/
```

### 5.1 环境配置

路径：

```text
configs/env/base_env.yaml
```

建议字段：

```yaml
simulation:
  simulator: IsaacSim
  framework: IsaacLab
  device: cuda
  headless: true
  num_envs: N_env
  physics_dt: dt_phys
  control_decimation: n_decimation
  episode_length_s: T_episode

task:
  name: Isaac-MultiRover-Gathering-Direct-v0
  n_agents: 4
  scene_dim: "2.5D/3D"
  explicit_goal_in_execution: false
  oracle_optimal_gather_point_in_training: true
  docking_considered: false
```

### 5.2 地形配置

路径：

```text
configs/env/terrain_lunar_simple.yaml
```

建议字段：

```yaml
terrain:
  type: lunar_heightfield_or_mesh
  use_handcrafted_features: true
  features:
    - slope
    - roughness
    - local_height_diff
    - obstacle_density
    - traversable_width
```

### 5.3 智能体与算法配置

路径：

```text
configs/agent/skrl_mappo_base.yaml
```

建议字段：

```yaml
algorithm:
  name: MAPPO
  library: SKRL
  framework: IsaacLab
  shared_actor: true
  centralized_critic: true
  gamma: gamma
  gae_lambda: lambda
  clip_epsilon: epsilon_clip
  entropy_coef: c_ent
  value_loss_coef: c_v
  ppo_epochs: E_ppo
  rollout_steps: L_rollout
```

### 5.4 规划与控制配置

路径：

```text
configs/task/multi_rover_gathering.yaml
```

建议字段：

```yaml
planner:
  action_type: local_subgoal_polar
  action_dim: 2
  action_fields: [rho, beta]
  output_coordinate: body_relative
  rho_max: rho_max
  beta_max: beta_max

trajectory_generator:
  input_type: local_subgoal
  output_type: time_stamped_trajectory
  n_trajectory_points: K
  geometry_method: line_or_arc
  terrain_height_query: true
  assign_timestamp: true
  assign_reference_speed: true

low_level_control:
  first_stage_mode: simplified_velocity_tracking
  command_type: body_twist
  command_fields: [v_cmd, omega_cmd]
  isaac_articulation_interface: TBD
```

### 5.5 奖励配置

路径：

```text
configs/reward/reward_base.yaml
```

建议字段：

```yaml
reward:
  weights:
    gather: w_g
    oracle: w_o
    energy: w_e
    safety: w_s
    motion: w_m
    consistency: w_c
    terminal: w_T

  coefficients:
    dmax_progress: alpha_1
    dispersion_progress: alpha_2
    oracle_mean_distance_progress: alpha_3
    path_length: alpha_4
    slope_cost: alpha_5
    turn_cost: alpha_6
    terrain_cost: alpha_7
    obstacle_collision: alpha_8
    inter_agent_collision: alpha_9
    near_distance: alpha_10
    subgoal_turn: alpha_11
    subgoal_stagnation: alpha_12
    action_consistency: alpha_14
```

### 5.6 实验配置

路径：

```text
configs/experiment/exp_001_minimal.yaml
```

建议用途：

1. 指定基础环境；
2. 指定奖励配置；
3. 指定算法配置；
4. 指定日志路径；
5. 指定训练步数；
6. 指定评估间隔；
7. 指定是否保存视频。

---

## 6. 脚本入口

### 6.1 `scripts/train.py`

训练入口。

职责：

1. 读取实验配置；
2. 创建 Isaac Lab 环境；
3. 创建 SKRL-MAPPO agent；
4. 执行训练；
5. 保存 checkpoint；
6. 保存训练日志。

运行形式：

```bash
python scripts/train.py --config configs/experiment/exp_001_minimal.yaml
```

### 6.2 `scripts/play.py`

策略回放入口。

职责：

1. 加载 checkpoint；
2. 创建环境；
3. 执行 deterministic policy；
4. 保存视频或轨迹数据。

### 6.3 `scripts/eval.py`

批量评估入口。

职责：

1. 加载训练模型；
2. 在固定测试场景集上运行；
3. 统计成功率、完成时间、碰撞率和 oracle gap；
4. 输出结果表格。

### 6.4 `scripts/debug_env.py`

环境调试入口。

职责：

1. 检查 reset 是否正常；
2. 检查 step 是否正常；
3. 检查多 rover 状态是否正确；
4. 检查动作是否能推进仿真。

### 6.5 `scripts/debug_observation.py`

观测调试入口。

职责：

1. 打印 actor observation 维度；
2. 检查是否存在 NaN；
3. 检查 oracle 信息是否泄漏到 actor；
4. 检查邻居 mask 是否正确。

### 6.6 `scripts/debug_reward.py`

奖励调试入口。

职责：

1. 输出各奖励分项；
2. 检查奖励量级；
3. 检查成功/失败终端奖励；
4. 检查 oracle 平均距离下降量。

---

## 7. 单元测试结构

测试目录位于：

```text
tests/
```

### 7.1 `test_observation.py`

测试内容：

1. actor 观测维度是否正确；
2. neighbor slot 是否正确填充；
3. mask 是否正确；
4. oracle 信息是否未进入 actor。

### 7.2 `test_reward.py`

测试内容：

1. $D_{\max}$ 下降时 $r_{\text{gather}}$ 是否为正；
2. $\sigma_p^2$ 下降时奖励是否为正；
3. $\bar d^{*}$ 下降时 $r_{\text{oracle}}$ 是否为正；
4. 碰撞时是否产生负奖励；
5. 静止不动时是否存在停滞惩罚。

### 7.3 `test_termination.py`

测试内容：

1. 满足几何聚集条件时是否成功；
2. 未达到保持步数时是否不成功；
3. 碰撞时是否失败；
4. 超时时是否失败。

### 7.4 `test_action_interpreter.py`

测试内容：

1. $\rho$ 是否被限制在 $[0,\rho_{\max}]$；
2. $\beta$ 是否被限制在 $[-\beta_{\max},\beta_{\max}]$；
3. 极坐标到局部子目标转换是否正确；
4. 局部坐标到世界坐标转换是否正确。

### 7.5 `test_trajectory_generator.py`

测试内容：

1. 输出轨迹点数量是否为 $K$；
2. 时间戳是否单调递增；
3. 轨迹高度是否可查询；
4. 航向角是否连续；
5. 轨迹生成失败时是否有回退逻辑。

---

## 8. 数据流关系

### 8.1 训练数据流

```text
Isaac Sim scene
    ↓
gathering_env.py
    ↓
observation.py  → actor observation o_i
state.py        → critic state s
    ↓
SKRL-MAPPO actor / critic
    ↓
action_interpreter.py
    ↓
trajectory_generator.py
    ↓
simple_controller.py
    ↓
Isaac Sim physics step
    ↓
reward.py + termination.py
    ↓
SKRL rollout buffer
```

### 8.2 actor 执行流

```text
o_i(t)
  → shared actor
  → [rho_i, beta_i]
  → local subgoal
  → local trajectory
  → [v_cmd, omega_cmd]
  → Isaac Sim execution
```

### 8.3 critic 训练流

```text
all rover true states
  + team geometry statistics
  + terrain summary
  + oracle features
  → centralized critic
  → V_phi(s)
```

### 8.4 oracle 信息流

```text
external optimal gathering algorithm
  → p^{*}(t)
  → d_i^{*}(t)
  → mean distance bar d^{*}(t)
  → critic state + oracle reward + evaluation metrics
```

约束：

```text
oracle information must not enter actor observation
```

---

## 9. 开发阶段规划

### 9.1 阶段 0：环境与资产检查

目标：

1. 确认 Isaac Sim 与 Isaac Lab 可运行；
2. 确认 SKRL 可调用；
3. 确认 rover 资产能加载；
4. 确认 4 个 rover 可在同一场景中实例化。

产出：

1. `tools/check_asset.py`
2. `scripts/debug_env.py`
3. 最小场景运行截图或视频。

### 9.2 阶段 1：最小多智能体环境

目标：

1. 完成 `gathering_env.py`；
2. 完成 reset/step；
3. 完成低维动作输入；
4. 完成简化控制执行；
5. 实现无复杂地形的平面集合任务。

产出：

1. 可运行环境；
2. 随机策略下环境可推进；
3. 无 NaN；
4. 轨迹可视化。

### 9.3 阶段 2：奖励与终止逻辑

目标：

1. 实现自组织聚集奖励；
2. 实现 oracle 平均距离下降量奖励；
3. 实现碰撞和超时失败；
4. 实现几何成功判据。

产出：

1. `reward.py`
2. `termination.py`
3. `test_reward.py`
4. `test_termination.py`

### 9.4 阶段 3：SKRL-MAPPO 训练闭环

目标：

1. 接入 SKRL；
2. 跑通 MAPPO 训练；
3. 验证 reward 曲线是否正常；
4. 验证成功率是否优于随机策略。

产出：

1. `scripts/train.py`
2. `scripts/play.py`
3. 首版 checkpoint；
4. 首版训练曲线。

### 9.5 阶段 4：地形特征增强

目标：

1. 加入坡度、粗糙度、高差等手工地形特征；
2. 加入障碍物；
3. 加入地形代价奖励；
4. 验证策略是否能避开高风险区域。

产出：

1. `terrain_features.py`
2. `configs/env/terrain_lunar_simple.yaml`
3. 地形消融实验结果。

### 9.6 阶段 5：轨迹与控制增强

目标：

1. 将直线轨迹扩展为圆弧或 Bézier；
2. 加入速度跟踪误差统计；
3. 根据 rover asset 控制接口替换底层控制适配器；
4. 完成闭环控制稳定性分析。

产出：

1. `trajectory_generator.py`
2. `simple_controller.py`
3. 跟踪误差统计结果。

### 9.7 阶段 6：实验与论文结果

目标：

1. 完成 oracle reward 消融；
2. 完成通信半径消融；
3. 完成动作维度消融；
4. 完成地形复杂度消融；
5. 整理论文图表。

产出：

1. `evaluate/metrics.py`
2. `notebooks/experiment_result_analysis.ipynb`
3. 论文实验结果表格与曲线。

---

## 10. 最小可运行版本范围

第一版最小可运行版本只包含以下内容：

| 模块 | 是否实现 |
|---|---|
| 4 个 rover 实例 | 是 |
| 平面或简单地形 | 是 |
| 低维动作 $[\rho,\beta]$ | 是 |
| 简化子目标解释器 | 是 |
| 简化轨迹生成器 | 是 |
| 简化速度控制器 | 是 |
| 几何集合奖励 | 是 |
| oracle 平均距离下降量奖励 | 是 |
| SKRL-MAPPO 训练 | 是 |
| 复杂轮地接触建模 | 否 |
| 力矩级能耗建模 | 否 |
| 可学习通信 | 否 |
| 多锚点轨迹输出 | 否 |
| 机械拼接/对接 | 否 |

---

## 11. 后续扩展路径

### 11.1 规划动作增强

动作空间可按以下顺序增强：

$$
[\rho,\beta]
\rightarrow
[\rho,\beta,v_{\text{ref}}]
\rightarrow
[(\rho_1,\beta_1),(\rho_2,\beta_2)]
\rightarrow
\{(\Delta s_m,\Delta l_m,\Delta h_m,\Delta\psi_m)\}_{m=1}^{M}.
$$

### 11.2 地形输入增强

地形输入可按以下顺序增强：

```text
handcrafted features
→ local elevation patch
→ DEM patch encoder
→ depth / lidar features
```

### 11.3 控制接口增强

控制接口可按以下顺序增强：

```text
simplified velocity tracking
→ wheel velocity control
→ steering + speed control
→ torque control
```

### 11.4 通信机制增强

通信机制可按以下顺序增强：

```text
neighbor state sharing
→ compressed neighbor statistics
→ learnable message passing
→ graph neural network communication
```

---

## 12. 当前冻结结论

当前项目脚手架冻结如下：

1. 项目采用 Isaac Sim / Isaac Lab 作为仿真与训练环境；
2. 训练算法采用 SKRL-MAPPO；
3. 智能体数量固定为 $N=4$；
4. 执行阶段不提供显式集合点；
5. actor 仅使用局部观测；
6. critic 使用全局状态与 oracle 辅助信息；
7. 通信机制采用邻居状态共享；
8. actor 输出低维动作 $[\rho,\beta]$；
9. 轨迹由确定性轨迹生成器生成；
10. 第一阶段采用简化速度跟踪控制器；
11. rover 底层 articulation 控制接口暂不冻结；
12. 第一版优先实现最小可训练闭环。

---

## 13. 下一步实现顺序

建议下一步按以下顺序建立项目：

1. 创建项目目录；
2. 创建 Isaac Lab extension 包；
3. 创建 `multi_rover_gathering` 任务目录；
4. 编写 `gathering_env_cfg.py`；
5. 编写 `gathering_env.py` 的 reset/step 空壳；
6. 编写 `action_interpreter.py`；
7. 编写 `trajectory_generator.py`；
8. 编写 `simple_controller.py`；
9. 编写 `observation.py` 与 `state.py`；
10. 编写 `reward.py` 与 `termination.py`；
11. 编写 `scripts/debug_env.py`；
12. 编写 `scripts/train.py`；
13. 接入 SKRL-MAPPO；
14. 跑通随机策略环境；
15. 跑通最小训练闭环。

---

这份脚手架文档的重点是**让项目文件结构和模块边界清楚**。后续进入代码阶段时，应优先保证 `gathering_env.py`、`action_interpreter.py`、`trajectory_generator.py` 和 `simple_controller.py` 四个文件能够形成最小闭环。 
