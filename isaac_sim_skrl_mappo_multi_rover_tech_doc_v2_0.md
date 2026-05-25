# 多月球车自组织集合局部参考轨迹规划技术文档
## 基于 Isaac Sim / Isaac Lab 与 SKRL-MAPPO 的实现方案（V2.0）

---

## 1. 文档范围

本文档用于定义多月球车自组织集合任务在 NVIDIA Isaac Sim / Isaac Lab 环境下的技术架构、信息边界、强化学习建模方式、观测空间、状态空间、动作空间、奖励函数、网络结构、轨迹生成流程、训练流程、软件架构与实施计划。

本项目采用 Isaac Sim / Isaac Lab 作为主要训练与仿真平台，采用 SKRL 实现 MAPPO 多智能体强化学习算法。当前阶段的研究重点为规划模块。为降低系统初期实现难度，actor 不直接输出高维轨迹点序列，而输出低维局部子目标动作；局部参考轨迹由确定性轨迹生成器生成，底层控制采用简化速度跟踪方式，待月球车物理模型与控制接口明确后再替换为更精细的 articulation 控制接口。

---

## 2. 任务定义

设系统包含 $N=4$ 个同构月球车。第 $i$ 个智能体在时刻 $t$ 的状态记为 $x_i(t)$，位置为 $p_i(t)\in\mathbb{R}^3$，速度为 $v_i(t)$，航向角为 $\psi_i(t)$。

任务目标是在月面 2.5D/3D 地形条件下，使多个智能体通过局部感知与邻域通信完成自组织集合。执行阶段不向智能体显式提供集合点。智能体仅依据自身状态、邻居状态共享信息、局部地形特征和局部聚集态势进行分散决策。

训练阶段允许引入外部算法给出的最优集合点 $p^{*}(t)$。该点仅用于 centralized critic、oracle 辅助奖励和评估指标计算，不作为 actor 的观测输入。该设计保持集中式训练、分散式执行的信息边界。

当前阶段不考虑机械拼接与精细对接，仅以几何聚集成功作为任务完成标准。

---

## 3. 总体技术路线

主技术路线定义为：

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

系统分为五层。

第一层为 **Isaac Sim 物理仿真层**。该层负责机器人实体、月面地形、障碍物、碰撞检测、接触关系、物理时间推进和仿真状态读取。

第二层为 **Isaac Lab 任务环境层**。该层负责多智能体 reset、step、观测构造、动作解释、奖励计算、终止判定、oracle 辅助信息计算与并行环境管理。

第三层为 **SKRL-MAPPO 训练层**。该层实现 centralized training with decentralized execution。actor 执行期仅接收局部观测，critic 训练期接收全局状态和 oracle 辅助信息。

第四层为 **动作解释与轨迹生成层**。该层将 actor 输出的低维局部子目标动作 $[\rho_i,\beta_i]$ 转换为局部参考轨迹 $\mathcal{T}_i(t)$。

第五层为 **简化控制执行层**。该层将局部参考轨迹转换为车体速度命令 $[v_i^{\text{cmd}},\omega_i^{\text{cmd}}]$，再由底层适配器转换为 Isaac Sim 可执行的控制目标。后续可根据月球车模型将简化控制命令替换为轮速、转向角或力矩控制接口。

---

## 4. 符号约定

| 符号 | 含义 |
|---|---|
| $N$ | 智能体数量，当前固定为 4 |
| $i,j$ | 智能体索引 |
| $t$ | 离散规划时刻 |
| $\Delta t_p$ | 规划刷新周期 |
| $\Delta t_c$ | 控制执行周期 |
| $x_i(t)$ | 第 $i$ 个智能体全状态 |
| $p_i(t)\in\mathbb{R}^3$ | 第 $i$ 个智能体位置 |
| $v_i(t)$ | 第 $i$ 个智能体速度 |
| $\psi_i(t)$ | 第 $i$ 个智能体航向角 |
| $o_i(t)$ | 第 $i$ 个智能体局部观测 |
| $s(t)$ | centralized critic 输入状态 |
| $a_i(t)$ | 第 $i$ 个智能体动作 |
| $p^{*}(t)$ | 外部算法给出的最优集合点 |
| $d_i^{*}(t)$ | 第 $i$ 个智能体到 $p^{*}(t)$ 的距离 |
| $\bar d^{*}(t)$ | 平均最优点距离 |
| $\bar p(t)$ | 团队几何中心 |
| $D_{\max}(t)$ | 团队最大 pairwise 距离 |
| $\sigma_p^2(t)$ | 团队分散度 |
| $\mathcal{T}_i(t)$ | 第 $i$ 个智能体的局部参考轨迹 |
| $K$ | 轨迹生成器输出的轨迹点数量 |

其中：

$$
\bar p(t)=\frac{1}{N}\sum_{i=1}^{N}p_i(t),
$$

$$
D_{\max}(t)=\max_{i,j}\|p_i(t)-p_j(t)\|,
$$

$$
\sigma_p^2(t)=\frac{1}{N}\sum_{i=1}^{N}\|p_i(t)-\bar p(t)\|^2,
$$

$$
d_i^{*}(t)=\|p_i(t)-p^{*}(t)\|,
$$

$$
\bar d^{*}(t)=\frac{1}{N}\sum_{i=1}^{N}d_i^{*}(t).
$$

---

## 5. 系统运行闭环

系统运行闭环定义为：

$$
\text{Isaac Sim 场景状态}
\rightarrow
\text{Isaac Lab 观测构造}
\rightarrow
\text{SKRL-MAPPO actor 输出低维动作}
\rightarrow
\text{动作解释器生成局部子目标}
\rightarrow
\text{轨迹生成器生成局部参考轨迹}
\rightarrow
\text{简化速度跟踪控制器生成控制命令}
\rightarrow
\text{Isaac Sim 物理仿真推进}
\rightarrow
\text{奖励、终止与下一观测计算}.
$$

在每个规划步中，Isaac Lab 环境从 Isaac Sim 场景中读取所有月球车状态、地形信息、邻居关系和碰撞信息，构造 actor 局部观测 $o_i(t)$ 与 critic 全局状态 $s(t)$。actor 输出低维局部子目标动作后，环境内部动作解释器将其转换为局部参考轨迹，并通过简化控制器转换为仿真控制命令。Isaac Sim 推进物理仿真后，Isaac Lab 环境计算奖励、终止标志和下一时刻观测。

---

## 6. 信息边界

### 6.1 执行期信息

执行阶段，第 $i$ 个智能体可访问：

$$
o_i(t)=
\left[
o_i^{\text{ego}}(t),
o_i^{\text{nbr}}(t),
o_i^{\text{ter}}(t),
o_i^{\text{agg}}(t)
\right].
$$

其中，$o_i^{\text{ego}}(t)$ 为自车状态，$o_i^{\text{nbr}}(t)$ 为邻居状态共享信息，$o_i^{\text{ter}}(t)$ 为局部地形手工特征，$o_i^{\text{agg}}(t)$ 为局部聚集态势特征。

执行期 actor 不可访问 $p^{*}(t)$、$d_i^{*}(t)$、$\bar d^{*}(t)$ 等 oracle 信息。

### 6.2 训练期信息

训练阶段 centralized critic 可访问：

$$
s(t)=
\left[
s^{\text{agent}}(t),
s^{\text{team}}(t),
s^{\text{ter}}(t),
s^{\text{oracle}}(t)
\right].
$$

其中，$s^{\text{agent}}(t)$ 为全部智能体真值状态，$s^{\text{team}}(t)$ 为团队几何统计量，$s^{\text{ter}}(t)$ 为地形摘要，$s^{\text{oracle}}(t)$ 为外部最优集合点及其距离统计量。

### 6.3 通信机制

当前阶段通信机制采用邻居状态共享。设第 $i$ 个智能体的通信邻居集合为 $\mathcal{N}_i(t)$，则其可接收 $\mathcal{N}_i(t)$ 内邻居的状态摘要。当前阶段不引入可学习消息通信，可学习通信作为后续改进方向保留。

---

## 7. 观测空间

actor 观测空间定义为：

$$
o_i(t)=
\left[
o_i^{\text{ego}}(t),
o_i^{\text{nbr}}(t),
o_i^{\text{ter}}(t),
o_i^{\text{agg}}(t)
\right].
$$

| 分量 | 内容 | 来源 | actor 可见 |
|---|---|---|---|
| $o_i^{\text{ego}}$ | 自车位置、姿态、速度、角速度、底盘姿态代理量 | Isaac Sim rover 状态 | 是 |
| $o_i^{\text{nbr}}$ | 邻居相对位置、相对速度、相对航向、可见性 mask | Isaac Lab 根据通信半径计算 | 是 |
| $o_i^{\text{ter}}$ | 坡度、粗糙度、高差、障碍密度、可通行宽度 | Isaac Sim 地形几何 / 高度场 | 是 |
| $o_i^{\text{agg}}$ | 邻域质心、平均邻距、最近邻距离、局部分散度 | 邻居状态统计 | 是 |
| $p^{*}$ 相关量 | 最优集合点及距离统计 | 外部算法 | 否 |

当前阶段不直接输入图像、点云或 DEM patch，而采用低维手工地形特征，以降低训练难度。

---

## 8. critic 状态空间

centralized critic 输入定义为：

$$
s(t)=
\left[
s^{\text{agent}}(t),
s^{\text{team}}(t),
s^{\text{ter}}(t),
s^{\text{oracle}}(t)
\right].
$$

| 分量 | 内容 | critic 可见 | actor 可见 |
|---|---|---|---|
| $s^{\text{agent}}$ | 4 个 rover 的真值状态 | 是 | 否 |
| $s^{\text{team}}$ | $D_{\max}$、$\sigma_p^2$、$\bar p$、平均邻距等 | 是 | 否 |
| $s^{\text{ter}}$ | 全局或局部地形摘要 | 是 | 否 |
| $s^{\text{oracle}}$ | $p^{*}$、$d_i^{*}$、$\bar d^{*}$、$\|\bar p-p^{*}\|$ | 是 | 否 |

critic 使用 oracle 信息的目的为改善价值估计与训练引导，不改变执行阶段的分散策略输入。

---

## 9. 动作空间

为降低第一阶段训练难度，actor 不输出多锚点轨迹骨架，而输出低维局部子目标动作：

$$
a_i(t)=
\left[
\rho_i(t),\beta_i(t)
\right].
$$

其中：

$$
\rho_i(t)\in[0,\rho_{\max}],
$$

$$
\beta_i(t)\in[-\beta_{\max},\beta_{\max}].
$$

$\rho_i(t)$ 表示车体系下局部子目标距离，$\beta_i(t)$ 表示局部子目标方位角。对应的车体系局部子目标点为：

$$
p_{i,\text{sub}}^{b}(t)=
\begin{bmatrix}
\rho_i(t)\cos\beta_i(t)\\
\rho_i(t)\sin\beta_i(t)
\end{bmatrix}.
$$

若需要高程信息，则由 Isaac Sim 地形查询或局部地形函数补全：

$$
z_{i,\text{sub}}=H(x_{i,\text{sub}},y_{i,\text{sub}}).
$$

单智能体动作维度为：

$$
d_a=2.
$$

多智能体动作张量形状为：

$$
[N,2].
$$

| 字段 | 记号 | 维度 | 坐标系 | 约束 | 含义 |
|---|---|---:|---|---|---|
| 局部子目标距离 | $\rho_i$ | 1 | 车体系 | $[0,\rho_{\max}]$ | 决定局部前进尺度 |
| 局部子目标方位角 | $\beta_i$ | 1 | 车体系 | $[-\beta_{\max},\beta_{\max}]$ | 决定局部运动方向 |

---

## 10. 轨迹生成器

轨迹生成器输入为低维局部子目标动作：

$$
G_{\text{traj}}:\ [\rho_i(t),\beta_i(t)] \mapsto \mathcal{T}_i(t).
$$

输出为带时间戳的局部参考轨迹：

$$
\mathcal{T}_i(t)=
\{(x_{i,k},y_{i,k},z_{i,k},\psi_{i,k},t_{i,k},v_{i,k})\}_{k=1}^{K}.
$$

轨迹生成流程如下：

1. 将 $[\rho_i,\beta_i]$ 转换为车体系局部子目标点；
2. 将局部子目标点转换到世界坐标系；
3. 查询地形高度或局部高程函数，补全 $z$；
4. 使用直线段、圆弧段或二阶 Bézier 曲线生成短局部路径；
5. 根据固定速度、限速规则和地形代价分配时间戳；
6. 输出固定长度 $K$ 的 time-stamped local trajectory。

当前阶段轨迹生成器采用确定性规则，不由 RL 学习速度剖面和轨迹曲率。

---

## 11. 简化控制执行层

由于月球车底层控制接口尚未确定，第一阶段采用简化速度跟踪控制器。控制器根据局部参考轨迹生成车体速度命令：

$$
u_i(t)=
\left[
v_i^{\text{cmd}}(t),
\omega_i^{\text{cmd}}(t)
\right].
$$

其中，$v_i^{\text{cmd}}$ 为期望前向速度，$\omega_i^{\text{cmd}}$ 为期望偏航角速度。

底层适配器根据最终 rover asset 的控制接口，将 $[v_i^{\text{cmd}},\omega_i^{\text{cmd}}]$ 转换为 Isaac Sim articulation 可接受的控制目标。可选接口包括：

1. 左右轮速度目标；
2. 转向角 + 前向速度目标；
3. 关节速度目标；
4. 关节力矩目标；
5. 简化底盘控制目标。

当前阶段不冻结具体 articulation 控制接口，仅要求接口能够接收简化速度跟踪器输出。

---

## 12. 集合成功判据

集合成功采用纯几何聚集判据。设：

$$
\bar p(t)=\frac{1}{N}\sum_{i=1}^{N}p_i(t),
$$

$$
D_{\max}(t)=\max_{i,j}\|p_i(t)-p_j(t)\|,
$$

$$
\sigma_p^2(t)=\frac{1}{N}\sum_{i=1}^{N}\|p_i(t)-\bar p(t)\|^2.
$$

若在连续 $n_{\text{hold}}$ 个判定时刻内同时满足：

$$
D_{\max}(t)\le \varepsilon_D,
$$

$$
\sigma_p^2(t)\le \varepsilon_\sigma,
$$

$$
\|v_i(t)\|\le \varepsilon_v,\quad i=1,\dots,N,
$$

则判定集合成功。

其中，$\varepsilon_D$ 为最大 pairwise 距离阈值，$\varepsilon_\sigma$ 为团队分散度阈值，$\varepsilon_v$ 为速度阈值，$n_{\text{hold}}$ 为连续保持步数。

---

## 13. 奖励函数

总奖励定义为：

$$
r_t
=
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

### 13.1 自组织聚集奖励

$$
r_{\text{gather}}(t)
=
\alpha_1\big(D_{\max}(t-1)-D_{\max}(t)\big)
+
\alpha_2\big(\sigma_p^2(t-1)-\sigma_p^2(t)\big).
$$

该项衡量团队几何收缩趋势，不依赖显式集合点。

### 13.2 oracle 辅助奖励

定义平均最优点距离：

$$
\bar d^{*}(t)
=
\frac{1}{N}\sum_{i=1}^{N}\|p_i(t)-p^{*}(t)\|.
$$

oracle 辅助奖励固定采用平均距离下降量：

$$
r_{\text{oracle}}(t)
=
\alpha_3\big(\bar d^{*}(t-1)-\bar d^{*}(t)\big).
$$

该项仅用于训练塑形，不作为执行期输入。

### 13.3 能耗代理奖励

第一阶段采用代理项：

$$
r_{\text{energy}}(t)
=
-\alpha_4 L(t)
-\alpha_5 C_{\text{slope}}(t)
-\alpha_6 C_{\text{turn}}(t)
-\alpha_7 C_{\text{terrain}}(t).
$$

其中，$L(t)$ 为路径长度或位移代价，$C_{\text{slope}}$ 为坡度代价，$C_{\text{turn}}$ 为转向代价，$C_{\text{terrain}}$ 为高风险地形代价。

后续若获得关节力矩和关节速度，可扩展为物理能耗近似：

$$
C_{\text{energy}}(t)
=
\sum_{\tau=t}^{t+\Delta t_p}
\sum_q
|\tau_q(\tau)\dot q(\tau)|\Delta t_c.
$$

该扩展不作为第一阶段要求。

### 13.4 安全惩罚

$$
r_{\text{safety}}(t)
=
-\alpha_8 C_{\text{obs}}(t)
-\alpha_9 C_{\text{agent}}(t)
-\alpha_{10} C_{\text{near}}(t).
$$

其中，$C_{\text{obs}}$ 为障碍碰撞惩罚，$C_{\text{agent}}$ 为车间碰撞惩罚，$C_{\text{near}}$ 为危险近距惩罚。

### 13.5 运动质量奖励

由于第一阶段动作空间已降为 $[\rho,\beta]$，运动质量项简化为：

$$
r_{\text{motion}}(t)
=
-\alpha_{11}C_{\text{turn}}(t)
-\alpha_{12}C_{\text{subgoal}}(t).
$$

其中，$C_{\text{turn}}(t)$ 用于抑制过大局部转向，$C_{\text{subgoal}}(t)$ 用于抑制长期输出过小 $\rho_i$ 导致的停滞行为。

### 13.6 一致性奖励

$$
r_{\text{consistency}}(t)
=
-\alpha_{14}
\sum_{i=1}^{N}
\left\|
\begin{bmatrix}
\rho_i(t)\\
\beta_i(t)
\end{bmatrix}
-
\begin{bmatrix}
\rho_i(t-1)\\
\beta_i(t-1)
\end{bmatrix}
\right\|_2^2.
$$

### 13.7 终端奖励

$$
r_{\text{terminal}}(t)=
\begin{cases}
+R_{\text{succ}}, & \text{集合成功},\\
-R_{\text{fail}}, & \text{碰撞/越界/超时},\\
0, & \text{其他}.
\end{cases}
$$

---

## 14. 网络结构

### 14.1 actor 网络

actor 采用参数共享结构。每个智能体共享同一套策略网络参数。输入为局部观测 $o_i(t)$，输出为 2 维连续动作：

$$
a_i(t)=[\rho_i(t),\beta_i(t)].
$$

actor 由四个编码分支和一个共享 MLP 主干组成：

| 分支 | 输入 | 作用 |
|---|---|---|
| ego encoder | $o_i^{\text{ego}}$ | 编码自车状态 |
| neighbor encoder | $o_i^{\text{nbr}}$ | 编码邻居共享状态 |
| terrain encoder | $o_i^{\text{ter}}$ | 编码局部地形手工特征 |
| local-consensus encoder | $o_i^{\text{agg}}$ | 编码局部聚集态势 |

输出头为连续动作分布参数。动作采样后通过有界映射约束到：

$$
\rho_i\in[0,\rho_{\max}],
\quad
\beta_i\in[-\beta_{\max},\beta_{\max}].
$$

### 14.2 critic 网络

critic 采用 centralized value network。输入为全局状态 $s(t)$，输出团队共享状态价值：

$$
V_\phi(s(t)).
$$

critic 输入包含全部智能体真值状态、团队几何统计量、地形摘要和 oracle 辅助信息。当前阶段智能体数量固定为 4，因此 critic 可采用全局拼接输入和 MLP 主干结构。

---

## 15. SKRL-MAPPO 训练流程

训练流程包括：

1. Isaac Lab 创建并行多智能体任务环境；
2. 每个环境包含 4 个 rover 智能体；
3. actor 根据局部观测输出 $[\rho,\beta]$；
4. Isaac Lab 环境内部将动作转换为参考轨迹和控制命令；
5. Isaac Sim 推进物理仿真；
6. Isaac Lab 计算奖励、终止标志与下一观测；
7. SKRL 收集 rollout；
8. MAPPO 使用 centralized critic 计算优势并更新 actor/critic。

---

## 16. 软件架构

建议项目结构如下：

```text
project/
├── source/
│   └── lunar_rover_tasks/
│       ├── lunar_rover_tasks/
│       │   ├── tasks/
│       │   │   └── multi_rover_gathering/
│       │   │       ├── __init__.py
│       │   │       ├── gathering_env.py
│       │   │       ├── gathering_env_cfg.py
│       │   │       ├── observation.py
│       │   │       ├── reward.py
│       │   │       ├── termination.py
│       │   │       ├── action_interpreter.py
│       │   │       ├── trajectory_generator.py
│       │   │       └── simple_controller.py
│       │   ├── assets/
│       │   │   └── rover/
│       │   └── utils/
│       └── setup.py
├── scripts/
│   ├── reinforcement_learning/
│   │   └── skrl/
│   │       ├── train.py
│   │       └── play.py
│   └── tools/
├── configs/
│   ├── env/
│   ├── agent/
│   └── terrain/
├── docs/
└── outputs/
```

模块职责如下：

| 模块 | 职责 |
|---|---|
| `gathering_env.py` | Isaac Lab 多智能体任务环境 |
| `gathering_env_cfg.py` | 场景、机器人、地形、仿真步长、并行环境数量配置 |
| `observation.py` | actor 观测与 critic state 构造 |
| `reward.py` | 奖励函数计算 |
| `termination.py` | 成功/失败判据 |
| `action_interpreter.py` | $[\rho,\beta]$ 到局部子目标的转换 |
| `trajectory_generator.py` | 局部子目标到 time-stamped trajectory 的转换 |
| `simple_controller.py` | 简化速度跟踪控制器 |
| `assets/rover/` | 月球车 USD / URDF 资产 |
| `scripts/reinforcement_learning/skrl/` | SKRL-MAPPO 训练与评估脚本 |

---

## 17. 实验配置模板

```yaml
task:
  name: Isaac-MultiRover-Gathering-Direct-v0
  n_agents: 4
  scene_dim: "2.5D/3D"
  explicit_goal_in_execution: false
  oracle_optimal_gather_point_in_training: true
  docking_considered: false

simulation:
  simulator: IsaacSim
  framework: IsaacLab
  device: cuda
  headless: true
  num_envs: N_env
  physics_dt: dt_phys
  control_decimation: n_decimation
  episode_length_s: T_episode

terrain:
  type: lunar_heightfield_or_mesh
  use_handcrafted_features: true
  features:
    - slope
    - roughness
    - local_height_diff
    - obstacle_density
    - traversable_width

robot:
  asset_type: usd_or_urdf
  control_interface: TBD
  base_frame: base_link
  wheel_joints: TBD
  sensor_set: TBD

communication:
  mode: neighbor_state_sharing
  radius: R_comm

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
  geometry_method: line_or_arc_or_bezier
  terrain_height_query: true
  assign_timestamp: true
  assign_reference_speed: true

low_level_control:
  first_stage_mode: simplified_velocity_tracking
  command_type: body_twist
  command_fields: [v_cmd, omega_cmd]
  isaac_articulation_interface: TBD

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

---

## 18. 评估指标

评估指标分为四类。

任务指标包括集合成功率、平均完成时间、最大 pairwise 距离、团队分散度、平均邻距和碰撞率。

oracle 最优性指标包括最终平均最优点距离 $\bar d^{*}(T)$、最终最大最优点距离、团队质心到 $p^{*}(T)$ 的距离和累计 oracle gap。

轨迹指标包括局部子目标变化幅度、轨迹长度、转向幅度、轨迹生成失败率和相邻规划周期动作差异。

闭环指标包括速度跟踪误差、Isaac Sim 物理执行下的任务成功率和碰撞/越界比例。

---

## 19. 实施计划

### 19.1 第一阶段：最小可训练 Isaac Lab 环境

目标是跑通 Isaac Lab + SKRL-MAPPO 的最小闭环。

本阶段实现：

1. 4 个 rover 实例加载；
2. 自组织集合任务 reset / step；
3. 局部观测构造；
4. centralized critic state 构造；
5. 低维动作 $[\rho,\beta]$；
6. 简化轨迹生成器；
7. 简化速度跟踪器；
8. 几何集合奖励；
9. oracle 平均距离下降量奖励；
10. 基础碰撞与终止判据。

本阶段不实现：

1. 多锚点轨迹输出；
2. 速度剖面学习；
3. 关节力矩级能耗建模；
4. 可学习通信；
5. 精细对接/拼接。

### 19.2 第二阶段：地形特征与安全约束增强

加入坡度、粗糙度、局部高差、障碍密度和可通行宽度等地形特征，增强安全惩罚与地形代价。

### 19.3 第三阶段：控制接口替换

在月球车资产和底层控制接口明确后，将简化速度跟踪器替换为对应 articulation 控制接口，例如轮速控制、转向角控制或力矩控制。

### 19.4 第四阶段：规划输出增强

在 2 维动作收敛后，逐步增强规划输出：

$$
[\rho,\beta]
\rightarrow
[\rho,\beta,v_{\text{ref}}]
\rightarrow
[(\rho_1,\beta_1),(\rho_2,\beta_2)]
\rightarrow
\{(\Delta s_m,\Delta l_m,\Delta h_m,\Delta\psi_m)\}_{m=1}^{M}.
$$

### 19.5 第五阶段：实验与消融

开展以下实验：

1. 无 oracle 辅助奖励 vs 有 oracle 辅助奖励；
2. critic 不接入 oracle 信息 vs critic 接入 oracle 信息；
3. 不同通信半径；
4. 不同动作维度；
5. 不同地形复杂度；
6. 不同能耗建模方式；
7. 简化速度控制 vs articulation 控制接口。

---

## 20. 当前冻结结论

当前技术路线冻结为：

1. 采用 Isaac Sim / Isaac Lab 构建多月球车自组织集合训练环境；
2. 采用 SKRL 实现 MAPPO 训练；
3. 执行阶段不显式提供集合点；
4. 通信机制采用邻居状态共享；
5. actor 输出低维局部子目标动作 $[\rho,\beta]$；
6. 轨迹由确定性轨迹生成器生成；
7. 第一阶段底层执行采用简化速度跟踪控制器；
8. critic 训练期接收全局状态与 oracle 辅助信息；
9. oracle 辅助奖励采用平均距离下降量；
10. 底层 rover articulation 控制接口暂不冻结；
11. 后续在训练稳定后再扩展动作维度、控制精度和能耗建模。

---

## 21. 当前仍保留的待定项

| 待定项 | 当前处理方式 | 后续处理 |
|---|---|---|
| rover 资产格式 | `usd_or_urdf` 占位 | 根据模型来源确定 |
| 底层控制接口 | `TBD` | 根据 articulation 结构确定 |
| 轨迹几何生成方式 | line / arc / Bézier 均可 | 第一版建议直线或圆弧 |
| 地形高度查询方式 | Isaac Sim 场景查询 | 根据地形资产格式实现 |
| 奖励权重 | 符号占位 | 训练阶段调参 |
| 控制周期与规划周期 | 符号占位 | 根据仿真效率和稳定性设定 |

---
