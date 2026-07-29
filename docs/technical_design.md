# 技术设计

本文是短版技术摘要，只定义任务建模、信息边界、观测/状态/action、轨迹控制链、reward、网络接口和评估判据。长期技术路径管理见主目录 `多月球车自组织集合局部参考轨迹规划技术文档.md`，工程目录见 [scaffold.md](scaffold.md)，当前实施计划见 [implementation_plan.md](implementation_plan.md)。

## 任务与路线

任务目标是让 4 个同构 rover 在有限通信和局部感知条件下自组织集合到隐含目标区域。当前训练主路径是 torch-vectorized proxy 环境；Isaac Sim / Isaac Lab / PhysX 当前用于 checkpoint 级 high-fidelity closed-loop evaluation，不参与每次 PPO/MAPPO 采样更新。

每个 proxy episode 在 reset 时先固定该环境的 procedural terrain runtime，再采样初始 rover 状态，并执行一次地形约束的粗到细最优集合点搜索。搜索结果作为该 episode 固定的训练期 oracle，不在每个 control step 随 rover 位置重新漂移。

核心闭环：

```text
actor observation
-> policy 输出归一化 action
-> [rho, beta] 局部子目标
-> 局部参考轨迹
-> 简化速度控制
-> proxy 运动学状态更新
-> reward / termination / metrics
```

## 信息边界

- 默认 Actor 执行期只使用自车、邻居、地形手工特征和局部聚合特征。
- 默认 schema 不包含 `p*`、oracle 距离或 oracle 距离下降量。显式执行目标 schema 只额外广播车体系下的相对向量和距离，不提供世界坐标、搜索代价或诊断字段。
- Centralized critic state 和 reward shaping 可以使用训练期 oracle 信息。
- 视觉观测当前不进入 policy input；地形以车体系局部结构化网格进入策略。
- Oracle 搜索可以读取该 episode 的全局 rover 真值位置和 procedural terrain runtime；它属于训练期特权计算，不改变去中心化执行边界。

## Actor Observation

默认 schema 为 `ego_v3_local_terrain_grid`，形状为：

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

`ego_v6_gather_slot_goal` 是显式执行目标的 89 维扩展：在上述 86 维后追加 `[local_dx, local_dy, normalized_distance]`。环境先搜索真实地形可行 `oracle_point`，再在其周围构造等角对称槽位，并用 reset 时的最小总行驶距离排列把槽位固定分配给 rover。各槽位的算术均值严格等于 `oracle_point`；因此策略学习的是到自身槽位的局部控制，而不是将所有车压向同一几何中点。该字段没有世界系坐标、agent ID、oracle 目标函数或平整度测量。`ego_v7_gather_site_and_slot_goal` 将共同搜索点三元组与槽位三元组拼为 92 维输入；它是已实现的诊断 schema，exp068 的 BC 闭环结果差于 v6，当前不作为训练主线。`task.execution_slot_reward_target` 默认关闭；显式启用后只将 dense oracle-progress 的距离目标从共享 `oracle_point` 改为每辆 rover 的固定槽位。该开关不改变搜索、Critic、实际质心平整度或终止语义。

`observation.communication_radius` 是当前唯一允许从 experiment YAML 覆盖的 observation 字段。取值 `>0` 表示有限通信；取值 `<=0` 表示临时取消通信距离限制，所有非自身 rover 都作为可见邻居参与 neighbor slots、aggregation、visible-local teacher 和子目标过滤器可见邻居计算。`max_neighbors`、`ego_dim`、`neighbor_dim`、`terrain_dim` 和 `aggregation_dim` 会改变模型输入接口，本轮不开放配置覆盖。

地图尺度由 `safety.world_xy_limit` 和 `terrain.crater_field_size` 控制。新工程探针把 `world_xy_limit=12.5`、`crater_field_size=25.0`，对应 `25 m × 25 m` 训练区域。Actor 的局部地形网格仍保持 `5×5×2=50` 维，不随地图扩大自动扩展感知面积。

为了避免 25m 地图只在边界上变大而 reset 分布仍停留在旧小范围，训练环境新增 `initial_state` 配置。默认仍为旧的 `3–4 m` 环形初始队形、中心 `±1 m`、jitter `0.35 m`；新长训可显式扩大 `spawn_radius_min/max` 和 `center_xy_range`，但这不改变观测/状态接口。

`initial_state` 还支持训练期课程：训练脚本可设置 `progress_timestep_override`，让 reset 分布从 `curriculum_start_*` 线性过渡到目标分布。独立 checkpoint eval 默认不设置该 override，因此评估目标始终是最终难度，而不是课程早期难度。

## Critic State

Critic state 形状为：

```text
(num_envs, state_dim)
```

它包含全部 rover 真值状态、队形几何信息、地形摘要和仅训练使用的 oracle 特征。该信息只服务 centralized critic、reward shaping 和评估指标，不进入 actor 执行期输入。Critic 总维度保持 54；地形 5 维摘要改为平均绝对高差、最大上升、最大下降、平均风险和最大风险。

## 最优集合点与平整度

训练期 `oracle_point` 不再直接取几何中点或几何中位数。默认 `terrain_aware_multiresolution` 搜索在初始 rover 包围盒外扩 `1.5 m` 的区域内执行：

1. 评估 `9×9` 粗网格，并加入几何中位数和团队质心种子；
2. 围绕当前最优点执行两层 `5×5` 局部细化；
3. 搜索边界预留完整集合 footprint，并裁剪到 `world_xy_limit`；
4. 优先在满足平整度硬约束的候选中选择目标函数最小值；当 `robustness_radius>0` 时，该硬约束要求候选中心以及 `robustness_samples` 个等角质心偏移的完整圆盘全部合格，因而搜索的是可执行平整盆地而非偶然平整的单点；
5. 局部搜索无可行点时，默认扩展到预留 footprint 后的全部世界边界：对 `33×33` 粗网格执行完整约束与目标评估，保留其中最优的 32 个 beam 种子，再执行两层 `5×5` 多分辨率细化；
6. 全局回退仅处理局部失败环境，并按每批 8 个环境限制峰值显存；若全局仍无可行点，则返回惩罚增强目标最小的退化点并标记 `feasible=false`。

候选目标函数联合最小化团队平均行进距离、最远 rover 行进距离、各车到候选点的路径地形风险、路径高差和平整度代价。候选及实际集合位置使用同一个世界系圆盘定义：

```text
radius: 0.75 m
sampling: center + 3 rings x 12 samples = 37 points
height_range <= 0.18 m
max height gradient <= 0.25
```

`0.25` 是无量纲高度梯度模长，对应约 `14.0°` 坡角。`robustness_radius=0` 保持单中心搜索；正值则要求中心和环上全部执行偏移均满足该圆盘约束，并把最坏偏移的高度范围和坡度写入搜索 telemetry。Oracle 点与 terrain runtime 在 episode 内固定；实际成功位置则始终取当前团队质心，并在每一步状态推进后重新评估其圆盘。搜索包络只是 reset 时的保守约束，绝不能替代或放宽实际质心 hard gate。最终 `feasible=false` 的退化点仅用于保持 telemetry 和张量结构连续，对应环境的 oracle 距离进度 shaping 置零，避免奖励把 rover 拉向不满足硬约束的位置。默认机制不向 Actor 或 Critic 增加字段，接口为 `86 / 54`；`ego_v6_gather_slot_goal` 是受显式契约保护的执行期例外，Actor/Critic 为 `89 / 54`，只读取自身槽位的车体系相对特征。

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

当前 proxy 动力学是 2D/2.5D torch-vectorized 运动学状态更新。地形开启时会查询 procedural heightfield / crater proxy 特征并施加速度缩放，但不包含质量、惯量、轮地接触、打滑、沉陷、悬挂或 PhysX contact。

本轮新增的工程路径允许通过配置切换：

- `trajectory_generator.geometry_method: line | quintic`。`line` 保留旧实验兼容；`quintic` 使用 2D quintic Hermite 曲线，起点切向为当前 yaw，终点切向指向 subgoal，输出 `Trajectory` 接口不变。
- `low_level_control.kinematic_model: unicycle | bicycle`。`unicycle` 保留旧默认；`bicycle` 将控制器输出的 yaw-rate demand 转成 steering clamp，并用 terrain speed scaling 后的前向速度计算实际 yaw-rate。
- `formation_center_correction` 是默认关闭的末端共同平移器：只有 dmax 和 dispersion 到达配置倍数以内时，才把裁剪并缩放后的“固定槽位均值减实际质心”世界系偏移加到每个 rover 子目标。`formation_center_correction_require_flatness_failure=true` 会再要求上一状态实际质心的 37 点平整圆盘失败；共同平移严格保留槽位相对几何与两两间距，且只为重新进入真实平地服务。搜索点、实际质心平整 gate 和 success predicate 均不被替换。`terminal_slot_capture` 则可选地把子目标向各自专属槽位 blend，仍从不把多车目标合并为几何中点。两者均位于 filter 后、轨迹与 safety projection 前；当前后验反例表明 slot capture 应保持关闭。

这仍是 proxy，不包含真实轮地接触或 PhysX contact。

## Reward 与终止

Reward 由以下部分组成：

- 自组织集合 reward：鼓励队形收缩和距离目标集合区域更近。
- Oracle 辅助 reward：仅训练期使用，用于距离进展 shaping。
- 能耗代理 reward：约束过大的速度和角速度命令。
- 安全惩罚：约束 rover 间最小距离和碰撞。
- Actual-centroid flatness reward：仅在队形接近几何成功区间时，对实际团队质心圆盘的平整度改善提供 shaping。
- 运动质量 reward：鼓励平滑、有效的运动。
- 一致性 reward：鼓励局部协同。
- 终端 reward：根据成功、碰撞、超时等终止原因给出 episode 级反馈。

Actual-centroid flatness shaping 在每一步状态推进后计算。设实际质心圆盘的高度范围为 \(\Delta h_t\)、最大高度梯度为 \(s_t\)，配置硬阈值为 \(H_{\max}\) 与 \(S_{\max}\)，则：

\[
C_t=\operatorname{clip}\left(
\max\left(\frac{\Delta h_t}{H_{\max}},\frac{s_t}{S_{\max}}\right),
0,3
\right).
\]

\(C_t\le 1\) 与 actual-centroid flatness hard gate 完全等价。以几何 success dmax 阈值 \(d_g\) 和激活倍数 \(m>1\) 定义：

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
\lambda_p(P_{t-1}-P_t)
-\lambda_e a_t\operatorname{ReLU}(C_t-1).
\]

该项在 \(d_{\max}\ge m d_g\) 时关闭，在 \(d_g<d_{\max}<m d_g\) 时线性激活，在 \(d_{\max}\le d_g\) 时完全激活。进展量是 gated potential 差 \(P_{t-1}-P_t\)，因此跨 activation 边界时使用前后两步各自的 \(aC\)，而不是用当前 \(a_t\) 统一缩放裸 cost 差。任意闭合往返满足 \(\sum_t(P_{t-1}-P_t)=P_0-P_T=0\)，消除了“在高 activation 区获得平整化正奖励、在低 activation 区低成本撤销”的循环；excess 项始终非正，不会重新引入正循环收益。

exp064 使用 \(H_{\max}=0.18\,\mathrm{m}\)、\(S_{\max}=0.25\)、\(d_g=1.25\,\mathrm{m}\)、\(m=2.0\)、\(\lambda_p=2.0\)、\(\lambda_e=0.02\)，对应 `2.50 m -> 1.25 m` 的激活区间；总 reward 通过 `reward.weights.flatness=1.0` 加入该项。项目默认配置把 flatness weight、progress coefficient 和 excess coefficient 都设为 `0.0`，因此该 shaping 默认关闭，不改变既有实验语义。

该 shaping 的位置与 oracle reward 严格分离：它只读取状态推进前后的实际 `metrics.centroid`、`metrics.dmax` 和当前 terrain runtime 的圆盘地形，不读取 `oracle_point` 或 oracle 搜索结果。训练 telemetry 记录 `centroid_flatness_cost_mean`、`centroid_flatness_progress_mean`、`centroid_flatness_activation_mean`；其中 progress 是 \(P_{t-1}-P_t\)，不是裸 \(C_{t-1}-C_t\)。reward breakdown 记录 `reward_raw_flatness`、`reward_weight_flatness` 与 `reward_contribution_flatness`。这些字段只用于训练奖励与诊断，不进入 Actor observation 或 Critic state；Actor/Critic 维度继续保持 `86 / 54`，不存在执行期 oracle 泄漏。

集合 instant success 现在同时要求 `dmax`、dispersion、全部 rover 速度、可选最近邻安全间距以及实际团队质心圆盘平整度通过。平整度使用半径 `0.75 m` 的 37 点圆盘，并要求 `height_range <= 0.18 m`、`max_gradient <= 0.25`；任一步不满足都会重置 success hold count。终止条件包括连续保持后的集合成功、碰撞、安全边界失败和 episode timeout。不要只用训练 reward 判断成功，严格结论以机器可读评估结果为准。

## 网络与训练接口

- Actor 是同构多智能体策略，输入去中心化 observation，输出每车 `[rho, beta]`。默认兼容路径为 `mlp_v1`；新工程路径 `branched_v1` 将 86 维 observation 拆成 ego、neighbor、terrain、aggregation 四个编码分支，再接共享 MLP 主干。
- Critic 使用 centralized state，服务 MAPPO/PPO 训练。默认兼容路径为 `mlp_v1`；新工程路径 `structured_v1` 将 54 维 state 拆成 agent states、team stats、terrain summary、oracle state 后再做 value trunk。
- SKRL-MAPPO 训练通过 `MultiRoverGatheringSKRLEnv` 和 `isaaclab-multi-agent` wrapper 接入；该 wrapper 是接口层，不代表 PhysX 训练。
- Checkpoint metadata 必须记录 `observation_schema_version`、`actor_obs_dim`、`critic_state_dim`、Actor/Critic 架构、运动学模型和轨迹生成方法。旧 schema 或缺少 schema metadata 的 checkpoint 明确拒绝，不自动迁移；当前 schema 但缺少架构 metadata 的旧 checkpoint 按 `mlp_v1` 兼容路径加载。

## 评估判据

Proxy strict gate 默认写成：

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

地形平整度成功门改变了 `success_rate` 和 `timeout_rate` 的统计语义，但不改变模型输入维度。旧纯几何 gate 下的历史成功率或 strict pass 不能与新结果直接比较；做算法对照时，必须用同一 terrain runtime、同一 oracle 搜索配置和同一 centroid 平整度 gate 重新评估 checkpoint。

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
