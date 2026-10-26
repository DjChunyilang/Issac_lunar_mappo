# 当前训练与规划接口规范

更新：2026-10-25。主体保留固定月面接口；最新[active 配置](architecture/planning_pipeline.md)在坑分档基础上完成MPPI车身覆盖修复，百万步训练已结束。旧配置按原接口复现。旧尺度、诊断和历史候选接口全文见[完整快照](archive/formal_training_transition_2026-10-22/docs/interface_spec.md)。

## 配置与版本身份

| 接口 | 当前身份 |
| --- | --- |
| 固定训练分布 | `rendezvous_fixed_lunar_1km_v1`，课程／晋级关闭 |
| 任务 | `finite_task_upstream_v1`，地形失败语义 `known_terrain_terminal_v1` |
| 障碍输入／观测 | 当前`local_obstacle_input_v2`／`obstacle_ground_known_actor_v1`；旧v1保留 |
| A1 概率契约／环境动作桥接 | `t11_polar_hold_contract_v1`／`polar_hold_upstream_v1` |
| 决策时间 | `t11_decision_time_v1` |
| 规划协议／适配器 | `planner_protocol_v2`／`mppi_upstream_v4` |
| 当前风险／安全模式 | `cost_reward_only_v1`／`projection_disabled` |

源码、配置、物理 profile、地图、atlas、观测、动作、规划和时间身份一起参与恢复校验。引入新观测的历史训练使用全新 Actor／Critic／Adam／价值归一化状态；本次MPPI修复没有改变观测或训练网络。历史版本不能依靠相同维度或文件名静默恢复。

## 观测与信息边界

Actor 输入形状 `[env,4,523]`，Critic `[env,1439]`。维度由实际 encoder 与配置构建。Actor 仅含本车状态、合法收到的邻车信息和本车格记忆，Critic 可接实际团队信息及驻留计数；Oracle 预留的九维保持零。

局部主区域为 112×3，通道表示障碍接近风险、地面 cost、known 交叠面积比例。障碍标签 0.2 m，普通地形 0.8 m；粗格来源完整进入视场后才能揭示对应子格，未知不填成无障碍。全图和 atlas 只由环境／感知所有者持有；`GridTerrain` 仅持已揭示的只读局部 tile。排队快照按内容不可变，恢复保存私有记忆、地图、anchor、时间和 revision。

邻车基础状态、目标、预测、有效性和年龄按同一 sender 绑定；缺失或过期消息不能用真值补齐。任务剩余时间为归一化本回合时钟，Actor 每车一份、Critic 每环境一份；末状态与 reset 后状态不能混用。

## 动作、时序与规划

A1 使用 MOVE／HOLD 条件分布。MOVE 的极坐标连续分量保存原采样 latent 并使用相应变换 log-prob；HOLD 仅计算离散分支概率，连续分量不进入该样本的策略损失。规划回退、执行限幅与请求续发不产生新的策略概率样本。

策略每 5 s 采样一次、推进一次 GRU，目标冻结在接收车固定 odometry frame；每 1 s 刷新合法约束并续发／重规划，0.2 s 执行。请求经过既有延迟队列，模拟延迟按底层步量化。HOLD 冻结请求时位姿并按执行限制减速；HOLD、实际停稳和任务成功分别判定。

| 对象 | 当前内容 |
| --- | --- |
| `UpstreamIntent` | `goal_xy[2]`、`hold`、`timestamp`、`expires_at`；可选 `terminal_yaw`、`speed_cap`，`stop_at_goal` 默认 false |
| `UpstreamPlannerInput` | env／agent 身份、本车 pose／实际速度、意图、只读局部地形、sender 邻车预测、上一实际控制、now、显式时间网格、工作区／物理 spec；可携带信息时刻与实际控制历史 |
| 输出轨迹 | 复用 `TimedTrajectory` 与 `PlannerFeedback`；状态 `[41,3]`，控制和实际速度 `[40,2]`，时间 `[41]` |
| 生命周期 | `plan`、`reset`、`state_dict`、`load_state_dict`；公开批调度保持输入顺序和实例状态隔离 |

距离为米，角度为弧度，时间为回合内秒。预测控制保持轮速等效 `[v,omega]` 语义，实际速度单独保存；前一步输入必须是实际下发控制。请求在有效期内使用，过期进入既定停车行为，不延长旧目标寿命。规划固定 40×0.2 s，不能套用旧 T03 的 15 步形状。

## 当前风险与实际执行

当前 soft-hazard 模式保留 MPPI 既有 prescreen、动力学与执行限制，关闭预测完整足迹／停车储备和每步安全投影／停车准入。完整复核及执行 verified 字段为 false，规划 `valid` 不代表完整足迹认证；没有当前有效候选时不把旧计划自动当作有效结果。

`oriented_rectangle_cover_disks_v1` 用 54 个覆盖圆盘估计已观测障碍间距，仅用于软 cost／奖励。near／overlap 分别乘预测或实际时长；实际奖励在执行后位姿查询运动前的合法障碍快照，取四车均值。标签接近或覆盖本身不新增终端类型。

实际地形失败与停车仍用即时传感原网格的 713 个足迹支持点及既有物理判据。前进硬上限 10 km/h、倒车上限 0.5 m/s；完整执行限制以当前解析物理 profile 为准。旧 strict 路径、预测足迹复核及安全投影仍可供历史诊断，不能默认带入当前训练。

## 陨石坑分档候选接口

候选实验为 [upstream_1km_crater_classes_v1.yaml](../configs/experiment/upstream_1km_crater_classes_v1.yaml)，引用[同名地形配置](../configs/terrain/upstream_1km_crater_classes_v1.yaml)，输入身份为 `local_obstacle_input_v2`。以下直径阈值是本轮 **Proxy 任务假设**，不是轮地动力学测得的车辆通过极限；正式 `local_obstacle_input_v1` 配置和历史 checkpoint 语义保留。

| 坑口直径 D | 驾驶与反馈 |
| --- | --- |
| D < 0.5 m | 小坑，不计坑标签风险和附加惩罚 |
| 0.5 m ≤ D < 1.8 m | 中坑，允许通过，局部风险幅值为 0.1；规划轻代价与执行轻罚按穿入比例和时长计算 |
| D ≥ 1.8 m | 大坑，规划拒绝侵入；执行后实际车身投影与坑口出现正面积交叠，判为 `terrain_failure`，复用一次性失败惩罚 |

驾驶表面由原始底图、岩石层及既有残差层组成，去掉全部人工 crater 层；中小坑不再通过旧坑壁坡度、粗糙度、台阶或底盘间隙规则触发驾驶失败。底图与岩石的原有检查保留。停车审查和最终成功条件继续使用原始含坑地形，因此“允许驶过”不等于“允许在该处停车完成任务”。这些检查都是平面 Proxy 的几何规则，不代表已经模拟触底、掉坑或翻车动力学。

环境的大坑判定使用实际 `5.015 × 3.2 m` 车身矩形与坑口圆的精确交叠，不加 0.3 m 安全余量，几何相切不算进入。历史规划配置使用保守圆盘覆盖估距，最新显式SAT配置见下文；MPPI预筛以及最终重生成轨迹、停车／刹车分支均检查已观测大坑，侵入返回`large_crater`。这些配置不恢复其他预测完整足迹拦截。

私有格分别保存岩石／大坑、中坑和大坑的距离场，仅从已揭示标签更新；小坑不进入这些风险场。Actor 523／Critic 1439 维及三通道布局不变，障碍风险取原风险与 `0.1 × 中坑风险` 的较大值。规划、观测和奖励不读取隐藏坑真值；只有环境执行终止及 reset 出生筛选使用完整 atlas 的精确大坑几何。出生筛选先排除实际车身侵入大坑的候选，再使用同一派生驾驶表面审查。

中坑规划代价为 `0.5/s × 穿入比例`；原始执行奖励为 `-0.05/s × 四车平均穿入比例`，查询执行后位姿在运动前合法快照中的距离。每个团队回合该奖励分量累计最多扣 2 分；无效学习步不计入，异步 reset 只清相应环境额度，完整恢复保存已扣额度。输入配置指纹、atlas 身份和规划／奖励 metadata 隔离新旧语义，出生采样器另保存 `birth_driving_atlas` 身份；相同网络维度不允许跨语义静默恢复。

## MPPI车身覆盖调参接口

[upstream_1km_mppi_body_geometry_v1.yaml](../configs/experiment/upstream_1km_mppi_body_geometry_v1.yaml)在 `upstream_scale.mppi_parameters` 增加两个可选有限数值：`footprint_margin_m >= 0` 和 `footprint_cover_spacing_m > 0`，仅允许 `mppi_upstream_v4` 且启用局部障碍格的配置。未指定的值继承物理余量／地形覆盖间距；两者都未指定时，覆盖几何和metadata保持历史值，新字段不写入metadata。

指定任意一个参数时，规划使用 `planning_footprint_cover=cell_center_disks_v1`：圆心位于矩形网格单元中心，半径为单元半对角线，覆盖完整车身。当前选定零余量与0.2米间距，共416点。MPPI批量预筛、最终重生成轨迹和HOLD／刹车审查均使用同一规划视图；原视图的tile共享，覆盖点不原地修改，不影响Actor观测、执行奖励或实际地形失败判定。

几何参数、覆盖身份及采样参数参与规划器恢复检查。原配置继续原义恢复，新配置不能静默读取旧参数的规划热状态；跨配置使用旧策略需要显式迁移并新建run。本轮诊断先完整恢复旧状态、逐值核对，再施加记录在案的参数干预，0学习更新。实现与CPU／CUDA、大坑侵入、恢复检查证据见[执行修复](experiments/t11_mppi_execution_repair_2026-10-25.md)。

## MPPI查询优化与SAT接口

[upstream_1km_mppi_grid_sat_v1.yaml](../configs/experiment/upstream_1km_mppi_grid_sat_v1.yaml)在`upstream_scale.mppi_parameters`显式设置`large_crater_query=observed_grid_sat_v1`，仅用于V4分档局部格，要求`footprint_margin_m=0`；选用512采样×3轮。该字段进入规划metadata和恢复校验；不设置时省略metadata字段并保持旧圆盘行为。

SAT将已揭示大坑格的严格并集合并为矩形，再用空间桶与四轴分离测试判断实际车身是否正面积侵入。CPU和CUDA共享几何，相切不算侵入。大坑查询此时只返回正／负硬检查符号，不能解释为米制距离；普通障碍和中坑仍用原距离场与416点覆盖。两个软通道共用坐标变换／格索引，评分算术保持。未知格不从atlas补齐，环境实际大坑判定保持矩形—圆。

`GridTerrain.obstacle_distance(..., include_known=True)`默认保持原距离和完整已知范围标志；最终大坑审查传`include_known=False`，第二项返回`None`，省去原本被丢弃的计算。后续完整reviewer若启用，仍独立检查已知范围，实际感知、停车与失败规则保持。

T11的`make_collector`和入口校验向`runtime_config`传`iterations=None`，表示继承实验YAML轮数；训练与独立评估均遵守该值。通用局部诊断调用不传该参数时保留历史2轮默认。旧两轮训练配置保持，新SAT配置实际运行3轮；不把旧2轮热状态改标为3轮恢复。

`UpstreamMPPIKernel.plan_batch(..., record_samples=True)`及独立规划池默认完整采样证据；`run_upstream_t11.make_collector`将`env.pool.record_samples`设为`False`，GPU先按原稳定顺序选择待审控制，只回传候选及统计。此开关不改变选择、评分、随机数或热状态身份；关闭时`last_samples=None`、`full_samples_recorded=false`，需要完整诊断时须在采样前显式开启。证据见[查询与迭代优化](experiments/t11_mppi_query_optimization_2026-10-25.md)。

## 本轮短训奖励与评估

新训练配置为[upstream_1km_far_crater_reward_v1.yaml](../configs/training/upstream_1km_far_crater_reward_v1.yaml)。`continuous_cost_scale=0.02`仅缩放旧非势持续扣分：远距／离散、平坦度超额、能量、地形、车间距、障碍near／overlap。势差、事件奖罚、中坑0.05/s及2分上限保持；默认尺度1.0省略新增metadata字段，旧奖励状态与算术保持。新尺度及`continuous_cost_scope`参与恢复身份校验。

`evaluation_mode=bounded_fixed_window_v1`只改变评估安排，训练回合仍1500 s。初始化／最终各在固定4起点运行sample和mean各60秒，各预留2400、共4800独立评估步，无中途与未见图评估。`full_episode_evaluation=false`；记录`observed_seconds`、`censored`、窗口失败率、Dmax／离散度缩减，`mean_actual_speed`用实际路程除各自有效时间。旧`mean_applied_speed`保留回放语义，新结果不用该字段；短窗不满足完整收敛判据。

## 终止、奖励与优势

`terminal_reason`、`bootstrap_allowed`、`learning_valid` 分别表示结束原因、尾值许可和学习样本有效性。真实任务终端不 bootstrap；有效缓冲切分保留尾值，不 reset 环境或 GRU。异步结束只重置相应环境和 recurrent 状态。

已知行驶地形超限为 `terrain_failure`（code 6），仍是有效学习终端，后继势／bootstrap 为零；缺测或无效环境为 `environment_invalid`（code 5），不对无效尾状态估值、不进入学习损失。失败优先于成功，合法状态在截止步完成驻留仍可成功。

成功由实际状态的队形、间距、共同区域、地形、速度与姿态漂移条件共同判定，连续保持 5 s／25 步；中断即清零。当前几何阈值最大车间距 12 m、离散度 30 m²，不能使用旧小车 2 m 阈值解释本 run。

保留奖励的折扣势差、真正终端后继势为零、终端一次结算及原始／折扣回报记录。当前地形失败代价为 100，奖励配置与障碍分量见实际 training YAML；旧说明中的 55 不是当前有效值。

令底层折扣为 `g`、底层 GAE 系数为 `l`，一次决策实际推进 `D` 步：

$$
R=\sum_{k=0}^{D-1}g^k r_k,\qquad
\delta=R+g^D V_{next}-V_{start}.
$$

GAE 递推系数为 `(g*l)^D`，真实终端的 `Vnext=0`。当前 1 s 参考系数 gamma=0.9975992791345717、lambda=0.99，统一派生 0.2 s 系数；中途终止使用实际 D。奖励／观测采用配置中的 log1p，价值目标使用 running normalization。

`action_effective` 与 `learning_valid` 分开：在动作尚不能生效前发生的终端，保留奖励、Critic 样本及 GAE，只从 Actor 损失、熵、优势标准化和策略统计排除。当前 `terminal_actor_credit=effective`，Actor／Critic 分开裁剪；不因此删除整段轨迹或额外 reset。

2026-10-24修复批次入口：正duration的 `learning_valid=false` 记录仅在 `terminal_reason=5`（environment_invalid）、`terminated=true`、`bootstrap_allowed=false` 时允许进入批次处理，随后由既有mask排除Actor／Critic／归一化目标并切断GAE，不拒绝其余有效记录。输入张量及动作log probability的非有限数、矛盾的终端标记仍停止训练。证据见[恢复记录](experiments/t11_crater_reward_recovery_2026-10-24.md)。

## 百万步续训评估与恢复边界

2026-10-24新增配置 `upstream_1km_far_crater_reward_1m_v1.yaml` 使用 `periodic_fixed_window_v1`：每10万累计底层步在批次边界评估，初始／中途／最终均保留60秒`far_sample/mean`面板；最终另有1500秒`full_far_sample/mean`。独立评估最多88,800步，其中最终预留62,400步。训练任务和学习参数与短训完全相同；旧`bounded_fixed_window_v1`仍只有初末评估。

完整继承49,871步，新增上限1,000,000，故本次`--budget=1049871`。新run显式迁移仅评估元数据与预算／调度身份，环境、优化器及随机状态逐值不变，原严格恢复校验保持。见[本轮记录](experiments/t11_crater_reward_long_training_2026-10-24.md)。下文短训和历史预算保留对应启动时语义。

故障恢复分支从62,628步完整保存点继续；原已消耗25,551步包含未更新的12,794步。剩余974,449步，故恢复分支 `--budget=1037077`；新增授权总消耗＝分支 `actual_base_steps` −49,871＋12,794。初始评估继承原49,871步策略结果及2,400步账目，不再运行；独立评估剩余86,400步。原checkpoint与events保持原义。

## 保存、预算与日志

`run_upstream_t11.py --training-config ... --budget N` 使用当前分支累计实际底层步上限。当前SAT从零训练run的N=1,000,000，从0开始，无旧计数。历史短训N=50,000，另一个历史恢复分支N=1,497,631、曾继承261,149步；恢复时仍不能把剩余数作为累计上限。更早 run 消耗与独立评估分别计费，见[优化续训记录](experiments/t11_optimized_training_resume_2026-10-23.md)。内部 `curriculum` 状态键继续复用容器，不表示当前运行课程。

本轮每批保存完整恢复点及四owner分片；旧正式配置仍为每5批。恢复需保持算法、配置、源码及地图身份；不得只恢复 learner 却冒充完整环境续跑。普通批切分保留热启动和 GRU，异步 reset 只清理对应实例。快照不导出求解器对象。

评估使用冻结策略和独立环境／RNG／GRU，结果绑定 `training_base_steps`，不进入 PPO。所有学习与恢复保存实时 TensorBoard；训练横轴使用 `actual_base_steps`，评估消耗单独记录。具体运行和恢复规则见[训练手册](runbooks/train_proxy.md)及[输出管理](references/output_management.md)。
