# 当前局部目标接口规范

## 价值目标归一化候选（2026-10-19）

`TrainingOptions.value_normalization`支持默认`none`与显式`running`。启用后，`AblationPolicy`注册`RunningValueNormalization`的count／mean／M2 buffer，并在policy与训练元数据写入语义身份；默认关闭时保持旧权重键与元数据。`policy.value()`始终返回原奖励单位，包含采样、bootstrap和评估尾值；PPO在原单位GAE之后、每批一次更新有效团队return统计，Critic拟合归一化目标。统计不按PPO轮次／车辆重复，随checkpoint恢复，不允许跨语义直接载入。

现有`--mode repair-validation --budget 20000`增加`--value-normalization running --comparison-run <已完成combo>`，只运行一个全新初始化候选，复用原组合对照；其余模式拒绝这两个非默认参数。训练前验证配置、优化器其余选项、初始参数及固定面板一致。候选独立评估／检查上限4千底层步，原两组默认入口仍为8千步。详情见[本轮记录](experiments/t11_value_normalization_trial_2026-10-19.md)。

## 三项组合闭环验证接口（2026-10-19）

`run_upstream_t11.py --mode repair-validation --budget 20000`顺序运行baseline与三修复组合，复用原采样／skrl更新／checkpoint路径；固定A1、seed23、4×4环境、5 s决策与1 s折扣参考周期，每批4决策／环境、5轮PPO。`near_far_balanced_v1`固定8近／8远，所有实际物理推进按每组2万步计费，独立评估suite上限8千步。该准入不继承旧signal100k或20万步语义。

`UpstreamEnvironment(initial_control=...)`可显式设置每车初始线／角速度，形状为`(E,4,2)`；写入环境身份，异步reset及load后继续恢复该初速。默认None保留旧初速与元数据。近距候选每次reset为0.1 m/s；Process采样元数据额外记录cohort，禁止跨初态协议直接恢复。命令、预算与验收见[记录](experiments/t11_combined_learning_validation_2026-10-19.md)。

## 2026-10-19：独立梯度裁剪与完整更新归档

`TrainingOptions.gradient_clipping`支持`global`和`actor_critic`。前者保留旧全局裁剪及原options元数据；后者按原`gradient_limit`分别裁剪Actor与独立Critic，并显式写入options元数据，禁止跨模式直接恢复Adam/学习状态。`run_upstream_t11.py --mode decision-credit --gradient-clipping actor_critic`启用候选；旧入口不接受非默认裁剪。共享PPO更新仍由skrl和原trainer共同调用。

每轮日志保留裁剪前总范数，新增`actor_gradient_norm`、`critic_gradient_norm`、`actor_gradient_scale`和`critic_gradient_scale`。有界实验的`interactions_*`归档补齐更新前`obs/state/hidden`以及CPU更新随机数`update_rng`；CPU learner可结合对应checkpoint完整复演概率及优化。历史归档缺字段仍属于不完整证据，不自动补造或覆盖。实现与对照见[顺序修复记录](experiments/t11_learning_repair_execution_2026-10-19.md)。

同入口新增`--initial-continuous-std-scale {1,0.25}`，默认1保留原初始化。0.25只在全新learner建立时将可学习`log_std`减去`log(4)`，Actor均值、HOLD门控和Critic权重不变。采样与log-prob使用同一个新分布，不在执行后缩放动作；PPO可继续更新标准差。非默认初始化写入run身份和有效配置，恢复时加载已学习权重，不重复缩放，也不允许更换初始化身份后直接恢复旧学习状态。

`--terminal-actor-credit {all,effective}`默认`all`维持旧协议；`effective`写入`TrainingOptions`和checkpoint身份，拒绝跨语义恢复。两个collector共同增加布尔`action_effective`：真终端的实际duration不大于`ceil(simulated_latency/dt)`时为假，因为请求尚不能控制终止之前的底层步。它与`learning_valid`分开，**不删除奖励、Critic样本或GAE链条，也不额外重置GRU**；仅Actor标准化、PPO目标、熵与策略变化统计使用两者交集。若整批无有效Actor样本，Actor参数和Adam槽不推进；策略比率/KL日志为null，skrl不统计该空项。诊断日志新增`actor_transitions`及`actor_excluded_terminals`，更新归档保存该掩码。

## 2026-10-18：T11持续决策与时间折扣接口

`make_collector`、`UpstreamAblationRollout`和`ProcessUpstreamRollout`新增可选`learning_time: LearningTimeConfig`。省略时保留旧元数据和1 s行为；显式配置支持A0/A1，研究比较入口固定A1。配置字段为`policy_decision_period_s`（1或5）、`discount_reference_period_s`（0.2或1）、`gamma`和`lam`（默认0.99/0.95）。gamma/lambda相对于参考周期，统一派生底层系数；`TrainingOptions.gamma/lam`继续表示底层系数，必须与collector一致。

5 s区间只采样一次并推进一次GRU，固定odometry目标；每1 s更新合法约束、续发新票据，0.2 s执行／安全检查不变。一次区间只存一条PPO转移，`duration`仍为每环境实际底层步数，完整区间为25。环境中途终止保留终端观察，到下一决策入口才reset，其他环境继续；缓冲切分不冒充任务终端。

区间奖励`R=sum(g**k*r_k)`；变时长TD为`R+g**D*V(next)-V(start)`，GAE递推系数`(g*l)**D`。真实终端后继势和bootstrap为0，非终端保留尾值。奖励、生产／诊断更新及skrl使用显式同源系数。

新元数据增加`learning_time`，checkpoint增加`base_steps`（所有实际推进，包括无效学习步）和`decisions`（实际执行的团队策略决策），旧`transitions`仍指有效团队样本。新协议的JSON侧车记录这些计数与时间参数。普通续发不生成概率样本；不允许跨时间语义恢复或导入旧learner。新`--mode decision-credit`每组固定最多200,000底层环境步，按完整向量区间预留，独立于原`signal100k`。

完整命令、三组条件和评估口径见[实现记录](experiments/t11_decision_learning_implementation_2026-10-18.md)。以下按历史阶段保留原接口说明。


2026-10-13追加：独立新车辆尺度已接通S00参数、S01公共执行基础、S02的40步规划器核心及S03感知／通信／停车组件。版本接口见文末；完整环境、Gate B和T11尚未通过，见[S02](experiments/upstream_scale_planners_2026-10-13.md)与[S03记录](experiments/upstream_scale_s03_components_2026-10-13.md)。下述T03–T10接口保留旧尺度与15步语义。

日期：2026-10-11。当前实现与版本边界见[规划架构](architecture/planning_pipeline.md)。旧86维等多代接口全文见[历史快照](archive/interface_spec_snapshot_2026-10-09.md)，不作为当前 exp167 或 P0 输入布局。

## 版本与观测

| 配置 | 任务契约 | 观测契约 | 合成地图 Actor / Critic | 三通道真实图接口 Actor / Critic |
| --- | --- | --- | --- | --- |
| exp167 | legacy_v1 | local_goal_v1 | 410 / 990 | 522 / 1438 |
| p0_contract_v2 / t03_planner_protocol | finite_task_v2 | local_goal_clock_v2 | 411 / 991 | 523 / 1439 |
| t04–t10 独立诊断 | finite_task_v2 | local_intent_clock_v3 | 411 / 991 | 523 / 1439 |

Actor 形状为 `[env, 4, obs_dim]`，Critic 为 `[env, state_dim]`。两者的底层 schema 保持现有多尺度结构，新增时钟位于完整输入末尾。实际模型维度依据配置和运行张量构建，不能用历史常数猜测。

`remaining_time = clamp(1 - step_count / max_episode_steps, 0, 1)`，是已知任务截止下的本地时钟；每车获得一份，Critic 每环境一份。终止前状态保留0时钟，reset后的状态恢复1；两者不得混用。

新版附加30维邻车消息（3×10）按同一次通信 snapshot 的 `sender_indices` 填槽，与基础51维邻车信息对应。缺失 sender 不填真值，过期、稀疏或不支持的消息版本不暴露目标／轨迹；元数据保留年龄、有效期和版本。sender 只用于关联，不新增数值ID网络特征。

## 请求、规划与执行

独立修复配置新增 `local_goal.terrain_memory_version`（旧默认 `quantized_latest_v1`，新诊断 `direct_batch_v3`）与 `diagnostic_nmpc_adapter`（旧默认 `nmpc_polar_v1`）。记忆版本记录在环境 `terrain_contract.local_memory_version`、解析配置与快照；旧默认不新增该键，保持原恢复语义。新路径恢复前校验记忆版本、数量、容量、分辨率与支持半径。`local_terrain_samples_v1` 的快照结构不变；只读查询不访问全局地图，未知边界不放宽。

`leader_local_site_v8` 是独立无学习脚本，输入仍为 `ScriptObservation`（本车状态、只读地形、合法消息、sender 绑定预测）。候选点须完整本地支持，实际到达重新复核后冻结当前位置；followers 根据合法 leader 停稳消息接近槽位。脚本状态版本 8 拒绝旧脚本，HOLD、实际停稳与环境终端成功仍分别判定。

`leader_local_site_v9` 的状态版本为 9，拒绝 v8 状态；独立低风险停车配置采用它。仅本车完整支持的 0.9 m／37 点锚点区域满足高度差 ≤0.08 m、坡度 ≤0.10 及原通行门槛时，恢复 0.2／0.3 m 捕获／退出；未知或坡地保持严格行为。合法消息要求、实际停车、安全层与环境成功判据不变。v8 坡地回合与 v9 平坦回合的验证范围分别记录。

`leader_local_site_v10` 状态版本 10 拒绝 v9 状态。接近合法槽位且本车实际已停稳时，可依据当前位置 0.1 m／9 点的合法低风险支持请求 HOLD，记录 `common_site_certified=false`。它不要求把未知锚点区域认作可行场地；完整轨迹复核、限幅、实际团队地形及驻留仍按原契约，验证范围见修复记录。

`nmpc_recovery_v4 / merit_recovery_v4` 使用适配器状态版本 4。公开输入／输出和 15×0.2 s 网格不变；短参考只改变代价目标，原请求约束与输出复核保留。一次规划最多四次尝试，共享时限，诊断保留每次参考、求解状态及选择来源；快照不含求解器对象且拒绝旧状态。完整边界和证据见[修复记录](experiments/gathering_failure_repairs_2026-10-11.md)。

动作版本 `local_goal_request_v1`：每车4维有界连续 `[dx,dy,yaw,speed]`；保存采样前 latent，log-prob 使用 tanh Jacobian。零动作的速度上限为0.575 m/s，不是 HOLD。

`LocalGoalRequest` 保存 odometry frame 下冻结目标、速度上限、时间戳、有效期和协议版本。`for_hold(pose, now)` 显式冻结当前位置／航向、速度上限0；环境 `step(..., hold_mask=...)` 仅供诊断，runner 不调用该覆盖入口。运动中的请求按现有变化率限制减速，可能因停止约束选择 brake；反馈不隐瞒该 fallback。

`TimedTrajectory`、`PlannerFeedback`、`NeighborPrediction` 继续复用协议版本1，NMPC revision为 `prefix_consistency_v2`。HOLD、实际停车、连续成功驻留分别判定。T04已在独立版本实现条件混合分布；T03/T04协议见后文。

## 新版任务与转移

新判据为全部车辆满足线速≤0.05 m/s、角速≤0.1 rad/s，并满足最大距离≤2 m、离散度≤0.75 m²、已有最小间距与地形门，连续保持5 s（0.2 s控制周期下25步）。任一条件破坏，驻留计数清零。

`TerminalReason` 张量编码：0 none、1 success、2 collision、3 out_of_bounds、4 task_deadline、5 invalid_terrain。同时发生时优先级：invalid_terrain > collision > out_of_bounds > success > task_deadline。合法状态在截止同一步完成驻留仍为成功。

| 情况 | terminated | truncated | bootstrap_allowed | learning_valid |
| --- | --- | --- | --- | --- |
| 成功／碰撞／越界／任务截止 | true | false | false | true |
| 无效地形 | false | true | false | false |
| 有效宏步／普通缓冲切分 | false | false | true | true |
| 未执行的环境行 | false | false | false（v2） | false |

`MacroTransition` 增加上述原因及两个mask；保留实际 duration、reward、终止前 Actor/Critic、reset后 Actor/Critic 和 episode/trace。外部有效截断可用 truncated=true、bootstrap_allowed=true 表示并切断 GAE trace；普通缓冲切分本身不重置环境或 GRU。

无效地形所在宏步整体不用于学习，即使其中较早的底层步有效。无效后继不进入 Critic，GAE 不跨无效样本传播；PPO 的归一化、Actor、Critic和熵损失均只使用有效行。网络前先屏蔽无效输入，避免 `0 * NaN` 污染梯度。终止／reset切断循环状态，序列边界从采样时保存的 hidden 开始。

## 奖励与统计

保留底层 gamma=0.99、lambda=0.95 及原代价权重。设 D/S 是距离／离散度相对新阈值的非负超限量，E 为平坦度 cost 超过1的部分，A为原平坦度激活函数，h/H为归一化驻留进度：

- gather势为 `-dmax_progress*D - dispersion_progress*S`；原level代价也改为仅对超限量收费。
- flatness势为 `-centroid_flatness_progress*A*E`，保留原excess代价尺度。
- hold势为 `-success_hold_step*(1-clamp(h/H,0,1))`，不再逐步累计当前进度。

每个底层步使用 $F_t=\gamma\Phi_{t+1}-\Phi_t$ 对应的势差（实现中为浮点配置值）；真正任务终端的后继势为0，普通有效采样切分保留势尾项。各分量继续按原权重合成；能耗、安全、地形代价保持。无效地形转移返回0学习奖励并屏蔽损失，不施加任务超时惩罚。

成功奖励130、碰撞／越界失败惩罚55、任务截止惩罚80沿用配置，仅按对应终端原因结算一次。碰撞物理代价仍独立保留。没有新增时间成本；折扣偏好不解释为严格成功概率最优化。

宏步内仍逐项折扣；GAE 使用实际低层时长对应的 gamma 和 gamma×lambda 次幂。每环境跨宏步累计 raw_return、discounted_return、duration_low_steps，以及加权 reward_terms（分量和等于原始总回报，允许浮点误差）。结束后只重置对应环境；累加器进入环境快照以保持恢复一致。

## 兼容与迁移

省略新版字段时使用旧任务／观测契约；旧 exp167 YAML 不修改。新增配置通过显式版本组合校验，错误组合提前拒绝。

运行和 checkpoint 元数据包含 task_contract_version、observation_version、action_version、planner_revision；还检查完整配置及实现指纹。指纹覆盖项目包源码、runner、配置加载和场景生成依赖。新版拒绝缺失合同或版本不匹配的 checkpoint；旧 checkpoint 的历史复现应使用匹配代码版本，不能静默续训。

正式训练暂停由新配置和 runner 的 finite_task_v2 门控共同执行；共享 core.step 拒绝将此配置误用于旧原语入口；无学习审计使用 `scripts/audit_p0_contracts.py`。接口通过不代表整个 Gate A、任务收敛或物理闭环通过。

## T03：统一规划协议 v1

`configs/experiment/t03_planner_protocol.yaml` 继承 P0，使用相同任务／观测／四维动作与维度，独立启用 `planner_protocol_v1`、`nmpc_bridge_v1` 和 planner_state_version=1。exp167及P0配置继续走 legacy_direct，元数据不增加这些字段。实现集中在 local_goal/protocol.py；阶段验收见[T03记录](experiments/t03_planner_protocol_validation_2026-10-10.md)。

### 输入与信息边界

| 字段 | 形状／含义 |
| --- | --- |
| env_id / agent_id | 非负整数；批次中身份唯一，每项对应独立规划器 |
| ego_pose | [3]，接收车固定 odometry frame 下x/y/yaw，m/rad |
| ego_velocity | [2]，实际有符号前向速度／角速度，m/s、rad/s |
| intent | LocalIntent：goal_xy[2]、hold、timestamp、expires_at、可选terminal_yaw和speed_cap |
| terrain | TerrainView，局部传感记忆的独立快照，只暴露lookup/safe/local_ball/free_disks |
| neighbors | IdentifiedNeighbor(sender_id, prediction)有序序列，复用NeighborPrediction；NMPC最多3个邻车 |
| previous_control | [2]，上次下发差速执行的轮速等效v/omega，不是受坡度影响后的实际速度 |
| now / times | 回合内秒数；times=now+arange(16)*0.2 |

局部目标和预测均已经转换到接收车固定坐标系。TerrainView只复制白名单内的局部传感字段，不复制core、全局地形或Oracle引用；不允许以未知区域填充可通行地形。邻车身份来自构造该预测的sender，不按距离重新关联。过期消息可形成已有保守预测，但时间、身份及预测版本必须合法。

LocalIntent.from_request无损保留四维请求。LocalIntent.for_hold冻结请求时位置和航向；HOLD的速度上限固定为0。运动意图的航向／速度允许缺省，但T03桥接尚不生成其默认值，返回unsupported_intent。HOLD必须携带冻结航向，调用for_hold可满足该条件。`PolicyAction(rho,beta,hold)`仅为T04目标语义，尚无混合分布。

时间有效边界包含expires_at；now更大时返回expired_intent并刹车，禁止复用旧计划。请求来自未来、反向有效期、非法形状、非有限值及不支持版本在求解前报错。规划网格固定15步、3秒；请求有效期与轨迹预测时域是不同概念。

### 输出、回退与执行

PlannerOutput持有既有TimedTrajectory和PlannerFeedback；predicted_states/nominal_controls是轨迹属性视图，不复制平行数组。状态[16,3]、控制[15,2]、实际预测速度[15,2]、时间[16]，首状态与输入位姿一致。

| 输出情况 | valid | fallback_reason |
| --- | --- | --- |
| feasible / feasible_iteration_limit / feasible_prefix | true | 空字符串；前缀原始拒绝细节保留在feedback |
| previous（通过当前状态复核） | true | 原求解／候选失败原因 |
| brake | false | 求解失败、超时、expired_intent、unsupported_intent等具体原因 |

无效规划的限幅刹车仍可进入现有应急执行链，不构成地形／碰撞安全证书。total_cost为null，直到规划器提供明确定义的可靠代价；不能将null当0或用不同内部目标直接排名。

execution_passthrough先跟踪轨迹，再经过直通安全接口，返回nominal_control、applied_control和projection_disabled。环境记录这些值、身份、有效性、回退原因及代价。applied_control指发送给代理差速执行器的命令，实际速度仍由地形、轮速限制和动力学决定。

### 生命周期与恢复

Planner协议包含plan、reset、state_dict和load_state_dict；plan_batch按调用方顺序返回，支持空批和稀疏批，不允许一个实例在同一批承担多辆车。当前CPU实现支持现有线程池；未来GPU打包属于适配器内部，不修改策略接口。

NMPC状态封套记录protocol、adapter、state_version和planner_revision，负载只含previous轨迹及blocked时长。环境不访问适配器的NMPC内部字段，不序列化求解器。恢复前验证合同与状态数量；适配器拒绝错误版本／形状，失败时恢复已有规划器状态，环境core不应用失败快照。环境reset只重置指定环境的实例，rollout切分不触发reset。

新诊断路径拒绝缺少协议元数据的P0/旧快照，runner沿用完整合同、配置及源码指纹校验。源码指纹包含新协议模块。正式训练门控继续有效；历史复现应使用匹配源码版本。

## T04：条件极坐标动作与观测 v3

独立 `t04_polar_hold.yaml` 继承T03，新增动作polar_hold_v2、观测local_intent_clock_v3及rollout载荷polar_hold_rollout_v1；任务finite_task_v2、规划协议和轨迹版本保留。旧exp167/P0/T03配置仍为原动作与观测。新版Actor/Critic维度仍为合成图411/991、三通道图523/1439，但语义版本不同，禁止凭维度相同加载旧checkpoint。

### 策略动作与概率

PolicyAction包含physical[env,agent,2]和布尔hold[env,agent]，physical是rho/beta；默认范围分别为[0,1.6 m]、[-pi,pi rad]。HybridSample保存动作、二维原始latent、log_prob、following hidden和rollout版本；motion_active等于非HOLD。

先采样Bernoulli分支，只为MOVE行采样Normal latent。MOVE经tanh与仿射变换映射至物理极坐标；HOLD的连续动作和latent均填零，占位不参与动作似然。确定性分支在logit>0时选择HOLD，等于0选择MOVE；MOVE使用tanh(mean)。HOLD头采用小增益权重与零bias，初始概率在0.5附近，不对所有状态强制为0.5。

HOLD的log_prob为log(p)；MOVE为log(1-p)+log(q)。q从保存的latent计算，包含稳定tanh Jacobian及log(rho_max/2)+log(beta_max)的仿射修正。密度定义在物理极坐标空间，不从规划控制或目标xy反推。熵为Bernoulli解析熵加(1-p)乘当前MOVE分布的重参数化连续熵估计。

关闭熵正则时，HOLD动作似然不向连续头传播梯度；完整分布熵仍可能更新连续头，与当次采样为HOLD不矛盾。HOLD正常参与价值学习和离散策略损失。无效转移在进入网络／分布前屏蔽，不能以HOLD替代无效转移分类。

### 意图、观测与通信

MOVE直接生成LocalIntent，目标由请求时车体系极坐标转换到固定odometry frame，航向和速度约束为None。HOLD调用for_hold冻结请求时位置／航向。所有请求有效期1 s，决策周期5步、控制步长0.2 s、规划15步不变。rho=0仍是MOVE。

新版以intents为请求源，goals中不构造伪四维请求；拒绝hold_mask外部覆盖。动作整批校验后才修改环境状态。缺省约束的MOVE在现有NMPC桥接中返回unsupported_intent并刹车；T05之前不允许将它报告为运动闭环。测试专用适配器只验证输入消费和记录。

基础ego的请求速度／请求航向两个槽位为保留零值；新版附加观测前10项如下：

| 索引 | 含义 |
| --- | --- |
| 0–1 | 冻结目标相对当前车体的xy |
| 2 | HOLD请求标志 |
| 3 | 意图存在且未过期 |
| 4 | 请求剩余有效期比例，截断至[0,1] |
| 5–6 | blocked_seconds/10、原规划反馈状态编码/5 |
| 7–9 | 实际前向速度、实际角速度、局部地形known比例 |

reset后无意图，目标／HOLD／有效期均置零。原75项局部地形与30项邻车消息布局保持；Critic依旧追加每车前10项，末尾任务时钟保持。过期意图保留历史目标与请求标志，但有效性和剩余有效期为0。

通信目标取发送方发布的LocalIntent。stop_intent只在请求有效、明确HOLD、实际线速≤0.05 m/s且角速≤0.1 rad/s时由发送方发布；阈值比较使用与任务状态相同的张量精度。接收方继续遵守FULL／SPARSE、sender、时效和版本权限，不能补入未通信真值。保留零值的旧请求速度槽不能解释为停车。

### Rollout、版本及运行入口

新版rollout记录hold、motion_active、physical action、二维latent及采样时log_prob；载荷版本必须匹配。PPO校验物理动作与latent变换的一致性，拒绝被控制量替换或被篡改的动作；序列长度16、实际duration折扣、异步done与无效样本处理沿用P0规则。

配置、checkpoint、环境快照、manifest及评测元数据统一由contract_metadata生成，包括新动作／观测、规划协议、rollout版本及动作范围。旧checkpoint和范围不匹配的版本明确拒绝，不迁移优化器状态。原评测报告写死四维动作版本的路径已改为完整合同来源。

正式训练仍暂停。旧完整任务评测对T04提前拒绝并指出T05依赖；诊断入口为scripts/audit_polar_hold.py，测试适配器记录test_only，合成PPO测试单独标记optimizer更新次数，不生成训练checkpoint。验收入口见[T04记录](experiments/t04_polar_hold_validation_2026-10-10.md)。

## T05：基线适配与版本隔离

`t05_planner_baselines.yaml`继承T04，显式启用nmpc_polar_v1/state2。注册表统一工厂与元数据；非NMPC不构造acados。旧nmpc_bridge_v1保持原默认约束及快照规则，仍拒绝缺省MOVE约束。新NMPC适配只在适配层生成缺省航向／1.15 m/s速度上限，按意图冻结、保存恢复，不改策略原始动作或概率。

确定性line/quintic复用原参考几何和跟踪器，经幅值／轮速／变化率限制后差速预测；TimedTrajectory是实际代理预测，不是几何点。公共execution函数承载原NMPC预测、复核和停车前缀，旧入口委托后通过配对验收才能标为等价。

47库版本exp156_primitive47_v1；PrimitiveReplayInput(context, primitive_id, library_version)是明确的原生载荷，不是PolicyAction。冻结0 HOLD、1–39前进、40–42倒车、43–44旋转、45–46让行。统一输出／失败记录，新适配限幅独立于旧categorical执行语义；旧categorical配置和checkpoint不迁移。原生载荷只通过统一诊断入口，不传入polar环境step。回放的请求时间沿用context；过期强制刹车。

新增诊断入口scripts/diagnose_planners.py，无优化器、无训练checkpoint。旧训练和六单元评测门控继续保留。阶段实施与验收入口见[计划](implementation_plan.md)。

## T06/T07：批量 MPPI 与本地查询

新配置t06_mppi_base.yaml和t07_mppi_terrain_neighbor.yaml使用planner_protocol_v1、polar_hold_v2、finite_task_v2、local_intent_clock_v3；分别选择mppi_base_v1与mppi_terrain_neighbor_v1，状态版本1。协议、适配器、完整MPPI配置、种子、动力学和代价版本进入运行元数据／快照校验。旧配置、categorical和旧checkpoint不自动切换。

### 批量、运动与生命周期

单条plan和有序plan_batch共用Torch内核；按设备、精度及完整MPPIConfig分组。同配置的不同工厂内核也可合批；每车的身份、请求、控制均值、上一轨迹、回合计数和RNG仍属于独立句柄。NMPC仍逐实例调度，MPPI不使用逐车线程池采样。

| 数据 | 形状 |
| --- | --- |
| 原始高斯控制／受限控制／实际预测速度 | [B,K,15,2] |
| 预测状态／名义均值 | [B,K,16,3] / [B,15,2] |
| 代价／数值可用标记／硬可行标记 | [B,K] |

默认CPU、float32、K=512、2轮、噪声标准差[0.35,0.7]、温度1、seed23，网格固定0.2 s×15。CPU集成使用K=128；小契约用64；512/1024/2048均真实调用。float64用于解析参考。温度不与学习gamma/lambda混用。

幅值边界为v±1.15、omega±2.4，轮速等效v±0.188omega不超过18×0.098。先限制幅值及轮速，再沿上一实际下发控制到目标的线段按比例满足dv≤0.3、domega≤0.6；所有后继控制继续复核。中点积分保存实际速度，T07可选解析地形缩放，最小0.25。普通MOVE不补造终端航向；可选航向仅显式提供时加入圆周误差代价，速度字段仍是上限。

HOLD不采样，冻结请求时位姿并限幅减速；移动中停车可返回valid=false的刹车反馈。请求过期直接刹车，禁止复用旧目标。普通切分保留热启动；按实际经过的控制步数移位并追加停车尾部，从当前实际位姿重新预测。reset只清空对应环境，其他句柄RNG不受影响。

state_dict导出CPU控制均值、上一轨迹、请求、last_now、身份／回合，以及保存到CPU的RNG字节状态和原始rng_device，不含acados对象。CPU导出不改变随机数生成器所属设备。load先校验形状、有限性、配置、版本及身份，再修改状态；缺失请求字段也拒绝恢复。同设备类型、同版本恢复要求复现，CPU/CUDA跨类型热恢复显式拒绝。

### 代价、权重与输出

阶段目标距离平方权重2，终端12；控制0.05v²+0.03omega²，变化0.5dv²+0.2domega²，倒车0.2min(v,0)²。T07加地形、未知和邻车分量，独立开关。学习奖励保持T02语义，不使用规划代价替代奖励。

原始控制为Z=U+epsilon，epsilon服从独立高斯，执行控制为确定性映射g(Z,previous_control)。重要性分数使用原始epsilon，保持：

$$
S_k=J_k+\tau\sum_t U_t^\top\Sigma^{-1}\epsilon_{k,t},\qquad
w_k=\frac{\exp(-(S_k-S_{\min})/\tau)}{\sum_j\exp(-(S_j-S_{\min})/\tau)}.
$$

$$
U_t\leftarrow U_t+\sum_k w_k\epsilon_{k,t}.
$$

数值不可用样本排除，全部不可用时权重为0，不计算无效softmax；进入旧计划／刹车回退。硬不可行但数值有限的样本可参与风险优化，不能直接执行。重要性修正不混入物理代价分量。

输出依次检查更新均值、末轮最低物理代价的可行采样、带完整停车尾段的可行前缀、复核过的移位旧计划、限幅刹车。均值必须重新预测，不能继承样本的可行标记。最终轨迹仍通过CPU复核；valid仅表示此离散模型／信息边界下通过复核，刹车valid=false，不是连续安全认证。

total_cost按最终实际选中状态／控制批量计算，排除重要性修正；HOLD或没有可靠有效解时为None。日志记录各物理分量、有效样本比例、权重ESS、原始噪声指纹、候选拒绝及回退来源；末轮采样NPZ记录原始／受限控制、状态、实际速度、各分量、权重和重要性分数。CPU耗时区分打包、预测、代价、更新、样本复核、最终复核及端到端；不作为GPU结论。

### 只读地形与邻车边界

TerrainView.snapshot返回local_terrain_samples_v1：只含本地xy、五维features、observed_at和support_radius，数组只读且与实时记忆脱离。TerrainBatch按有效掩码打包，并沿查询点分块（默认256）；不分配完整[B,K,N,M]。空记忆、支持范围外与未知洞返回未知，不插值补空。MPPI通过TerrainView.conservative()显式启用等距冲突的最大坡度／最小可通行度规则，CPU最终复核使用同一视图；旧T03/T04默认TerrainView及live memory查询语义保持。

五维仍为高度、x/y坡度、既有粗糙度占位、可通行度代理。当前传感粗糙度占位为0，不构造粗糙度代价；可通行度来自1-risk，不把它解释成测量粗糙度。地形风险为2×((slope/0.8)²+((1-traversability)/0.75)²)，每未知节点诊断代价20；未知样本不能有效执行。硬边界为有支持、traversability≥0.25、坡度模长≤0.8，每段9个插值点。

NeighborBatch按IdentifiedNeighbor.sender_id绑定位置、半径和有效掩码。直接消费环境已有的延迟／过期保守预测，不重复膨胀半径，不补取未通信真值。软代价在半径外0.2 m内采用归一化平方风险、权重8；无邻车为0。硬边界复用段内插值和相对位移/16的补偿。关闭软地形／未知／邻车代价不会关闭硬复核；地形动力学另有开关。

验收和失败记录见[T07](experiments/t07_mppi_terrain_neighbor_validation_2026-10-10.md)，入口见[运行手册](runbooks/planner_diagnostics.md)。CPU局部闭环通过不解除旧全任务评测或正式训练门控；T05–T07配置的安全接口保持projection_disabled，T08–T10独立配置和证据见下文；独立CUDA诊断的数值门槛见下节。

### 独立CUDA配置与证据边界

`t07_mppi_terrain_neighbor_cuda.yaml` 和 `t10_gathering_no_learning_cuda.yaml` 分别继承既有CPU诊断，显式设置 `local_goal.planner_device: cuda`、`mppi_query_chunk: 1024`、`training_paused: true`。适配器、协议、动作与观测版本保持继承配置，设备和query chunk由完整MPPIConfig进入运行／恢复校验。`simulation.device` 仍为CPU；GPU仅执行MPPI张量内核，感知、通信、差速代理、OSQP与最终协议轨迹复核仍在CPU。

每个设备分别冷启动独立run。CPU/CUDA热状态因完整配置及 `rng_device` 不匹配拒绝恢复，拒绝时不修改目标句柄；同设备的身份绑定、异步reset和公开状态恢复继续保持原生命周期契约。GPU数值对照使用同一显式注入噪声，不能仅凭同seed断言不同设备RNG逐值一致。

`scripts/benchmark_mppi_devices.py` 在整调用前后执行CUDA同步，计时包含CPU打包、传输、规划与最终host复核；预热计算内核，每次调用reset热状态。K=512/1024/2048、GPU B=1/4/16/64及默认CPU B=1分别记录，同噪声SHA绑定配对结果。既有内部分阶段耗时缺少同步，只表示异步dispatch，不构成GPU实际耗时证据。面板仅使用冻结平坦合法局部输入，不代替地形／邻车性能或集合验收。

2026-10-11首次独立GPU诊断契约17项为16通过、1失败、0跳过；旧v1的`cross_offset` float32控制量最大绝对误差 `1.7166e-5`，未通过既定 `atol=1e-5, rtol=1e-5` 对照。该原始失败与旧配置保持原状，独立证据见[首次CUDA诊断记录](experiments/mppi_cuda_diagnostic_2026-10-11.md)。既有诊断报告的 `cuda_validation: deferred` 不自动变成完整验收通过。

### 稳定数值v2与查询内存预算

新`mppi_terrain_neighbor_stable_v2`适配器／规划版本使用状态版本2，数值版本`fp32_fp64_accum_v2`。原始采样、受限控制、运动预测和控制均值仍为float32；邻车位置差、距离、平方风险先转float64再计算，重要性分数、归一化权重及加权扰动累计使用float64，更新后的均值转回运行精度。不能仅将float32累计结果末尾转double。采样公式、代价权重、动力学及硬执行边界不变；旧v1仍按原规则执行，v1状态不得静默恢复为v2。

v2元数据增加`accumulation_dtype=float64`、`terrain_query_memory_budget_bytes=268435456`、`terrain_query_budget_version=pairwise_temporaries_v1`和预算公式`max(1,min(query_chunk,budget//(B*M*(8*element_size+4))))`。B为本次打包实例数，M为最大本地样本数，`element_size`为地形张量每元素字节数。256 MiB预算用于估计查询的成对临时张量；有效query chunk可小于配置1024，但不改变最近邻／保守等距语义。它不是整个规划器、原始采样或CUDA缓存分配器的总显存上限，实际allocated／reserved高水位另行记录。

独立新增`t06_mppi_base_cuda.yaml`保留基础`mppi_base_v1`／状态1；`t07_mppi_terrain_neighbor_cuda_v2.yaml`、`t08_safety_projection_cuda_v2.yaml`、`t09_planner_benchmark_cuda_v2.yaml`、`t10_gathering_no_learning_cuda_v2.yaml`采用稳定适配器／状态2。五配置保持CPU host、CUDA规划、query chunk1024与训练暂停；T08起沿用CPU安全层。设备、运行／累计精度、预算及完整MPPI元数据进入运行与恢复校验。

`scripts/validate_mppi_cuda.py`要求实际CUDA及当前完整GPU契约JUnit，产出独立`mppi_cuda_acceptance_v1`。证据绑定源码、测试源码、JUnit、manifest与指标SHA，检查必要测试0失败／0跳过、K512／1024／2048×B1／4／16／64负载、复杂合法输入、float64参考、生命周期和容量边界。

2026-10-11稳定v2的CUDA数值／负载验收已完成：66项CUDA契约、695项联合回归及其中18项真实acados全部通过、0失败／0跳过；runtime严格验收的36GPU面板、9个B1 CPU配对、27个复杂输入对照及容量边界通过。生产证据分别为`outputs/runs/mppi_cuda_validation/contracts_v2_seed23_rtx2060_20261011_02/`与`acceptance_v2_seed23_rtx2060_20261011_02/`。[新GPU Gate B汇总](../outputs/runs/t09_planner_benchmark/acceptance_seed23_cuda_v2_20261011_01/metrics/summary_0000.json)确认完整7000／630／252／14及所有门槛通过，`gate_b_cuda=true`、`gate_b_cpu=false`。[T10正式汇总](../outputs/runs/t10_gathering_no_learning/acceptance_seed23_cuda_v2_20261011_01/metrics/summary_0007.json)确认144回合完整、18例简单门槛全部通过，`t10_cuda_functional_pass=true`；110成功、24终端地形不满足、4规划拒绝及6任务截止，无碰撞／执行限制违反。[完整逐步审计](../outputs/runs/t10_gathering_no_learning/acceptance_seed23_cuda_v2_20261011_01/metrics/trace_audit_0000.json)通过117744次执行记录、4108个宏步／公开快照及710／710个要求采样事件，错误为0。状态见[本轮CUDA验收记录](experiments/mppi_cuda_validation_2026-10-11.md)，冻结范围见[计划](cuda_acceptance_plan.md)。

47回放沿用共享诊断配置的任务／观测合同，但实际动作载荷是PrimitiveReplayInput和显式primitive_id；该路径不消费polar_hold_v2策略概率，不用配置中的动作字段推断原生47与极坐标等价。


## T08 安全执行契约

仅独立诊断配置启用 `safety_projection=discrete_cbf_qp_v1`；版本与状态版本1写入运行／恢复元数据，依赖 `osqp==1.0.4` 通过可选 `safety` extra提供。旧配置默认 `disabled`，继续 `projection_disabled`，不构造OSQP。

`SafetyInput` 只接受冻结的 `PlannerInput` 及接收车odometry中的工作区半平面 `workspace_A @ p <= workspace_b`，不接受core或Oracle。工作区按0.14 m代理圆留边；邻车半径使用已有预测，不重复膨胀。地形是中心路径支持，仍非完整足迹。

`ExecutionDecision` 在原名义／下发控制和状态之外新增 `correction`、`reason`、`model_feasible`、`constraint_ids`、`qp_status`、`residual`、`seconds`。安全通过状态为 `unchanged/projected/verified_brake`；未复核刹车为 `unverified_brake`。规划 `valid` 另存，执行复核不能将无效规划提升为成功。

共享投影按中点差速解析Jacobian迭代最多3次。QP无松弛，容差1e-6、4000次迭代、固定rho；仅接受 `solved` 并通过非线性9点、轮速、幅值和变化率复核。邻车端点满足离散 `h_next >= 0.8 h_current`；段内使用既有相对位移裕量。局部线性化、有限点和代理模型不构成连续安全或真实月面认证。

HOLD/过期保持已有限幅停车；HOLD速度上限0表示停车意图，不能解释为即时跳至零控制。求解失败、超时、拒绝明确刹车；非有限输入报环境失效。公开 `reset/state_dict/load_state_dict` 隔离每个环境／车辆，快照只含版本、模型配置和上一控制，无求解器对象。

实际下发写入下一规划 `previous_control`。启用安全的路径保留名义规划，另生成实际首控制与停车尾段的执行预测；完整复核才广播，失败撤销轨迹载荷。消息的 `stop_intent` 继续由实际速度与有效HOLD判断。FULL消息中的朝向从合法通信基础特征恢复，仅新安全／脚本诊断使用。

## T09/T10 无学习证据协议

T09 `planner_cases_v1` 冻结1000个合法局部输入，七设置7000条原生输出，各输出使用同一次求解分别计算安全开关；公共评测代价与规划器内部代价分开。原生47枚举完整编号，平局取小编号。30案例×三个种子×七设置的630恢复记录、252局部回合及B4/16的14面板另存。分片25案例，不完整分片拒绝续跑。

`EvidenceRun` 绑定全部实际任务源码、诊断脚本、解析配置、依赖版本；结果原子写入且拒绝覆盖。恢复需要显式 `--resume` 并匹配指纹，状态pickle限本地可信诊断使用，有SHA校验、无acados对象。中断遗留与重试轨迹分别保留，不算规划失败。

T10 `leader_slot_v1` 只消费 `ScriptObservation`：本车位姿、实际速度、只读地形、合法 `ScriptMessage` 和sender绑定预测。脚本状态包括首次合法leader朝向及槽位HOLD滞回。FULL载荷不完整／过期时HOLD，停稳邻车不触发编号永久等待。动作每5步按既有极坐标解码，普通MOVE位移不超过1.6 m。

`LocalGoalEnvironment.step(..., diagnostic_callback=...)` 为无学习证据提供每底层步回调，在结束环境重置前读取当次采样。T10每5步公开恢复环境、脚本、安全与规划状态；不加载网络、创建优化器或训练checkpoint。CPU正式144回合要求同源码的CPU Gate B证据；有显式宏步预算的接线pilot不能充当T10验收。

CUDA证据协议为`planner_cuda_evidence_v1`，新run记录`planner_device=cuda`、`simulation_device=cpu`及当前MPPI元数据，状态从`pending`开始。T09七设置的suite标记CUDA，但line／quintic／原生47／NMPC实际设备仍为CPU，三种MPPI设置才使用CUDA；逐条设备和数值版本须匹配。完整覆盖7000／630／252／14、恢复与简单闭环、当前源码绑定联合回归和18真实acados、完整GPU契约及runtime严格证据均通过，才输出`gate_b_cuda=true`；此时`gate_b_cpu=false`，不重写历史CPU结论。

正式CUDA T10使用`t10_evidence_driver_cuda_v1`，必须提供`--gate-b`、`--cuda-junit`与`--cuda-acceptance`；逐项核对版本、设备、运行／累计精度、预算、完整配置、SHA和Gate B覆盖。仍仅允许T09不导入的T10消费者文件差异，其他实际依赖必须与生产快照一致。144矩阵和两个规划器各九例简单主验收全部成功才标CUDA集合通过；`--max-macros`保留为独立pilot，不能绕过正式门控。

独立CUDA接线pilot限定 `flat_compact/mppi512/safety on/duration 180/max-macros 30`，消费真实传感与合法通信，OSQP在CPU。宏步限制是另设的实际执行预算，不构成144回合正式GPU验收。正式新GPU全矩阵仍需与当前实际依赖源码匹配的新Gate B生产证据，保留旧源码guard；不能以GPU可用、局部数值对照或pilot替代正式门控。

## S00／S01 新尺度公共执行接口

2026-10-13追加v2：[修复记录](experiments/upstream_scale_review_v2_repair_2026-10-13.md)。`upstream_scale_s01_review_v2.yaml`显式启用`footprint_review_v2/batch_ties_v2`，元数据增加两项版本并将geometry改为`footprint_upstream_v2`；cache key与公开恢复绑定新身份。旧S00／S01配置省略两项时保留原元数据和标量／v1语义，只作历史复现。

`UpstreamRuntimeConfig`及嵌套规范为冻结dataclass；设计文件原始SHA与完整设计内容、规范、配置、设备、种子及各版本绑定恢复身份。`metadata`返回独立副本，`validate_state_identity`在加载前拒绝缺失／不同身份。参数审核和基础代理不构造acados或优化器；旧配置解析分支保留，新配置在旧core／环境／训练评测入口被拒绝。

`limit_control_numpy([v,w], previous, spec)`与`limit_control_torch(...[...,2]...)`使用同一三模式可行集合：低速、正向行驶和倒车行驶，联合方向相关变化率、幅值、轮速及轮加速度。改向经过零，低速边界先减角速度。该映射只是执行器限幅，不能认证地形或碰撞。`midpoint_*`返回预测位姿与实际速度；实际速度与轮速等效下发控制单列。

`SafetyStopEnvelope`复用`TimedTrajectory`，记录`delay_steps/commitment_steps/braking_steps/stationary_steps/fits_horizon/information_time/activation_time/conservative_center_distance`。储备轨迹长度可以小于40步，与S02固定40步的规划输出分别记录。信息时间储备包含已排队保持控制前缀，实际执行必须从生效时间开始，不能重复等待。无完整N40储备或足迹复核失败时，`AdmissionDecision`明确为`unverified_brake`并记录动态限速及过渡超速；HOLD只沿限幅停车，不重采样目标。

`FootprintTerrain`只复制`TerrainSnapshot`位置、五维特征、时间和支持半径，不能保存地图owner、环境core或Oracle。等距点采用危险高度／坡度及保守通行代理。`FootprintReviewer`检查四轮接触平面、去趋势高度RMS、台阶、车底净空和间隔不超过0.1 m的矩形足迹／扫掠；每段至少9时刻。邻车`FootprintNeighbor`带sender与绝对查询时间，其半径已含几何／通信不确定度，不再次膨胀消息年龄；查询超出覆盖时报错。工作区和邻车使用圆包络，地形保留矩形／接触检查，不将包络半径从0.155 m支持圆扣除。

`UpstreamActuatorProxy.step`消费`[E,4,2]`目标与局部合法地形增益，返回名义／实际下发、实际速度、生效时间、几何违反及状态；每实例下一步使用实际下发作为`previous_control`。公开`reset(env_ids, poses)`只清空指定环境，`state_dict/load_state_dict`完整校验后才修改状态，快照仅CPU数据。该代理不提供Actor、奖励、任务终止或通信链，不能替代S03环境集成。

v2新增`FootprintNeighbor.for_spec(sender_id,times,positions,radii,spec)`：绑定规范指纹与`identified_neighbor_upstream_v2`，要求半径至少为两车有效包络之和；复核也检查半径、版本和指纹。未绑定／旧尺度载荷返回明确拒绝，不重新叠加消息年龄。公共`FootprintReviewer`默认v2，普通轨迹和信息时间储备都根据同一本地特征重算实际速度和后继，`conservative_reserve`不能绕过核对；float64与float32容差分别为1e-10/1e-8和1e-5/1e-5。

`FootprintTerrain(query_version="batch_ties_v2")`默认每块最多256点，先统计球邻点数，使一次邻点列表合并不超过65536条；保留原标量k=1/k=2、五维原始通道及等距排序语义。`query_version="scalar_ties_v1"`为配对／历史路径。只读视图记录规范指纹，v2拒绝不同规范的地形输入；内存／查询错误不能静默标记为可行。

## S02独立40步规划接口

`configs/experiment/upstream_scale_s02.yaml`显式选择`planner_execution_version=upstream_planning_v1`，复用S01的`footprint_review_v2/batch_ties_v2`。`local_goal/upstream_factory.py`只接收该执行版本；S00／S01配置仍不能通过此工厂构造规划器。当前工厂仅接受`offline_simulated_latency`，MPPI可以选择CPU／CUDA，NMPC及几何适配器在CPU执行。请求CUDA不可用时直接失败。

协议集中在`local_goal/upstream_planning.py`，版本为`planner_protocol_v2`。状态坐标使用接收车固定odometry frame，距离为米、航向为弧度、时间为回合内秒数；`workspace`固定为`(xmin,ymin,xmax,ymax)`，传给足迹复核时转换成`[min_xy,max_xy]`。不接受全局地图、环境core或Oracle作为输入。

| 对象 | 字段与约束 |
| --- | --- |
| `UpstreamIntent` | `goal_xy/hold/timestamp/expires_at`；可选`terminal_yaw/speed_cap`及显式`stop_at_goal`，数组自有只读，HOLD不由零距离或速度上限零推断 |
| `UpstreamPlannerInput` | `env_id/agent_id/ego_pose/ego_velocity/intent/terrain/neighbors/previous_control/now/times/workspace/spec`，另含可选`information_pose/information_time`及协议版本 |
| `UpstreamPlannerOutput` | 复用`TimedTrajectory/PlannerFeedback`，另含`valid/fallback_reason/total_cost/diagnostics`；可靠最终代价缺失时为`None` |
| 固定预测网格 | `dt=0.2 s`、`N=40`，状态`[41,3]`、控制／实际速度`[40,2]`、时间`[41]` |
| `Candidate` | `[40,2]`候选控制、来源、可选代价与诊断；进入共享执行限制和重新预测，不能直接把几何点当成实际状态 |

`now`是**控制实际生效时刻**，输出轨迹从该时刻及当前实际位姿开始。正模拟延迟必须提供原始`information_pose/information_time`，缺失时明确返回`valid=false/missing_information_state`。输入校验要求`now = information_time + ceil(simulated_latency / dt) × dt`，并依据同一局部地形核对期间保持上一实际控制所得的生效位姿；地形观测不能晚于信息时刻，邻车预测覆盖信息时刻至规划末端。模拟0.5 s延迟对应三个控制步，不能再把这三步添加到生效后的实际执行中。相邻调用按完整控制步推进；实际速度必须与上一轮速等效控制及已有地形速度缩放语义一致。

`PlanningAdapter.plan/reset/state_dict/load_state_dict`为公开生命周期；批内核可使用`validate_input/prepare/finish`先校验整批，再共享采样计算，最后逐输入执行独立完整复核。`prepare`直接处理HOLD、过期及缺失信息状态。HOLD执行既有限幅制动；过期不使用旧目标回退。所有有效输出包含完整限幅停车尾段，且同时通过当前轨迹和信息时刻停车储备复核。请求速度上限降低时，只允许已在制动过程中的受限过渡，不瞬时裁剪实际状态。

输出选择为更新均值、至多16个预筛候选、带停车尾段的前缀、按实际经过步数移位并重新预测的旧计划、明确刹车。更新均值复核与16个样本预算分开计数；前缀面板固定为20／10／5／1个运动步，实际可用前缀还须给完整停车留足时域。`sample_review_budget_exhausted/unreviewed_sample_count`保留未检查样本的信息，有限预算拒绝不能解释为证明不存在可行轨迹。紧急刹车始终`valid=false`，另记录`unverified_brake`；安全复核失败不变为规划成功。控制经共同限制或补停车尾段后，原候选代价不直接复用，MPPI重算最终物理分量；NMPC代价仍可为`None`。

| 适配器 | 独立实现 |
| --- | --- |
| `line_upstream_v1/quintic_upstream_v1` | 旧几何生成器只提供参考，采用弧长时间参数化、`planning_time`跟踪，再通过新尺度联合限幅和地形中点预测 |
| `primitive47_upstream_v1` | `UpstreamPrimitiveInput(context,primitive_id,library_version,library_fingerprint)`携带原生编号；新库`upstream_primitive47_v1`按新尺度冻结并记录内容指纹，旧47库不改写 |
| `nmpc_upstream_v1` | 新40步acados模型`midpoint_directional_corridor_v1`，显式缓存模型／规范／源码身份；航向与默认速度约束按意图冻结，走完整参考和停车走廊的本地支持选择 |
| `mppi_upstream_v1` | 共享`upstream_mppi_kernel_v1`批量计算，独立`upstream_mppi_normalized_cost_v1`，每车独立随机状态与热启动；默认512样本、两轮更新 |

新47库仍按0 HOLD、1–39前进、40–42倒车、43–44旋转、45–46让行排列，位移及参考速度按规范重新生成；倒车受0.5 m/s上限约束。编号与库指纹必须同时匹配，不能静默加载旧categorical动作／checkpoint。NMPC只在实际选择该适配器时导入并构造acados；求解器对象不进入快照。共同生命周期状态绑定配置、规范、实现、设备与状态版本，MPPI拒绝跨CPU／CUDA热恢复。

`UpstreamPlannerPool`按`(env_id,agent_id)`保存独立实例；批调度按内核身份分组MPPI并恢复输入顺序，NMPC及几何路径逐实例调用。`reset(env_id)`只清空对应环境；整池恢复先校验全部条目，再替换已有状态，避免后部坏条目造成部分恢复。

`LocalTerrainBatch`从本地快照建立空间哈希，查询点与桶占用双重分块，默认查询块256、临时预算64 MiB，不分配完整查询×记忆距离张量。最近邻等距以最终最近距离判定，格宽包含等距和支持边界裕量。CUDA使用同设备几何计算；真实等距或支持边界歧义通过所属CPU本地视图校正五通道，记录`ambiguous_queries/boundary_queries/cpu_tie_query_s`。这条局部查询子路径不把整套规划器切回CPU。未知返回零特征及`known=false`。`LocalNeighborBatch`保持sender、位置、有效掩码与半径一致，半径保留float64，不重复加通信年龄。

上述张量查询仅用于中心节点预筛；最终规划与停车储备继续在CPU执行矩形／接触／段内扫掠完整复核。新尺度传感、通信、任务终止和安全投影尚未由S02接入。诊断入口只推进实际首控制及同设备恢复面板，不等价于完整局部或集合闭环；[当前记录](experiments/upstream_scale_planners_2026-10-13.md)及[命令](runbooks/upstream_scale_diagnostics.md)分别保存实现状态与验收证据。

## T11动作概率准备接口

`local_goal/action_ablation.py`的`ActionContract`绑定A0–A3、尺度指纹、动作范围、请求有效期与概率测度。独立版本为`t11_polar_move_contract_v1/t11_polar_hold_contract_v1/t11_cartesian_request_contract_v1/t11_primitive47_contract_v1`，重放版本为`t11_action_replay_v1`。这些版本只用于准备，不能代替运行期任务／观测／规划／安全版本。

`ActionPayload`保存原始连续动作、布尔HOLD、int64原语编号及动作身份。A0/A1使用物理极坐标密度；A1复用既有条件分布，HOLD连续占位为零。A2保留归一化四维tanh动作与密度；A3连续维度为0，编号0–46附冻结库版本及指纹，独立于显式极坐标HOLD。A0零距离、A2速度上限零均保持MOVE。`decode_payload`返回不可变`PreparedRequest`，没有实际执行、停车或规划成功含义；A2方形支持范围的角点超过极坐标半径，执行桥接前须独立冻结比较约束。

`AblationPolicy`复用T04编码器／GRU／Critic；工厂逐值复制共享参数，只改变输出头。当前合成Actor/Critic维度411/991，元数据标记`synthetic_t04_layout_not_new_scale_environment`。`replay_action_sequence`先排除无效环境，再重算原采样概率；旧采样及序列初始隐状态固定为重放数据，梯度只沿当前策略重建的序列传播。真实结束清零相应GRU，普通序列切分保留尾状态。`clipped_actor_objective`只接受`[T,E,A]`ratio与`[T,E]`固定优势／有效性，用于无优化器梯度诊断。

`export_diagnostic_state/load_diagnostic_state`先完整校验`t11_action_diagnostic_state_v1`、动作／测度／尺度／网络／库身份和张量，再加载；跨方案、旧checkpoint或缺失身份均拒绝。准备模型的`diagnostic_only=true`由旧PPO入口提前拒绝。独立配置与命令见[T11准备记录](experiments/t11_action_preparation_2026-10-13.md)，正式环境及训练仍未接通。

## S03密集感知、通信与实际延迟历史

`SensorFrame`固定12231个直接局部采样，带`env_id/agent_id/timestamp/map_identity`、接收车固定odometry下的位置／本车位姿、五通道、有效与直接掩码；导出数组只读，不包含地形backend。`DenseTerrainMemory.observe(frame)`仅接受同车、同地图、单调时刻及同规范的帧，保留直接位置与危险冲突；`view()`返回脱离owner的`FootprintTerrain`。网络112点与115项附加输入的独立读取仍待接通，不将密集帧直接拼入Actor。

`RoverBroadcast/ReceivedMessage`绑定sender、episode、发布／接收时刻、位姿、实际速度及实际下发控制；FULL可携带目标、场地中心／布局朝向、执行轨迹和限速证书。稀疏载荷不暴露这些字段。场地中心与朝向必须成对；接收变换对位姿、目标、场地和轨迹一致应用，控制与实际体坐标速度不旋转。

`DistanceCommunicationCache.transmit`只用路由真值决定60／120 m消息权限，`snapshot`只读缓存及接收车本车状态。各槽位按缓存距离排序，`sender_indices`与目标／轨迹／预测始终同源；不存在消息时不补真值，过期时保留带sender的保守可达界。状态1 s、轨迹0.6 s按各自原始时间戳判断，重复投递不能刷新年龄。`stop_intent`由合法FULL HOLD及实际三项停稳阈值共同判断，不构成未来静止承诺。

`MotionBoundLedger`是执行owner的持续承诺登记。只有同规范、同episode、已登记且覆盖全部未来查询的证书可降低25/9 m/s上界；实际每条控制须经过`limit`，更高新证书不撤销旧未过期低上限。规划输入只接收生成后的`FootprintNeighbor`，不持有ledger、缓存或路由真值。快照恢复及异步reset分别校验并维护消息／证书epoch。

S03配置显式选择`actual_control_history_v1`。`UpstreamPlannerInput.actual_history`携带信息到生效时刻的`ActualControlHistory`，包含本车身份、规范、时间网格、实际位姿、控制／速度和信息时刻上一控制。须逐项满足联合限制、实际gain范围、midpoint及信息／生效端点，并匹配向上取整的延迟。过去使用实际反馈，未来仍使用冻结本地地形；真实历史与保守停车储备合计不能超过40步。旧`held_previous_control_v1`拒绝该新载荷，新路径也拒绝缺失历史；热状态不可跨时间版本恢复。

`pose_scope_cache_v1`只复用同一次轨迹内逐值相同位姿及停车标志的静态足迹报告，动态约束不缓存，退出即销毁。配置身份记录该选择，旧配置缺省维持无缓存语义。

`UpstreamPlannerPool.export_diagnostics()`独立导出`upstream_pool_diagnostics_v1`及每sender的末轮CPU采样／诊断副本，修改导出不影响后续执行或RNG。它不是热恢复接口，不包含求解器对象，不能把一轮导出标为两轮完整采样。

`LeaderSquareScript.decide`仅消费本车实际状态、本地视图和合法snapshot，返回带显式HOLD的意图、状态机及停车见证。leader冻结初始角点和布局，其他车解析独立场地中心；MOVE仍按原极坐标解码生成≤4 m滚动目标。完整停车终点合法且位于0.2 m捕获区才进入BRAKING，直到实际停稳；HOLD仅在停稳且误差超过0.4 m时重新接近。任务成功仍由`UpstreamTask.advance`的实际四车状态、逐车停车足迹、几何、速度、漂移与25步驻留决定。

这些公共组件不是可训练环境，完整高层排队续发、安全、奖励／观测和真实宏步接线尚未完成。证据范围见[S03记录](experiments/upstream_scale_s03_components_2026-10-13.md)。

## S03精度v2与密集记忆观测

`upstream_scale_s03_precision_v2.yaml`显式选择`mppi_upstream_v2`、状态版本2和`sampling_configured_actuator_fp64_v2`。高斯原始采样／均值仍按配置默认float32，原始扰动重要性公式保持不变；实际上一控制、联合投影、状态与实际速度使用float64。旧v1内核保持原样，工厂不共享不同内核，跨版本热恢复拒绝。局部cap用批量制动曲线保留合法过渡，并以float64容差复核；低速多边形epsilon不允许绕过cap。

`upstream_scale_s03_observation_v1.yaml`启用`dense_actor_clock_upstream_v1`。`ObservationInput`仅含本车实际状态、有限任务时钟、当前意图／公开反馈、私有地形和合法receiver绑定snapshot；重验sender、版本、episode、发布／接收时刻、年龄、有效掩码。没有core、全局地形、Oracle或真实团队驻留入口。

`UpstreamObservationEncoder.actor`返回自有只读`ActorObservation`：411维值、112点known掩码及身份／时刻／规范；保持N1的15＋51＋224＋5＋115＋1布局。三层网格分别7×9近车体、3×7中场、4×7前后远场，查询来自密集记忆而非Actor反解。高度为相对本地点除车高；风险为`max(1-traversability,(slope/driving_slope)^2,(RMS/roughness_rms)^2)`截至[0,1]，未知为高度0／风险1。

附加输入保留意图／反馈10、目标周边25×3、邻车3×10；消息目标及未来点与基础17维槽共享sender。归一化采用25/9线速、0.35角速、4 m局部目标、60 m通信及已知任务时限。实际驻留、dmax及集中状态只由`critic`读取；Critic保持991槽，9个Oracle槽明确置零，不将二维代理的z槽冒充绝对DEM高程。

消息v2为`sender_message_upstream_v2`，FULL可携带合法sender-local五通道，坡度按固定frame旋转；稀疏撤销新增载荷。旧v1拒绝该扩展。snapshot带`env_id/receiver_id/timestamp`；缓存v2使用`distance_1e-8_sender_v2`稳定等距排序，该分桶不参与硬通信、地形或碰撞阈值。

诊断驱动首次与恢复均通过同一协议构造。`--contracts-from`只用于驱动自身修复，要求其他生产源码／归档、测试集合／SHA、配置、依赖及settings全相同；原JUnit保留在原run，新增驱动契约在新run实际执行。它不能复用任何算法变化后的测试，不能产生完整S03、Gate B或训练门控。

最新实现与证据见[精度／观测记录](experiments/upstream_scale_s03_precision_observation_2026-10-13.md)。网络诊断只做零初态GRU及Critic前向，实际循环rollout、奖励、安全和T11执行桥接仍未连接。

## S03完整环境与T11执行桥接

2026-10-14独立`actual_loop_upstream_v1/v2`通过`UpstreamEnvironment.step_intents/reset/state_dict/load_state_dict`执行真实传感、通信、实际控制历史、任务和有限奖励。宏步输出包含实际duration、terminal_reason、bootstrap_allowed、learning_valid、终端观测、逐步记录和回报分量；普通切分不reset。新接线证据见[环境记录](experiments/upstream_scale_environment_2026-10-14.md)，历史前向诊断范围保留。

v1保留issue+1 s绝对有效期；v2将送达票据和活动执行租约分开。原始动作/目标/约束、issue和expires_at保存不改写；仅按时送达才建立activation+1 s租约，迟到票据进入原过期回退，旧活动租约不延期。版本与环境快照绑定。

`footprint_discrete_cbf_upstream_v1`是可选执行层，读取新profile、只读局部点集和sender预测；联合方向/低速/轮速/变化率可行域为三个二维QP。地形支撑圆盘线性化后独立进行完整足迹、扫掠、动力学及停车复核。`SafetyDecision`记录名义/下发控制、状态、原因、残差、约束来源和耗时。通过公开`execution_prediction()`重建实际首控制和停车尾段，完整复核失败则撤销广播载荷；规划有效性保持原输出。这不是实车/PhysX保证。

`UpstreamActionBridge`消费原始`ActionPayload`：A0/A1原解码，A2归一化方形连续映射到4 m圆盘、概率仍定义在原始动作上；A3传递原生编号并分别记录来源与新尺度库指纹，信息时刻冻结目标。零速度上限仍为MOVE。`UpstreamRollout`目前接A1真实环境与循环PPO/变时长GAE，实际结束清零GRU，普通缓冲边界保留。四组完整学习运行与统一验收仍待完成。

S04探索的`mppi_upstream_v3/v4`分别绑定状态3/4和独立核/代价身份，继承v2实际FP64运动/控制复核。v3平滑项按物理速度尺度归一化，初始化按目标距离/时域引导；v4的XY误差按剩余距离归一化，下限为固定捕获尺度，显式航向项不改变。`mppi_parameters`仅供v3/v4，支持四项代价倍率、两项噪声和温度的正有限覆盖，进入配置与核元数据；硬限制/任务阈值不作为覆盖项。周期角历史验证使用圆周差，仍拒绝真实姿态或位置不一致。公开`reset_agent(env_id,agent_id)`仅清空指定规划器热状态，普通切分不调用它。

## 四组实际循环准备与独立校准入口

`UpstreamAblationRollout.collect(1..16)`消费真实411/991维输入与原始A0–A3样本，通过同一桥接执行，保存duration、终止原因、学习有效掩码及有效尾值。GRU在实际结束/无效时清零，普通切分保留；公开恢复绑定环境、政策概率、执行库、设备及RNG身份。通信恢复后的按环境reset先复制只读时钟数组再清零。

`renew_legal_bounds`的租约仅依据本车上一实际控制和合法消息净空计算，覆盖预测时域；执行器实际强制该界。它对四组动作使用同一规则，不从动作分支或未通信真值猜测停车。

2026-10-16学习路径修复：在`braking_clearance_lease_v3`配置下，collector以制动距离上界和`0.8*净空/(预测时域+请求有效期+模拟延迟)`的较小值提出承诺，仍不低于上一实际控制。`execution_bound_policy=learned_horizon_braking_lease_v1`进入collector身份；原学习checkpoint需显式迁移这项身份，脚本承诺及环境配置身份不改。该界由执行器强制，8秒轨迹复核及安全阈值不变，见[修复与续训](experiments/t11_learned_lease_repair_2026-10-16.md)。

`PreparedUpstreamTrainer`拼接连续实际采样块，保留原始动作/log-prob和跨缓冲隐状态；采用现有有限任务/变时长GAE与循环PPO目标。公开状态包括collector、优化器、计数和版本，不含求解器。正式更新同时要求明确训练授权与独立准备审核；当前只有诊断更新和定义生成。

`LocalCase/FormalCase`指纹绑定实际感知点、特征/时间、sender预测、真实延迟历史、意图、初态和地图身份。校准种子101/103/107/109与正式23/31/47分开；正式训练准备的独立评估种子为151/157/163。gzip诊断状态保留原对象并校验SHA，只改变存储编码，跨版本/配置热恢复仍拒绝。

完整评测入口与训练准入是独立出口；当前结果范围见[本轮记录](experiments/upstream_scale_training_preparation_2026-10-14.md)，生成四组配置不解除训练暂停。

## T11实际运行与诊断状态v2（2026-10-15）

`PreparedUpstreamTrainer` v2支持连续128宏步采样、五轮PPO及优化器/GRU公开恢复。`diagnostic_update`至多两批、每批最多128宏步；使用与正式更新相同的数值实现，输出明确标记diagnostic_only，不能作为正式训练初态。`update`继续校验暂停授权、车辆profile与独立准入。新增诊断计数进入快照身份，旧v1热状态拒绝自动迁入。

`scripts/run_upstream_t11.py`记录有效团队宏步、原始动作/概率、实际执行和独立评估，支持从更新边界恢复；诊断只保存带SHA的诊断状态。正式模式要求完整Gate B、144集合、必要契约、CUDA及入口验收证据，当前未启动。独立评估在副本环境和RNG作用域中执行，不重置采样环境/GRU。

`leader_square_visibility_upstream_v3`将合法停稳HOLD邻车的优先放行半径与现有退出半径对齐；v4进一步提供带导航净空的局部路点。两者不修改任务阈值或实际控制/复核约束。旧v1/v2保留原语义，状态版本身份拒绝静默恢复。证据及未完成范围见[推进记录](experiments/t11_progress_repair_2026-10-15.md)。

运行入口版本`upstream_t11_session_v2`区分诊断与学习角色。旧运行配置保留`training_paused=true`且不含`training_session_version`；仅显式`upstream_t11_learning_v1`允许两个暂停字段同时为false。该角色供工厂/环境识别，并绑定配置和恢复身份；它本身不证明准入。CLI在转换前校验完整证据与明确解除暂停；正式更新拒绝暂停配置，诊断更新拒绝学习角色。当前只执行角色构造与有界诊断，正式训练仍暂停。

`flat_frame_pose_cache_v3`对混合朝向点云使用完整观测帧的规则网格子集证明支持，全部保留点必须具有相同平坦安全属性；只在足迹完整位于证明区域时复用报告，其余回到原复核。危险点、未知、邻车、控制和停车限制保持原检查，配置身份与v2分开。

2026-10-16等价计算优化保留同一接口：平坦段可批量返回原顺序首失败；完全静止轨迹在静态证明内一次构造全部原扫掠时刻，仍逐时刻检查邻车膨胀及采样余量。数值边界走原路径，不缓存动态验收。`LocalTerrainBatch`按一次lookup合并等距/边界CPU校正，保留五通道及内存预算。实测及CPU/GPU取舍见[训练交互优化](experiments/t11_interaction_acceleration_2026-10-16.md)。

2026-10-16原生计算优化保持上述接口及检查预算：FP64控制投影/空间哈希使用原生核，属性一致点集支持整条40步GPU预测；非均匀等距查询保留CPU权威校正。严格平坦见证内批量完整移动扫掠保留首失败及采样计数，数值边界回到原复核。CPU编译器、CUDA Toolkit及原生源文件指纹纳入运行/迁移依赖，见[原生优化记录](experiments/t11_native_acceleration_2026-10-16.md)。

2026-10-17[运动优化](experiments/t11_motion_optimization_2026-10-17.md)将私有记忆内部改为连续数组与原生稳定合并，保留直接点顺序、危险合并、时间戳、淘汰和readonly快照；阈值/梯度舍入边界由原NumPy判定。安全层每轮共享几何约束，三个控制模式仍独立剪枝与求解。环境默认6线程逐车计算，使用独立复核器，在各阶段完成后按原环境/车辆顺序提交，保留通信、GRU与RNG顺序；线程池不进入恢复状态。公开版本、checkpoint结构及完整检查预算不变。

`motion_lease_version`默认`clearance_horizon_lease_v2`，历史配置与元数据保持原样。独立`braking_clearance_lease_v3`以合法收到的邻车净空、本车实际上一控制和配置制动能力提出速度承诺；该版本进入配置/状态身份，不接受旧身份热恢复。连续制动距离公式只提出上界，不能替代完整离散足迹、邻车、延迟和停车储备复核。所有尚未到期的旧承诺继续强制；移动中的HOLD不立即承诺零速度。学习collector不按采样动作更新承诺，原动作概率与损失语义不变。接口实现与证据见[本轮记录](experiments/t11_speed_lease_repair_2026-10-15.md)。

原生47类局部评测的候选枚举使用独立零延迟信息时刻预览；选中原生编号仍进入原配置请求队列，实际规划按延迟后的生效时间及真实控制历史生成。预览配置具有独立身份，不复用为实际执行器，不删除旋转/让行类别。`PlanningAdapter`现已显式分派平坦帧v3，新旧真实NMPC轨迹配对通过。

T11可观测性追加：`t11_learning_diagnostics_v1`记录更新前有效优势/目标值尺度、解释方差及每轮ratio/clip/近似KL；梯度范数保持裁剪前定义，损失与优化器顺序不变。`t11_session_timing_v1`记录host采样/PPO/交互保存/评估/状态保存时间，不称GPU内核时间。`interactions`追加GAE输入字段；collector仅在返回的激活事件上附加实际规划valid/回退原因。旧产物允许缺少这些字段，离线工具标为未知，不补零或改写历史文件。恢复仍由原源码/配置身份校验。

独立评估定义固定每family/seed一个环境、四辆车，与`evaluate()`一致；训练环境数量不传播到独立场景文件。操作和证据见[并行准备记录](experiments/t11_parallel_preparation_2026-10-15.md)。

## 2026-10-16：首次100k学习信号准入

训练启动身份新增`signal`，与`diagnostic`和完整`train`分开。`upstream_t11_signal_readiness_v1`绑定当前源码、基础配置、profile及验证产物SHA，字段为`signal_training_ready=true`、`formal_training_ready=false`、`system_acceptance_complete=false`，只允许种子23、A0–A3、100k预算。完整矩阵不属于此角色的启动前置条件；完整`train`继续验证原系统证据。

启动器和`PreparedUpstreamTrainer.update`同时接入这一范围；没有修改任务、奖励、动作分布、GAE或PPO损失。signal仍执行真实学习并保存模型/优化器/GRU/RNG/环境checkpoint，源角色和预算不能静默升级。每16宏步的只读采样回调不改变随机流；每次完整更新后保存恢复点。自动18场景长评估在signal角色关闭，后续独立比较保留。见[当前T11计划](t11_handoff_plan.md)。


## skrl训练与常驻owner接口（2026-10-17）

2026-10-17 OOM修复新增`upstream_checkpoint.save_checkpoint/read_checkpoint`：真实训练的owner恢复点使用`t11_owner_shards_v1`，主文件记录中央学习状态与逐owner文件/SHA清单，最后提交；owner分片逐个本地保存/恢复，不经父进程汇集。旧完整单文件仍可读取。`*.learner.pt`为`t11_learner_only_v1`，只允许显式学习状态导入并重置环境/GRU，不能完整resume，队列选择恢复点时排除它。`owners24_v1`固定6owner/24环境，`owners16_v1`固定4owner/16环境，均每owner2线程、128宏步；不重定义`owners48_v1`。见[修复记录](experiments/t11_checkpoint_oom_repair_2026-10-17.md)。

- `SkrlUpstreamTrainer`身份`upstream_skrl_recurrent_mappo_v1`，元数据记录skrl版本、MAPPO与`duration_gae_shared_team_recurrent_v1`目标；普通load拒绝legacy身份，`migrate_legacy_state`显式验证原配置/设备/计数并导入。模型、Adam、环境、GRU与RNG都进入完整会话恢复。
- 入口默认`--trainer-backend skrl`；`--trainer-backend legacy`保留历史兼容。`--import-checkpoint`只用于新run，不能与`--resume-state`并用；源路径/SHA进入身份。跨设备热恢复继续拒绝，`--policy-device cuda`是显式冷启动设备选择。
- `ProcessUpstreamRollout`身份`t11_persistent_owner_central_policy_v1`。`--env-owners`必须整除`--num-envs`，`--owner-cpu-workers`控制各进程线程；owner seed=`seed+owner_id*envs_per_owner`。中央actor一次采样整批；同一批收齐后更新，保留各环境GRU/episode reset。真实训练的owner完整状态通过本地分片保存/恢复；原state_dict接口保留供兼容和小规模契约测试。错误直接返回traceback，无静默串行回退。
- owner记录的嵌套数组保留组内环境轴，附加`owner_id`与`env_offset`；episode使用全局env_id。owner数量/环境分组属于采样协议，不能拿旧checkpoint热恢复或混入冻结的signal四方案比较。
- `--gpu-service`要求owner模式，集中合并相同K/dtype/kernel的就绪请求；缓存每车地形，GPU结果回传原owner继续CPU验收。作为显式实验选项，默认关闭。
- `FootprintTerrain.readonly_snapshot`只供内部查询；公开`snapshot`仍独立复制。缓存不进入checkpoint，不替代动态复核。
- 回放阶段`prediction_s/cost_s/update_s`在无deadline CUDA路径为CUDA events测时；有deadline和CPU路径保留host同步计时。`end_to_end_s`始终是host整批延迟，不能把各字段混称整轮GPU耗时。

验证与运行记录：[深度优化及skrl迁移](experiments/t11_skrl_deep_optimization_2026-10-17.md)。


## 2026-10-20：1 km课程v2候选扩展（未启动）

`--curriculum-config configs/training/upstream_1km_curriculum_v2.yaml` 显式启用 `rendezvous_curriculum_1km_v2`，旧v1配置不继承新默认值。新增 `initial_level`、`maximum_stage_base_steps`、`initial_continuous_std_scale`、`initial_hold_probability`；中央collector与owner初始级别一致，恢复时由完整状态还原，不重新初始化分布。

`maximum_stage_base_steps` 按从 `stage_start_base_steps` 起的实际环境步计费，包含复习；每次采样同时对总预算和阶段余额预留完整向量决策。不足一个区间时保存结束，阶段停止原因 `stage_budget_exhausted`，不强制晋级；阶段结束可使用独立最终评估预留。用户SIGTERM保持原暂停语义，不触发阶段结束补评估。

固定评估由 `trend_evaluation_levels/seconds`、`initial_full_evaluation_levels`、`full_evaluation_levels`、`full_evaluation_at_base_steps`、`final_evaluation_levels` 定义。`trend_level_<i>_<mode>` 标记短时面板，未结束保留尾值；`level_<i>_<mode>` 才是完整回合面板，必须记录四个结束事件。各结果含 `full_episode_evaluation` 和 `observation_seconds`；只有完整面板进入原三检查点收敛筛选。启动入口校验最坏评估预留不超预算，v2准入另校验 `static_audit.curriculum_config` 与实际配置完全一致。

本扩展没有改变物理动作、奖励权重或安全约束。完整参数和预算见[六项审定](experiments/t11_six_point_review_2026-10-20.md)，当前训练保持停止，下节启动授权为历史记录。

## 2026-10-19：1 km 课程学习接口

`run_upstream_t11.py --mode train --curriculum-config configs/training/upstream_1km_curriculum.yaml` 进入显式 `curriculum_1km` 准入，要求用户释放暂停及已完成的前置奖励／几何／停车审核；限 A1、seed23、四 owner×四环境及 100万／200万实际底层步。该角色与原完整系统准入、signal100k 和短对照分开，不能把旧 checkpoint 静默迁入。当前实际授权与启动预算为 200 万步。

`make_collector`／`ProcessUpstreamRollout` 新增可选 `curriculum_config`。`SpawnCurriculum` 在 reset 时提供新姿态、出生级别与回合截止时间；晋级只影响未来 reset。Actor/Critic 时钟都读取该环境的 `task.deadlines`。owner 状态保存课程 RNG、级别与截止时间，中央状态保存晋级窗口／历史；终端回合携带实际 `curriculum_level`。不启用课程时保留原初始化与状态身份。

`UpstreamRewardSettings.gather_scaling` 默认为 `linear`，保持旧运算顺序和元数据；显式 `log1p` 同时作用于距离／离散度的势和状态成本，绑定奖励指纹。`UpstreamObservationEncoder(spatial_scaling='log1p')` 仅压缩已有合法相对位置及集中式几何尺度，411／991 维布局与消息掩码不变。课程使用 5 s 决策、1 s 折扣参考、gamma=.997；所有生产累积及 GAE 共用派生底层值。

新物理 profile 为 `upstream_vehicle_proxy_1km_v1.yaml`，世界 1000×1000 m、通信 300／600 m，其余车辆与规划物理限制沿用 v2。独立评估保留完整回合和单独预算，不进入 PPO；运行契约与证据见[长训练记录](experiments/t11_1km_curriculum_long_training_2026-10-19.md)。
