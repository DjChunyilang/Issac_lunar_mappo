# 历史状态快照：校准更新前

本页完整保留2026-10-14本轮更新前的状态内容；不再追加当前进度。最新事实见[当前状态](../current_status.md)。

# 当前状态

工程事实更新至2026-10-14，下一阶段计划形成于2026-10-12；历史完整任务性能仍截至2026-09-26。本页区分已确认事实与待实施计划。原详细快照见[归档](../archive/current_status_snapshot_2026-09-26.md)。

## 当前主线

- 已按[实施计划](../implementation_plan.md)推进 S00–S05 车辆尺度迁移。[设计 profile v2](../../configs/design/upstream_vehicle_proxy_v2.yaml)将前进硬上限设为 **10 km/h（2.777778 m/s）**，倒车暂为 **0.5 m/s**，**1 m/s 仅是首轮测试速度**。共同执行与完整环境已接通，八车给定观测的满速/停车及两个真实出生回合的MPPI集合已通过对应局部范围。独立校准与统一版本完整矩阵仍待完成，旧配置保持原语义，正式训练暂停。基础范围见[执行记录](../experiments/upstream_scale_foundation_2026-10-12.md)，最新闭环见[环境记录](../experiments/upstream_scale_environment_2026-10-14.md)。
- 地图采用真实轨道 DEM、插值表示及 NASA 模型坑石增强。[地图主线](../architecture/lunar_map_mainline.md)解释数据来源与边界；现有 7 张地图来自同一个 200×200 m NPB 窗口，不是 7 个独立地理区域。[验收记录](../experiments/nasa_sfd_v2_2026-09-24.md)包含 36 项地形测试及 GPU 无学习检查。坑石模型尚未针对 NPB 标定，处理网格不等于测量精度。
- [任务适配方案](../architecture/lunar_task_adaptation_v1.md)已接受集合、分层出生、停车足迹与驻留方向；新版 P0 诊断已采用集合与停稳参数。原 24 m 通信候选保留对照，下一阶段通信参数以[新尺度独立假设](../architecture/upstream_vehicle_proxy_assumptions_v2.md)为准；分层出生和逐车足迹仍待实施。
- [学习系统审查](../architecture/learning_system_review_2026-09-26.md)已完成首轮无学习核验。其 P0 时间／终止、邻车实体和奖励累计问题已在独立版本修复并通过契约回归；新网络或算法没有通过性能验收。
- 历史兼容入口是 [exp167 连续局部目标＋NMPC](../experiments/exp_167_local_goal_nmpc.md)。`prefix_consistency_v2` 已通过固定目标配对诊断，详见[规划修正记录](../experiments/exp_167_planner_correction.md)；这不是完整任务成功证明。

- [T00–T02 阶段](../archive/implementation_plan_t00_t02_snapshot_2026-10-10.md)已完成：[仓库审计](../experiments/t00_repository_audit_2026-10-09.md)、最小上下文入口与独立 `finite_task_v2` 契约。当前接口统一见[规划架构](../architecture/planning_pipeline.md)和[接口规范](../interface_spec.md)。

- [T03 统一规划协议](../experiments/t03_planner_protocol_validation_2026-10-10.md)已完成：独立诊断配置通过公开生命周期接入NMPC桥接，旧exp167/P0调用路径保留；新规划器在T05–T07独立配置接入。

- [T04与Gate A契约](../experiments/t04_polar_hold_validation_2026-10-10.md)已通过：条件极坐标／HOLD动作、观测v3、循环PPO概率和版本校验已接入。原T04桥接行为保留；完整低维MOVE的真实NMPC适配由T05新版本提供。

## 验收与 checkpoint

[T02 验收](../experiments/t02_p0_contract_validation_2026-10-09.md)：353 项通过，1 项 CUDA 一致性测试因 CPU 版 PyTorch 跳过；6 项真实 acados 运动／停车回归全部通过。只证明契约与局部执行，不代表整个 Gate A 或完整集合任务通过。

[T03 验收](../experiments/t03_planner_protocol_validation_2026-10-10.md)：409 项通过、1 项非必要 CUDA 一致性跳过；11 项真实 acados 全部通过，5 项新增桥接配对最大数值误差为0。只证明协议可替换及桥接兼容，不证明实时预算、安全投影或任务性能。

[T04验收](../experiments/t04_polar_hold_validation_2026-10-10.md)：453项通过、1项非必要CUDA一致性跳过，12项真实acados全部通过；Gate A各项契约有独立机器可读证据。这不构成Gate B/C或正式训练授权。

[T05–T07联合验收](../experiments/t07_mppi_terrain_neighbor_validation_2026-10-10.md)：本轮576项联合回归通过，另1项整数身份复现补充回归通过，0失败、1项CUDA一致性延期，18项真实acados全部通过；CPU基础与地形／邻车规划分别保留独立证据。

exp167 限额 pilot 训练前后各 96 场景均为 0 成功、0 碰撞、96 超时，未收敛。**当前没有通过完整任务验收的推荐 checkpoint**。旧 pilot 的 `outputs/runs/exp167_local_goal_nmpc/pilot_seed23_100k_cpu/checkpoints/latest.pt` 仅供失败诊断，不能直接续训新版规划器。Isaac/PhysX 高保真闭环验证尚未完成。

Active-DSTC 及 exp166 后续消融暂时搁置；相关历史产物在本机未恢复，不能补写未核验数值。正式网络训练继续暂停。

## 当前实施

2026-10-14[S03完整环境接线](../experiments/upstream_scale_environment_2026-10-14.md)：实际感知通信、原始请求/接收租约、控制反馈、任务、奖励、宏步与GRU已连通；新尺度足迹QP和四组动作桥接已实现，19项局部契约及1216项同源联合回归通过，零跳过、含30项真实acados。两次K512出生探索均运行20 s并产生实际接近；完整360 s探索继续运行，近槽位往返微动影响优先解除，已追加真实状态的温度/冷启动探针。完整集合、安全全出口、36例校准与全矩阵以及正式训练准备尚未全部完成；诊断PPO更新不计学习预算。

追加代价/闭环修复：独立MPPI v3/v4修正平滑和近目标归一化、满速初始化；圆周角历史误拒已修复，67项针对性回归通过并重跑30项真实acados，v4另59项通过（批次重合，不相加）。v3续跑第一环境前两车停车、第三车接近；v4从真实出生重新运行，29 s两环境首车均HOLD，第二车接近中。完整集合仍未验收，当前优先继续此闭环并进入独立36例校准，正式训练保持暂停。

v4最新真实出生探索两环境分别72/85 s完成集合，各25步驻留；3140条实际控制审核0碰撞/净空/执行限制违反，异步结束成立。安全投影关闭的两例是局部完整回合证据；安全开启配对正在运行，独立36例校准、完整新Gate B/144回合及四组学习准备未完成，不解除训练暂停。旧v2完整360 s两例均截止，原始失败保留。

2026-10-13[S03精度／真实观测](../experiments/upstream_scale_s03_precision_observation_2026-10-13.md)：独立MPPI v2以float32原始采样、float64执行器／预测修复低速阈值误拒；411／991维真实私有观测与消息v2已接线。169＋1019项冻结测试通过，零失败／跳过，包含30项真实acados；仅驱动恢复构造修正后另4项通过，原JUnit按相同依赖引用。新版八车K512两轮、32步停车／驻留、23步恢复及9组真实网络输入前向通过。完整环境、奖励／宏步、安全与GRU循环学习仍待实现，T11暂停条件不变。

2026-10-13[S03组件接线](../experiments/upstream_scale_s03_components_2026-10-13.md)：密集SensorFrame、私有记忆、合法通信、强制限速承诺、实际延迟历史、方格停车状态机和实际任务已实现。125＋1017项冻结测试通过，零失败／跳过，含30项真实acados；八车K512两轮真实批量规划后，32步实际停车、25步驻留及23步恢复通过。初态为已成形方格，未验证正式出生接近。新增float32低速阈值拒绝缺口已复现；完整Actor观测、奖励／宏步、安全和集合链仍待修复接线，T11预算与暂停条件不变。

2026-10-13[S02规划器接通](../experiments/upstream_scale_planners_2026-10-13.md)：独立40步协议、工厂、真实NMPC、line／quintic、新47类库及批量MPPI已实现。冻结598项契约与417项相关回归全部通过，零跳过，包含旧23项及新增7项真实acados。56个给定本地平坦观测案例完成全足迹／停车储备及实际首控制验证，10组公开恢复配对通过。追加容量矩阵54面板、九组复杂B1 CPU／CUDA同噪声配对通过；容量点集是合成fixture，实际感知闭环分别按S03验收，完整S02出口不由此自动通过。

2026-10-13[T11无学习准备](../experiments/t11_action_preparation_2026-10-13.md)：A0–A3独立动作头复用同一T04骨干，97项新契约与12个三种子CPU面板通过，保存768个原始采样，零环境交互／优化器更新。当前仅合成观测概率与恢复验证；新尺度执行桥接、完整动作表示筛选和Gate C未通过。A2方形位移范围及A3原生执行差异在接线前处理。

2026-10-13[S01复核v2修复](../experiments/upstream_scale_review_v2_repair_2026-10-13.md)：两项拒绝缺口已修复，独立配置启用完整规范绑定、地形动力学重算及分块保守查询；407项回归全部通过、零skip，八车满速／停车／恢复通过。同一严格复核下查询中位耗时13.47→3.33 s，完整轨迹逐值一致；仍未通过实时预算或完整环境接线，旧配置及证据保留。

2026-10-12追加[S01诊断](../experiments/upstream_scale_diagnosis_2026-10-12.md)：公共复核可接受过小邻车半径和与平地模型不符的低预测速度，须先补齐拒绝。CPU主要开销为417648次逐点等距球查询；批量原型约2 s，仍未满足实时预算，且尚未接入运行路径。原364项及满速停车保留为已测范围的事实，不能据此颁发完整复核／环境验收。

2026-10-12新尺度基础：[S00／S01记录](../experiments/upstream_scale_foundation_2026-10-12.md)包含364项联合回归（零失败／零跳过），八车实际达到10 km/h及满速停车、恢复和异步reset通过；旧配置未改。当前是给定本地观测的公共执行基础，完整环境集成和40步规划器未验收。全足迹CPU复核约6–14 s，实时预算未通过，后续需保守规则不变的查询优化及S02／S03接线。

**T05已验收；T06/T07 CPU契约与局部闭环已验收。** 下列初始记录中的CUDA延期是当时状态，本机后续CUDA验收独立见[新记录](../experiments/mppi_cuda_validation_2026-10-11.md)，不改写历史结果。独立记录：[T05基线](../experiments/t05_planner_baselines_validation_2026-10-10.md)、[T06基础MPPI](../experiments/t06_mppi_base_validation_2026-10-10.md)、[T07地形／邻车MPPI](../experiments/t07_mppi_terrain_neighbor_validation_2026-10-10.md)。统一无学习入口见[运行手册](../runbooks/planner_diagnostics.md)。

CPU已实际运行K=512/1024/2048；K=128给定局部观测的前进／转向／倒车达到0.2 m内。平坦后端通过现有传感与通信链驱动两环境×四车，13个底层步全部到达局部目标并停车，无碰撞、无限幅违反。地形和邻车风险实际改变规划；未知／不可通行输出保持硬边界。继承起伏地形的八车诊断仍有车辆在100步内未到达（0.8 m目标最大剩余误差0.425 m），不计成功，需T09继续分析风险成本与推进的取舍。

上述初始CPU局部证据与后续Gate B、完整集合任务及CUDA验收分别记录；Isaac/PhysX尚未通过，正式训练继续暂停。

[T08–T10冻结计划与追加快照](../archive/implementation_plan_t08_t10_snapshot_2026-10-12.md)保留旧尺度验收要求。T08代理安全投影契约已验收：625项通过、1项CUDA延期，18项既有真实acados全部通过；F821、链接／数学、卫生及diff检查通过。T09 CPU Gate B已验收：7000单次、630恢复复现、252局部回合和14批量面板完整执行，必要回归绑定同一规划源码，NMPC／MPPI512预定平坦闭环全部成功。复杂邻车窗口过期后的碰撞、不可行刹车和未知区域停滞仍保留，不宣称普遍安全。

T10完整144回合已执行并通过代理无学习集合功能验收：NMPC与MPPI K512各九例简单平坦主验收全部成功，全部矩阵共116/144成功，无代理碰撞及执行限制违反。坡地终端地形不满足和交叉队形NMPC持续拒绝独立分类，不计复杂场景成功。逐步控制、广播预测和采样事件覆盖复核通过；结束分类单独修正，原144回合记录未改。56项驱动／安全集成与16项分类专项测试通过，实际物理依赖仍绑定625项回归及18项真实acados。独立记录：[T08](../experiments/t08_safety_projection_validation_2026-10-10.md)、[T09](../experiments/t09_planner_benchmark_validation_2026-10-10.md)、[T10](../experiments/t10_gathering_validation_2026-10-10.md)。

## 最新诊断证据

本机完整MPPI CUDA数值与负载验收已通过，见[新验收记录](../experiments/mppi_cuda_validation_2026-10-11.md)：独立stable_v2采用float32预测及float64敏感代价／更新；66项CUDA契约、695项相关回归（含18项真实acados）全部通过，0失败、0跳过。36个K/B面板、9个B1同噪声CPU对照、27个复杂输入对照通过，B1完整调用约加速6.2–10.4倍，尚未通过0.2 s实时门槛。CPU环境、历史配置和[原v1诊断](../experiments/mppi_cuda_diagnostic_2026-10-11.md)独立保留。

新GPU T09已完整通过Gate B（7000单次、630恢复、252局部回合、14面板）；新T10完整144回合通过CUDA功能门槛，NMPC／MPPI各9/9简单平坦成功，总110/144成功、0代理碰撞及执行限制违反。新增六个NPB失败已定位本地记忆量化覆盖车下采样点，产生0.20 m最近距离超过0.155 m支持半径的缺口；频率分类为任务截止，不能解释为CUDA数值失败。六类公开快照重放及异步重置隔离通过；逐步记录完整审核通过，710/710必要采样事件保留。正式训练继续暂停。

本轮独立完成[局部记忆、坡地集合位置与 NMPC 停滞修复](../experiments/gathering_failure_repairs_2026-10-11.md)：`direct_batch_v3` 保留同格直接观测，支持半径／未知边界不放宽；`leader_local_site_v8` 采用合法本地选址与实际到达复核；`nmpc_recovery_v4` 有界冷启动／短参考恢复。78 个定向完整回合为 **76 成功**（对应旧子集 44 成功），原 34 失败恢复 **32**；NPB 24/24、NMPC 交叉 12/12、坡地 NMPC 12/12、坡地 MPPI 10/12，两规划器平坦各 9/9，无代理碰撞及执行限制违反。804 项冻结回归通过（含 23 项真实 acados、66 项 CUDA）；随后追加的停车版本专项 283 项通过，批次有重合、不相加。33,912 次执行及 595/595 必要采样事件审计通过，145 个历史文件 SHA 不变。

剩余两例为 MPPI 96 s 坡地：seed23 安全关闭、seed47 安全开启，截止时全部瞬时门槛满足但驻留仅 8／9 步；近槽位 MOVE 微动打断驻留，180 s 配对均成功。平坦停车退化已在独立 `leader_local_site_v10`／组合 v11 下完成九例对照：仅本车实际停稳、当前位置支持合法时请求 HOLD，不认证未知共同场地。该高层版本尚未完成坡地／NPB 全矩阵；坡地证据仍属于 v8。开发失败／中断与原始结果保留，历史 Gate B/T10 不替代新版本完整评测，正式训练继续暂停。

## 下一步

1. [完整环境接线](../experiments/upstream_scale_environment_2026-10-14.md)的局部契约已通过，优先推进真实出生的完整集合、停车捕获及安全开/关闭环。完成统一版本后开展S04校准、S05全矩阵和T11原预算筛选；训练继续暂停，既有充分依赖证据不重复昂贵运行。
2. 旧尺度剩余两例 96 s 坡地停车／驻留仅作机制回归，保持原成功阈值；不先重复昂贵的旧尺度全矩阵，也不把 78 回合定向子集计作完整验收。
3. T11的独立动作／概率准备已完成；冻结新尺度、高层、规划器、动态限速与安全的统一版本并通过完整 Gate B／144回合后，按[T11交接计划](../t11_handoff_plan.md)接通真实执行与动作表示筛选。历史 Gate B/T10 不能替代，Gate C及正式训练条件仍未满足。真实足迹、Isaac/PhysX、分层出生和地理独立数据的证据分别记录。
4. 迁移至后续服务器时，依[CUDA验收计划](../cuda_acceptance_plan.md)补验目标硬件数值、恢复、资源及吞吐，不从本机结果推断其他设备性能；CPU感知、最终复核及安全QP的耗时分别测量。

[任务尺度审计](../experiments/task_scale_audit_2026-10-11.md)已完成只读核对：T10 的 200 m NPB 底图仍在 25 m 工作区、同一地理锚点及约 1.3–3.4 m 出生距离下使用；1.6 m 子目标与速度、传感支持及预测时域须联合校准。槽位半径 0.8 m 超过共同终端地形半径 0.75 m，局部 HOLD 捕获存在几何裕量不足；参数迁移、逐车足迹和分层出生尚未据此实施或验收。

[全项目物理尺度依据](../references/project_physical_scale_reference_2026-10-11.md)保留取得上游输入前的车型对照。用户已提供最高车速 **10 km/h**、外形 **5.015×3.200×1.850 m**、轮径／轮距／轴距 **1.0／1.8／2.7 m**，并确认暂用差速模型、缺失值可合理推测。[车辆基线](../architecture/upstream_vehicle_scale_baseline_v1.md)保留已知输入；最新设计见[独立代理 v2 假设](../architecture/upstream_vehicle_proxy_assumptions_v2.md)与[profile v2](../../configs/design/upstream_vehicle_proxy_v2.yaml)。前进硬上限采用上游 10 km/h，实际速度须由当前合法观测支持的停车距离、地形与轮速共同约束，1 m/s 不再作为长期前进上限。加速／制动已在共同代理基础验证；滚动规划、完整时域及任务链仍待迁移，实车标定另行推进。设计profile不可作为运行实验加载，完整新尺度闭环及实车能力未验收。[v1 假设](../architecture/upstream_vehicle_proxy_assumptions_v1.md)与[v1 profile](../../configs/design/upstream_vehicle_proxy_v1.yaml)完整保留。

实验结果以机器可读 strict acceptance、独立 final eval 和 checkpoint status 为准；`outputs/` 不作为当前决策来源。具体任务顺序见[路线图](../roadmap.md)，实验入口见[索引](../experiments/README.md)。
