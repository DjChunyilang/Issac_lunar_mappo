> 历史完整快照（2026-10-22 正式训练主线切换前）。原文件：`docs/README.md`。保留当时正文，仅调整相对链接；当前状态见[现行入口](../../../current_status.md)。

# 项目文档入口

2026-10-22：[新感知采样全流程性能分解](../../../experiments/t11_obstacle_sampling_profile_2026-10-22.md)：200步／0更新配对，单owner策略步27.491 s；MPPI57.3%、感知／真实足迹23.2%、观测8.5%，CUDA格查询8640次／7.59 s。13组学习张量与动作／结果逐值一致；已列优化优先级，正式训练继续，生产源码不变。

2026-10-22：[停止旧训练，启动新感知／地形正式训练](../../../experiments/t11_obstacle_input_training_2026-10-22.md)：旧训练502,369步／52批保存退出，新Actor523／Critic1439、固定格增量感知与软障碍反馈从零训练；预算使用原200万步中的剩余1,497,631步。45项检查、四图72步准入及新run连续采样／TensorBoard6006通过，原结果与checkpoint保留。

2026-10-22：[固定栅格、增量感知与障碍输入实施](../../../experiments/t11_local_obstacle_input_2026-10-22.md)：0.2 m标签＋0.8 m普通地形、静态tile复用、三通道观测与软cost／奖励贯通；213项检查通过，前端2.09倍、4环境有界整体采样吞吐+28.45%，计费上界627／800底层步、0 PPO更新。实施验收后已按上方记录切换正式训练；[配置与架构规划](../../../architecture/local_obstacle_input_plan_2026-10-22.md)同步实际实现。

2026-10-22：[关闭预测足迹拦截，调整地形cost/奖励并启动正式训练](../../../experiments/t11_soft_hazard_training_2026-10-22.md)：108项检查通过，748步启动诊断；16环境13.53步/s。全新A1/seed23、200万学习底层步、1km远距月面和25分钟回合，无课程，TensorBoard实时保存。

2026-10-22：[16/24环境容量与进一步优化验证](../../../experiments/t11_cuda_footprint_optimization_2026-10-22.md#补充三并行容量与进一步查询优化验证)：1951步、0 PPO更新；24环境5.54步/s，比同轮16环境快7.4%，但保存时进程swap约1.07GiB。短测未OOM，正式默认保留16环境；无收益的查询候选已撤回，训练继续停止。

2026-10-22：[顺序优化与16环境小批复测](../../../experiments/t11_cuda_footprint_optimization_2026-10-22.md#补充二顺序优化缓存与轨迹批量查询)：修复缓存循环引用并合批GPU／CPU查询，81项检查通过；764步、0 PPO更新，采样4.63→5.30步／秒，显存峰值采样5140→2508 MiB，行为逐值一致，TensorBoard已保存。正式训练保持停止。

2026-10-22：[停止训练并优化GPU足迹复核](../../../experiments/t11_cuda_footprint_optimization_2026-10-22.md)：长训已停止；同状态短程配对约2.55倍提速、77项检查通过。另按用户要求完成优化版16环境小采样，764步、0 PPO更新、4.63步／秒，TensorBoard已保存；不自动恢复训练。

2026-10-21 23:35：[月面训练执行性能修复](../../../experiments/t11_lunar_runtime_performance_2026-10-21.md)：近邻查询与静态足迹复用提速，同状态执行逐值一致；新run保留原任务和200万步预算，冻结评估与四owner学习采样并行。

2026-10-21最新：[25分钟仅远距月面正式训练](../../../experiments/t11_far_fixed_lunar_launch_2026-10-21.md)：用户已授权1500 s完整回合与启动；1 km月面图、固定采样和成功率Pinned已接通，100步实际月面检查已通过，正式进程已启动（PID1654210），覆盖此前3600 s待审候选。

2026-10-21最新：[月面地形与完整回合时限修订](../../../experiments/t11_lunar_terrain_duration_review_2026-10-21.md)：按用户要求采用真实DEM＋NASA坑石增强；1200s偏紧，候选3600s，需联动复审折扣和完整评估预算。1km地图构建／有界加载尚待接通，训练未启动；覆盖下方平地／30万评估候选。

2026-10-21最新：[仅远距、无课程正式训练参数审计](../../../experiments/t11_far_fixed_parameter_audit_2026-10-21.md)：用户要求只用远距完整任务；候选R200–300 m、1200 s、取消晋级／复习／阶段上限。全参数与静态检查已交付，学习200万／评估30万步待审；训练未启动，覆盖此前短探针优先安排。

2026-10-21：[TensorBoard曲线失败复核](../../../experiments/t11_tensorboard_failure_review_2026-10-21.md)：两组训练均0/34成功，末批总体EV=.993掩盖停车组.213及终端信用偏移；回报骤降主要是回合混合结算。仅分析，新增学习／环境步为0。

2026-10-21：[后续训练强制记录 TensorBoard](../../../references/output_management.md#当前-t11-的实时监看与历史回放)。当前 T11／课程／短训练已接入；最近学习率对照与基线已有回放曲线，查看命令和标签见链接。训练保持结束。

2026-10-21当前：[现有PPO学习率有界对照完成](../../../experiments/t11_optimizer_step_probe_2026-10-21.md)。10,000学习＋3,050评估步，较大学习率改善部分推进，停车仍1/4、0/4，未过筛；全部核验通过，进程已结束。3次有效全HOLD中两次驻留23/25被下一MOVE打断，下一项先验证现有5／6 s决策同状态配对，尚未执行；旧长训练不恢复。

2026-10-21最新约束：[后续在现有整体框架内调整](implementation_plan.md)，通信、网络、奖励、训练只是举例；环境、动作表达、规划／控制等已有组成也可按证据修改。避免不断叠加额外子系统，按完整执行链确定下一项最小改动，不限定四类或预设调整顺序。当前仅修正路线，训练保持停止。

2026-10-21：[联合停车探索与续停验证完成](../../../experiments/t11_joint_parking_exploration_2026-10-21.md)：9,906学习步＋3,588评估步，获得84次全HOLD、12对有效连续HOLD及46个停车成功。满偏置停车4/4、4/4，撤除后仍1/4、0/4；正确停车多数获正优势，但网络尚未学出状态区分。保存退出，下一项聚焦HOLD门控，不自动扩训。

2026-10-21：[价值、停车与方向顺序验证完成](../../../experiments/t11_v2_ordered_repair_2026-10-21.md)：Critic留出误差81.86→39.48；停车1万步未改善，联合持续HOLD样本仍缺失；方向辅助使20s平均缩小量采样−1.433→2.437m、均值−1.776→4.345m，但左侧仍失败。实际14,650环境步后结束，下一项聚焦联合停车探索，不启动长训练。

2026-10-21：[v2训练失败详细分析](../../../experiments/t11_v2_failure_analysis_2026-10-21.md)：全量轨迹确认方向均值近乎固定前进、近成功HOLD缺乏状态区分；停车实际曝光1.08%，一次驻留4.8 s被新MOVE打断。Critic时限／课程估值失配明显；未发现奖励累积、GRU或概率重放错误。本次0新增环境步／更新，训练保持结束。

2026-10-21：[v2课程训练已结束](../../../experiments/t11_v2_curriculum_launch_2026-10-20.md)：第2级用满25万底层步，20批／100轮PPO，未晋级；185个训练回合全部超时，最终四组完整评估在采样／确定性执行下均0/4成功。完整checkpoint已核验，进程已退出，不自动续训。

2026-10-20 16:36：[v2训练进度](../../../experiments/t11_v2_curriculum_launch_2026-10-20.md)：12.8万底层步、10批／50轮PPO，checkpoint和实际优化已核验；仍在第2级，92个结束回合全部超时。HOLD约35%，主靠拢评估未优于初始化，最远组仍平均发散；按阶段预算继续运行。

2026-10-20 13:13：[v2课程新训练已启动](../../../experiments/t11_v2_curriculum_launch_2026-10-20.md)。新配置实际准入1,304步通过；A1/seed23、四owner×四环境、新网络／Adam，初始化checkpoint及四分片已核验。每级25万步上限，卡级结束；旧队列不恢复。

2026-10-20：[六项配置调整与审定](../../../experiments/t11_six_point_review_2026-10-20.md)已落地独立v2候选：12–30 m主训练、每级25万步上限、延长课程时限、lambda=.99／32决策序列、初始HOLD约.35、固定近中远评估。原配置保留，正式训练尚未启动。

2026-10-20：[下一次正式长训练前的完整参数审查](../../../experiments/t11_next_long_training_parameter_audit_2026-10-20.md)已完成。核对有效配置、生产代码及 checkpoint，列出出生／课程、奖励与信用、PPO／网络、感知通信、MPPI／安全和评估预算；241 个源码 SHA 与启动记录一致。参数未修改，等待审定下一轮方案，训练继续停止。

2026-10-20 11:33：[1 km 课程长训练已按用户要求停止](../../../experiments/t11_1km_curriculum_long_training_2026-10-19.md)。最终 1,614,642 底层步、264 次更新；第 264 批完整 checkpoint 与四分片已校验，所有训练进程退出。仍停留近距课程，未通过收敛筛选，不自动续训或补跑评估。

2026-10-19 17:58：[1 km 课程长训练已启动](../../../experiments/t11_1km_curriculum_long_training_2026-10-19.md)。地图 1000×1000 m、通信 300／600 m、随机出生逐级扩展至 150–300 m；奖励尺度与时间信用已核对，近距停车 4/4、四例实际奖励排序正确。单个 A1、seed23，最多 200 万实际底层步；首批 6,262 步、5 轮 PPO 和完整 checkpoint 已核验，当前继续运行，尚无收敛结论。

2026-10-19最新：[价值目标归一化短学习已完成](../../../experiments/t11_value_normalization_trial_2026-10-19.md)。19,863训练步；近距随机驻留1/4→2/4，但随机远距2.974 m仍低于自身初始化4.441 m、HOLD升高，整体筛选未通过。奖励权重与默认保持原值，本轮已保存退出。

2026-10-19最新：[三项组合闭环验证完成](../../../experiments/t11_combined_learning_validation_2026-10-19.md)。两组各约2万底层步、13批更新；组合随机远距最终2.61 m优于baseline，但低于自身初始化4.44 m，近距驻留仍随机1/4、均值4/4，未通过学习改善筛选。两组已保存退出，不追加预算。

2026-10-19执行更新：[按序修复与独立诊断已完成](../../../experiments/t11_learning_repair_execution_2026-10-19.md)：接通独立裁剪、小噪声初始化、终端前未生效动作的Actor信用屏蔽及完整更新归档。10项针对性检查通过，诊断按上界11,863/12,000步结束。冻结批次已修正停车信用方向，尚待新闭环学习比较；原三组不追加。

2026-10-19研究更新：[强化学习文献与三组结果深度复核](../../../experiments/t11_decision_credit_deep_review_2026-10-19.md)已完成，修正此前直接方向引导建议。下一项先检查价值基线与实际策略更新，再分离初始化先验、门控、方向噪声和停车状态覆盖；本轮未新增训练或环境步。

2026-10-19：[三组训练与行为比较已完成](../../../experiments/t11_decision_credit_results_2026-10-19.md)，各20万底层步、20次更新，正常退出。B仅减少跳变；C均值策略远距离靠拢明显，但随机策略和近距离驻留未改善。两组均未通过筛选，本轮结束，下一项研究合法局部方向引导与停车区分；下方运行描述为历史状态。

2026-10-18 19:26：[A→B→C三组短训练已启动](../../../experiments/t11_decision_credit_launch_2026-10-18.md)，每组最多20万底层环境步，独立评估共预留1.8万步。A组初始评估中，随后进入学习，B/C依次接续；下方暂停描述为历史状态。


2026-10-18：[高层决策与学习信号修复计划](../../../t11_decision_learning_repair_plan.md)聚焦提前HOLD、向心MOVE和目标跳变。[同状态对照](../../../experiments/t11_decision_credit_2026-10-18.md)已完成九个短程分支：持续靠拢奖励排序正确，减少目标跳变有益。[持续决策、时间折扣及三组有界比较入口已实现](../../../experiments/t11_decision_learning_implementation_2026-10-18.md)；研究学习尚未启动，训练保持暂停。

2026-10-18 13:05最新：用户要求暂停训练，全部训练与后续队列已退出。A1完成100k，A0保留79,872转移完整恢复点，A2/A3未启动。下一步优先定位高层目标与学习信号问题，见[当前状态](current_status.md)和[实施安排](implementation_plan.md)；下方继续运行描述为历史记录。

2026-10-18：[A1完成100k后的失败诊断](../../../experiments/t11_a1_failure_final100k_2026-10-18.md)：固定16环境段256回合均超时，末期约95%选择HOLD；MOVE执行正常，学习梯度仍强化提前停车。结果已保存，本轮未改训练或队列。

2026-10-18：[A0第35批集合失败诊断](../../../experiments/t11_a0_failure_update35_2026-10-18.md)：71,680有效转移、192回合均超时；高层目标缺少向心偏置且频繁换向，近期队形整体发散，执行复核通过率99.7466%。本轮只读，训练与队列继续。

2026-10-17最新：[四进程16环境长期续训](../../../experiments/t11_owner16_training_resume_2026-10-17.md)已从10,752转移完整恢复，保留128宏步与5轮PPO，继续A1原100k及后续四方案队列。下方验证退出为历史状态。

2026-10-17最新：[checkpoint OOM修复与完整保存验证](../../../experiments/t11_checkpoint_oom_repair_2026-10-17.md)完成：35项不同测试通过，四进程16环境连续两批更新/保存和新进程恢复通过，PSS峰值7.43GiB、整机最低可用5.21GiB、训练树swap为0。完整恢复点达10,752转移/85轮PPO，验证已退出。

2026-10-17最新：[六进程48环境skrl重启](../../../experiments/t11_skrl_owner48_restart_2026-10-17.md)已启动，替代旧两环境队列；继承3,584转移学习状态，独立记录新采样协议，监控整个进程树的硬件负载。

2026-10-17最新：[优化版skrl续训与硬件监测](../../../experiments/t11_skrl_training_restart_2026-10-17.md)完成前3批，完整恢复点达到2,816转移/55轮PPO；每批采样约167–169 s、更新约4.3–4.5 s，CPU平均205%、GPU7.6%。A1继续原100k，随后按原顺序推进A0/A2/A3。以下暂停为历史记录。

2026-10-17最新：[深度优化与skrl迁移](../../../experiments/t11_skrl_deep_optimization_2026-10-17.md)已交付；同状态吞吐再提高1.46倍，完整更新等价，常驻进程扩批最高实测约5.58转移/s。训练继续停止，当前推荐与证据见记录。

2026-10-17：[优化后深度性能分析](../../../experiments/t11_deep_performance_analysis_2026-10-17.md)完成，约半数关键路径仍在主线程，双独立采样进程容量探针吞吐提高1.53倍；后续按快照/打包复用、安全预测批处理和环境进程调度推进。这份分析保留优化前证据；实施结果见上方最新记录。

2026-10-17：[运动性能优化与测试](../../../experiments/t11_motion_optimization_2026-10-17.md)完成，短程同状态吞吐8.13倍，128宏步含重置回放保持原行为，125项相关测试通过。训练继续保持停止，恢复点不变。

2026-10-17最新：用户要求停止训练、优先优化实际运动时的采样性能。A1及四方案队列已退出，保留2,048个有效转移、8批/40轮PPO的完整checkpoint；未经用户明确启动不恢复。见[当前状态](current_status.md)，以下为此前运行记录。

2026-10-16最新：[学习速度承诺修复](../../../experiments/t11_learned_lease_repair_2026-10-16.md)在同状态对照中让8/8辆车产生位移，14项针对性测试通过。A1已从1,024个有效团队转移恢复100k训练，A0/A2/A3按原顺序排队；下述暂停与零运动为修复前记录。

2026-10-16晚间：[优化版真实训练试跑](../../../experiments/t11_optimized_training_trial_2026-10-16.md)完成一个A1续训批次，累计1,024个有效团队转移、20轮PPO；采样142.8 s，但车辆仍零运动。试跑已保存退出，四方案自动队列仍暂停，当前事实以[当前状态](current_status.md)为准。

2026-10-16最新决策与[首批真实结果](../../../experiments/t11_signal100k_start_2026-10-16.md)：A1已完成256有效团队转移/5轮PPO，但实际执行全部刹车、零运动；首次100k准入与完整系统验收分开，暂停剩余144/原语47，A1→A0→A2→A3各100k、种子23按同条件推进，见[T11当前计划](t11_handoff_plan.md)。下列历史“完整矩阵先于100k”的顺序已由用户此次指令替代，不能再作为启动阻塞。

## 阅读顺序

1. [当前状态](current_status.md)：当前主线、验收、checkpoint 和暂停条件。
2. [实施计划](implementation_plan.md)：S00–S05 车辆尺度迁移、10 km/h 前进硬上限与动态限速，以及统一新版本的完整 Gate B／144 回合验收门槛；研究全貌见[路线图](roadmap.md)。
3. [实验索引](../../../experiments/README.md)：找到对应实验后，读取配置、run 和机器可读结果。
4. [活跃规划架构与代码／配置索引](architecture/planning_pipeline.md)、[接口规范](interface_spec.md)、[技术说明](technical_design.md)。
5. 按任务查阅[运行手册](../../../runbooks)、[架构与设计](../../../architecture)、[输出管理](../../../references/output_management.md)。

长期技术背景见[设计基线](../../../design/README.md)。[历史归档](../../README.md)、[研究参考](../../../references)和[状态汇报](../../../status)只按需查阅，不能覆盖上述当前入口。

前一阶段的 CPU／CUDA 验收要求与定向修复追加完整保留在[T08–T10计划快照](../../implementation_plan_t08_t10_snapshot_2026-10-12.md)，不再作为新尺度的默认执行计划。

从仓库根目录默认运行 `rg` 时，`.rgignore` 会略过 `docs/archive/`、`docs/design/` 和 `docs/status/`；追溯这些目录时明确指定路径或使用 `rg --no-ignore`。

## 事实来源与边界

2026-10-16 15:35：[等价加速与四进程续跑](../../../experiments/t11_compute_acceleration_2026-10-16.md)记录约2倍真实宏步加速、12例集合成功及NPB预加载修复；最新生产者为`gather144_braking_v5_fast_20261016_02`。完整矩阵和训练门槛仍未通过。

2026-10-16最新事实见[矩阵续跑记录](../../../experiments/t11_matrix_continuation_2026-10-16.md)：v5的NMPC两例也已成功；原评测继续补齐局部/批量，新144集合已启动。用户已授权门槛满足后推进100k，当前门槛未满足，正式训练尚未开始。下文按日期保留历史事实。

最新事实见[10月15日晚间修复记录](../../../experiments/t11_speed_lease_repair_2026-10-15.md)：7000单次、630重复性及A1完整128宏步诊断批次完成；MPPI新v5两个完整回合成功，NMPC和剩余评测继续。正式训练保持暂停。

[等待评测期间的并行准备](../../../experiments/t11_parallel_preparation_2026-10-15.md)交付耗时实测、离线学习/执行报告、失败回放图、[动作消融比较协议](../../../t11_ablation_protocol.md)、[服务器迁移手册](../../../runbooks/t11_training_handoff.md)及[版本证据对应表](../../../t11_evidence_map.md)。这些准备不改变既定验收或训练暂停。

S03完整环境独立入口为`upstream_scale_s03_environment_v2.yaml`，见[接线与实际出生探索](../../../experiments/upstream_scale_environment_2026-10-14.md)。完整环境两例安全开/关出生集合均完成，四组真实循环学习准备与36例单次校准已运行，详见[本轮记录](../../../experiments/upstream_scale_training_preparation_2026-10-14.md)。九类校准闭环和新尺度全矩阵仍待完成；正式训练继续暂停。

新尺度公共执行配置为 `upstream_scale_s01_review_v2.yaml`，见[S01复核修复记录](../../../experiments/upstream_scale_review_v2_repair_2026-10-13.md)。`upstream_scale_s02.yaml`的40步基线、56案例及54容量面板已执行，见[S02记录](../../../experiments/upstream_scale_planners_2026-10-13.md)。新增`upstream_scale_s03.yaml`及[S03感知／通信／停车组件](../../../experiments/upstream_scale_s03_components_2026-10-13.md)，完整环境与实时门槛仍待完成。

S03最新独立入口为`upstream_scale_s03_observation_v1.yaml`，见[低速精度与真实观测记录](../../../experiments/upstream_scale_s03_precision_observation_2026-10-13.md)：原始采样与物理预测拆分精度，私有密集记忆与消息v2接入旧N1形状。完整循环学习／安全环境及新尺度全矩阵仍待完成。

- 运行参数以 `configs/` 和实际代码为准；文档描述须与其核对。
- 严格 proxy 结果优先看 `_suite/metrics/strict_acceptance.json`、`suite_summary.json`、独立 `metrics/final_eval_proxy.json` 与 `metrics/checkpoint_status.json`。
- GIF、截图、TensorBoard 曲线、训练 reward 和 checkpoint 文件名不能单独证明 strict pass。
- `outputs/` 是生成目录，默认不进入 Git。实验解释写在 `docs/experiments/`，产物目录规范见[输出管理](../../../references/output_management.md)。
- Isaac/PhysX 闭环评估与 proxy 训练分别记录，不把 proxy 结果写成物理训练结果。

Markdown 数学公式使用 `$...$` 和独立行的 `$$`。

现有运行配置包括T05基线、T06基础MPPI、T07地形／邻车MPPI及独立T08安全、T09评测、T10集合诊断；T04、T03、P0契约诊断及exp167历史对照继续保留，具体 allowlist 在规划架构中维护。新尺度已新增独立S00参数审计、S01执行基础与S02规划诊断，见[基础执行记录](../../../experiments/upstream_scale_foundation_2026-10-12.md)与[运行手册](../../../runbooks/upstream_scale_diagnostics.md)；实际任务链已通过独立S03环境接通，设计 profile 不属于运行 allowlist。根目录 [AGENTS.md](../AGENTS.md) 提供简短工作流程；历史主线不能覆盖当前接口。

[T05–T10无学习规划诊断手册](../../../runbooks/planner_diagnostics.md)提供规划器选择、固定场景、原生47回放、请求回放及现有传感链闭环命令。

[完整CUDA验收与后续诊断](../../../experiments/mppi_cuda_validation_2026-10-11.md)记录独立stable_v2配置、已通过的数值／负载、GPU千例和完整集合功能验收，以及复杂失败诊断；范围见[CUDA验收计划](../../../cuda_acceptance_plan.md)。[首次接通诊断](../../../experiments/mppi_cuda_diagnostic_2026-10-11.md)保留原v1数值失败和有限pilot证据。

[局部记忆、坡地集合位置与 NMPC 停滞修复](../../../experiments/gathering_failure_repairs_2026-10-11.md)记录独立版本、保守信息边界及定向完整回合对照；不替代修改后完整 Gate B/T10。

[T11交接计划](t11_handoff_plan.md)列出动作表示对照、契约准备及原规划的逐级筛选预算；须先通过新统一版本的完整 Gate B 与 144 回合，历史 T10 验收不替代该前置条件，正式训练继续暂停。

[T11无学习准备记录与复跑命令](../../../experiments/t11_action_preparation_2026-10-13.md)提供四组共享骨干的动作／概率诊断。独立`configs/ablation/`准备配置不能作为运行experiment加载；新尺度桥接与真实循环准备已另行实现，完整学习筛选尚未启动。

[任务尺度审计](../../../experiments/task_scale_audit_2026-10-11.md)区分地图与工作区、实际出生与配置值，核对局部目标、感知支持、车辆、通信和时限的匹配；参数建议仍需独立版本验证。

[全项目物理尺度依据](../../../references/project_physical_scale_reference_2026-10-11.md)保留取得上游输入前的车型对照。[上游车辆尺度基线](../../../architecture/upstream_vehicle_scale_baseline_v1.md)已记录 10 km/h、5.015×3.200×1.850 m 及轮径／轮距／轴距 1.0／1.8／2.7 m。最新入口为[差速代理 v2 假设](../../../architecture/upstream_vehicle_proxy_assumptions_v2.md)及[设计 profile v2](../../../../configs/design/upstream_vehicle_proxy_v2.yaml)：前进硬上限为 10 km/h（2.777778 m/s），实际速度按合法感知、制动、地形与轮速动态限制；倒车暂为 0.5 m/s，1 m/s 只用于首轮测试。独立执行基础已实际达到满速及停车，完整环境与规划器尚未验收；旧配置、[v1 假设](../../../architecture/upstream_vehicle_proxy_assumptions_v1.md)和[v1 profile](../../../../configs/design/upstream_vehicle_proxy_v1.yaml)保留原语义。
