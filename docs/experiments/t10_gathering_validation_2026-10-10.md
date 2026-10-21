# T10 K512无学习集合闭环

日期：2026-10-10，更新至2026-10-11。状态：T09 CPU Gate B已通过，正式144回合完整执行，T10代理无学习集合功能已验收。[T09记录](t09_planner_benchmark_validation_2026-10-10.md)、[实施计划](../implementation_plan.md)与[运行手册](../runbooks/planner_diagnostics.md)给出完整门槛；无Actor、Critic、优化器或训练checkpoint。

## 配置与信息边界

`configs/experiment/t10_gathering_no_learning.yaml`：MPPI K512、两轮原参数、两环境×四车、真实传感通信与本地记忆。`leader_slot_v1` 以合法FULL leader目标和朝向建立0.8 m三槽位，每5步续发极坐标动作，0.2/0.3 m HOLD滞回；缺失／过期HOLD、编号让行仅依合法预测，已停车邻车正常处理。无Oracle集合点、未通信真值或稠密注入。

六类场景×23/31/47×NMPC/MPPI512×安全开关×96/180 s，共144回合。保持现有通信，不适配24 m、出生分层或完整足迹。NPB使用已有real_base窗口，与解析地形身份分别记录；不是独立地理样本。

## 接线pilot

`outputs/runs/t10_gathering_no_learning/pilot_seed23_cpu_20261010_02/`：明确20宏步预算，仅检查K512平坦紧凑接线。三个种子均完成真实25步驻留，无碰撞、无执行限制违反；`metrics/episode_*.json` 与 `summary_0000.json` 保留。该run在正式评测源码冻结前建立，仍是pilot，不能代替Gate B或144回合。

首个2宏步pilot保留 `pilot_seed23_cpu_20261010_01/`，没有完整回合，不计成功证据。每底层步采样回调避免最终HOLD清空样本；名义选中轨迹、实际控制、安全复核及执行预测分开记录。

## 正式门槛与恢复

正式入口必须提供 `gate_b_cpu=true` 且全部实际依赖源码匹配的汇总。`t10_evidence_driver_v2` 只允许T09不导入的本消费者文件独立更新，逐文件验证其他依赖。每5底层步通过公开环境／脚本／规划器／安全生命周期保存诊断状态，SHA校验，不含acados对象；只重置结束环境。中断重试轨迹保留独立attempt，不能覆盖先前记录。采样首次／每25步／新失败事件保留，HOLD没有采样。

恢复额外绑定NPB配置、manifest与全部运行数组指纹，以及冻结场景和分批有效配置。已落盘回合在恢复重放后必须匹配原结束判据、回报及失败计数，只允许计算墙钟不同。最终汇总逐个核对144个场景／规划器／安全／时限组合和K512，拒绝用重复或错误场景补足数量。

驱动专项run `outputs/runs/t10_gathering_no_learning/driver_contract_seed23_cpu_20261011_01/`：56项安全／集成／证据驱动测试通过、0失败、0跳过。实际规划／执行／通信依赖仍沿用T09绑定的625项回归与18项真实acados；消费者增强不改规划算法或物理约束。

T10功能要求安全开启180 s三类简单平坦共9例，NMPC与MPPI512各全部成功，且全程无碰撞和限制违反；其余场景必须完整执行并分类。团队实际距离、离散、线速、角速、间距、地形及25步驻留全部由原任务链判断，刹车不计成功。

## 完整矩阵结果

正式run为 `outputs/runs/t10_gathering_no_learning/acceptance_seed23_cpu_20261011_01/`，不可变原始生产版本为 `t10_evidence_driver_v2`。`metrics/strict_acceptance.json` 绑定全部144组合、原始源码、回归与逐步记录；`metrics/episodes_144.csv` 和 `family_results_144.json` 便于复核。全部真实预算执行，没有降低K或截短困难场景。

**T10功能门槛通过：安全开启180 s的三类简单平坦场景，NMPC与MPPI512分别9/9成功；全部144回合共116成功，其余为真实失败。** 各设置结果如下；每行12例包含三个队形种子、安全开关与两个任务时限，不能作为12个独立地理样本。

| 场景 | 规划器 | 成功 | 代理段内碰撞事件 | 执行限制违反 |
| --- | --- | --- | --- | --- |
| crossing | nmpc | 8/12 | 0 | 0 |
| crossing | mppi512 | 12/12 | 0 | 0 |
| flat_compact | nmpc | 12/12 | 0 | 0 |
| flat_compact | mppi512 | 12/12 | 0 | 0 |
| flat_irregular | nmpc | 12/12 | 0 | 0 |
| flat_irregular | mppi512 | 12/12 | 0 | 0 |
| flat_reverse | nmpc | 12/12 | 0 | 0 |
| flat_reverse | mppi512 | 12/12 | 0 | 0 |
| npb | nmpc | 12/12 | 0 | 0 |
| npb | mppi512 | 12/12 | 0 | 0 |
| slope | nmpc | 0/12 | 0 | 0 |
| slope | mppi512 | 0/12 | 0 | 0 |

中等坡地失败包含终端地形不满足：leader冻结锚点及槽位不在线迁移，延长预算不能自动修复停车区域。交叉队形中NMPC的失败表现为持续求解拒绝；不将启动时三次消息缺失解释为整回合通信故障。复杂失败全部保留，不宣称全部地形集合成功或规划器普遍安全。

## 控制、采样与分类复核

最终逐步审计 `metrics/trace_audit_0001.json` 通过：102480个规划／执行实例记录，恰好等于全部回合实际底层步数乘四；实际下发值与执行预测首控制匹配。原始采样保留546次／7440次可采样调用，覆盖率7.34%，首次、每25步及新失败事件的546次应保留调用全部存在，所有原始采样形状均为512×15×2。HOLD没有采样，不纳入分母；保留的中断attempt单列，不增加完成回合数。

场景seed控制初始队形，MPPI规划seed保持原默认23，环境seed取解析配置，各批次分别记录。NPB配置、manifest及运行数组SHA绑定同一个real_base窗口。六物理核只用于互不重叠的CPU批次调度，每个进程单Torch线程；实际两环境×四车批量接口和异步结束均执行。辅助调度、SHA、命令及中断／恢复日志留在run中，中断不计规划失败。计算墙钟和模拟任务时间分开，不宣称0.2 s实时运行通过。

原生产记录的失败标签保持不变。补充run `outputs/runs/t10_gathering_no_learning/classification_review_seed23_cpu_20261011_02/` 仅复核结束后的分类，使用 `symptom_frequency_v2`，16项专项测试通过，全部144原始回合SHA未变。当前消费者版本 `t10_evidence_driver_v3` 只修改分类、汇总及版本字面量，AST检查确认物理、规划、安全、通信、动作、场景和恢复不变；旧v2热状态不能静默恢复到v3。必要18项真实acados证据沿用相同实际依赖，未重复昂贵物理实验。

复核后的分类为 `{'success': 116, 'terminal_terrain': 24, 'planning_rejection': 4}`，原标签单独保存在 `recorded_failure_categories`。通信／让行按高层决策频率，规划／执行按四车实际底层调用频率，阈值与版本明确记录；该分类只定位症状，不改变终止原因、回报、碰撞或任务成功。

辅助检查的失败日志保留：首次最终轨迹审核匹配到自身报告，修正文件匹配后通过，见正式run的 `logs/final_v2_step_01.log` 与 `final_v2_retry_step_01.log`。首次分类测试没有向子进程传入 `PYTHONPATH=scripts`，收集失败保留在 `classification_review_seed23_cpu_20261011_01/`；补齐路径后另建上述02 run，16项通过、无跳过。这些是审核／测试入口失败，不计作规划回合失败。

最终文档链接、数学分隔符、仓库卫生、Ruff F821及 `git diff --check` 全部通过，命令和结果保存在分类复核run的 `metrics/static_acceptance.json`。

## 结论与交接

T08代理安全投影、T09 CPU Gate B及T10代理无学习集合功能分别验收。Gate C、CUDA、真实车体足迹、24 m通信／出生分层等完整任务适配及PhysX仍待办。下一步准备[T11动作表示消融](../t11_handoff_plan.md)，当前没有推荐训练checkpoint，正式训练继续暂停。
