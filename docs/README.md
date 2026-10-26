# 项目文档入口

当前工作是正式训练及其结果驱动修复；消融在主方案取得可用结果后推进，T15 治理同步执行。现行顺序由[实施计划](implementation_plan.md)确定，历史阶段文件不再提供本轮训练准入条件。

## 当前入口

| 文档 | 职责 |
| --- | --- |
| [当前状态](current_status.md) | 当前训练状态、已核验结果与下一步 |
| [实施计划](implementation_plan.md) | 正式训练工作顺序、结束条件和延后范围 |
| [实验安排](experiment_plan.md) | 当前单主方案、既定评估及未来比较 |
| [规划架构](architecture/planning_pipeline.md) | 活跃执行链、代码索引与配置清单 |
| [接口规范](interface_spec.md) | 当前观测、动作、规划、时间与恢复契约 |
| [技术说明](technical_design.md) | 当前设计选择及研究问题 |
| [训练操作手册](runbooks/train_proxy.md) | 当前运行、监控、恢复及旧入口边界 |
| [Windows／WSL2 部署交接](runbooks/windows_wsl2_deployment_handoff_2026-10-23.md) | 10月26日新包、SAT三轮完整状态、局域网传输、依赖及有界验收 |
| [T15 治理清单](t15_governance.md) | 已完成治理、引用边界和按需拆分规则 |
| [路线图](roadmap.md) | 正式训练后的消融、泛化和物理验证 |
| [输出管理](references/output_management.md) | run、checkpoint、TensorBoard 与证据规范 |

## 当前实现与证据

- [SAT三轮MPPI从零百万步训练](experiments/t11_mppi_sat_long_training_2026-10-25.md)：全新初始化、SAT512×3；10月26日14:49已按用户指令停止，第71批完整保存点902,079步，日志末点911,630步；不自动恢复。
- [MPPI查询合并与迭代优化](experiments/t11_mppi_query_optimization_2026-10-25.md)：SAT接入、软查询合并、精简回传及审查优化；选3轮，整段耗时减少38–45%，短目标10／10、0失败。6,000诊断步、0学习，109项检查通过。
- [两种碰撞算法对照](experiments/t11_collision_algorithms_2026-10-25.md)：已观测栅格SAT硬查询快10.34倍、闭环耗时减少18–21%，短目标10／10、0失败；圆形真值对照整体耗时相近。3,600诊断步、0学习更新，生产后端未切换。
- [MPPI执行修复与调参](experiments/t11_mppi_execution_repair_2026-10-25.md)：修正规划车身覆盖过大，0.5米可达目标从0／10到达提高到10／10；原目标平均进展0.003→0.652米，仍含不可达目标。5,400诊断步、0学习更新，未增加行为子模块。
- [同保存态目标执行对照](experiments/t11_target_execution_probe_2026-10-25.md)：1,200诊断步、0学习更新；转向队友只让2／10辆MOVE车明显推进，其余仍停滞，后续候选筛查及修复见上项。
- [百万步训练详细分析](experiments/t11_crater_reward_detailed_analysis_2026-10-25.md)：解释学习曲线、方向选择、长时停滞、地形失败及奖励正分；下一步优先同状态目标执行对照，诊断未产生新学习步。
- [百万步故障修复、恢复及结束结果](experiments/t11_crater_reward_recovery_2026-10-24.md)：新增实际采样999,812步、独立评估53,860步，已按预算结束；最终短窗平均仍扩散，完整面板0/8成功；第82批checkpoint、owner分片、events及一次性总结齐备。
- [原百万步续训](experiments/t11_crater_reward_long_training_2026-10-24.md)：00:52异常退出的历史证据，新增采样25,551步，最后完整保存点62,628步。
- [坑分档后的奖励修复与5万步短训](experiments/t11_crater_reward_short_training_2026-10-23.md)：49,871学习步／4,427评估步已结束；短窗失败减少，平均队形仍扩大；最终checkpoint与events保留供部署交接。

- [奖励审计与算分实验](experiments/t11_reward_audit_2026-10-23.md)：172对保存轨迹全部偏向假设提前失败，理想成功算分也低于快速失败；明确持续成本尺度问题及“只罚不终止”的物理／事件边界，0新增环境／学习步。
- [本轮训练结束与问题分析](experiments/t11_final_training_review_2026-10-23.md)：总预算使用1,999,763步，障碍分支20,986个结束回合0成功；最终独立评估0/16，回报提高伴随存活缩短，下一步核查安全延续与提前失败的回报排序。
- [优化源码正式续训](experiments/t11_optimized_training_resume_2026-10-23.md)：第23批／261,149步完整状态承接、剩余预算及当前命令；[障碍输入初次切换](experiments/t11_obstacle_input_training_2026-10-22.md)保留准入与初始化边界。
- [局部障碍输入规划](architecture/local_obstacle_input_plan_2026-10-22.md)及[实施记录](experiments/t11_local_obstacle_input_2026-10-22.md)：固定格、增量感知和软反馈。
- [当前采样性能分解](experiments/t11_obstacle_sampling_profile_2026-10-22.md)及[优化实施](experiments/t11_sampling_optimization_2026-10-22.md)：隔离源码、104项检查、1,178步配对，四owner短测吞吐2.006倍；已按用户要求用于正式续训。
- [地图主线](architecture/lunar_map_mainline.md)、[车辆代理假设](architecture/upstream_vehicle_proxy_assumptions_v2.md)：当前物理与数据来源。
- [实验索引](experiments/README.md)：历史结果及复现资料。文件名中的 T11 保留来源身份，不表示仍执行原阶段门槛。

## 历史与边界

[本次完整文档快照](archive/formal_training_transition_2026-10-22/README.md)保留路线调整前的全文及相对链接。[归档索引](archive/README.md)、[设计基线](design/README.md)、[研究参考](references/)和[历史状态](status/)按需追溯。

[T11 交接入口](t11_handoff_plan.md)保留历史跳转；动作消融、旧完整矩阵和阶段门槛不自动接续。历史配置保留复现价值，启动配置以架构中的 active 清单及用户授权为准。

证据以有效配置、源码和原始 metrics 为准。TensorBoard、截图和 checkpoint 名称不能单独证明成功；Proxy 任务结果与 Isaac／PhysX 分开记录。`outputs/` 默认不提交。根目录 [AGENTS.md](../AGENTS.md)与两个仓库 Skill 只维护工作方式，不积累实验事实。

`.rgignore` 默认略过 `docs/archive/`、`docs/design/`、`docs/status/`；追溯时显式指定路径。数学公式使用 `$...$` 或独立行的 `$$`。
