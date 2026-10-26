# 当前技术说明

更新：2026-10-22。当前主方案以共享 N1 CNN＋GRU Actor 和集中式 Critic，学习条件极坐标 MOVE／HOLD 意图，通过 CUDA MPPI 在差速代理中执行；训练使用 skrl MAPPO。正式训练已经运行，工作顺序见[实施计划](implementation_plan.md)。

Actor 只使用本车状态、合法通信和私有局部记忆。固定格将已揭示的障碍接近风险、普通地面 cost 和 known 比例送入网络，并向规划与奖励提供一致的局部障碍信息；Critic 可使用团队状态，其 Oracle 预留槽为零。地图由环境／感知所有者持有，不能直接作为 Actor 或规划器输入。

当前采用固定远距月面、完整任务训练，无课程。策略每 5 s 产生一个意图，1 s 重规划／续发，0.2 s 执行；学习按真实持续时间计算折扣和 GAE。当前 soft-hazard 路径关闭预测完整足迹拦截和安全投影，保留执行限制、实际地形失败和停车判定，利用代价与奖励学习避险。

研究目标是持续靠拢、避险和自主停车驻留。先从正式训练取得主方案结果，消融延后；性能优化和功能修复根据现有轨迹与热点选择，不堆叠外置行为规则。

[规划架构](architecture/planning_pipeline.md)维护职责、活跃配置和源码入口；[接口规范](interface_spec.md)维护当前契约；[当前状态](current_status.md)记录真实结果。[旧技术说明](archive/formal_training_transition_2026-10-22/docs/technical_design.md)仅供追溯，Proxy 与物理验证分别评价。
