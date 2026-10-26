> 历史完整快照（2026-10-22 正式训练主线切换前）。原文件：`docs/technical_design.md`。保留当时正文，仅调整相对链接；当前状态见[现行入口](../../../current_status.md)。

# 当前技术说明

历史exp167保留四维局部目标请求；新诊断采用条件极坐标／HOLD意图和统一工厂，接入真实NMPC、确定性／47类基线及CPU批量地形与邻车MPPI。共享N1 CNN＋GRU Actor和集中式Critic的正式训练仍暂停。具体职责及入口统一见[规划架构](architecture/planning_pipeline.md)，字段、奖励和终止规则统一见[接口规范](interface_spec.md)。

`finite_task_v2` 在独立配置中修复有限任务时间、剩余时间观测、邻车身份、停车驻留及折扣塑形；旧 exp167 语义作为版本化对照保留。新停车阈值只构成诊断任务定义，尚未完成逐车足迹和共同作业区的全部物理假设验证。

Actor 和局部规划器保持去中心化信息边界；团队指标和奖励可使用集中状态，不能把真实集合目标或团队驻留真值加入 Actor。已有变时长折扣、采样 latent 和变换后 log-prob 继续复用。

目标研究路线为低维意图、可替换 NMPC/MPPI 规划器和可选安全层，统一协议已在 T03 冻结，动作与CPU规划器已完成契约及局部执行验收，T08代理安全投影、T09 CPU固定千例评测和T10简单集合门槛已验收；复杂地形失败保留，CUDA验证延期。正式训练仍暂停，当前无通过完整任务验收的推荐 checkpoint。

旧技术摘要保存于[历史快照](../../technical_design_snapshot_2026-10-09.md)；设计背景保留在[设计基线](../../../design/README.md)。性能、风险及实验状态以[当前状态](current_status.md)和验收记录为准。

T03独立诊断路径已实现统一规划协议与NMPC桥接，见[接口规范](interface_spec.md)和[T03验收](../../../experiments/t03_planner_protocol_validation_2026-10-10.md)。T04条件动作与Gate A已通过[契约验收](../../../experiments/t04_polar_hold_validation_2026-10-10.md)，T05–T07交付及限制见[T07验收](../../../experiments/t07_mppi_terrain_neighbor_validation_2026-10-10.md)，T08–T10实施面见[当前计划](implementation_plan.md)，代理安全投影、固定千例与无学习集合分别验收。
