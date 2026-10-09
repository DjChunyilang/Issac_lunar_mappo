# 当前状态

事实截至 2026-09-26；文档入口于 2026-10-09 整理。本页只记录已确认的当前状态。原详细快照见[归档](archive/current_status_snapshot_2026-09-26.md)。

## 当前主线

- 地图采用真实轨道 DEM、插值表示及 NASA 模型坑石增强。[地图主线](architecture/lunar_map_mainline.md)解释数据来源与边界；现有 7 张地图来自同一个 200×200 m NPB 窗口，不是 7 个独立地理区域。[验收记录](experiments/nasa_sfd_v2_2026-09-24.md)包含 36 项地形测试及 GPU 无学习检查。坑石模型尚未针对 NPB 标定，处理网格不等于测量精度。
- [任务适配方案](architecture/lunar_task_adaptation_v1.md)已接受为下一实施方向，包括暂定 24 m 完整通信范围、分层出生、停车足迹与驻留判据；运行代码和配置尚未切换。
- [学习系统审查](architecture/learning_system_review_2026-09-26.md)已完成首轮无学习核验。时间／终止语义、邻车实体对应和回报累计需先修正并验证；新网络或算法没有通过性能验收。
- 当前实现是 [exp167 连续局部目标＋NMPC](experiments/exp_167_local_goal_nmpc.md)。`prefix_consistency_v2` 已通过固定目标配对诊断，详见[规划修正记录](experiments/exp_167_planner_correction.md)；这不是完整任务成功证明。

## 验收与 checkpoint

exp167 限额 pilot 训练前后各 96 场景均为 0 成功、0 碰撞、96 超时，未收敛。**当前没有通过完整任务验收的推荐 checkpoint**。旧 pilot 的 `outputs/runs/exp167_local_goal_nmpc/pilot_seed23_100k_cpu/checkpoints/latest.pt` 仅供失败诊断，不能直接续训新版规划器。Isaac/PhysX 高保真闭环验证尚未完成。

Active-DSTC 及 exp166 后续消融暂时搁置；相关历史产物在本机未恢复，不能补写未核验数值。正式网络训练继续暂停。

## 下一步

1. 扩展并隔离真实地理训练／评估区域，再冻结地图比较协议。
2. 实施任务适配并做 96/180 s 无学习闭环配对；补充 GPU NMPC 资源验证。
3. 修正时间／终止和邻车信息合同，核验动作执行与回报语义；依据完整任务指标再决定训练预算。

实验结果以机器可读 strict acceptance、独立 final eval 和 checkpoint status 为准；`outputs/` 不作为当前决策来源。具体任务顺序见[路线图](roadmap.md)，实验入口见[索引](experiments/README.md)。
