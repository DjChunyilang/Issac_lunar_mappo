# 路线图

更新时间：2026-09-18。当前事实与产物入口见[current_status.md](current_status.md)。

## 当前执行点

exp167连续局部目标＋acados NMPC＋N1 GRU MAPPO已实现，限额pilot累计99,999次底层训练交互，训练前后均0/96成功。网络训练暂停，没有推荐checkpoint。

`prefix_consistency_v2`已完成输出轨迹/控制一致性、可行运动前缀和停车消息语义修正；固定短目标与长目标配对复测改善，但尚不代表完整任务收敛。详见[规划修正记录](experiments/exp_167_planner_correction.md)。

## 后续决策顺序

1. 保留pilot、固定目标诊断和修正后配对结果，严格区分规划版本。
2. 讨论复杂会车、目标切换、稀疏/过期通信与执行误差的完整闭环验证方案，优先检查规划失败、前缀介入和实际运动。
3. 依据完整任务成功、阻塞、耗时及模型差异，再决定是否恢复训练及预算；当前不自动启动新训练。
4. 正式proxy验收通过后，再制定Isaac/PhysX闭环验证方案。

## 保持暂停

- Active-DSTC及exp166后续消融。
- exp156架构比较、N2完整训练及额外种子。
- 旧pilot checkpoint向新规划版本直接续训。
- 仅凭固定目标到达、低碰撞或reward提升扩大训练。

历史路线图见[归档](archive/roadmap_before_exp167_2026-09-18.md)，其中的旧计划不再作为执行指令。
