# 路线图

计划更新时间：2026-10-12；工程事实截至 2026-10-11。当前事实与产物入口见[current_status.md](current_status.md)。

## 地图部分的确定主线

2026-09-25用户确认采用**真实轨道DEM＋插值/重采样表示＋NASA模型坑石增强**，详见[地图主线](architecture/lunar_map_mainline.md)。当前7张NPB地图、36项地形测试和四组32环境×32步GPU无学习检查已完成，见[验收记录](experiments/nasa_sfd_v2_2026-09-24.md)。

后续按以下顺序推进：

1. 扩展真实地理覆盖与有来源记录的地图窗口；需要时补充独立NASA区域。
2. 先隔离地理区域，再构造裁剪和增强；同地点各版本归同组，避免训练/测试泄漏。
3. 按[S00–S05尺度迁移计划](implementation_plan.md)和[差速代理 v2 假设](architecture/upstream_vehicle_proxy_assumptions_v2.md)，联合迁移车辆足迹、动态限速、合法感知、通信、出生与停稳判据，并完成新版本无学习闭环和必要 GPU NMPC 资源适配。[2026-09-26任务适配方案](architecture/lunar_task_adaptation_v1.md)保留原 24 m 通信与 96/180 s 对照语义；新版 P0 已采用停稳参数，完整新尺度适配仍待实施。
4. 固定算法与执行条件，配对比较真实底图、加坑、加坑石；据吞吐量提出训练预算，明确恢复后再训练。

纯随机地图和旧增强配置保留为历史/机制对照；LUPEX保留为补充来源与跨源评估候选。NASA分布尚未在NPB标定，插值格距不当作测量精度。旧撤回草案不恢复为执行依据，当前授权和边界统一由主线说明承接。

## 研究路线重设计

已完成[终端区域研究综述](references/terminal_feasibility_review_2026-09-19.md)、[完整设计](architecture/terminal_feasibility_redesign_v1.md)与[实验协议草案](experiments/terminal_feasibility_research_protocol.md)。新的研究顺序为：任务定义与可达见证 → 完整任务控制参照 → 局部信息与通信机制实验 → 同接口学习增益比较。方案尚未实施，不自动启动训练或恢复Active-DSTC。

## 当前执行点

T00 仓库审计、T01 上下文入口和 T02 P0 契约修复已完成，见[T00–T02计划快照](archive/implementation_plan_t00_t02_snapshot_2026-10-10.md)与[P0 验收](experiments/t02_p0_contract_validation_2026-10-09.md)。新契约采用独立配置，旧 exp167 保留历史语义。T03统一规划器协议已通过[独立验收](experiments/t03_planner_protocol_validation_2026-10-10.md)，T04低维意图／HOLD与Gate A已通过[契约验收](experiments/t04_polar_hold_validation_2026-10-10.md)，T05基线、T06/T07 CPU批量MPPI与局部闭环已通过[阶段验收](experiments/t07_mppi_terrain_neighbor_validation_2026-10-10.md)，该初始阶段CUDA补验延期；本机后续已独立通过CUDA契约／负载、新GPU Gate B及144回合CUDA集合功能，详见[CUDA记录](experiments/mppi_cuda_validation_2026-10-11.md)；T08代理安全接口、T09 CPU Gate B及T10完整144回合集合功能已分别验收；两规划器简单平坦各9/9成功，复杂失败保留。随后[局部记忆、坡地选址与 NMPC 停滞修复](experiments/gathering_failure_repairs_2026-10-11.md)完成 78 个定向回合、76 成功；剩余两例 96 s 坡地仅作机制回归，平坦停车修正版尚未通过完整矩阵。

最新优先任务是[S00–S05车辆尺度迁移](implementation_plan.md)。[设计 profile v2](../configs/design/upstream_vehicle_proxy_v2.yaml)采用 **10 km/h（2.777778 m/s）前进硬上限**，按合法感知、制动、地形与轮速动态限速；倒车暂为 **0.5 m/s**，**1 m/s 仅用于首轮测试**。独立S00参数及S01公共执行基础已接通，见[记录](experiments/upstream_scale_foundation_2026-10-12.md)；完整规划与环境链仍待迁移，旧配置保留。旧尺度两例不触发昂贵的旧全矩阵优先重跑。T11须等待新统一版本的完整 Gate B 与 144 回合，具体见[交接计划](t11_handoff_plan.md)；Gate C与正式训练条件仍未满足。

总体顺序以实施计划为准：独立尺度契约与执行限制 → 动态限速、合法感知与预测协议 → 新尺度集合与出生 → 统一版本完整 Gate B／144 回合 → 通过门槛后的动作／规划器消融。地图独立区域与完整任务适配仍是后续验证条件，当前不恢复正式训练。[T08–T10快照](archive/implementation_plan_t08_t10_snapshot_2026-10-12.md)、[v1 假设](architecture/upstream_vehicle_proxy_assumptions_v1.md)和[v1 profile](../configs/design/upstream_vehicle_proxy_v1.yaml)保留历史语义。

exp167连续局部目标＋acados NMPC＋N1 GRU MAPPO已实现，限额pilot累计99,999次底层训练交互，训练前后均0/96成功。网络训练暂停，没有推荐checkpoint。

`prefix_consistency_v2`已完成输出轨迹/控制一致性、可行运动前缀和停车消息语义修正；固定短目标与长目标配对复测改善，但尚不代表完整任务收敛。详见[规划修正记录](experiments/exp_167_planner_correction.md)。

## 当前实现的技术待办

以下是exp167仍需解决的工程问题；后续项目研究的总体顺序以本页上方的重设计和实验协议为准。

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
