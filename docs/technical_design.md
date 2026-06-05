# 技术设计说明

第一阶段实现遵循 `isaac_sim_skrl_mappo_multi_rover_tech_doc_v2_0.md` 的范围：

- 4 个同构 rover agent。
- 去中心化 actor observation。
- 中心化 critic state。
- `[rho, beta]` 局部子目标 action。
- 确定性直线轨迹生成器。
- 简化速度跟踪控制。
- 几何集合 reward 和 oracle 距离进展 reward。

