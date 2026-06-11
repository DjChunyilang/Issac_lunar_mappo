# 技术设计说明

第一阶段实现遵循 `isaac_sim_skrl_mappo_multi_rover_tech_doc_v2_0.md` 的范围：

- 4 个同构 rover agent。
- 去中心化 actor observation。
- 中心化 critic state。
- `[rho, beta]` 局部子目标 action。
- 确定性直线轨迹生成器。
- 简化速度跟踪控制。
- 几何集合 reward 和 oracle 距离进展 reward。

## 当前接口约束

- actor observation 不包含 oracle；critic state 和 reward 可以使用训练期 oracle。
- actor observation schema 为 `ego_v2_speed_angular`，详见 [interface_spec.md](interface_spec.md)。
- 通信半径由 `cfg.observation.communication_radius` 驱动；维度相关 observation 字段暂不开放配置覆盖。
- `reward.coefficients.obstacle_collision` 已移除。当前没有 obstacle collision 输入链路，未实现的 reward 配置项不得保留在 base 配置中。
