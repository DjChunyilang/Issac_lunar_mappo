# 路线图

## 立即处理

1. 将 exp008 保持为当前已验证的 3-seed terrain-aware proxy 结果。
2. 将 exp009 视为强地形诊断实验，而不是严格成功结果。
3. 诊断 exp009 seed31 失败 episode。
4. 在继续长训练前，先原型化成功区附近的 control/reward 改动。

## 近期工作

- 新增成功 hold 失败原因指标：`dmax_ok`、`dispersion_ok`、`speed_ok` 和 hold count 分布。
- 为失败 episode 增加定点 rollout debug 图。
- 长预算训练前，先用 seed31 短 run 对比 reward/control 变体。
- PhysX 继续作为验证和展示层，不进入主训练 loop。

## 中长期工作

- 如果成功区稳定性仍然脆弱，重新审视动作表示。
- 如果强地形直接训练仍不稳定，加入更严格的地形强度 curriculum。
- 构建可重复的报告生成器，从 `_suite/metrics/*.json` 自动更新实验文档。

