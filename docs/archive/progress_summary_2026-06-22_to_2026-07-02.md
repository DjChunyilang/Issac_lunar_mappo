# 进度总结 2026-07-02

> 归档日志：这是 2026-06-22 至 2026-07-02 阶段总结。当前状态、最新结果和下一步计划请优先阅读 `docs/current_status.md`、`docs/experiments/README.md` 和对应实验文档。

## 1. 总体进展


第一，把训练链路从固定地形推进到了随机月面地形，补齐了局部地形观测、共享 MAPPO 更新、独立评估和可视化。固定地图已经能 strict 通过，说明基本训练链路是成立的。

第二，围绕随机地形下的安全集合做了系统消融。现在已经明确：过硬的安全过滤会导致 rover 不敢集合，过软又会碰撞；引入 mutual path safety 和低层 control safety 后，成功率和碰撞率可以同时达标。当前最好结果成功率约 `97.6%`、碰撞率约 `1.37%`，只剩 `1%` 左右 episode 超时。

第三，开始推进更高难度的新环境栈，包括结构化网络、bicycle proxy、quintic 轨迹和更大地图。工程链路已经跑通，两轮 40M 训练也完成了，但新环境栈从零训练还没有恢复成功集合。下一步我会先把随机地形 strict pass 收尾，同时拆解新环境栈里到底是哪一项增加了学习难度。

## 2. 阶段性工作进展


### 一、实验配置的修改

- 补充局部地形观测，让 rover 不只知道邻居在哪里，也能看到周围地形是否难走。
- 修通 MAPPO 训练、checkpoint、评估和可视化流程，让结果可以稳定复现和横向比较。
- 先用固定地图做 baseline，验证基本训练链路确实能让 rover 学会集合。



### 二、把任务从固定地图推进到随机月面地形

加入随机坑、坡度和粗糙度后，固定地图上有效的策略不再稳定。rover 有时会走向风险更高的地形，或者在靠拢过程中发生碰撞；只看单次训练曲线也不足以说明方法可靠。

- 加入随机月面地形、地形通行风险和速度影响，让训练任务更接近真实复杂地表。
- 新增评估脚本，建立独立评估和多 seed 诊断，同时看成功率、碰撞率、超时率和距离收缩。

rover 已经能形成明显集合趋势，随机地形 baseline 的 `success` 一度达到 `96%` 以上。但问题从“会不会集合”变成了“能不能又快、又安全、又稳定地集合”：降低碰撞时容易超时，追求快速集合时又容易碰撞。

### 三、排查安全过滤与集合进度的矛盾


上一阶段的核心矛盾是安全和集合进度互相冲突。安全约束太硬，rover 会变得过于保守；安全约束太软，rover 会继续集合，但碰撞率会上升。



- 在策略输出子目标之后加入 filter，如果原始子目标危险，就在一批候选子目标里换一个更安全、同时还能继续靠近集合目标的点。
- 增加路径采样的filter，不只检查子目标终点是否危险，也检查从当前位置走过去的整段路径是否经过高风险地形。
- filter增加 mutual path safety 项，引入对邻居的运动的预测，而不是简单把邻居当成静止障碍物。
- 加密路径采样，更细地比较候选子目标的安全性、地形风险和集合进度。


filter 能明显改善碰撞问题，也证明 mutual path safety 是有效方向；但只靠子目标过滤还不够。它能让 rover 更安全地靠近成功区域，却不能完全解决最后一段“已经接近成功，但停不稳、保持不住、最终超时”的问题。

### 四、处理“最后差一点”的末段稳定问题



训练已经能把大多数 episode 带到成功区域附近，但最后一小段容易反复震荡。任务要求 rover 不只是短暂靠近，还要在安全距离内稳定保持一段时间；一旦末段停不稳，就会变成 timeout，甚至在最后阶段发生碰撞。



- 增加 success hold reward，鼓励 rover 进入成功区域后稳定保持，而不是靠近后又散开。
- 增加 timeout penalty。
- 在低层控制输出后增加 safety projection，也就是在真正执行速度和方向之前再做一次安全修正，减少即将发生的相对运动冲突。
- 增加 success-zone stabilizer，专门处理接近成功后的稳定保持问题。

阶段结果：

| 关键结果 | 数值 | 备注 |
| --- | ---: | --- |
| 首次 success/collision 同时达标 | success `0.9072`，collision `0.0127` | 说明安全和集合不是必然冲突。 |
| 进一步提升后 | success `0.9336`，collision `0.0088` | timeout 仍是主要瓶颈。 |
| 当前最佳长训 | success `0.9756`，collision `0.0137`，timeout `0.0107` | 随机地形已经接近 strict，只剩少量超时。 |
| 最佳复评诊断 | success `0.9795`，collision `0.0107`，timeout `0.0098` | 比当前最佳长训略好，但还不是独立长训结果。 |

当前随机地形路线已经接近可汇报成果，成功率和碰撞率都能达标，主要剩余问题是约 `1%` 的 episode 还会卡在末段保持条件上。下一步不能再只追求更强的安全约束，而是要把最后这部分 timeout 清掉，同时不把碰撞率重新抬高。

## 4. 当前最好结果

### 固定地图结果

固定地图 pure RL 单 seed 已经通过 strict gate：

| 指标 | 结果 |
| --- | ---: |
| 最大距离收缩比例 | `0.1318` |
| 成功率 | `0.9990` |
| 碰撞率 | `0.0010` |
| 超时率 | `0.0000` |


### 随机地形当前最佳结果

`exp038`

| 指标 | strict 要求 | 当前结果 | 是否达标 |
| --- | ---: | ---: | --- |
| 最大距离收缩比例 | `<= 0.20` | `0.1590` | 达标 |
| 成功率 | `>= 0.90` | `0.9756` | 达标 |
| 碰撞率 | `<= 0.02` | `0.0137` | 达标 |
| 超时率 | `= 0` | `0.0107` | 未达标 |

随机地形下已经达到高成功率和低碰撞率，距离 strict pass 只剩少量超时失败。

展示图：

![随机地形训练曲线](outputs/runs/exp038_randomized_terrain_success_zone_stabilizer/pure_rl_seed23_20m_success_zone_stabilizer_timeout320/figures/training_curves.png)

![随机地形收敛曲线](outputs/runs/exp038_randomized_terrain_success_zone_stabilizer/pure_rl_seed23_20m_success_zone_stabilizer_timeout320/figures/convergence_curves.png)

![随机地形安全诊断](outputs/runs/exp038_randomized_terrain_success_zone_stabilizer/pure_rl_seed23_20m_success_zone_stabilizer_timeout320/figures/safety_diagnostics.png)

![随机地形 rollout](outputs/runs/exp038_randomized_terrain_success_zone_stabilizer/pure_rl_seed23_20m_success_zone_stabilizer_timeout320/videos/proxy_eval_rollout.gif)

## 5. 新环境栈进展

在随机地形接近收敛后，开始提高环境和控制模型复杂度。这一部分不是为了替代当前最好结果，而是为后续把问题做得更接近真实 rover 运动提前铺路。

遇到的问题：

前面的随机地形路线已经能说明训练方法有效，但环境和控制模型仍然偏简化。继续往下做，必须逐步增加模型复杂度，否则后续很难说明方法能适应更复杂的 rover 运动和更大地图。

- Actor/Critic 从普通 MLP 扩展为结构化网络，把自车、邻居、地形和聚合信息分支编码。
- Proxy 模型从 unicycle 扩展到 bicycle。
- 局部参考轨迹从直线段扩展到 quintic 曲线。
- 地图扩大到 `25 m x 25 m`。

目前进展：

- 训练闭环已验证无问题。
- 两组 40M env-step 长训已经完成。

训练结果：

| 实验 | 目的 | 结果 |
| --- | --- | --- |
| 直接新环境栈长训 | 检查复杂环境能否直接从零学会集合 | 失败，success `0`，timeout `1.0`。 |
| initial-state curriculum | 降低冷启动难度，逐步恢复目标初始分布 | 有进步，`dmax_ratio` 从 `0.8596` 改善到 `0.4796`，但 success 仍为 `0`。 |


下一步需要继续诊断，判断到底是训练量、结构化网络、bicycle proxy、quintic 轨迹，还是大地图导致冷启动变难。
