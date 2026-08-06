# exp136：失败 episode 的重复车辆对冲突触发审计

## 目的

exp135 证明 pair-repeated 冲突比单步冲突具有显著更强的碰撞结果相关性。exp136 进一步核对原始 B2 启用规则中的 episode 级条件：失败 episode 是否普遍包含多个重复车辆对冲突。

该实验只判断 B2 的冲突触发条件，不实现图注意力，也不改变训练、奖励、通信或控制链路。

## 统计定义

对上三角车辆对 \((i,j)\)，当现有 `trajectory_conflicts.repeated` 从 0 变为 1 时，记为一次重复车辆对冲突事件：

\[
e_{ij}(t)
=
\mathbb I\left[
r_{ij}(t)=1\land r_{ij}(t-1)=0
\right].
\]

episode \(k\) 的重复冲突事件数为：

\[
C_k^{\mathrm{repeat}}
=
\sum_t\sum_{i<j}e_{ij}(t).
\]

episode 只在环境返回 `done=true` 时计入；采样尾部尚未结束的 episode 右删失，不进入统计。失败 episode 定义为完成时 `success=false`，并按 `collision`、`out_of_bounds`、`timeout` 记录原因。

## 正式协议

- 冻结 exp125 `relative_quintic` seed23 best；
- 数据种子：`28023`、`29023`；
- 每个种子 128 个环境、512 步；
- 使用原策略动作标准差采样；
- reset 时车辆对重复状态由环境跟踪器清空；
- 事件计数不得跨 episode；
- Actor checkpoint 摘要保持不变。

## 预注册门限

B2 的 episode 级冲突条件仅在以下门限全部满足时成立：

- 每个种子至少获得 100 个完成的失败 episode；
- 每个种子至少 20% 的失败 episode 包含一个或更多重复车辆对冲突事件；
- 将零事件 episode 纳入后，失败 episode 的 \(C_k^{\mathrm{repeat}}\) 中位数不低于 2；
- Actor checkpoint 完全不变。

通过仅表示 B2 的“重复冲突触发条件”成立。B2 仍要求 B0 或 B1 先满足基础收敛条件，因此本实验通过后也不得直接实现或训练图注意力。

## 停止规则

任一门限失败，则 B2 的冲突触发条件不成立，不继续按图注意力方向扩展。不得通过改变事件定义、排除零事件失败 episode 或扩大结果窗口来补偿。

## 正式结果

| 指标 | seed28023 | seed29023 | 门限 |
| --- | ---: | ---: | ---: |
| 完成 episode | 149 | 136 | 描述量 |
| 成功 episode | 8 | 6 | 描述量 |
| 失败 episode | 141 | 130 | \(\ge100\) |
| 含重复冲突的失败 episode | 100% | 100% | \(\ge20\%\) |
| 失败 episode 重复事件均值 | 20.79 | 20.18 | 描述量 |
| 失败 episode 重复事件中位数 | 17 | 18 | \(\ge2\) |
| 失败 episode 10% 分位数 | 7 | 9 | 描述量 |
| 失败 episode 90% 分位数 | 36 | 34.1 | 描述量 |

按失败原因分层：

| 原因 | seed28023 episode/命中率/中位数 | seed29023 episode/命中率/中位数 |
| --- | ---: | ---: |
| collision | `120 / 100% / 16` | `109 / 100% / 18` |
| timeout | `21 / 100% / 20` | `21 / 100% / 21` |

全部预注册门限均通过，Actor checkpoint 摘要保持不变。结果不仅超过 20% 触发下界，而且两个种子的所有失败 episode 均包含重复冲突；即使在分布的 10% 分位数，事件数仍达到 `7/9`。

## B2 启用矩阵

| 条件 | 状态 |
| --- | --- |
| exp135 证明 repeated 指标具有结果相关性 | 通过 |
| 失败 episode 重复冲突触发条件 | 通过 |
| B0 或 B1 基础收敛 | 未通过 |
| 当前允许实现/训练 B2 | 否 |

## 结论

exp136 状态为 `b2_conflict_trigger_met_base_not_converged`。重复冲突是失败 episode 的普遍现象，不是少量碰撞 episode 的偶发诊断量；timeout episode 同样包含大量重复冲突。

但按照当前规划，B2 仍不能启动，因为 B0/B1 基础收敛前置条件未满足。新证据同时暴露了该前置条件可能形成循环：B0 的主要失败现象正是 B2 试图处理的动态邻接冲突。若要解除此前置条件，必须在执行计划中显式登记为一次有证据支撑的规则修订，并保持“只替换 neighbor encoder、单一4M、不得与GRU组合”的范围；不能把本实验直接解释为已经授权 B2。

## 产物

- `outputs/runs/exp136_failed_episode_repeated_conflicts/frozen_exp125_seed23/`
- `metrics/failed_episode_repeated_conflicts.json`
