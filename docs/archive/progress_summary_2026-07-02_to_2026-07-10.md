# 阶段工作总结：2026-07-02 至 2026-07-10

> 本文为历史阶段归档。当前状态与最新实验结论请以 [当前状态](../current_status.md) 和 [实验索引](../experiments/README.md) 为准。文中的图表与动画链接指向本地 `outputs/` 生成目录，不作为仓库内长期保存的静态资源。

## 1. 总体进展

一，新环境栈已经从完全不会集合推进到接近 strict。`exp043/exp044` 直接训练和 initial-state curriculum 后仍然 `success=0`，本阶段通过 local-success bootstrap、terminal hold release、terminal convergence release 和 terminal drive，把新环境栈 success 从 `0.1846` 逐步提高到 `0.9844`，说明复杂环境不是不可学，而是需要先把策略带入 local success basin。

二，当前最好主结果是 `exp051`。它保留 `exp048` 的 reward / filter / control safety 主体，只隔离 PPO 稳定性调整，最终达到 dmax `0.1836`、success `0.9883`、collision `0.0020`，三项达标；唯一失败是 timeout `0.0098`，也就是约 `1%` episode 没有在 320 steps 内稳定通过 success hold。

三，围绕剩余 timeout 做了系统负结果排查。增强 terminal spacing、增强 hold/timeout shaping、提前 entropy taper、调 PPO clip、加入 terminal pairwise reward、提高 gamma、降低 GAE lambda、提高 value loss、给 Actor/Critic 暴露 terminal gate 特征、只给 critic 加 min_pairwise，均没有超过 `exp051`。这说明剩余问题不是简单加惩罚、改 PPO 超参或换 checkpoint 就能解决。

## 2. 工作概览


本阶段新增的总览图如下。可以看到：从 `exp045` 到 `exp048/051`，success 快速恢复；从 `exp048` 以后，大部分实验的 dmax、success、collision 都能接近或达到 strict，但 timeout 始终没有清零。

![exp045-exp062 指标总览](../../outputs/runs/_comparisons/exp045_exp062_20260710/figures/exp045_exp062_metrics_overview.png)

## 3. 阶段性工作进展

### 一、快速迭代拉回成功率

遇到的问题：

上一阶段新环境栈已经跑通，但直接 40M 长训和 initial-state curriculum 都没有产生成功 episode。结构化网络、bicycle proxy、quintic 轨迹、25m 地图和更复杂 reset 一起上来后，策略很难从零学到“稳定集合”。

解决方法：

- `exp045` 缩小 reset 难度，做 local-success bootstrap，让 rover 先在更容易成功的初始分布里学会局部集合。
- `exp046` 降低末段 filter / control safety 的介入强度，避免还没进入成功区就被过度保守约束卡住。
- `exp047` 继续释放 terminal convergence，让策略更主动地完成最后收缩。
- `exp048` 加强 terminal drive 和 dispersion 收缩，把策略从“能靠近”推到“基本能成功”。

阶段结果：

| 实验 | 主要目的 | dmax | success | collision | timeout | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| exp045 | local-success bootstrap | `0.2734` | `0.1846` | `0.0000` | `0.8174` | 首次恢复局部 success 信号。 |
| exp046 | terminal hold release | `0.2424` | `0.6123` | `0.0000` | `0.3877` | 已进入 local success basin。 |
| exp047 | terminal convergence release | `0.2132` | `0.7188` | `0.0059` | `0.2764` | 收敛继续改善。 |
| exp048 | terminal drive / dispersion tightening | `0.1866` | `0.9844` | `0.0020` | `0.0137` | dmax / success / collision 达标，只剩 timeout。 |

图上能直接看到这个恢复过程：success 从 `0.1846` 逐步爬到 `0.9844`，timeout 从 `0.8174` 降到 `0.0137`。

![exp045-exp051 success/timeout 恢复过程](../../outputs/runs/_comparisons/exp045_exp062_20260710/figures/exp045_exp051_recovery_success_timeout.png)

`exp048` 是第一个让新环境栈接近 strict 的关键节点：

![exp048 训练曲线](../../outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_drive/figures/training_curves.png)

![exp048 候选 checkpoint 曲线](../../outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_drive/figures/candidate_eval_curves.png)

![exp048 随机地形高度图](../../outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_drive/figures/terrain_height_map.png)

![exp048 proxy rollout](../../outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_drive/videos/proxy_eval_rollout.gif)


可以看到快速迭代后问题转换为末段 timeout / hold 问题。

### 二、尝试修复末端问题

遇到的问题：

`exp048` 的剩余失败集中在末段，理论上可以加强 terminal spacing、hold reward 或 timeout penalty

解决方法：

- `exp049` 针对 terminal spacing 做增强，希望把最后最近邻间距灰区拉开。
- `exp050` 回到 `exp048` 主体，改为更克制地增强 hold / timeout shaping，并配合 PPO 学习率和 clip 微调。

阶段结果：

| 实验 | 修改方向 | dmax | success | collision | timeout | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| exp048 | terminal drive baseline | `0.1866` | `0.9844` | `0.0020` | `0.0137` | 当前节点最好。 |
| exp049 | terminal spacing 更强 | `0.1884` | `0.8926` | `0.0010` | `0.1064` | 过强间距修正降低 success，timeout 变差。 |
| exp050 | hold / timeout shaping | `0.1847` | `0.9590` | `0.0059` | `0.0352` | 比 exp049 好，但仍差于 exp048。 |

![exp038/exp048/exp049 候选评估对比](../../outputs/runs/_comparisons/exp038_exp048_exp049_20260707/figures/candidate_eval_comparison.png)

![exp048/exp049/exp050 对比](../../outputs/runs/exp050_structured_bicycle_quintic_map25_terminal_hold_rl_tune/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_hold_rl_tune/figures/exp048_exp049_exp050_candidate_comparison.png)

新的问题：

“末段间距不够”确实是问题，但不能直接用全局更强的 spacing/filter 或 hold shaping 解决。强行拉开距离会破坏 success hold，导致 timeout 反而升高。

### 三、回到 exp048 主体，只隔离 PPO 稳定性，得到当前最好结果 exp051

遇到的问题：

`exp049/050` 说明直接改 reward/filter/control 容易扰动已学到的策略。因此下一步要尽量少动任务定义，只看 PPO 训练稳定性是否能减少尾部 timeout。

解决方法：

`exp051` 回到 `exp048` 的 reward、filter、control safety 主体，只做 PPO 稳定性调整，包括学习率、clip、entropy schedule 和 initial log std。目标是保持已有高成功率和低碰撞率，同时减少训练不稳定带来的尾部失败。

阶段结果：

| 指标 | strict 要求 | exp051 结果 | 是否达标 |
| --- | ---: | ---: | --- |
| dmax reduction ratio | `<= 0.20` | `0.1836` | 达标 |
| success rate | `>= 0.90` | `0.9883` | 达标 |
| collision rate | `<= 0.02` | `0.0020` | 达标 |
| timeout rate | `= 0` | `0.0098` | 未达标 |

`exp051` 是当前新环境栈最好候选。它没有 strict pass，但已经把问题压缩到约 `1%` timeout。

![exp051 训练曲线](../../outputs/runs/exp051_structured_bicycle_quintic_map25_ppo_stability/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_stability/figures/training_curves.png)

![exp051 候选 checkpoint 曲线](../../outputs/runs/exp051_structured_bicycle_quintic_map25_ppo_stability/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_stability/figures/candidate_eval_curves.png)

![exp048/exp050/exp051 对比](../../outputs/runs/exp051_structured_bicycle_quintic_map25_ppo_stability/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_stability/figures/exp048_exp050_exp051_candidate_comparison.png)

### 四、 PPO、reward 和 value horizon 消融

做的工作：

- `exp052` 提前 entropy taper，测试是否更早收敛能减少 timeout。
- `exp053` 小幅提高 near reward，测试更强安全间距是否有用。
- `exp054/exp055` 做 PPO clip 扫描。
- `exp056/exp057` 加 terminal pairwise reward，测试只在末段惩罚最近邻 gap 是否有效。
- `exp058/exp059/exp060` 分别检查 gamma、GAE lambda 和 value loss 权重。

主要结果：

| 实验 | 改动方向 | dmax | success | collision | timeout | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| exp051 | 当前基线 | `0.1836` | `0.9883` | `0.0020` | `0.0098` | 当前最好。 |
| exp052 | entropy 更早退火 | `0.1863` | `0.8955` | `0.0059` | `0.0986` | 过早收窄探索会退化。 |
| exp053 | near reward 小幅增强 | `0.2049` | `0.6416` | `0.0039` | `0.3545` | 全局间距惩罚会推散队形。 |
| exp054 | PPO clip `0.16` | `0.1972` | `0.7168` | `0.0029` | `0.2803` | clip 过窄，更新受限。 |
| exp055 | PPO clip `0.20` | `0.1850` | `0.9824` | `0.0029` | `0.0146` | 能恢复高成功率，但不优于 exp051。 |
| exp056 | terminal pairwise reward | `0.1864` | `0.9873` | `0.0010` | `0.0117` | 接近但仍差于 exp051。 |
| exp057 | stricter terminal pairwise | `0.1850` | `0.9697` | `0.0059` | `0.0254` | 触发更严格反而扰动 hold。 |
| exp058 | gamma `0.995` | `0.1991` | `0.7451` | `0.0020` | `0.2529` | 更长 horizon 拖慢末段收敛。 |
| exp059 | GAE `0.90` | `0.1927` | `0.6904` | `0.0127` | `0.2988` | 更短 trace 破坏 terminal convergence。 |
| exp060 | value loss `0.75` | `0.1837` | `0.9736` | `0.0000` | `0.0264` | critic loss 更强也没有清 timeout。 |

几组对比图能直接看出：这些微调有的能保持 dmax/collision，但都没能把 timeout 清零，也没有超过 `exp051`。

![PPO clip sweep 对比](../../outputs/runs/exp055_structured_bicycle_quintic_map25_ppo_clip20/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_clip20/figures/exp051_exp054_exp055_clip_sweep_comparison.png)

![terminal pairwise reward 对比](../../outputs/runs/exp057_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward/figures/exp051_exp056_exp057_terminal_pairwise_comparison.png)

![value horizon 对比](../../outputs/runs/exp059_structured_bicycle_quintic_map25_gae090/pure_rl_seed23_40m_structured_bicycle_quintic_map25_gae090/figures/exp051_exp058_exp059_value_horizon_comparison.png)

![value learning 对比](../../outputs/runs/exp060_structured_bicycle_quintic_map25_value075/pure_rl_seed23_40m_structured_bicycle_quintic_map25_value075/figures/exp051_exp058_exp059_exp060_value_learning_comparison.png)


通过以上实验，我认为剩余 timeout 不是简单的“探索太多/太少”“clip 不合适”“critic 不够强”或“末段 reward 不够强”。真正瓶颈需要从 success gate 本身和末段几何关系里找。

### 五、做 checkpoint seed sweep 和 success gate 诊断，定位 timeout 本质

遇到的问题：

`exp051` timeout 只有 `0.0098`，可能只是 checkpoint 选择或 eval seed 的偶然波动。因此需要验证：附近 checkpoint 是否有更稳的点，不同 eval seed 下 timeout 是否可能为 0？timeout episode 到底失败在哪个 gate？

解决方法：

- 对 `exp051` 附近 `012288 / 013312 / 014336` 三个 checkpoint 做 `4` 个 eval seed 复验。
- 对 `exp062` 附近 `015360 / 016384 / 017408` 三个 checkpoint 也做 `4` seed 复验。
- 新增 success gate diagnostics，按 dmax、dispersion、speed、min_pairwise 分解 timeout 失败原因。
- `exp061` 直接把 terminal gate 特征加入 Actor/Critic，测试显式观测是否能解决末段保持。
- `exp062` 不改 Actor，只给 critic state 加 min_pairwise，测试 critic 可观测性是否能帮助 value 学到末段间距。

复验结果：

| 实验 | checkpoint | 4-seed timeout mean | strict pass count | timeout zero count | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| exp051 | `012288` | `0.0205` | `0/4` | `0/4` | 不如当前 best。 |
| exp051 | `013312` | `0.0134` | `0/4` | `0/4` | 附近最稳，但仍不 strict。 |
| exp051 | `014336` | `0.0295` | `0/4` | `0/4` | 更差。 |
| exp062 | `015360` | `0.0217` | `0/4` | `0/4` | 不优于 exp051。 |
| exp062 | `016384` | `0.0161` | `0/4` | `0/4` | exp062 最好，但仍不如 exp051 sweep。 |
| exp062 | `017408` | `0.0195` | `0/4` | `0/4` | 不优于 exp051。 |

![exp051/exp062 checkpoint seed sweep](../../outputs/runs/_comparisons/exp045_exp062_20260710/figures/exp051_exp062_checkpoint_sweep_timeout.png)

success gate 诊断显示，timeout 的主要失败项是最近邻安全间距 `min_pairwise`，不是速度，也不是整体 dmax / dispersion。

| 诊断 | timeout 数量 | dmax 失败 | dispersion 失败 | speed 失败 | min_pairwise 失败 |
| --- | ---: | ---: | ---: | ---: | ---: |
| exp051 | `15` | `2/15` | `0/15` | `0/15` | `15/15` |
| exp060 | `19` | `2/19` | `1/19` | `0/19` | `18/19` |
| exp062 | `13` | `5/13` | `1/13` | `0/13` | `8/13` |

![timeout gate 诊断](../../outputs/runs/_comparisons/exp045_exp062_20260710/figures/success_gate_timeout_failure_rates.png)

`exp061/exp062` 结果：

| 实验 | 改动方向 | dmax | success | collision | timeout | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| exp061 | terminal gate 特征进入 Actor/Critic | `0.1890` | `0.8506` | `0.0205` | `0.1289` | Actor 变激进，整体退化。 |
| exp062 | 只给 critic 加 min_pairwise state | `0.1832` | `0.9736` | `0.0059` | `0.0205` | 保持三项达标，但 timeout 不优于 exp051。 |

新的问题：

现在可以明确：当前最好策略基本会集合，也足够低速，但极少数 episode 最后停在成功间距门槛附近。它们通常不是没到集合区域，而是最近邻距离没有稳定超过 `0.42 m`，导致 success hold 无法累计到 `8` 步。

## 4. 当前最好结果

综上所述，当前最适合汇报的新环境栈结果是 `exp051`：

| 指标 | strict 要求 | exp051 | 是否达标 |
| --- | ---: | ---: | --- |
| dmax reduction ratio | `<= 0.20` | `0.1836` | 达标 |
| success rate | `>= 0.90` | `0.9883` | 达标 |
| collision rate | `<= 0.02` | `0.0020` | 达标 |
| timeout rate | `= 0` | `0.0098` | 未达标 |

总的来说：

> 新环境栈已经从完全不成功恢复到高成功率，当前最好结果 dmax、success、collision 都已经过 strict，剩余问题非常集中：约 `1%` episode 在末段最近邻间距门槛附近没有稳定 hold 到 `8` 步，因此 timeout 还没有清零。

## 5. 总结与规划

一，最后 `1%` timeout 不能靠简单加强安全/间距惩罚解决。`exp049/053/056/057` 都说明，过强的 spacing 或 pairwise reward 会扰动已经学到的集合行为。

二，PPO 超参不是当前主瓶颈。entropy、clip、gamma、GAE、value loss 的多组消融都没有超过 `exp051`。

三，直接把 terminal gate 特征给 Actor 会退化。`exp061` 说明显式 gate margin 可能诱发更激进或更饱和的动作；`exp062` 的 critic-only 可观测性也没有解决 timeout。

四，下一步不再继续迭代追求timeout清零，转向继续推进完善仿真训练环境的配置，包括地图尺寸、车等的各类参数的进一步确认，调研和尝试增加集合位置地形的平坦度判断
