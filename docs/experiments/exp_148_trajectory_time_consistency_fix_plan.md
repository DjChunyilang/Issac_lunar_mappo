# exp148：quintic轨迹时间一致性修正与B0重新筛选

## 决策依据

exp147在两个完整96秒验证种子上确认了系统性执行契约失配：

- `78.25%–79.45%` 的非零路径按0.2 s时间戳计算会超过1.35 m/s最大线速度；
- 按1.15 m/s参考速度所需时域与当前声明时域之比中位数为 `2.51–2.57`；
- 实际一步位移只占规划弧长的 `1.77%`；
- 控制器固定使用的第2个轨迹点只覆盖规划弧长的 `4.83%`；
- 地形奖励和连续冲突诊断仍评价整条路径。

因此，当前B0训练结果不能继续用于判断quintic局部路径学习能力。exp148选择修正轨迹的物理时间语义，不截断地形奖励。截断到第一个跟踪段会把可评价路径缩短到约5%，无法支持地形相关绕行，故不采用该方案。

## 唯一工程改动

执行链路保持不变：

```text
局部观测与分级通信
→ 共享Actor
→ 局部子目标
→ quintic轨迹
→ bicycle控制器
```

只修改现有 `trajectory_generator.py`、`simple_controller.py` 和非干预式冲突诊断，不新增网络或在线修正模块。

### 1. 按弧长生成轨迹时域

对每辆车的quintic采样点计算弧长：

\[
L_i
=
\sum_{k=0}^{K-2}
\left\|\mathbf p_{i,k+1}-\mathbf p_{i,k}\right\|_2.
\]

轨迹预测时域定义为：

\[
H_i
=
\max\left(
\frac{L_i}{v_{\mathrm{ref}}},
\Delta t
\right),
\]

其中 $v_{\mathrm{ref}}=1.15\,\mathrm{m/s}$，
$\Delta t=0.2\,\mathrm{s}$。第 $k$ 个采样点的时间戳为：

\[
t_{i,k}
=
\frac{k}{K-1}H_i.
\]

零长度路径的时域固定为一个规划周期，避免除零和零时域。

### 2. 按物理前视时间选择跟踪点

控制器不再固定选择索引1，而在每条轨迹上插值获取 $t=\Delta t$ 的位置和航向。若路径时域小于或等于 $\Delta t$，直接使用终点。控制律、`k_linear=3.2`、`k_angular=3.2`、速度上限、转角上限和bicycle积分均保持不变。

该修改只使控制器跟踪“0.2秒后应到达的位置”，不增加轨迹跟踪器、MPC或安全投影。

### 3. 连续冲突按真实时间对齐

不同车辆的 $H_i$ 可能不同。冲突诊断先在每个环境内建立从0到该环境最大预测时域的公共时间网格，再分别对每辆车的轨迹作分段线性插值；超出单车时域后保持在终点。随后才计算同一时间戳的车辆间距离。

该结果仍只写诊断日志，不修改Actor动作、局部子目标或控制命令。

## 工程门限

实现后必须通过：

- line和quintic路径几何点在相同输入下与修正前完全一致；
- 非零路径的终点时间戳等于 $\max(L/v_{\mathrm{ref}},\Delta t)$；
- 时间戳单调、有限，零长度路径无NaN；
- 插值跟踪点对应 $t=0.2\,\mathrm s$，且位于轨迹起终点之间；
- 公共时间网格上的冲突计算能区分同位置异时通过与同时间相交；
- 地形风险对固定轨迹点的数值完全不变；
- Actor观测仍为101维，Actor参数摘要和固定探针动作变化均为0；
- 安全投影、后处理、槽位目标和BC保持关闭；
- CPU小环境与CUDA 256环境smoke无NaN，控制量不越界。

冻结动作复核还必须满足：

- 时间戳隐含速度超过1.35 m/s的比例不高于5%；
- $H_i^{\mathrm{ref}}/H_i$ 中位数位于 `[0.99, 1.01]`；
- 控制器不再固定依赖轨迹点索引1；
- Actor和通信缓存不读取新增的集中式信息。

任一工程门限失败则停止，不启动训练，也不通过调整 `rho_max`、参考速度、轨迹点数或控制增益补偿。

## B0重新筛选

工程门限全部通过后，旧checkpoint只保留为历史对照，新策略必须随机初始化。只允许重跑一次原B0 4M筛选：

```yaml
episode_duration: 96 s
episode_steps: 480
parallel_envs: 2048
rollout_length: 64
algorithm: shared-joint MAPPO
bc_updates: 0
init_checkpoint: null
seed: 23
```

通信、101维Actor、奖励系数、初始状态课程和严格去中心化边界保持exp125不变。唯一实验变量是轨迹时间一致性修正。

4M门限仍使用原B0标准：无数值异常、网络分支有效更新、动作标准差大于 $10^{-4}$、末四分之一dmax相对首四分之一降低至少30%、出现非零success、collision不超过10%、terrain-contrast动作MSE大于0.02、正常地形下实际执行路径风险至少降低5%。

未通过则停止，不继续修改控制器或奖励。通过也只允许重新制定40M计划，不直接启动40M。

## 明确不做

- 不改变Actor结构、通信内容或101维观测；
- 不增加GRU、GNN、注意力或学习消息；
- 不恢复BC、PPO-Lagrangian、逐车信用或成对安全Critic；
- 不增加安全投影、方向mask、槽位修正或末段覆盖；
- 不调整奖励权重、`rho_max`、参考速度、轨迹点数或控制增益；
- 不把冲突诊断变成奖励或硬约束；
- 不迁移旧checkpoint。

## 当前状态

时间一致性修正、CPU/CUDA smoke、冻结动作双种子工程复核及一次随机初始化 B0 4M 筛选均已完成。工程门限全部通过，但 4M 收敛门限未通过；状态为 `screen_failed`，不启动 40M。

## 实施结果

新增配置为 `configs/experiment/exp148_decentralized_b0_trajectory_time_consistent.yaml`。历史配置继续采用原时间语义，exp148 单独启用：

```yaml
trajectory_generator.time_parameterization: arc_length_reference_speed
low_level_control.tracking_point_mode: planning_time
```

工程复核得到：

- CPU、CUDA 256 环境及冻结策略双种子审计均无 NaN 或控制越界；
- 时间戳隐含速度超过 $1.35\,\mathrm{m/s}$ 的比例由 exp147 的 `78.25%–79.45%` 降为 `0`；
- $H_i^{\mathrm{ref}}/H_i$ 中位数为 `1.0000001`；
- Actor 参数摘要与固定探针动作完全不变；
- 完整 96 秒冻结复核中，单步实际弧长利用率中位数由约 `1.77%` 提高到 `11.82%–12.05%`。

工程门限记录为 `outputs/runs/exp148_trajectory_time_consistency_fix/_suite/metrics/engineering_gate.json`。

## 4M 筛选结果

seed23 使用 2048 个并行环境、2048 个训练时步，共 `4,194,304` 次环境交互。训练过程有限且各网络分支确实更新：

- 首四分之一到末四分之一的平均 dmax 从 `7.0379` 降至 `3.2090`，降幅 `54.40%`；
- Actor、neighbor encoder、terrain encoder 参数变化范数分别为 `2.0889/0.8290/0.3220`；
- 训练后动作标准差为 `0.4462`；
- 训练期间 success episode 数为 `0`。

冻结近距评测结果为：

| dmax ratio | success | collision | timeout | 最终实际质心平整率 |
| ---: | ---: | ---: | ---: | ---: |
| `0.3528` | `0` | `0.9990` | `0.0010` | `0.0869` |

terrain contrast 中，正常地形与地形置零的动作 MSE 为 `0.00383`，低于 `0.02` 门限；正常地形路径风险为 `0.34139`，地形置零为 `0.33999`，相对变化为 `-0.414%`，没有达到风险降低 5% 的要求。近距评测的完整消息比例为 `1.0`、平均消息年龄为 `0`，因此本次失败不能归因于远距通信。

十项 B0 筛选中六项通过，以下四项失败：

- 未出现非零 success episode；
- collision 高于 10%；
- terrain contrast 动作 MSE 未超过 0.02；
- 正常地形路径风险未降低 5%。

正式门限记录为 `outputs/runs/exp148_trajectory_time_consistency_fix/_suite/metrics/b0_screen_seed23_4m_time_consistent_screen_summary.json`。

## MAPF 失败归因

对 best checkpoint 追加两个独立地形种子、每种子 128 环境乘 512 步的冻结失败 episode 诊断。两个种子分别完成 `405/388` 个 episode，success 均为 `0`，全部因碰撞结束；失败 episode 的重复车辆对冲突命中率均为 `100%`，事件数中位数均为 `5`，10%–90% 分位区间均为 `2–17`。

这说明修正后的策略已经能够更充分地执行集合方向动作，但学习结果表现为持续靠近并反复发生车辆对冲突，而不是安全集合。B2 的冲突触发条件虽然成立，B0/B1 基础收敛条件仍不成立；结合 exp137 已失败的单变量图注意力实验，本结果不授权 B2、GRU、可学习消息或任何在线安全修正。

诊断记录为 `outputs/runs/exp148_trajectory_time_consistency_fix/post_fix_failed_episode_conflicts_dualseed/metrics/failed_episode_repeated_conflicts.json`。

## 最终决策

exp148 证明轨迹执行契约修正是必要且正确的工程改动，但它没有使现有 B0 训练目标收敛。按照预注册停止规则：

- 保留时间一致性修正作为新实验的物理执行基线；
- 不启动 B0 40M，不调整控制增益、参考速度、轨迹点数或动作尺度；
- 不增加 B1/B2/B3、PPO-Lagrangian、安全投影或后处理；
- 下一步只允许基于现有日志和冻结 checkpoint 分析碰撞终止前的奖励、动作与重复冲突时序，形成单一训练信用假设后再另行预注册。
