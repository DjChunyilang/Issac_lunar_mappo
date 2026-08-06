# 严格去中心化收敛训练、通信架构与 MAPF 诊断研究计划

本文是当前唯一执行计划。当前事实见 [current_status.md](current_status.md)，总体架构见 [architecture/overall_plan_v3.md](architecture/overall_plan_v3.md)，实验结果见 [experiments/README.md](experiments/README.md)。

## 当前实施状态

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| B0 分级通信与 101 维 Actor | 工程实现完成 | 主配置为 `configs/experiment/exp125_decentralized_tiered_b0_pure_rl.yaml` |
| 通信、兼容性和执行不变性测试 | 工程实现完成 | 覆盖 12 m 边界、缓存隔离和旧 checkpoint 拒绝 |
| 连续轨迹冲突诊断 | 工程实现完成 | 仅写日志，不修改动作或控制 |
| 地形拓扑分类与 vanilla CBS | 基础工具完成 | 只用于离线诊断和集中式上界 |
| B0 4M screen | 已完成，未通过 | seed23 六组同族诊断均失败；按停止规则不启动 40M |
| C0 零和地形信用 4M 对照 | 已完成，未通过 | 地形风险方向改善但 collision `0.6484`；停止且不扫参数 |
| exp127 联合动作 Critic 诊断 | 已完成，未通过 | 完整 episode 的 16 步 MSE 仅改善 `2.16%`；C1 停止 |
| exp128 奖励分量辨识诊断 | 已完成，未通过 | gather 可辨识，safety/terrain 不可辨识；不改变训练信用 |
| exp129 配对动作干预 | 已完成 | 地形与安全方向整体正交；不启动固定权重训练 |
| exp130 Actor 梯度冲突审计 | 已完成，通过诊断门限 | 两个种子负余弦比例均为 `46.875%`；允许一次 C2 筛选 |
| exp131 C2 主任务优先投影 | 已完成，未通过 | 投影有效但 collision `0.6680`；停止且不扫参数 |
| exp132 逐车安全势函数审计 | 已完成，未通过 | 仅 75% rollout 有事件，总体 trace std `0.8660`；不训练 C3 |
| exp133 连续近距逐车信用审计 | 已完成，未通过 | 密度约 17%，但首次激活仍晚；不训练 C3-near |
| exp134 近距信用冲突提前量 | 已完成，未通过 | 真实碰撞提前量充分，但预测冲突覆盖率仅约 36%–38% |
| exp135 重复冲突结果相关性 | 已完成，通过诊断门限 | repeated 结果率约 5%，8 步碰撞召回 100% |
| exp136 失败 episode 重复冲突 | 已完成，通过触发门限 | 100% 失败 episode 命中，中位事件数 `17/18` |
| exp137 B2 单跳图注意力例外 | 已完成，未通过 | dmax改善但collision `0.7881`，重复冲突未一致下降；在基础gate停止 |
| exp138 安全聚合辨识诊断 | 已完成，未通过 | worst-pair动作增益接近0，均值稀释不是瓶颈；不启动4M |
| exp139 逐车局部安全信用辨识 | 已完成，未通过 | raw跨种子不稳定，零和中心化动作增益仅 `8.22%/7.20%`；不重开C3-near |
| exp140 非零和逐车近距信用 | 已完成，未通过 | collision `0.2295`，重复冲突升至 `19/20`；停止且不扫参数 |
| exp141 collision cost-value审计 | 已完成，通过前置门限 | AUROC `0.939/0.933`、Brier改善 `35.6%/30.0%`；允许制定Lagrangian计划 |
| exp142 collision PPO-Lagrangian | 已完成，未通过 | 碰撞显著下降但success为0、timeout `0.9746`；以回避集合换取安全，停止且不扫参数 |
| exp143 B0时域/约束竞争审计 | 已完成，未通过 | B0后段仍改善，但约束后success严格零增长条件未满足；不直接启动12M |
| exp144 B0多种子checkpoint趋势 | 已完成，未通过 | 5/5种子几何改善，但3/3 terrain contrast路径风险恶化；停止深度训练假设 |
| exp145 统一逐车任务回报审计 | 已完成，未通过 | 集合/地形可辨识，安全动作增益仅`9.58%`、最差`8.12%`；不训练 |
| exp146 最近邻成对安全耦合 | 已完成，未通过 | 条件动作增益未达15%；不实现成对安全Critic |
| exp147 轨迹执行契约审计 | 已完成，确认失配 | 约79%路径时间戳速度超限，实际每步仅执行规划弧长约1.77% |
| exp148 轨迹时间一致性修正 | 工程通过，4M未通过 | 速度违例降为0、弧长利用率约12%；4M success `0`、collision `0.9990`，停止在40M前 |
| exp149 碰撞参与者信用可行性 | 已完成，通过诊断门限 | 典型碰撞只涉及2辆车，终止前8/16步召回约100%；只授权一次exp150筛选 |
| exp150 真实碰撞参与者Actor信用 | 工程通过，4M未通过 | success `0`、collision `0.9990`、重复冲突`9/8`；停止该信用方向，不启动40M |
| exp151 碰撞信用因果有效性 | 已完成，未通过 | 8/16步局部可行动率最小`0.5100/0.5616`；停止终止信用，只允许冻结可控性分解 |
| exp152 动作—规划—控制可控性分解 | 已完成，确认瓶颈 | 8步无约束可行动率最小`0.6648`，控制传递率最低`0.8069`；瓶颈位于动作—quintic层 |
| exp153 动作范围与quintic几何分离 | 已完成，混合结果 | endpoint恒可达，网格quintic最小`0.7536`；动作范围或quintic均非单一瓶颈，只允许双车联合干预 |
| B1、B3及B2后续扩展 | 未启用 | B1无消息年龄依据；B2已否决；B3缺少B2前置条件 |

## 1. 研究目标与总体原则

本轮只深入解决三个问题：

1. 建立符合执行期严格去中心化要求的分级通信。
2. 让 Pure RL 直接学习地形相关路径与安全集合。
3. 使用 MAPF 拓扑和冲突分析解释策略失败，不继续增加在线修正模块。

执行链路固定为：

```text
局部观测与分级通信
→ 共享 Actor
→ 局部子目标
→ quintic 轨迹
→ bicycle 控制器
```

Actor 输出后不得由集中式模块修改。新主线统一关闭：

```yaml
planner.subgoal_filter.enabled: false
safety_projection_enabled: false
projection_directional_agent_scale: false
success_zone_damping_enabled: false
formation_center_correction_enabled: false
terminal_slot_capture_enabled: false
flat_geometry_capture_enabled: false
dynamic_terminal_slot_goal_enabled: false
explicit_goal_in_execution: false
```

同时固定 `bc_updates=0`、`init_checkpoint=null`。不广播集合点、槽位、全局质心或集中式路径。历史过滤器和安全投影代码保留，但严格去中心化 schema 在配置加载时拒绝启用这些覆盖。安全只由碰撞惩罚、碰撞终止、最近邻成功门限、路径地形风险和实际质心平整度表达。

## 2. 分级通信与 101 维 Actor

### 2.1 12 m 内完整消息

每个真实规划步刷新一次，每个邻居发送 12 维：

\[
\left[
\Delta x_b,\Delta y_b,
\Delta v_{x,b},\Delta v_{y,b},
\cos\Delta\psi,\sin\Delta\psi,
h_{\mathrm{abs}},h_{\mathrm{rise}},h_{\mathrm{drop}},
r_{\mathrm{mean}},r_{\mathrm{max}},q
\right].
\]

位置、速度和航向均相对接收车辆；五维地形摘要来自发送车辆自己的 \(5\times5\) 局部地形网格；完整消息取 \(q=1\)。不发送原始地形网格、动作意图、网络状态和 Oracle 信息。

### 2.2 12 m 外稀疏消息

远距只保留上次发送时的位置和航向快照，速度与地形维全部清零：

\[
T(d)=1+3\,\operatorname{clip}
\left(
\frac{d-12}{25\sqrt{2}-12},0,1
\right)\ \mathrm{s}.
\]

令消息年龄为 \(a\)，则：

\[
q=0.5\exp\left(-\frac{a}{T(d)}\right).
\]

缓存只在 reset 和真实环境步更新。重复读取观测不能刷新消息。进入 12 m 后下一规划步恢复完整消息；离开 12 m 时立即清除速度和地形字段；远距间隔内未发送状态不得进入 Actor。

### 2.3 观测与聚合

严格去中心化 schema 为 `ego_v8_decentralized_tiered`：

\[
10+3\times12+50+5=101.
\]

- ego：10 维；
- neighbor：最多 3 个邻居，共 36 维；
- terrain：本车局部地形网格，共 50 维；
- aggregation：只由本车缓存计算的 5 维统计量。

\[
\left[
\frac{n}{3},
\frac{\bar d}{D},
\frac{d_{\max}^{\mathrm{cache}}}{D},
\bar q,
\frac{\bar a}{4}
\right],
\qquad D=25\sqrt{2}.
\]

B0 使用 `branched_v5`，保留 ego、neighbor、terrain、aggregation 四分支，只把 neighbor encoder 输入改为 36 维。旧 86、89、92 维 checkpoint 必须拒绝加载。

### 2.4 Oracle 边界

Oracle 只允许用于集中式 Critic、奖励、诊断和离线评测。固定自车观测与通信缓存后，改变 Oracle、槽位、全局量或未发送邻车状态，不得改变 Actor 观测、动作、局部轨迹和控制命令。

## 3. MAPF 拓扑与冲突诊断

理论依据为 [Lee et al., IEEE Transactions on Robotics, 2026](https://doi.org/10.1109/TRO.2025.3641865)。该论文只支撑约束搜索、拓扑和冲突分析，不作为 GNN、GRU 或可学习通信的依据。

### 3.1 地形拓扑

离线可执行地形图节点必须通过现有坡度和可通行性门限，边表示经过地形风险检查的局部可执行连接。拓扑分析使用对称化无向图。

\[
BC(v)=
\sum_{\substack{s\neq v\\t\neq v\\s\neq t}}
\frac{\sigma(s,t\mid v)}{\sigma(s,t)}.
\]

\[
S_{10}=
\frac{\sum_{v\in V_{\mathrm{top10\%}}}BC(v)}
{\sum_{v\in V}BC(v)}.
\]

分类规则：

- `Open`：不可通行节点比例小于 5%；
- `Bottleneck`：不属于 Open 且 \(S_{10}\ge0.60\)；
- `Mixed`：其余情况。

拓扑标签只用于离线分层和课程采样，不进入 Actor。

### 3.2 连续轨迹冲突

\[
d_{ij}^{\min}=
\min_k
\left\|
\mathbf p_i(t_k)-\mathbf p_j(t_k)
\right\|_2.
\]

若 \(d_{ij}^{\min}<d_{\mathrm{safe}}\)，记录预测冲突；同一车辆对连续两个规划步冲突，记录重复冲突。诊断记录冲突数、重复冲突、解除步数、消息年龄和完整消息比例，但不改变执行。

### 3.3 分层评测和启用判断

近距、远距评测分别按 Open、Mixed、Bottleneck 报告 dmax、success、collision、timeout、平整率、路径风险、冲突数和消息年龄。

- 高消息年龄组 timeout 比低年龄组高至少 10 个百分点，或失败与消息年龄的 Spearman 相关系数绝对值不小于 0.30：允许启动 B1。
- 至少 20% 的失败 episode 出现重复冲突，且其中位数不小于 2：允许启动 B2。
- timeout 高而预测冲突少：继续分析地形路径和平整集合，不增加 CBS 约束。
- collision 已低于 strict 门限时，不得因预测冲突恢复安全投影。

### 3.4 离线 CBS 上界

vanilla CBS 只在近距/远距与三类拓扑各 32 个冻结 episode 上运行，共 192 个场景。允许使用 Oracle 可行集合区域和集中式终端配置匹配，只比较 makespan、路径长度、风险和冲突可行性。CBS 路径不得用于 BC、Actor、Critic 输入或在线控制，也不能替代 strict pass。

## 4. 候选网络及启用顺序

### B1：单层 GRU

仅在 B0 基础收敛且消息年龄诊断达到门限后实现和训练。四分支编码拼接后接单层 128 维 GRU，序列长度 16，burn-in 8；Actor 循环化，Critic 保持前馈；reset、terminated 和 truncated 清空对应隐藏状态。依据为 [Foerster et al., NeurIPS 2016](https://proceedings.neurips.cc/paper_files/paper/2016/hash/c7635bfd99248a2cdef8249ef7bfbef4-Abstract.html)。

### B2：单跳图注意力

仅在动态邻接或重复冲突达到门限后实现和训练，首轮不与 GRU 组合。每个 12 维邻居消息映射为 32 维节点特征；ego 形成查询；使用 4 头、每头 12 维的单跳注意力，输出 48 维。消息掩码和 \(q\) 参与聚合，邻居排列不得影响输出。

依据为 [Tolstaya et al., CoRL 2020](https://proceedings.mlr.press/v100/tolstaya20a.html)、[Jiang et al., ICLR 2020](https://openreview.net/forum?id=S8icDSeqfvy)、[Jiang and Lu, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/6a8018b3a00b69c008601b8becae392b-Abstract.html) 和 [Das et al., ICML 2019](https://proceedings.mlr.press/v97/das19a.html)。

exp137已按上述定义完成一次显式例外4M并失败。实现仅为历史复现保留，当前不再训练、调参或与GRU组合。

### B3：八维可学习消息

只有 B2 正式通过且固定消息消融确认容量不足后才实现。发送端只依据自身 ego 与局部地形生成 8 维 `tanh` 消息；12 m 外清零；接收端复用 B2；首轮不叠加 GRU。

依据为 [Sukhbaatar et al., NeurIPS 2016](https://proceedings.neurips.cc/paper_files/paper/2016/hash/55b1927fdafef39c48e5b73b5d61ea60-Abstract.html) 和 [Foerster et al., NeurIPS 2016](https://proceedings.neurips.cc/paper_files/paper/2016/hash/c7635bfd99248a2cdef8249ef7bfbef4-Abstract.html)。B0 为默认主线；MAPPO 强基线依据为 [Yu et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html)。

## 5. Pure RL 训练课程

```yaml
episode_duration: 96 s
episode_steps: 480
parallel_envs: 2048
rollout_length: 64
algorithm: shared-joint MAPPO
bc_updates: 0
init_checkpoint: null
```

前 4096 训练时步采用 `spawn_radius=2.4–3.4 m`。随后 8192 步线性扩展到：

```text
spawn_radius = 4.5–6.5 m
center_xy_range = 2.0 m
jitter = 0.40 m
```

40M 正式训练后 8192 步对 Open、Mixed、Bottleneck 等比例采样。拓扑标签只控制采样，不进入策略。

## 6. 训练晋级和停止条件

### B0 4M screen

seed23、2048 环境、2048 训练时步，约 419 万环境交互。进入 40M 必须同时满足：

- 无 NaN、Inf 或梯度异常；
- Actor、neighbor encoder 和 terrain encoder 均更新；
- 动作标准差大于 \(10^{-4}\)；
- 末四分之一平均 dmax 比首四分之一降低至少 30%；
- 出现非零成功 episode；
- collision 不超过 10%；
- 正常地形观测与地形置零观测的动作均方差大于 0.02；
- 正常观测路径风险比地形置零对照低至少 5%。

失败后只分析奖励占比、terrain 梯度、通信利用率、消息年龄和 MAPF 冲突，不启动 40M 或新增修正模块。

#### 已完成结果与当前决策

seed23 已完成六组同族 4M screen，结果为 `0/6` 通过。完整对比见 [exp125 实验记录](experiments/exp_125_decentralized_tiered_b0_pure_rl.md)。其中：

- `relative_quintic` 的评测 dmax ratio 最好，为 `0.2047`，但 success 仅 `0.0518`；
- `relative_only` 是唯一得到正路径风险改善的运行，为 `+0.88%`，但 collision 达到 `0.4980`；
- 近距 screen 的消息全部为 12 m 内完整消息，没有远距消息年龄因素；
- Actor、neighbor encoder 和 terrain encoder 均有效更新，失败不是工程链路中断。

因此B0决定为 `stop_before_40m`。当时B1通信年龄和B2重复冲突条件均未满足；后续exp135/136补足了B2冲突证据，但exp137的一次性例外4M仍在基础gate失败，故当前依然不启用任何40M或候选架构扩展。

#### exp148 时间一致性修正后的重新筛选

exp148只修正既有 `局部子目标→quintic→bicycle` 链路的时间契约：轨迹时域改为 $H=\max(L/v_{\mathrm{ref}},\Delta t)$，控制器插值跟踪 $t=\Delta t$ 的轨迹状态，连续冲突按公共物理时间网格对齐。路径几何、Actor、通信、奖励、课程和控制增益均未改变。工程门限全部通过，时间戳速度违例率为0，冻结双种子单步弧长利用率由约 `1.77%` 提高到 `11.82%–12.05%`。

随后按原协议从随机初始化运行一次seed23 4M。训练dmax首末四分之一降低 `54.40%`，网络分支更新且动作非退化；但训练success数为0，冻结评测 dmax ratio `0.3528`、success `0`、collision `0.9990`、timeout `0.0010`。terrain contrast动作MSE为 `0.00383`，正常地形路径风险比置零对照高 `0.414%`。十项screen门限中有四项失败，故状态为 `stop_before_40m`。

两个新地形种子的失败episode诊断分别完成 `405/388` 个episode，全部因碰撞结束，且100%包含重复车辆对冲突，中位事件数均为5。近距消息完整比例为1、消息年龄为0，因此不满足B1依据；B2冲突触发虽成立，但基础收敛前置条件失败且exp137已经否决单独图注意力。当前只允许冻结分析碰撞终止前的奖励、动作和重复冲突时序，不启动40M或新的网络/控制实验。完整记录见 [exp148](experiments/exp_148_trajectory_time_consistency_fix_plan.md)。

逐车信用诊断也已完成：当前 `shared_joint` 将同一团队 GAE advantage 复制给四辆车；在 245,760 个车辆样本上，相对路径风险与一步 TD advantage 代理的 Pearson 相关为 `-0.0123`，其与单车质心进度、最近邻距离变化和团队 dmax 进度的相关绝对值均不超过 `0.012`。这支持信用稀释假设，但不等同于已经选定新的奖励算法。

#### exp149–exp150 碰撞参与者信用审计与筛选

exp149使用exp148的t1024/t2048两个checkpoint和两个新数据种子完成冻结审计。四个组合分别包含 `420/423/331/348` 个完整碰撞episode；碰撞参与车辆数中位数均为2，未参与车辆比例约为50%。最终碰撞对在终止前8/16步的repeated recall约为 `99.7%–100%`，首次重复冲突命中提前量中位数为 `14–21` 步。该证据只授权使用真实终止碰撞对，预测和重复冲突仍不得进入训练。

exp150据此只改变训练期Actor信用：把幅值为155的碰撞相关团队终止贡献按实际碰撞参与者分配，并以逐车零和残差、`0.25`固定系数加入Actor advantage。环境reward、集中式Critic、观测、通信、轨迹、控制和评测均保持exp148不变。工程门限全部通过，最大逐步零和误差为 `9.54e-7`，团队reward保持误差为0。

唯一一次seed23 4M筛选中，训练dmax降低 `59.22%`，但success episode仍为0；独立评测dmax ratio `0.2756`、success `0`、collision `0.9990`、timeout `0.0010`。terrain contrast动作MSE只有 `0.000810`，路径风险改善只有 `0.0466%`。双种子失败episode重复冲突中位数为 `9/8`，高于exp148基准的 `5/5`。因此exp150状态为 `stopped_at_4m_gate`：不启动40M，不扫描信用系数、trace、惩罚或碰撞距离，不与其他机制组合。完整协议和结果见 [exp149](experiments/exp_149_collision_participant_credit_feasibility.md) 与 [exp150](experiments/exp_150_collision_participant_actor_credit_plan.md)。

exp151随后在冻结exp150的t1024/t2048策略上，以双种子随机策略采样动作执行局部反事实干预。旁路候选只改变一辆车的一维动作，不进入环境执行；候选必须同时满足地形风险增加不超过`0.01`、规划终端dmax退化不超过`0.02 m`，并把目标碰撞对的物理时间对齐轨迹距离提高至少`0.02 m`。四个组合均通过Actor和执行不变性检查。

8步时域任一碰撞参与者存在可行替代动作的比例为`0.6840/0.7126/0.5114/0.5100`，16步为`0.7331/0.7361/0.5767/0.5616`；跨组合最小值没有达到预注册的`0.70/0.60`。两名参与者在8步时域均可行动的比例只有`0.1977–0.2853`，等量信用支持率只有`0.1777–0.2454`。因此终止参与者身份不能稳定提供可执行的局部避碰方向，exp151状态为`local_avoidance_not_actionable_stop_credit`。下一步只允许冻结分解动作表示、quintic、低层控制和地形/集合约束的可控性，不允许继续设计终止信用或启动训练。完整记录见 [exp151](experiments/exp_151_collision_credit_causal_validity.md)。

本审计同时修正了MAPF诊断中的一个时间轴问题：原冲突函数以全队最长轨迹构造公共时间网格，使第三辆车的轨迹时长能够改变目标车辆对的距离。现在每个车辆对只使用自身两条轨迹的公共物理时域，并由第三车时长不变性测试覆盖。该修正只影响离线/日志冲突量，不修改Actor、reward或控制。

exp152按完全相同的动作、checkpoint和数据种子，将局部可行动性分为无约束、仅地形、仅集合和联合约束四层，并回溯最佳安全候选是否实际改变bicycle控制命令。四个组合对exp151的联合层重建误差均为0。

t1024的8步无约束可行动率为`0.8436/0.8475`，t2048降为`0.6648/0.6676`；跨组合最小值未达到`0.70`。相反，在无约束层已经可行动的样本中，控制传递率仍为`0.8069–0.9091`，基线线速度饱和率为0、角速度饱和率最高仅`1.29%`。最佳候选约`79.9%–84.0%`来自方位角动作。因此当前首个失败层是动作—quintic局部可控性，而非低层控制饱和；训练后期还使这类可控性明显下降。下一步只允许冻结比较现有动作范围、局部扰动覆盖与quintic几何，不启动训练。完整记录见 [exp152](experiments/exp_152_action_planning_controllability_decomposition.md)。

exp153用现有动作边界内的4个局部候选、4个轴向极值候选和8点二维网格，分别计算quintic、相同终点line参考和endpoint距离。四组合的局部层均精确重建exp152。t2048的8步网格quintic可行动率提高到`0.7841/0.7536`，但仍未达到`0.80`，且范围恢复率最小只有`0.0645`；单独扩大局部覆盖没有形成一致结论。line最小仅`0.6275`且不稳定优于quintic，quintic几何损失最多`1.14%`，因此也不替换quintic。

四组合endpoint可行动率均为1，说明子目标范围本身足够；但endpoint到完整路径的途中交叉损失从t1024约`14%`升至t2048的`34.7%–37.2%`。轴向候选与二维网格的可行动率完全相同，说明同时改变单车的两维动作不能新增可行动场景。exp153状态为`mixed_action_geometry_bottleneck`，不授权动作或quintic修改。唯一后续证据路径是冻结的碰撞对双车联合动作干预，以验证问题是否来自相互轨迹协调；验证前不增加GNN、通信、在线规划或新训练信用。完整记录见 [exp153](experiments/exp_153_action_range_quintic_geometry_audit.md)。

### C0：零和地形信用残差对照

为直接验证信用稀释假设，只增加一个训练期 Actor advantage 对照，不改变环境执行、网络结构、集中式 Critic、团队奖励均值、通信、课程和 strict gate。

车辆 \(i\) 的动作相关相对路径风险为：

\[
\Delta R_{t,i}
=R\!\left(\tau_{t,i}(\rho,\beta)\right)
-R\!\left(\tau_{t,i}(\rho,0)\right).
\]

定义即时地形信用及车辆间中心化残差：

\[
c_{t,i}=-\Delta R_{t,i},
\qquad
\widetilde c_{t,i}=c_{t,i}-\frac{1}{N}\sum_{j=1}^{N}c_{t,j}.
\]

因此每个环境步都有：

\[
\sum_{i=1}^{N}\widetilde c_{t,i}=0,
\qquad
\frac{1}{N}\sum_{i=1}^{N}
\left(r_t^{\mathrm{team}}+\widetilde c_{t,i}\right)
=r_t^{\mathrm{team}}.
\]

集中式 Critic 仍使用原团队奖励计算 return 和 GAE。Actor 的车辆特定信用使用有限时域 trace：

\[
C_{t,i}=\widetilde c_{t,i}
+\gamma\lambda_c(1-d_t)C_{t+1,i},
\qquad \lambda_c=0.95.
\]

在同一 rollout 内对全部车辆的 \(C_{t,i}\) 联合标准化，最终 Actor advantage 为：

\[
A^{\mathrm{actor}}_{t,i}
=A^{\mathrm{team}}_t
+\alpha\,\operatorname{Norm}(C_{t,i}),
\qquad \alpha=0.25.
\]

选择联合标准化而非逐车标准化，是为了保留同一环境步内车辆之间的相对信用。该机制不属于完整 COMA；它只借鉴 agent-specific counterfactual advantage 的信用分配思想。相关理论动机见 [Foerster et al., AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11794)。本轮不增加动作条件 Q critic、反事实动作采样或额外价值头。

唯一配置为 `configs/experiment/exp126_decentralized_b0_centered_terrain_credit.yaml`。先完成 CPU 和 CUDA smoke，再运行一次 seed23 4M screen。除原 B0 全部门限外，还必须满足：

- 每步中心化信用和的绝对误差不超过 \(10^{-6}\)；
- 加入信用前后的团队奖励均值误差不超过 \(10^{-6}\)；
- 最后一次更新的信用 trace 标准差大于 \(10^{-4}\)；
- Critic sample 数量和 team return 语义与 B0 保持一致。

只有原 B0 4M 全部门限和上述不变量同时通过，才允许重新讨论 40M。任一条件失败即停止 C0，不扫描 \(\alpha\)、\(\lambda_c\) 或地形奖励权重，也不追加第二种信用算法。

#### C0 已完成结果

seed23 4M 正式运行选择 `t=1024` checkpoint。训练首末四分之一 dmax 降幅为 `-1.70%`；独立评测为 dmax ratio `0.6932`、success `0.0029`、collision `0.6484`、timeout `0.3486`。terrain contrast 的动作 MSE 提高到 `0.0198`，路径风险改善到 `+2.74%`，但仍未达到 5%。预测冲突和重复车辆对冲突分别升至 `0.2006/步` 与 `0.1927/步`。

所有信用不变量均通过，故结论是方法失败而非工程错误：车辆特定地形信用能够改变路径选择，但缺少车辆交互的边际责任会严重破坏碰撞安全。C0 状态为 `stopped`，不启动 40M、不调整 \(\alpha\) 或 \(\lambda_c\)，也不恢复安全投影。完整记录见 [exp126 实验记录](experiments/exp_126_decentralized_b0_centered_terrain_credit.md)。

### 已完成：动作条件反事实 Critic 的离线可行性诊断

现有高水平方法给出两个边界：COMA 使用集中式动作条件 Critic 和单车反事实 baseline 解决共享奖励信用，但原方法通过离散动作边缘化计算 baseline；当前项目的每车动作是连续二维 \([\rho,\beta]\)，不能直接照搬求和形式。[Foerster et al., AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11794)

MADDPG 和 FACMAC 支持连续多智能体动作及集中式 joint-action Critic，但会把当前 on-policy MAPPO/V-Critic 主线替换为确定性策略梯度，FACMAC 还增加价值分解与 mixing critic，改动范围明显大于本轮允许的单一对照。[Lowe et al., NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html)、[Peng et al., NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html)

因此没有直接实施 COMA、MADDPG、FACMAC 或 attention critic，而是先完成一个不更新 Actor 的离线可行性诊断：

1. 冻结 exp125 `relative_quintic` Actor 和 Critic，按原近距课程采集带随机策略采样的状态、联合动作、团队奖励、连续冲突和终止序列。
2. 使用 16 步 bootstrapped team return 作为监督目标，分别训练参数量相近的 state-only 诊断 Critic 与 state-plus-joint-action 诊断 Critic。
3. 在独立地形 seed 上比较 held-out MSE，并逐车替换动作计算反事实 \(\Delta Q_i\)，检查其与该车参与的预测冲突、实际最近邻变化和路径风险变化的关系。
4. 两个诊断模型都只写入 `outputs`，不得进入 Actor、控制器、环境奖励或正式 checkpoint。

只有同时满足以下条件，才允许在下一版计划中提出 C1：

- joint-action Critic 的 held-out MSE 比 state-only Critic 至少降低 15%；
- 单车反事实 \(\Delta Q_i\) 与车辆冲突/安全结果的秩相关绝对值至少为 0.30；
- 独立地形 seed 上结论方向一致；
- 诊断不改变冻结策略 rollout。

若未达到门限，则停止反事实 Critic 方向，不能因为 C0 的失败直接扩展为新 Actor-Critic 算法。

#### exp127 结果与决策

正式诊断改为覆盖完整 96 秒 episode：128 个训练环境和每个验证种子 64 个环境均运行 480 步。16 步目标共使用 59,520 个训练样本；两个验证地形种子的预测冲突参与率为 `17.06%/16.03%`。联合动作 Critic 的 held-out MSE 平均只改善 `2.16%`，最差地形种子改善 `1.89%`，反事实量的最大安全秩相关仅 `0.0896`。完整 episode 的 1 步和 4 步复核也分别为 `-0.019%/-0.114%`。

因此 exp127 状态为 `stop_before_c1`。不实施 C1，不增加联合动作价值头，不扫描 Critic 容量或时域参数。完整记录见 [exp127 实验记录](experiments/exp_127_joint_action_critic_feasibility.md)。

### exp128：奖励分量动作可辨识性诊断

为区分“动作接口无效”和“奖励语义未与动作对齐”，冻结同一策略，在完整 episode 上用匹配的多输出状态模型与状态—联合动作模型拟合即时加权奖励分量。关键任务分量 `gather`、`safety` 和 `terrain` 预先要求在平均值及每个独立地形种子上均达到 15% MSE 改善。

结果显示 `gather` 的动作增益为 `51.00%`，但 `safety`、`terrain` 和总奖励分别只有 `0.10%`、`-0.26%` 和 `0.13%`。已有诊断量中，预测冲突参与和最近邻距离变化的动作增益为 `17.56%/56.18%`，说明动作与动力学链路有效，主要问题集中在安全与地形奖励的当前表达。

exp128 未通过关键分量门限，状态为 `stop_reward_credit_candidate`。当前不改变训练信用、不启动 4M。下一步只允许做冻结状态配对动作干预，直接测量单车动作对路径风险、预测冲突和最近邻变化的局部因果响应；该诊断完成前不增加奖励项、网络或后处理模块。完整记录见 [exp128 实验记录](experiments/exp_128_reward_component_identifiability.md)。

#### exp129 配对动作干预结果

冻结状态配对干预已完成。对 15,360 个车辆—状态样本分别计算相对路径风险梯度与轨迹最近邻安全裕量梯度。二者目标方向余弦的中位数为 `0.0202`；强一致 ​\(\cos\theta>0.5\) 占 `27.64%`，强冲突 ​\(\cos\theta<-0.5\) 占 `26.94%`。动作对两项目标均有显著局部响应，但优化方向整体近似正交。

因此不启动固定地形—安全权重对照，不扫描信用系数，也不恢复方向性安全投影。任何后续训练目标必须先形成一个单一、状态相关的多目标信用假设并重新制定门限；exp129 的局部梯度不得进入在线 Actor 或控制器。完整记录见 [exp129 实验记录](experiments/exp_129_paired_action_interventions.md)。

### exp130 与 C2：主任务优先梯度投影

exp130 在两个独立地形种子上重建实际团队 GAE policy gradient 与 exp126 地形信用 gradient。全 Actor 负余弦批次比例均为 `46.875%`，terrain encoder 为 `50%`，辅助/主梯度范数中位比约为 `1.14`。冲突比例超过 20% 的预设门限且梯度非退化，因此只允许实施一次 C2 4M screen。完整审计见 [exp130 实验记录](experiments/exp_130_actor_gradient_conflict_audit.md)。

C2 不修改奖励、Critic target、Actor结构、通信、执行链路或 exp126 的地形信用定义。令团队 PPO 与熵正则形成主 Actor 梯度 ​\(g_p\)，地形信用 surrogate 形成辅助梯度 ​\(g_a\)。先做非对称冲突投影：

\[
\widehat g_a=
\begin{cases}
g_a-
\dfrac{g_p^{\mathsf T}g_a}
{\lVert g_p\rVert_2^2}g_p,
&g_p^{\mathsf T}g_a<0,\\[6pt]
g_a,&g_p^{\mathsf T}g_a\ge0.
\end{cases}
\]

再将辅助范数限制为不超过主梯度：

\[
\widetilde g_a=
\widehat g_a
\min\left(1,
\frac{\lVert g_p\rVert_2}
{\lVert\widehat g_a\rVert_2+\varepsilon}
\right),
\qquad
g_{\mathrm{actor}}=g_p+0.25\widetilde g_a.
\]

其中 `0.25` 沿用 exp126，不扫描。由于 ​\(g_p^{\mathsf T}\widetilde g_a\ge0\) 且 ​\(\lVert\widetilde g_a\rVert\le\lVert g_p\rVert\)，最不利的正交情形下主方向夹角不超过 ​\(\arctan 0.25\approx14.0^\circ\)。该方法受 [Yu et al., NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) 的梯度冲突处理思想启发，但属于非对称主任务优先变体。

C2 的唯一配置从 exp126 继承，只把 Actor credit 的组合方式从 advantage 直接相加改为上述梯度投影。先完成 CPU/CUDA smoke，再运行一次 seed23 4M。除原 B0 门限外必须满足：

- 每次投影后 ​\(g_p^{\mathsf T}\widetilde g_a\ge-10^{-8}\)；
- Actor 合成梯度与主梯度余弦不低于 `0.970`；
- 训练中负余弦批次比例不低于 20%，否则该机制没有实际作用；
- 团队 reward、Critic return 和执行动作语义与 B0 完全一致；
- collision 不超过 10%，且不得以碰撞提前终止换取 timeout 下降；
- 路径风险改善至少 5%，terrain contrast 动作 MSE 大于 `0.02`；
- dmax 首末四分之一降幅至少 30%，并出现非零成功 episode。

任一条件失败即停止 C2，不扫描投影系数、信用系数或范数上限，不启动 40M，也不改用对称 PCGrad、CAGrad 或约束 RL。

#### exp131 已完成结果

seed23 4M 正式运行的投影工程门限全部通过：最后一次更新冲突比例 `65.625%`，投影后最小主梯度内积为 `5.56e-9`，最低主方向余弦为 `0.970142`。但训练 dmax 首末四分之一降幅只有 `13.45%`；`t=2048` 评测为 dmax ratio `0.3525`、success `0.0361`、collision `0.6680`、timeout `0.2959`，路径风险改善仅 `2.38%`。`t=1024` 的 collision 也达到 `0.6719`。

因此 C2 状态为 `stopped`，不启动 40M、不调整任何投影或信用参数。结论是：保护团队 PPO 梯度只能防止辅助梯度直接反向，不能从安全动作辨识度不足的团队主梯度中产生车辆级安全信用。下一步若继续，只允许先离线审计基于现有最近邻成功门限的车辆级安全势函数，不新增在线约束或预测冲突奖励。完整记录见 [exp131 实验记录](experiments/exp_131_primary_projected_terrain_credit.md)。

### exp132：现有最近邻门限的逐车安全势函数审计

exp132 只使用既有的 \(d_{\mathrm{safe}}=0.42\ \mathrm{m}\) 成功门限。令 \(m_i(s)\) 为车辆 \(i\) 的最近邻距离：

\[
\Phi_i(s)=-\max\left(d_{\mathrm{safe}}-m_i(s),0\right),
\]

\[
c_{i,t}^{\mathrm{raw}}=\Phi_i(s_{t+1})-\Phi_i(s_t),
\qquad
\widetilde c_{i,t}
=c_{i,t}^{\mathrm{raw}}
-\frac{1}{N}\sum_j c_{j,t}^{\mathrm{raw}}.
\]

该量只用于冻结策略的梯度审计，不进入环境 reward、Critic target、Actor 输入或控制链路。正式协议使用 exp125 `relative_quintic` checkpoint、两个独立数据种子、每种子 128 环境乘 512 步，并抽取 32 个 4,096 样本梯度批次。

只有当两个种子的原始和中心化信用激活率均不低于 1%，正负事件均不低于 0.25%，安全/团队梯度范数中位比均位于 \([0.05,20]\)，地形—安全负余弦批次比例均不低于 20%，零和误差不超过 \(10^{-6}\)，且 checkpoint 不变时，才允许制定 C3 4M 计划。未通过时停止，不增加在线预测冲突奖励、安全约束或参数扫描。完整预注册见 [exp132 实验记录](experiments/exp_132_agent_safety_potential_credit.md)。

#### exp132 已完成结果

两个种子的原始信用激活率为 `4.174%/4.090%`，中心化激活率为 `7.677%/7.680%`，安全/团队梯度范数中位比为 `0.6083/0.5443`，地形—安全负余弦批次比例为 `46.875%/59.375%`。但是每个种子的 8 段 rollout 都只有 6 段出现安全事件，总体归一化 trace 标准差均为 `0.8660`，低于预注册下界 `0.90`。

因此 exp132 状态为 `stopped`。不启动 C3、不重新解释或放宽门限，也不将 \(0.42\,\mathrm m\) 门限信用直接写入训练。该结果表明安全梯度存在但激活偏晚；若继续，只允许先预注册并离线审计现有 `safety.near_distance` 连续项的逐车信用，不得同时改变权重、网络或控制链路。

### exp133：现有连续近距项的逐车信用审计

exp133 不新增安全项，而是分解当前环境已经使用的团队近距奖励。令 \(d_{\mathrm{near}}=0.72\,\mathrm m\)：

\[
\Phi_i^{\mathrm{near}}(s)
=-
\max\left(d_{\mathrm{near}}-m_i(s),0\right),
\]

\[
\widetilde c_{i,t}
=
\Phi_i^{\mathrm{near}}(s_{t+1})
-
\Phi_i^{\mathrm{near}}(s_t)
-
\frac{1}{N}\sum_j c_{j,t}^{\mathrm{raw}}.
\]

正式审计使用两个新数据种子 `22023/23023`、每种子 128 环境乘 512 步，以及 32 个 4,096 样本梯度批次。除激活率、梯度范数、地形冲突和零和不变量外，必须满足至少 7/8 段 rollout 激活、最迟第 2 段首次激活、“环境×rollout”平均激活率不低于 25%。

所有预注册条件通过时只允许制定一次 C3-near 4M 计划；失败则停止，不扫描距离、权重或 trace 参数。完整口径见 [exp133 实验记录](experiments/exp_133_agent_near_distance_credit.md)。

#### exp133 已完成结果

连续近距信用的原始激活率提高到 `17.540%/16.704%`，中心化激活率为 `25.633%/24.966%`，环境×rollout 平均激活率为 `43.945%/42.578%`。但 seed22023 的前两段 rollout 仍完全没有事件，首次激活索引为 `2`，激活 rollout 比例只有 `75%`，总体 trace std 为 `0.8660`。seed23023 对应指标为索引 `1`、`87.5%` 和 `0.9354`。

因此 exp133 按预注册规则停止，不启动 C3-near，也不继续扩大距离或扫描权重。若继续，只允许先离线评估现有 `0.72 m` 信用相对预测冲突和实际碰撞的提前步数；该提前量诊断不修改训练或执行链路。

### exp134：连续近距信用的冲突提前量

exp134 将每辆车连续参与预测冲突的区间合并为事件，并记录事件开始后多少步进入 `0.72 m`；对实际碰撞车辆记录从首次进入 `0.72 m` 到碰撞的规划步数。正式数据使用新种子 `24023/25023`、每种子 128 环境乘 512 步。

晋级要求包括：每种子至少 20 个碰撞车辆事件和 200 个预测冲突事件；95% 碰撞事先进入近距区；90% 至少提前 1 步、50% 至少提前 2 步且中位数不少于 2；70% 预测冲突在解除前被覆盖，覆盖延迟中位数不超过 8 步。任一条件失败即停止距离势函数信用方向。完整定义见 [exp134 实验记录](experiments/exp_134_near_credit_lead_time.md)。

#### exp134 已完成结果

两个种子分别包含 `222/244` 个碰撞车辆事件，全部在碰撞前进入 `0.72 m`，且全部至少提前 2 个规划步；提前量中位数为 `74/67` 步、10% 分位数为 `48/46` 步。真实碰撞相关门限全部通过。

但是，`19,744/19,730` 个车辆级预测冲突事件中，只有 `37.956%/35.646%` 在解除前进入近距区，低于预注册的 70%。因此 exp134 和 C3-near 仍按规则停止。该差异说明单步预测轨迹冲突包含大量随下一次重规划解除的事件；下一步只允许离线比较单步冲突与连续重复冲突的结果相关性，不将其写入 reward。

### exp135：单步与重复冲突的结果相关性

exp135 将车辆连续参与预测冲突的区间合并为事件；事件内出现 pair-level repeated 标记时归为重复事件。碰撞结果只允许出现在事件期间或结束后 4 步内，且不得跨 episode；每个碰撞车辆事件另外回看前 8 步的重复冲突参与。

只有当重复事件样本不少于 500、碰撞结果率不少于 1%、结果率至少为非重复事件的 2 倍、碰撞召回率不少于 80%，且近距覆盖率至少高 20 个百分点时，才保留重复冲突作为 B2/MAPF 的主要决策指标。通过不代表启用 B2，也不重新授权 C3-near。完整协议见 [exp135 实验记录](experiments/exp_135_repeated_conflict_outcomes.md)。

#### exp135 已完成结果

两个种子分别记录 `14,632/14,352` 个非重复事件和 `4,556/4,511` 个重复事件。非重复事件的碰撞结果均为 0；重复事件分别有 `238/221` 个碰撞结果，结果率为 `5.224%/4.899%`。重复冲突对碰撞车辆的 8 步召回率均为 100%，近距覆盖率比非重复事件高 `37.533/39.243` 个百分点。

因此 exp135 状态为 `retain_repeated_conflict_metric`。所有单步冲突继续写日志，但不得用于 B2 或奖励决策；架构判断只看 pair-repeated 指标。该结果不满足 B2 的全部启用条件，因为 B0 基础收敛尚未通过，且失败 episode 级重复冲突占比尚未完成正式审计。

### exp136：失败 episode 的重复冲突触发条件

exp136 以 pair-level `repeated` 从 0 到 1 作为事件起点，在每个完成 episode 内统计事件数。失败 episode 包括 collision、out-of-bounds 和 timeout，采样尾部未完成的 episode 不计入。

只有当每个种子至少有 100 个失败 episode、至少 20% 的失败 episode 包含重复事件，且把零事件 episode 纳入后事件数中位数仍不低于 2，B2 的冲突触发条件才成立。通过不改变 B0 基础收敛前置条件。完整协议见 [exp136 实验记录](experiments/exp_136_failed_episode_repeated_conflicts.md)。

#### exp136 已完成结果

两个种子获得 `141/130` 个失败 episode，含重复车辆对冲突的比例均为 100%，事件数中位数为 `17/18`、10% 分位数仍为 `7/9`。collision 和 timeout 失败分别也全部命中。exp136 状态为 `b2_conflict_trigger_met_base_not_converged`。

因此 B2 的重复冲突证据条件已经满足，但当前启用矩阵仍为：指标有效、episode 触发通过、基础收敛失败、B2 不允许实现。该结果揭示原前置条件存在潜在循环依赖；如需修订，必须另行预注册为单一 B2 screen，不能默认为本实验已经解除限制。

### exp137：B2 单跳图注意力一次性例外筛选

exp135/136 的新证据表明，重复车辆对冲突不是少量失败的伴随现象，而是两个种子全部失败 episode 的共同特征。由于该现象正是 B2 的目标问题，继续要求 B0 在不具备动态邻接聚合能力时先行通过基础收敛，可能形成循环前置条件。

因此仅授权一次预注册的 seed23 B2 4M 例外筛选。该例外只把36维展平 `neighbor_encoder` 替换为每邻居独立编码、四头单跳注意力和48维聚合输出；ego、terrain、aggregation、Actor主干、Critic、通信物理量、奖励、课程和执行链路均保持不变。工程gate、基础收敛gate或候选性能gate任一失败即停止，不授权40M，也不得追加GRU、多跳传播、学习消息、新奖励或安全后处理。完整协议见 [exp137 实验记录](experiments/exp_137_decentralized_b2_graph_attention.md)。

#### exp137 已完成结果

工程gate全部通过，但4M基础收敛gate失败。B2独立评测dmax ratio从B0的 `0.2047` 改善到 `0.1707`，同时collision从 `0.0967` 急剧升至 `0.7881`，success为 `0.0439`；训练dmax只降低 `20.79%`。terrain动作MSE为 `0.00121`，路径风险降低比例为 `-0.00062`，两项terrain gate仍失败。

相同双种子重复冲突复核中，失败episode命中率仍为100%，事件中位数由B0的 `17/18` 变为 `16/22`，未实现两种子一致下降。exp137状态为 `stopped_at_base_gate`，不运行候选远距对比、不启动40M，并停止继续扩大或调参图注意力结构。

### exp138：现有安全奖励聚合语义辨识诊断

exp128已经表明最近邻距离变化具有明确动作信息，而当前团队safety奖励不可辨识；exp137进一步排除了邻居排列建模不足这一解释。exp138因此只离线比较当前近距gap的四车平均与最危险车辆对最大值聚合，碰撞项、距离阈值和权重全部保持不变。

只有worst-pair聚合在两个验证种子上的动作辨识增益均不低于15%，相对当前mean聚合均提高至少10个百分点，且目标激活率均不低于5%，才允许另行制定一次单变量4M。失败后不扫描温度、top-k、距离或权重。完整协议见 [exp138 实验记录](experiments/exp_138_safety_aggregation_identifiability.md)。

#### exp138 已完成结果

两个验证种子的安全目标激活率为 `28.906%/25.837%`，但worst-pair动作辨识增益仅为 `0.0973%/0.00016%`，相对mean聚合只提高 `0.0120/0.0166` 个百分点。目标重构误差为0，Actor参数完全不变。

因此exp138状态为 `stop_safety_aggregation_change`，不启动聚合改动4M。B0、B2和安全聚合均失败后，当前约束范围内不再存在有证据支持的网络或团队reward微调；继续训练前必须明确修订是否允许逐车训练信用或约束优化目标。

### exp139：逐车局部安全信用动作辨识审计

exp139不直接修改训练，而在严格去中心化Actor的101维本车观测上，测量加入本车动作后对raw近距势函数信用、零和中心化信用和重复冲突参与的预测增益。只有raw与中心化信用在两个验证种子上都达到15%动作增益、激活率达到exp133门限，并保持exp134真实碰撞提前量证据，才形成一次C3-near边界修订建议。重复冲突仅作对照，不进入reward。完整协议见 [exp139 实验记录](experiments/exp_139_local_safety_credit_identifiability.md)。

#### exp139 已完成结果

正式审计使用262,144个训练agent-step和两个各131,072个agent-step的验证种子。raw信用激活率为 `17.244%/18.524%`，动作增益为 `15.791%/14.901%`：均值越过15%，但第二个种子未越过逐种子门限。逐步零和中心化信用虽然具有 `26.318%/27.512%` 的激活率，动作增益却只有 `8.219%/7.203%`。重复冲突参与的动作增益仅为 `2.283%/3.333%`。

中心化零和误差、Actor参数变化和探针动作变化均为0，exp134真实碰撞提前量证据保持成立。故exp139按预注册规则失败，状态为 `stop_local_safety_credit_reconsideration`。不启动C3-near、不扫描信用形式或权重，也不将重复冲突写入奖励。当前计划内可执行的B0、B2、团队安全聚合与零和逐车信用方向均已达到停止条件；任何新的4M训练都必须先显式修订研究边界，而不能继续以诊断名义扩展训练模块。

### exp140：非零和逐车近距信用边界修订

为继续推进而不扩展执行模块，本计划只开放一次组件级训练例外：保留exp139中的raw逐车近距势函数差分，不执行逐步跨车中心化，并以固定`0.25`尺度加入Actor advantage。团队reward、集中式Critic、地形奖励、网络、通信和执行链路均保持exp125 `relative_quintic`不变。

该修订的目的仅是验证中心化造成的责任损失是否真实影响学习。4M必须相对B0将collision至少降低30%，同时限制success与dmax退化，并在两个冻结诊断种子中将失败episode重复冲突中位数均降低20%。任一条件失败则永久停止该方向；通过也只能制定下一份统一地形—安全信用计划，不能直接进入40M。完整公式、工程不变量和门限见[exp140实验记录](experiments/exp_140_agent_local_near_credit_screen.md)。

#### exp140 已完成结果

工程gate、CPU/CUDA smoke、Pure RL初始化和团队reward保持不变量均通过。4M训练完成32次联合更新，训练dmax降低 `23.92%`，独立评测为dmax ratio `0.2397`、success `0.0439`、collision `0.2295`、timeout `0.7266`。collision相对B0的 `0.0967` 增加而非降低；双种子失败episode重复冲突中位数也由 `17/18` 增至 `19/20`。terrain动作MSE仅 `0.00246`，路径风险降低 `0.085%`。

因此exp140状态为 `stopped_at_component_gate`。它证明“恢复raw动作辨识信息”不足以形成长期安全策略，不允许统一信用计划、参数扫描或40M。后续若研究约束优化，必须先冻结验证cost value/advantage是否可估计，不能直接新增cost critic和乘子。

### exp141：碰撞cost value可估计性前置审计

exp141冻结exp125策略，只拟合由54维集中式state预测未来64步内真实collision终止的诊断分类器。与训练集发生率常数基线比较，要求双验证种子的AUROC不低于0.75、AUPRC不低于正标签率3倍、Brier改善不低于15%，且事件覆盖和Actor不变量同时通过。失败后不实现约束优化；通过也只允许制定PPO-Lagrangian计划。完整协议见[exp141实验记录](experiments/exp_141_collision_cost_value_feasibility.md)。

#### exp141 已完成结果

训练集有482个collision episode，正标签率 `12.368%`。两个验证种子的AUROC为 `0.9392/0.9331`，AUPRC为 `0.6231/0.5792`，Brier改善为 `35.59%/30.00%`；Actor参数和探针输出完全不变。所有门限通过，状态为 `allow_lagrangian_plan_only`。

### exp142：真实碰撞PPO-Lagrangian组件筛选

exp142只允许新增一个读取现有54维集中式state的cost critic，以及一个按真实collision rate更新的标量乘子。cost为collision终止指示，Actor使用标准化reward advantage减去乘子加权cost advantage；团队reward、reward Critic、101维Actor和执行链路不变。固定cost GAE、dual公式和4M门限见[exp142计划](experiments/exp_142_collision_lagrangian_component_plan.md)。本阶段仍不授权40M。

#### exp142 已完成结果

工程门限、CPU/CUDA smoke和32次联合更新均正常，cost critic、reward Critic和Actor均发生有效更新。训练期episode等效碰撞率的首末四分之一均值由 `1.0963` 降至 `0.05447`，下降 `95.03%`；最终乘子为 `1.0484`。独立评测collision为 `0.02539`，双诊断种子的重复冲突中位数均由 `17/18` 降至0。

但独立评测success为 `0`、dmax ratio为 `0.68146`、timeout为 `0.97461`，terrain动作MSE仅 `0.000259`。约束目标促使策略回避接近，而没有学会兼顾集合与避碰。因此exp142状态为 `stopped_at_component_gate`：不启动40M，不扫描collision预算、dual参数、PID项或cost网络容量，不把cost critic写入正式执行架构。

### exp143–exp144：B0训练深度假设复核

exp143首先使用冻结日志检查4M是否在明显平台期结束。B0从1024到2048 checkpoint的dmax相对改善 `29.08%`、collision降低 `90.09%`、success增加 `3.71` 个百分点，末四分之一训练dmax仍改善 `14.35%`，且2048步只达到4096步课程warmup的一半。由于exp142在cost/reward尺度竞争后仍新增2个success，未满足预注册的严格零增长条件，exp143没有直接授权12M。

exp144随后在五个相同场景评测种子上配对复核B0的t1024/t2048 checkpoint。dmax、collision和success在5/5种子全部改善，平均变化分别为 `29.76%`、`89.94%` 和 `+4.41` 个百分点。然而三个terrain-contrast种子中，动作MSE从约 `0.00057` 增至约 `0.00746` 的同时，路径风险改善从正值变为负值；平均趋势恶化 `0.2508` 个百分点。

因此“加深相同B0训练会自然学会地形规划”的假设被否决。后续不启动12M/40M，不扫描地形奖励权重。若继续，只允许先定义并冻结审计一个统一的逐车局部任务回报，要求同一语义同时对集合、地形和安全具有动作辨识度；不得继续分别叠加单项信用。

### exp145：统一逐车局部任务回报前置审计

exp145把逐车到其余车辆质心的距离进度、逐车相对quintic路径风险和逐车最近邻安全势函数进度作为同一组离线目标。诊断只比较 `local_obs` 与 `local_obs+own_action` 回归器，不更新策略。三个原始分量必须在两个验证种子上分别达到15%动作增益，标准化等权统一目标必须达到20%，同时满足事件覆盖、正负地形样本和Actor冻结不变量，才允许形成逐车advantage训练计划。完整公式与停止规则见[exp145计划](experiments/exp_145_unified_agent_local_task_reward_identifiability_plan.md)。

正式结果中，局部集合、地形和统一目标的跨种子平均动作增益分别为 `77.58%/28.44%/25.69%`，但局部安全只有 `9.58%`，最差验证种子为 `8.12%`。安全事件覆盖和方差均通过，Actor完全冻结。因此exp145按规则停止，不删除安全分量、不降低门限、不实现统一逐车advantage。

### exp146：最近邻成对安全动作耦合审计

exp146保持相同冻结数据，只增加训练诊断可见的最近邻动作，比较纯观测、本车动作、邻车动作和成对动作四个同容量回归器。邻车动作乱序使成对模型MSE平均恶化 `91.36%`，证明成对关系包含真实信号；但成对总增益的最差种子为 `23.93%`，已知邻车后的本车增益和已知本车后的邻车增益最差为 `12.17%/14.09%`，均未达到预注册门限。因此不实现最近邻成对安全Critic，也不重新开启COMA或通用联合动作Critic。

### exp147–exp148：轨迹执行契约修正

信用方向停止后，exp147检查现有 `Actor→局部子目标→quintic→bicycle` 链路本身。当前生成器无论路径长短都把12点完整路径标为0.2 s，控制器却固定跟踪索引1并在下一环境步重规划；`reference_speed=1.15 m/s`未参与时间戳或跟踪点选择。双种子中 `78.25%–79.45%` 的非零路径要求超过1.35 m/s最大速度，参考速度所需时域中位数是声明时域的 `2.51–2.57` 倍，实际一步只执行规划弧长的约 `1.77%`。

该失配使地形奖励和连续冲突诊断评价整条路径，而动力学几乎只执行其开头。exp148随后保持所有策略、通信、奖励、课程与安全配置不变，只按弧长/参考速度生成预测时域，在0.2 s处插值选择跟踪点，并把冲突诊断重采样到公共真实时间网格。工程门限全部通过，但旧checkpoint不迁移的seed23随机初始化4M重新筛选仍因success、collision和terrain contrast失败，故不进入40M。完整结果见[exp148记录](experiments/exp_148_trajectory_time_consistency_fix_plan.md)。

同时已经修正筛选语义：路径风险沿实际 quintic 轨迹采样；checkpoint 评测按其训练时步恢复初始状态课程；主配置显式启用实际质心平整度奖励。

### B0 40M formal

4M 通过后从随机初始化重新训练 seed23，共 20480 个训练时步、约 4194 万环境交互。近距和远距都通过后，再独立训练 seed31 和 seed47。

### 候选架构门限

- B1：远距 success 提高至少 10 个百分点，或 timeout 相对下降至少 20%；collision 恶化不超过 2 个百分点，近距 success 下降不超过 5 个百分点。
- B2：满足 B1 性能门限，并通过排列不变性、单邻居屏蔽鲁棒性和重复冲突下降检查。
- B3：满足 B2 门限；消息置零后性能显著下降；跨 12 m 后学习消息完全消失；不得出现缓存外泄漏。

候选失败后立即停止，不用后处理、额外奖励或辅助损失补偿。GRU 与图注意力只有分别通过后，才可在下一轮计划中讨论组合。

## 7. 正式验收

近距和远距都必须满足：

\[
\mathrm{dmax\ ratio}\le0.20,
\qquad
\mathrm{success}\ge0.90,
\]

\[
\mathrm{collision}\le0.02,
\qquad
\mathrm{timeout}=0.
\]

成功 episode 还必须通过实际质心平整度、最近邻距离、dmax、dispersion 和低速保持 gate。三个种子全部通过后，才生成正式动画、进入 PhysX 闭环验证并更新 `docs/technical_design.md`。MAPF/CBS 结果不计入 strict pass。

## 8. 测试、日志和产物

通信测试覆盖 11.9、12.0、12.1 m 与地图最大距离，验证更新周期、消息质量、出入 12 m 字段切换、重复读取不刷新和缓存外状态不泄漏。

网络测试覆盖 101 维切片、旧 checkpoint 拒绝和 Oracle 执行不变性。MAPF 测试覆盖 Open/Mixed/Bottleneck 分类、轨迹交叉、重复冲突和诊断不干预执行。

通信和冲突日志新增：

```text
full_message_ratio
sparse_message_ratio
mean_message_age
mean_update_period
far_pair_ratio
predicted_conflict_count
repeated_pair_conflict_count
mean_conflict_resolution_steps
message_age_at_conflict
full_message_conflict_ratio
```

离线拓扑输出包含：

```text
topology_class
blocked_ratio
bc_mean
bc_variance
bc_high_region_ratio
```

结果仍写入 `outputs/runs/<experiment>/<run_id>/`。GIF、单个 checkpoint 和 TensorBoard 曲线不能替代机器可读 strict 评测。

## 9. 明确不做

- 方向性安全约束或方向性 mask；
- 行为克隆；
- 在线 CBS、CBSw/P 或 PBS；
- 固定车辆优先级；
- 集合点、槽位或 CBS 路径广播；
- leader election、显式一致性协议或多跳 GNN；
- 未经独立验证就叠加 GRU 和图注意力；
- 12 m 外学习消息；
- 本轮同时研究丢包、噪声和随机时延；
- 用后处理掩盖策略未收敛。
