# exp157：共同站点可辨识性与低层可达性诊断

更新时间：2026-08-19。

## 1. 目的与证据边界

`exp157`不直接实现完整D-STC，而是先拆开两个问题：

- H0：当前局部地形、12 m通信图和47维原语是否提供了必要的信息与可控性；
- H1：当四辆车获得同一个可行站点区域条件时，N1低层Actor能否完成到达、避碰、收紧和稳定保持。

H1中的共同站点由训练环境在reset时给定，只用于低层能力上界诊断。它不是最终去中心化选址机制，也不能用于证明车辆已经学会共同发现和选择平地。

## 2. H0冻结审计

H0使用`exp156`固定的六分层1152场景，不训练策略、不生成教师动作。审计包括：

- 站点区域是否落入至少一辆车的当前多尺度采样范围；
- 12 m通信图是否连通，从而在信息语义允许时具备传播证据的可能；
- 站点势场在场景整体SE(2)变换下是否保持不变；
- 成功门限是否允许四车在站点区域内形成安全终端几何；
- 47维原语在历史和人工死锁场景中是否存在联合解。

首轮结果为：

| 指标 | 结果 |
| --- | ---: |
| 固定场景数 | 1152 |
| 12 m通信图连通率 | 100% |
| 至少一车初始可见且可传播站点证据 | 40.97% |
| 站点势场SE(2)最大误差 | $4.17\times10^{-7}$ |
| 原语可解除冲突场景 | 12/12 |

分层差异显著：近距Open和Mixed的初始证据传播可能率分别为83.85%和86.98%，近距Bottleneck为0；远距Open、Mixed和Bottleneck分别为25.00%、42.19%和7.81%。因此，当前瞬时局部地形即使通信图连通，也经常没有任何车辆在reset时观测到目标区域。

H0只证明H1低层诊断可以启动。候选提取、跨车数据关联和commit共识尚未实现，因此`decentralized_site_selection_ready=false`。

## 3. H1接口

H1采用临时N1 CNN基线和47维差速原语。Actor观测为：

$$
15+3\times17+336+5=407.
$$

三个地形尺度由原来的相对高度、风险两通道扩展为：

```text
[relative_height, risk, feasible_site_potential]
```

对局部采样点 $x$、站点中心 $c$、可行区域半径 $R$，势场定义为：

$$
H(x)=\exp\left[
-\frac{1}{2}
\left(
\frac{\max(\lVert x-c\rVert_2-R,0)}{\sigma}
\right)^2
\right],
$$

其中 $R=0.75\ \mathrm m$、$\sigma=2.0\ \mathrm m$。该空间通道与地形网格对齐，不使用直接的车体系方向/距离目标。Critic仍接收原950维状态，不复制新增势场。

H1恢复`oracle`进展奖励权重0.5，以隔离低层goal-conditioned控制能力；保持Pure RL、BC关闭、随机初始化、无安全投影和无Actor后动作覆盖。

## 4. 训练预算与判读

H1使用N1、seed23、256并行环境、rollout 64和2400 policy iterations，总计39,321,600环境交互。课程、熵调度及六分层配对评测与`exp156`一致。

结果解释规则固定为：

- H1通过而原N1失败：共同站点可辨识性/协调是主要瓶颈，才允许规划完整D-STC；
- H1仍失败：47维原语、低层奖励、控制或终端稳定仍存在核心问题，禁止直接实现完整高层协议；
- H1不能成为推荐checkpoint，也不能替代严格去中心化验收。

## 5. 入口与产物

配置与入口：

```text
configs/experiment/exp157_h1_site_belief_n1.yaml
scripts/audit_exp157_h0.py
scripts/run_exp157_h0_h1.py
```

产物：

```text
outputs/runs/exp157_site_belief_diagnostic/h0_frozen_audit/
outputs/runs/exp157_h1_site_belief_n1/h1_n1_seed23_full_2400iter/
```

## 6. H1运行状态

H1在134,400/153,600训练步后中断，当前没有训练进程，也没有生成最终1152场景配对评测。最后一次课程诊断仍为success 0、collision约0.112、timeout约0.888。该run状态为：

```text
lifecycle_status: incomplete_interrupted
eligible_for_baseline: false
resume_allowed: false
```

其 `ppo_timestep_134400.pt` 只允许作为exp158冻结奖励可辨识性审计的行为策略。exp158的H1-GAE与H1-DAE必须从相同随机初始化重新完整训练，不能把本run续训结果作为配对基线。
