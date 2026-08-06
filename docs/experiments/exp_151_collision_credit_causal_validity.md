# exp151：碰撞参与者信用因果有效性审计

## 目的

exp149证明真实碰撞责任通常只涉及四辆车中的一对；exp150据此把碰撞相关Actor信用按真实终止参与者重新分配，但唯一一次4M筛选仍为success `0`、collision `0.9990`，重复冲突中位数由`5/5`升至`9/8`。

本实验不修改训练，只检验exp150隐含的两个假设：

1. 最终碰撞参与车辆在碰撞前1–16个规划步确实存在局部可执行的避碰动作；
2. 同一碰撞对中的两辆车具有近似对称的动作责任，因此可以接受相同幅值的负信用。

若局部反事实动作不能改善安全距离，终止参与者身份不足以提供可学习方向；若只有一辆车具有明显改善能力，则对两名参与者等量惩罚同样不成立。

## 冻结设置

- 配置：`configs/experiment/exp150_collision_participant_actor_credit.yaml`；
- checkpoint：exp150的`t1024`与`t2048`；
- 数据种子：`46023`、`47023`；
- 每个组合：128个环境、512个规划步；
- 动作：冻结策略分布采样，使用checkpoint中的标准差，随机生成器由数据种子固定；
- 分析时域：碰撞终止前`1、2、4、8、16`步；
- 局部动作扰动：`delta=0.15`；
- Actor、Critic、环境reward、通信、轨迹生成、控制和状态推进均不得改变。

使用随机策略动作而不是deterministic mean，是为了复现exp150训练时“采样动作—终止信用”的真实对应关系。干预动作只在旁路中计算轨迹，不送入环境执行。

## 局部反事实定义

在状态 $s_t$，冻结策略采样动作记为 $\mathbf a_t$。对车辆 $r$ 的两维动作分别作正负扰动，候选集合为：

\[
\mathcal U_r(\mathbf a_t)
=
\left\{
\operatorname{clip}(\mathbf a_{t,r}\pm0.15\mathbf e_k,-1,1)
\mid k\in\{1,2\}
\right\}.
\]

其他车辆动作保持不变。沿exp148以后的物理时间戳对齐quintic轨迹，车辆对 $(i,j)$ 的最小预测距离为：

\[
d_{ij}(\mathbf a_t)
=
\min_{\tau}
\left\|
\mathbf p_i(\tau;\mathbf a_t)
-\mathbf p_j(\tau;\mathbf a_t)
\right\|_2.
\]

车辆 $r$ 对该车辆对的最佳局部安全改善定义为：

\[
g_{r,ij}(t)
=
\max_{\mathbf u\in\mathcal U_r}
\left[
d_{ij}(\mathbf u,\mathbf a_{t,-r})
-d_{ij}(\mathbf a_t)
\right].
\]

同时计算候选轨迹地形风险变化 $\Delta R_r$ 和规划终端队伍dmax变化 $\Delta D$。若候选满足：

\[
\Delta R_r\le0.01,
\qquad
\Delta D\le0.02\ \mathrm m,
\]

则认为该候选没有以明显增加地形风险或破坏集合进度换取避碰。只在这些候选中取最大值，得到 $g^{\mathrm{feasible}}_{r,ij}(t)$。

对于最终实际碰撞对 $(i,j)$，在终止前 $h$ 步分别读取：

\[
g_i=g^{\mathrm{feasible}}_{i,ij}(t_c-h),
\qquad
g_j=g^{\mathrm{feasible}}_{j,ij}(t_c-h).
\]

当 $g_i+g_j>0$ 时，局部反事实责任份额为：

\[
s_i=\frac{\max(g_i,0)}{\max(g_i,0)+\max(g_j,0)},
\qquad
s_j=1-s_i.
\]

责任不对称度定义为：

\[
A_{ij}=2\left|s_i-\frac12\right|.
\]

$A_{ij}=0$表示局部改善能力完全对称，$A_{ij}=1$表示只有一辆车具有有效改善方向。

## 主要指标

每个checkpoint—种子组合、每个时域分别统计：

- `any_actionable_rate`：至少一名参与者满足 $g^{\mathrm{feasible}}\ge0.02\ \mathrm m$ 的碰撞对比例；
- `both_actionable_rate`：两名参与者均满足上述门限的比例；
- `participant_actionable_rate`：逐参与车辆满足门限的比例；
- `equal_credit_supported_rate`：两名参与者均可行动，且 $s_i,s_j\in[0.25,0.75]$ 的比例；
- `responsibility_asymmetry`：$A_{ij}$ 的分布；
- `locally_optimal_or_insensitive_rate`：参与车辆的最佳改善不超过`0.005 m`的比例；
- `nonparticipant_pair_gain_abs_max`：修改非参与车辆动作对最终碰撞对轨迹距离的数值影响；
- 地形风险和规划终端dmax的候选代价；
- exp150信用trace在各时域的理论权重 $(\gamma\lambda_c)^h$。

非参与车辆的轨迹不构成最终碰撞对，因此其单步旁路动作不应改变 $d_{ij}$。该数值检查用于揭示exp150零和补偿项是否把正信用分配给当前碰撞因果链之外的车辆。

## 预注册判定

所有四个checkpoint—种子组合必须先满足：

- 至少100个完整碰撞episode；
- checkpoint Actor摘要与探针动作保持不变；
- 环境实际执行动作仍为冻结策略采样动作，旁路候选从未送入`env.step`；
- `nonparticipant_pair_gain_abs_max <= 1e-6`；
- 所有指标有限。

随后按以下顺序判定：

1. 若四个组合在8步时域的`any_actionable_rate`均不低于70%，且16步时域均不低于60%，说明现有动作和quintic链路在碰撞前存在局部避碰能力。
2. 在满足第1项的前提下，若四个组合的8步`equal_credit_supported_rate`均低于50%，或者8步责任不对称度中位数均不低于0.50，则否决“同一碰撞对等量信用”假设，只允许继续做冻结的反事实difference-advantage可估计性审计，不授权训练。
3. 若第1项失败，则停止碰撞参与者信用方向；下一步应检查动作表示、规划时域或低层可控性，不得继续设计新的终止信用。
4. 若第1项通过且等量信用得到支持，但exp150仍失败，则只允许审计采样动作score与延迟信用的协方差和有效样本量，不得直接调整trace或信用系数。

无论本实验结果如何，exp151本身不授权4M、12M或40M训练。

## 明确不做

- 不把预测冲突、反事实距离或责任份额写入reward；
- 不修改exp150 checkpoint；
- 不执行旁路候选动作；
- 不恢复安全投影、方向mask或后处理；
- 不新增Critic、GNN、GRU或可学习消息；
- 不扫描动作扰动、可行动门限、地形门限或dmax门限；
- 不以局部预测改善替代strict evaluation。

## 产物路径

```text
outputs/runs/exp151_collision_credit_causal_validity/
  frozen_exp150_dual_checkpoint_dualseed/
    config/experiment.yaml
    metrics/collision_credit_causal_validity.json
    run_manifest.json
  _suite/
    metrics/suite_summary.json
    run_manifest.json
```

## 当前状态

正式冻结审计已经完成。首次运行发现原公共时间冲突函数使用全队最长轨迹建立时间网格，使第三辆非参与车辆的轨迹时长能够改变目标车辆对的冲突距离，产生最高约`0.09 m`的伪影响。该问题只污染诊断，不改变Actor、轨迹生成或控制。修正后，每个车辆对只使用自身两条轨迹的公共物理时域，并新增“第三车时长变化不影响目标车辆对”的回归测试；随后使用完全相同的checkpoint、种子和门限重新运行。

四个有效组合分别包含`322/339/348/347`个完整碰撞episode，全部工程门限通过。非参与车辆对最终碰撞对的距离影响严格为0，Actor摘要、探针动作和环境实际执行动作均保持不变。

主要结果如下：

| checkpoint—seed | 8步任一参与者可行动 | 8步两车均可行动 | 8步等量信用支持 | 8步责任不对称中位数 | 16步任一参与者可行动 |
| --- | ---: | ---: | ---: | ---: | ---: |
| t1024—46023 | 0.6840 | 0.2853 | 0.2454 | 0.6300 | 0.7331 |
| t1024—47023 | 0.7126 | 0.2463 | 0.2287 | 0.7870 | 0.7361 |
| t2048—46023 | 0.5114 | 0.2216 | 0.2045 | 0.5513 | 0.5767 |
| t2048—47023 | 0.5100 | 0.1977 | 0.1777 | 0.5083 | 0.5616 |

8步`any_actionable_rate`最小值为`0.5100`，低于预注册的`0.70`；16步最小值为`0.5616`，低于`0.60`。因此状态为`local_avoidance_not_actionable_stop_credit`，终止参与者信用方向停止。

次要结果也表明等量责任假设较弱：8步两车均可行动比例只有`0.1977–0.2853`，等量信用支持率只有`0.1777–0.2454`。但因可行动性前置门限已经失败，本实验不据此授权反事实difference advantage。下一步只允许冻结分解“动作表示、quintic轨迹、低层控制、地形风险约束和集合进度约束”中哪一层限制了局部避碰，不授权任何新训练。

正式汇总位于：

```text
outputs/runs/exp151_collision_credit_causal_validity/
  _suite/metrics/suite_summary.json
  frozen_exp150_dual_checkpoint_dualseed/
    metrics/collision_credit_causal_validity.json
    run_manifest.json
```
