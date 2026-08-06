# exp152：动作—规划—控制可控性分解

## 目的

exp151表明，碰撞前8/16步能够同时改善安全距离、地形风险和集合进度的局部动作比例不足，因此终止碰撞信用方向已经停止。但exp151尚不能区分以下原因：

1. 两维Actor动作本身不能产生有效的避碰quintic轨迹；
2. 轨迹可以改善，但低层控制器对动作扰动不敏感或处于饱和区；
3. 避碰方向存在，但被地形风险条件排除；
4. 避碰方向存在，但会破坏集合进度；
5. 地形和集合条件分别可满足，但没有同一个候选同时满足两者。

本实验只做冻结分解，不修改训练或执行链路。

## 冻结设置

- 配置：`configs/experiment/exp150_collision_participant_actor_credit.yaml`；
- checkpoint：exp150的`t1024`与`t2048`；
- 数据种子：`46023`、`47023`；
- 每个组合：128个环境、512个规划步；
- 策略动作：使用checkpoint标准差的冻结随机采样；
- 分析时域：碰撞前`1、2、4、8、16`步；
- 候选动作：与exp151完全相同的四个`delta=0.15`单维正负扰动；
- 地形风险容差：`0.01`；
- 规划终端dmax容差：`0.02 m`；
- 可行动安全距离增益：`0.02 m`。

门限和数据均不因exp151结果调整。候选只在旁路计算，不进入`env.step`。

## 分层可行动性

对最终碰撞对 $(i,j)$、参与车辆 $r\in\{i,j\}$ 和候选动作 $u$，安全距离改善仍定义为：

\[
g_{r,ij}(u)
=d_{ij}(u,\mathbf a_{-r})-d_{ij}(\mathbf a).
\]

分别定义四种最佳改善：

\[
g^U_{r,ij}=\max_u g_{r,ij}(u),
\]

\[
g^R_{r,ij}
=\max_{u:\,\Delta R_r(u)\le0.01}g_{r,ij}(u),
\]

\[
g^D_{r,ij}
=\max_{u:\,\Delta D(u)\le0.02}g_{r,ij}(u),
\]

\[
g^{RD}_{r,ij}
=\max_{u:\,\Delta R_r(u)\le0.01,\,\Delta D(u)\le0.02}
g_{r,ij}(u).
\]

其中：

- $U$：不施加地形或集合条件；
- $R$：只保留地形条件；
- $D$：只保留集合进度条件；
- $RD$：同时保留两项条件，对应exp151的可行动定义。

对每个车辆对，只要任一参与车辆的相应最佳改善不低于`0.02 m`，即记为该层可行动。

## 约束阻塞定义

只在 $U$ 层已经可行动的碰撞对中计算：

- `terrain_blocked_rate`：$U=1$且$R=0$；
- `dmax_blocked_rate`：$U=1$且$D=0$；
- `cross_constraint_incompatibility_rate`：$R=1$、$D=1$但$RD=0$；
- `combined_feasibility_loss_rate`：$U=1$且$RD=0$。

这些指标分别表示地形约束单独阻塞、集合约束单独阻塞、两个条件需要不同候选才能满足，以及两项条件合并后的总损失。

## 动作维度和控制传递

分别记录半径动作和方位角动作产生的最佳安全改善：

\[
g^\rho_{r,ij},\qquad g^\beta_{r,ij}.
\]

由此统计最佳候选来自`radius`或`bearing`的比例。

对产生 $g^U$ 的候选，使用与环境完全相同的quintic轨迹和bicycle控制器计算控制变化。归一化控制响应为：

\[
C_r
=
\sqrt{
\left(\frac{\Delta v_r}{v_{\max}}\right)^2
+
\left(\frac{\Delta\omega_r}{\omega_{\max}}\right)^2
}.
\]

若 $C_r\ge0.05$，认为动作扰动已经有效传递到低层控制。另行记录基线和候选的线速度、角速度饱和比例，以区分控制器不敏感与控制器饱和。

## 预注册门限与决策树

四个checkpoint—种子组合必须先满足：

- 至少100个完整碰撞episode；
- Actor摘要和探针动作不变；
- 环境执行动作与冻结策略采样动作完全一致；
- 非参与车辆对目标碰撞对的影响不超过`1e-6`；
- exp151的$RD$可行动率能够以误差不超过`1e-6`重建；
- 所有指标有限。

随后依次判定：

1. 若8步$U$层可行动率跨组合最小值低于70%，或16步低于60%，判定为`action_quintic_bottleneck`。下一步只允许冻结比较现有动作范围、quintic几何和预测时域，不授权训练。
2. 若第1项通过，但8步控制传递率跨组合最小值低于70%，判定为`low_level_control_bottleneck`。下一步只允许低层控制契约审计，不改变Actor或reward。
3. 若第1、2项通过，但8步$RD$可行动率仍低于70%，判定为`objective_tradeoff_bottleneck`：
   - 若四个组合的`terrain_blocked_rate`均不低于25%，且均比`dmax_blocked_rate`高至少10个百分点，判定地形条件主导；
   - 若四个组合的`dmax_blocked_rate`均不低于25%，且均比`terrain_blocked_rate`高至少10个百分点，判定集合条件主导；
   - 其余情况判定为耦合约束主导。
4. 若$U$、控制传递和$RD$均通过，则与exp151矛盾，状态为诊断无效，只允许修复诊断。

无论结果如何，exp152本身不授权4M、12M或40M。只有最小瓶颈被四个组合一致识别后，才允许形成下一份单变量计划。

## 明确不做

- 不扫描动作范围、候选间隔或容差；
- 不执行候选动作；
- 不修改Actor、Critic、reward、通信、轨迹或控制；
- 不恢复安全投影或预测冲突奖励；
- 不把局部反事实结果作为BC教师；
- 不直接实现新的动作维度、控制器或多目标奖励。

## 产物路径

```text
outputs/runs/exp152_action_planning_controllability_decomposition/
  frozen_exp150_dual_checkpoint_dualseed/
    config/experiment.yaml
    metrics/controllability_decomposition.json
    run_manifest.json
  _suite/
    metrics/suite_summary.json
    run_manifest.json
```

## 当前状态

正式冻结审计已经完成，四个checkpoint—种子组合均通过全部工程门限。联合约束层对exp151的五个时域重建误差均为0；Actor摘要、探针动作、环境执行动作和非参与车辆不变量全部成立。

| checkpoint—seed | 8步无约束 $U$ | 8步地形 $R$ | 8步集合 $D$ | 8步联合 $RD$ | 控制传递率 | 方位角最佳比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t1024—46023 | 0.8436 | 0.7485 | 0.8344 | 0.6840 | 0.9091 | 0.8400 |
| t1024—47023 | 0.8475 | 0.7625 | 0.8240 | 0.7126 | 0.8824 | 0.7993 |
| t2048—46023 | 0.6648 | 0.5909 | 0.6165 | 0.5114 | 0.8419 | 0.8120 |
| t2048—47023 | 0.6676 | 0.5874 | 0.5989 | 0.5100 | 0.8069 | 0.8240 |

16步无约束可行动率分别为`0.8742/0.8768/0.7472/0.7020`，全部超过`0.60`；但8步跨组合最小值只有`0.6648`，低于预注册的`0.70`。因此状态为`action_quintic_bottleneck`。

低层控制不是首要瓶颈：在已经存在无约束避碰轨迹的样本中，8步控制传递率为`0.8069–0.9091`，超过`0.70`；基线线速度饱和率为0，角速度饱和率最高仅`1.29%`。最佳避碰候选约`79.9%–84.0%`来自方位角动作，说明当前局部安全可控性主要由转向维度提供。

地形和集合条件会继续降低可行动率，但没有在预注册决策树中成为首个失败层。t2048相对t1024的8步无约束可行动率下降约18个百分点，表明训练后期策略访问的碰撞状态更难通过当前局部动作—quintic映射修正，而不是控制器无法跟踪已产生的轨迹。

下一步只允许冻结比较现有动作范围、局部扰动覆盖和quintic几何，不授权训练，也不修改动作范围或轨迹生成器。

正式汇总位于：

```text
outputs/runs/exp152_action_planning_controllability_decomposition/
  _suite/metrics/suite_summary.json
  frozen_exp150_dual_checkpoint_dualseed/
    metrics/controllability_decomposition.json
    run_manifest.json
```
