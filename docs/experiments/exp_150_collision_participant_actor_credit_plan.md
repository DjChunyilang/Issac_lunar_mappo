# exp150：真实碰撞参与者 Actor 信用组件筛选计划

## 目的

exp149 表明典型碰撞只涉及四辆车中的一对，但现有 shared-joint MAPPO 将同一团队碰撞终止回报复制给所有车辆。碰撞对在终止前 8/16 步的 repeated recall 约为 `99.7%–100%`，参与车辆的实际最近邻闭合量也显著高于未参与车辆。

本实验只改变训练期 Actor 的碰撞终止信用分配，不改变环境 reward、集中式 Critic、Actor结构、观测、通信、轨迹、控制、终止条件和评测协议。

## 信用定义

碰撞终止步 $t$ 的实际参与车辆集合为 $\mathcal A_t$，参与车辆数为 $m_t$，车辆总数为 $N=4$。当前碰撞相关团队终止贡献的绝对幅值为：

\[
C
=
w_{\mathrm{safety}}c_{\mathrm{collision}}
+w_{\mathrm{terminal}}c_{\mathrm{failure}}
=100+55=155.
\]

原团队信用对每辆车均为 $-C$。保持逐车均值不变的目标分配为：

\[
r_{t,i}^{\mathrm{allocated}}
=
-C\frac{N}{m_t}
\mathbb I[i\in\mathcal A_t].
\]

进入 Actor 的即时零和残差为：

\[
c_{t,i}
=
r_{t,i}^{\mathrm{allocated}}-(-C)
=
-C\left(
\frac{N}{m_t}\mathbb I[i\in\mathcal A_t]-1
\right).
\]

非碰撞步取 $c_{t,i}=0$。因此：

\[
\sum_{i=1}^{N}c_{t,i}=0,
\qquad
\frac{1}{N}\sum_i
\left(-C+c_{t,i}\right)=-C.
\]

该残差只使用碰撞终止后的实际位置和既有 `collision_distance=0.28 m` 判断参与车辆。若碰撞终止却没有检测到参与车辆，训练必须立即报错，不能回退到全队信用。

## Actor advantage

对残差使用与现有 MAPPO rollout 一致的有限时域 trace：

\[
G^{c}_{t,i}
=c_{t,i}
+\gamma\lambda_c(1-d_t)G^{c}_{t+1,i},
\qquad
\lambda_c=0.95.
\]

四辆车联合标准化后，以固定系数 $\alpha=0.25$ 加入 Actor advantage：

\[
A^{\mathrm{actor}}_{t,i}
=A^{\mathrm{team}}_t
+0.25\operatorname{Norm}(G^{c}_{t,i}).
\]

`0.25` 与此前单组件 Actor credit 审计保持一致，本轮不扫描。集中式 Critic 仍只学习原团队 return；环境返回给四辆车的 reward 必须保持完全相同。

## 唯一配置差异

配置从 exp148 继承，只允许增加：

```yaml
experiment.name: exp150_collision_participant_actor_credit
algorithm.training_semantics: exp150_collision_participant_actor_credit
algorithm.actor_credit_assignment: collision_participant_centered
algorithm.actor_credit_scale: 0.25
algorithm.actor_credit_trace_lambda: 0.95
algorithm.actor_credit_gradient_mode: additive_advantage
```

继续固定：

```yaml
bc_updates: 0
init_checkpoint: null
collision_constraint_enabled: false
```

## 工程门限

训练前必须验证：

- 非碰撞步信用严格为0；
- 单一碰撞对时参与车辆残差为 `−155`、未参与车辆为 `+155`；
- 多车辆碰撞时按 $N/m_t$ 自动缩放；
- 每个环境步残差和绝对值不超过 $10^{-5}$；
- 分配后逐车均值与原团队碰撞贡献误差不超过 $10^{-5}$；
- 环境 reward、Critic state/return、Actor动作和执行控制在安装信用包装前后完全一致；
- predicted/repeated conflict、Oracle、槽位和未通信状态不进入信用计算；
- CPU小环境与CUDA 256环境 smoke 无NaN，信用 trace 非退化；
- exp148 时间一致性、101维观测和严格去中心化测试继续通过。

## 4M筛选

工程门限通过后，从随机初始化运行唯一一次 seed23 4M：

```yaml
episode_duration: 96 s
parallel_envs: 2048
rollout_length: 64
training_timesteps: 2048
environment_interactions: 4194304
```

必须同时通过原 B0 全部门限：

- 无NaN、Inf或梯度异常；
- Actor、neighbor encoder、terrain encoder均有效更新；
- 动作标准差大于 $10^{-4}$；
- 训练末四分之一平均dmax相对首四分之一降低至少30%；
- 出现非零success episode；
- 独立评测collision不超过10%；
- terrain contrast动作MSE大于0.02；
- 正常地形下路径风险比地形置零对照低至少5%。

并额外满足：

- Actor信用在训练中实际激活；
- 信用逐步零和误差不超过 $10^{-5}$；
- 团队reward保持误差不超过 $10^{-6}$；
- source reconstruction误差不超过 $10^{-5}$；
- 双种子失败episode的重复冲突中位数均低于exp148的5；
- 不得以timeout替代collision：timeout不得高于exp148超过10个百分点。

任一条件失败则停止，不启动40M，不扫描 $\alpha$、$\lambda_c$、collision penalty、failure penalty或参与判定距离。

## 明确不做

- 不把predicted/repeated conflict写入reward或Actor输入；
- 不增加cost critic、Lagrangian乘子、GNN、GRU或注意力；
- 不恢复安全投影、方向mask、槽位目标或后处理；
- 不修改地形reward、集合reward或控制参数；
- 不与exp126、exp140或exp142机制组合；
- 不迁移exp148 checkpoint。

## 当前状态

工程实现、CPU/CUDA smoke和唯一一次seed23 4M组件筛选均已完成。工程门限全部通过：信用只在真实碰撞终止时激活，环境团队reward保持误差为0，逐步零和误差最大为 $9.54\times10^{-7}$，且Actor、neighbor encoder和terrain encoder均发生有效更新。因此以下性能失败不是信用公式未生效或工程链路中断造成的。

4M训练中，dmax首末四分之一由 `7.2509` 降至 `2.9567`，降幅为 `59.22%`，但没有产生任何success episode。独立评测结果为：

\[
\text{dmax ratio}=0.2756,\qquad
\text{success}=0,
\]

\[
\text{collision}=0.9990,\qquad
\text{timeout}=0.0010.
\]

terrain contrast动作MSE仅为 `0.000810`，正常地形相对置零地形的路径风险改善为 `0.0466%`，均未达到门限。两个冻结诊断种子的失败episode重复冲突中位数分别为 `9` 和 `8`，高于exp148基准的 `5`，没有满足“两个种子均下降”的必要条件。

因此状态为 `stopped_at_4m_gate`，`forty_m_authorized=false`。真实碰撞参与者信用虽然能够在数值上无偏地区分参与与未参与车辆，但没有形成安全集合或地形相关路径；本方向停止，不扫描 $\alpha$、$\lambda_c$、碰撞惩罚、失败惩罚或参与判定距离，也不与其他信用、网络或约束模块组合。

正式汇总位于：

```text
outputs/runs/exp150_collision_participant_actor_credit/
  _suite/metrics/engineering_gate.json
  _suite/metrics/component_screen_summary.json
  participant_credit_seed23_4m/metrics/component_gate.json
```
