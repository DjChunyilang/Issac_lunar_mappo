# exp141：真实碰撞cost value冻结可估计性审计

## 目的

exp140说明，逐车近距势函数差分即使恢复动作辨识信息，也会增加真实碰撞和重复冲突。因此不再把安全表示为新的Actor奖励整形。exp141只检验约束优化的必要前提：在冻结B0策略下，现有集中式state是否能稳定预测未来一个rollout内的真实碰撞事件。

本实验不新增cost critic、乘子或训练损失；诊断分类器不进入checkpoint、Actor、环境或执行链路。

## 目标与模型

令 \(z_t\) 为当前54维集中式Critic state，\(e_t\) 表示该步是否以真实车辆碰撞终止。固定时域 \(H=64\)，定义：

\[
y_t^{\mathrm{collision}}
=
\mathbb I\!\left[
\exists k\in\{0,\ldots,H-1\}:e_{t+k}=1
\right].
\]

目标窗口在episode终止时截断，禁止跨reset传播标签。拟合两层128单元ELU分类器：

\[
V_c(z_t)=\sigma\!\left(f_\theta(z_t)\right).
\]

训练使用带正类权重的BCE；评测前从logit中减去正类权重的对数，以恢复原始训练分布先验。比较对象是只输出训练集碰撞窗口发生率的常数模型。

该审计只检查cost value能否降低稀疏碰撞回报的估计方差，不宣称它能够解决逐车责任。约束优化的理论边界参考[Achiam et al., “Constrained Policy Optimization,” ICML 2017](https://proceedings.mlr.press/v70/achiam17a.html)和[Stooke et al., “Responsive Safety in Reinforcement Learning by PID Lagrangian Methods,” ICML 2020](https://proceedings.mlr.press/v119/stooke20a.html)。本项目若后续采用PPO-Lagrangian，也不能宣称等同于CPO。

## 正式协议

- 策略：冻结exp125 `relative_quintic` seed23 best；
- 时域：64步；
- 训练：512环境×512步，seed39023；
- 验证：两个独立种子40023、41023，各256环境×512步；
- 模型种子：7、17、29；
- 网络：两层128单元ELU；
- 训练：30 epochs、batch 4096；
- 输入只含当前集中式state，不含joint action、Oracle点、CBS路径或未来信息；
- Actor参数和固定探针动作必须完全不变。

## 晋级门限

只有以下条件全部满足，才允许制定一次PPO-Lagrangian计划；本实验本身不授权训练：

- 训练集真实collision episode不少于30；
- 每个验证种子真实collision episode不少于20；
- 每个验证种子的正标签率不低于0.5%；
- 三个模型种子的平均AUROC在每个验证种子上均不低于0.75；
- 平均AUPRC在每个验证种子上均不低于该种子正标签率的3倍；
- 相对常数发生率模型的Brier score改善在每个验证种子上均不低于15%；
- Actor参数摘要和探针动作变化均为0；
- 目标构造通过终止截断测试，不跨episode泄漏。

## 停止规则

任一门限失败，则不实现cost critic、CPO、PPO-Lagrangian或PID乘子，也不扫描预测时域、网络容量和正类权重。通过时只允许先写单一约束优化计划，仍需明确cost定义、预算、乘子更新和4M停止门限。

## 当前状态

已完成，状态为 `allow_lagrangian_plan_only`。本实验只授权制定一次PPO-Lagrangian组件计划，不直接授权训练或40M。

## 正式结果

训练集包含229,888个窗口样本、482个真实collision episode，正标签率为 `12.368%`。两个验证种子各包含114,944个样本，结果如下：

| 指标 | seed40023 | seed41023 | 门限 |
| --- | ---: | ---: | ---: |
| collision episode | `247` | `228` | `>=20` |
| 正标签率 | `12.663%` | `11.637%` | `>=0.5%` |
| 三模型平均AUROC | `0.9392` | `0.9331` | `>=0.75` |
| 三模型平均AUPRC | `0.6231` | `0.5792` | `>=3×正标签率` |
| Brier改善 | `35.59%` | `30.00%` | `>=15%` |

目标构造未跨reset，Actor参数摘要和固定探针输出变化均为0。全部预注册门限通过，说明现有集中式state足以为真实collision约束提供低方差cost value基线。

## 结论

允许制定一次PPO-Lagrangian组件筛选计划，但结论边界保持为：

- 不采用在线安全投影或方向性mask；
- 不把近距势函数、预测冲突或MAPF标签写入cost；
- cost只使用真实collision终止；
- cost critic只在集中训练期使用，不进入Actor或执行链路；
- 该结果不证明约束策略一定收敛，也不授权40M。

## 产物路径

- `outputs/runs/exp141_collision_cost_value_feasibility/frozen_exp125_seed23/`
- 正式指标：`outputs/runs/exp141_collision_cost_value_feasibility/frozen_exp125_seed23/metrics/collision_cost_value_feasibility.json`
