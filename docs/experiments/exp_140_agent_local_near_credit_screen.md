# exp140：逐车非零和近距信用组件筛选

## 目的

exp139表明，现有 \(0.72\,\mathrm m\) 近距势函数在保留逐车raw差分时，对本车动作的验证增益为 `15.791%/14.901%`；逐步跨车中心化后只剩 `8.219%/7.203%`。这说明中心化会删除两辆相互接近车辆共同承担的安全责任，但现有证据仍不足以直接将该信号纳入正式训练。

exp140只进行一次组件级4M筛选，检验非零和逐车近距信用能否在真实PPO更新中减少碰撞和pair-repeated冲突。即使通过，本实验也不授权40M；它只决定是否允许下一步讨论统一的地形—安全信用分解。

## 唯一变量

基线固定为exp125 `relative_quintic`。定义车辆 \(i\) 的最近邻距离为 \(m_i(s)\)，近距势函数为：

\[
\Phi_i^{\mathrm{near}}(s)
=
-\max\!\left(0.72-m_i(s),0\right).
\]

逐车一步信用为：

\[
c_{i,t}^{\mathrm{near}}
=
\Phi_i^{\mathrm{near}}(s_{t+1})
-
\Phi_i^{\mathrm{near}}(s_t).
\]

与exp133/139不同，本实验不执行逐步跨车中心化。对rollout计算：

\[
u_{i,t}
=
c_{i,t}^{\mathrm{near}}
+
\gamma\lambda_c(1-d_t)u_{i,t+1},
\]

并在完整“车辆×时间×环境”联合批次上作一次均值—标准差归一化：

\[
\widehat u_{i,t}
=
\frac{u_{i,t}-\mathbb E[u]}{\operatorname{Std}(u)+10^{-8}}.
\]

Actor使用：

\[
A_{i,t}^{\mathrm{actor}}
=
A_t^{\mathrm{team}}
+0.25\widehat u_{i,t}.
\]

固定 \(\lambda_c=0.95\)。`0.25`与exp126的单一Actor信用尺度一致，只用于一次筛选，不扫描。团队reward、集中式Critic输入、Critic return和执行期策略均保持不变。

逐车信用分配的研究动机参考[Foerster et al., “Counterfactual Multi-Agent Policy Gradients,” AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11794)。势函数差分的形式参考[Ng, Harada and Russell, “Policy Invariance Under Reward Transformations,” ICML 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)。本实验不是COMA，也不宣称逐车Actor-only信用满足经典势函数整形的策略不变性条件；引用只用于说明多智能体责任分解和势函数差分的来源。

本轮不采用CPO或PID-Lagrangian。它们需要新增cost critic、约束预算和乘子动态，会一次改变多个训练机制。相关方法仅保留为未来边界选项：[Achiam et al., ICML 2017](https://proceedings.mlr.press/v70/achiam17a.html)、[Stooke et al., ICML 2020](https://proceedings.mlr.press/v119/stooke20a.html)。

## 保持不变的边界

- 101维严格去中心化Actor观测；
- 12 m分级通信及缓存语义；
- `branched_v5`共享Actor和前馈集中式Critic；
- Pure RL、随机初始化、`bc_updates=0`；
- 96 s / 480步、2048并行环境、rollout 64；
- relative-quintic地形奖励、实际质心平整度奖励和初始状态课程；
- 所有安全投影、末段覆盖、槽位修正和显式目标保持关闭；
- 不使用预测冲突、重复冲突、Oracle、CBS路径或缓存外邻车状态生成信用。

## 启动前工程门限

只有以下检查全部通过才运行4M：

- 新assignment只允许在`shared_joint`下启用；
- 每步raw信用与势函数差分的最大误差不超过 \(10^{-6}\)；
- raw信用的车辆均值等于团队平均近距势函数变化，误差不超过 \(10^{-6}\)；
- 团队reward逐元素完全不变；
- 集中式Critic GAE输入及target语义不变；
- 非零和policy信用不被误标为逐步零和；
- exp126的`terrain_relative_centered`历史行为与零和不变量保持兼容；
- 101维接口、checkpoint兼容性和Oracle执行不变性继续通过；
- CPU小环境与CUDA 256环境smoke无NaN，Actor发生更新且信用trace标准差大于 \(10^{-4}\)。

## 4M组件门限

使用seed23、2048环境、2048训练时步，共4,194,304次环境交互。与相同预算的exp125 `relative_quintic`比较。

exp140只在以下条件全部满足时允许制定后续统一信用计划：

- 无NaN、Inf或梯度异常，Actor、neighbor encoder和terrain encoder均更新；
- 动作标准差大于 \(10^{-4}\)；
- 出现非零成功episode；
- 训练末四分之一dmax相对首四分之一至少降低20%；
- 独立评测collision不高于 `0.0677`，即相对B0的 `0.0967` 至少降低30%；
- 独立评测success不低于 `0.0318`，即相对B0的 `0.0518` 最多下降2个百分点；
- 独立评测dmax ratio不高于 `0.2547`，即相对B0最多恶化0.05；
- 两个冻结诊断种子的失败episode重复冲突事件中位数均相对B0降低至少20%；
- 训练期raw信用激活率不低于8%，最终trace标准差大于 \(10^{-4}\)；
- 团队reward保持误差不超过 \(10^{-6}\)。

terrain-contrast继续报告，但不作为exp140组件门限。原因是本实验只验证安全责任；它无权因安全改善而进入40M，后续统一信用计划仍必须重新满足terrain门限。

## 停止规则

任一门限失败即停止非零和近距信用方向，不扫描scale、trace、距离或归一化方式，不叠加terrain信用、梯度投影、Lagrangian或新网络。通过时也只允许先形成一个单变量的统一信用计划，不直接启动40M。

## 当前状态

已完成，状态为 `stopped_at_component_gate`。不授权40M，也不允许继续扫描非零和近距信用。

## 正式结果

工程gate全部通过。CPU 32环境和CUDA 256环境的256步smoke均完成4次联合更新；最终信用trace标准差约为1，最大团队reward保持误差为0，最大势函数重构误差为 \(2.05\times10^{-8}\)。历史exp126零和信用误差仍为0。

正式训练完成32次shared-joint更新和4,194,304次环境交互，耗时约107.6 s。结果如下：

| 指标 | B0 relative-quintic | exp140 | exp140门限 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 训练dmax降低比例 | `26.94%` | `23.92%` | `>=20%` | 通过 |
| 独立评测dmax ratio | `0.2047` | `0.2397` | `<=0.2547` | 通过 |
| success | `0.0518` | `0.0439` | `>=0.0318` | 通过 |
| collision | `0.0967` | `0.2295` | `<=0.0677` | 失败 |
| timeout | `0.8516` | `0.7266` | 描述性 | 主要被collision替代 |
| raw信用平均激活率 | — | `12.07%` | `>=8%` | 通过 |
| terrain动作MSE | `0.00732` | `0.00246` | 描述性 | 仍无地形响应 |
| 路径风险降低比例 | `-0.163%` | `0.085%` | 描述性 | 仍无地形响应 |

冻结冲突复核结果为：

| seed | B0失败episode重复事件中位数 | exp140中位数 | 相对变化 | 门限 |
| ---: | ---: | ---: | ---: | ---: |
| 28023 | `17` | `19` | `+11.76%` | 至少降低20% |
| 29023 | `18` | `20` | `+11.11%` | 至少降低20% |

exp140没有减少目标冲突，而是同时提高碰撞率和重复冲突。timeout下降不能解释为更快收敛，因为大量episode改为更早collision终止。所有工程不变量、参数更新、信用密度和集合能力保护项均通过，故失败属于方法语义而非实现错误。

## 结论

非零和raw近距势函数trace不能作为后续训练组件。取消逐步中心化虽然恢复了可辨识信号，但该信号没有产生正确的长期避碰行为；单纯提高逐车动作责任并不等于建立安全约束。

按预注册规则：

- 不调整`actor_credit_scale`、`actor_credit_trace_lambda`或`near_distance`；
- 不与terrain信用、梯度投影或图注意力叠加；
- 不启动统一信用计划或40M；
- 如果继续研究训练安全，只能先对“独立cost critic与约束优化是否可估计”做冻结可行性审计，不能直接实施CPO/PID-Lagrangian。

## 产物路径

- run：`outputs/runs/exp140_agent_local_near_credit/local_near_seed23_4m/`
- suite：`outputs/runs/exp140_agent_local_near_credit/_suite/`
- 工程gate：`outputs/runs/exp140_agent_local_near_credit/_suite/metrics/engineering_gate.json`
- 组件gate：`outputs/runs/exp140_agent_local_near_credit/_suite/metrics/component_screen_summary.json`
- 冲突复核：`outputs/runs/exp140_agent_local_near_credit/local_near_seed23_4m/diagnostics/failed_episode_repeated_conflicts/metrics/failed_episode_repeated_conflicts.json`
