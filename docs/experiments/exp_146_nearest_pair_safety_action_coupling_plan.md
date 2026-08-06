# exp146：最近邻成对安全动作耦合审计计划

## 研究动机

exp145已经证明，逐车集合进度和逐车地形结果可以由本车观测与本车动作稳定辨识，但逐车安全进度的平均动作增益只有 `9.58%`，最差验证种子只有 `8.12%`。安全目标的事件覆盖率和方差均已通过，因此失败不能归因于样本过少。最近邻距离由一对车辆共同决定，更可能的原因是：只加入车辆 \(i\) 的动作，无法解释邻车 \(j\) 同时采取的动作。

exp146只检验这一项成对耦合假设。它不修改Actor、环境奖励、通信、执行链路或训练算法，也不重新调整exp145门限。

## 冻结数据与安全目标

继续冻结exp125 `relative_quintic` 的t2048 checkpoint，使用与exp145相同的数据协议：

- 训练：seed `39023`，128环境，480步；
- 验证：seed `40023, 41023`，每个种子64环境，480步；
- 模型初始化：seed `7, 17, 29`；
- Actor观测固定为101维，Actor参数和探针动作必须完全不变。

对每个物理步，在执行动作前选择车辆 \(i\) 的最近邻：

\[
j(i,t)=\underset{k\ne i}{\operatorname{argmin}}
\left\|\mathbf p_{i,t}-\mathbf p_{k,t}\right\|_2.
\]

安全势函数仍固定使用现有 `safety.near_distance=0.72 m`：

\[
\Phi_{i,t}^{s}
=
-\max\left(0,0.72-d_{i,t}^{\mathrm{near}}\right),
\qquad
s_{i,t}=\Phi_{i,t+1}^{s}-\Phi_{i,t}^{s}.
\]

最近邻索引和邻车动作只用于集中训练可行性诊断，不进入Actor观测或执行期消息。

## 对照回归器

四个回归器采用完全相同的两层128单元ELU结构，只改变输入：

\[
\begin{aligned}
F_o &: \mathbf o_i,\\
F_i &: [\mathbf o_i,\mathbf a_i],\\
F_j &: [\mathbf o_i,\mathbf a_j],\\
F_{ij} &: [\mathbf o_i,\mathbf a_i,\mathbf a_j].
\end{aligned}
\]

其中：

- \(\mathbf o_i\) 是车辆 \(i\) 的101维严格去中心化观测；
- \(\mathbf a_i\) 是车辆 \(i\) 的2维动作；
- \(\mathbf a_j\) 是数据采集时最近邻 \(j(i,t)\) 的2维动作。

定义三项动作增益：

\[
I_{ij}
=
\frac{\operatorname{MSE}_{o}-\operatorname{MSE}_{ij}}
{\operatorname{MSE}_{o}},
\]

\[
I_{i\mid j}
=
\frac{\operatorname{MSE}_{j}-\operatorname{MSE}_{ij}}
{\operatorname{MSE}_{j}},
\qquad
I_{j\mid i}
=
\frac{\operatorname{MSE}_{i}-\operatorname{MSE}_{ij}}
{\operatorname{MSE}_{i}}.
\]


其中，\(I_{i\mid j}\) 检查在已知邻车动作后，本车动作是否仍提供独立解释量；\(I_{j\mid i}\) 检查exp145缺失的信息是否确实来自邻车动作。

另将验证集中的邻车动作作确定性乱序，计算：

\[
D_{\mathrm{shuffle}}
=
\frac{\operatorname{MSE}_{ij}^{\mathrm{shuffle}}
-\operatorname{MSE}_{ij}}
{\operatorname{MSE}_{ij}}.
\]

该对照用于排除“仅增加两维输入和参数量就能改善回归”的解释。

## 晋级门限

只有以下条件在两个验证种子上全部满足，才允许形成“训练期最近邻成对安全Critic”计划：

- \(I_{ij}\ge25\%\)；
- \(I_{i\mid j}\ge15\%\)；
- \(I_{j\mid i}\ge15\%\)；
- \(D_{\mathrm{shuffle}}\ge10\%\)；
- 安全目标标准差大于 \(10^{-4}\)；
- 安全目标非零比例不低于8%；
- 最近邻索引不进入Actor输入；
- Actor参数摘要和固定探针动作变化均为0。

门限失败后，不扫描距离阈值、回归器容量、时域或样本重加权，也不启动训练。门限通过也只允许形成训练计划，不能直接启动4M。

## 与既有负结果的边界

exp127拟合的是团队多步return，联合动作在完整episode上没有达到可辨识门限；exp146拟合的是现有0.72 m安全势函数的一步、最近邻成对结果。二者的目标语义不同，因此exp146不能推翻exp127，也不能据此重新引入通用联合动作Critic。

若exp146通过，后续候选只能是训练期、单跳、最近邻成对安全责任估计；执行期Actor仍只读取101维本地观测和通信缓存。该思路与集中训练阶段使用反事实基线分配多智能体信用的原则一致，可参考 [Foerster et al., “Counterfactual Multi-Agent Policy Gradients,” AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11794)，但本轮不实现COMA。

## 明确不做

- 不把邻车动作加入Actor或通信消息；
- 不增加在线安全投影、方向mask或CBS约束；
- 不恢复PPO-Lagrangian；
- 不叠加地形信用、图注意力、GRU或学习消息；
- 不修改0.72 m门限或安全目标定义；
- 不使用碰撞样本过采样、类别加权或辅助损失；
- 不把离线诊断模型作为可部署checkpoint。

## 当前状态

审计已完成，状态为 `stop_pair_local_safety_credit_direction`。

| 指标 | 跨种子平均 | 最差验证种子 | 门限 | 判定 |
| --- | ---: | ---: | ---: | --- |
| 成对动作相对纯观测增益 | 25.25% | 23.93% | 25% | 未通过 |
| 已知邻车动作后的本车增益 | 13.27% | 12.17% | 15% | 未通过 |
| 已知本车动作后的邻车增益 | 14.34% | 14.09% | 15% | 未通过 |
| 邻车动作乱序退化 | 91.36% | 89.07% | 10% | 通过 |

邻车动作乱序后性能大幅退化，说明成对关系是真实信号；但本车和邻车各自的条件边际解释量均未稳定达到门限，且seed40023的成对总增益也不足25%。按照停止规则，不降低门限、不扫描模型容量或时域、不实现成对安全Critic。

机器可读事实源：`outputs/runs/exp146_nearest_pair_safety_action_coupling/_suite/metrics/audit_summary.json`。
