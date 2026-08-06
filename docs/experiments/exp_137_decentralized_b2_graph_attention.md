# exp137：B2 单跳图注意力一次性例外筛选

## 目的

exp135 已证明 pair-repeated 冲突比单步预测冲突更接近实际碰撞结果；exp136 进一步发现两个数据种子的全部失败 episode 都包含重复车辆对冲突，零事件 episode 纳入统计后事件数中位数仍为 `17/18`。因此，B0 的基础收敛失败可能正由 B2 试图解决的动态邻接与重复交互冲突主导，原有“B0/B1 先收敛，B2 才可实现”的条件存在循环前置风险。

本实验仅授权一次 seed23 的 B2 4M 筛选，用于检验这一因果假设。该例外不授权40M训练，不改变正式验收标准，也不允许同时引入GRU、学习消息、新奖励、安全投影或动作后处理。

## 唯一变量

基线固定为 exp125 的 `relative_quintic` 版本。两者保持相同的101维观测、分级通信缓存、Pure RL MAPPO、集中式Critic、奖励、初始状态课程、quintic轨迹和bicycle控制器。

唯一变化是将 `branched_v5` 的36维展平邻居编码器替换为 `branched_v6_graph_attention`：

1. 三个12维邻居消息分别映射为32维节点特征；
2. 32维ego编码形成四个查询头；
3. 每个注意力头的key、query和value维度均为12；
4. 四头输出拼接为48维，保持Actor主干输入宽度不变；
5. 消息质量 \(q\) 与有效邻居掩码参与权重归一化；
6. 图中只有本车和通信缓存内的单跳邻居。

注意力权重定义为：

\[
e_{ih}^{(j)}=
\frac{
\left\langle Q_h e_i,K_h n_{ij}\right\rangle
}{\sqrt{12}},
\]

\[
\alpha_{ih}^{(j)}=
\frac{
m_{ij}q_{ij}\exp\!\left(e_{ih}^{(j)}-e_{ih}^{\max}\right)
}{
\sum_k m_{ik}q_{ik}\exp\!\left(e_{ih}^{(k)}-e_{ih}^{\max}\right)+\varepsilon
},
\]

\[
z_{ih}=\sum_j\alpha_{ih}^{(j)}V_h n_{ij}.
\]

其中，\(m_{ij}\) 仅表示缓存槽位是否有效；无有效邻居时 \(z_{ih}=0\)。发送者索引不进入网络，邻居排列不应改变输出。

## 配置与预算

- 配置：`configs/experiment/exp137_decentralized_b2_graph_attention.yaml`
- 初始化：随机初始化，`bc_updates=0`，`init_checkpoint=null`
- seed：23
- 并行环境：2048
- rollout：64
- 训练时步：2048
- 环境交互：4,194,304
- episode：96 s / 480步
- 对照checkpoint：exp125 `b0_screen_seed23_4m_relative_quintic/checkpoints/best.pt`

## 启动前工程门限

只有以下检查全部通过才启动4M：

- 任意邻居排列下Actor均值输出最大绝对误差不超过 \(10^{-6}\)；
- `q=0` 的无效邻居内容变化不影响输出；
- 无邻居时输出有限且邻居聚合严格为零；
- 固定本车观测与通信缓存后，Oracle、槽位和未发送状态不影响动作与控制；
- B0和B2 checkpoint架构元数据互不兼容，不允许部分加载；
- CPU小环境与CUDA 256环境smoke均无NaN，neighbor encoder发生更新，动作标准差大于 \(10^{-4}\)。

## 4M筛选门限

首先沿用B0的全部基础门限，包括训练dmax降低30%、collision不超过10%、非零成功episode以及terrain contrast两项门限。任一基础门限失败即停止，不启动40M。

若基础门限通过，再与同预算B0进行候选架构比较：

- 远距success至少提高10个百分点，或远距timeout相对下降至少20%；
- 近距success下降不超过5个百分点；
- 近距和远距collision相对B0均不得恶化超过2个百分点；
- 随机屏蔽一个有效邻居后，B2的success下降与collision上升不得同时劣于B0；
- 失败episode的重复冲突事件中位数相对B0至少下降20%。

这些比较使用相同地图种子、通信频率、评测时长与初始状态分布。B0原始4M checkpoint也重新在同一评测入口运行，避免使用不一致的历史口径。

## 严格标准

exp137本身只是4M候选筛选，不以单次checkpoint宣称正式收敛。任何后续40M仍须满足近距和远距：

\[
\mathrm{dmax\ ratio}\le 0.20,
\qquad
\mathrm{success}\ge 0.90,
\]

\[
\mathrm{collision}\le 0.02,
\qquad
\mathrm{timeout}=0.
\]

成功episode还必须通过实际质心平整度、最近邻安全距离、dmax、dispersion和低速保持条件。

## 结果

工程门限全部通过：排列置换的Actor输出最大绝对误差为 \(4.47\times10^{-8}\)，无效邻居泄漏误差和无邻居聚合输出均为0。CPU与CUDA smoke中的neighbor encoder参数改变量分别为 `0.0585/0.0459`，动作标准差分别为 `0.0358/0.0462`，且参数有限、BC更新为0。

4M训练耗时约109.9 s，完成32次shared-joint更新。正式基础gate结果如下：

| 指标 | B0 relative-quintic | B2图注意力 | 门限/判断 |
| --- | ---: | ---: | --- |
| 训练dmax降低比例 | 0.2694 | 0.2079 | B2低于0.30，失败 |
| 独立评测dmax ratio | 0.2047 | 0.1707 | B2改善并通过0.20门限 |
| success | 0.0518 | 0.0439 | 未形成成功改善 |
| collision | 0.0967 | 0.7881 | 明显恶化，远高于0.10 |
| timeout | 0.8516 | 0.1680 | 主要被更早collision终止替代，不能解释为收敛改善 |
| terrain动作MSE | 0.00732 | 0.00121 | 低于0.02，失败 |
| 路径风险降低比例 | -0.00163 | -0.00062 | 低于5%，失败 |

B2通过了数值稳定、参数更新、动作非退化和非零成功episode等工程项，但在训练dmax、collision及两项terrain contrast条件上失败。因此状态为 `stopped_at_base_gate`，未运行远距候选对比，也未启动40M。

## 重复冲突复核

停止后只进行了冻结checkpoint的离线失败归因。使用与exp136相同的 `28023/29023` 数据种子、128环境和512步：

| seed | B0失败episode | B0事件中位数 | B2失败episode | B2事件中位数 | B2 collision失败 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 28023 | 141 | 17 | 146 | 16 | 143 |
| 29023 | 130 | 18 | 135 | 22 | 125 |

两种子中，B2失败episode仍有100%包含pair-repeated冲突。事件中位数相对B0分别变化为 `-5.9%/+22.2%`，未达到预注册的下降20%条件；同时失败构成进一步向collision集中。图注意力没有消除目标冲突，只使几何收缩更激进。

## 产物路径

- run：`outputs/runs/exp137_decentralized_b2_graph_attention/b2_screen_seed23_4m_relative_quintic/`
- suite：`outputs/runs/exp137_decentralized_b2_graph_attention/_suite/`
- 工程gate：`outputs/runs/exp137_decentralized_b2_graph_attention/_suite/metrics/engineering_gate.json`
- 4M gate：`outputs/runs/exp137_decentralized_b2_graph_attention/_suite/metrics/b2_screen_summary.json`
- 重复冲突复核：`outputs/runs/exp137_decentralized_b2_graph_attention/b2_screen_seed23_4m_relative_quintic/diagnostics/failed_episode_repeated_conflicts/metrics/failed_episode_repeated_conflicts.json`

## 停止规则

若工程门限、B0基础门限或候选架构门限任一失败，立即停止B2方向。本轮不通过扩大注意力层数、增加多跳传播、叠加GRU、调整奖励或恢复后处理进行补偿。

## 结论

exp137不能晋级。结果否定了“仅靠排列不变的动态邻居聚合即可解决当前收敛问题”的假设：B2改善了几何收缩，却显著放大碰撞，且没有建立地形路径选择能力。B0、B1和B2均不得进入40M；下一步应回到共享团队advantage下的安全信用表达，而不是继续扩展网络结构。
