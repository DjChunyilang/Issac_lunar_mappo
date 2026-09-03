# 总体架构规划 v3：严格去中心化站点证书与差速轨迹原语

## 1. 系统边界

当前已部署的低层执行数据流保持：

```text
本车状态、多尺度局部地形和分级通信缓存
→ 共享Categorical Actor
→ 47维差速轨迹原语
→ 时标化quintic或定时航向轨迹
→ 左右轮命令
```

Actor输出之后不存在集中式修正。Oracle、真实全队状态、集中式Critic、MAPF诊断和未通信状态均不得进入执行链。

`exp160/161`验证后的Active-DSTC候选链为：

```text
任务特定frontier探索与有限候选belief
→ 12 m图上的proposal摘要转发及必要重连
→ 保守物理关联与四车全签site commit
→ 去中心化47维原语best response
→ 差速控制
```

截至2026-08-20，静态proposal/commit、有限轮候选共识和12个死锁场景原语优化分别通过组件门限，但完整闭环尚未实现。R4若采用，将替换Pure RL Actor，而不是接在Actor之后修改动作。候选源车、转发车辆和接收车辆均不得读取Oracle或集中式地图。

训练采用CTDE：strict共享Actor只读取本地295维观测；H1诊断Actor读取407维局部站点势场观测；统一Critic读取950维集中式状态。Critic和DAE奖励模型均不随Actor部署到车辆。

## 2. Actor观测

总维度为：

$$
15+3\times17+224+5=295.
$$

### 2.1 Ego分支

v10基础ego采用车体坐标系速度，绝对平面位置和绝对世界航向槽位被规范化，不形成朝向世界原点的捷径。基础10维之后加入：

$$
[\Delta x_b^{\mathrm{plan}},\Delta y_b^{\mathrm{plan}},
v^{\mathrm{plan}},\Delta\psi^{\mathrm{plan}},z_i].
$$

### 2.2 Neighbor分支

每车最多读取3个17维缓存消息。12 m内信息完整；12 m外只保留低频位置和航向，速度、地形、轨迹摘要及协调令牌全部清零。Actor不得根据sender索引建立固定身份策略。

### 2.3 Terrain分支

三个车体坐标系网格覆盖0.2 m、0.4 m和0.8 m三种分辨率，最大前向范围4.0 m。每点包含相对高度和风险，合计224维。该范围覆盖47个原语的终点和前向预判区域，不构建全局地图。

### 2.4 Aggregation分支

5维统计只由本车通信缓存计算，不查询全队实时状态。它描述有效邻居比例、缓存距离、消息质量和消息年龄，不包含实时全局dmax或dispersion。

## 3. Actor状态

N0、N1和N2保持相同分支输出：ego 32维、neighbor 48维、terrain 64维、aggregation 16维。

- N0使用展平地形MLP；
- N1使用共享多尺度CNN及空间池化；
- N2使用共享多尺度CNN和13条前进quintic路径的路径条件采样，所有47个动作同时读取三尺度地形上下文。

三种结构均使用47维Categorical logits。N0/N1完整训练均出现成功率地板；N2在修复CUDA非连续采样网格后只完成工程smoke，完整训练已取消。当前临时采用N1共享多尺度CNN作为后续诊断的统一Actor接口，不将其视为strict通过或最终架构。暂不加入GRU、GNN或可学习消息，避免把共同选址、动作接口和信用改造同时叠加。

## 4. 动作与轨迹

47个动作由1个hold、39个前进、3个倒车、2个原地转向和2个S形让行组成。轨迹携带 `primitive_type`、`motion_direction` 和 `planned_yaw_delta`。

平移动作使用quintic位姿轨迹。倒车的几何切向沿车体后方，但车体航向不被改写为路径切向；S形让行动作的末端切向与起始车体方向平行。原地转向的位置恒定，航向在定时时域内沿最短角度插值。

## 5. 差速控制

左右轮速度为：

$$
\omega_L=\frac{v-\frac{b}{2}\omega}{r},
\qquad
\omega_R=\frac{v+\frac{b}{2}\omega}{r}.
$$

参数为 $r=0.098\ \mathrm m$、$b=0.376\ \mathrm m$，轮速上限为 $18\ \mathrm{rad/s}$。代理环境先独立裁剪左右轮，再由裁剪结果反算有效车体速度，避免代理执行超出PhysX接口能力。

## 6. 集中式Critic

Critic输入为：

$$
54+4\times224=950.
$$

每辆车的8维全局运动状态和224维多尺度地形分别编码为32维，再融合为32维。四车融合特征经mean/max聚合，与团队、地形摘要和Oracle分支拼接。该结构只改善训练期价值估计，不改变Actor可观测信息。

## 7. 训练与执行信息边界

| 信息 | Actor | 通信消息 | Critic | 诊断/离线评测 |
| --- | --- | --- | --- | --- |
| 本车局部状态与地形 | 是 | 发送端摘要 | 是 | 是 |
| 上一规划步已承诺轨迹 | 本车可见 | 12 m内可见 | 可见 | 是 |
| Oracle集合点 | 否 | 否 | 是 | 是 |
| 全局质心、dmax、dispersion | 否 | 否 | 是 | 是 |
| 未发送邻车真实状态 | 否 | 否 | 是 | 是 |
| MAPF/CBS路径或优先级 | 否 | 否 | 否 | 仅离线 |

exp158在训练侧增加一个可选DAE分支：标准GAE与DAE共享完全相同的Actor、Critic、环境奖励和执行接口。DAE奖励模型只读取集中状态、联合动作和查询车辆索引，批量估计47种候选动作的团队即时奖励。该输出只构造逐车advantage，不写入Actor观测、通信缓存或控制器。

exp159新增另一个互斥训练语义 `analytical_prd_loo`。它不加载DAE reward model，也不增加Critic head；只从车辆 $i$ 的raw团队advantage中减去当前步、仅由其他车辆状态和动作解析计算的LOO基线。团队reward和Critic return保持原值。GAE、DAE和ALO-PRD三种语义不得叠加。

## 8. 可信评测层

正式评测使用固定的1152场景清单：六个距离/拓扑分层各192个episode。清单保存初始位置、航向、地形运行参数和内容哈希，所有架构必须精确复现。

每层使用Clopper–Pearson单侧置信界和dmax ratio单侧bootstrap上界验收，并报告IQM与架构间配对bootstrap差异。64 episode中间评测只用于诊断。

## 9. 候选技术的后置条件

- Active-DSTC：exp163在修复内部可行域的Bottleneck上通过1152场景DISCOVER/VERIFY/EXCHANGE/COMMIT门限；下一步只剩delta通信和R4 GATHER完整闭环。exp163尚未通过最终dmax/success gate；

- DAE：工程接口已在exp158实现；只有冻结因果和奖励可辨识性门限通过后才允许H1完整训练，H1三seed通过后才允许295维strict训练；
- ALO-PRD：exp159工程接口与无偏性检查通过，但双seed梯度方差只降低7.06%/0.42%，A-H1门限失败，当前不训练；
- GRU：仅当失败与合法历史消息中的可恢复时序信息相关时考虑；
- GNN/注意力：仅当固定展平邻居对动态邻接或排列鲁棒性构成瓶颈时考虑；
- 可学习通信：仅当固定17维物理消息消融确认通信内容本身构成瓶颈时考虑。

GRU、GNN、注意力和学习通信不得与exp158同时启用。DAE在H1中只验证共同站点已知条件下的低层信用，不把H1结果解释为严格去中心化共同选址已经解决。
