# 当前状态

更新时间：2026-08-13。

## 当前主线

当前低层基线来自 `exp156`，正在通过`exp157`拆分验证共同选址与低层可达性：

```text
H0冻结信息/原语审计
→ H1 407维共同站点区域条件观测
→ N1共享多尺度CNN Categorical Actor（诊断基线）
→ 47维差速轨迹原语
→ 时标化位姿轨迹
→ 左右轮差速控制
```

保持Pure RL、shared-joint MAPPO和CTDE；BC、安全投影、方向性mask、集中式动作修正、在线MAPF、DAE、GRU、GNN和可学习通信均未启用。

## exp155状态

`exp155`已停止。N0在停止命令到达前恰好完成153,600训练时步，但其动作和评测接口已被否决，因此不进入架构排名或strict汇总。N1/N2没有启动。

N0产物保留在：

```text
outputs/runs/exp155_full_rl_ablation/n0_seed23_full_2400iter/
```

其run manifest标记为 `lifecycle_status: stopped_design_revision`，checkpoint状态仍为 `candidate`，且 `eligible_for_suite: false`。

该run的最终诊断为success 0、collision约0.0495、timeout约0.9505、dmax ratio约0.141，确定性hold比例约0.780。该结果只作为旧动作空间hold塌缩证据。

## exp156已完成工程

- 47维动作：hold、39个前进、3个倒车、2个原地转向和2个S形让行；
- 轨迹新增运动方向、计划航向变化和原语类型；
- 差速代理采用Jackal的 $r=0.098\ \mathrm m$、$b=0.376\ \mathrm m$ 和 $18\ \mathrm{rad/s}$ 轮速限制；
- 295维Actor观测和17维分级通信；
- v10 ego移除世界平移/旋转捷径，速度转换到车体坐标系；
- 950维统一多尺度Critic；
- 主线Oracle奖励权重为0；
- 队形整体旋转和每车初始航向独立随机；
- 熵衰减覆盖完整153,600训练时步；
- 记录归一化熵、有效动作数、hold概率和四类动作族概率；
- checkpoint明确区分295/47/Categorical与旧291/40/Gaussian接口；
- 生成六分层、每层192个episode的固定配对场景清单；
- 实现Clopper–Pearson、dmax bootstrap、IQM和配对bootstrap统计；
- 增加活动文档公式分隔符静态检查。

三种Actor参数量为：

| 结构 | 参数量 |
| --- | ---: |
| N0 | 83,343 |
| N1 | 106,591 |
| N2 | 31,752 |

均低于12万参数门限。

## 已完成验证

- exp156专项测试全部通过；
- 历史exp155多尺度和分级通信回归测试通过；
- Oracle、槽位和集中式诊断量不改变轨迹、控制与轮命令；
- 平坦场景整体SE(2)变换后，Actor观测和logits保持不变；
- CPU真实shared-joint MAPPO smoke通过；
- N0、N1、N2的CUDA 256环境真实MAPPO smoke均通过；
- 12个冻结场景的47原语覆盖审计通过，其中5个来自exp155后期全hold状态；
- 覆盖审计的无碰撞解除率为100%，倒车、原地转向和S形让行均参与过有效联合解；
- 1152场景 `scenario_manifest` 已生成。

Smoke只证明工程链路和梯度更新有效，不代表策略收敛。

## exp156训练结论与exp157诊断

`exp156` 当前没有训练进程。N0和N1均完成2400 iterations、39,321,600环境交互及同一1152场景清单的六分层配对评测：

| 结构 | success | collision | timeout | dmax ratio | strict分层 |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0 | 0 | 0.0729 | 0.9271 | 0.2242 | 0/6 |
| N1 | 0.0017 | 0.0130 | 0.9852 | 0.2367 | 0/6 |

两种结构均处于成功率地板，不能据此作有效的网络优劣排名。N1仅因保留多尺度空间结构且collision较低，被指定为后续接口研究的**临时CNN基线**；其checkpoint仍为 `candidate`，不是推荐策略、strict通过结果或最终架构。

N2完整训练已取消。原N2 run在第一次PPO更新时因非连续张量进入CUDA `grid_sample` 而退出；该问题已通过在采样前显式物化连续特征图和网格修复。修复后完成256环境、64训练时步的真实CUDA smoke，4次Actor/Critic更新均完成，参数变化非零且无NaN。smoke产物位于：

```text
outputs/runs/exp156_smoke/cuda_n2_contiguous_fix_256env_64step/
```

N2仍只保留为工程可运行候选，不再投入39,321,600交互，也不进入exp156架构排名。运行状态以以下文件为准：

```text
outputs/runs/exp156_differential_multiscale_ablation/_suite/suite_status.json
```

当前尚不存在exp156推荐checkpoint、六分层strict pass或正式动画。Oracle奖励消融、seed31/47和新的信用算法均暂停；下一项决策先验证共同选址与低层到达能否被分离，而不是继续扩大编码器比较。

“共同平地选择—终端稳定”耦合问题的候选路线为 `D-STC`（Decentralized Site-and-Trajectory Commitment）：有限本地地形 belief、分布式站点承诺、goal-conditioned Actor 与短时轨迹承诺。`D-STC` 是本项目对多篇研究原则的组合命名，不是已有论文中可直接复现的标准算法，也尚未决定采用。

`exp157` H0已完成1152个固定场景审计。12 m通信图连通率为100%，但“至少一辆车初始观测到共同站点且可传播证据”的总体比例仅40.97%，近距Bottleneck为0；这说明通信连接不等于站点信息已经可用。47维动作审计仍为12/12场景存在联合解，站点势场SE(2)误差为 $4.17\times10^{-7}$。H0尚未实现候选关联和commit，因此不能声称去中心化共同选址已就绪。

H1使用407维空间站点势场、N1 CNN和相同39.3M Pure RL预算，作为“共同可行区域已知时低层能否完成任务”的受控上界。H1不是最终去中心化策略，其结果只用于决定是否值得实现完整D-STC。详见[exp157实验记录](experiments/exp_157_site_belief_diagnostic.md)与[共同选址综述](references/decentralized_site_trajectory_coordination_review.md)。

完整接口、预算和验收方法见[实施计划](implementation_plan.md)。
