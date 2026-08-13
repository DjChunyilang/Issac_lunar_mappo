# 去中心化共同选址与终端协调文献检索日志

检索日期：2026-08-13。

本日志对应[研究综述](decentralized_site_trajectory_coordination_review.md)和[证据矩阵](decentralized_site_trajectory_coordination_evidence_matrix.csv)。研究与正在运行的 `exp156` 并行进行，没有暂停训练或修改训练配置。

## 1. 研究问题

- RQ1：在没有中央目标广播的情况下，多车如何由私有局部地形形成同一终端区域？
- RQ2：分级、异步和陈旧通信下，哪些分布式一致方法仍具有可迁移意义？
- RQ3：共同站点已形成后，如何表达短时未来意图并避免终端同时动作歧义？
- RQ4：应继续使用端到端 MAPPO，还是改为 goal-conditioned 层级策略或分布式优化？
- RQ5：哪些信息可以在严格去中心化执行中合法传播，哪些属于 Oracle 或集中式泄漏？
- RQ6：怎样将候选选择和终端稳定作为一个联合问题研究，同时避免继续叠加互不相关的模块？

## 2. 来源与质量边界

直接检索和核验的正式来源包括：

- NeurIPS Proceedings；
- PMLR 的 ICML、AISTATS 和 CoRL 正式论文页；
- ICLR/OpenReview 的正式录用页；
- Robotics: Science and Systems 正式论文集；
- IEEE Xplore、IEEE DOI 和作者机构正式存档；
- Elsevier DOI 页面；
- 项目内历史实验文档和机器可读运行状态。

核心论据优先使用 NeurIPS、ICML、ICLR、RSS、CoRL、IEEE Transactions on Robotics 和 IEEE Robotics and Automation Letters 的正式论文。经典分布式优化、任务分配与 options 论文不受年份限制，但只承担基础理论作用。

当前环境没有订阅版 Web of Science、JCR 或中科院分区数据库访问权限。因此：

- 不声称已经查询不可访问的付费数据库；
- 不根据非官方转载推断中科院或 JCR 分区；
- 期刊分区无法实时核验时，在证据矩阵中明确标注；
- 高水平会议依据正式会议论文集，而不是引用量或搜索排名认定。

## 3. 检索式

### Q1：共同知识、部分可观测和去中心化协调

```text
("decentralized multi-agent" OR "Dec-POMDP")
AND ("common knowledge" OR consensus OR agreement)
AND (communication OR local observation)
```

### Q2：分布式地图、belief和异步共识

```text
("multi-robot" OR "multi-agent")
AND (distributed mapping OR collaborative mapping OR local belief)
AND (consensus OR ADMM OR asynchronous OR sparse communication)
```

### Q3：共同站点、rendezvous和任务分配

```text
("multi-robot" OR "multi-agent")
AND (rendezvous OR "goal selection" OR "task allocation")
AND (decentralized OR distributed OR local communication)
```

### Q4：轨迹承诺和多车时空协调

```text
("multi-agent trajectory" OR "multi-robot trajectory")
AND (decentralized OR asynchronous OR distributed)
AND (commitment OR communication delay OR pairwise constraints)
```

### Q5：长时层级策略和目标条件策略

```text
("goal-conditioned policy" OR hierarchical reinforcement learning OR options)
AND (planning OR navigation OR multi-robot)
```

### Q6：行星平地与安全站点

```text
(planetary OR lunar OR terrain)
AND ("safe landing site" OR "site selection" OR flat region)
AND (detection OR optimization OR traversability)
```

### Q7：反方路线

```text
(GNN OR GRU OR attention OR learned communication OR roles)
AND (decentralized MARL OR multi-robot)
AND (coordination OR goal OR navigation)
```

## 4. 引文追踪种子

以下论文用于前向、后向和相邻领域追踪：

- MACKRL；
- Foerster et al. 的学习通信；
- RACER；
- ROAM；
- iMESA；
- RAMEN；
- CBAA/CBBA；
- MADER 和 RMADER；
- 分布式协方差控制；
- Model-Based RL for Decentralized Multiagent Rendezvous；
- Planning with Goal-Conditioned Policies；
- Active Neural SLAM。

最终证据矩阵纳入 31 篇正式论文，覆盖公共知识、学习通信、分布式地图与一致估计、平地候选、任务分配、轨迹协调、形成稳定和 goal-conditioned 学习七类证据。矩阵不是引用数量排名，而是与当前两个耦合问题的项目适配表。

## 5. 三组独立研究与交叉评审

本轮调用三个独立子研究方向，并要求其分别提出方案和反驳：

### A组：MARL与共同目标形成

重点检索公共知识、分布式约束交集、局部图策略和学习通信。结论是：最强路线为带物理语义的候选 belief 与分布式共识；GRU、GNN和无语义 latent message 不能恢复未传播的站点信息。

### B组：多机器人轨迹与终端稳定

重点检索 MADER/RMADER、分布式MPC、ADMM、形成控制、CBF和ORCA。结论是：当前端点摘要不足以表示连续时空占据；最小改进是共享上一规划步已经承诺的短时轨迹。CBF、ORCA或MPC若作为 Actor 后覆盖会违反当前执行契约。

### C组：地形belief、候选站点与层级决策

重点检索行星站点检测、主动地图、候选关联、拍卖、共识和 options。结论是：候选检测、候选验证分工与最终 all-to-one 站点承诺必须区分；CBBA 适合验证分工，不适合直接证明共同站点一致。

根代理将三组建议压缩为一个共同假设：`D-STC`，即高层站点承诺和低层短时轨迹承诺共享相同的“本地生成、邻居复制、版本化、带年龄”的设计原则。

## 6. 纳入标准

论文需至少满足一项：

- 给出严格去中心化或分布式执行机制；
- 讨论局部通信、陈旧消息或异步一致；
- 处理多机器人共同目标、任务分配、地图融合或终端形成；
- 处理已承诺轨迹、两两轨迹约束或分布式滚动优化；
- 给出可迁移到 goal-conditioned 层级策略的正式算法。

同时必须能够明确区分论文事实与项目推断。单机器人站点检测论文可支持“区域检测与选择分离”，但不能用于证明多车共识；多车地图一致论文可支持本地 belief 和共识原则，但不能直接证明月面集合成功。

## 7. 排除或后置标准

以下方向不进入首轮实现：

- 只有 arXiv 或 Workshop、未核实正式录用的论文；
- 执行期需要中央服务器、全局地图或联合动作选择的方法；
- 把中央搜索点换一个名称后广播给 Actor；
- 没有消息语义、版本或边界审计的无约束 latent communication；
- 在 Actor 后执行 CBF-QP、ORCA、CBS或集中式MPC覆盖；
- 同时叠加 DAE、GRU、GNN、options 和学习消息；
- 直接把凸 ADMM 收敛结论套到非凸、不连通的月面候选区域；
- 把任务分配拍卖的收敛性误写为 all-to-one 共同站点共识。

## 8. 仓库证据核对

研究同时核对了 `exp063`、`exp097`、`exp109`、`exp112`、`exp125` 和当前 `exp156`：

- terrain-aware Oracle 可行不代表 Actor 能到达同一平地；
- 延长 episode 只能消除部分慢收敛，不能解决残余平整度与几何耦合；
- 更强平地搜索会增加行程并损害几何收紧；
- 更早终端修正可能使车辆离开平整区域；
- 大多数 timeout 不是只差连续 hold；
- N0 的 96/144/192 秒配对诊断没有出现成功，说明当前 exp156 N0 的主要问题不是时域边界。

这些项目事实用于决定研究优先级，不替代论文的外部证据。

## 9. 可复核性和更新规则

- 论文条目、正式来源和项目结论见[证据矩阵](decentralized_site_trajectory_coordination_evidence_matrix.csv)。
- 公式、架构、路线评分和停止条件见[研究综述](decentralized_site_trajectory_coordination_review.md)。
- 后续新增论文必须记录正式出处、解决问题、执行期信息和对项目的限制。
- 若能访问官方 JCR 或中科院数据，可另行补充分区年份；不得用当前年份的非官方网页倒推历史分区。
- 当前训练结束前，不把 `D-STC` 写入 `docs/technical_design.md`，也不把它描述为已采用架构。
