# 多智能体信用分配文献检索日志

检索日期：2026-08-12。

本日志记录[信用分配综述](marl_credit_assignment_review.md)的检索范围、筛选过程、质量边界和可复核来源。检索结果不是根据引用量自动排序，最终结论以正式论文原文、项目适配性和CTDE边界为准。

## 1. 研究问题

- RQ1：共享团队 advantage 是否会掩盖单车边际贡献？
- RQ2：个体、车辆对、子团队和全团队信用分别适用于什么任务结构？
- RQ3：96秒/480步任务是否需要专门的时间信用方法？
- RQ4：哪些方法可以保留本地 Actor 的严格去中心化执行？
- RQ5：在四辆车、40维离散动作、共享 Actor 和 MAPPO 下，最小可信改造是什么？

## 2. 检索来源

直接检索和核验的来源包括：

- PMLR 的 ICML、AISTATS、L4DC 正式论文页；
- NeurIPS Proceedings；
- OpenReview 和 ICLR 正式会议记录；
- AAAI Proceedings；
- ACM/DOI 的 KDD 论文记录；
- IEEE Xplore 元数据、IEEE DOI 和作者机构正式存档；
- AAMAS/IFAAMAS 正式论文集；
- Springer、Frontiers 和期刊 DOI 页面；
- OpenAlex 仅用于候选发现和去重，不用于判定论文质量。

当前执行环境没有订阅版 Web of Science、JCR 或中科院分区数据库访问权限。因此：

- 不声称已经执行不可访问数据库中的检索；
- 不根据非官方转载推断 JCR 或中科院分区；
- 期刊分区无法核实时在证据矩阵中明确标记；
- 核心算法结论优先由正式录用的 ICML、NeurIPS、ICLR、AAAI、KDD、AISTATS 和 AAMAS 论文支持。

## 3. 检索式

### Q1：反事实和差分优势

```text
("multi-agent reinforcement learning" OR MARL)
AND ("credit assignment" OR "counterfactual advantage" OR "difference reward")
AND ("centralized training decentralized execution" OR CTDE)
```

### Q2：价值与策略分解

```text
("cooperative MARL")
AND ("value decomposition" OR "factored critic"
     OR "policy decomposition" OR Shapley)
```

### Q3：机器人和群体应用

```text
("multi-robot" OR swarm OR "autonomous vehicle")
AND ("reinforcement learning")
AND ("credit assignment" OR "difference reward")
```

### Q4：时间信用

```text
("temporal credit assignment" OR "delayed reward")
AND ("multi-agent" OR cooperative)
AND ("actor-critic" OR "policy gradient")
```

### Q5：部分可观测和通信

```text
("partial observability" OR "communication constraint")
AND ("cooperative MARL")
AND ("credit assignment" OR coordination)
```

## 4. 候选发现数量

使用 OpenAlex 对2017-01-01至2026-08-12进行补充发现，各检索式返回的未筛选记录总数为：

| 检索式 | 原始记录数 |
| --- | ---: |
| Q1 | 483 |
| Q2 | 448 |
| Q3 | 584 |
| Q4 | 1,224 |
| Q5 | 1,001 |

这些总数包含主题偏离、重复、预印本和低相关论文，不代表纳入数量。每组只取相关度最高的100条用于候选池，按 DOI 或 OpenAlex ID 去重后得到328条唯一记录。

筛选流程为：

```text
328 条去重候选
→ 94 条题名/摘要相关记录
→ 62 条正式发表或需要全文核验的记录
→ 48 条写入证据矩阵
   ├─ 41 条核心、基础或直接相邻高水平证据
   ├─ 4 条领域/综述补充
   └─ 3 条明确排除记录
```

在41条核心及相邻证据中，对 DAE、COMA、MAPPO、HAPPO、Optimal Baseline、LICA、MACA、FACMAC、DOP、QMIX、QPLEX、SHAQ、STAS、HCA、CCA、COCOA、RUDDER、PRD 等18条进行了公式和实现边界精读。

## 5. 引文追踪

前向和后向追踪的种子论文为：

- COMA；
- DAE；
- MAPPO；
- QMIX；
- LICA；
- FACMAC；
- SHAQ；
- HCA；
- PRD；
- Multi-Agent Deep Reinforcement Learning: A Survey。

引文追踪用于发现 Difference Rewards Policy Gradients、Optimal Baseline、QTRAN、QPLEX、DOP、MACA、STAS、CCA、COCOA 和 Collective Actor-Critic 等论文。最终书目信息回到官方论文页或 DOI 核验。

## 6. 纳入标准

核心证据需满足：

1. 已正式录用或正式发表；
2. 研究合作式共享奖励、CTDE、信用分配、价值分解或与其直接相关的多智能体策略梯度；
3. 给出可辨认的算法、公式或理论结论；
4. 可以明确区分训练期和执行期输入；
5. 对四车共享 Actor、40维动作或机器人任务至少有一项可迁移意义。

经典 difference reward、potential-based shaping 和 MARL 综述可突破2017年下限，但只用作理论基础。

## 7. 排除标准

以下记录不进入算法结论：

- 未录用投稿、被拒论文和只有预印本的未核验工作；
- 需要集中式在线动作选择或执行期全局状态的方法；
- 只研究竞争博弈、LLM agent、自然语言子目标或与机器人控制无可解释迁移关系的工作；
- 只有 reward arrival delay，而不研究动作到结果因果延迟的方法；
- 仅改变探索、角色或通信而没有信用机制的论文；
- 无法核实正式出版来源的二手总结。

相邻问题论文可以进入证据矩阵，但必须标记为 `core-adjacent` 或 `adjacent`，不能用来单独授权信用算法。

## 8. 质量核验规则

- 会议质量以官方正式录用页和正式论文集为准；
- 期刊论文记录 DOI、卷期和论文类型；
- 只有获得当年度官方 JCR 或中科院数据时才标注分区；
- 作者主页、机构存档和 arXiv 可用于获得全文，但不能替代正式发表状态；
- OpenReview 中的 `Published as a conference paper`、会议 poster/oral 状态与未录用 submission 严格区分。

## 9. 可复核性

- 完整逐篇字段位于[marl_credit_assignment_evidence_matrix.csv](marl_credit_assignment_evidence_matrix.csv)；
- 关键正式文献书目信息位于[marl_credit_assignment.bib](marl_credit_assignment.bib)；
- 算法公式和项目接口映射位于[marl_credit_assignment_technical_appendix.md](marl_credit_assignment_technical_appendix.md)；
- 研究结论与唯一推荐位于[marl_credit_assignment_review.md](marl_credit_assignment_review.md)。

后续新增文献必须追加检索日期、来源、正式发表状态和纳入理由，不得只把论文链接加入列表。
