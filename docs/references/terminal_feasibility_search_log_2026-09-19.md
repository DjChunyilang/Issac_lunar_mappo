# 终端可行区域研究：检索与证据日志

日期：2026-09-19。配套[综述及27篇清单](terminal_feasibility_review_2026-09-19.md)。

## 1. 范围与方法

本轮是围绕具体科学问题的定向文献调查，不是声称穷尽全部文献的系统综述。先阅读仓库已有共同选址与信用分配综述，再按信息结构、主动感知、分布式运动、奖励和评估五条线回到论文正式来源核查；新增覆盖2025年死锁控制与2026年目标条件学习。

优先使用IEEE/期刊出版页、NeurIPS/ICML/ICLR/RSS/AAAI正式论文页、作者机构网页和作者开源仓库。搜索结果中的DBLP、聚合站、自动综述只用来发现线索，不作为技术结论依据。没有根据影响因子排序挑论文；正式来源、相关性和假设可迁移性更重要。

本轮纳入27篇。旧综述中同一论文不因为被重复引用而算新增论文；同一MADER/RMADER工作的预印本与期刊版不重复计数。未核实正式发表的2025/2026预印本不进入核心证据表。

## 2. 检索主题与代表性检索式

| 主题 | 检索式/入口 | 筛选目的 |
| --- | --- | --- |
| 信息与共同选择 | `Decentralized Stochastic Control Partial History Sharing`; `Partially Observable Multi-agent RL Quasi Efficiency`; `MACKRL`; `SeqComm` | 区分共享参数、共享观测和共同决策依据 |
| 主动探索 | `RACER decentralized exploration`; `distributed sequential greedy assignment`; `non-myopic multi-robot information gathering`; `iHERO scarce communication` | 验证运动获取信息、探索分工、主动重连 |
| 地形风险 | `STEP stochastic traversability`; `Robust constrained learning-based NMPC off-road` | 地形不确定性与模型误差，而非只看平地标签 |
| 运动与死锁 | `MADER`; `Robust MADER communication delay`; `Distributed RHC nonholonomic`; `Deadlock-Aware Control Multiple Safety Constraints` | 区分无碰撞轨迹、递归可行、稳定性、完整到达 |
| 学习与塑形 | `MAPPO surprising effectiveness`; `COMA`; `policy invariance reward transformations`; `HIRO`; `HER`; `PRM-RL` | 识别信息、任务抽象和奖励不变性的不同作用 |
| 近期方法 | `Scaling Goal-conditioned Reinforcement Learning Multistep Quasimetric Distances`; `ROAM active mapping robot teams` | 核查2025/2026正式来源，避免只依赖历史工作 |
| 评价 | `Deep Reinforcement Learning Edge Statistical Precipice` | 多seed、区间、配对比较和样本层级 |

偶然返回的偏好学习、通用VLA、金融通信和无关文章未纳入。不能据检索未命中断言没有类似工作。

## 3. 阅读深度与关键核查

| 文献 | 本轮访问深度 | 核对内容与边界 |
| --- | --- | --- |
| R01共同信息 | IEEE CSS正式记录＋作者预印本PDF相关段落 | coordinator为分析表示；共同与私人信息、处方政策；不冒充运行时必须集中 |
| R03信息共享 | PMLR正式页＋50页PDF中§2—4相关段落 | 可观测性、信息演化、策略无关belief；均衡与团队最优不是同一保证 |
| R04 SeqComm | NeurIPS正式页＋全文§4.2及相关讨论 | 全通信推导与局部通信版本有区别，不能平移理论保证 |
| R10 STEP | RSS正式页＋全文相关方法段落 | 不确定性、CVaR与MPC组合；单车通行不能代替团队终端任务 |
| R12 RMADER | RA-L正式元数据/作者机构页＋作者全文HTML | §I—II的最大时延与delay check关系、完全失联排除、实验中死锁与实际时延越界的限制 |
| R20势函数塑形 | 作者原论文PDF；因文本编码异常，对关键页面做图像阅读 | 折扣形式、MDP适用条件和吸收终止边界；不将理论写成MAPPO收敛保证 |
| R05/R09/R11/R14/R27 | 官方/作者摘要、正式来源和作者代码说明 | 机制、发表身份和迁移边界；未逐行复核证明或复现实验 |
| R02/R06/R07/R08/R13/R15—19/R21—26 | 正式论文/出版页摘要、书目信息及必要原文检索 | 主张限制在可核验的研究目标、方法和公开结论，不报告未核对的消融数值 |

PDF临时阅读文件放在系统临时目录，不作为项目产物提交。少数arXiv PDF下载出现SSL错误，RMADER改用作者全文HTML，RACER使用IEEE正式摘要及作者页面；IEEE部分页面直接打开触发反机器人检查，采用已索引的正式摘要和作者来源交叉核查。这些限制不影响文献链接真实性，但不能声称全文精读了全部27篇。

## 4. 书目信息校正

- MADER DOI含2021，正式卷期为IEEE T-RO 38(1)，2022；文献表同时说明在线与卷期年份。
- RMADER本轮引用扩展RA-L论文，9(2)，2024，DOI为`10.1109/LRA.2023.3342561`；不写成T-RO论文，也不与ICRA 2023会议版混为一篇不同贡献。
- STEP核心引用RSS 2021版本；后续DARPA SubT扩展稿只作背景，没有把预印本年份当正式期刊年份。
- MAPPO正式来源是NeurIPS 2022 Datasets and Benchmarks Track；不是2021正式会议论文。
- R25已在ICLR 2026正式论文页核实题名、作者和摘要，不仅凭arXiv日期推断录用。

## 5. 项目证据核查

本轮阅读 `docs/README.md`、`current_status.md`、实验索引、exp063/064/164/166/167相关记录及已有选址综述。代码核查包括 `metrics.py`、`reward.py`、`termination.py`、`local_goal/contracts.py`、`memory.py`、`environment.py`、`learning.py`及当前配置。

本机原始结果重点核对exp167的pilot summary、前后评测和修正结果文件。早期结果主要来自文档，exp166后续消融沿用用户历史陈述，不补造数值。没有重跑长训或高保真模拟。

额外解析检查为本项目独立推导：平坦度圆盘不蕴含全部车体覆盖、平面坡度限制、激活塑形的符号与折扣循环、物理时间折扣。这些不归功于某篇论文，也不当作真实rollout中的因果效应。

## 6. 仍待研究的问题

尚未证明新组合架构全局完备、任意通信下安全或任意局部观测下可学习。新颖性检索需要在算法形式确定后继续收窄，重点比较风险受限任务相关信息规划、分布式区域选择与终端可达集合。当前综述足以提出可检验方案，不能替代这些证明和实验。
