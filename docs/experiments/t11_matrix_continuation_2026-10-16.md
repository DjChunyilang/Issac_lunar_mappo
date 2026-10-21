# T11继续执行：完整矩阵、v5集合与依赖审核

日期：2026-10-16。11:18（Asia/Shanghai）快照。用户要求继续收齐评测、扩大v5集合、核对证据，并在既定门槛满足后推进100k。该学习阶段已有条件授权；当前门槛仍未满足，正式交互为0，没有训练checkpoint。

## 目标行为与验收

- 原冻结benchmark达到7000/630/252/14完整覆盖，保留全部失败；以原汇总器判断Gate B，不从覆盖数推断通过。
- v5完成六类、三种子、两规划器、安全开/关、180/360 s的144回合；安全开/360 s简单三类的18条全部成功且无规定违反。
- 每份复用结果绑定原生产者、配置和SHA，实际改变执行的分支须补证据；不得把新旧版本成功拼为统一验收。
- 满足上述和现有契约/设备审核后启动100k信号检查；不自动升级后续预算。

## 已结算结果

原v5 run为`outputs/runs/upstream_scale_s04/braking_lease_v5_full_pairs_20261015_01/`。

| 规划器 | compact种子23 | compact种子31 |
| --- | --- | --- |
| MPPI512 | 128.8 s成功 | 203.8 s成功 |
| NMPC | 250.4 s成功 | 284.6 s成功 |

四条结果均实际驻留25步，违规列表为空。旧v4 NMPC对应两例均在360 s截止，因此新速度承诺的改善已经转化为两例完整集合成功。结果文件仍记录未获复核的刹车：MPPI两例863/1496次、NMPC两例1536/2102次；不能由成功宣称所有控制获安全认证。模拟秒数不等于计算墙钟，也不据两例作规划器总体排名。

原benchmark run为`outputs/runs/upstream_scale_s05/formal1000_nativepreview_cpu_20261015_01/`：7000单次、630重复一致、173局部、12批量；局部140完成、33未完成。NMPC局部31/36成功，MPPI512/1024/2048分别34/33/31条成功，各自36条均已运行。预定NMPC/MPPI512简单平坦子集8/8成功；其余几何基线尚未全部运行。

33条未完成的可观测分类为：23条结束时仍在捕获范围外、8条进入捕获范围但停车驻留未完成、2条环境失效。分类保留原`terminal_reason`、误差、驻留、恢复次数、违规列表和逐文件SHA。它不将近目标失败直接归因于控制器，也不把环境失效改成超时。

## 实际续跑与冻结边界

原几何进程在保存的恢复点附近中断，原日志保留；中断不计规划失败。已分别启动line、quintic、primitive47三个互不重叠的局部worker，继续原冻结源码、配置、种子和状态。记录为原run的`logs/geometry_split_20261016_01.json`。

六项缺失批量与局部闭环独立运行，不再等待原语局部全部结束。11:18已有line/quintic四项完成，剩余primitive47的B4/B16。调度记录为`logs/batch_early_20261016_01.json`；新汇入器仍在覆盖达到后执行原正式汇总。新增批量在其他任务并发期间运行，耗时标记为并发资源面板，不替代独占机器延迟。原7000条单次规划耗时保持原值。

新集合run：`outputs/runs/upstream_scale_s05/gather144_braking_v5_20261016_01/`。

- [续跑入口](../../scripts/continue_upstream_gathering.py)为NMPC/MPPI各生成72条结果，全部简单门槛优先；进程锁阻止同一规划器重复worker。
- 两个进程从新run的`config/source_snapshot/`执行，正在补compact种子47，随后irregular、reverse三种子，再继续完整余项。每宏步沿原入口保存公开状态，失败不删案例。
- 原四条v5结果经有效配置、依赖版本、源码影响逐项核对后硬链接引用；原文件字节和耗时不变。`config/completed_episode_inheritance.json`绑定原manifest、每条结果和差异解释。
- 与原四例生产者相比，运行相关变化仅为seed列表的JSON表示修复；学习collector新增日志不由无学习入口调用，实际共用的`renew_legal_bounds` AST一致，其他学习/准备脚本不执行。没有迁移不同版本热状态。

源码、配置和依赖均已归档，地图数组原身份保留。没有更改物理、安全、任务阈值，没有降低K512或缩减两轮更新。

## 版本核对与必须补齐的证据

核对产物：`outputs/runs/upstream_scale_t11/continuation_dependencies_20261016_01/metrics/`，含`source_comparison.json`和逐项`dependency_actions.json`。

| 差异 | 复用范围 | 必须补齐 |
| --- | --- | --- |
| 无学习日志、续跑调度、seed JSON表示 | 原结果按原SHA引用；四条v5有效配置一致 | 调度覆盖/重复拒绝及恢复检查 |
| 训练角色、collector记录、训练定义 | 原暂停配置分支与已有诊断结果 | 最终学习角色和准入绑定；不以诊断更新代替学习效果 |
| v3平坦帧复核与规划器分派 | 已有独立参考/真实NMPC配对的受测范围；原v2结果继续有效 | 最终候选复核与规划的配对范围核对；旧延迟不能标成v3速度 |
| 新制动距离速度承诺 | 不影响已冻结静态输入中的消息值；旧局部回合仍仅证明旧承诺 | v5实际局部闭环与144集合，不能只靠旧局部成功 |
| v3优先迟滞/v4净空绕行 | 固定局部目标的旧证据不证明新高层集合 | 同版本完整144及简单18条门槛 |
| CPU与CUDA | 本机各自原数值/容量证据 | 最终配置的设备依赖核对；迁移其他服务器时补受影响项 |

这份审核列明动作，不签发`dependency_audit_passed=true`。原7000/630不无条件重跑，也不直接改标为v5全矩阵。最终审核仍须绑定实际受影响验证和完整结果。

## 训练准备与可追溯报告

基于v5重新生成A0–A3定义和18个独立评估场景，位于`outputs/runs/upstream_scale_t11/v5_training_definitions_20261016_01/`。`config/conditional_training_authorization.json`记录本轮用户的条件授权；尚未解除运行暂停，不构造正式训练checkpoint。

[只读进度入口](../../scripts/report_upstream_continuation.py)持续写入`outputs/runs/upstream_scale_t11/continuation_progress_20261016_01/metrics/progress_*.json`。按覆盖变化追加，含原始结果路径/SHA和症状分类；矩阵收齐后停止，不自动签发准入或启动学习。

现有准入器已实际调用，结果保存在`outputs/runs/upstream_scale_t11/continuation_readiness_20261016_01/metrics/readiness.json`，为false。该次调用仅绑定已有不完整汇总，没有附上最终contracts/CUDA/training_entry证据，显示missing不代表历史测试不存在。首次命令误写覆盖文件名，在发布前失败；已通过实际文件匹配重跑，日志保留。

## 本轮验证

新增调度矩阵覆盖、简单门槛优先、重复/不完整拒绝、复杂失败保留，加既有JSON恢复检查，共4项通过、0失败/skip（2.10 s）。结果及测试SHA存于新集合run的`logs/continuation_contract_tests.json`。未修改物理执行或求解器代码，不重复既有真实acados回归。

新增两脚本、测试的Ruff F821通过。其余文档/仓库检查记录另存本轮进度run的`logs/`；以实际日志结果为准。后台计算尚未结算的案例不记为通过，正式训练仍待真实门槛。

11:23追加：覆盖增加至7000/630/175/13，局部140条完成。原语B4已完成，B16继续；三种几何局部与两个集合进程仍在推进。10份修改文档334个本地链接及公式检查、默认25文档公式、仓库卫生、全相关源码F821和diff检查通过。四例继承另核对17份支持包源码，全部相同；`writing-block.md`字节未变。
