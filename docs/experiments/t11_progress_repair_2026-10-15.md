# T11推进：评测恢复、集合停滞与训练运行流程

日期：2026-10-15。目标是完成既有T11准备缺口，正式训练保持暂停。前置事实见[10月14日记录](upstream_scale_training_preparation_2026-10-14.md)，本轮不覆盖其结果。

后续完成结果见[晚间速度承诺修复记录](t11_speed_lease_repair_2026-10-15.md)：本文以下“运行中”是当时快照；A1完整128宏步及MPPI v3两例已结束，NMPC v4两例均截止，630重复性已全部完成。旧快照和失败证据保留。

## 目标行为与验收

1. 恢复冷/热规划器状态并继续拒绝跨车辆恢复；复用相同生产代码、配置、输入下的7000条规划结果，继续630/252/14矩阵。
2. 消除HOLD迟滞与优先放行条件冲突；以原出生、K512、实际感知通信和360 s任务重新验证。
3. 定位NMPC低速停滞，在合法邻车信息内尝试更有净空的路点，原控制限制、通信速度承诺、停车储备和安全复核继续生效。
4. 补齐训练入口，验证完整循环PPO、独立评估、公开恢复及暂停保护；有界诊断更新不计100k预算。

## 评测恢复

旧run：`outputs/runs/upstream_scale_s05/formal1000_selected_cpu_reviewv2_20261014_03/`。

- 7000次单次规划全部完成。Line/Quintic/47原语/NMPC/MPPI512/1024/2048有效输出分别984/984/999/904/989/989/992。有效输出包括前缀与回退，不代表目标完成。
- 后续阶段于2026-10-15 08:09退出：未绑定车辆的冷快照被加载到已绑定实例，触发身份拒绝。失败日志保留。
- [评测入口](../../scripts/benchmark_upstream_planners.py)改为公开`reset→load_state_dict`，分别检查冷/热状态、控制、实际速度、时间及有效性/回退类别，身份保护不变。[针对性测试](../../tests/test_upstream_benchmark_replay.py)3项通过，包含真实acados。

新run：`outputs/runs/upstream_scale_s05/formal1000_replayfix_cpu_20261015_01/`。

`config/inherited_plan_evidence.json`记录原manifest、全部行及输入SHA、原始状态来源。继承时确认唯一生产源码差异为评测脚本的重复性步骤；原构造/规划耗时不改标。`config/source_snapshot/`含独立冻结执行目录，工作区后续修改不会进入评测。三个不重叠的规划器组各自按repeat→local→batch运行并记录退出状态，最终汇总；覆盖未完成或失败时不判Gate B通过。

2026-10-15 14:12（Asia/Shanghai）快照：7000/7000单次、184/630重复、14/252局部、0/14批量。184项冷/热恢复全部一致；NMPC组90项重复全部结束。14条局部中12条完成，未记录执行违反。两条未完成均为`crater_seed31_001`的NMPC安全开/关：误差0.00531/0.00540 m、驻留0、60 s截止。恢复原本地记忆复核发现实际速度满足停稳，但足迹坡度0.10574/0.10576（约6.04°）超过5°停车上限；行驶复核通过、停车复核返回`parking_slope`。保留该失败，不能改标目标完成。证据为本轮`metrics/progress_1416.json`及`crater_parking_diagnosis.json`，后者绑定两份原末态SHA。

## 集合停滞的具体见证

此前四回合均已结束：MPPI种子23于166 s成功，种子31截止；NMPC种子23/31均截止。四例未记录执行限制/净空/包络违反，但仍有`unverified_brake`，不能称全部控制安全认证。原144回合完整矩阵尚未执行。

MPPI种子31中，2号车最终距槽位0.307947 m，处于合法HOLD迟滞区（捕获0.2 m、退出0.4 m）；3号车仍要求它进入0.2 m才放行。2号车保持HOLD、3号车持续等待。原按累计未验证刹车次数归类的`safety_or_stopping_reserve_rejection`不足以解释该回合，不能当作已确认根因。

独立`leader_square_visibility_upstream_v3`将优先放行半径对齐既有退出半径，仍要求合法、及时、FULL、HOLD、实际停稳的邻车消息。捕获、退出、成功、控制及碰撞阈值不变；历史v1/v2保持旧行为，新旧状态拒绝静默混用。

配置：[priority v3](../../configs/experiment/upstream_scale_s04_priority_recovery_cuda_v3.yaml)。实际完整重跑：`outputs/runs/upstream_scale_s04/priority_v3_mppi_full_seed23_31_cuda_20261015_01/`。本记录写入时运行中，尚不能声称完整回合已修复。

恢复NMPC原300 s快照后，两辆活动车实际下发约0.00537/0.00404 m/s，合法邻车已停稳、半径6.548968 m。边缘路点的厘米级净空通过既有速度承诺规则压低速度。实际记录中QP多数unchanged；“安全层一直拒绝”不是准确的全部解释。

独立`leader_square_visibility_upstream_v4`先离开过小净空，再采用最多1 m、在最终槽位处收敛的导航余量。实际障碍半径、消息有效期、速度承诺与安全复核均不变。两个保存状态的几何见证确认首段净空增加，完整效果待实测。

配置：[clearance v4](../../configs/experiment/upstream_scale_s04_clearance_recovery_cuda_v4.yaml)。完整NMPC研究重跑：`outputs/runs/upstream_scale_s04/clearance_v4_nmpc_full_seed23_31_cpu_20261015_01/`。不计作原冻结版本Gate B或144回合验收。

## T11运行流程

[运行入口](../../scripts/run_upstream_t11.py)接通四方案、真实采样、完整循环PPO、独立评估、原采样/执行记录和恢复。引擎升级为`upstream_recurrent_ablation_trainer_v2`，历史热状态不静默迁入。

- diagnostic模式最多两批、每批不超过128宏步，使用同一完整更新实现；仅保存诊断状态，正式交互数为0。
- train模式在环境构造前要求显式解除暂停及完整、校验SHA的同源准入证据，当前未执行。诊断模型不能直接成为正式训练初态。
- 正式预算按有效团队宏步计。A0/A1/A2共用配置规划器，A3保留原生执行身份。
- 独立评估使用151/157/163种子、独立环境/模型副本及RNG作用域，保持训练环境和GRU状态。诊断只做有界入口检查，不冒称18场景完整评估。
- 自动保存/恢复、独立评估和运行预算已接通；长期学习表现、完整准入和Gate C仍待后续证明。

证据：`outputs/runs/upstream_scale_t11/replay_and_preparation_repair_20261015_01/`。恢复3项通过；优先/旧集合批次16通过1测试构造失败，修正非法负时间fixture后5项通过；追加训练运行、准备及净空见证12项通过。批次重合不相加。首次真实CLI发现归档缺少依赖元数据，已补齐，失败run保留。

实际CLI证据（均位于`outputs/runs/upstream_scale_t11/`，K64、CPU、有界诊断）：

| run | 实际结果 | 范围 |
| --- | --- | --- |
| `session_a0_resume_cpu_20261015_01` | 第一进程4团队宏步/5轮PPO；独立第二进程恢复至累计8宏步/10轮PPO | 两批均完成独立一例一步评估、保存及优化器恢复 |
| `session_a2_cpu_20261015_01` | 2宏步/5轮PPO完成 | 原始样本与执行记录保存 |
| `session_a3_cpu_20261015_01` | 2宏步/5轮PPO完成 | 使用原生47类执行身份 |
| `session_a1_full128_cpu_20261015_02` | 128宏步完整批次运行中 | 尚无完成更新结果，不计通过 |

上述结果由`metrics/cli_short_sessions.json`绑定各run摘要SHA。正式训练交互全部为0，没有生成训练checkpoint。诊断评估只验证入口，不是完整18场景结果。128宏步批次沿原生产源码继续执行，不将后续工作区改动注入运行进程。

后续修复了正式入口的运行身份接线：`upstream_t11_session_v2`在准入与明确解除暂停检查之后才构造`upstream_t11_learning_v1`运行配置；配置解析、工厂、环境、collector与更新入口都识别该身份。原诊断配置仍暂停、元数据不增加学习身份；不同角色状态不能互换。已实际验证学习角色的环境构造（零交互、零优化器更新）及诊断角色完整更新，9项测试通过、无skip；源码快照和JUnit为`config/runtime_role_source/`、`metrics/runtime_role.source.json`和`runtime_role.xml`。这不是解除当前训练暂停。

[准入汇总脚本](../../scripts/audit_upstream_t11_readiness.py)检查已有计划要求的完整矩阵、必要证据与依赖对应关系，绑定文件SHA。缺失/未通过的证据输出`formal_training_ready=false`；汇总本身不解除暂停。相关3项测试通过。当前缺少完整矩阵，尚未产生可用于启动训练的通过记录。

## 平坦混合观测帧的精确复核加速

原`flat_grid_pose_cache_v2`要求整份记忆形成完整规则网格；车辆转向后的多帧混合点云通常不满足，加速失效。新增独立`flat_frame_pose_cache_v3`从最近完整观测帧中证明局部支持，仍要求全部保留点具有一致安全平坦属性；旧危险点否决捷径，支持范围外使用原完整查询，不跨越未知空洞。

配置：[frame review v3](../../configs/experiment/upstream_scale_s04_frame_review_cuda_v3.yaml)。新旧复核9项契约通过。两份实际移动中65536点记忆各检查30个位姿，与原复核结果一致：CPU时间分别0.19180→0.02308 s、0.19469→0.02439 s；原grid v2均未建立捷径。`metrics/actual_frame_performance.json`保存记忆状态SHA、生产源码和测量。这里只证明位姿复核环节，未证明整体吞吐或实时运行；当前冻结评测、集合及128宏步进程未切换该配置。

## 未完成出口

630/252/14完整覆盖、同版本144集合、九例简单集合双规划器全部成功及最终依赖绑定审核尚未完成。T11的100k→500k×3→2M×3预算、Gate C及正式训练均未启动。优先推进实际失败修复和评测，不重复相同依赖下已经充分通过的昂贵回归。

本轮静态核验：Ruff F821、仓库卫生、25份默认文档及10份修改文档的数学分隔符、342个本地链接、`git diff --check`均通过。日志位于本轮`logs/{f821,hygiene,math}_final.log`，链接报告为`metrics/final_doc_links.json`；用户`writing-block.md`未修改。
