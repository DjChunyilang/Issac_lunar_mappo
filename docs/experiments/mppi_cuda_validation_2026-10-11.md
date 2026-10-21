# MPPI 完整 CUDA 验收与后续诊断

日期：2026-10-11。状态：CUDA数值／负载验收、新GPU Gate B、T10完整矩阵功能门槛及逐步证据／公开快照重放全部通过。正式训练继续暂停。依据[CUDA验收计划](../cuda_acceptance_plan.md)，沿用[第五阶段冻结要求](../implementation_plan.md)。[首次CUDA诊断](mppi_cuda_diagnostic_2026-10-11.md)及全部历史CPU结果独立保留。

## 已确认的实现与证据

采用独立适配器 `mppi_terrain_neighbor_stable_v2`、状态版本2及数值版本 `fp32_fp64_accum_v2`。采样、控制和运动预测保持float32；邻车距离／风险及重要性修正、权重和更新累计使用float64，再导出float32控制。旧v1公式、配置与快照语义保持原状，旧状态不能静默迁移。

地形查询按256 MiB临时张量估算预算自适应分块。B64、每实例4096个本地样本的容量查询已实际验证：query chunk由1024缩为28，峰值allocated约156.5 MiB；这是容量查询证据，不是B64/K2048/4096点完整规划延迟。

最终冻结源码后的必要CUDA契约实际执行66项，相关联合回归695项，合计761项全部通过，0失败、0跳过；包含18项实际运行的真实acados回归。JUnit、命令及源码／测试指纹位于 `outputs/runs/mppi_cuda_validation/contracts_v2_seed23_rtx2060_20261011_02/`。先前contracts `_01`保留为开发证据；runtime测试场景修正后完整重跑了最终`_02`，未复用旧源码指纹宣称通过。

runtime `outputs/runs/mppi_cuda_validation/acceptance_v2_seed23_rtx2060_20261011_02/metrics/strict_acceptance.json` 为 `passed=true`：平坦、坡地、邻车交叉三类场景各覆盖K512/1024/2048与B1/4/16/64，共36面板；9个B1同噪声CPU对照、九类场景×三个种子的27个复杂对照全部通过。每个性能面板预热一次、测量三次，CUDA同步计时。B表示独立环境的单车输入；两环境×四车的真实集成另外验收。

首轮runtime `_01`在邻车交叉B4触发合法输入校验：测试打包复制了sender身份，却把接收车改成同一sender，产生self-neighbor。该轮25个已完成面板及失败日志完整保留；修正诊断打包为独立环境、固定接收车编号后另建`_02`，完整重跑36面板。问题来自诊断场景构造，不能计作规划器失败，也不能将部分结果计作完整验收。

T06独立CUDA配置还实际运行了现有传感／通信链的两环境×四车K512平坦局部闭环，达到0.2 m以内，原始记录为 `outputs/runs/mppi_cuda_validation/configuration_v2_seed23_rtx2060_20261011_01/metrics/t06_sensor.json`。

## 耗时、资源与边界

本机RTX2060 6 GiB、驱动580.178.04、CUDA runtime12.6、PyTorch2.10.0+cu126，OSQP1.0.4。正式runtime时GPU与物理CPU0独占，CPU1–5运行冻结CPU基线；Torch／BLAS各单线程。以下为B1完整规划调用p50，与相同输入及注入噪声的CPU调用对比，包含打包、最终CPU复核及导出。

| 场景 | K512 GPU / 加速 | K1024 GPU / 加速 | K2048 GPU / 加速 |
| --- | --- | --- | --- |
| 平坦 | 225 ms / 7.96倍 | 357 ms / 9.16倍 | 690 ms / 10.42倍 |
| 坡地 | 315 ms / 6.21倍 | 454 ms / 8.64倍 | 871 ms / 8.60倍 |
| 邻车交叉 | 313 ms / 6.46倍 | 435 ms / 8.69倍 | 800 ms / 9.99倍 |

36面板峰值GPU allocated 1055.26 MiB、reserved 1852 MiB；进程RSS累计高水位1597.29 MiB。reserved包含此前缓存，不能解释为当前轨迹张量实际占用。4096样本容量专项只测B64／128查询点，chunk=28、allocated156.50 MiB；它不是K2048／4096样本整次规划的性能结果。

独立同步profile外层调用264.11 ms，其中候选复核111.63 ms、运动预测89.27 ms。它们是包含内部地形查询和传输开销的函数时域，不能与内部或嵌套计时机械相加。最终CPU重预测、协议复核及数据导出仍消耗墙钟时间；GPU接通不等于0.2 s实时预算通过。详细统计与来源SHA位于 `outputs/runs/mppi_cuda_diagnostic/t10_failure_cpu144_readonly_20261011_01/metrics/runtime_latency_resources.json`。

## 运行与配置

本机RTX2060、PyTorch2.10.0+cu126，独立环境 `.venv_terrain_cuda`；感知、通信、差速代理、OSQP及最终轨迹复核仍在CPU。新增独立配置从T06基础到T10集合逐阶段继承，详见[架构索引](../architecture/planning_pipeline.md)和[诊断手册](../runbooks/planner_diagnostics.md)。

新runtime验收使用 `scripts/validate_mppi_cuda.py`，新的Gate B要求当前完整CUDA JUnit、runtime严格验收和同源码必要回归；正式T10继续要求完整144回合及两规划器各9/9简单集合。逐步记录由 `scripts/audit_gathering_traces.py` 审核。正式训练、真实车体足迹与Isaac/PhysX仍各自暂停或待验。

## 原CPU失败的后续诊断

旧144回合原始结果仍为116成功、28失败，0碰撞、0执行限制违反；两规划器各九例简单平坦全部成功。只读复核48个坡地／交叉回合的2942份最新attempt轨迹及终端状态，重新计算任务门槛与原记录一致，原始分类未改。分析位于 `outputs/runs/mppi_cuda_diagnostic/t10_failure_cpu144_readonly_20261011_01/metrics/cpu144_failure_analysis.json` 及 `mechanism_summary.json`。

- MPPI的12个坡地失败只有终端地形条件未满足。最大间距1.675–1.802 m、离散度0.660–0.706 m²及停车均达标；任务区域高度差0.18038–0.18332 m超过0.18 m，最大坡度0.26436–0.27318超过0.25。固定leader槽位的理想布局本身最大坡度0.26846，仍超限；延长停车不能改变该门槛。这是现有37点任务区域测量，不是完整真实车体足迹。
- NMPC的12个坡地失败还存在每个种子一辆车持续QP失败、零控制和原停车轨迹复用，槽位误差1.73–2.07 m；间距、离散度及终端地形均未满足。
- NMPC的4个交叉seed23失败中，三辆跟随车每回合475／895次求解均返回状态4 `ACADOS_QP_FAILURE`，复用可行但静止的旧计划；`valid=true`不代表取得进展。目标及路径在合法局部记忆内，缺失通信仅发生在启动，单次最大求解0.241 s，远小于10 s上限。

在独立CPU5诊断中，用公开snapshot、原配置及同SHA求解器二进制只执行三次acados，三辆车均复现相同status4，输出状态／控制／速度／时间与历史逐元素一致。内部统计显示第五个SQP子问题返回 `MIN_STEP`，IPM迭代16／19／16，未达到50次上限。离线复核导出的末次线性化QP，HiGHS在`1e-9`容差下判不可行，Farkas证书驻点残差不超过`1.39e-17`；冲突集来自地形支持约束下界与控制变化率约束，未含邻车约束。

这一证据定位到复现过程中的线性化子问题冲突；初始非线性问题的等式／不等式残差均为零，不能据此称原OCP无解。公开snapshot不含原live dual／内部QP状态，因此只承诺公共恢复边界下相同输入与输出复现。未放宽控制、地形或碰撞条件，也未改动求解器设置。统计、原输入指纹、QP与证书位于 `outputs/runs/mppi_cuda_diagnostic/acados_crossing_seed23_cpu5_20261011_01/`。

这些任务／求解失败与CUDA数值失败分别记录。GPU可减少计算时间，不能放宽终端条件或使旧NMPC自动获得有效运动解。下述新GPU完整144回合在冻结条件下重新分类，不以CPU历史结果代替。

## GPU T09：完整千例与Gate B

新run `outputs/runs/t09_planner_benchmark/acceptance_seed23_cuda_v2_20261011_01/` 实际完成七设置×1000单次、630恢复复现、252局部回合和14批量面板。`metrics/summary_0000.json` 的 `gate_b_cuda=true`、`gate_b_cpu=false`；覆盖、恢复复现、预定简单闭环、实际设备及源码绑定均通过，并绑定上述最终66／695项JUnit和runtime严格验收。CPU旧Gate B保持独立，不用于替代本轮GPU证据。

案例文件与旧CPU冻结语料SHA一致。四种基线实际仍在CPU，三种MPPI在CUDA使用稳定v2；主调用单Torch线程，调度与完整命令位于run的`config/commands_actual.json`、`scheduling_actual.json`。47类局部闭环采用输出目录内的四个互斥工作器，每个9回合，脚本SHA及36回合无重叠预检保留；原生产源码未修改。

全部单次文件及280分片的7000行逐对象一致，输入指纹唯一且齐全。三种MPPI各962个有效输出、38个刹车，38个无效HOLD／`speed_limit`案例ID与旧CPU相同；`valid=true`与候选被拒绝、前缀或旧计划复用分别统计，不能作为目标完成率。

局部闭环的复杂失败保留原定义。固定消息在0.6 s过期后，保守预测半径扩大；该夹具末端半径4.6 m超过固定已知支持域，完整时域没有同时满足地形与邻车条件的终点。脚本邻车继续运动，刹车也可能被撞，`unverified_brake`不计安全。未知目标距离支持域的下界超过0.2 m，且固定夹具不刷新记忆，因此停车不能被计作到达。诊断只针对这些固定输入，不能推广为实际通信闭环全局不可达。

单次完整质量、输入／分片核验和几何复核位于 `outputs/runs/mppi_cuda_diagnostic/t10_failure_cpu144_readonly_20261011_01/metrics/t09_single_quality_complete.json`、`t09_single_integrity_verification.json` 和 `t09_failure_geometry_verification.json`。CPU使用旧v1与CPU RNG，CUDA使用stable_v2与CUDA RNG；同病例类别对照不等于同噪声逐值回放，也不证明差异由设备单独造成。

最终只读对照 `metrics/t09_quality_final.json` 绑定官方summary SHA：252局部回合共162成功、33碰撞、57截止，执行限制违反为0；旧CPU为162成功、34碰撞、56截止。唯一结局类别变化是K2048的`cross_000`安全关闭回合由碰撞变为截止，不能据此宣称安全收益。K512／1024各24/36成功、6碰撞；K2048为24/36成功、5碰撞。28个固定未知边界回合全部未到达。630次同设备恢复全部通过、最大误差0，14面板输出／安全类别与旧CPU相同；不同版本之间的类别变化另行记录，不能解释为恢复失败。

## GPU T10：完整144回合

新run `outputs/runs/t10_gathering_no_learning/acceptance_seed23_cuda_v2_20261011_01/` 完整执行144回合，未设置`--max-macros`。NMPC使用CPU、MPPI固定CUDA K512／两轮；实际传感、通信缓存、局部记忆和任务终止链接通，两环境×四车运行，96／180 s预算完整保留，没有学习交互或优化器更新。

全部工作器正常退出后，公开`--aggregate`生成`metrics/summary_0007.json`，SHA为`77c5a89f7cbc8b190612e8992f13b8b032bde72f31e563d56736049bc4aeb06c`。`t10_cuda_functional_pass=true`；144覆盖、实际设备、数值证据及GPU Gate B绑定均通过。两规划器安全开启180 s的简单平坦主验收分别9/9成功，总18/18；全部回合无代理碰撞、段内代理碰撞或执行限制违反。

| 规划器 | 成功 / 完整回合 | 失败 |
| --- | --- | --- |
| CPU NMPC | 56/72 | 12坡地、4交叉 |
| CUDA MPPI-512 | 54/72 | 12坡地、6 NPB |
| 合计 | 110/144 | 34 |

最终质量对照见诊断run的`metrics/t10_quality_final.json`。新增六个NPB回合的终端门槛只有离散度未满足：seed23约0.75028 m²、seed31安全开启约0.81138 m²，均超过0.75 m²且驻留计数为零，不能通过容差放宽记作成功。

官方频率分类为24个`terminal_terrain`、4个`planning_rejection`和6个`task_deadline`；频率标签与独立机制诊断分别记录。旧CPU矩阵为116/144成功，新增六个NPB截止失败不计成功，也不能归为CUDA数值契约失败。CPU v1与CUDA stable_v2使用不同随机流，不能把两个完整回合称为同噪声配对。

完整执行及复核命令位于run的`config/commands.json`与`final_review_commands.json`。CPU基线最初分配独立物理核，其他CPU工作器结束后，NMPC坡地线程池扩展到空闲CPU1–5，原配置和源码不变；`config/scheduling_actual_updates.json`保留时间、命令与原因。此类全回合耗时跨越两种affinity，不能作为固定核延迟比较。

## 新NPB停滞的支持缺口

新CUDA回合的NPB seed23在安全开／关、96／180 s均失败，seed31只在安全开启两种预算失败；其余NPB回合成功。目标有本地支持，实际地图与最近样本的可通行度／坡度达标，但当前点距最近记忆样本约0.20 m，超过0.155 m支持半径。当前点参与硬复核，导致无可行候选和持续零控制。

首次失效前的轨迹有效；已核对的seed23安全关闭及seed31安全开启首次失效链中，执行后的位姿与预测误差分别约`6.3e-8`和`1.4e-7` m。感知更新将车下`[0,0]`及前方`[0.2,0]`两个物理有效采样量化到同一`np.rint`记忆键，后写点覆盖车下点。公开前后快照之间的五个低层步在CPU纯感知重放中逐值一致，包括样本、坐标、特征及时间戳，定位为量化更新产生的支持缺口，未发现明显执行偏差或真实地图空洞。

安全开启时，段内起点对应Jacobian为零，支持圆盘产生`0·u ≤ -0.015975`的矛盾约束；现场QP `primal infeasible`与这一独立几何证据分别记录。未放宽支持或控制约束；停车只能记录为未验证刹车，不能获得安全认证。静止轨迹的邻车复核通过不代表所有采样候选或全局任务可达。

部分诊断、原命令与CPU分析源码位于 `outputs/runs/mppi_cuda_diagnostic/t10_failure_cpu144_readonly_20261011_01/metrics/cuda_npb_diagnostic_0000.json`、`cuda_npb_diagnostic_0001.json`及该run的`logs/`；完整矩阵对照、48个坡地／交叉终端复核及独立生产终止函数检查全部通过，分别见同目录的`t10_outcome_comparison_0000.json`、`t10_complex_terminal_validation_0001.json`及`t10_complex_source_termination_0000.json`。NPB完整24例的CPU终端复核追加在`cuda_npb_diagnostic_0002.json`；`cuda_npb_diagnostic_0003.json`最终37/37检查通过，绑定24个完整配对、48个终端门槛复算、六个记忆更新见证、158个生产源码及官方`summary_0007.json`，输入读前／读后SHA未变。后续优先以独立版本修复本地记忆覆盖、保守特征与直接观测语义，再重跑对应闭环；本轮冻结评测不静默修复或调权重。

## 逐步证据与公开生命周期

`metrics/trace_audit_0000.json` 审核通过，SHA为`44fd09705f01d8a607ac832e561d434d3c6f12c7e71a7a7d4e72fb8a8e34078b`：4108份最新宏步轨迹、4108个提交状态快照、117744次规划执行调用，完整覆盖全部已完成回合；实际下发与广播预测首控制一致。10170次真正采样调用中保留710次原始采样，占6.98%；首次／每25步／新失败规则要求的710个事件全部覆盖，HOLD分支不采样。原始张量均为512×15×2，没有把事件覆盖率当作保存所有采样的比例。

另建`outputs/runs/mppi_cuda_validation/public_state_replay_v2_seed23_rtx2060_20261011_01/`，在正式工作器退出后独占GPU，从六类场景的可信公开快照实际恢复。每类两次五步续跑与原轨迹在固定`atol=rtol=1e-5`内一致，两次控制逐值相等、最大误差0；每类另一次Oracle／其他环境改动及稀疏批次检查不改变本环境执行，异步reset不改变另一环境RNG和安全层状态。`metrics/cuda_public_state_replay_0000.json`为`passed=true`、`source_matches=true`；命令、分析源码及输入快照SHA独立保存，没有写回正式回合。

独立终止诊断补充脚本首轮错误读取`asdict`中未包含的派生配置属性，失败日志保留；另建v2脚本使用生产`max_episode_steps`后全部检查通过。该错误属于诊断入口，不是生产回合或任务判据失败。其原代码、修正版本与重跑日志仍在诊断run的`logs/`，不覆盖失败证据。

## GPU完整评测的严格门槛

- 已完成：相关必要联合回归及18项真实acados，全部实际通过且没有必要skip。
- 已完成：K512/1024/2048、B1/4/16/64完整CUDA负载、复杂场景和同噪声数值对照。
- 已完成：新GPU T09的7000单次、630重复、252局部回合、14批量面板及GPU Gate B。
- 已完成：新GPU T10完整144回合、实际感知通信、18/18简单集合门槛、逐步证据和六类公开快照重放。

最终`metrics/final_source_validation_0000.json`确认158个生产源码在CUDA契约、回归、runtime、T09、T10和公开快照重放六处绑定一致，五套新CUDA配置保持显式设备与训练暂停。文档链接、数学分隔符、仓库卫生、Ruff F821、安装脚本语法及diff检查另存`metrics/static_final_0001.json`。新文件检查首轮包装误将`git --no-index`正常差异退出码1计为失败，原日志保留；重新判定检查内容及退出码后通过，未修改生产文件。

CUDA数值／负载、GPU Gate B和T10 CUDA功能状态分别记录；复杂场景失败不会计作成功，性能不会解释为实时或真实月面安全认证。
