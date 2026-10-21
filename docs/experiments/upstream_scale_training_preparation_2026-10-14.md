# 新尺度安全闭环、独立校准与四组真实学习准备

日期：2026-10-14。当前为正式训练前准备；没有启动正式训练、消耗T11分级预算或生成训练checkpoint。之前的[环境接线与代价修复](upstream_scale_environment_2026-10-14.md)保留原始结果，本记录只追加本轮证据。

## 目标与当前结论

修复三个实际缺口：完整环境与实际控制/通信同步、独立新尺度校准与统一评测、四组动作的真实循环采样/更新。当前完整环境的两例安全开/关配对和四组学习入口已实际运行；36例单次规划校准已完成。九类真实局部闭环已完成并通过S04出口；统一版本完整Gate B和144集合回合尚未全部完成，训练准入保持false。

## 安全开启的实际出生集合

配置：[upstream_scale_s04_target_safety_probe_cuda_v4.yaml](../../configs/experiment/upstream_scale_s04_target_safety_probe_cuda_v4.yaml)。run：`outputs/runs/upstream_scale_s03/environment_target_v4_safety_seed23_cuda_20261014_01/`。

- 两环境各四车、K512两轮，通过实际密集SensorFrame、私有记忆、距离通信和任务链运行；68/77 s完成集合，各满足25步驻留。之前安全关闭配对为72/85 s，两组均限这两个平坦出生案例。
- `metrics/execution_audit.json`审核2900条实际控制，零包络碰撞、零净空/幅值/轮速/变化率违反。625次QP为solved且unchanged；本场景的名义控制已经可行，不能据此宣称发生了安全修正。临界修正另有公共QP契约证据。
- 状态次数为unchanged625、verified_brake1777、unverified_brake498。未验证刹车主要涉及未来保守邻车界；实际回合无碰撞，不把这些控制写成安全认证。
- `metrics/broadcast_audit.json`审核全部保存宏步快照的1539条消息：带轨迹的消息与sender复核后的实际执行预测一致；309条未通过复核的预测已撤销，错误为0。原执行审计中的pending字段不覆盖，完成结果由独立广播审计补充。

## 四组真实循环学习入口

`UpstreamAblationRollout`接通A0/A1原始极坐标、A2原始归一化四维动作和A3原生编号；执行桥接保持原始采样、log-prob与库身份。每次动作前仅从本车状态和合法消息建立强制速度租约，不将HOLD解释为未经强制的零速承诺。

修复了通信缓存恢复后的异步reset：恢复时的只读时钟数组必须先复制再清零。原失败日志保留，之后实际恢复/回合重置通过。

run：`outputs/runs/upstream_scale_t11/actual_heads_seed23_31_47_cpu_cuda_20261014_01/`。A0–A3各seed23/31/47，真实两环境各采两宏步，规划诊断K64：

| 项目 | 实测 |
| --- | --- |
| 面板 | 12 |
| 有效团队宏步转移 | 48 |
| 每车动作决策 | 192 |
| ratio重放最大误差 | 0 |
| 有限诊断优化器更新 | 12 |
| 正式训练交互/预算 | 0 |

这些2 s诊断只验证真实观测、动作、有限任务bootstrap、变时长GAE、GRU普通切分/异步重置和更新接线，不计100k信号检查或集合学习成功。

`outputs/runs/upstream_scale_t11/actual_learning_contracts_seed23_cpu_cuda_20261014_01/`的原批次141项：140通过、1失败、零skip，30项真实acados全部实际通过。唯一失败是测试精确比较0.6与实际浮点0.6000000000000001；只修改测试容差，原失败JUnit保留，受影响两项重跑通过。见`metrics/executed_summary.json`、`recheck_summary.json`和两份JUnit；重合项不相加。

## 独立36例校准

配置：[upstream_scale_s04_calibration_cuda_v4.yaml](../../configs/experiment/upstream_scale_s04_calibration_cuda_v4.yaml)。run：`outputs/runs/upstream_scale_s04/calibration_36_seed101_103_107_109_cuda_20261014_03/`。

九类各四例，seed101/103/107/109与正式评测分离。输入来自实际12231点感知采样、只读局部快照、实际延迟历史和sender绑定预测；解析地图身份明确，不称为独立DEM地理样本。相同标准高斯扰动用于参数对照，单Torch线程、K512、两轮。

七组各最多三值，实际15份不同配置、540次完整规划，1632.1 s；540份结果均valid。按预先规定的独立物理指标选择，而非规划器内部total_cost。选定参数写入[独立配置](../../configs/experiment/upstream_scale_s04_selected_cuda_v4.yaml)：

| 参数 | 选定值 |
| --- | --- |
| goal / terrain / neighbor / smooth倍率 | 2 / 0.5 / 0.5 / 0.5 |
| 线速/角速原始噪声标准差 | 0.375 / 0.15 |
| 温度 | 0.5 |

这只完成单次规划校准；`calibration_summary.json`的`s04_accepted=false`不覆盖。实际局部闭环独立在`calibrated_nine_families_seed101_cuda_20261014_01/`执行，每类最多300步；已经完成的平坦/坡地/月坑/通道误差分别0.033/0.063/0.108/0.054 m且有停车驻留。迎面例300步截止，误差1.470 m，原冻结分类为可行性未知，保留失败，不算完成。交叉280步完成、误差0.189 m；满速HOLD60步实际停车/驻留，减速位移9.369 m不算返回请求位置；倒车45步完成、误差0.098 m。未知边界300步截止、误差0.517 m，保留受阻。`closure_summary.json`确认九类全部执行、预定四类可达场景全部完成、零执行契约违反、`s04_accepted=true`；原单次规划JSON仍保持其独立false标记。

之前的`flat_closure_probe_seed101_cuda_20261014_01/`以探索参数完成45步、误差0.099 m；它不替代选定参数九类闭环。校准准备的01/02 run没有完整执行调用，03为当前实际生产者。

## 训练定义与完整矩阵入口

`prepare_upstream_t11.py`生成四组实际运行定义与18个独立评估场景（seed151/157/163），保持411/991输入、128 GRU及同seed共享骨干初始化；A3明确原生执行差异。预算保留每候选100k信号检查、通过者500k×3、保留者2M×3。

`PreparedUpstreamTrainer`按连续实际宏步收集完整循环序列，提供五轮PPO、原概率重放/无效掩码/有限任务GAE和公开恢复；训练更新要求独立readiness与训练授权同时成立。当前没有调用正式更新。

run：`outputs/runs/upstream_scale_t11/definitions_a0_a3_selected_seed23_cuda_20261014_02/`。3项契约通过，包含实际安全开启采样、暂停拒绝、原动作不变和恢复/出生/压缩状态检查。01入口把安全版本误写为通用代号而被解析器拒绝，改为实际`footprint_discrete_cbf_upstream_v1`；失败日志保留。定义生成不等于训练准入。

`benchmark_upstream_planners.py`已提供1000固定输入、7000单次、630恢复、252实际局部回合和14批量面板。47类单次与闭环均按完整原生候选枚举及独立公共代价选择，保留选择成本；安全开/关从同一原生输出计算。原输入生成run为`upstream_scale_s05/formal1000_selected_cpu_cuda_20261014_01/`；当前执行run02显式复用其不可变输入及SHA，不重复生成。

`evaluate_upstream_gathering.py`提供新分层出生、NMPC/MPPI512、安全开/关和180/360 s的144回合，真实传感/通信/脚本/任务及双环境批处理。每5底层步保存可恢复公开状态；gzip压缩只改变存储，校验SHA，不包含求解器。选中轨迹/控制逐步保存，原始MPPI样本首次、每25步及新失败事件保存。尚未产出完整新Gate B或144回合验收。

## 下一出口

完成九类校准闭环及失败定位后，冻结全部实际依赖并执行完整S05矩阵；先尝试NMPC实际正式出生集合，避免只做组件检查。验收按必要回归、样本覆盖和真实成功门槛判定，复用未变化依赖的既有证据。新版本完整门槛与训练准备审核全部通过前，正式训练继续暂停。


## 追加：正式出生缺口和执行成本

NMPC原版的零控制移位热启动会将后续方向状态约束锁在零，求解成功却不推进。独立`nmpc_upstream_v2`只在远于捕获区且全序列零控制/无位移时重建引导初值，原模型/代价/完整复核继续使用。`nmpc_zero_mode_recovery_contracts_cpu_20261014_01/`5项实际求解器契约通过，正常冷解配对采用atol=1e-7/rtol=1e-6。

直接指向槽位的滚动目标在大型已停车辆周围会进入受阻区域；新增独立`leader_square_visibility_upstream_v2`，依据合法sender预测的保守半径建立可见图，只选择原4 m范围内的中间目标，保持冻结集合场地/最终槽位/停车状态机。图不认证地形或执行；实际轨迹继续完整复核。两组原NMPC探索已显式切分并保存公开状态，`diagnostic_cut.json`标记未完成完整回合，不计规划失败或S05通过。新的NMPC/MPPI各双环境正式出生探索run为`visibility_nmpc_two_episodes_seed23_31_cpu_20261014_01/`和`visibility_mppi_two_episodes_seed23_31_cuda_20261014_01/`，目前仍运行。

`flat_grid_pose_cache_v2`只在局部点集能证明完整规则格、恒定高度/零坡度/通行且全车采样位于支持矩形内部时复用原完整物理报告；未知洞/危险属性/边界及非规则点集回到原查询。动态限幅、运动、邻车、时间及工作区检查仍逐点执行。原声明54项契约通过；强引用缓存和极端整数范围拒绝补齐后，公共执行56项全部通过，零失败/skip，含原30项真实acados和5项新NMPC恢复契约。原批次与实际生产源码各自保留，不相加。

相同已观测旋转平坦格、同一实际运动预测、360次检查的报告逐项一致，PoseCachedReviewer2.837 s、FlatGridPoseReviewer0.041 s，见`visibility_flat_contracts_20261014_01/metrics/performance_probe.json`。这是单条局部复核成本，不代表整个闭环或非平坦地形获得相同加速。

并行CPU评测触发acados每次重构共享库的竞态，原run`upstream_scale_s05/formal1000_selected_cpu_reviewv2_20261014_02/`保存缺库日志及`construction_interruption.json`。独立NMPCv2已采用文件锁、同模型编译一次/后续加载，库mtime及冷解配对验证通过；`nmpc_cache_recovery_contracts_cpu_20261014_01/`6项实际求解器契约通过。

当前CPU千例run为`upstream_scale_s05/formal1000_selected_cpu_reviewv2_20261014_03/`。唯一生产代码差异是独立NMPCv2的构造缓存，97条其他规划器产物按原文件SHA与原manifest明确复用；NMPC行全部重新执行，原构造/规划耗时保留原来源。三块CPU工作区间不重叠、每进程单Torch线程，原始CPU输入不重新生成。完整7000/630/252/14及144回合仍未全部完成，不颁发训练准入。
