# 活跃规划架构与执行面

2026-10-20：1 km课程候选 `configs/training/upstream_1km_curriculum_v2.yaml` 沿用 `run_upstream_t11.py --curriculum-config` 入口。`upstream_curriculum.py` 与 `upstream_process_rollout.py` 共享显式初始级别；`upstream_curriculum_experiment.py` 实现每级物理步预留、独立探索初值、固定趋势／完整评估及预算核算。原v1配置保持旧分支，当前训练暂停；参数、检查及版本边界见[六项审定](../experiments/t11_six_point_review_2026-10-20.md)。

2026-10-19：`local_goal/upstream_value_normalization.py`实现可选价值目标累计矩，`action_ablation.AblationPolicy.value`统一还原奖励单位，`upstream_training.recurrent_mappo_update`按每批更新统计并训练归一化Critic。沿用repair-validation入口运行单变量候选，原组合结果复用为对照；恢复身份、预算及命令见[记录](../experiments/t11_value_normalization_trial_2026-10-19.md)。

2026-10-19：`run_upstream_t11.py --mode repair-validation`通过`upstream_learning_validation.py`编排2万底层步baseline／组合候选，复用`upstream_decision_experiment.run_arm`。`upstream_learning_cohort.py`定义相同近／远混合初态与停车评估；接口和预算见[记录](../experiments/t11_combined_learning_validation_2026-10-19.md)。

2026-10-18：T11候选时间层由`local_goal/upstream_decision.py`实现，单进程与多owner共用`step_decision`，保持原规划／控制频率。`run_upstream_t11.py --mode decision-credit`调度`scripts/upstream_decision_experiment.py`的三组有界研究会话，固定初态评估位于`upstream_decision_evaluation.py`，评估轨迹不进入PPO。研究学习尚未启动；接口与运行条件见[实现记录](../experiments/t11_decision_learning_implementation_2026-10-18.md)。


2026-10-14实际环境入口见[环境记录](../experiments/upstream_scale_environment_2026-10-14.md)。`upstream_environment.py`连接密集感知、私有记忆、合法通信、原始请求/执行租约、统一规划器、跟踪、可选新尺度QP、实际执行、任务、奖励与宏步；`upstream_rollout.py`连接A1 GRU/PPO，`upstream_action_bridge.py`冻结四组目标解码及原生编号。配置为`upstream_scale_s03_environment_v2.yaml`，安全及CUDA/原生47各有独立配置。局部契约通过，完整出生集合、S04/S05、四组学习入口和正式训练准备仍在推进。

末段代价与初始化探索使用`upstream_mppi_calibration.py`及`upstream_scale_s04_target_probe_cuda_v4.yaml`；v1/v2保留原算法，v3/v4独立核、代价与状态版本。当前版本用于机制修复和实际出生探索，不代表独立36例校准或新Gate B通过。

日期：2026-10-14。新尺度S02已接通40步规划核心及容量矩阵；S03实际环境的感知、通信、任务、奖励/宏步、安全和GRU局部契约已接线，完整出生集合、校准、实时与新Gate B仍待验收。旧exp167/P0及T05–T10保留各自尺度、接口和证据。验证边界见[当前状态](../current_status.md)，字段定义见[接口规范](../interface_spec.md)。

## 历史与P0兼容执行链

```text
局部传感 + 合法通信缓存 + 剩余任务时间（v2）
→ 共享 N1 CNN + GRU Actor → 四维 tanh Gaussian / latent / log-prob
→ 固定 odometry frame 的 LocalGoalRequest
→ prefix_consistency_v2 NMPC / 轨迹复核 / 旧计划回退 / 限速刹车
→ TimedTrajectory + PlannerFeedback → 差速执行
→ 实际奖励与结束事件 → 变时长 MacroTransition
→ 集中 Critic / duration_gae / 序列 PPO（正式训练暂停）
```

Actor 和规划器只读取本车可获得的信息；Critic、团队奖励和任务验收可以使用集中状态。Actor 不接收真实团队驻留计数或 Oracle 集合点。旧hold_mask测试覆盖入口不参与PPO采样；T04条件HOLD是独立版本的真实策略分支。

## 活跃代码与配置索引

代码包位于 `source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/`。

| 对象 | 入口 | 定位 |
| --- | --- | --- |
| P0 诊断配置 | `configs/experiment/p0_contract_v2.yaml` | finite_task_v2/local_goal_clock_v2；训练暂停 |
| 历史对照配置 | `configs/experiment/exp167_local_goal_nmpc.yaml` | legacy_v1/local_goal_v1；复现与回归 |
| 新契约审计 | `scripts/audit_p0_contracts.py` | 无优化器、无求解器调用 |
| 旧契约审计 | `scripts/audit_learning_contracts.py` | 保留旧问题数值基线 |
| 环境与信息边界 | `local_goal/environment.py` | 传感、通信、宏步、实际执行、统计、恢复 |
| 数据类型 | `local_goal/contracts.py` | 请求、轨迹、反馈、邻车预测及版本元数据 |
| 学习 | `local_goal/learning.py` | 时钟编码、真实采样概率、有效样本 GAE/PPO |
| 任务与奖励 | `termination.py`、`reward.py` | 新语义显式启用；共享旧入口保留默认行为 |
| 配置与 runner | `scripts/_common.py`、`scripts/train_local_goal_nmpc.py` | 合同验证、暂停门、checkpoint和依赖指纹 |
| 验证 | `tests/test_p0_learning_contracts.py`、既有 local_goal/通信/任务测试 | P0解析、兼容与真实 acados |

以上两项配置及下述T03–T10诊断配置构成本阶段allowlist，其他配置通过[T00 路线清单](../experiments/t00_repository_audit_2026-10-09.md)按需追溯；存在文件不等于允许恢复训练。

## 研究目标与交接

2026-10-11 追加独立修复，事实见[验收记录](../experiments/gathering_failure_repairs_2026-10-11.md)：`direct_batch_v3` 保留同格直接采样，`leader_local_site_v8` 只由合法本地观测选址，`nmpc_recovery_v4` 有界冷启动／短参考恢复。旧配置默认行为保留，定向修复不颁发修改后的完整 Gate B/T10。

| 修复执行面 | 配置 | 入口与边界 |
| --- | --- | --- |
| 局部记忆单项 | `t10_memory_repair_cuda_v4.yaml` | `memory.py`；支持半径／危险阈值不变，旧高层与原 NMPC 保留 |
| 坡地选址单项 | `t10_local_site_repair_v8.yaml` | `gathering_script.py`；当前代理终端区域的本地检查，非真实完整车体足迹 |
| 求解恢复单项 | `t10_nmpc_recovery_v5.yaml` | `planner.py`、`baselines.py`、`registry.py`；最多四次共享时限的真实 NMPC 调用 |
| 组合诊断 | `t10_combined_repairs_v9.yaml` | MPPI stable_v2 K512 权重保留；NMPC 使用恢复 v4 |
| 低风险停车开发对照 | `t10_local_site_repair_v9.yaml` / `t10_combined_repairs_v10.yaml` | `leader_local_site_v9` 要求完整锚点观测，紧凑／不规则六例成功，倒车开发中断保留 |
| 已停稳捕获 | `t10_local_site_repair_v10.yaml` / `t10_combined_repairs_v11.yaml` | `leader_local_site_v10` 只依据本车已停稳及当前位置支持请求 HOLD，不认证未知共同场地；当前真实回合验证三类平坦 |
| 定向完整回合 | `scripts/diagnose_gathering_repairs.py` | 原传感／通信／实际执行链，无学习，明确不生成正式门控 |

上述配置为本轮追加诊断 allowlist。`v2–v7` 选址及旧修复配置为开发复现资料，不能从其存在推断通过；v8 的 CUDA 平坦停车退化另有完整失败证据。异步 reset／公开快照沿用环境接口；恢复校验记忆、脚本、规划器版本，禁止跨版本静默加载。

T03 已接入独立诊断协议，验收见[T03记录](../experiments/t03_planner_protocol_validation_2026-10-10.md)；T04 已在独立诊断路径加入 `[rho,beta]+HOLD`；T05–T07已实现CPU批量terrain-aware MPPI及完整基线，验收见[T07记录](../experiments/t07_mppi_terrain_neighbor_validation_2026-10-10.md)。[首次CUDA诊断](../experiments/mppi_cuda_diagnostic_2026-10-11.md)保留v1数值失败；独立稳定v2的CUDA数值／负载、新GPU Gate B与T10完整矩阵的简单集合功能已通过，复杂失败独立保留。安全投影在独立T08配置实现，验收状态见当前记录。

47 类原语与 line/quintic 作为冻结基线保留；Active-DSTC、PRD 和旧过滤器保留复现入口。没有经过引用分析和回归的目录迁移或删除。

本阶段沿用既有车辆代理、地形、通信和出生边界。新停稳阈值是诊断任务条件，不构成完整车体足迹、真实月面安全或 PhysX 验证。

整个项目已收到上游最高车速 10 km/h、外形 5.015×3.200×1.850 m、轮径／轮距／轴距 1.0／1.8／2.7 m，见[车辆尺度基线](upstream_vehicle_scale_baseline_v1.md)。用户确认差速模型并授权缺失值采用合理假设，最新依据为[独立参数 profile v2 与推导](upstream_vehicle_proxy_assumptions_v2.md)。8 s／40 步不兼容旧15步协议，继续独立迁移；S00／S01基础事实见下述入口，取得输入前的[车型对照](../references/project_physical_scale_reference_2026-10-11.md)和旧代理验收分别保留。

## S00／S01 新尺度基础执行面（2026-10-12）

2026-10-13当前入口为`upstream_scale_s01_review_v2.yaml`：严格复核绑定邻车几何、规范及局部地形动力学，查询按256点／65536邻点条目分块；运行／状态记录独立版本。407项与八车满速／停车／恢复通过，完整环境及实时未通过，详见[修复记录](../experiments/upstream_scale_review_v2_repair_2026-10-13.md)。旧S00／S01显式选择原v1／标量路径，当前公共复核默认v2；不将历史兼容入口用于新验收。

独立运行schema为`upstream_runtime_v1`，配置为`upstream_scale_s00.yaml`／`upstream_scale_s01.yaml`，只用于参数及公共执行诊断。`scripts/_common.py`按schema返回冻结的`UpstreamRuntimeConfig`，旧core／环境／trainer显式拒绝此类型。设计profile仍不可直接加载为experiment，不隐式迁移旧配置。

```text
独立profile＋文件SHA → UpstreamSpec／配置身份
→ 给定合法本地观测 → FootprintTerrain／FootprintReviewer
→ 信息时间stop_envelope／admit_control
→ 明确推进注入延迟保持控制
→ 联合限幅 → UpstreamActuatorProxy实际中点执行
→ 实际下发控制、实际速度、恢复及异步reset证据
```

| 入口 | 范围 |
| --- | --- |
| `local_goal/upstream_spec.py` | 参数、版本、规范／配置指纹和静态队形 |
| `local_goal/upstream_execution.py` | NumPy参考、Torch批量联合限幅、实际速度及停车储备 |
| `local_goal/upstream_footprint.py` | 本地点集、接触／矩形地形、圆包络邻车／工作区、停止走廊 |
| `local_goal/upstream_proxy.py` | 两环境×四车CPU执行与公开生命周期；未接完整任务链 |
| `scripts/audit_upstream_scale.py` | S00严格参数审计 |
| `scripts/diagnose_upstream_execution.py` | S01真实代理加速／停车，要求CPU及显式离线延迟 |
| `scripts/upstream_evidence.py` | 完整JUnit身份、来源、run归属与零skip校验 |

[基础证据](../experiments/upstream_scale_foundation_2026-10-12.md)与后续[S01复核v2记录](../experiments/upstream_scale_review_v2_repair_2026-10-13.md)分别保留原批次结果；[复跑手册](../runbooks/upstream_scale_diagnostics.md)给出命令。S01提供40步规范及停止储备，S02另以显式配置接通下述规划器，不能用S01配置默认启用新规划路径。旧MPPI、NMPC、安全和通信入口保留旧尺度语义；新尺度传感通信／任务与安全链仍待完整接通，正式训练暂停。

## S02新尺度40步规划核心（2026-10-13）

独立配置`configs/experiment/upstream_scale_s02.yaml`新增到无学习诊断allowlist，必须选择`planner_execution_version=upstream_planning_v1`及严格复核／查询v2。工厂拒绝S00／S01未接线配置、实时模式、非MPPI的CUDA选择及缺失训练暂停。实现及本轮证据见[独立记录](../experiments/upstream_scale_planners_2026-10-13.md)，完整S02尚未验收。

```text
规范／配置身份＋给定合法本地观测＋信息时刻本车状态
→ UpstreamIntent / 原生47载荷 → UpstreamPlannerInput（40×0.2 s）
→ UpstreamPlannerPool按车辆隔离状态、按内核身份批调度
  ├─ line / quintic / 新47：参考＋控制器＋新尺度实际预测
  ├─ NMPC：独立40步acados模型＋参考／停车走廊本地支持
  └─ MPPI：CPU或CUDA批预测／局部预筛／重要性更新
→ 共同限幅、补完整停车尾段、重新预测
→ CPU完整接触／矩形扫掠及信息时刻停车储备复核
→ UpstreamPlannerOutput → 代理实际首控制／同设备恢复证据
```

| 入口 | 当前责任 |
| --- | --- |
| `local_goal/upstream_planning.py` | 独立协议v2、输入／信息时刻校验、停车尾段、完整复核、前缀／旧计划／刹车与生命周期 |
| `local_goal/upstream_factory.py` | 显式工厂、内核分组、输出顺序、每车状态、异步reset及整池原子恢复 |
| `local_goal/upstream_baselines.py` | line／quintic控制器回放、新47库、原生编号与冻结请求 |
| `local_goal/upstream_nmpc.py` | 新尺度40步求解器、可行初值阶段模式、走廊支持、冷／热恢复及独立缓存 |
| `local_goal/upstream_mppi.py` | 真批量MPPI、归一化代价、每实例随机状态、16样本完整复核预算与超时记录 |
| `local_goal/upstream_native.py`、`.cu/.cpp/.h` | FP64原生控制核、局部空间哈希、属性一致点集40步融合预测及稳定CPU直接点记忆合并；编译失败显式报错 |
| `local_goal/upstream_tensor_views.py` | 有界空间哈希、sender绑定邻车预筛、局部CPU等距校正与设备管理 |
| `scripts/diagnose_upstream_planners.py` | 冻结源码／测试来源，七设置、全47编号、双设备MPPI及公开恢复面板 |
| `configs/experiment/upstream_scale_s02.yaml` | 新执行版本、N40、默认K512／两轮、模拟0.5 s延迟，正式训练暂停 |

协议以`now`表示控制实际生效时刻，信息时间储备另外保留排队期间的保持控制。二者使用同一地形动力学核验，不把延迟重复执行。滚动目标与`stop_at_goal`分别表达；HOLD、超时及停车成功由各自字段／实际状态判定，预测尾部停车不自动构成任务驻留。

新47库版本为`upstream_primitive47_v1`，保留类别顺序但按新尺度生成端点与参考速度，并绑定内容指纹。旧47库与categorical入口继续用于历史复现；T11准备模型目前没有通过桥接自动切换到新库。

MPPI仅在预测、代价及更新阶段采用共享张量内核，最终完整足迹复核逐实例在CPU执行。CUDA空间哈希的等距／支持边界歧义使用所属CPU本地快照保持五通道语义，次数与耗时明确记录；未知仍为硬拒绝。16样本预算以外的候选保持未复核状态，不将停止搜索解释为无解证明。

本轮联合契约、旧路径相关回归与实际规划面板分别保存。当前诊断输入是给定本地平坦快照，执行证据为实际首控制与同设备恢复；传感链、局部持续闭环、完整负载矩阵、实时预算、安全投影和集合任务未由此验证。Gate B／T11及正式训练条件保持未通过。

追加的54容量面板与九组复杂B1配对已完成，合成点集范围见[S02记录](../experiments/upstream_scale_planners_2026-10-13.md#追加容量负载矩阵完成)；实际传感链继续由下述S03独立验证。

## S03密集感知、合法通信及停车组件（2026-10-13）

`upstream_scale_s03.yaml`加入独立无学习组件诊断allowlist；旧环境／trainer仍拒绝该运行类型。当前不是可训练新尺度环境，证据与失败记录见[S03记录](../experiments/upstream_scale_s03_components_2026-10-13.md)。

```text
DenseSensor（独立owner采样）→ SensorFrame → 每车DenseTerrainMemory
实际广播＋仅用于路由的距离 → DistanceCommunicationCache → sender绑定snapshot
本车实际反馈＋本地视图＋合法消息 → LeaderSquareScript → 方格意图
冻结信息视图＋ActualControlHistory → S03显式时序 → 统一规划工厂
完整未来轨迹＋实际排队路径＋N40停车储备 → 实际限幅下发
MotionBoundLedger持续强制 → 合法未来可达界
实际逐车停车足迹＋团队几何／漂移 → UpstreamTask
```

| 入口 | 当前责任 |
| --- | --- |
| `local_goal/upstream_sensing.py` | 独立12231点帧、连续数组/原生稳定合并的私有直接点记忆、去趋势RMS及readonly导出 |
| `local_goal/upstream_communication.py` | FULL／稀疏权限、sender／时效、执行强制证书与保守预测 |
| `local_goal/upstream_timing.py` | 真实控制历史、实际运动一致性、过去足迹及未来停车总储备 |
| `local_goal/upstream_gathering.py` | 场地中心、编号槽位、合法等待与停车状态机 |
| `local_goal/upstream_task.py` | 实际驻留、漂移、逐车停车条件、终端优先级及异步reset |
| `local_goal/upstream_review_cache.py` | 单次复核内静态位姿查询复用；不缓存动态验收 |
| `scripts/diagnose_upstream_s03.py` | 冻结组件／旧路径回归、八车MPPI／实际停车及公开恢复诊断 |
| `configs/experiment/upstream_scale_s03.yaml` | 实际历史／缓存版本、K512两轮、CPU代理及离线延迟；训练暂停 |

Actor 112点与115项附加观测、集中Critic、奖励／宏步学习、安全QP及完整高层排队调度尚未由组件脚本接通。完整出生分布、集合回合、S04校准及S05同版本全矩阵仍是T11前置门槛。

追加[精度与观测接线](../experiments/upstream_scale_s03_precision_observation_2026-10-13.md)：`upstream_scale_s03_observation_v1.yaml`继承独立`mppi_upstream_v2`，默认原始采样float32、执行器／预测float64，旧v1保持原样。`upstream_observation.py`从私有密集记忆与receiver绑定消息构造411／991维，保持N1布局，新增消息v2及明确归一化；任务驻留只进入Critic，Oracle槽为零。Actor数据及网络前向已接入组件脚本，实际循环rollout、奖励／安全与完整环境仍待完成。

| 新入口 | 范围 |
| --- | --- |
| `local_goal/upstream_mppi_precision.py` | 独立物理精度、局部cap预筛／复核、状态2及CPU／CUDA数值 |
| `local_goal/upstream_observation.py` | 私有Actor、集中Critic、receiver／时刻／sender核对及物理归一化 |
| `configs/experiment/upstream_scale_s03_precision_v2.yaml` | 仅切换新MPPI实现，旧配置不迁移 |
| `configs/experiment/upstream_scale_s03_observation_v1.yaml` | 消息v2／实际记忆观测；训练暂停，非可训练环境 |
| `tests/test_upstream_mppi_precision.py`、`tests/test_upstream_observation.py` | 真实CPU／CUDA、物理参考、信息隔离与网络形状 |
| `tests/test_upstream_s03_diagnostics.py` | 同一协议冷构造／恢复与驱动限定的证据复用 |

## T11独立动作概率准备（2026-10-13）

`local_goal/action_ablation.py`复用T04 N1／GRU／Critic并提供A0–A3动作头、原始载荷、概率重放及诊断状态；`scripts/diagnose_t11_actions.py`执行合成输入契约与四方案三种子面板。配置`configs/ablation/t11_action_contracts.yaml`仅用于准备，**不加入运行experiment allowlist**。记录见[T11无学习准备](../experiments/t11_action_preparation_2026-10-13.md)。

```text
不可变尺度身份＋合成T04观测 → 同一网络骨干＋四组独立动作头
→ 原始动作／latent／分支／log-prob → 循环概率与梯度诊断
→ PreparedRequest（未连接PlannerInput／环境）
```

A2归一化密度与方形位移范围、A3原生47编号及旧执行尺度单独记录。旧PPO拒绝该准备模型；新尺度执行桥接、统一评测和学习预算等待S02–S05。这里的通过不修改T11／Gate C状态。

## T03 独立诊断执行链

`t03_planner_protocol.yaml`继承P0；local_goal/protocol.py承载协议、NMPC桥接和工厂。local_goal/environment.py通过plan_batch调度每车独立实例，快照仅使用公开生命周期方法。原exp167/P0仍走原调用链。

```text
原四维采样／明确诊断HOLD → LocalIntent
→ PlannerInput（本车状态、TerrainView、IdentifiedNeighbor、显式时间网格）
→ Planner协议 / NMPCAdapter → PlannerOutput（复用轨迹与反馈）
→ 轨迹跟踪 → 安全接口直通（projection_disabled）→ 差速执行
```

LocalIntent中可选航向／速度字段为T04/T05留出接口，本阶段不生成低维意图默认约束。过期请求与不支持意图显式刹车；刹车输出valid=false且保留原因。测试专用适配器证明输入可替换，不算确定性规划基线或MPPI实现。

## T04 动作诊断执行链

`t04_polar_hold.yaml`是新增独立诊断配置。local_goal/actions.py定义条件动作分布、PolicyAction、HybridSample和极坐标解码；learning.py按动作版本选择策略头与PPO概率重放。environment.py从真实LocalIntent生成新版观测／通信，不填充伪航向和速度约束。

```text
N1 + GRU → Bernoulli HOLD / 条件二维Normal MOVE
→ 保存原始分支、latent、物理动作与log_prob
→ LocalIntent → T03 Planner协议
  ├─ HOLD：真实NMPC限幅减速
  └─ MOVE：测试专用适配器验收；NMPC完整适配待T05
→ 新版观测与宏步转移 → 合成PPO契约测试（无正式训练）
```

T04使用local_intent_clock_v3，维度保持但观测语义变化；动作及rollout各自版本化，拒绝旧checkpoint。旧配置调用链保留。scripts/audit_polar_hold.py与tests/test_polar_hold.py提供独立诊断和概率／循环学习验证；全任务MOVE评测与正式训练均未开放。

## T05–T07 新诊断实施面

注册表local_goal/registry.py集中适配器、版本及工厂；local_goal/execution.py承载原代理预测／复核与停车前缀。baselines.py提供nmpc_polar_v1、deterministic_line_v1、deterministic_quintic_v1和primitive47_adapter_v1。新的低维NMPC适配与旧T03/T04桥接版本隔离。

统一无学习诊断入口为scripts/diagnose_planners.py。冻结47类原语使用显式原生编号载荷，其动作语义不压缩为低维意图；保留旧categorical训练入口，不启动训练。阶段完成状态以[当前状态](../current_status.md)和独立验收记录为准，实施范围见[计划](../implementation_plan.md)。

## CPU批量规划执行链与索引

```text
固定目标脚本／有序请求回放（无策略权重与优化器）
→ LocalIntent / PlannerInput / 传感本地快照 / sender邻车预测
→ registry工厂
  ├─ nmpc_polar_v1：冻结默认约束 → 原prefix_consistency_v2求解器
  ├─ line/quintic：几何参考 → 逐步控制器 → 差速预测
  ├─ 原生47载荷：冻结解码 → 原生成器／控制器 → 执行限幅
  └─ MPPI：配置分组 → 共享Torch内核（B,K张量）
       → 原始采样 → 凸限幅映射 → 中点预测／可选地形缩放
       → 物理代价／数值标记／段内硬复核 → 原始重要性更新
       → 均值／可行采样／停车前缀／旧计划／刹车
→ 最终CPU轨迹复核 → 跟踪 → 安全接口直通 → 实际差速执行
→ 原任务判据、异步reset及无学习记录
```

| 入口 | 职责 |
| --- | --- |
| local_goal/registry.py | 版本组合、惰性求解器构造、共享MPPI内核 |
| local_goal/execution.py | 原代理预测、限幅、9点复核、停车前缀和移位 |
| local_goal/baselines.py | nmpc_polar_v1、line/quintic、47原生回放与生命周期 |
| local_goal/mppi.py | 运动／代价／原始权重更新、批调度、独立RNG与热启动 |
| local_goal/tensor_views.py | 只读局部样本分块查询及sender邻车打包；不持有环境 |
| configs/experiment/t05_planner_baselines.yaml | 完整基线诊断，默认nmpc_polar_v1 |
| configs/experiment/t06_mppi_base.yaml | CPU基础MPPI，平坦模型、软扩展关闭 |
| configs/experiment/t07_mppi_terrain_neighbor.yaml | CPU地形／邻车MPPI，独立风险和动力学开关 |
| configs/experiment/t07_mppi_terrain_neighbor_cuda.yaml | 独立CUDA规划、query chunk 1024；CPU感知／代理与最终复核 |
| configs/experiment/t06_mppi_base_cuda.yaml | 基础MPPI CUDA／状态1；继承原基础模型 |
| configs/experiment/t07_mppi_terrain_neighbor_cuda_v2.yaml | 稳定数值CUDA地形／邻车、状态2、256 MiB查询预算 |
| configs/experiment/t08_safety_projection_cuda_v2.yaml | 稳定v2 CUDA规划＋现有CPU代理安全投影 |
| configs/experiment/t09_planner_benchmark_cuda_v2.yaml | 七设置完整评测，三种MPPI CUDA、四基线CPU |
| configs/experiment/t10_gathering_no_learning_cuda_v2.yaml | 新GPU Gate B门控下K512完整无学习集合 |
| configs/experiment/primitive47_library_v1.json | 47个索引与实际解码语义、库指纹冻结 |
| scripts/diagnose_planners.py | 统一固定场景、原生载荷、有序请求与现有传感链局部闭环 |
| scripts/install_planner_cuda.sh、requirements-planner-cuda.txt | 在独立CUDA环境补OSQP 1.0.4；保留旧CPU环境与驱动 |
| scripts/benchmark_mppi_devices.py | 同噪声CPU/GPU同步整调用耗时、K/B面板及显存；冻结平坦局部输入 |
| scripts/validate_mppi_cuda.py | 当前CUDA契约绑定、复杂输入、36负载面板、容量与同步阶段诊断；独立严格证据 |
| tests/test_mppi_cuda.py | 真实GPU数值、查询分块、身份RNG／恢复与跨设备拒绝；失败不计通过 |
| tests/test_planner_baselines.py、tests/test_mppi*.py、tests/test_planner_diagnostic_entry.py | 真实NMPC、数值／状态／风险作用和诊断入口验收 |

MPPI的热状态属于每个(env_id,agent_id)句柄，共享内核不保留目标或热启动控制。TerrainBatch和NeighborBatch只由输入快照构造；MPPI显式选择保守地形视图，旧T03/T04等距查询保持历史行为；只沿时间、优化轮次及查询分块循环，B/K维度张量计算。CPU逐车循环限定在消息打包、独立RNG调用、最终协议导出和CPU复核，不以逐车调用内核代替批量预测。

既有T05–T07结果是CPU局部规划与代理闭环。平坦传感链两环境×四车达到局部目标；继承起伏地形有未到达样例，保留到T09分析。不把停车前缀计作到达，不将未知洞填成可行空间。首次v1 CUDA诊断的17项GPU契约为16通过、1失败、0跳过，`cross_offset` float32控制误差 `1.7166e-5` 未通过既定 `atol=1e-5, rtol=1e-5` 对照；原始失败和历史run保留。完整足迹与PhysX分别延期；安全投影及完整集合按T08–T10独立门槛验收。

基础及首次v1独立CUDA配置将 `local_goal.planner_device` 设为CUDA，并设置 `mppi_query_chunk: 1024`；稳定v2配置还显式切换适配器与规划版本，见下文。`simulation.device` 仍为CPU。共享MPPI内核在GPU完成采样、预测、代价与批量复核，协议导出、最终host复核、代理执行及OSQP在CPU。CPU/CUDA分别冷启动新run，完整配置与RNG设备进入热状态校验，跨设备热恢复拒绝。同步设备面板覆盖K=512/1024/2048、GPU B=1/4/16/64，默认CPU对照B=1；整调用前后同步，包含打包、传输与host复核。旧内部phase计时仅反映异步dispatch，不能当作GPU真实耗时。命令与测量范围见[诊断手册](../runbooks/planner_diagnostics.md)。

### 稳定v2 CUDA执行与证据

`mppi_terrain_neighbor_stable_v2`使用状态版本2、`fp32_fp64_accum_v2`。raw采样、运动预测和控制均值仍为float32，邻车位置差／距离／风险及重要性权重累计使用float64，均值更新后转回运行精度。旧v1没有切换算法或状态语义，恢复不能隐式迁移。

稳定v2将地形成对查询临时张量估算预算固定为256 MiB。打包后按`max(1,min(query_chunk,budget//(B*M*(8*element_size+4))))`确定有效查询块，保持B／K张量化和原保守查询；预算版本`pairwise_temporaries_v1`与完整参数写入MPPI元数据。该预算不约束全部存活规划张量、CPU采样导出或CUDA allocator reserved显存，最大容量及实际显存仍必须测量。

`validate_mppi_cuda.py`先绑定当前GPU JUnit和源码；flat／slope／cross分别运行K512／1024／2048×B1／4／16／64共36GPU面板，B1以相同噪声与CPU配对。九类×种子23／31／47另有K128的27个合法输入配对；4096点本地记忆在B64、每实例128个query点下单独核验。容量查询不是B64／K2048／4096点整规划延迟。同步阶段profile单独插入barrier，作为诊断，不混入普通延迟排名。

当前CUDA数值／负载验收已完成：66项CUDA契约、695项联合回归及其中18项真实acados全部通过，0失败／0跳过；36GPU面板、9个B1 CPU配对、27个复杂输入对照和容量边界的runtime严格验收通过。证据run为`outputs/runs/mppi_cuda_validation/contracts_v2_seed23_rtx2060_20261011_02/`及`acceptance_v2_seed23_rtx2060_20261011_02/`。

[新GPU Gate B严格汇总](../../outputs/runs/t09_planner_benchmark/acceptance_seed23_cuda_v2_20261011_01/metrics/summary_0000.json)已通过：7000单次、630恢复、252局部回合、14面板完整执行，重复性、简单闭环、设备及生产源码核验通过，`gate_b_cuda=true`、`gate_b_cpu=false`。[T10正式汇总](../../outputs/runs/t10_gathering_no_learning/acceptance_seed23_cuda_v2_20261011_01/metrics/summary_0007.json)确认完整144回合、两规划器各9例简单门槛通过，`t10_cuda_functional_pass=true`；NMPC 56／72成功、MPPI 54／72成功，无碰撞和执行限制违反。复杂失败为24终端地形、4规划拒绝和6任务截止。[全量逐步审计](../../outputs/runs/t10_gathering_no_learning/acceptance_seed23_cuda_v2_20261011_01/metrics/trace_audit_0000.json)通过117744次执行记录、4108个宏步／公开快照与710／710个要求采样事件，错误为0。

[公开状态CUDA重放](../../outputs/runs/mppi_cuda_validation/public_state_replay_v2_seed23_rtx2060_20261011_01/metrics/cuda_public_state_replay_0000.json)覆盖六类实际集合场景，每类两次冷构造恢复及一次信息隔离重放、各执行5个底层步。同设备恢复控制误差为0，与历史轨迹符合既定容差；Oracle／其他环境／稀疏batch变化不改变合法执行，异步reset保持状态隔离。恢复行为与全量逐步审计分别证明。

三项结论按[追加计划](../cuda_acceptance_plan.md)分别记录，进度／证据见[本轮CUDA验收](../experiments/mppi_cuda_validation_2026-10-11.md)。`mppi_cuda_acceptance_v1`、`planner_cuda_evidence_v1`绑定runtime及生产证据；新T09／T10完整依赖校验保持有效，不由首次pilot、runtime或旧CPU结果推定T10通过，正式训练持续暂停。


## T08–T10 安全、评测与集合入口

```text
T09冻结合法局部输入 / T10真实传感与通信 → 统一工厂 → PlannerOutput
→ 原轨迹跟踪 → SafetyProjector（新配置启用）
→ 非线性9点复核 / 明确刹车 → 实际控制 → 原差速代理
→ 下一步previous_control使用实际值
→ 修正首控制＋停车尾段 → 完整复核 → 广播执行预测或撤销轨迹
→ 实际任务条件、驻留、结束统计、异步重置
```

| 入口 | 职责 |
| --- | --- |
| local_goal/safety.py | 版本化OSQP、解析Jacobian、无松弛投影、非线性复核、执行预测 |
| local_goal/benchmark.py | 冻结九类局部输入、公共评价代价、合法预测的局部推进 |
| local_goal/gathering_script.py | leader_slot_v1，合法FULL锚点、编号让行、HOLD滞回 |
| scripts/planner_evidence.py | 指纹、依赖、原子只增写入、公开状态校验恢复 |
| scripts/reconstruct_t07_cases.py | 使用原下发控制重建四个T07失败状态，显式保留重建标签 |
| scripts/benchmark_planners.py | 1000案例、七设置、配对安全、重复性、局部回合与B4/16面板 |
| scripts/evaluate_gathering_no_learning.py | K512与NMPC、144真实传感集合回合，正式入口受Gate B门控 |
| scripts/audit_gathering_traces.py | 完整回合实际控制、广播首控制、采样覆盖及公开恢复状态审计；只追加结果 |
| configs/experiment/t08_safety_projection.yaml | 可选OSQP、安全投影启用，训练暂停 |
| configs/experiment/t09_planner_benchmark.yaml | CPU正确性、NMPC 10 s、MPPI关闭墙钟截止 |
| configs/experiment/t10_gathering_no_learning.yaml | 两环境四车、K512、合法脚本，无学习 |
| configs/experiment/t10_gathering_no_learning_cuda.yaml | 独立K512 CUDA规划接线pilot，真实感知／安全执行仍在CPU |
| tests/test_safety_projection.py、tests/test_stage5_diagnostics.py | 解析／真实QP、恢复、信息隔离、真实规划器共用执行层 |
| tests/test_t10_evidence_driver.py、tests/test_cuda_evidence_gates.py | 完整矩阵消费者、当前源码／设备／数值／证据SHA门控与恢复契约 |

FULL消息只把sender已经合法发布的目标和执行预测变换至接收车坐标系；脚本无peer真值入口。T09局部观测与T10实际传感证据分开，不以稠密fixture替代感知集成。每个主延迟调用B1、Torch单线程；并发评测进程使用固定物理核，隔离于学习与其他评测，运行亲和性单独记录。

CUDA集合pilot限定 `flat_compact/mppi512/safety on/duration 180/max-macros 30`，用于真实感知、通信和代理执行接线；不构成144回合正式GPU验收。正式新GPU全矩阵需要与当前实际依赖源码匹配的新Gate B生产证据，保留旧源码guard、配置／依赖指纹及恢复边界。正式训练暂停继续有效。

新CUDA T09只在完整7000单次、630恢复、252局部回合、14面板、当前源码绑定联合回归／18真实acados及GPU数值／runtime严格证据通过时产生`gate_b_cuda`，同时保持`gate_b_cpu=false`。CUDA正式T10采用`t10_evidence_driver_cuda_v1`，要求`--gate-b --cuda-junit --cuda-acceptance`，逐项校验适配器、精度、预算、设备及SHA；NMPC仍CPU、MPPI512 CUDA，完整144回合与两规划器各九例简单门槛分别记录。缺例或失败时门控保持未通过。


## 新尺度校准、完整评测与四组学习准备入口

以下为独立新尺度入口，历史15步入口保留原契约。当前范围及失败见[本轮准备记录](../experiments/upstream_scale_training_preparation_2026-10-14.md)。

| 代码/配置 | 职责 |
| --- | --- |
| `local_goal/upstream_ablation_learning.py` | 四组真实采样、有限任务/GAE、GRU普通切分保留、诊断更新及公开恢复 |
| `local_goal/upstream_training.py` | 共享循环PPO目标、训练准入与legacy恢复兼容 |
| `local_goal/upstream_skrl.py` | 当前默认skrl Model/MAPPO/RandomMemory与显式旧checkpoint迁移 |
| `scripts/upstream_process_rollout.py` | 独立常驻owner采样协议，中央策略、同步更新与owner恢复 |
| `scripts/upstream_checkpoint.py` | 更新边界逐owner本地分片保存/恢复、独立学习备份及清单校验，避免父进程汇集完整环境 |
| `scripts/upstream_gpu_service.py` | 可选集中GPU评估服务、变化地形缓存与请求合批 |
| `local_goal/upstream_cases.py` | 独立36例密集感知、真实延迟历史和sender输入 |
| `scripts/calibrate_upstream_planner.py` | 七组最多21候选、相同标准噪声和独立物理指标选择 |
| `configs/experiment/upstream_scale_s04_selected_cuda_v4.yaml` | 540次独立单次规划选定参数，硬物理/任务门槛不变 |
| `scripts/upstream_local_closure.py` | 固定局部目标、停止终点捕获、实际传感/通信/执行及公开恢复 |
| `scripts/diagnose_upstream_local_closures.py` | 九类实际感知校准闭环，独立保留受阻/未知失败 |
| `scripts/upstream_s05_cases.py` | 千例正式输入、新分层出生及回归身份 |
| `scripts/benchmark_upstream_planners.py` | 7000/630/252/14完整矩阵、47枚举、配对安全及续跑 |
| `scripts/evaluate_upstream_gathering.py` | 新尺度144集合回合、NMPC/MPPI512、双环境及恢复 |
| `scripts/upstream_run_state.py` | gzip公开诊断状态和SHA，不覆盖已有结果 |
| `scripts/prepare_upstream_t11.py` | 四组运行定义、原学习预算和18个独立评估场景；零正式交互 |

S03完整环境已接线；两例实际集合与四组真实循环准备分别通过对应范围。S04九类闭环、新S05完整矩阵和训练准入仍待完成。本机CUDA实际运行与目标训练服务器验收分别记录。

## 2026-10-15恢复与T11运行入口

- `scripts/benchmark_upstream_planners.py`通过公开reset/load完成冷/热状态重复性检查。新run显式继承7000条相同生产依赖结果，使用归档代码独立执行后续阶段。
- `local_goal/upstream_visibility.py`增加独立v3优先迟滞及v4导航净空；新配置分别用于原出生集合研究，不改历史基线。
- `local_goal/upstream_training.py` v2与`scripts/run_upstream_t11.py`接通完整循环更新、独立评估、预算、状态保存/恢复和正式训练门控；有界诊断不计分级学习预算。
- 运行器`upstream_t11_session_v2`在准入之后构造独立学习角色，工厂/环境/collector身份对应；`scripts/audit_upstream_t11_readiness.py`汇总既定矩阵与依赖SHA，当前结果未满足准入。
- `local_goal/upstream_flat_review.py`新增`flat_frame_pose_cache_v3`，配置`configs/experiment/upstream_scale_s04_frame_review_cuda_v3.yaml`；用于混合朝向平坦记忆，旧配置与正在运行的冻结评测不切换版本。

当前评测与集合修复状态见[推进记录](../experiments/t11_progress_repair_2026-10-15.md)，完整Gate B/144集合及正式训练尚未完成。

2026-10-15追加：[速度承诺修复记录](../experiments/t11_speed_lease_repair_2026-10-15.md)给出原出生MPPI两例成功、NMPC真实短程对照和完整A1批次结果。新增入口如下，历史配置与当前冻结评测不自动切换。

| 入口 | 职责 |
| --- | --- |
| `local_goal/upstream_speed_lease.py` | 依据合法消息提出制动距离速度承诺；账本继续强制旧承诺，执行仍完整复核 |
| `configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml` | 独立速度承诺v3、导航净空v4和平坦帧v3的完整回合研究配置 |
| `scripts/diagnose_upstream_nmpc_stall.py` | 从绑定SHA的公开失败状态进行有界真实求解器对照 |
| `scripts/upstream_local_closure.py::native_preview_config` | 原生47类信息时刻枚举；选中编号仍经实际延迟队列执行 |

公共规划器已经接通`flat_frame_pose_cache_v3`分派；真实acados配对保持轨迹数值一致。新速度承诺进入脚本与学习collector的独立配置路径，不修改条件HOLD动作概率。A1的128宏步/5轮更新属于有界诊断，正式运行仍需原定准入。

T11准备分析入口：`scripts/analyze_upstream_t11.py`离线读取原样本/执行记录；`profile_upstream_t11.py`做有界CPU嵌套计时；`report_upstream_preparation.py`生成可导出的诊断图；`compare_upstream_evidence.py`输出源码差异清点；`package_upstream_transfer.py`及`verify_upstream_transfer.py`制作/核验源码与运行地图包。运行器和trainer增加纯记录字段，保持原损失及执行行为，详见[并行准备](../experiments/t11_parallel_preparation_2026-10-15.md)。

2026-10-16续跑入口：`scripts/continue_upstream_gathering.py`在冻结生产者内按规划器分工，优先全部18条简单门槛，继续144矩阵并保留失败；`scripts/report_upstream_continuation.py`只读追加覆盖及失败症状，不授予训练准入。实际进度和依赖复用见[本轮记录](../experiments/t11_matrix_continuation_2026-10-16.md)。

## 2026-10-16：100k信号角色与完整验收分离

用户授权的当前执行顺序为A1→A0→A2→A3，各100k、种子23；剩余144/原语47评测暂停且不阻塞此阶段。`audit_upstream_t11_readiness.py --scope signal100k`生成独立准入，`run_upstream_t11.py --mode signal`及底层trainer执行真实PPO，`run_upstream_t11_signal_suite.py`顺序调度。完整系统准入字段保持false，原完整train路径不受此授权替代。

任务/奖励/动作、K512两轮及128宏步/5轮PPO保持原定义，使用当前等价加速实现。每16宏步记录实际采样/动作，每次更新保存恢复点；signal不自动插入18场景长评测。当前规则见[T11计划](../t11_handoff_plan.md)，上文旧准入顺序作为历史保留。

2026-10-16学习采样承诺修复：`upstream_ablation_learning.renew_legal_bounds`在制动距离限制之外保留完整预测时域的位移预算，防止所有学习车辆同时承诺较高速度而导致可达圆覆盖静止尾段。collector以`learned_horizon_braking_lease_v1`标记该执行行为，A0–A3共用且不读取采样动作。脚本、规划和安全复核未改；同状态短程对照8车均产生位移后，A1从1,024个有效转移恢复100k，详见[证据](../experiments/t11_learned_lease_repair_2026-10-16.md)。


## 2026-10-17：skrl默认入口与计算执行路径

`run_upstream_t11.py`默认`--trainer-backend skrl`，实际模型/采样/内存/Adam归属`RecurrentTeamMAPPO`。四车共享actor与集中critic在一个team角色中保留rover轴；变时长GAE、完整GRU序列与HOLD概率继续由项目契约扩展update，采用同步on-policy调度。`PreparedUpstreamTrainer`保留准入/旧实现，只有显式legacy选项采用旧优化器路径。

地形只读几何/KD树供内部快照共享；每车GPU索引按几何身份缓存，HOLD/MOVE/评分共享一次打包，候选与诊断按dtype批量下载。安全约束批处理、原生停车序列和记忆scratch减少CPU调用；观察、执行预测、停车报告、中心查询合为每车任务。非deadline GPU阶段使用events延迟计时，deadline与安全判定保持原语义。

`--env-owners N`启用新采样协议：每进程长期持有完整环境组，中央策略全批采样，宏步边界只传必要数组/执行记录，更新期间不预采下一批。环境ID在owner内稳定，外部记录携带owner_id/env_offset；seed按owner分组派生，因此禁止将旧完整checkpoint静默导入新协议。可选`--gpu-service`独立进程集中评估同时就绪请求，默认关闭，由端到端结果决定取舍。框架与设备、owner数量/worker数均进入会话身份。

实现、硬件负载、长回放和完整PPO更新证据见[本轮记录](../experiments/t11_skrl_deep_optimization_2026-10-17.md)。原两环境协议的GPU PPO实测更慢，默认CPU学习/CUDA规划。未恢复正式长训练。


## 2026-10-19：1 km 课程长学习执行面

独立物理入口 `configs/experiment/upstream_1km_curriculum_v1.yaml` 绑定 `configs/design/upstream_vehicle_proxy_1km_v1.yaml`，学习入口仍为 `scripts/run_upstream_t11.py`，通过 `--curriculum-config configs/training/upstream_1km_curriculum.yaml` 调用 `scripts/upstream_curriculum_experiment.py`。平地世界 1000×1000 m，完整／稀疏通信 300／600 m；不缩放旧 DEM，局部感知和车辆安全约束保留。

`local_goal/upstream_curriculum.py` 管理七级随机出生、成功率晋级和易场景复习。环境在异步 reset 应用课程，逐回合截止时间同时进入任务及 Actor/Critic；中央和 owner checkpoint 分别保存晋级窗口与出生 RNG／实际级别。奖励和观测对数尺度为显式配置，旧线性默认与 checkpoint 身份不变。前置检查由 `scripts/audit_upstream_curriculum.py` 记录；执行与评估预算、冻结源码及完整命令见[实验记录](../experiments/t11_1km_curriculum_long_training_2026-10-19.md)。
