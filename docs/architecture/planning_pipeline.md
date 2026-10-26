# 活跃规划架构与执行面

更新：2026-10-25。本页描述坑分档训练链及最新MPPI执行修复；旧百万步训练已结束；最新SAT三轮配置于20:18开始从零百万步训练。旧阶段正文见[完整快照](../archive/formal_training_transition_2026-10-22/docs/architecture/planning_pipeline.md)。研究工作顺序见[实施计划](../implementation_plan.md)，具体字段见[接口规范](../interface_spec.md)。

## 当前执行链

```text
环境／感知所有者持有月面地图和障碍 atlas
  → 本车视场增量揭示 → 私有格记忆／只读局部视图
  → 本车状态＋合法 sender 消息 → 三通道区域观测
  → 共享 N1 CNN＋GRU Actor → A1 极坐标 MOVE／HOLD
  → 固定 odometry 目标／请求队列 → 统一规划器工厂
  → CUDA MPPI v4 → 控制限幅 → 差速 Proxy 实际执行
  → 实际足迹／停车／任务终止 → 奖励与变时长转移
  → 集中式 Critic／GAE → skrl MAPPO 更新
```

策略每 5 s 采样一次并冻结目标，每 1 s 续发／重规划，底层控制周期 0.2 s。固定规划网格为 40×0.2 s，状态 `[41,3]`、控制与实际速度 `[40,2]`、时间 `[41]`。4 owner × 4 环境共 64 车；各 owner 内同配置 MPPI 使用张量批处理，策略／PPO 当前在 CPU，MPPI 在 CUDA。

当前 `cost_reward_only_v1` 路径关闭预测完整足迹拦截、停车储备及每步安全投影／准入；保留执行限幅、MPPI 既有预筛和实际地形／停车判定。`valid` 不表示完整足迹安全认证。旧 strict 路径继续留存，不能将其验收结论直接套用当前行为。

Actor 523、Critic 1439 维；Actor 三通道为障碍接近风险、地面 cost、known 比例。Critic 另接团队状态，Oracle 槽保留为零。私有格记忆只持已经揭示的 tile，规划器没有全图、atlas 或未通信车辆真值入口。当前驾驶在相同足迹支持位置查询去坑派生面，并加精确大坑侵入判定；停车仍查询原始地形。

当前已启用计算等价采样优化：格查询共享不可变tile池并合批读取所需通道，MPPI内部错误标记在批次边界检查，最终执行轨迹合批评分；感知只构造一次最终物理视图，先筛粗格再展开细格，按实际障碍依赖更新距离场，并使用隔离reviewer的既有双worker。宏步只编码必要观测边界，保留提前终止快照。规划采样数、预测时域、物理查询点与判据不变；实现证据见[优化记录](../experiments/t11_sampling_optimization_2026-10-22.md)。

## Active 配置清单

| 角色 | 当前选择 | 使用边界 |
| --- | --- | --- |
| 最新规划实验入口 | [upstream_1km_mppi_grid_sat_v1.yaml](../../configs/experiment/upstream_1km_mppi_grid_sat_v1.yaml) | 已观测大坑SAT、软查询合并、512采样×3迭代；短目标10／10、0失败；已用于本轮从零百万步训练 |
| 前轮车身覆盖入口 | [upstream_1km_mppi_body_geometry_v1.yaml](../../configs/experiment/upstream_1km_mppi_body_geometry_v1.yaml) | 零规划余量、0.2米单元中心覆盖、512采样×4迭代；保留圆盘大坑查询供复现 |
| 已结束训练的实验入口 | [upstream_1km_crater_classes_v1.yaml](../../configs/experiment/upstream_1km_crater_classes_v1.yaml) | 百万步续训原参数、512采样×2迭代；供历史恢复 |
| 当前学习参数 | [upstream_1km_far_crater_reward_1m_v1.yaml](../../configs/training/upstream_1km_far_crater_reward_1m_v1.yaml) | 奖励与评估安排沿用；本轮新网络／Adam、100万步独立预算 |
| 障碍／地形输入 | [upstream_1km_crater_classes_v1.yaml](../../configs/terrain/upstream_1km_crater_classes_v1.yaml) | 0.2 m分档标签、0.8 m派生驾驶地形；停车仍查原始地形 |
| 物理设计依赖 | [upstream_vehicle_proxy_v2.yaml](../../configs/design/upstream_vehicle_proxy_v2.yaml) | 由有效配置解析；不是独立训练入口 |
| 父配置及地图／atlas | 实验继承链、训练 YAML 的四张 scene、障碍 YAML 的 atlas | 共享依赖；不因父配置存在而授权另开旧训练 |

T11训练与独立评估入口显式继承实验YAML中的MPPI轮数，已修复旧工具固定覆盖为2轮的问题；通用历史局部诊断仍默认2轮。本轮实际kernel与YAML均为3轮，证据见[启动审计](../experiments/t11_mppi_sat_long_training_2026-10-25.md)。

这是一份**当前使用边界清单**，不更改加载器允许的历史配置。原 YAML 的 `training_paused: true` 由训练入口在已授权的学习 runtime 中显式处理；不要全局改父配置来消除门控。实际解析配置、版本和命令保存在 run 的 `config/`。

当前 run、剩余预算和 TensorBoard 由[当前状态](../current_status.md)维护；原始参数及准入见[训练记录](../experiments/t11_obstacle_input_training_2026-10-22.md)，优化源码承接与恢复命令见[续训记录](../experiments/t11_optimized_training_resume_2026-10-23.md)。旧实验 run ID 和脚本中的 T11／curriculum 名称保留来源与恢复兼容，固定模式实际关闭课程及晋级。

## 当前陨石坑分档与奖励

候选入口 [upstream_1km_crater_classes_v1.yaml](../../configs/experiment/upstream_1km_crater_classes_v1.yaml) 引用[同名地形配置](../../configs/terrain/upstream_1km_crater_classes_v1.yaml)，使用 `local_obstacle_input_v2`。它保留现有感知、私有格、MPPI、环境和奖励职责，已用于当前短训。上一轮obstacle_input_v1实验／训练配置和checkpoint保留。

按坑口直径划分：小于 0.5 m 不计坑标签惩罚，0.5–不足 1.8 m 可通过并轻罚，达到 1.8 m 为不可通行大坑。这两个边界是 Proxy 假设。驾驶查询从底图、岩石和残差层重算，移除全部人工坑层；停车和最终成功条件继续查询原始含坑地形。出生筛选与驾驶使用相同表面，并先排除实际车身侵入大坑的位置。岩石语义保持。

局部标签经原视场规则进入私有格，再向原三通道观测、CPU／CUDA MPPI 和执行奖励提供分档风险；隐藏标签不进入距离场。软风险保留圆盘覆盖估距；大坑硬检查在历史配置使用圆盘，在最新显式配置使用已观测栅格SAT，候选预筛和最终轨迹审查保持一致。环境在执行后用实际车身矩形与坑口圆精确判定，不加安全余量；失败仍通过 `terrain_failure` 和既有终端奖励结算。预测几何与实际任务判定分别记录。

2026-10-25的[MPPI修复](../experiments/t11_mppi_execution_repair_2026-10-25.md)在现有规划职责内提供 `footprint_margin_m`／`footprint_cover_spacing_m`。显式新配置使用零额外余量、0.2米单元中心圆盘，完整覆盖实际车身；MPPI采样与最终重生成审查共享这一几何。仅浅复制局部视图的覆盖点，已感知tile、观测、奖励和物理失败判据保持。未设参数时继续使用原覆盖及metadata；新几何身份不允许静默恢复旧规划器热状态。新配置的噪声0.2／0.1、目标权重4、4次迭代同样只影响MPPI，没有新增避障子模块。

中坑风险幅值 0.1，规划代价系数 0.5/s；执行奖励系数 0.05/s，按穿入比例和四车均值累计，每团队回合最多扣 2 分，额度参与 reset／恢复。新配置、派生 atlas、规划／奖励 metadata 及出生 atlas 身份隔离行为；旧 v1 继续复现。尺寸、信息时序和恢复细节见[分档接口](../interface_spec.md#陨石坑分档候选接口)。本次训练另以`continuous_cost_scale=0.02`缩放旧持续非势成本，事件奖罚、势差和中坑项保持；新网络初始化，前后60秒固定窗口评估，不作完整收敛判定，见[短训记录](../experiments/t11_crater_reward_short_training_2026-10-23.md)。

## 活跃代码职责索引

2026-10-25的[查询与迭代优化](../experiments/t11_mppi_query_optimization_2026-10-25.md)在既有实现中合并普通障碍／中坑的变换与查格，评分逐值保持；大坑通过`large_crater_query=observed_grid_sat_v1`选择实际车身对已观测占用格的SAT。最新配置为512采样×3轮，其他噪声与权重保持。`make_collector`使用精简候选回传，完整采样记录可由`env.pool.record_samples=True`启用。最终大坑审查以`include_known=False`省去调用方未使用的完整已知范围计算，实际感知、停车和失败检查保持。6,000步对照与109项检查完成，整段耗时减少38–45%；没有新增行为模块。

以下 `local_goal` 位于 [multi_rover_gathering/local_goal](../../source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/local_goal/)，`terrain_data` 位于[地形包](../../source/lunar_rover_tasks/lunar_rover_tasks/terrain_data/)。

| 职责 | 当前源码 |
| --- | --- |
| 训练入口与固定月面调度 | [run_upstream_t11.py](../../scripts/run_upstream_t11.py)、[upstream_curriculum_experiment.py](../../scripts/upstream_curriculum_experiment.py) |
| 多进程采样、checkpoint、实时日志 | [upstream_process_rollout.py](../../scripts/upstream_process_rollout.py)、[upstream_checkpoint.py](../../scripts/upstream_checkpoint.py)、[training_tensorboard.py](../../scripts/training_tensorboard.py) |
| 环境与任务 | `local_goal/upstream_environment.py`、`upstream_task.py`、`upstream_proxy.py` |
| 感知、私有格与输入身份 | `upstream_sensing.py`、`upstream_grid.py`、`upstream_obstacle_spec.py` |
| 观测、合法消息与邻车预测 | `upstream_observation.py`、`upstream_communication.py` |
| 动作分布、网络与意图桥接 | `action_ablation.py`、`actions.py`、`learning.py`、`upstream_action_bridge.py`、`upstream_ablation_learning.py` |
| 奖励、决策宏步、采样、优势与 PPO | `upstream_reward.py`、`upstream_decision.py`、`upstream_rollout.py`、`upstream_training.py`、`upstream_skrl.py` |
| 规划协议与工厂 | `upstream_planning.py`、`upstream_factory.py` |
| MPPI 当前实现及共享基类 | `upstream_mppi.py`、`upstream_mppi_precision.py`、`upstream_mppi_calibration.py` |
| 执行、延迟与足迹判定 | `upstream_execution.py`、`upstream_queue.py`、`upstream_timing.py`、`upstream_footprint.py` |
| 地图、增强与障碍 atlas | `terrain_data/batched.py`、`nasa_sfd.py`、`obstacle_atlas.py`、`scene_build_bounded.py` |

## 基线、历史与诊断边界

| 分类 | 范围 | 处理 |
| --- | --- | --- |
| ACTIVE | 上述正式入口、配置和实际依赖 | 按当前实验维护 |
| BASELINE | NMPC、line／quintic、47 原语、A0/A2/A3 分支 | 后续比较，当前不排队；共享代码继续保留 |
| LEGACY_REPRODUCIBILITY | exp167、旧尺度 T03–T10、旧 T11／课程／strict 启动组合 | 按对应历史配置和证据复现，不能作为当前默认 |
| DIAGNOSTIC | 性能剖析、局部回放、动作／价值拟合和原矩阵工具 | 按具体问题有界使用；不成为自动训练前置流程 |
| DEAD | 本轮未证明任何代码属于此类 | 不删除 |

分类针对入口与行为分支，不能据此整文件搬走。`mppi_upstream_v4` 实际继承 precision／calibration 共享实现；`action_ablation.py` 同时服务 A1 和其他表示。工厂按选择惰性构造求解器，历史可选后端保持原边界。

T15 的引用审查、配置与文档治理见[清单](../t15_governance.md)。大文件拆分由实际维护缺口触发；没有恢复身份和受影响行为验证的迁移不进入运行链。
