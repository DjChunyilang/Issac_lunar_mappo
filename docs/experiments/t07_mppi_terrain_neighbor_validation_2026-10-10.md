# T07：地形与邻车 MPPI 的 CPU 验收

日期：2026-10-10。状态：**T05已验收；T06/T07 CPU契约与局部闭环已验收；CUDA验证延期。** 正式训练继续暂停。范围依据[实施计划](../archive/implementation_plan_t05_t07_snapshot_2026-10-10.md)，前置证据分别见[T05](t05_planner_baselines_validation_2026-10-10.md)、[T06](t06_mppi_base_validation_2026-10-10.md)。不把原453项基线直接记作新阶段通过。

## 实现及版本

独立t07_mppi_terrain_neighbor配置使用mppi_terrain_neighbor_v1，保留planner_protocol_v1、finite_task_v2、polar_hold_v2和local_intent_clock_v3。注册表工厂、Torch共享内核与逐身份热启动／RNG接入现有环境；旧exp167/P0/T03/T04配置行为保留，训练及旧六单元评测门控保持。

TerrainView提供local_terrain_samples_v1只读快照，仅含本地位置、已有五维特征、时间和支持半径；Torch按256查询点分块，不持有全局DEM、core或Oracle。空记忆、支持外和空洞仍未知，MPPI显式选择保守等距规则，旧T03/T04默认视图查询保持历史行为。当前粗糙度通道无有效数据，不新增粗糙度代价；可通行度明确为1-risk代理。

地形／未知／邻车软代价独立开关，硬复核始终启用。每段9点，地形阈值沿用traversability≥0.25、坡度≤0.8；邻车直接使用sender绑定的合法预测及已有半径，不重复膨胀或补取真值。T07默认采用现有解析速度缩放，最小0.25。学习奖励、gamma/lambda及混合采样概率不改变。

最终加权均值重新预测并CPU复核；依序尝试可行采样、带停车尾段的前缀、复核旧计划和限幅刹车。输出代价按实际选中状态／控制统一批量计算，排除重要性项。HOLD与过期不采样；快照包含请求状态，跨CPU/CUDA热状态拒绝恢复。安全接口仍为projection_disabled，不构成CBF或连续安全认证。

## 验收命令与证据

Run：outputs/runs/t07_mppi_terrain_neighbor/acceptance_seed23_cpu_20261010_01/。完整命令、源代码和解析配置在run_manifest.json与config/中。核心联合回归：

```bash
ACADOS_SOURCE_DIR="$PWD/.venv_isaaclab/acados" \
LD_LIBRARY_PATH="$PWD/.venv_isaaclab/acados/lib" OMP_NUM_THREADS=1 \
.venv_isaaclab/bin/python -m pytest \
  tests/test_planner_baselines.py tests/test_mppi.py \
  tests/test_mppi_terrain_neighbor.py tests/test_planner_diagnostic_entry.py \
  tests/test_polar_hold.py tests/test_planner_protocol.py \
  tests/test_p0_learning_contracts.py tests/test_local_goal_nmpc.py \
  tests/test_local_goal_diagnostics.py tests/test_reward.py \
  tests/test_observation.py tests/test_termination.py \
  tests/test_tiered_communication.py tests/test_exp156_differential.py \
  tests/test_config_wiring.py tests/test_lunar_training_maps.py \
  --junitxml=outputs/runs/t07_mppi_terrain_neighbor/acceptance_seed23_cpu_20261010_01/metrics/contracts_acceptance_compat_final.xml \
  -ra --disable-warnings
```

最终**576项联合回归通过、另1项身份复现追加回归通过、0失败，1项既有CUDA一致性因CPU环境延期**（292.32 s）；**18项真实acados实际通过**，包含已有12项及新增4项MOVE配对、2项polar HOLD／过期。配对atol=1e-7、rtol=1e-6，逐项误差保留。新增测试范围为65项基线、24项基础MPPI、20项地形／邻车、10项诊断入口、3项配置和1项旧地形兼容检查；这是本轮实际执行结果。

float64解析运动／权重采用atol=1e-10、rtol=1e-8；float32单条／批量采用atol=1e-5、rtol=1e-5。覆盖换序、稀疏批次、查询分块、三个种子、配置分组、异步reset和恢复；全无效／极端代价、均值不可行而采样可行、旧计划复核通过／拒绝均有真实分支测试。Oracle与未通信真值的观测／邻车／MPPI控制不变性通过。

| 证据 | 内容 |
| --- | --- |
| metrics/contracts_acceptance_compat_final.xml、logs/contracts_acceptance_compat_final.log | 最终计数、必要真实求解器和CUDA延期原因 |
| metrics/source_snapshot_compat_final_start.json、metrics/source_snapshot_compat_final.json | 最终验收前后逐文件指纹及一致性 |
| metrics/*_loop_final.json及对应*_samples/ | 固定目标、选中轨迹、末轮原始采样／控制／实际速度、各代价、权重及噪声 |
| metrics/scenario_*_final.json | 风险、受阻、未知、交叉／迎面、停稳、缺失与过期消息诊断 |
| metrics/summary.json、run_manifest.json、config/ | 命令、源码／配置版本、种子、设备、资源与结论 |
| logs/static_checks.log | 链接、公式、卫生、F821、diff及源代码一致性 |

诊断无策略权重、优化器或训练checkpoint。联合回归中原有合成PPO更新仅属于学习契约测试，不构成训练恢复。

## 局部闭环与作用证据

给定合法局部观测，K=128、最多100步：前进(seed23)误差0.196731 m，转向(seed31)0.197998 m，倒车(seed47)0.162433 m，均达到0.2 m内。JUnit另保存seed23三类目标的误差及步数，并实际检验MOVE→HOLD的限幅停车。

现有传感链的平坦诊断使用配置生成flat_proxy后端，再通过原112点感知、局部记忆、通信和差速执行运行，**没有注入稠密测试地图**。两环境×四车、K=128、seed23，在13个底层步内全部到达0.8 m局部目标并停车；最终误差0.171167–0.196663 m，0碰撞、0执行限制违反。记录8实例真实合批、局部记忆数量与每步名义／下发控制。结果为局部目标到达，不是多车集合任务成功。

同噪声的风险开启／关闭测试改变候选排序和选中控制；邻车输入也改变规划。关闭软代价不关闭硬边界；单位场景验证未知／不可通行前缀有完整停车尾段，交叉与迎面受阻诊断明确刹车，不能计作到达。单次诊断在未知边界内选择完整可行短轨迹也只说明有效执行，不能用valid推断目标完成。

## 失败与证据边界

- 继承原起伏地形的八车0.8 m目标在100步仍有3车未到达，最大剩余误差0.425424 m；0.35 m补充诊断仍有1车未到达，剩余0.299406 m。两者无碰撞、无限幅违反，但local_pass=false。记录在metrics/sensor_loop_seed23_retry1.json和sensor_short_loop_seed23.json。不更改固定代价权重来隐藏结果；T09必须分析地形风险成本、随机更新与目标推进的取舍。
- 首次传感脚本未将HOLD连续占位清零，触发动作契约拒绝；保留logs/sensor_loop_seed23.log及已有采样，修正后重新运行。平坦场景单独明确地图身份，不替代上述失败记录。
- T05新增真实移动HOLD的测试初版只期待speed_limit，但真实求解器合法返回solver_failure并限幅刹车。修正为接受已定义失败反馈，保留logs/tests_extended.log和重跑证据，不把失败解声明为可行。
- 初轮联合回归558项通过、1项报错文案兼容失败；版本拒绝行为正确，修复planner revision文案后相关268项重跑通过，再冻结源码执行最终公共接口576项。原失败日志不覆盖。

已有原语历史可控性原始产物缺失不补写；这些CPU结果不构成规划器质量／速度排名、复杂地形普遍可达、完整集合、GPU吞吐或PhysX通过。

## 交接

T08安全接口与T09固定场景评测可进入下一阶段，继承起伏地形未到达样例作为必需评测输入。CUDA后续服务器补验同噪声数值、设备／精度、恢复边界、显存，以及K=512/1024/2048、B=1/4/16/64的同步计时与传输开销。本机延期项不记作通过。

Gate B、T10完整无学习集合、Isaac/PhysX及正式训练仍待后续门槛，不生成或推荐训练checkpoint。

收尾兼容审查将TerrainView.conservative()限定为MPPI的显式输入模式；默认视图与原live memory的等距选择保持一致。新增独立回归同时检查旧语义、保守规则和与实时记忆的隔离。公共视图修改后重新执行完整联合与18项真实acados，旧575项结果及其指纹保留。

RNG收尾将合法Python int与NumPy integer身份归一为同一数值键，避免同一(env_id,agent_id)因标量类型改变噪声。追加回归证明二者初次规划和reset后随机状态一致；基础MPPI25项重跑全部通过（47.82 s）。Python整数请求与已记录的128样本轨迹状态／控制逐值完全一致，见metrics/rng_python_reference_equivalence.json。此修改不影响NMPC依赖；576项联合源码指纹与随后RNG源码指纹分别保留，合计577项独立测试通过，1项CUDA延期。
