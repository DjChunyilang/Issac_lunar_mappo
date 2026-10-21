# 完整 CUDA 验收与后续诊断计划

日期：2026-10-11。状态：CUDA数值／负载、新GPU Gate B、T10完整矩阵的简单集合功能及逐步证据审计已通过；复杂失败保留，正式训练暂停。依据[原规划](../writing-block.md)的T06–T10及[第五阶段冻结要求](implementation_plan.md)。三项结论分别记录，实际证据见[本轮CUDA验收](experiments/mppi_cuda_validation_2026-10-11.md)。

本轮66项CUDA契约及695项联合回归全部通过、0失败、0跳过，18项真实acados实际执行通过。runtime严格验收完整通过36个GPU性能面板、9个B1同噪声CPU配对及27个复杂输入对照；独立容量和同步阶段诊断通过。契约run为`contracts_v2_seed23_rtx2060_20261011_02`，runtime run为`acceptance_v2_seed23_rtx2060_20261011_02`，均位于`outputs/runs/mppi_cuda_validation/`。旧v1失败与历史CPU结果仍保留；GPU Gate B与T10完整矩阵由下述独立生产评测证明。

## 前置事实与交付边界

[首次CUDA诊断](experiments/mppi_cuda_diagnostic_2026-10-11.md)已确认本机RTX 2060及cu126环境可用，GPU真实感知K512集合pilot三例成功；17项GPU契约中保留一项float32邻车数值对照失败。旧T05–T10 CPU验收及原始结果保持原状，平坦性能面板和三例pilot不能替代本轮完整验收。

- 保持`finite_task_v2`、`polar_hold_v2`、`local_intent_clock_v3`、时域15×0.2 s、两轮更新及原规划代价、执行限制和通信语义。
- `simulation.device`仍为CPU；Torch MPPI在CUDA计算，感知、通信、代理动力学、OSQP和最终轨迹复核在CPU。CUDA不可用时明确报错，不静默回退。
- 不加载Actor／Critic、学习权重或优化器，不生成训练checkpoint。真实车体足迹、24 m通信、出生分层、Isaac／PhysX及实时认证继续独立待办。
- 不覆盖旧配置、失败日志、run、热启动状态或用户原规划。跨CPU／CUDA热状态恢复仍明确拒绝。

## 版本与独立配置

地形／邻车数值修复使用独立适配器`mppi_terrain_neighbor_stable_v2`、状态版本2、数值版本`fp32_fp64_accum_v2`。运行预测和控制仍为float32，邻车位置差／距离／风险与重要性权重累计采用float64；公式、原始扰动及硬边界保持不变。旧`mppi_terrain_neighbor_v1`及其失败夹具保留，旧热状态不能静默迁移。基础MPPI保留`mppi_base_v1`。

本轮已新增以下独立配置，实际解析配置、版本元数据和源码指纹分别进入诊断产物：

| 配置路径 | 诊断用途 |
| --- | --- |
| `configs/experiment/t06_mppi_base_cuda.yaml` | 基础CUDA运动、采样、更新、停车及回退；基础适配器v1 |
| `configs/experiment/t07_mppi_terrain_neighbor_cuda_v2.yaml` | 独立稳定数值版地形／邻车CUDA |
| `configs/experiment/t08_safety_projection_cuda_v2.yaml` | CUDA规划共用CPU安全投影 |
| `configs/experiment/t09_planner_benchmark_cuda_v2.yaml` | 完整七设置固定评测 |
| `configs/experiment/t10_gathering_no_learning_cuda_v2.yaml` | K512完整无学习集合矩阵 |

所有配置显式保持训练暂停。query chunk与显存分块规则版本化并记录；改变设备、精度、适配器或数值规则时另起冷启动run。

## 执行顺序与出口

### 1. 修复数值与设备契约

保留`cross_offset`原失败输入和既定误差门槛，追踪运动、地形／邻车代价、重要性修正及两轮权重更新。不得删除案例、降低容差或以xfail代替通过。

- float64解析参考采用`atol=1e-10、rtol=1e-8`；运行float32的CPU／CUDA同噪声、单实例／批次／换序／分块对照采用`atol=1e-5、rtol=1e-5`。
- 验证中点运动、幅值／轮速／变化率限制、原始噪声重要性修正、极端代价、全部无效样本、非有限输入及空批次。
- 验证MOVE、移动HOLD、过期请求、未知／不可通行、段内邻车交叉、稀疏／过期预测和完整回退顺序；加权结果必须重新预测与复核。
- 验证每实例RNG、输入顺序、稀疏批次、异步reset、同设备恢复及CPU／CUDA热状态拒绝；恢复拒绝不得部分修改实例状态。
- 在真实CUDA上核对张量设备、协议维度、sender绑定及Oracle／未通信真值／其他环境变化的执行不变性。

出口为全部必要CUDA契约实际通过、0失败、0跳过，数值失败诊断与修复版本均可追溯。CUDA可用与完整数值通过分别记录。

本步已完成：66项完整CUDA契约实际通过，原`cross_offset`在既定容差下通过；旧v1的失败产物保留，修复仅在稳定v2生效。

### 2. 完整负载与复杂场景诊断

冻结K=512／1024／2048、B=1／4／16／64、种子23／31／47及实际注入噪声，覆盖平坦、中坡、坑边、窄通道、迎面、交叉、HOLD、倒车和未知边界。另覆盖空记忆、非均匀本地记忆和4096点容量边界，不能用441点平坦面板替代实际感知负载。

查询沿内存块循环，车辆和候选维度采用张量计算；分块不能跨越未知空洞或改变保守等距规则。按B、M和显存预算限制查询块，完整运行最大批次，保留OOM或拒绝日志；不能通过减小K或B补写原矩阵通过。

每组独立记录构建、预热、CPU打包、数据传输、GPU预测／代价／更新／复核及同步端到端耗时，保存p50／p95／p99、吞吐、GPU峰值allocated／reserved显存、CPU内存和实际重复次数。整调用计时前后CUDA同步；GPU阶段使用明确的同步计时方法，旧异步dispatch计时不作为阶段性能结论。重复数较少时说明分位数统计边界，不宣称实时预算通过。

出口为完整K×B及场景覆盖、同噪声数值对照、资源边界和合法输入输出通过。性能结果按设备、负载和精度报告，不推断其他服务器吞吐。

本步已完成：flat／slope／cross各运行全部K×B，共36GPU面板，B1共9组CPU配对；九类×种子23／31／47共27组K128复杂输入对照通过。B64／4096点记忆的容量查询边界及同步阶段诊断通过；容量查询不是B64／K2048／4096点整规划性能。完整九类各K的规划器比较已在下述新T09生产评测中完成。

### 3. 新GPU Gate B生产评测

实现、配置、CUDA门控和诊断脚本冻结后创建新run。完整执行原T09范围：

- 单次规划1000例×七设置，共7000条；九类各110例及10个固定回归，不按表现删除输入。
- 重复性30例×七设置×种子23／31／47，共630条。
- 局部闭环18例×七设置×安全开关，共252回合；B4／16共14面板。
- 七设置仍为line、quintic、原生47枚举、真实NMPC及MPPI512／1024／2048。四种基线维持CPU，MPPI采用独立CUDA稳定数值版；所有设置消费相同保守局部输入。

单次规划的安全开关复用同一原生输出；局部闭环独立恢复相同初态和RNG。保持原100步预算及实际停车／25步驻留判据，停车前缀和刹车不能计作目标完成。

新的必要JUnit与完整源码快照绑定，实际执行18项真实acados；失败或必需skip不得通过。历史CPU事实可按相同实际依赖引用，但不能把旧结果复制到新run或重贴源码标签。新`gate_b_cuda`与历史`gate_b_cpu`分开，核对设备、适配器、精度／数值版本、公共参数及完整覆盖。

出口为完整覆盖、可重复性、必要回归与失败分类通过，预定简单平坦NMPC和MPPI512闭环全部成功。复杂碰撞、不可行刹车及未知区域停滞保留实际结局，不宣称普遍安全。

本步已通过：[新GPU Gate B严格汇总](../outputs/runs/t09_planner_benchmark/acceptance_seed23_cuda_v2_20261011_01/metrics/summary_0000.json)确认`gate_b_cuda=true`、`gate_b_cpu=false`，7000单次、630恢复、252局部回合、14面板完整覆盖，重复性、简单闭环、设备与生产源码核验通过。原CPU Gate B证据保持原状。

### 4. 正式K512 CUDA集合诊断

新GPU Gate B通过且当前实际依赖源码匹配后，运行六类×三个种子×NMPC／MPPI512×安全开关×96／180 s，共144个完整回合；不使用pilot的`--max-macros`预算。

本步完整矩阵及简单集合功能已通过：[正式汇总](../outputs/runs/t10_gathering_no_learning/acceptance_seed23_cuda_v2_20261011_01/metrics/summary_0007.json)确认144回合完整、18例简单门槛全部成功，`t10_cuda_functional_pass=true`。固定K512、两轮，保持96／180 s和安全开关；NMPC 56／72成功、MPPI 54／72成功，无碰撞和执行限制违反。[全量逐步审计](../outputs/runs/t10_gathering_no_learning/acceptance_seed23_cuda_v2_20261011_01/metrics/trace_audit_0000.json)通过117744次执行记录、4108个宏步／公开快照及710／710个要求采样事件，错误为0。

保持`leader_slot_v1`、真实传感、通信缓存、本地记忆、代理动力学、任务终止、两环境×四车及异步结束。固定K512、两轮更新，不注入稠密地图；通过公开生命周期保存可恢复状态，不含求解器对象或训练checkpoint。

功能出口为安全开启180 s的三类简单平坦九例，NMPC与MPPI512各9／9完成集合、无碰撞、无执行限制违反。其余回合完整执行并分类；坡地终端区域不满足、NMPC持续拒绝及其他复杂失败不能因GPU加速或平坦通过改写为成功。

实际失败保留24个终端地形不满足、4个规划拒绝及6个任务截止。新增6个MPPI NPB失败与本地记忆量化后当前位置失去查询支持有关，诊断距离0.20 m超过0.155 m支持半径；CUDA仍正常工作，未知边界不放宽。旧CPU成功不能推定新CUDA轨迹同样完成；本轮定位证据见[验收记录](experiments/mppi_cuda_validation_2026-10-11.md)，后续修正另起run。

[公开状态重放](../outputs/runs/mppi_cuda_validation/public_state_replay_v2_seed23_rtx2060_20261011_01/metrics/cuda_public_state_replay_0000.json)已通过：六类实际场景、每类两次冷构造恢复及一次信息隔离重放、各5底层步；控制重复误差0，与历史轨迹符合既定容差。异步reset、Oracle／其他环境变化及稀疏batch不改变合法执行，同设备CUDA随机状态恢复通过，源码绑定一致。

另审核全部逐步名义轨迹、实际下发控制、安全结果、修正广播和下一次`previous_control`一致性；原始采样按首次／每25步／新失败事件保留并报告覆盖率。简单门槛未满足时，完整矩阵执行状态与功能验收状态分别记录。

## 排程、证据与交接

单GPU独占运行CUDA测试、资源面板和MPPI评测。CPU基线可在独立物理核并行，先核对CPU内存高水位；固定Torch／BLAS线程数，避免争用污染延迟。分片互不重叠，续跑严格校验配置、源码、依赖、场景与版本指纹；运行中断不计规划失败。

产物遵守[输出管理](references/output_management.md)，另建`outputs/runs/<experiment_id>/<run_id>/`，不覆盖首次CUDA诊断和旧CPU证据。保存解析配置、设备／驱动／依赖、全部版本、场景／地图身份、种子和噪声SHA、源码指纹、命令、JUnit、逐轮误差、恢复状态、采样、轨迹、安全残差、结束原因、回报分量、失败分类、耗时与资源。

每阶段完成文档链接、数学分隔符、仓库卫生、Ruff F821和`git diff --check`。最终分别报告完整CUDA契约／负载验收、GPU Gate B、T10 CUDA无学习集合的真实状态。缺例、必要测试失败／跳过、资源边界失败或简单集合门槛不满足时，相应严格验收保持未完成；正式训练持续暂停。

运行入口见[规划诊断手册](runbooks/planner_diagnostics.md)，后续学习契约准备仍见[T11交接计划](t11_handoff_plan.md)。本计划不自动解除训练暂停。
