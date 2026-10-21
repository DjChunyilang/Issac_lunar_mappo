# MPPI CUDA 接通与耗时诊断

日期：2026-10-11。状态：本机GPU可用，独立CUDA配置和真实感知K512集合pilot已运行；完整CUDA数值门槛尚未通过。历史T07–T10仍是CPU验收，正式训练继续暂停。

本页为首次v1诊断快照；后续独立stable_v2修复、完整CUDA数值与负载验收及GPU闭环评测见[后续验收记录](mppi_cuda_validation_2026-10-11.md)，原失败和产物保留。

## 慢的原因

此前按计划延期CUDA，使用 `.venv_isaaclab` 的 `torch==2.10.0+cpu`，T07–T10配置继承 `local_goal.planner_device: cpu`。GPU没有参与MPPI规划。实际驱动和硬件健康：RTX 2060、驱动580.178.04；既有 `.venv_terrain_cuda` 的PyTorch 2.10.0+cu126、CUDA runtime 12.6能够执行真实GPU张量计算。`nvidia-smi` 显示的CUDA 13.0是驱动能力信息，不是当前PyTorch runtime。

只读复核冻结CPU证据，MPPI样本地形／邻车硬复核占主要耗时：

| 记录 | 样本复核占比 | 预测占比 | 代价占比 |
| --- | --- | --- | --- |
| T09 K512 | 79.05% | 10.63% | 9.02% |
| T09 K1024 | 80.47% | 9.83% | 9.05% |
| T09 K2048 | 80.99% | 9.54% | 9.10% |
| T10 K512 | 81.10% | 9.33% | 9.05% |

T09分母为各设置1000次规划累计时间，890次调用包含采样内核计时；T10取最新attempt，并将复制到多车行的共享内核计时按批次去重。2820个采样批次累计33710.82 s，占48个MPPI回合批次累计时间的96.33%。这些是累计进程时间，不等于并发运行的实际经过时间。最终CPU轨迹复核仅占采样内核累计时间约0.07%。具体原始路径、分母和去重规则见[CPU profile](../../outputs/runs/mppi_cuda_validation/contract_seed23_rtx2060_20261011_03/metrics/cpu_profile.json)。

该热点已使用Torch并支持CUDA。小查询块会产生大量GPU kernel调用：K512一次段内复核有69120个查询点，chunk256对应270个查询块。扩大块大小减少调度开销，但显存随批量规模、查询块和本地记忆样本量增长。NMPC、47类枚举、OSQP、传感、通信和证据写入仍在CPU，不能由MPPI加速推断整个T09/T10等比例加速。

## 已完成的接通

- 复用既有CUDA环境，仅补充 `osqp==1.0.4`；`pip check`通过。CPU环境安装前后freeze逐字一致，没有更换驱动或CPU PyTorch。
- 新增[安装入口](../../scripts/install_planner_cuda.sh)及[可选依赖](../../requirements-planner-cuda.txt)，实际执行安装入口并验证GPU计算和OSQP版本。
- 新增独立[T07 CUDA配置](../../configs/experiment/t07_mppi_terrain_neighbor_cuda.yaml)和[T10 CUDA配置](../../configs/experiment/t10_gathering_no_learning_cuda.yaml)，规划设备为CUDA，query chunk为1024；代理环境及OSQP仍在CPU，训练暂停。
- 新增[同步性能入口](../../scripts/benchmark_mppi_devices.py)和[真实GPU契约测试](../../tests/test_mppi_cuda.py)。保持既有MPPI代价、采样、动力学、硬复核和历史CPU配置。

安装日志和依赖前后记录位于 `outputs/runs/mppi_cuda_validation/diagnostic_seed23_rtx2060_20261011_01/logs/`；[依赖检查](../../outputs/runs/mppi_cuda_validation/diagnostic_seed23_rtx2060_20261011_01/metrics/dependency_verification.json)证明CPU环境未改变。

## 同噪声同步性能面板

run：`outputs/runs/mppi_cuda_validation/diagnostic_seed23_rtx2060_20261011_01/`。使用冻结平坦合法局部输入，441个本地地形样本，两轮更新；CPU和CUDA均在同一cu126环境、单Torch线程运行，注入相同seed23噪声并记录SHA。每组先预热计算内核，再测三次冷规划器状态调用；每次整调用前后CUDA同步。计时包括CPU打包、传输、预测、代价、更新及最终host复核，构建／预热单独保存。17组测量完成，全部输出有效。

| K，B=1 | CPU chunk256，中位秒 | CUDA chunk1024，中位秒 | 加速比 |
| --- | --- | --- | --- |
| 512 | 1.3845 | 0.1979 | 7.00 |
| 1024 | 2.7282 | 0.2867 | 9.51 |
| 2048 | 5.3910 | 0.4857 | 11.10 |

B1同噪声配对全部通过既定控制容差及有效性／反馈类别对照，最大控制误差 `1.40e-6`。这只是该固定输入的配对结果，不能代替后述复杂邻车数值契约。

K512/B1查询块面板：CUDA chunk256为0.4824 s，chunk1024为0.1979 s，chunk2048为0.1524 s；对应Torch峰值分配显存5.03／11.69／20.73 MiB。chunk2048在这个小记忆场景更快；独立配置保留1024，给大本地记忆和批量留空间，未修改历史CPU配置。

| GPU B | K512，中位秒 | K1024，中位秒 | K2048，中位秒 |
| --- | --- | --- | --- |
| 1 | 0.1979 | 0.2867 | 0.4857 |
| 4 | 0.2851 | 0.4371 | 0.7682 |
| 16 | 0.8057 | 1.4256 | 2.6785 |
| 64 | 3.0104 | 5.4837 | 10.4217 |

最大Torch峰值分配为B64/K2048的1257.07 MiB，峰值保留量另见对应JSON；不包括驱动上下文等全部显存。面板内存仅适用于441样本的本地记忆，不能推广到全部地图。具体硬件、噪声、配置、轨迹和重复测量见[性能汇总](../../outputs/runs/mppi_cuda_validation/diagnostic_seed23_rtx2060_20261011_01/metrics/device_benchmark.json)。每组仅保存末轮首实例原始采样，覆盖范围明确记录。

既有MPPI内部分阶段GPU计时没有同步，只反映异步dispatch，不能作为真实GPU阶段耗时。三次短测不足以建立完整p95/p99或0.2 s实时门槛；本次加速结论仅来自同步整调用时间，没有推断完整任务加速比。

## 实际感知集合pilot

run：`outputs/runs/mppi_cuda_validation/gathering_seed23_rtx2060_20261011_01/`。通过独立T10 CUDA配置，实际传感、通信缓存、本地记忆、合法脚本、差速执行和CPU安全投影运行；没有注入稠密地图、Actor/Critic或优化器。固定K512、两轮更新，两环境×四车批量；第二批只启用一个环境。seed23／31／47的平坦紧凑场景全部自然结束，实际25步驻留，无碰撞及执行限制违反。

| 队形seed | 实际模拟秒 | 最大车间距离，m | 离散度，m² | 驻留步 |
| --- | --- | --- | --- | --- |
| 23 | 9.0 | 1.7195 | 0.7200 | 25 |
| 31 | 9.0 | 1.7084 | 0.6979 | 25 |
| 47 | 9.0 | 1.7119 | 0.7130 | 25 |

两批计算墙钟分别约11.71 s／7.58 s，不把共享批次时间重复累计为两回合时间。18个公开生命周期快照均记录CUDA规划／RNG设备，9份原始采样均为512×15×2。配置明确 `simulation.device: cpu`、`planner_device: cuda`，证明GPU接入真实环境而非仅运行张量示例。结束判据与证据见[pilot检查](../../outputs/runs/mppi_cuda_validation/gathering_seed23_rtx2060_20261011_01/metrics/pilot_acceptance.json)。

本次限定最多30宏步，任务截止仍为180 s；三个回合在预算内成功结束。不是144回合GPU矩阵，也不替代既有CPU Gate B或T10验收。现有summary／manifest的通用 `cuda_validation: deferred` 未改；CUDA接通和本次有限覆盖以独立指标为准。正式入口的源码匹配门控保持有效，不能将旧CPU热状态或不匹配的Gate B直接用于新GPU正式验收。

## 数值门槛与待办

最终真实GPU契约run：`outputs/runs/mppi_cuda_validation/contract_seed23_rtx2060_20261011_03/`，**17项中16通过、1失败、0跳过**。覆盖本地地形支持／未知／等距冲突、查询块、同噪声规划、RNG身份隔离、顺序、reset、同设备恢复和跨设备热状态拒绝。必要失败保留，未删除夹具、降低容差或标记xfail。

失败为 `test_cpu_cuda_same_injected_noise[cross_offset]`：控制最大误差 `1.7166e-5`、状态最大误差约 `1.7929e-5`，未满足既定 `atol=1e-5, rtol=1e-5`。硬可行性和usable掩码完全相同；float32邻车距离／风险代价误差经过两轮权重更新放大。相同案例的float64控制误差仅 `2.42e-14`。详见[最终JUnit](../../outputs/runs/mppi_cuda_validation/contract_seed23_rtx2060_20261011_03/metrics/pytest_cuda.xml)和[逐轮误差诊断](../../outputs/runs/mppi_cuda_validation/contract_seed23_rtx2060_20261011_02/metrics/cross_offset_float32_boundary.json)。

该失败涉及浮点一致性，GPU执行、地形硬边界和本次集合pilot并未因此失败；**完整CUDA数值门槛仍未通过**。内部双精度邻车代价可作为独立版本修复方向，需保持公式、噪声和执行边界，实际复测该失败及批量契约，不能仅靠CPU参考或放宽误差门槛宣布修复。本轮没有改动规划算法。

后续补齐数值契约、复杂地形／邻车性能和完整GPU评测覆盖，再考虑GPU Gate B／完整集合、真实足迹及PhysX。CUDA不会修复既有坡地终端区域不满足或NMPC持续求解拒绝；这些功能问题见[T10记录](t10_gathering_validation_2026-10-10.md)。本轮不重跑昂贵千例／144回合、不创建训练checkpoint，正式训练继续暂停。复现命令见[诊断手册](../runbooks/planner_diagnostics.md)。

Ruff F821、仓库卫生、数学分隔符、修改文档的222个本地链接、Shell语法及 `git diff --check` 通过；测量后实际任务／诊断源码指纹未变，记录见[静态检查](../../outputs/runs/mppi_cuda_validation/diagnostic_seed23_rtx2060_20261011_01/metrics/static_acceptance.json)。没有修改NMPC、动力学、通信或安全实现，因此不重复既有18项真实acados回归。最终[诊断状态](../../outputs/runs/mppi_cuda_validation/diagnostic_seed23_rtx2060_20261011_01/metrics/diagnostic_acceptance.json)将GPU接通／有限性能覆盖和未通过的完整CUDA门槛分别记录。
