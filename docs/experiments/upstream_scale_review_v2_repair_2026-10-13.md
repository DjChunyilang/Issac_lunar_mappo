# S01复核缺口修复与批量查询接入v2

日期：2026-10-13。完成[前次诊断](upstream_scale_diagnosis_2026-10-12.md)的两项拒绝缺口，并在独立v2配置启用规则不变的分块查询。**两项缺口修复通过；查询优化已接入；S01完整环境及实时门槛仍未通过。** 正式训练暂停。

## 版本与目标行为

新入口为[upstream_scale_s01_review_v2.yaml](../../configs/experiment/upstream_scale_s01_review_v2.yaml)，继承原S01并显式启用 `execution_review_version=footprint_review_v2`、`terrain_query_version=batch_ties_v2`。原S00／S01 YAML、设计profile v1／v2和用户原规划不改；不改变车辆、速度、支持半径、坡度、制动、任务成功门槛。

旧配置解析缺省为原 `footprint_review_v1/scalar_ties_v1`，元数据字段及cache key计算保留本轮前逻辑，只用于历史复现；当前旧S01元数据与2026-10-12冻结run逐项相同。早期S00与当前规范的指纹差异来自此前将坡度速度缩放0.75纳入规范，该旧状态继续拒绝，不自动迁移。新配置记录 `geometry_version=footprint_upstream_v2`、两项实现版本及独立cache key，公开快照拒绝新旧互载。新公共 `FootprintReviewer` 默认v2；旧复核需要显式选择v1，不用于当前验收。

| 缺口／瓶颈 | v2目标与实现 |
| --- | --- |
| 邻车正半径不足以保证几何 | `FootprintNeighbor.for_spec`绑定完整规范指纹及 `identified_neighbor_upstream_v2`；构造和复核均检查至少包含两车有效包络。过小半径、缺失／不匹配规范或旧版本拒绝；合法既有不确定度不重复膨胀 |
| 可任意声明0.25–1速度增益 | 从同一只读地形重新计算坡向／通行增益、实际速度和中点后继。float64用atol=1e-10/rtol=1e-8，float32用1e-5/1e-5；未知拒绝。保守储备只是输入控制的时序约定，也执行同样核对 |
| 逐点等距处理 | 每块最多256个查询点，先统计邻点数量，限制每次ragged邻点合并最多65536条。批量球查询及分段归并保留危险高度、最大坡度、最小通行代理、原邻点顺序和原第4通道来源；空／稀疏／覆盖外语义不变 |

源码仍在 `local_goal/upstream_footprint.py`／`upstream_spec.py`；`scripts/diagnose_upstream_execution.py`从配置传播复核／查询版本，当前默认新配置。旧NMPC／MPPI执行模型及其acados约束未改，不据本轮基础修复宣称七设置已适配。

## 反例与回归

修改前先运行三项必要拒绝测试：过小半径、普通轨迹伪低速、保守储备伪低速，全部因错误放行而失败。原失败JUnit保存在 `outputs/runs/upstream_scale_s01/review_v2_red_seed23_cpu_20261013_01/metrics/contracts.xml`，日志复制到正式新run；开发117项通过记录另存，不与正式批次累加。

正式run：

```text
outputs/runs/upstream_scale_s01/review_v2_acceptance_seed23_cpu_20261013_01/
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_upstream_spec.py tests/test_upstream_execution.py \
  tests/test_config_wiring.py tests/test_cuda_evidence_gates.py \
  tests/test_upstream_review_v2.py \
  --junitxml=outputs/runs/upstream_scale_s01/review_v2_acceptance_seed23_cpu_20261013_01/metrics/contracts.xml

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv_isaaclab/bin/python scripts/diagnose_upstream_execution.py \
  --config configs/experiment/upstream_scale_s01_review_v2.yaml \
  --run-dir outputs/runs/upstream_scale_s01/review_v2_acceptance_seed23_cpu_20261013_01 \
  --resume \
  --junit outputs/runs/upstream_scale_s01/review_v2_acceptance_seed23_cpu_20261013_01/metrics/contracts.xml
```

**407项通过，0失败、0跳过**，含42项新增v2专项；覆盖半径阈值、指纹／版本、合法不确定度不重复加入、正反向／旋转／最小地形增益、错坡向、伪后继、float32、完整非平坦模型停车储备、1／7／256分块精确配对、危险等距、缺失支撑、车底／台阶以及密集重合点的分配上限。

原两项反例现在均拒绝：中心距1 m／半径0.1 m返回 `neighbor_radius_below_geometry`；平地伪0.25速度返回 `velocity_gain_inconsistent`。结果在 `metrics/review_v2_rejections.json`，两项修复通过字段在 `metrics/review_v2_acceptance.json`；来源绑定完整收集测试身份及生产SHA。

## 实际代理执行与性能边界

两环境×四车实际执行122个底层步，达到1／1.5／2／25/9 m/s，满速实际停车；无执行限制、物理包络或净空违反，热恢复精确一致，异步reset隔离。满速制动位移仍约9.368889 m，信息时间保守储备仍为40步、12.146667 m中心位移，加车头／余量为14.954167 m。不是实车或PhysX验证。

本次四档足迹复核墙钟分别约1.49／1.93／2.47／**3.26 s**。同一v2严格复核下，标量／批量查询的三次原生配对中位数为 **13.468／3.334 s**，约 **4.04倍**；完整选中状态／控制／速度／时间逐值一致。详见 `metrics/query_v2_benchmark.json`，其中同时保存进程峰值RSS、256点分块和65536邻点条目上限，以及独立计时runner源码SHA。

前次2.04 s只是无分块保护的诊断原型；新实现增加规范／模型复核和最坏邻点分配保护，不能直接作同条件速度排名。新v2已真实启用查询优化，但仍超过0.2 s控制周期及0.5 s延迟预算；`realtime_pass=false`，S01完整环境、S02、S03及Gate B继续false。

当前是给定12231点合法本地观测的离线执行，不把0.5 s注入仿真延迟当作墙钟达标；真实传感通信、实时延迟界、安全投影与规划器仍待接线。

## 交接

静态与保存边界检查通过：Ruff F821、仓库卫生、diff空白、默认25份及额外10份修改文档数学检查，292条本地链接有效。生产源码与正式run快照一致，旧S01合同逐项一致，原规划SHA不变；现有acados及旧执行依赖源码未改，未重复昂贵求解器／旧任务矩阵。结果保存为 `metrics/static_and_preservation_checks.json`。

修复和分块查询为[实施计划](../implementation_plan.md)提供可用的v2公共复核基础。后续40步规划协议／各适配器必须传递该规范与sender绑定预测，并从合法地形生成与执行一致的状态，不能迁入旧半径或热状态。真实acados23项与新尺度求解器、CUDA、校准、千例及144回合仍按S02–S05门槛完成；本轮未调用学习权重、优化器或创建训练checkpoint。
