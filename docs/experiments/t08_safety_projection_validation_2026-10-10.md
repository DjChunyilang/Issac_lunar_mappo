# T08 代理安全投影验收

日期：2026-10-10。状态：最终联合回归通过；静态检查及门槛结果见下文。计划见[实施入口](../implementation_plan.md)，实现见[接口](../interface_spec.md)和[规划架构](../architecture/planning_pipeline.md)。正式训练暂停。

## 实现与边界

独立配置 `configs/experiment/t08_safety_projection.yaml` 启用 `discrete_cbf_qp_v1`，OSQP 1.0.4、float64、无松弛、固定参数，最多3次线性化。解析中点Jacobian、既有轮速／幅值／变化率、9点本地地形与sender邻车复核；合法请求的HOLD与过期限幅刹车，不重造运动目标。规划valid与安全model_feasible分开。

实际下发反馈下一规划；通信仅发布实际首控制与停车尾段复核成功的执行预测，失败撤销轨迹。旧配置保持disabled。公开安全生命周期不导出求解器，异步重置隔离环境。

代理圆半径0.14 m，沿用已有通信预测半径，地形仍中心路径支持；不是完整足迹、真实月面或连续安全认证，未开展PhysX。

## 证据与失败记录

开发run：`outputs/runs/t08_safety_projection/acceptance_seed23_cpu_20261010_01/`。所有先前失败与重跑保留：`initial.log/xml`、`contracts.log`、`safety_solver.xml`、`stage5_retry1.xml`、`shared_planners.xml`，以及对应logs。首次测试暴露HOLD速度上限被误作瞬时执行限幅，已修正；未设置acados环境变量的失败原样保留，随后实际求解器重跑通过。恢复测试暴露局部记忆／缓存与快照引用共享，已改深拷贝并验证同一状态可反复恢复。

`stage5_retry1.xml` 为38项新增安全／诊断回归；真实NMPC与MPPI共用层追加两项在 `shared_planners.xml` 通过。以上不能代替最终联合验收。

联合命令设置 `ACADOS_SOURCE_DIR`、`LD_LIBRARY_PATH`、OMP／OpenBLAS单线程，覆盖安全、阶段诊断、基线、MPPI、协议、P0、动作、任务、通信、配置与地图测试。JUnit `contracts_stage5_final.xml`；源码快照与manifest保留。既有18项真实acados必须全部实际通过，非必要CUDA一致性继续延期。

## 结论

最终冻结源码联合回归已经完成；独立T09千例和T10集合不能计入T08契约测试数。正式训练继续暂停。

源码冻结后的最终run为 `outputs/runs/t08_safety_projection/acceptance_seed23_cpu_20261010_04/`，JUnit `metrics/contracts.xml` 及同名源码JSON；run02保留中断联合记录与首次MOVE／恢复测试。源码记录完善后重新完整执行，不复用中途变更源码的联合批次。

最终 `acceptance_seed23_cpu_20261010_04/metrics/strict_acceptance.json`：**T08代理安全投影契约通过**。625项通过、1项CUDA一致性延期、0失败，既有18项真实acados全部通过；新增阶段45项独立回归通过。`regression_acceptance.json`确认JUnit与当前源码一致，`static_acceptance.json`确认链接、数学、卫生、F821与diff检查通过。acados库指纹另存 `config/acados_libraries.json`。本结论不包含Gate B、集合、实时或连续安全认证。
