# S02新尺度规划器接通与部分验收

日期：2026-10-13。状态：**40步规划核心、给定本地观测诊断及B1同噪声配对通过；完整S02、实际感知集合及T11学习尚未验收。**

依据[实施计划](../implementation_plan.md)，本轮目标是将新车辆尺度接入统一规划协议、完整基线和Torch MPPI，继续补齐[T11原计划](../t11_handoff_plan.md)的前置条件。未缩减100k→500k×3→2M×3的学习预算。本轮诊断没有学习环境交互或优化器更新，正式训练继续暂停。

## 实现与版本

独立运行配置为[upstream_scale_s02.yaml](../../configs/experiment/upstream_scale_s02.yaml)，显式启用`planner_execution_version=upstream_planning_v1`，保留`footprint_review_v2/batch_ties_v2`。旧S00／S01省略该字段时，元数据及缓存身份保持原样；旧15步路径继续独立。新工厂拒绝未接线版本及尚未完成的实时测量模式。

- `upstream_planning.py`提供40×0.2 s输入／输出与生命周期、实际控制生效时间、完整停车尾段、信息时刻储备及回退。只接受本地只读地形和sender绑定预测。
- `upstream_factory.py`延迟构造所选适配器。非NMPC路径在禁止导入acados的独立子进程中仍实际运行。MPPI按内核身份真正批计算，各车保留独立随机状态、热启动、异步reset与公开恢复。
- `upstream_nmpc.py`使用真实40步acados模型、方向相关变化率、低速转向与轮速约束，地形支持覆盖参考和停车走廊。纯侧向目标的SQP分支问题通过固定可行热启动阶段模式修复；最终仍经共同限幅和完整足迹复核。
- `upstream_baselines.py`提供line／quintic实际控制预测及原生47编号回放。新库保留编号和类别，按新尺度生成端点与参考速度；内容指纹为`4ce1ff4d048871917c523ef870ba6d72a8036840c6cb6d99c8118f9d6f231812`。旧库及T11准备模型没有自动迁移。
- `upstream_mppi.py`使用两轮高斯扰动、原始扰动重要性修正、float64敏感累计、方向相关控制限制和完整停车尾段。更新均值、最多16个预筛样本、前缀、旧计划及刹车依次复核；预算耗尽和未复核数量单列。
- `upstream_tensor_views.py`使用有界空间哈希查询本地快照。真实等距／支持边界歧义由所属CPU本地快照校正五通道，并记录次数和耗时；最终矩形、接触、扫掠及停车储备复核仍在CPU完成。

接口、架构和命令分别见[接口规范](../interface_spec.md)、[规划流水线](../architecture/planning_pipeline.md)和[运行手册](../runbooks/upstream_scale_diagnostics.md)。

## 冻结证据

正式run：

```text
outputs/runs/upstream_scale_s02/planners_seed23_cpu_cuda_20261013_01/
```

组合证据为[metrics/s02_evidence_summary.json](../../outputs/runs/upstream_scale_s02/planners_seed23_cpu_cuda_20261013_01/metrics/s02_evidence_summary.json)，绑定配置、源码、JUnit、硬件和下列产物SHA。

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 新尺度联合契约 | 598通过，0失败／0跳过 | `metrics/contracts.xml`及执行前冻结的`.source.json` |
| 相关旧路径与动作契约 | 417通过，0失败／0跳过 | `metrics/related.xml/related_acceptance.json` |
| 真实acados | 旧23项及新增7项实际执行通过 | 上述两个JUnit；旧23节点集合逐项匹配 |
| 统一诊断 | 56案例全部通过轨迹、独立储备及实际首控制核对 | `metrics/summary.json`与`samples/*_execution.json` |
| 公开恢复 | 10设置下一步配对通过 | `samples/*_restore.json` |
| CPU／CUDA同噪声 | B1，K512／1024／2048全部通过 | `metrics/same_noise_acceptance.json` |

两批测试文件不重合，合计1015项。原有23项包括容易漏计的`test_real_planners_share_safety_and_actual_control_feedback[nmpc]`；新增7项是实际求解与公开桥接用例，不把纯模型表达式测试计作真实求解。

56案例为CPU七设置中的47个原生编号与其余六设置，另加三个CUDA MPPI设置。输入为给定合成平坦本地快照，使用默认0.5 s注入延迟、实际推进三个保持控制步后执行首控制；没有使用S03传感链。恢复配对使用公开生命周期和新信息时刻，容差`atol=1e-7/rtol=1e-6`。

追加三个同噪声面板保存run内生产脚本及其SHA，使用`atol=1e-5/rtol=1e-5`。选中状态最大差为`6.883121017597915e-10`，控制／实际速度最大差为`4.656612873077393e-10`，样本代价最大差为`2.474431141763489e-6`，权重最大差为`3.919651270223312e-9`。这只证明该B1平坦面板范围，完整K/B与复杂输入矩阵仍待执行。

## 耗时与采样覆盖

本机RTX2060、单Torch线程，普通诊断单次规划＋完整复核耗时如下；这些数字不是分位延迟或实时通过证据。

| 设置 | CPU秒 | CUDA秒 |
| --- | --- | --- |
| line | 3.956 | — |
| quintic | 3.923 | — |
| NMPC | 3.993 | — |
| MPPI512 | 4.345 | 3.336 |
| MPPI1024 | 6.070 | 3.711 |
| MPPI2048 | 11.178 | 4.717 |

构建、规划、额外独立储备与代理执行分别记录。GPU计时在边界同步；内核另记录打包、预测、代价、更新及完整复核耗时。`process_peak_rss_kib`是同一进程累计峰值，不能解释为各设置独立增量内存。

六个实际MPPI首调用均保存末轮全部K样本，包括原始控制、proposal mean、原始扰动、受限控制、状态、物理代价、权重和有效掩码。每调用实际执行两轮，采样文件只覆盖末轮一轮，索引为1；不能称两轮采样100%覆盖。选中轨迹、实际首控制和公开状态逐案例保存。

独立本地查询微基准见`outputs/runs/upstream_scale_s02/tensor_views_seed31_cpu_cuda_20261013_01/metrics/query_benchmark_v2.json`：每view65536点，B2×K64×N41共5248查询，与CPU五通道参考一致。CPU中位27.6 ms，CUDA同步中位37.9 ms；仅查询耗时，不能替代完整规划复核耗时。

## 开发失败与保留边界

`logs/development/index.json`保存47份开发、失败和重跑文件来源及SHA。关键修复包括浮点零残差导致方向切换延后、停车步数舍入、地形tie跨分块传递扩张、支持边界漏查、CUDA设备别名、最后一轮／复核后截止未检查，以及无效采样NaN导致严格JSON日志无法保存。失败产物均保留，最终验收使用冻结run内的完整JUnit。

与前一T11准备run的169个生产文件核对，168个保持原样，唯一已有文件修改是`upstream_spec.py`增加显式执行版本。用户输入`writing-block.md`和旧47库SHA保持一致。源码字节、设计profile、解析配置和依赖身份已归档到该run，不覆盖T02–T10证据。

## 后续出口

当前组合结果保持`s02_full_acceptance_pass=false`、`s03_actual_sensor_communication_task_pass=false`、`s05_new_gate_b_pass=false`、`t11_learning_pass=false`及`gate_c_pass=false`。继续完成完整负载与复杂配对、S03密集感知／合法通信／集合任务、新尺度安全集成、S04校准和S05七设置千例与144回合，再接T11真实动作桥接及原预算筛选。

实时预算、PhysX与目标服务器性能分别验收。当前有效规划及停车尾段不代表目标完成、任务驻留或完整集合成功。

## 追加：容量负载矩阵完成

`outputs/runs/upstream_scale_s02/load_matrix_seed23_cpu_cuda_20261013_01/`的[严格汇总](../../outputs/runs/upstream_scale_s02/load_matrix_seed23_cpu_cuda_20261013_01/metrics/strict_acceptance.json)确认54个面板调用完整结束、九组B1同噪声CPU／CUDA配对通过。GPU覆盖平坦、4°坡地、交叉邻车三类，K512／1024／2048、B1／4／16／64，共36组；CPU覆盖同场景九组B1及平坦九组B4／16／64。没有降低K、减少两轮更新或静默换设备。

配对状态最大差`4.963075e-8`、控制最大差`1.862645e-8`、权重最大差`5.736113e-8`；样本代价最大绝对差`0.002064545`，通过原定相对／绝对联合容差，不能将它写成绝对误差小于`1e-5`。全部样本导出仍仅覆盖每调用末轮。

输入是65536点合成容量fixture，并非真实`SensorFrame`历史；B64是64条合法协议请求，不是64车环境闭环。范围由[coverage_notes.json](../../outputs/runs/upstream_scale_s02/load_matrix_seed23_cpu_cuda_20261013_01/metrics/coverage_notes.json)明确冻结。大规模调用包含CPU完整足迹复核，平坦GPU K2048/B64约294.6 s、交叉约744 s，平坦CPU K2048/B64约772 s；没有通过实时门槛，也不据此进行规划器延迟排名。

矩阵只令`resource_matrix_pass`及`same_noise_complex_b1_pass`为true，完整S02／S03／S05／T11标记仍为false。接续的真实感知、通信及任务组件实现与证据见[S03记录](upstream_scale_s03_components_2026-10-13.md)。上文“待执行”描述保留为首次冻结时的范围，此追加记录更新容量矩阵状态。
