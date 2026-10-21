# T03：统一规划器协议验收

日期：2026-10-10。基线提交：`4a79b0d5faf6d65037422c352ed91f827c8c40f1`，在已验收的 T00–T02 未提交工作区上实施。阶段计划见[实施计划](../implementation_plan.md)，字段语义见[接口规范](../interface_spec.md)。

状态：T03 协议验收通过；不代表 Gate A、完整任务或安全层验收通过。

## 修改与边界

独立配置 `configs/experiment/t03_planner_protocol.yaml` 继承 P0，启用 planner_protocol_v1 / nmpc_bridge_v1。任务、观测、四维动作及 prefix_consistency_v2 保持；旧 exp167/P0 保留原调用路径。

新协议提供 LocalIntent、带时间网格的 PlannerInput、复用轨迹／反馈的 PlannerOutput、只读局部 TerrainView、带 sender 的邻车预测、公开生命周期和有序批调度。环境可以替换协议适配器，不访问NMPC热启动成员；只有适配器处理其私有状态。

过期意图和未支持的低维运动意图明确刹车；已复核轨迹与旧计划valid=true，紧急刹车valid=false。安全接口直通并记录projection_disabled、名义控制和实际下发命令。它们不是安全投影或可行性证书。HOLD冻结请求时位姿并沿用减速限幅，成功仍由实际任务判定。

本阶段没有正式训练、混合动作、MPPI、CBF/QP、完整T05基线或性能比较。测试替身仅证明协议可替换，不计为研究基线。24 m通信、分层出生、完整足迹和地理独立数据仍待后续实施。

## 配置与证据

Run：`outputs/runs/t03_planner_protocol/contract_seed23_cpu_20261010_01/`，seed=23，CPU。该目录为独立ignored产物，不覆盖T02结果。

| 文件 | 含义 |
| --- | --- |
| config/experiment.yaml | 解析后的诊断配置 |
| run_manifest.json | 基线、工作区状态、版本、配置与实现指纹、验收命令 |
| metrics/source_snapshot.json | 执行验收时依赖源码的逐文件SHA256 |
| metrics/contracts_acceptance.xml | 最终相关回归JUnit记录 |
| logs/contracts_acceptance.log | 完整测试日志及acados原始输出 |
| metrics/summary.json | 通过／失败／跳过数量、真实求解器数量和门槛结论 |
| metrics/paired_errors.json | 从JUnit提取的确定性、环境及实际acados配对误差 |
| logs/static_checks.log | 链接／归档完整性、公式、卫生、F821和diff检查 |

源码指纹覆盖项目包及实际runner／配置加载依赖；配置记录规划协议、适配器和状态版本。不同版本及缺失合同的快照均拒绝恢复。未创建训练checkpoint。

## 验证命令

从仓库根目录执行；acados库与Python环境沿用T02已验证安装。

```bash
ACADOS_SOURCE_DIR="$PWD/.venv_isaaclab/acados" \
LD_LIBRARY_PATH="$PWD/.venv_isaaclab/acados/lib" \
OMP_NUM_THREADS=1 .venv_isaaclab/bin/python -m pytest \
  tests/test_planner_protocol.py tests/test_p0_learning_contracts.py \
  tests/test_local_goal_nmpc.py tests/test_local_goal_diagnostics.py \
  tests/test_reward.py tests/test_observation.py tests/test_termination.py \
  tests/test_tiered_communication.py tests/test_exp156_differential.py \
  tests/test_config_wiring.py tests/test_lunar_training_maps.py \
  --junitxml=outputs/runs/t03_planner_protocol/contract_seed23_cpu_20261010_01/metrics/contracts_acceptance.xml \
  -ra --disable-warnings
```

配对使用相同输入和初始规划器状态：确定性替身逐值一致；真实求解器检查状态、控制、实际速度和时间，atol=1e-7、rtol=1e-6，反馈类别与原因一致；不比较墙钟耗时。真实配对将求解时限设为10 s，生产诊断默认时限仍为0.15 s，不能据此宣称满足实时预算。

## 覆盖与验收结果

- 协议：形状、时间网格、坐标转换、有效期含边界、未知版本、非法数值、地形快照隔离、轨迹有效性与反馈一致性。
- 行为：正常解、迭代限制、前缀反馈映射、求解失败／超时、旧计划复核通过／拒绝、过期意图、unsupported_intent、移动HOLD限幅。
- 状态：多环境有序／稀疏批次、异步任务结束／reset、公开生命周期替换、热启动隔离、快照恢复、失败恢复不修改core。
- 信息：sender换槽、缺失／稀疏／过期／不支持消息，以及Oracle和未通信真值不影响本地预测；保留T02学习／任务边界回归。
- 实际acados：原有6项回归，新增前进、倒车、转向、移动HOLD和热启动恢复后倒车的5项配对。

最终结果：**409 项通过、1 项 CUDA 一致性检查因 CPU 环境跳过，0 失败；11 项真实 acados 测试全部实际通过。** 新协议文件55项，加上自动纳入的新配置检查，相比T02验收净增56项。5项真实桥接配对的最大轨迹／控制／实际速度／时间误差均为0；确定性替身和环境配对误差也为0。

最终源码 AST 指纹：`06f3de26b2eec1763b10b49540a6ee018a7c10bc3443c28954421b9457053d9a`。完整相关回归用时98.58 s，日志记录467项警告计数；本次使用--disable-warnings，未展开逐条警告，acados原始QP提示仍保留。此耗时不是规划吞吐评测。

文档链接、公式、仓库卫生、Ruff F821及git diff --check通过。所有代码探索测试均无失败；前期未启用真实求解器的跳过不计为验收，最终11项必要真实求解器测试均未跳过。

## 探索日志与限制

收尾静态检查发现旧技术说明快照链接的日期被误改；已恢复到原归档文件，重跑链接检查通过。失败日志单独保存在logs/static_link_failure.log，不覆盖最终通过日志。

先运行无求解器相关回归（55通过、6项真实acados未启用），再运行新增纯协议（40通过，扩展后49通过）；这两批不是最终验收。加入真实acados后新增协议首批53通过，完整回归进一步加入热启动恢复和地形字段隔离。

日志保留原始输出；移动HOLD可能触发ACADOS_MINSTEP并走限幅刹车，验收检查的是回退契约及新旧一致性，不能将其写成求解可行。真实前缀算法回归沿用既有诊断，新增协议测试检查前缀状态的映射。

T03通过后交接T04，并供T05/T06使用；Gate A仍依赖混合动作概率契约，正式训练继续暂停。此处不报告任务成功率、吞吐优势或真实月面安全结论。
