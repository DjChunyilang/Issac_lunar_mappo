# T05：完整规划基线验收

日期：2026-10-10。状态：CPU契约及基线适配已验收，正式训练暂停。范围见[实施计划](../archive/implementation_plan_t05_t07_snapshot_2026-10-10.md)，前置[T04](t04_polar_hold_validation_2026-10-10.md)证据不计入本次结果。

交付注册表、纯函数execution、新nmpc_polar_v1/state2、line/quintic物理预测适配、冻结47库及原生回放载荷、统一diagnose_planners入口和t05_planner_baselines配置。旧桥接、checkpoint语义和categorical入口保留；非NMPC工厂不构造求解器。

Run：outputs/runs/t05_planner_baselines/acceptance_seed23_cpu_20261010_01。测试命令：

```bash
ACADOS_SOURCE_DIR="$PWD/.venv_isaaclab/acados" \
LD_LIBRARY_PATH="$PWD/.venv_isaaclab/acados/lib" OMP_NUM_THREADS=1 \
.venv_isaaclab/bin/python -m pytest \
  tests/test_planner_baselines.py tests/test_planner_protocol.py \
  tests/test_local_goal_nmpc.py tests/test_polar_hold.py -ra --disable-warnings

# 补齐P0真实HOLD回归（同一acados环境）
.venv_isaaclab/bin/python -m pytest \
  tests/test_p0_learning_contracts.py::test_real_acados_hold_decelerates_with_slew_limits
```

181项联合通过（152.78 s）＋补齐1项P0真实HOLD（8.76 s），共182项通过、0失败、0跳过，16项真实acados。真实求解器用例包括前置运动／停车、桥接／恢复、策略HOLD及新增4项低维MOVE配对；所有均实际执行。新增前／侧／后／零距离配对atol=1e-7、rtol=1e-6；后方目标首控制为倒车。具体名称与逐项误差保存在metrics/tests_retry1.xml。

统一诊断的line、quintic局部目标误差分别0.152864 m、0.189093 m，均低于0.2 m。47类原生载荷全部回放，旋转／倒车索引保留。这些使用给定合法平坦局部观测，不代表真实传感闭环、任务集合成功或性能排名。

初次新测试独立加载未设置scripts导入路径导致收集失败，修正顺序后重跑；logs/tests_initial.log及metrics/tests_initial.xml保留。新适配源代码、测试、配置及公共依赖指纹与run manifest另存。Ruff F821与git diff --check通过；最终联合静态检查在T07收尾时更新。

CUDA延期，未进行训练或生成训练checkpoint。历史47可控性报告保留；其旧原始outputs不因本次回放补写验收。下一阶段T06基础MPPI。

最终收尾联合回归见[T07记录](t07_mppi_terrain_neighbor_validation_2026-10-10.md)：65项基线测试全部通过，18项真实acados包含上述已有12项及6项新增。新增完整冻结库内容验证、原生context拒绝、真实polar移动HOLD及过期；库落盘configs/experiment/primitive47_library_v1.json。移动HOLD因停车上限可能返回solver_failure并刹车，这是合法失败反馈，初次过窄断言与修正日志保留。最终源码与解析配置另存，不覆盖首次基线指纹。
