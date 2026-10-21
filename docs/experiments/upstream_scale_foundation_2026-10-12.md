# 新车辆尺度：S00参数入口与S01公共执行基础

2026-10-13后续：[v2修复](upstream_scale_review_v2_repair_2026-10-13.md)补齐两项拒绝并启用保守分块查询，407项及八车满速／停车／恢复通过；本页保留原364项与原run，完整环境门槛仍未通过。

日期：2026-10-12。范围是[新尺度计划](../implementation_plan.md)的参数与执行基础。**S00基础已验收；S01公共执行基础通过，完整环境集成未验收；S02–S05待实施。** 不将本记录作为完整新尺度Gate B／T10、CUDA或PhysX证据，正式训练暂停。

同日后续[复核与耗时诊断](upstream_scale_diagnosis_2026-10-12.md)复现过小邻车半径、错误地形速度增益可被公共复核接受；原364项未覆盖这些拒绝边界。本页原通过／满速执行仍是既有输入的事实，不颁发完整复核证明。诊断只新增独立产物，未改本页原run。

## 配置、实现与版本

设计输入为[profile v2](../../configs/design/upstream_vehicle_proxy_v2.yaml)，原始文件SHA256为 `1b62f5a7c04c79e77092aac5a0bd82b6369089fafae837c77bba5e2eb839d51c`；v1及用户原规划未改。独立运行配置为[S00](../../configs/experiment/upstream_scale_s00.yaml)及[S01](../../configs/experiment/upstream_scale_s01.yaml)。前进硬上限25/9 m/s、倒车0.5 m/s；1 m/s只作为显式首轮测试。默认CPU、seed23、两环境×四车、0.2 s、预定N40；旧协议及历史配置仍N15。

包位于 `source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/local_goal/`：

| 文件 | 已接通范围 |
| --- | --- |
| `upstream_spec.py` | 冻结车辆／网格／任务规范、设计SHA、完整配置与规范指纹、恢复身份、静态队形和归一化 |
| `upstream_execution.py` | 独立NumPy与Torch联合控制投影、中点积分、实际速度、停车序列、信息时间停止储备 |
| `upstream_footprint.py` | 分离矩形地形与圆包络几何；四轮接触、去趋势RMS、台阶、净空、扫掠、邻车及工作区；停车走廊准入 |
| `upstream_proxy.py` | CPU八车共同模型执行、实际控制记录、异步reset、公开快照和拒绝加载前先完整校验 |

旧core、旧LocalGoalEnvironment及旧训练／评测入口拒绝新尺度配置。未把未来规划器适配器标识的登记解释为实现完成；真实协议v2、NMPC缓存和工厂在S02迁移。地形代理速度增益沿用通行代理及坡向指数缩放，最小0.25、坡度缩放系数0.75在不可变规范中记录；它不是车轮土壤标定。

## 命令及证据

S00初次严格run：`outputs/runs/upstream_scale_s00/acceptance_seed23_cpu_20261012_03/`，309项全部通过，27项新规范契约。此前证据初始化调用曾因旧device函数访问缺失的`local_goal_contract`失败，已改为显式新类型分派；失败原因保留在成功run的 `logs/setup_failure_prior_attempt.json`。没有通过伪造旧字段兼容新配置。

当前S01基础run：`outputs/runs/upstream_scale_s01/acceptance_seed23_cpu_20261012_03/`。此前 `_01` 的361项及真实执行记录完整保留；随后 `_02` 的363项及实际执行保留；当前 `_03` 补充邻车可达半径增长的段内裕量后另起run，不覆盖原产物。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_upstream_spec.py tests/test_upstream_execution.py \
  tests/test_config_wiring.py tests/test_cuda_evidence_gates.py \
  --junitxml=outputs/runs/upstream_scale_s01/acceptance_seed23_cpu_20261012_03/metrics/contracts.xml

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv_isaaclab/bin/python scripts/diagnose_upstream_execution.py \
  --config configs/experiment/upstream_scale_s01.yaml \
  --run-dir outputs/runs/upstream_scale_s01/acceptance_seed23_cpu_20261012_03 \
  --resume \
  --junit outputs/runs/upstream_scale_s01/acceptance_seed23_cpu_20261012_03/metrics/contracts.xml
```

创建run与pre-test来源记录的复跑步骤见[手册](../runbooks/upstream_scale_diagnostics.md)。脚本要求完整收集的测试身份一致、零失败／零skip、JUnit属于该run、测试与生产源码SHA一致；只写测试数量无法通过。产物包括 `run_manifest.json`、完整解析规范、命令／失败日志、JUnit与来源、逐步执行、四档停车储备、CPU热快照和严格状态。无优化器及训练checkpoint。

## 结果与边界

当前联合回归 **364项通过，0失败，0跳过**。包含54项新执行／足迹／证据门控、27项参数契约，以及历史配置与CUDA证据门控兼容测试；本批没有运行真实CUDA计算或acados，不能把门控测试当设备／求解器验收。

八车实际执行共122个底层步、976条车辆转移：

| 检查点 | 实际速度（m/s） | 信息时间（s） | 注入延迟后的生效时间（s） | 完整储备步数 |
| --- | --- | --- | --- | --- |
| 首轮 | 1.0 | 5.0 | 5.6 | 18 |
| 第二档 | 1.5 | 8.4 | 9.0 | 24 |
| 第三档 | 2.0 | 11.8 | 12.4 | 30 |
| 硬上限 | 25/9 | 16.6 | 17.2 | 40 |

0.5 s注入延迟向上取整为3个底层步，执行代理实际推进这3步保持控制，再按生效时间下发一次；未额外重复等待。满速从静止在此次分档流程16.6 s时实际达到，包含中间档位的延迟／保持，不以一条8 s预测宣称达到。独立连续加速契约覆盖70次加速更新、14 s底层网格。

八车由满速实际停车，执行位移均约 **9.368889 m**。信息时间停车储备使用先保持再减速的保守覆盖，满速制动35步，加3步延迟、1步新控制和1步静态尾段，共40步；保守中心总位移 **12.146667 m**，加车头及0.30 m余量为 **14.954167 m**。实际执行与覆盖上界分别记录。移动HOLD、低速转弯提前减角、倒车过零、轮加速度及延迟／制动敏感性均有契约测试；1 s延迟的满速储备超出N40时明确拒绝，限幅刹车保持 `unverified_brake`，不认证安全。

执行限制、物理包络重叠与静态净空违反均为0。公开恢复后的满速制动逐值一致，只重置环境0不影响环境1。足迹用例覆盖轮下未知、车底未知／障碍、危险等距点、去趋势坡面、台阶、旋转扫掠及邻车段内交叉。

本诊断输入为**给定的12231点本地平坦观测**，不是现有传感通信任务链。足迹全复核CPU耗时约6–14 s，满速约14 s，不能视为0.2 s实时或0.5 s墙钟预算通过。动态限速目前是停车走廊公共准入；安全QP、实时延迟界、实际环境链与所有规划器使用共同限幅仍待集成。`strict_acceptance.json`中 `s01_full_environment_integration_pass`、`s02_planners_pass`、`s03_perception_communication_task_pass`、`s05_gate_b_pass`均为false。

## 后续出口

本轮静态检查全部通过：Ruff `F821`、仓库卫生、`git diff --check`、默认25份文档数学检查及额外11份修改文档数学／围栏检查；11份文档共318条本地链接有效。验收后生产源码与run快照逐项一致，结果记入 `metrics/static_checks.json`。新增检查只写独立生成产物，不覆盖历史结果。

继续S01完整执行集成及S02：网格v2、完整七设置、真实acados23项及新增新尺度用例、CPU／CUDA数值与恢复；随后S03合法感知通信与集合状态机。足迹查询当前为正确性参考，需保留保守等距规则的批处理／分块实现和独立配对证据，不能放宽支持半径改善耗时。历史全矩阵不替代S04校准及S05新尺度重新验收。
