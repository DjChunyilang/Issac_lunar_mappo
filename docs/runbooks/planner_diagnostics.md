# T05–T10 无学习规划诊断

训练保持暂停。本入口消费统一规划协议，使用固定目标／显式原语回放，不加载策略权重或优化器。当前结论见[状态](../current_status.md)，字段见[接口](../interface_spec.md)。每次复跑选择新的run_id和输出文件；命令拒绝覆盖已有JSON及采样NPZ。

## 选择规划器

当前追加修复见[独立记录](../experiments/gathering_failure_repairs_2026-10-11.md)。定向完整回合使用 `diagnose_gathering_repairs.py`，它不会生成或替代 Gate B/T10，正式训练继续暂停。先选新的 run_id：

```bash
ACADOS_SOURCE_DIR="$PWD/.venv_isaaclab/acados" \
LD_LIBRARY_PATH="$PWD/.venv_isaaclab/acados/lib" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv_terrain_cuda/bin/python scripts/diagnose_gathering_repairs.py \
  --config configs/experiment/t10_memory_repair_cuda_v4.yaml \
  --run-dir outputs/runs/gathering_failure_repair/npb_memory_cuda_new \
  --family npb --variant mppi512 --safety on --duration 180
```

`--family` 可选 NPB、坡地、交叉及三个平坦类，脚本自动运行种子 23／31／47，原任务截止和传感链保留。NPB 单项使用 `t10_memory_repair_cuda_v4.yaml`；坡地单项 `t10_local_site_repair_v8.yaml`；交叉求解恢复 `t10_nmpc_recovery_v5.yaml`；组合为 `t10_combined_repairs_v9.yaml`。`--variant nmpc` 消费配置中的 `diagnostic_nmpc_adapter`。安全开／关及 96／180 s 要求分别运行并保存。

CUDA 平坦组合对照采用独立 `t10_combined_repairs_v11.yaml`，高层为已停稳捕获 v10；它保留严格 v8 的独立坡地证据，不声称 v10 坡地／NPB 全矩阵已完成。锚点区域完整支持时可依据低风险区域请求 HOLD；锚点不完整时，只能在本车已实际停稳、当前位置有低风险支持时请求 HOLD，不认证未知共同场地。实际团队地形／停稳／驻留仍按原任务判定。v8 平坦 seed31 截止及 v9 的开发中断保留，不覆写或补标成功。

已存在 run 仅在源码／配置／设置指纹匹配时加 `--resume`；修改实现后另起 run，不迁移历史热状态。`--max-macros` 只用于开发截断，不能用于完整回合验收。运行后 `metrics/episode_*.json`、逐步 trace、采样及公开状态分别核对；结果本身不会解除训练暂停。

| 适配器 | 配置 | 行为 |
| --- | --- | --- |
| nmpc_polar_v1 | configs/experiment/t05_planner_baselines.yaml | 低维MOVE补足并冻结NMPC约束 |
| deterministic_line_v1 / deterministic_quintic_v1 | 同上 | 几何参考经控制器和差速预测；不是插值状态 |
| primitive47_adapter_v1 | 同上 | 原生编号0–46；旋转、倒车和让行不折叠为极坐标 |
| mppi_base_v1 | configs/experiment/t06_mppi_base.yaml | 平坦模型基础MPPI，仍保留硬执行复核 |
| mppi_terrain_neighbor_v1 | configs/experiment/t07_mppi_terrain_neighbor.yaml | 地形模型、风险及邻车代价；CPU本地闭环 |
| mppi_terrain_neighbor_v1 | configs/experiment/t07_mppi_terrain_neighbor_cuda.yaml | 独立CUDA规划诊断；代理执行及最终复核在CPU |
| mppi_base_v1 | configs/experiment/t06_mppi_base_cuda.yaml | 独立基础CUDA；状态1与基础模型保持 |
| mppi_terrain_neighbor_stable_v2 | configs/experiment/t07_mppi_terrain_neighbor_cuda_v2.yaml | float32预测／float64累计、状态2、查询内存预算 |
| mppi_terrain_neighbor_stable_v2 | configs/experiment/t08_safety_projection_cuda_v2.yaml | CUDA规划与现有CPU安全投影 |
| mppi_terrain_neighbor_stable_v2 | configs/experiment/t09_planner_benchmark_cuda_v2.yaml | 新完整七设置评测，四基线CPU |
| mppi_terrain_neighbor_stable_v2 | configs/experiment/t10_gathering_no_learning_cuda_v2.yaml | 新GPU Gate B门控下完整K512集合 |

原exp167/P0/T03/T04配置不切换适配器。NMPC需要本地acados，安装方法见[旧运行手册](local_goal_nmpc.md)。MPPI及确定性路径不需要casadi/acados。

```bash
OMP_NUM_THREADS=1 .venv_isaaclab/bin/python scripts/diagnose_planners.py \
  --config configs/experiment/t07_mppi_terrain_neighbor.yaml \
  --planner mppi_terrain_neighbor_v1 --samples 128 --seed 23 \
  --goal .8 0 --steps 100 \
  --output outputs/runs/t07_mppi_terrain_neighbor/replay_seed23_cpu_new/metrics/diagnostic.json
```

默认是给定合法局部观测。--scene支持flat、risk、blocked、unknown、cross、head_on、stopped、missing、expired；除简单可达场景外，刹车／前缀停车只能证明边界行为，不计作到达。--steps 0为单次规划，--goal设置固定odometry目标；请求每5步续期，目标不随车移动。

```bash
# 现有传感、通信、差速执行及任务结束链，两环境×四车
OMP_NUM_THREADS=1 .venv_isaaclab/bin/python scripts/diagnose_planners.py \
  --config configs/experiment/t07_mppi_terrain_neighbor.yaml \
  --planner mppi_terrain_neighbor_v1 --samples 128 --seed 23 \
  --sensor-env --scene flat --goal .8 0 --steps 100 \
  --output outputs/runs/t07_mppi_terrain_neighbor/sensor_seed23_cpu_new/metrics/diagnostic.json

# 冻结库47项原生回放
OMP_NUM_THREADS=1 .venv_isaaclab/bin/python scripts/diagnose_planners.py \
  --config configs/experiment/t05_planner_baselines.yaml \
  --planner primitive47_adapter_v1 --primitive all \
  --output outputs/runs/t05_planner_baselines/primitive_seed23_cpu_new/metrics/diagnostic.json
```

--sensor-env --scene flat通过配置生成平坦后端，随后消费其现有传感链和真实局部记忆，不注入稠密测试地图；--scene inherited使用诊断配置的原地形，失败记录保留。达到误差阈值后使用HOLD并实际限幅停车。它不是完整集合任务，也不做PhysX验证。长目标未达到时local_pass=false，保留误差和回退，不能解释成成功。

## 有序请求回放

--requests接受JSON列表，且必须--steps 0。每条包含goal或hold，以及可选env_id、agent_id、pose、velocity、previous_control、now、timestamp、expires_at、terminal_yaw、speed_cap；所有位置在接收车固定odometry frame。每个身份保留独立热启动。原语回放另含primitive_id。

```json
[
  {"env_id":0,"agent_id":0,"goal":[0.8,0],"now":0},
  {"env_id":0,"agent_id":0,"pose":[0.1,0,0],"previous_control":[0.6,0],"hold":true,"now":0.2},
  {"env_id":0,"agent_id":0,"goal":[0.8,0],"now":2,"timestamp":0,"expires_at":1}
]
```

回放使用--scene选择的合法局部观测；不会从全局地形补足未知。记录末轮原始采样和受限控制、预测／实际速度、代价分量、权重与重要性分数；输出JSON记录选中轨迹、状态、回退、首控制与耗时，传感集成另记录名义／下发控制和projection_disabled。配置、源代码指纹和验收命令以对应run manifest为准。

## 设备与后续验收

默认CPU float32；解析测试使用float64。显式请求不可用CUDA时提前报错。基础及首次v1独立CUDA配置设置 `local_goal.planner_device: cuda` 和 `mppi_query_chunk: 1024`；稳定v2配置还显式设置 `planner_adapter` 与 `planner_revision` 为 `mppi_terrain_neighbor_stable_v2`。继承的 `simulation.device` 仍为CPU；感知、通信、代理执行、OSQP和最终轨迹复核在CPU。正式训练持续暂停。

本机已有 `.venv_terrain_cuda`，使用PyTorch 2.10.0+cu126。下列安装入口为该环境补充OSQP 1.0.4并检查GPU；通过CPU环境的pip管理目标环境，不更换 `.venv_isaaclab` 的CPU PyTorch或驱动。目标环境缺失时脚本调用既有 `install_terrain_cuda.sh` 创建独立环境。

```bash
bash scripts/install_planner_cuda.sh

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv_terrain_cuda/bin/python scripts/diagnose_planners.py \
  --config configs/experiment/t07_mppi_terrain_neighbor_cuda.yaml \
  --planner mppi_terrain_neighbor_v1 --samples 512 --seed 23 \
  --sensor-env --scene flat --goal .8 0 --steps 100 \
  --output outputs/runs/mppi_cuda_diagnostic/sensor_seed23_cuda_new/metrics/diagnostic.json
```

CPU与CUDA分别冷启动新run。设备、完整MPPI配置和RNG设备进入快照校验，跨CPU/CUDA热状态恢复明确拒绝；不能将旧CPU run或checkpoint改名后接续。既有诊断JSON中的 `cuda_validation: deferred` 保留其历史验收边界，实际CUDA配置、数值测试和同步测量由独立证据说明。

2026-10-11首次v1 GPU契约测试为17项：16通过、1失败、0跳过。`cross_offset` 的float32控制量最大绝对误差为 `1.7166e-5`，既定 `atol=1e-5, rtol=1e-5` 对照失败；原失败夹具、配置和产物完整保留。测试细项、run和证据见[首次CUDA诊断记录](../experiments/mppi_cuda_diagnostic_2026-10-11.md)，不能用平坦场景一致或GPU可用代替完整数值验收。稳定v2的新验收入口见下文。

### 同噪声与同步耗时面板

`scripts/benchmark_mppi_devices.py` 对K=512/1024/2048及GPU B=1/4/16/64做独立测量，默认CPU对照为B=1。CPU/GPU使用同一注入噪声并保存SHA；每次计时前后 `torch.cuda.synchronize()`，包含CPU打包、设备传输、规划和最终host复核。每次重置规划器热状态，先预热计算内核，构建、预热与稳态调用分开记录。此面板使用冻结的平坦合法局部输入，不能作为地形／邻车完整性能矩阵或任务验收。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv_terrain_cuda/bin/python scripts/benchmark_mppi_devices.py \
  --config configs/experiment/t07_mppi_terrain_neighbor_cuda.yaml \
  --samples 512 1024 2048 --batches 1 4 16 64 --cpu-batches 1 \
  --chunks 256 1024 2048 --repeats 3 \
  --run-dir outputs/runs/mppi_cuda_diagnostic/device_seed23_cuda_new
```

`--cpu-batches 1 4 16 64` 可扩大CPU配对覆盖。query chunk面板只在GPU B=1及第一个K设置比较256/1024/2048，其余GPU面板使用独立配置的1024。旧MPPI内部phase计时未做CUDA同步，反映异步dispatch，不能用作GPU真实耗时或据此排名。主同步计时及显存高水位记录在 `metrics/<device>_b<B>_k<K>_q<chunk>.json`，同噪声差异和加速比汇总在 `metrics/device_benchmark.json`；该汇总明确要求另行完成CUDA契约测试。

T05–T07默认安全接口保持直通；独立T08配置启用代理CBF/QP，T09/T10分别验收固定评测与集合。代理有限点复核不构成连续安全或PhysX通过。

## T08–T10 新诊断

T08仅新配置启用OSQP 1.0.4：

```bash
.venv_isaaclab/bin/python -m pip install 'osqp==1.0.4'
```

历史配置仍直通，MPPI基础入口不要求安装acados；T09完整对照需要真实acados。环境变量必须提供真实源码和共享库路径，必要求解器测试拒绝skip。

```bash
export ACADOS_SOURCE_DIR="$PWD/.venv_isaaclab/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv_isaaclab/bin/python scripts/benchmark_planners.py \
  --config configs/experiment/t09_planner_benchmark.yaml \
  --run-dir outputs/runs/t09_planner_benchmark/full_seed23_cpu_new
```

默认覆盖1000案例、七设置、630恢复检查、252局部回合、B4/16面板14组。每25例写只增分片，单例先原子保存，重跑必须 `--resume` 且源码／配置／依赖一致。`--phase single/repeat/loop/panel` 可分阶段，`--variants` 不改变冻结验收范围；`--start/--stop` 必须25的倍数。不同进程处理同一run时使用互不重叠的单例区间，或不同设置；不并发写同一重复／闭环记录。固定物理核、单线程，记录运行命令和亲和性，避免将共享核干扰解释为算法差异。

`--phase aggregate --regression-junit <path>` 汇总完整证据，JUnit同名 `.source.json` 保存执行前实际源码快照。必要联合测试、既有18项真实acados及恢复重复性均通过，NMPC与MPPI512全部预定平坦局部回合成功，才输出 `gate_b_cpu=true`。没有JUnit、漏例、必要skip、失败或源码不符保持false。CPU正确性关闭MPPI墙钟截止，NMPC10 s，不是实时验收。

```bash
.venv_isaaclab/bin/python scripts/benchmark_planners.py \
  --run-dir outputs/runs/t09_planner_benchmark/full_seed23_cpu_new \
  --resume --phase aggregate --regression-junit /absolute/path/contracts.xml

.venv_isaaclab/bin/python scripts/evaluate_gathering_no_learning.py \
  --config configs/experiment/t10_gathering_no_learning.yaml \
  --run-dir outputs/runs/t10_gathering_no_learning/full_seed23_cpu_new \
  --gate-b /absolute/path/t09/metrics/summary_NNNN.json
```

T10正式入口拒绝没有通过Gate B或实际依赖源码不同的证据。当前 `t10_evidence_driver_v3` 仅允许T09不导入的本消费者文件更新；规划、环境、通信和评测依赖仍逐文件匹配T09生产快照，差异记录到gate证据。默认144回合，两环境×四车真实batch，MPPI固定512；`--family/--variant/--safety/--duration` 可分段，不能缩小最终验收范围。`--max-macros` 仅明确预算的接线pilot，其manifest与正式run隔离。已有状态只有显式 `--resume`、版本匹配和SHA一致才恢复；诊断pickle仅用于本地可信状态，不是训练checkpoint。

恢复还核对分场景有效配置、NPB配置／manifest／全部运行数组SHA，以及冻结的18场景列表。场景seed生成初始队形，规划seed保持T07默认23，环境seed按解析配置记录，三者分开保存。各回合批次独立acados构建目录，支持不同矩阵分段并行；仍不得并发执行同一批次。构建耗时、CPU进程内存高水位独立记录。中断时已落盘回合须与恢复重放的结束原因、回报和失败计数一致；计算墙钟可不同，原结果不覆盖。

每5步保存公开生命周期状态；每步记录名义轨迹、实际控制和安全结果。原始MPPI采样首次、每25底层步以及新的失败／回退保留；HOLD无采样，不能把空记录算采样覆盖。执行预测复核失败撤销轨迹广播，日志中的 `unverified_brake` 不能解释为安全认证。复杂失败与未完成回合单独分类；只有完整矩阵和两规划器各9例安全开启180 s简单集合全部成功、无碰撞和限制违反，才通过T10。

独立CUDA接线pilot使用真实感知、通信和代理执行，限定30宏步：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv_terrain_cuda/bin/python scripts/evaluate_gathering_no_learning.py \
  --config configs/experiment/t10_gathering_no_learning_cuda.yaml \
  --family flat_compact --variant mppi512 --safety on --duration 180 \
  --max-macros 30 \
  --run-dir outputs/runs/mppi_cuda_diagnostic/gathering_seed23_cuda_pilot_new
```

`--duration 180` 设置任务时钟预算，`--max-macros 30` 另限制实际接线诊断步数；自然结束的回合按实际判据记录，未结束的预算截断不能计作任务超时。pilot不代替144回合正式GPU验收。正式新GPU全矩阵需要与当前实际依赖源码匹配的新Gate B生产证据，不能放宽旧源码guard或复用不匹配的CPU证据。[首次CUDA诊断记录](../experiments/mppi_cuda_diagnostic_2026-10-11.md)保留当时未通过的数值边界，稳定v2本轮已完成的数值／负载、Gate B与完整T10功能结果见下文；完整足迹及PhysX仍有独立门槛，正式训练持续暂停。

正式144回合由冻结v2完成，v3仅修正结束后的症状频率分类；独立复核run保留144原始SHA。旧v2run及热状态拒绝直接在v3下续跑；复现原生产版本需恢复其完整源码快照，v3新运行必须另建run，不绕过指纹校验。

## 稳定v2 CUDA完整验收与正式诊断

本轮CUDA数值／负载验收已完成：66项CUDA契约、695项联合回归及其中18项真实acados全部通过，0失败／0跳过；runtime的36GPU面板、9个B1 CPU配对、27个复杂输入对照、容量边界及同步阶段诊断通过。证据run为`outputs/runs/mppi_cuda_validation/contracts_v2_seed23_rtx2060_20261011_02/`与`acceptance_v2_seed23_rtx2060_20261011_02/`。[新GPU Gate B汇总](../../outputs/runs/t09_planner_benchmark/acceptance_seed23_cuda_v2_20261011_01/metrics/summary_0000.json)已通过完整7000／630／252／14及重复、简单闭环、设备与源码核验，`gate_b_cuda=true`、`gate_b_cpu=false`。[T10正式汇总](../../outputs/runs/t10_gathering_no_learning/acceptance_seed23_cuda_v2_20261011_01/metrics/summary_0007.json)确认144回合完整、18例简单门槛通过，`t10_cuda_functional_pass=true`；110成功、24终端地形失败、4规划拒绝及6任务截止，无碰撞／执行限制违反。[全量逐步审计](../../outputs/runs/t10_gathering_no_learning/acceptance_seed23_cuda_v2_20261011_01/metrics/trace_audit_0000.json)通过117744次执行记录与4108个宏步／公开快照，错误为0；要求采样事件710／710保留，占10170次实际采样调用的6.981%，HOLD无采样。

冻结范围见[CUDA计划](../cuda_acceptance_plan.md)，当前状态和证据见[本轮CUDA验收记录](../experiments/mppi_cuda_validation_2026-10-11.md)。以下命令复现数值／负载、Gate B及T10功能验收；通过结论须由实际结果证明，正式训练暂停继续有效。使用`.venv_terrain_cuda`，host保持CPU，不接续旧CPU或v1热状态。

`mppi_terrain_neighbor_stable_v2`为独立规划版本，状态2、数值`fp32_fp64_accum_v2`：raw采样／预测／控制均值float32，邻车距离风险及重要性权重累计float64，更新均值转回float32。v2的成对地形查询估算预算为256 MiB，有效块按`max(1,min(query_chunk,budget//(B*M*(8*element_size+4))))`选择；配置1024是上限，预算不代表总显存上限。原v1与基础`mppi_base_v1`保持原语义。

先冻结所有任务和`scripts/*.py`，在新目录实际运行完整GPU测试并保存执行前来源；同名`.source.json`使用包含`source_snapshot`和`test_sources`的对象，不能沿用普通回归只有源码字典的格式。目录和run_id每次重新选择，失败日志保留。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PYTHONPATH=scripts .venv_terrain_cuda/bin/python - <<'PY'
from pathlib import Path
from planner_evidence import json_write, source_snapshot, cuda_test_sources
path = Path('outputs/runs/mppi_cuda_validation/contracts_seed23_cuda_v2_new/metrics/pytest_cuda.xml')
path.parent.mkdir(parents=True, exist_ok=True)
json_write(path.with_suffix('.source.json'),
           {'source_snapshot': source_snapshot(), 'test_sources': cuda_test_sources()})
PY

PYTHONPATH=scripts .venv_terrain_cuda/bin/python -m pytest \
  tests/test_mppi_cuda.py -q \
  --junitxml=outputs/runs/mppi_cuda_validation/contracts_seed23_cuda_v2_new/metrics/pytest_cuda.xml

.venv_terrain_cuda/bin/python scripts/validate_mppi_cuda.py \
  --config configs/experiment/t09_planner_benchmark_cuda_v2.yaml \
  --cuda-junit outputs/runs/mppi_cuda_validation/contracts_seed23_cuda_v2_new/metrics/pytest_cuda.xml \
  --repeats 3 \
  --run-dir outputs/runs/mppi_cuda_validation/runtime_seed23_cuda_v2_new
```

`validate_mppi_cuda.py`要求实际CUDA、完整当前JUnit、0失败／0跳过及一致来源；不支持把旧失败证据改名续用。它运行flat／slope／cross各12个K512／1024／2048×B1／4／16／64面板，共36面板，并在B1以相同噪声做CPU配对；九类×三个种子另有27个K128对照。4096本地点、B64、每实例128个query点是单独容量边界，明确不是B64／K2048／4096点整规划性能。至少三次计时的p95／p99是描述统计，不是延迟尾部认证。

指标保存于`metrics/panel_<family>_b<B>_k<K>.json`、`cpu_<tag>.json`、`complex_inputs.json`、`memory_capacity.json`与`synchronized_stages.json`。普通面板采用预热kernel、冷句柄和整调用前后CUDA同步，包含host打包、传输及最终复核。阶段profile另插barrier，其额外开销不混入主延迟排名。`metrics/strict_acceptance.json`绑定所有指标SHA、硬件、完整planner元数据、JUnit、测试源码与生产源码；必要条件不足时退出失败。

随后完整运行新T09。联合回归JUnit须绑定当前源码、至少600项、0失败／0跳过及18项实际真实acados；执行前用`json_write(junit_path.with_suffix('.source.json'), source_snapshot())`保存普通回归来源。CUDA JUnit和runtime严格证据使用上一步产物，五份配置的公共MPPI参数必须一致。

```bash
export ACADOS_SOURCE_DIR="$PWD/.venv_isaaclab/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH=scripts:source/lunar_rover_tasks

.venv_terrain_cuda/bin/python - <<'PY'
from pathlib import Path
from planner_evidence import json_write, source_snapshot
path = Path('outputs/runs/mppi_cuda_validation/contracts_seed23_cuda_v2_new/metrics/regression.xml')
json_write(path.with_suffix('.source.json'), source_snapshot())
PY

.venv_terrain_cuda/bin/python -m pytest \
  tests/test_config_wiring.py tests/test_exp156_differential.py \
  tests/test_local_goal_diagnostics.py tests/test_local_goal_nmpc.py \
  tests/test_lunar_training_maps.py tests/test_mppi.py tests/test_mppi_terrain_neighbor.py \
  tests/test_observation.py tests/test_p0_learning_contracts.py \
  tests/test_planner_baselines.py tests/test_planner_diagnostic_entry.py \
  tests/test_planner_protocol.py tests/test_polar_hold.py tests/test_reward.py \
  tests/test_safety_projection.py tests/test_stage5_diagnostics.py \
  tests/test_termination.py tests/test_tiered_communication.py \
  tests/test_t10_evidence_driver.py tests/test_cuda_evidence_gates.py \
  --junitxml=outputs/runs/mppi_cuda_validation/contracts_seed23_cuda_v2_new/metrics/regression.xml

.venv_terrain_cuda/bin/python scripts/benchmark_planners.py \
  --config configs/experiment/t09_planner_benchmark_cuda_v2.yaml \
  --run-dir outputs/runs/t09_planner_benchmark/full_seed23_cuda_v2_new \
  --phase all \
  --regression-junit outputs/runs/mppi_cuda_validation/contracts_seed23_cuda_v2_new/metrics/regression.xml \
  --cuda-junit outputs/runs/mppi_cuda_validation/contracts_seed23_cuda_v2_new/metrics/pytest_cuda.xml \
  --cuda-acceptance outputs/runs/mppi_cuda_validation/runtime_seed23_cuda_v2_new/metrics/strict_acceptance.json
```

新run完整覆盖7000单次、630恢复、252局部回合及14批量面板。MPPI使用配置的稳定v2 CUDA，其余四基线仍CPU，单条记录明确实际设备和数值版本。完整覆盖、必要回归、重复性、简单闭环及GPU严格证据均通过才产生`gate_b_cuda=true`，`gate_b_cpu`仍为false；缺例／失败保持`cuda_validation=incomplete`。`--phase`、`--variants`和分片可调度互不重叠任务，最终范围不缩小；后续调用显式`--resume`，代码、配置和依赖不得中途改变。

分片运行结束后单独聚合。`--resume`只校验并接续同一run，不覆盖已完成记录；每次聚合追加新的`summary_NNNN.json`，使用实际完整通过的文件作为T10输入。

```bash
.venv_terrain_cuda/bin/python scripts/benchmark_planners.py \
  --config configs/experiment/t09_planner_benchmark_cuda_v2.yaml \
  --run-dir outputs/runs/t09_planner_benchmark/full_seed23_cuda_v2_new \
  --resume --phase aggregate \
  --regression-junit outputs/runs/mppi_cuda_validation/contracts_seed23_cuda_v2_new/metrics/regression.xml \
  --cuda-junit outputs/runs/mppi_cuda_validation/contracts_seed23_cuda_v2_new/metrics/pytest_cuda.xml \
  --cuda-acceptance outputs/runs/mppi_cuda_validation/runtime_seed23_cuda_v2_new/metrics/strict_acceptance.json
```

正式CUDA T10需要新Gate B汇总及相同CUDA证据，使用已完成门槛的实际`summary_NNNN.json`路径：

```bash
.venv_terrain_cuda/bin/python scripts/evaluate_gathering_no_learning.py \
  --config configs/experiment/t10_gathering_no_learning_cuda_v2.yaml \
  --run-dir outputs/runs/t10_gathering_no_learning/full_seed23_cuda_v2_new \
  --gate-b /absolute/path/t09_cuda/metrics/summary_NNNN.json \
  --cuda-junit outputs/runs/mppi_cuda_validation/contracts_seed23_cuda_v2_new/metrics/pytest_cuda.xml \
  --cuda-acceptance outputs/runs/mppi_cuda_validation/runtime_seed23_cuda_v2_new/metrics/strict_acceptance.json
```

CUDA消费者版本`t10_evidence_driver_cuda_v1`要求设备、运行／累计精度、查询预算、完整MPPI配置及所有证据SHA匹配，保持原逐文件源码guard。默认完整144回合，NMPC仍CPU，MPPI512 CUDA；无`--max-macros`。真实传感／通信、异步环境及25步驻留均实际执行，两个规划器安全开启180 s简单平坦各9／9成功、无碰撞／限制违反后才通过功能门槛。其他复杂失败完整分类，不能以停车或加速解释为成功。

T10分段任务全部结束后聚合，再审核完整144回合的实际控制、广播执行预测、采样事件和公开恢复状态；下一次规划使用实际控制的行为由相关契约测试覆盖。聚合从该run已绑定的Gate B／CUDA证据读取并复核，审计只追加新的`trace_audit_NNNN.json`；两者均不能把部分矩阵改记为完整通过。

```bash
.venv_terrain_cuda/bin/python scripts/evaluate_gathering_no_learning.py \
  --config configs/experiment/t10_gathering_no_learning_cuda_v2.yaml \
  --run-dir outputs/runs/t10_gathering_no_learning/full_seed23_cuda_v2_new \
  --resume --aggregate

.venv_terrain_cuda/bin/python scripts/audit_gathering_traces.py \
  --run-dir outputs/runs/t10_gathering_no_learning/full_seed23_cuda_v2_new \
  --require-full --cuda
```

运行时单GPU独占；CPU基线可在独立物理核处理互不重叠记录，核对内存和亲和性。所有阶段新run、只增产物、失败／重跑日志和恢复SHA保留；旧CPU及首次CUDA结果不覆盖。完整CUDA通过也不代表实时、真实足迹或PhysX通过，正式训练持续暂停。
