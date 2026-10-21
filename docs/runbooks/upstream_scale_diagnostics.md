# 新尺度无学习诊断与T11准备

## S03实际环境探索

新入口直接运行真实密集感知、私有记忆、距离消息、排队实际反馈、有限任务和宏步奖励：

```bash
.venv_terrain_cuda/bin/python scripts/diagnose_upstream_environment.py \
  --config configs/experiment/upstream_scale_s03_environment_cuda_v2.yaml \
  --run-id environment_spawn_seed23_cuda_new --scenario spawn_flat
```

CPU使用`upstream_scale_s03_environment_v2.yaml`；完整足迹QP使用`upstream_scale_s03_environment_safety_v2.yaml`。`--max-steps 100`可作有边界探索，提前切分不算截止或集合成功。每5步保存公开快照，首次和每25步保存MPPI末轮样本，run拒绝覆盖。`--restore-state`只接受带SHA的本地诊断快照，新run记录恢复来源；跨源码探索续跑不计冻结S05验收。

结果为`metrics/environment_summary.json`及`trace.jsonl`，按实际结束及驻留判断完成。当前入口不运行千例矩阵、不颁发训练准入。范围见[环境接线记录](../experiments/upstream_scale_environment_2026-10-14.md)。

S00／S01开放参数与共同执行基础，S02新增独立40步规划核心及给定观测面板。S01复核修复见[记录](../experiments/upstream_scale_review_v2_repair_2026-10-13.md)，S02实现／验收进展见[独立记录](../experiments/upstream_scale_planners_2026-10-13.md)，完整门槛见[计划](../implementation_plan.md)。各入口不加载学习权重或优化器。S01脚本要求CPU，S02可选择CPU／CUDA MPPI；两者均要求显式离线模拟延迟。旧[T05–T10手册](planner_diagnostics.md)用于旧尺度复现。

## S02七设置、双设备与公开恢复

使用独立`configs/experiment/upstream_scale_s02.yaml`和新run_id。`--prepare`先保存解析配置、全部生产源码、设计profile、运行依赖以及必需测试来源；测试后以完全相同设置`--resume`进入诊断。必需集合包含真实CUDA测试和真实acados测试，使用已配置这些依赖的`.venv_terrain_cuda`；`.venv_isaaclab`的CPU版Torch不能替代该集合。准备与诊断均保留`--device both --restore-panel`，否则恢复时设置指纹不匹配。

从仓库根目录执行：

```bash
export ACADOS_SOURCE_DIR="$PWD/.venv_isaaclab/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv_terrain_cuda/bin/python scripts/diagnose_upstream_planners.py \
  --config configs/experiment/upstream_scale_s02.yaml \
  --run-dir outputs/runs/upstream_scale_s02/planners_seed23_cpu_cuda_new \
  --device both --restore-panel --prepare

.venv_terrain_cuda/bin/python -m pytest -q \
  tests/test_upstream_spec.py tests/test_upstream_execution.py \
  tests/test_upstream_review_v2.py tests/test_upstream_planning.py \
  tests/test_upstream_nmpc.py tests/test_upstream_mppi.py \
  tests/test_upstream_tensor_views.py tests/test_upstream_baselines.py \
  tests/test_upstream_factory.py tests/test_upstream_planner_diagnostics.py \
  tests/test_config_wiring.py \
  --junitxml=outputs/runs/upstream_scale_s02/planners_seed23_cpu_cuda_new/metrics/contracts.xml \
  > outputs/runs/upstream_scale_s02/planners_seed23_cpu_cuda_new/logs/contracts.log 2>&1

.venv_terrain_cuda/bin/python scripts/diagnose_upstream_planners.py \
  --config configs/experiment/upstream_scale_s02.yaml \
  --run-dir outputs/runs/upstream_scale_s02/planners_seed23_cpu_cuda_new \
  --device both --restore-panel --resume \
  --junit outputs/runs/upstream_scale_s02/planners_seed23_cpu_cuda_new/metrics/contracts.xml \
  > outputs/runs/upstream_scale_s02/planners_seed23_cpu_cuda_new/logs/diagnostic.log 2>&1
```

准备命令同时打印来自脚本`TEST_FILES`的测试命令。源码或测试变化后必须建立新run，不给已执行的JUnit补写来源。诊断要求完整、同源、同run、零失败及零跳过的必需测试；只有通过这一步才会调用七设置面板。

`--device both`运行CPU的line、quintic、新47、NMPC、MPPI-512／1024／2048，另运行三个CUDA MPPI设置。47编号逐项实际回放，因此计划覆盖共56个调用案例；这只是请求覆盖量，不代表已完成数量或成功数量。`--restore-panel`对每个非原语设置及原语HOLD保存公开状态，在下一次有新信息时刻的输入上比较连续执行和恢复执行，容差`atol=1e-7/rtol=1e-6`。主机代理始终在CPU；每例先实际推进一次注入延迟，再执行实际首控制并核对预测。

`--device cpu/cuda`可以选择单种MPPI设备；NMPC及几何适配器仍在CPU，必需测试集合不因此放宽。MPPI默认关闭墙钟截止，NMPC默认宽裕10 s求解预算；这些耗时不是0.2 s实时通过证据。输入为给定合成平地本地快照，未运行传感链或完整任务回合。

产物检查范围：

- `config/source_snapshot/`、`contracts.source.json`和各`panel_*.json/contracts_*.json`保存生产者／测试来源及每设置身份；acados缓存位于该run的`state/acados_cache/`。
- `samples/*_execution.json`保存名义轨迹、信息时刻储备、独立复核和实际首控制；`*_restore.json`保存恢复误差；原始MPPI末轮采样与公开状态保存为本地诊断快照。
- `metrics/summary.json`列出实际调用、有效性、回退和逐例失败；`strict_acceptance.json`只记录此次局部面板范围，完整S02、CPU／CUDA完整矩阵、Gate B、T11、实时和PhysX保持未通过。
- 旧路径P0、通信、奖励／终止、动作概率及既有23项真实acados回归另保存JUnit和来源；不能把面板默认范围当作这些回归已完成。

当前`--resume`用于从已准备的同源run继续进入诊断，不提供逐案例断点续跑。已有终端记录会被拒绝；若中断已经生成面板JSON或采样，保存该run并换新run重做，不能覆盖原文件或将中断记成规划失败。S02诊断快照不是训练checkpoint，给定平地首控制／恢复通过也不代表完整S02或T11通过。

## S01新run及测试来源

从仓库根目录执行，下例使用新的run_id；同名已存在时拒绝覆盖。修改源码后重新命名，不续写旧run。先创建绑定完整源码／配置的manifest及测试来源，再执行测试和诊断。

```bash
PYTHONPATH=scripts .venv_isaaclab/bin/python - <<'PY'
from pathlib import Path
from planner_evidence import EvidenceRun, json_write, file_sha256
from diagnose_upstream_execution import REVIEW_V2_TEST_FILES

run = EvidenceRun(
    Path("outputs/runs/upstream_scale_s01/review_v2_seed23_cpu_new"),
    "configs/experiment/upstream_scale_s01_review_v2.yaml",
    settings={"stage": "S01_shared_execution_foundation", "torch_threads": 1},
)
json_write(run.path / "metrics/contracts.source.json", {
    "source_snapshot": run.identity["source_snapshot"],
    "test_sources": {name: file_sha256(name) for name in REVIEW_V2_TEST_FILES},
})
PY

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_upstream_spec.py tests/test_upstream_execution.py \
  tests/test_config_wiring.py tests/test_cuda_evidence_gates.py \
  tests/test_upstream_review_v2.py \
  --junitxml=outputs/runs/upstream_scale_s01/review_v2_seed23_cpu_new/metrics/contracts.xml \
  > outputs/runs/upstream_scale_s01/review_v2_seed23_cpu_new/logs/contracts.log 2>&1

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv_isaaclab/bin/python scripts/diagnose_upstream_execution.py \
  --config configs/experiment/upstream_scale_s01_review_v2.yaml \
  --run-dir outputs/runs/upstream_scale_s01/review_v2_seed23_cpu_new \
  --resume \
  --junit outputs/runs/upstream_scale_s01/review_v2_seed23_cpu_new/metrics/contracts.xml \
  > outputs/runs/upstream_scale_s01/review_v2_seed23_cpu_new/logs/diagnostic.log 2>&1
```

该诊断实际推进两环境×四车并检查1／1.5／2／25/9 m/s、满速停车、延迟生效时间、热恢复和异步reset；使用给定合法本地平坦观测。`metrics/strict_acceptance.json`只颁发公共执行基础通过，不会颁发完整环境或Gate B。当前严格分块复核约3 s，尚未达到实时预算。两项拒绝与查询版本记录在metrics/review_v2_acceptance.json，日志中的计算墙钟不能替代模拟时间。旧S01 YAML仍选原v1／标量语义，仅供历史复现，不用于当前新验收。

## 只运行S00审计

使用 `configs/experiment/upstream_scale_s00.yaml`，创建run时settings为 `{"stage": "S00_contract_foundation", "profile_sha256": cfg.spec.profile_sha256}`。必需测试为 `tests/test_upstream_spec.py`、`tests/test_config_wiring.py`、`tests/test_cuda_evidence_gates.py`，其SHA与生产快照写入该run的 `metrics/contracts.source.json`，再运行pytest保存 `metrics/contracts.xml`。

```bash
.venv_isaaclab/bin/python scripts/audit_upstream_scale.py \
  --config configs/experiment/upstream_scale_s00.yaml \
  --run-dir outputs/runs/upstream_scale_s00/review_v2_seed23_cpu_new \
  --resume \
  --junit outputs/runs/upstream_scale_s00/review_v2_seed23_cpu_new/metrics/contracts.xml
```

S00审计不调用运动规划器，不能证明MPPI／NMPC已迁移。`configs/design/upstream_vehicle_proxy_v2.yaml`不含experiment，直接运行仍报错。恢复时规范、完整配置、种子、设备、网格及版本必须匹配；快照是本地可信诊断状态，不是训练checkpoint。

## S03合法通信／实际延迟／停车组件

[独立记录](../experiments/upstream_scale_s03_components_2026-10-13.md)对应`upstream_scale_s03.yaml`：K512、两轮、CPU代理、0.5 s注入延迟，显式选择实际控制历史及静态位姿缓存。新run先冻结源码和测试来源，再运行两批完整测试。实际CUDA及acados必要测试不能跳过。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv_terrain_cuda/bin/python scripts/diagnose_upstream_s03.py \
  --run-dir outputs/runs/upstream_scale_s03/legal_communication_seed23_new \
  --prepare-only

ACADOS_SOURCE_DIR="$PWD/.venv_isaaclab/acados" \
LD_LIBRARY_PATH="$PWD/.venv_isaaclab/acados/lib" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=scripts \
.venv_terrain_cuda/bin/python - <<'PY'
from pathlib import Path
import subprocess
import sys
from diagnose_upstream_s03 import TEST_FILES, DRIVER_FILES, RELATED_FILES
run = Path("outputs/runs/upstream_scale_s03/legal_communication_seed23_new")
for name, files in [("contracts", (*TEST_FILES, *DRIVER_FILES)), ("related", RELATED_FILES)]:
    with (run / f"logs/{name}.log").open("x") as log:
        subprocess.run([sys.executable, "-m", "pytest", "-q", *files,
            f"--junitxml={run}/metrics/{name}.xml"], stdout=log,
            stderr=subprocess.STDOUT, check=True)
PY

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv_terrain_cuda/bin/python scripts/diagnose_upstream_s03.py \
  --run-dir outputs/runs/upstream_scale_s03/legal_communication_seed23_new \
  --resume \
  > outputs/runs/upstream_scale_s03/legal_communication_seed23_new/logs/diagnostic.log 2>&1
```

输入为已经接近集合的7.2 m方格fixture。诊断真实采样八条MPPI MOVE、通过持续强制低速证书执行，随后使用合法leader场地及停车状态机请求HOLD；检查实际排队反馈、驻留、消息／规划器／记忆／脚本公开恢复和异步reset。MOVE推进与任务停车分别报告；这里不执行10–50 m出生、完整集合、Actor／奖励／训练或安全QP，不产生S03完整门控。

同名run不覆盖；中断或失败后的既有产物保留，修改源码后换新run。详细JSON而非日志末行决定通过范围。

### S03精度／真实观测与驱动修复

启用实际Actor／Critic构造时，上述prepare及resume命令均加`--config configs/experiment/upstream_scale_s03_observation_v1.yaml`。该配置使用MPPI v2、消息v2及411／991布局；每五步和结束事件保存观测帧。共享网络仅作零隐状态前向，完整循环学习／奖励／安全仍未接通。见[精度与观测记录](../experiments/upstream_scale_s03_precision_observation_2026-10-13.md)。

若只有`diagnose_upstream_s03.py`本身修复、其他生产及测试／配置依赖全部未变，可显式复用实际执行过的合同run。新run仍须执行驱动契约及真实组件诊断；原JUnit保持在原run，不能把复制文件写成重新执行。新增文件、算法修改、任何测试SHA／配置／依赖变化均拒绝复用。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv_terrain_cuda/bin/python scripts/diagnose_upstream_s03.py \
  --config configs/experiment/upstream_scale_s03_observation_v1.yaml \
  --run-dir outputs/runs/upstream_scale_s03/precision_observation_seed23_repair_new \
  --contracts-from outputs/runs/upstream_scale_s03/precision_observation_seed23_cpu_cuda_20261013_01 \
  --prepare-only

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv_terrain_cuda/bin/python -m pytest -q tests/test_upstream_s03_diagnostics.py \
  --junitxml=outputs/runs/upstream_scale_s03/precision_observation_seed23_repair_new/metrics/contracts.xml \
  > outputs/runs/upstream_scale_s03/precision_observation_seed23_repair_new/logs/contracts.log 2>&1

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv_terrain_cuda/bin/python scripts/diagnose_upstream_s03.py \
  --config configs/experiment/upstream_scale_s03_observation_v1.yaml \
  --run-dir outputs/runs/upstream_scale_s03/precision_observation_seed23_repair_new \
  --contracts-from outputs/runs/upstream_scale_s03/precision_observation_seed23_cpu_cuda_20261013_01 \
  --resume \
  > outputs/runs/upstream_scale_s03/precision_observation_seed23_repair_new/logs/diagnostic.log 2>&1
```

旧S03无观测配置仍保留复现入口；上面的旧说明“不执行Actor”仅指该配置。新观测配置增加数据与网络前向，`recurrent_rollout_connected/s03_full_acceptance_pass/t11_pass`仍保持false。


## 独立校准和真实循环学习准备

36例单次校准从独立默认值开始，先准备再同源续跑：

```bash
.venv_terrain_cuda/bin/python scripts/calibrate_upstream_planner.py \
  --config configs/experiment/upstream_scale_s04_calibration_cuda_v4.yaml \
  --run-id calibration_new --prepare
.venv_terrain_cuda/bin/python scripts/calibrate_upstream_planner.py \
  --config configs/experiment/upstream_scale_s04_calibration_cuda_v4.yaml \
  --run-id calibration_new --resume
.venv_terrain_cuda/bin/python scripts/diagnose_upstream_local_closures.py \
  --config configs/experiment/upstream_scale_s04_selected_cuda_v4.yaml \
  --run-id calibrated_closures_new \
  --calibration-summary outputs/runs/upstream_scale_s04/calibration_new/metrics/calibration_summary.json
```

`--families`只作定向探索，缺类时不颁发完整S04验收。结果以单次`calibration_summary.json`和独立`closure_summary.json`分别解释，不能改写单次结果中的false门控。

四组真实GRU采样/诊断更新与正式运行定义分别执行：

```bash
.venv_terrain_cuda/bin/python scripts/diagnose_upstream_ablation_learning.py \
  --config configs/experiment/upstream_scale_s04_target_probe_cuda_v4.yaml \
  --run-id actual_heads_new
.venv_terrain_cuda/bin/python scripts/prepare_upstream_t11.py \
  --config configs/experiment/upstream_scale_s04_selected_cuda_v4.yaml \
  --run-id training_definitions_new
```

前者是12次有界诊断更新，后者零更新/零交互；均不是100k正式学习。源码、概率、执行库和实际配置绑定恢复，旧checkpoint不迁入。

## 新S05完整矩阵

仅在选定参数校准闭环出口满足后执行完整评测。以下run名须新建；首次prepare冻结1000输入，续跑复核全部指纹，已完成分片不覆盖：

```bash
.venv_terrain_cuda/bin/python scripts/benchmark_upstream_planners.py \
  --config configs/experiment/upstream_scale_s04_selected_cuda_v4.yaml \
  --run-id gate_b_new --phase prepare
.venv_terrain_cuda/bin/python scripts/benchmark_upstream_planners.py \
  --config configs/experiment/upstream_scale_s04_selected_cuda_v4.yaml \
  --run-id gate_b_new --resume --phase all
.venv_terrain_cuda/bin/python scripts/evaluate_upstream_gathering.py \
  --config configs/experiment/upstream_scale_s04_selected_cuda_v4.yaml \
  --run-id gathering144_new
```

真实acados使用本机已有`ACADOS_SOURCE_DIR`和库路径，不能以skip替代通过。`--phase plan/repeat/local/batch/report`可分别运行；`--start/--stop`分片及`--variants/--families`定向调用不降低完整计数。NPB读取已接受的real_base场景及数组SHA，不重新放大地图。正式集合必须144完整回合、两规划器各9简单长时限回合成功；局部停车不能计集合成功。

完整集合和局部闭环每宏步保存公开恢复状态。gzip是无损存储优化，SHA不符拒绝恢复；不能跨生产源码/配置续接严格验收。已有不可变输入可在新run显式记录原产物SHA后复用，执行生产者仍独立冻结。当前事实与失败见[本轮记录](../experiments/upstream_scale_training_preparation_2026-10-14.md)，生成入口不解除训练暂停。

## T11运行入口诊断与恢复

以下命令只运行有上限的实际采样/PPO诊断，不启动100k训练，不生成训练checkpoint。每次使用新run；源码/配置改变后不可继续旧热状态。

```bash
.venv_terrain_cuda/bin/python scripts/run_upstream_t11.py \
  --config configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml \
  --run-id session_a0_resume_cpu_20261015_01 --representation A0 \
  --mode diagnostic --seed 23 --rollout-steps 4 --samples 16 --num-envs 1 \
  --diagnostic-updates 2 --updates-this-invocation 1
```

同一冻结源码下，第二次使用相同命令并追加该run的`--resume-state .../state/diagnostic_session_000001.pkl.gz`，验证第二批持续采样及优化器恢复。上述A0 run已由两个独立进程完成8宏步/10轮PPO；它属于原session v1源码，当前v2不能直接续写。已完成的run只读查看，不重复执行覆盖；新诊断使用新run-id。诊断评估只验证一例一步的入口，不能标记独立18场景通过。

正式模式仍要求明确解除暂停及完整准入证据，当前不提供启动正式预算的推荐命令。结果与错误见[推进记录](../experiments/t11_progress_repair_2026-10-15.md)。

`scripts/audit_upstream_t11_readiness.py --config <配置> --run-id <新标识>`可生成缺口清单；证据参数`--gate-b/--gathering/--contracts/--cuda/--training-entry/--dependency-audit`指向已有run中的只读记录。缺失记录不会判通过；汇总命令不启动训练、不解除暂停。

## 速度承诺和中断恢复研究

独立配置`upstream_scale_s04_braking_lease_cuda_v5.yaml`启用`braking_clearance_lease_v3`及平坦帧v3；原配置不切换。`scripts/diagnose_upstream_nmpc_stall.py --help`列出公开失败状态的短程对照参数；`--source-run`只接受本仓库生成目录，`--snapshot/--scene-config`须来自同一run。`baseline/lease_cap/braking_lease/braking_lease_cap`均为有界研究，不能当作完整回合。

实际v5完整研究run为`outputs/runs/upstream_scale_s04/braking_lease_v5_full_pairs_20261015_01/`，冻结脚本`logs/run_variant.py`执行原出生NMPC/MPPI512各两例。只从该run的`config/source_snapshot/`运行，保留原依赖环境，避免工作区变化进入续跑；`execute_pair`会校验配置并恢复最新有校验文件的公开状态。已完成回合自动跳过。中断日志保留，恢复输出写新文件；启动前核对进程，不能并发续跑同一案例。

原生47枚举修复后的评测run为`outputs/runs/upstream_scale_s05/formal1000_nativepreview_cpu_20261015_01/`；继承清单保留原生产源码与耗时。其`logs/finish_matrix_resume_01.py`只汇入已完成且状态SHA正确的未变化路径结果，并继续批量和最终汇总。操作记录见`logs/resume_20261015_2320.json`，当前事实见[本轮记录](../experiments/t11_speed_lease_repair_2026-10-15.md)。
