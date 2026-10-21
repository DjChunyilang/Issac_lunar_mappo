# T11训练服务器交接

日期：2026-10-15。当前交付源码/运行数据和诊断工具，正式训练暂停；目标训练服务器尚未指定。训练实验定义见[比较协议](../t11_ablation_protocol.md)，现有本机性能不外推为目标GPU吞吐。

## 可迁移包

[打包工具](../../scripts/package_upstream_transfer.py)生成`artifacts/source_and_runtime_data.tar.gz`和SHA清单，包含实际工作区源码、配置、测试、依赖入口及NPB运行数组。使用工作区文件而非仅Git HEAD，因为当前实现仍有未提交内容。包不含模型、优化器、venv或acados编译缓存；不把诊断权重作为训练初态。

```bash
.venv_terrain_cuda/bin/python scripts/package_upstream_transfer.py \
  --run-id training_transfer_20261015_01
```

上面的run如已存在只读使用，重新打包须换run-id。解包到目标服务器独立目录后，在该目录执行：

```bash
python3 scripts/verify_upstream_transfer.py --manifest transfer_manifest.json --root .
```

清单保留文件字节数、SHA、生产机Python/包版本和acados提交。地图保持仓库相对路径；同一NPB窗口只是开发样本，不增加地理独立性。传输完整性不等于目标硬件验收。

## 环境准备

本仓库已有安装入口可复用：`scripts/install_proxy_stack.sh`建立Python3.12 Proxy环境，历史目录名`.venv_isaaclab`不意味着安装了PhysX；`scripts/install_nmpc.sh`固定acados提交；`scripts/install_terrain_cuda.sh`与`scripts/install_planner_cuda.sh`建立独立Torch/CUDA和OSQP环境。先检查目标机与包清单的差异，再选择对应入口，不复制本机二进制venv。

现有CUDA安装脚本固定Torch2.10/cu126并含本机`sm_75`检查，不能直接把这个检查当作任意新GPU的兼容判据。目标GPU确定后按其实际架构、驱动和可用构建选择独立环境版本，记录变化，再运行已有同噪声及恢复验证。这里不指定尚未核实的驱动版本或吞吐。

acados库路径在运行前设置为目标目录：

```bash
export ACADOS_SOURCE_DIR="$PWD/.venv_isaaclab/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

2026-10-16起，当前计算实现还包含`upstream_native.cu/.cpp/.h`，迁移包必须保留它们。CPU路径首次调用需要C++17编译器；CUDA路径另需可被`CUDA_HOME`发现的CUDA Toolkit及`nvcc`。本机验证为GCC13.3、nvcc13.0.88、Torch2.10.0+cu126和sm_75。首次调用在Torch扩展缓存编译，编译失败直接报错；基准预热排除首次编译时间。`.cu/.cpp/.h`已纳入源码指纹和迁移打包。数值/吞吐证据见[原生优化记录](../experiments/t11_native_acceleration_2026-10-16.md)。

NMPC需要在新机器重建缓存。MPPI执行不依赖acados，但完整七设置评测需要该基线。模型当前位于CPU，MPPI可在GPU；不能仅凭CUDA可用就称整条采样链在GPU执行。

## 最小启动与资源测量

先验证包、依赖及实际设备，再从已有测试中选择受迁移影响的同噪声、恢复和真实求解器用例；相同依赖的原有充分证据复用。不要默认重跑所有历史实验。

CPU计时入口如下，仅做2宏步有界诊断和5轮更新，不生成训练checkpoint：

```bash
.venv_terrain_cuda/bin/python scripts/profile_upstream_t11.py \
  --config configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml \
  --run-id target_cpu_profile_01 --num-envs 1 --macros 2 --samples 64
```

`--num-envs 2`另建run检查批量增益。正式K512和成熟记忆状态另测，不能由K64冷启动推算100k预算。GPU测量需要同步设备计时，并把CPU足迹/通信/保存成本分开；本次CPU嵌套计时工具明确拒绝CUDA配置。

磁盘按两部分规划：源码/地图包大小由manifest给出；交互/诊断状态按实际每批字节数和保存频率估算。现有A1的128宏步交互压缩约492 KiB、完整诊断状态约7.5 MiB，只是当前单环境例，不能线性当作所有场景固定大小。保留阶段末和定期恢复点；不在线删除已有证据或改变记录覆盖率。

## 恢复与正式启动边界

运行器为`scripts/run_upstream_t11.py`。诊断与训练身份隔离；恢复需源版本、配置、角色、设备、环境数与公开状态一致。CPU/CUDA热状态不跨设备静默迁移。场景种子列表持久化已修复，失败日志保留。

目前只提供准备和诊断命令。最终配置、目标硬件、完整Gate B/144回合与依赖审核齐备后，再使用正式角色和明确解除暂停的入口启动100k；诊断状态不转为正式训练初态。现有准入检查不因迁移包或本页而解除。

## 2026-10-16首次100k独立准入（当前执行规则）

用户已明确调整顺序：[当前T11计划](../t11_handoff_plan.md)覆盖上文“完整矩阵先于100k”的旧启动条件。剩余144和原语47暂停，不阻塞首次信号检查；完整验收仍未通过。

```bash
.venv_terrain_cuda/bin/python scripts/audit_upstream_t11_readiness.py \
  --config configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml \
  --run-id signal_admission_new_run \
  --scope signal100k --authorize-signal100k \
  --signal-validation outputs/runs/upstream_scale_t11/closure_performance_20261016_01/metrics/targeted_validation_summary.json
```

实际已启动的顺序suite为`outputs/runs/upstream_scale_t11/signal100k_seed23_20261016_01/`，准入在其`metrics/readiness.json`。源码/config保存在`config/source_snapshot/`；后续从该冻结目录继续，不能用已变化的工作区源码静默恢复。

```bash
.venv_terrain_cuda/bin/python scripts/run_upstream_t11.py \
  --config configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml \
  --run-id new_signal_a1_run --representation A1 --mode signal \
  --authorize-signal100k --readiness /absolute/path/to/owned/readiness.json \
  --seed 23 --budget 100000 --samples 512 --num-envs 2 --rollout-steps 128
```

上述是独立run示例，不要重复启动正在运行的suite。`run_upstream_t11_signal_suite.py`负责A1→A0→A2→A3顺序调度。各组独立新初始化，A0/A1/A2共用MPPI，A3保留原生47。每16宏步记录采样，每128宏步实际更新5轮PPO，每次更新保存checkpoint。signal角色不调用18场景长评估；它与完整`train`角色、旧`diagnostic`角色分别记录。

真实学习指标在`metrics/update_*.json`，实际动作/反馈在`state/interactions_*.pkl.gz`，恢复点在`checkpoints/update_*.pt`及SHA sidecar。有效样本、参数更新与实际移动分别检查。来源、角色、配置、GRU、环境、优化器和RNG必须匹配；一次完整更新之前退出，没有完整学习恢复点时不能把已记录部分采样伪装成完成更新。

500k/2M没有获得此次授权，signal准入也不授予完整系统验收。原13个集合结果（12成功、1个reverse截止失败）和228条局部评测及最新状态保持原地，暂停记录在`signal_release_20261016_01/metrics/evaluation_pause.json`。
