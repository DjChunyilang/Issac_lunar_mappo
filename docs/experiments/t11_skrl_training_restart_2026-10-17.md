# 优化版skrl续训与前3次更新硬件监测

2026-10-17按用户“使用优化后的框架再次开始训练，使用skrl框架训练，顺便监控一下前几次更新的训练的硬件开销”执行。目标为从有效恢复点继续A1，并通过3个真实更新批次报告当前吞吐和硬件开销。监测结束后继续原100k学习，不扩大预算。

## 运行与来源

- Suite：`outputs/runs/upstream_scale_t11/skrl_signal100k_seed23_20261017_01/`。
- A1：同级`skrl_signal100k_seed23_20261017_01_a1/`。
- 原恢复点：`signal100k_leasefix_seed23_20261016_01_a1/checkpoints/update_000008.pt`，2,048有效团队转移、8批/40轮PPO。
- 已验收迁移副本：`skrl_deep_optimization_20261017_01/checkpoints/skrl_migrated_update_000008.pt`，SHA256 `3aedb82ef5943e55fa0dda607765198771a093da66031109bb1c86019801f125`。
- 本次仅给该副本附加当前独立session身份，完整保留trainer状态；父链、SHA和操作见A1的`metrics/checkpoint_adoption.json`。原checkpoint不覆盖，公开`load_state_dict`执行兼容校验。
- 优化依据：[深度优化/skrl迁移](t11_skrl_deep_optimization_2026-10-17.md)；已核对其生产文件哈希与本次实现一致。suite的`config/producer/`冻结scripts/source/configs，outputs链接至原产物根目录。

实际训练为skrl 2.1.0 MAPPO。A1＋MPPI-512、两轮MPPI、种子23、两环境×四车、128宏步/批、每批5轮PPO，预算仍累计100,000个有效团队转移。任务、奖励、动作和学习速度承诺不变。MPPI在CUDA执行；策略/PPO使用CPU，依据同批CPU4.40s、GPU7.47s的已有结果选择。6线程逐车计算沿用优化默认；未启用改变原采样协议的48环境owner模式。

A1完成后保留既有A0→A2→A3顺序，各100k。A1异常退出则队列停止，不跳过失败。剩余144矩阵与原语47评测仍暂停。

## 命令与最小验证

启动前10项相关测试通过，0失败/跳过，耗时6.53s。JUnit与输出分别在suite的`metrics/entry_validation.xml`和`logs/entry_validation.log`。原优化等价性证据直接复用，未追加大型前置测试。

```bash
.venv_terrain_cuda/bin/python -m pytest -q -o addopts= \
  tests/test_upstream_skrl_migration.py tests/test_upstream_signal_admission.py \
  --junitxml=outputs/runs/upstream_scale_t11/skrl_signal100k_seed23_20261017_01/metrics/entry_validation.xml
```

训练命令逐参数保存在suite的`metrics/launch.json`。它使用冻结入口和如下关键参数；运行包装只给collect/update添加阶段时间戳，不修改计算或随机数。

```text
--config configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml
--representation A1 --mode signal --authorize-signal100k
--seed 23 --budget 100000 --samples 512 --num-envs 2 --rollout-steps 128
--trainer-backend skrl --policy-device cpu
--resume-state <A1 run>/checkpoints/update_000008.pt
```

suite的`logs/prepare.py`保存显式状态接续过程，`logs/supervise.py`负责独立运行和后续队列，`logs/instrumented_train.py`提供阶段时间戳，`logs/hardware_monitor.py`读取硬件；辅助脚本哈希在manifest内。训练日志为`logs/A1_training.log`。未修改生产算法代码。

## 硬件监测口径与实测

监测第9、10、11批，500ms读取训练进程`/proc`和NVML，分别报告采样、PPO及两阶段合计。CPU100%表示一个逻辑核；RSS包含共享页，另记PSS/USS。GPU利用率、显存、功率与温度为整卡口径。监控是独立进程，结束不停止训练。

原始采样为A1的`metrics/hardware_raw.jsonl`，分阶段边界为`hardware_phases.jsonl`，逐批汇总为`hardware_update_000009.json`至`000011.json`；`timing_*.json`另记录交互/快照保存开销。监控自身CPU开销写入`hardware_monitor_summary.json`。

硬件为Intel i7-10750H（6核12线程）、RTX 2060 6 GiB；完整身份保存在suite的`config/hardware.json`。

| 实际更新批次 | 采样时间 | 5轮PPO | CPU平均 | GPU平均 | RSS峰值 | 显存峰值 |
| --- | --- | --- | --- | --- | --- | --- |
| 9 | 166.58 s | 4.32 s | 203.49% | 7.67% | 1,722.73 MiB | 688.06 MiB |
| 10 | 168.85 s | 4.34 s | 204.01% | 7.57% | 1,779.56 MiB | 688.06 MiB |
| 11 | 169.11 s | 4.48 s | 208.22% | 7.48% | 1,836.63 MiB | 688.06 MiB |

CPU/GPU平均值为采样与PPO窗口加权合计。采样阶段CPU平均206.05%/206.59%/211.01%，PPO阶段约103–105%。单个采样区间CPU最高336.11%、GPU最高14%；GPU平均功率约27.6–29.3 W，温度最高72°C。内存峰值约1.79 GiB，PSS最高1,825.51 MiB，显存占用约0.67 GiB。监控自身累计CPU25.99 s/墙钟524.60 s，约占一个逻辑核的4.95%，不计入训练进程CPU数据。

三批连同交互与checkpoint保存的已计时阶段共519.31 s，每批约173.10 s，吞吐1.479有效团队转移/s。PPO占采样加更新的约2.5%，当前主要开销仍是环境交互；原两环境协议下GPU没有满载。按这3批吞吐估计A1从2,816转移到100k还需约18.25小时；这是局部运行速度外推，车辆行为和场景变化会影响时长。独立owner扩批性能不套用于本次原协议。

## 学习结果与交接

前3批768/768团队转移有效，新增15轮PPO；完整恢复点达到2,816转移、11批/55轮PPO，`checkpoints/update_000011.pt`的SHA已核验。实际checkpoint标记为skrl 2.1.0，30个策略参数张量相对恢复点发生变化，参数、观测、log-prob和全部学习指标均有限。最高实际线速0.480 m/s、角速0.350 rad/s，车辆持续运动。窗口内两次回合结束均为`task_deadline`，没有集合成功；这是本次早期学习结果，不阻塞继续采样。

机器可读汇总为suite的`metrics/first_three_updates_summary.json`，包含实际更新、有限性、参数变化、终止记录、资源统计和时间外推。3批监测已完成并退出，A1继续运行至原100k，之后监督进程依次推进A0/A2/A3。当前任务的启动与初期硬件监测结束条件已达到。

本轮只修改受影响的状态/计划/索引文档并新增实验记录；生产代码沿用已验收版本。文档新增链接、空白及`git diff --check`通过，运行辅助脚本Ruff `F821`通过。
