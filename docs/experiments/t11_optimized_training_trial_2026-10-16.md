# T11性能优化后的真实训练试跑（2026-10-16）

## 结果与当前状态

按用户“尝试性能优化后的训练”的指令，使用原生CPU/CUDA计算核和完整轨迹批处理完成一个真实A1续训批次，包含128宏步、256个有效团队转移及5轮PPO。采样142.791 s、PPO更新4.320 s，交互与checkpoint保存合计0.291 s。相比优化前3个真实批次的采样中位数844.275 s，本批采样快5.91倍；主要阶段总耗时快5.76倍。

累计学习进度为1,024/100,000个有效团队转移、4批、20轮PPO，30个参数张量发生改变，最大改变量0.00056396。checkpoint已保存，试跑进程在完整批次边界退出；原训练进程及A0/A2/A3队列仍暂停，不自动启动长队列。完整系统验收仍未完成。

**计算提速成立，运动问题没有解决。** 本批5,120个车级底层执行步全部为`unverified_brake`，实际速度最大值为零；其中有效MOVE覆盖2,545个执行步，也全部停住。没有碰撞或间距违规，驻留计数始终为零。不能把静止无碰撞、正奖励或Critic误差下降解释为集合成功。

## 固定条件与恢复来源

- 配置：`configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml`，A1、种子23、2环境×4车、MPPI-512、两轮优化、40步/8 s时域。
- 保留任务、奖励、动作、合法感知通信、物理限幅及安全条件；128宏步/批、5轮PPO、原优化器状态及随机数状态均保留。
- 父恢复点：`outputs/runs/upstream_scale_t11/signal100k_seed23_20261016_01_a1/checkpoints/update_000003.pt`，768个有效转移、15轮PPO。
- 新run：`outputs/runs/upstream_scale_t11/optimized_trial_20261016_01_a1/`。使用`config/producer/`完整冻结代码启动新进程，没有SIGCONT旧代码进程。
- 显式迁移仅更新checkpoint外层`identity.source_snapshot`。16个源码变化全部属于已有优化证据；配置、依赖、协议和trainer状态保持一致，公共恢复接口继续执行校验。迁移来源、SHA及差异在`metrics/migration.json`，没有覆盖原checkpoint。
- 父进程中未提交的第4批仍保留在暂停内存中；新进程从完整第3批重新采样，不把父进程的部分日志重复计入学习预算。

复用[原生优化记录](t11_native_acceleration_2026-10-16.md)的271项通过测试与16宏步精确行为配对，核对最终源码SHA一致，没有新增大型前置测试。运行时确认加载了原生CPU和CUDA动态库。

## 实际学习与执行观察

| 指标 | 本批结果 |
| --- | --- |
| 有效团队转移 | 256/256 |
| 策略动作数／HOLD比例 | 1,024／50% |
| 观测、动作概率、loss、梯度、权重 | 均有限 |
| 首轮重要性比率／KL | 1／0 |
| Critic MSE | 7.151→6.524 |
| 最后一轮近似KL | 0.00001320 |
| 接受请求 | 1,024/1,024 |
| 实际运动 | 0；全部5,120个车级执行步刹车 |
| 任务完成 | 本批无回合结束，无成功证据 |

执行拒绝持续记录`neighbor_swept_clearance`。规划统计包含HOLD，不能将985个`valid=false`全部当作MOVE规划失败。长期邻车预测膨胀与执行租约之间的交互仍是待修复方向；这次只验证优化后真实学习，不改变安全语义，也没有完成该行为问题的因果修复。

按本批主要阶段耗时线性外推，单组100k约16小时，剩余98,976个转移约15.8小时。**这仅是平坦、静止单批的计算量估计**，不是完成时间承诺，不能推广到运动或复杂地形。旧3批与新批的动作不同，因此5.91倍是实际吞吐对比；同状态的因果配对仍以此前16宏步4.23倍证据为准。

## 证据与复现

run内主要产物：

- `logs/launch_trial.py`、`logs/launch.json`：显式迁移、冻结生产者和完整启动命令，限制`--updates-this-invocation 1`。
- `metrics/validation_reuse.json`、`metrics/readiness.json`：已有测试复用及当前源码的有界signal准入；`formal_training_ready=false`。
- `metrics/collection_000004_*.json`、`metrics/update_000004.json`、`metrics/timing_000004.json`：逐段采样及真实优化器更新。
- `state/interactions_000004.pkl.gz`、`metrics/offline_analysis.json`、`metrics/execution_timeline.jsonl`：原始动作概率、执行反馈和离线分类。
- `metrics/trial_summary.json`：性能、参数变化、累计进度及外推边界。
- `checkpoints/update_000004.pt`及其SHA侧车：新的可恢复学习状态，不是已验收部署模型。

训练使用已有`run_upstream_t11.py`，没有修改训练入口或算法。分析命令为：

```bash
.venv_terrain_cuda/bin/python outputs/runs/upstream_scale_t11/optimized_trial_20261016_01_a1/logs/analyze_trial.py
```

产物采用追加写入，以上命令已经执行；重跑分析需另建输出位置。后续优先在冻结失败样本上解决MOVE执行拒绝，再从新版完整恢复点开展短批次行为验证，确认实际推进后继续100k和A0/A2/A3比较。144矩阵及原语47剩余评测保持暂停。

本次修改文档的247个本地链接、6份文档数学分隔符、仓库卫生、两个运行辅助脚本的Ruff F821及`git diff --check`均通过，记录在`metrics/document_checks.json`。训练算法未改，不重复已有271项优化回归。
