# T11学习速度承诺修复与100k续训（2026-10-16）

2026-10-17停止更新：用户要求先停止训练、专注性能优化。A1与自动队列已退出，训练CUDA上下文已释放；累计完成2,048个有效团队转移、8批/40轮PPO，最新完整状态为新A1 run的`checkpoints/update_000008.pt`，SHA为`a57e8bffe72dcf1234e0a1b2e05a93c3c0445a02d01fee75aa9c753a1888d7be`。原结果、部分采样日志和全部checkpoint保留；未提交的内存采样不冒充恢复点。停止命令及时间见suite的`metrics/training_stop_1792167454069851305.json`。后续训练等待性能达到用户可接受程度及明确启动指令；下文保留停止前实验记录。

## 问题、修复与结果

[优化试跑](t11_optimized_training_trial_2026-10-16.md)中A1全程零运动。原因已用同一个学习checkpoint的短程对照验证：学习collector直接套用了脚本的制动距离速度承诺，但规划和安全层复核的是完整8秒轨迹。所有学习车辆都可能同时MOVE，非零速度上界在8秒内形成的邻车可达圆相互覆盖，连静止停车尾段也被拒绝。

修复学习路径的`renew_legal_bounds`，让速度承诺同时满足原制动距离限制和完整时域的位移预算：

$$
v_{\mathrm{proposal}}=\min\left(v_{\mathrm{braking}},\frac{0.8\max(d,0)}{H+T_{\mathrm{request}}+T_{\mathrm{latency}}}\right).
$$

其中净空$d$只来自本车状态和合法收到的消息，$H=8$秒；承诺不会低于本车上一实际控制，也不会撤销已有未到期的更低承诺。该速度界由执行器实际强制，原预测、8秒复核、车体足迹、QP、碰撞阈值全部保留。规则不检查策略采样动作，A0–A3共用；脚本集合路径的制动承诺保持原行为。

在相同A1恢复点、相同策略/RNG、2环境×4车、MPPI-512两轮条件下，各运行8个宏步，不更新优化器：

| 指标 | 原学习承诺 | 修复后的时域承诺 |
| --- | ---: | ---: |
| 有实际位移的车辆 | 0/8 | 8/8 |
| 最大车辆净位移 | 0 | 0.746 m |
| 最大实际线速度 | 0 | 0.320 m/s |
| `unverified_brake` | 320/320 | 8/320 |
| `verified_brake` | 0 | 159/320 |
| 通过的未修改控制 | 0 | 153/320 |
| 碰撞／间距违规 | 0／0 | 0／0 |
| 有效团队转移 | 16/16 | 16/16 |
| 采样墙钟 | 8.74 s | 100.36 s |

全部8辆车移动，说明先前的零运动卡点已解除；仍保留8次未通过复核的刹车，不能写为全部执行通过认证。车辆运动使感知、地形记忆与完整轨迹复核更频繁，静止时的100k约16小时估计不再适用；后续以真实训练采样为准，不由8宏步外推稳定工期。

## 验证与行为身份

14项针对性测试通过、0失败/跳过，覆盖学习承诺的时域预算、未通信真值变化不影响本车承诺、上一控制及旧承诺保持、旧collector状态拒绝、原脚本承诺、学习角色及signal准入。命令：

```bash
.venv_terrain_cuda/bin/python -m pytest -q -o addopts= \
  tests/test_upstream_learned_lease.py tests/test_upstream_speed_lease.py \
  tests/test_upstream_training_runtime.py tests/test_upstream_signal_admission.py \
  --junitxml=outputs/runs/upstream_scale_t11/learned_lease_repair_20261016_01/metrics/regression.xml
```

仅修改`upstream_ablation_learning.py`中的学习承诺及其元数据。学习collector的`execution_bound_policy`由`braking_clearance_lease_v3`变为`learned_horizon_braking_lease_v1`，旧身份不能静默恢复。任务、奖励、动作分布、网络、MPPI、执行器及安全实现未改；没有新增配置或重跑无关的大型评测。

修复证据目录：`outputs/runs/upstream_scale_t11/learned_lease_repair_20261016_01/`。`logs/probe.py`保存同状态对照，`metrics/before.json`和`metrics/horizon.json`记录两组结果，`state/`保存原始动作、执行和末状态；`config/upstream_ablation_learning_before.py`保留修改前源码，`run_manifest.json`记录父checkpoint及原SHA。其后的训练使用实际修复代码。

## 已恢复的训练

用户已授权“修复问题后继续训练”。新suite为：

```text
outputs/runs/upstream_scale_t11/signal100k_leasefix_seed23_20261016_01/
outputs/runs/upstream_scale_t11/signal100k_leasefix_seed23_20261016_01_a1/
```

A1从`optimized_trial_20261016_01_a1/checkpoints/update_000004.pt`续训，起点1,024个有效团队转移、4批、20轮PPO，剩余预算98,976。保持种子23、128宏步/批、5轮PPO、K512两轮、两环境及360秒任务时限；A1完成后按原授权顺序执行A0、A2、A3，各100k，不增加预算。

新生产代码完整冻结在suite的`config/producer/`。迁移仅修改外层源码身份和两处collector承诺身份，策略、优化器、GRU、RNG、环境、旧承诺及累计进度保留；新A1的`metrics/migration.json`记录字段变化与父子checkpoint SHA。此前1,024条静止转移仍属于历史训练数据，不能重标为修复后的数据。四方案比较需保留这段A1前缀差异。

启动命令与进程信息见修复run的`logs/resume_training.py`及新suite的`logs/launch.json`；逐16宏步采样、每批PPO与checkpoint写入新A1目录。原144矩阵和原语47剩余评测继续暂停。

启动后首段真实训练采样已完成：16宏步、32/32有效转移，累计采样计数1,056；最高下发线速0.376 m/s、角速0.288 rad/s，观测与log-prob均有限。该段墙钟238.4 s，当前第5批仍在采样，最新完整学习checkpoint仍为1,024个转移。证据为`metrics/collection_000005_016.json`，不能将这32条未提交批次数据写成已经完成新的PPO更新。

新进程确认恢复并产出运动样本后，已退出被替代的旧暂停进程，释放其RAM/CUDA上下文；所有磁盘证据与checkpoint不变，旧内存中的未提交批次不计入学习。操作记录在修复run的`metrics/old_process_retirement.json`。修改文件的Ruff F821、文档链接及数学分隔符、`git diff --check`通过。
