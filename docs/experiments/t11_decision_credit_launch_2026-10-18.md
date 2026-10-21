# 三组决策时间／奖励信用实验启动

完成更新：三组于2026-10-19 08:10前全部结束，退出码0，各完成200,000底层环境步及20次采样更新。最终比较见[结果记录](t11_decision_credit_results_2026-10-19.md)，两组候选均未通过预设筛选；下方进程及初始化描述为启动时记录。

用户于2026-10-18明确要求“启动三组实验”。已于北京时间19:26启动顺序A→B→C，采用新run；历史A0/A1恢复点与旧队列保留。

## 配置与预算

三组均为A1网络、seed 23、全新Actor/Critic与Adam，四owner×四环境，CPU skrl学习、CUDA MPPI K512×2，每批5轮PPO。

| 组 | 决策周期 | 折扣参考周期 | 每环境每批决策数 | 底层步上限 |
| --- | --- | --- | --- | --- |
| A | 1 s | 0.2 s | 125 | 200,000 |
| B | 5 s | 0.2 s | 25 | 200,000 |
| C | 5 s | 1 s | 25 | 200,000 |

三个标签表示时间协议对照，不是旧A0/A1/A2动作表示。独立评估每组预留6,000步，合计18,000步。完整区间预算不足即退出；失败停止后续组，不自动重试或增加预算。

## 实际启动

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv_terrain_cuda/bin/python -u scripts/run_upstream_t11.py \
  --config configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml \
  --run-id decision_credit_seed23_200k_20261018_01 \
  --representation A1 --mode decision-credit --decision-credit-arm all \
  --budget 200000 --env-owners 4 --num-envs 16 \
  --far-checkpoint outputs/runs/upstream_scale_t11/oom_fix_owner16_20261017_01_a1/checkpoints/update_000061.pt \
  --release-training-pause
```

监督进程PID 4040564，实验进程PID 4040565；后台独立会话，输出重定向到run日志。监督脚本只记录进程资源与退出码，不自动重启训练。

目录均位于`outputs/runs/upstream_scale_t11/`：

- suite：`decision_credit_seed23_200k_20261018_01/`。
- A/B/C：同名前缀加`_a/`、`_b/`、`_c/`，按顺序创建与运行。
- 启动与监控：同名前缀加`_launch/`；`metrics/launch.json`记录命令和PID，`logs/experiment.log`记录实际输出，`metrics/resources.jsonl`每5 s记录进程树PSS、swap及系统可用内存，`metrics/exit.json`在退出后写入。

A组初始完整checkpoint `checkpoints/update_000000.pt`及四个owner分片已创建；已进入初始状态评估。初始评估完成后进入训练采样，B/C待前组完成后自动接续。行为效果尚无结论。

## 检查与边界

复用[实现记录](t11_decision_learning_implementation_2026-10-18.md)中的45项通过测试与CUDA K512检查，没有重复运行全仓验证。启动前没有其他训练进程，磁盘剩余约225 GiB。首段进程树PSS约2.97 GiB，系统可用约7.55 GiB；这是初始化／评估阶段观测，不代表训练峰值。

首次启动包装器因缺少可选`psutil`模块，在创建run或启动任何实验前退出；已改用标准库和`/proc`记录资源，没有安装依赖、没有消耗学习或评估预算。算法和已验证源码未修改。

训练更新、评估结果和剩余预算以各组metrics为准。此记录不授予完整集合或PhysX验收。
