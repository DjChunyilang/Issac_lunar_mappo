# exp167：连续局部目标＋NMPC 完整闭环

## 目的

按用户批准的首轮方案，将原语选择改为连续阶段目标请求，由 acados NMPC 生成带时间的轨迹，再执行通用差速跟踪。保留四车自主选址、去中心化执行与 96 秒真实任务判据。首轮只交付受预算限制的闭环 pilot，不比较正式多种子性能。

## 配置

- `configs/experiment/exp167_local_goal_nmpc.yaml`。
- 独立入口 `scripts/train_local_goal_nmpc.py`，便捷启动 `scripts/run_local_goal_nmpc.sh`。
- N1 多尺度 CNN、128 维 GRU、4 维 tanh Gaussian 请求；每 1 秒决策。
- acados v0.5.5，3 秒时域、0.2 秒底层周期、15 区间，最多 4 个规划线程。
- CPU PyTorch，4 环境，seed23，rollout32，序列16。
- 训练上限 100,000 底层环境交互或 7,200 秒训练墙钟，先到即停；评估独立计时。
- 为把40＋640次开发短训也计入本轮上限，实际pilot命令进一步限制为99,320次交互、7,140秒，不追加额度。
- 训练 near Open，碰撞明确终止。评估 near/far × Open/Mixed/Bottleneck，各16 episode；Bottleneck30坑。
- Active-DSTC、Oracle奖励、原语惩罚、模仿学习及新信用算法均关闭。

安装、接口细节与恢复命令见 [运行手册](../runbooks/local_goal_nmpc.md)。

## 严格标准

完整任务仍要求 dmax≤1.25 m、dispersion≤0.30、每车速度≤0.25 m/s、最小间距≥0.42 m、真实质心区域平整并连续保持8个底层步。碰撞距离0.28 m，96秒超时；规划器停止不等于成功。

本轮16 episode/层仅用于诊断，不宣称达到历史192 episode/层的置信区间验收标准，不把奖励提升或低碰撞当成分层优于原方法的证据。

## 结果表

当前状态：完整 pilot 已结束，完整任务未收敛；网络训练暂停。2026-09-18完成无网络固定目标诊断，详见[诊断记录](exp_167_fixed_goal_diagnosis.md)。

| run | 底层交互 | 结果性质 |
| --- | ---: | --- |
| `smoke_seed23_40_cpu` | 40 | 发现首控制区间变化率约束缺失，不作为算法结果 |
| `smoke_seed23_640_cpu` | 640 | 修正后真实 acados＋循环 PPO 更新通过，非完整任务评测 |
| `pilot_seed23_100k_cpu` | 99,319 | 前后均0/96成功、0碰撞、96超时；156次更新、113.5分钟 |

加上开发短训累计99,999次交互，未超预算。正式短训205个回合中1成功、204超时，不代表稳定成功；规划失败/拒绝比例54.6%、减速回退40.9%。CPU吞吐14.6次底层环境交互/秒，进程峰值内存约660 MiB。六层前后场景hash一致。

640 交互 smoke：规划复核/求解失败比例9.53%，紧急减速9.14%，规划耗时中位数23.35 ms、p95 46.40 ms，含写 checkpoint 的实测吞吐约11.83底层环境交互/秒。此数据只用于工程成本估计，不是正式 pilot 的性能结果。

## 失败分析

开发过程中发现 acados 初始区间不自动继承中间区间的非线性约束，导致首步违反控制变化率。已显式设置 `con_h_expr_0`、`lh_0`、`uh_0`，并补充真实前进/倒车/原地转向/停车回归测试。该修正不改变用户任务或成功标准。

冻结前还修正了“合法停止/已到位被计作阻塞”的反馈错误，并将缓存改为等价的定长数组存储，避免每次重复拼接全部历史样本。首次训练前评估主动停止并保留在 `development_prefreeze_seed23/`，没有启动训练，不计入最终前后对比。新路线18项测试包含真实acados四类运动、异常/非数值输出、变时长GAE、终止状态、GRU序列和缓存恢复；静态检查通过。

随后复核发现逐时刻绑定上一条轨迹的单个支持圆盘会造成不必要的路径锁定，已改为各时刻共享本地观测支持圆盘的并集，并增加真实求解器目标反向切换测试。仍不查询全局地形、不增加独立路径搜索。对应预冻结评估保留于 `development_terrain_constraint_seed23/`，同样没有进入训练，不混入最终配对结果。最终冻结实现的23项新路线测试全部通过（包含5项真实求解器测试）；acados只报告API弃用警告。

全仓库复跑记录：752通过、5跳过、6失败；失败均为换机后缺失历史 `outputs/.../paired_configs/near_open.json` 或 `far_bottleneck.json`，不是新路线回归。此计数早于最后新增的5项边界测试；完整日志保留在 `development_terrain_constraint_seed23/metrics/regression_tests.log`。

已知方法风险：未知区域和离散观测支持圆盘会限制前瞻，局部 NMPC 可能陷入局部极小值；旧消息增长的不确定范围可能引发阻塞。达到 SQP 迭代上限但通过执行复核的解单独统计，不称为求解收敛。减速回退不承诺真实环境无碰撞。

## 产物路径

```text
outputs/runs/exp167_local_goal_nmpc/pilot_seed23_100k_cpu/
  config/experiment.yaml
  config/*_frozen_scenarios.json
  checkpoints/initial.pt
  checkpoints/latest.pt
  metrics/baseline_eval.json
  metrics/final_eval_proxy.json
  metrics/summary.json
  metrics/train_metrics.jsonl
  metrics/*_trajectories.json.gz
  run_manifest.json
```

上述pilot原始结果已生成。`latest.pt` 包含优化器、调度器、配置版本、环境/缓存/GRU和随机状态，不代表正式推荐 checkpoint。

## 结论

新闭环及参数更新可运行，但首轮完整任务没有稳定成功。后续固定目标诊断已在无网络情况下复现规划—执行问题，不能将失败全部归于学习层。旧路线、用户原有未提交修改与 CPU PyTorch 环境保留。

## 下一步

保持网络训练暂停。用户已批准并完成规划层`prefix_consistency_v2`修正，同批固定目标结果见[修正记录](exp_167_planner_correction.md)；这不是新版网络训练结果。下一步先关注持续运动、近距协作和稀疏消息闭环，不得因固定目标或单元测试通过自动追加训练预算。
