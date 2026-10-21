# T11训练交互与复核计算优化

日期：2026-10-16。按用户要求依次实施CPU批量复核、CPU进程并行和GPU批量计算对照。训练及四方案队列保持暂停；本轮只有固定checkpoint的有界采样回放，没有PPO更新，没有增加训练预算。

## 结果与选用实现

当前A1恢复状态下，原实现每4个宏步中位耗时30.240 s；选用实现为25.483 s，墙钟降低15.73%，采样吞吐提高18.66%。一个宏步同时推进两个环境、八辆车、五个底层步，折合每宏步7.560→6.371 s，有效团队转移0.265→0.314条/s。

| 实现 | 4宏步墙钟中位数/s | 相对原实现吞吐 | 判断 |
| --- | --- | --- | --- |
| 原实现 | 30.240 | 1.000倍 | 固定恢复点基线 |
| 仅批量返回首个拒绝 | 30.457 | 0.993倍 | 单项没有整体收益 |
| 上项＋两环境复核进程 | 28.395 | 1.065倍 | 仅早期版本有小幅收益 |
| CPU整条静止轨迹批量复核 | 25.804 | 1.172倍 | 保留 |
| 上项＋GPU查询同步优化 | 25.483 | 1.187倍 | 最终采用，CPU串行调度 |
| 最终实现＋两环境复核进程 | 25.718 | 1.176倍 | 比最终串行慢，不接入 |
| 最终实现＋CUDA批量邻车扫掠 | 25.706 | 1.176倍 | 比CPU扫掠慢，不接入 |

每行均为同一恢复点的3次独立恢复、每次4个连续宏步；GPU在计时边界同步。预热、恢复、序列化不计入墙钟，全部变体保留2环境、K512、两轮MPPI及策略/规划器随机状态。测量期间原训练进程为停止状态。

最后从`config/source_before/`显式加载冻结包复跑基线3次，中位30.583 s，68个实际加载模块的SHA逐项核验；与选用实现的非耗时字段仍完全一致。该复跑对应约1.20倍吞吐，正文保守使用首次基线的1.19倍，不把短窗口波动作为额外收益。证据为`baseline_frozen.json`及`baseline_frozen_vs_combined_serial.json`。

每个变体首个4宏步载荷与基线配对：动作、原始概率、观测、奖励、实际控制、规划输出/拒绝、任务、局部记忆、模型/GRU和RNG等10,392个非耗时叶字段逐值一致；只有显式墙钟字段不同。各变体并未重新训练模型。

## 保留的修改

1. `upstream_flat_review.py`的平坦扫掠批量检查可直接返回原顺序的首个失败及`checked_poses`，避免发现邻车冲突后再次逐点扫描。地形证明未覆盖、未知和数值边界仍执行原路径。
2. 对状态完全不变、控制与实际速度均为零的完整轨迹，静态地形/动力学检查复用一次；按每段原采样密度构造全部未来时刻，批量检查邻车位置及不断扩大的预测半径。保留半采样余量、第一失败原因和检查点计数；没有缓存最终动态通过状态，也没有减少8 s预测时域。
3. `upstream_tensor_views.py`在小占用桶使用已知上界，减少每查询块的GPU标量同步；将同一次lookup的等距/支持边界校正合并到一次CPU往返。五通道危险合并、未知支持边界、float64几何和原内存预算保持不变。

运行配置、物理/任务阈值、动作概率和checkpoint格式没有新增版本。代码身份由本run单独记录；不改写原训练生产者。

## 进程与GPU实验为何未接入

CPU原型以两个持久进程按环境处理四辆车的安全投影/执行预测，GPU规划仍保留原批次，主进程按原车辆顺序提交结果。静止轨迹批量化之后，进程通信和局部地形传递抵消了并行收益。已撤下运行入口及进程池代码，原型源码保留在本run的`config/combined_workers/`和`config/retired_parallel_prototype/`。

CUDA原型一次处理一条完整静止轨迹的全部邻车扫掠距离，保留float64和边界回查。真实输入为3个邻车×360个时刻，100次微基准中位耗时：CPU 0.0343 ms，CUDA含上传/回传0.1871 ms；数值最大绝对差为0。该规模下GPU路径约慢5.45倍，因此完整复核仍由CPU承担；这不是对未来更大常驻GPU批次的结论。

一个相同宏步的嵌套host计时解释了收益：

| 区段 | 早期首拒绝优化后/s | 最终实现/s |
| --- | --- | --- |
| 全部轨迹复核，116次累计 | 1.750 | 0.624 |
| 执行前安全投影，40次累计 | 0.745 | 0.219 |
| 执行后预测，40次累计 | 1.018 | 0.516 |
| 规划池 | 3.727 | 3.541 |
| 记忆更新 | 0.526 | 0.524 |

这些区段存在嵌套，不能直接相加；它们不用于计算表首的端到端加速比。剩余主要成本是MPPI预测/代价与环境记忆更新。

## 验证、复现和边界

157项针对性测试通过，零失败、零跳过：完整/平坦/缓存复核、危险与边界、CPU/CUDA局部查询、MPPI精度/校准、真实安全QP和环境闭环。新增9例覆盖静止轨迹的未来半径扩张、邻车穿越、净空/采样数量边界、未知、非法控制、状态不一致和停车；沿用原同段扫掠比较。未增加完整系统验收矩阵。

```bash
.venv_terrain_cuda/bin/python -m pytest -q -o addopts= \
  tests/test_upstream_equivalent_acceleration.py \
  tests/test_upstream_frame_review.py tests/test_upstream_visibility_flat_review.py \
  tests/test_upstream_review_cache.py tests/test_upstream_review_v2.py \
  tests/test_upstream_tensor_views.py tests/test_upstream_mppi_precision.py \
  tests/test_upstream_mppi_calibration.py tests/test_upstream_safety.py \
  tests/test_upstream_environment.py
```

原始证据：

```text
outputs/runs/upstream_scale_t11/interaction_acceleration_20261016_01/
  run_manifest.json
  config/source_before/
  config/combined_serial/
  config/combined_workers/
  config/cuda_sweep/
  metrics/efficiency_summary.json
  metrics/baseline_vs_*.json
  metrics/cpu_spans.json
  metrics/selected_spans.json
  metrics/regression_final.xml
  logs/bench.py
  logs/compare.py
  logs/regression_final.log
```

固定输入为`signal100k_seed23_20261016_01_a1/checkpoints/update_000003.pt`，768个有效团队转移；SHA见本run manifest。诊断通过同一公开恢复契约加载，不写入原run。可在仓库根目录用新的结果标签重放：

```bash
.venv_terrain_cuda/bin/python outputs/runs/upstream_scale_t11/interaction_acceleration_20261016_01/logs/bench.py replay_baseline --source-version baseline
.venv_terrain_cuda/bin/python outputs/runs/upstream_scale_t11/interaction_acceleration_20261016_01/logs/bench.py replay_selected
.venv_terrain_cuda/bin/python outputs/runs/upstream_scale_t11/interaction_acceleration_20261016_01/logs/bench.py replay_workers --workers 2 --source-version combined_workers
.venv_terrain_cuda/bin/python outputs/runs/upstream_scale_t11/interaction_acceleration_20261016_01/logs/bench.py replay_cuda --gpu-review
```

这里的收益针对当前大量静止刹车的真实训练状态，不外推为运动状态下的100k完成时间；零运动行为未由性能修改解决。原训练及suite两个进程保持暂停，原checkpoint、日志不变。暂停进程仍载入旧实现，直接SIGCONT不会启用新代码；后续恢复需明确记录新源码与原checkpoint的继承关系。本轮没有签发新训练准入或恢复训练。
