# T11原生计算核与环境批处理优化（2026-10-16）

## 结果与范围

在RTX 2060 6GB、i7-10750H、本轮同一A1恢复点上，4宏步三次测量中位数从25.279降至6.683秒，吞吐提高3.78倍、耗时减少73.56%。延长至16宏步各一次配对为97.155→22.952秒，吞吐提高4.23倍。两组配对分别检查10,392和21,804个字段，排除墙钟与明确列出的资源统计后，行为字段全部精确一致。最终相关回归271项通过，0失败、0错误、0跳过。

本轮保留实测最快的CPU/GPU混合实现。规划预测、局部查询及代价张量留在GPU；任务、通信、OSQP及非均匀地形的权威等距校正仍在CPU。**没有实现完整GPU环境后端，也不把有限候选的最优结果称为全局最优。** 40步融合预测适用于属性一致的合法本地点集；仍查询真实支持点、未知洞及支持边界。其他地形继续逐步预测，使用新的原生投影和查询核。完整平坦轨迹批处理需要足迹支持见证，其他情况保留原完整复核。

训练及队列保持SIGSTOP，未更新优化器、未修改原始checkpoint，也没有恢复100k训练。当前恢复点仍是零实际运动的训练状态；此次数据说明计算效率，不能解决邻车长期膨胀导致的执行拒绝，也不能将该倍率直接推广至复杂地形运动回合。

## 固定条件与计算实现

配置：`configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml`；A1、种子23、2环境×4车、MPPI-512、两轮、40步/8秒、FP64物理计算和完整安全复核保持不变。没有缩短时域、减少候选或放松安全条件。

1. 原生MPPI：新增`upstream_native.cu/.cpp/.h/.py`。固定12约束的二维控制投影避免大型候选广播张量；空间哈希在CUDA核内查找，非均匀等距属性继续使用原CPU源顺序。相同属性点集的一整条40步预测融合为一次启动，查询结果供后续GPU代价复用；输入状态发生原位修改时使缓存失效。关闭fast math和FMA收缩，保留原精度及候选优先顺序。
2. CPU环境：C++执行控制投影、限制检查和控制序列验证；地形记忆以数组保存最新时间，几何改变时再更新逐行时间；复用不变地形索引、正向支持见证、不可变规范指纹和相同位姿的地形增益。纯静止尾段一次积分后构造全部原始时刻。平地感知避免对全零高度重复执行拟合矩阵乘法。
3. 数据常驻与完整复核：地形几何未变时复用GPU索引；在严格平坦见证内批量计算整条移动轨迹的全部原扫掠时刻，检查控制、动力学、工作区及邻车，保留首失败原因和已检查位姿数。精度、支持、采样数和碰撞数值边界交回原路径处理；不缓存动态验收结果。另实际比较GPU感知，但未采用。

`setup.py`已包含原生源文件；`planner_evidence.source_snapshot()`和迁移打包入口也纳入`.cu/.cpp/.h`，避免只指纹Python而漏掉实际计算核。配置、adapter和checkpoint结构不变。

## 逐步实测

以下均为同一恢复点、4宏步、三次重复的中位数；各次重新恢复状态。先单独预热，计时边界同步CUDA，排除构造、恢复、首次编译及文件保存，包含真实环境交互和完整复核，不执行PPO。

| 候选 | 秒/4宏步 | 决策 |
| --- | ---: | --- |
| 本轮修改前，即上一轮约19%优化后的版本 | 25.279 | 冻结基准 |
| 原生CUDA投影和查询 | 15.365 | 保留 |
| 融合预测、CPU原生投影和记忆更新 | 11.864 | 保留 |
| 地形索引/见证/增益复用及指纹缓存 | 8.470 | 保留 |
| 上一行加GPU感知 | 8.839 | 不采用 |
| 平地感知和静止预测进一步批量化 | 8.197 | 保留 |
| 整条移动轨迹批量复核 | 6.763 | 保留 |
| 静止初态直接构造速度约束曲线 | 6.683 | 最终保留 |

最后一项增益约1.2%，本轮停止继续堆叠局部调整，转入延长配对与回归。GPU感知相对对应混合版本慢约4.4%，并使一辆车记忆坐标最大相差`3.55e-15`；未通过精确状态一致性要求，原失败记录保留。

| 延长配对 | 修改前 | 最终实现 |
| --- | ---: | ---: |
| 16宏步/32团队转移 | 97.155 s | 22.952 s |
| 有效团队转移/秒 | 0.329 | 1.394 |

延长配对各一次，用于验证较长连续采样及行为一致性；不将它表述为多次统计中位数。4宏步最终三次分别6.828、6.683、6.590秒。当前基准仍使用完整恢复点的同一状态，正式训练暂停。

## 正确性、依赖与复现

新增原生回归覆盖标量多边形参考、低速/倒车/满速边界、控制失败优先级、GPU融合与逐步预测、运动、未知支持、非有限噪声，以及返回轨迹原位修改后的查询失效。原有扫掠对照显式关闭新的批量路径，以继续比较真正的标量参考。最终271项涵盖执行器、感知记忆、地形查询、MPPI精度/校准、复核、安全、环境和训练入口；没有扩大到完整系统矩阵或PhysX验收。

新增运行依赖是C++17编译器；CUDA执行还需要带`nvcc`的CUDA Toolkit。首次调用按源码和GPU架构编译到Torch扩展缓存，此后复用。实测工具链为GCC13.3、nvcc13.0.88、Torch2.10.0+cu126、sm_75；不使用fast math。编译失败显式报错，没有悄悄改设备或减少采样。详情见[迁移手册](../runbooks/t11_training_handoff.md)。

原始run：

```text
outputs/runs/upstream_scale_t11/native_acceleration_20261016_01/
```

重要证据：`metrics/efficiency_summary.json`、`metrics/source_final.json`、`metrics/regression_final.xml`、`metrics/baseline_vs_final.json`、`metrics/baseline16_vs_final16.json`、`metrics/baseline_vs_cuda_sensor.json`。`config/source_before/`保留本轮之前的完整源码，`config/final/`保存最终修改文件；原checkpoint路径及SHA在`run_manifest.json`。中间候选测量用于本轮取舍，最终/基准提供完整重放来源。

从仓库根目录重放时使用新的label，避免覆盖产物：

```bash
.venv_terrain_cuda/bin/python outputs/runs/upstream_scale_t11/native_acceleration_20261016_01/logs/bench.py verify_baseline --source-version baseline --macros 4 --repeats 3
.venv_terrain_cuda/bin/python outputs/runs/upstream_scale_t11/native_acceleration_20261016_01/logs/bench.py verify_final --source-version final --macros 4 --repeats 3
.venv_terrain_cuda/bin/python outputs/runs/upstream_scale_t11/native_acceleration_20261016_01/logs/compare.py verify_baseline verify_final
```

原执行延长配对使用`baseline16 --source-version baseline --macros 16 --repeats 1`与`final16 --source-version final --macros 16 --repeats 1`；GPU感知候选使用`cuda_sensor --cuda-environment`。完整测试命令保存在`logs/regression_command.txt`。

后续恢复学习需要重新启动使用新代码的进程并依据已有授权及当前源码完成对应恢复准备；直接SIGCONT旧Python进程会继续使用其已加载的旧实现。本轮不自动恢复、杀死或替换暂停进程。
