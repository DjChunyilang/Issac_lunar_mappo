# T11 集合评测等价加速与四进程续跑

日期：2026-10-16，事实截至15:45（Asia/Shanghai）。用户选择“先尽量加速，再按实测决定”。本轮保留144回合、K512、两轮优化、原场景/种子分组及成功判据；不启动学习。

## 结论与当前执行

真实热状态配对中，一个宏步（两环境、八车、五个底层步）的MPPI耗时从24.323 s降到12.327 s，NMPC从21.409 s降到10.010 s，分别加速1.97倍和2.14倍。全部非墙钟字段逐值相等。四个场景工作进程已经实际续跑；完成的12回合原样继承，全部成功、违规列表为空，仍有未认证刹车记录，不能解释为全部控制获得安全认证。

当前生产者为：

```text
outputs/runs/upstream_scale_s05/gather144_braking_v5_fast_20261016_02/
```

NMPC和MPPI各两个进程；每个场景/种子对独占进程锁，保留原来的23/31两环境批次及47单环境批次，不改变随机数归属。完成结果、历史耗时与源代码身份原样引用；最新完整公共状态恢复后继续，最多重算中断时尚未保存的一个宏步。旧结果不覆盖，不把旧耗时重新标为加速版耗时。

原冻结benchmark继续原代码，当前7000/7000单次、630/630重复性、228/252局部、14/14批量。局部剩余24条均为原语47枚举，属于另一条长尾，不能把144集合的加速收益直接套用到它。

## 瓶颈与实现

本机为i7-10750H、6物理核/12逻辑线程、约16 GB内存、RTX2060 6 GB。原两个集合进程各占约一个CPU核；GPU抽样利用率低。MPPI已经使用CUDA，但局部记忆、CPU足迹复核和环境执行占据大量墙钟，因此不能简单归因于显卡不够快。

原热状态的带分析器宏步为31.64 s：记忆刷新约12.07 s，安全复核约10.23 s，规划调度约3.94 s。调用树有嵌套，不能直接相加。日志JSON序列化仅约0.01 s，不是本次主要优化对象。带分析器耗时不用于计算上述加速比。

本轮实际修改：

1. 相同直接观测重用只读几何和空间索引，只推进新观测的时间；保留历史危险点和其原观测时间。改变位置或属性仍走保守合并。索引坐标批量计算。
2. 属性完全相同的地形查询免除重复等距属性合并；仍执行最近邻支持半径判断，不跨越未知空洞。
3. 同一只读地形共享静态平坦支撑证明；邻车、时间、执行限幅、工作区和最终通过状态不缓存。
4. 同一控制段的原扫掠点批量检查。任何接近边界、危险或无法证明的情况回到原标量路径，保留第一个失败原因及检查点计数。
5. 已在原执行器多边形内的目标控制，其修正目标值为零，直接返回同一结果；不可行目标继续原投影算法。
6. 场景工作队列从两进程改为四进程；每个任务加独占锁，防止重复执行和并发写入同一状态。

原配置、物理模型、任务判据及公开状态格式保持原义。实现源指纹变化由新run显式记录；没有原地修改正在运行的冻结代码。

## 实测与时间判断

| 规划器 | 原代码宏步 | 优化后宏步 | 加速比 | 配对结果 |
| --- | --- | --- | --- | --- |
| MPPI-512 | 24.323 s | 12.327 s | 1.97 | 2515个叶字段核对，除耗时外逐值相等 |
| NMPC | 21.409 s | 10.010 s | 2.14 | 1370个叶字段核对，除耗时外逐值相等 |

配对覆盖执行记录、名义轨迹、实际控制、反馈、奖励、任务状态和每车局部记忆。它是两个真实热状态各一个宏步的配对，不能称完整144回合配对。

四进程实际运行平坦倒车段、同时有原语评测进程占用CPU时，最初采样窗口为：

| 工作进程 | 环境数 | 平均每宏步墙钟 | 连续间隔样本数 |
| --- | --- | --- | --- |
| NMPC，种子23/31 | 2 | 10.90 s | 12 |
| NMPC，种子47 | 1 | 5.09 s | 20 |
| MPPI，种子23/31 | 2 | 15.22 s | 8 |
| MPPI，种子47 | 1 | 7.28 s | 20 |

实测合计约0.649环境模拟秒/墙钟秒。若132个剩余回合都按平均270 s预算、并维持这一平坦段速率，条件外推为约15.3小时。**这不是完整矩阵ETA，也没有证明总加速恰为四倍。** 提前成功会缩短时间，复杂地形、内存增长及拒绝/回退分支可能延长。

15:45再次读取四进程各最近20个状态间隔，条件外推约17.1小时，说明短窗吞吐会波动；当前应表述为约15–17小时的平坦段外推，仍不能充当复杂场景整矩阵ETA。详细数据见`metrics/parallel_throughput_*.json`及`logs/parallel02.log`。

额外采用K512、真实传感链执行单环境四车各三个宏步：坡地10.67/10.48/13.41 s，NPB为11.53/10.12/14.71 s。它们证实复杂地形已经更慢，不能承诺所有剩余回合15小时内完成。继续以实际状态写入间隔和完整回合耗时修正；此次没有足够证据保证整批数小时内完成。

## 提前发现的NPB入口问题

复杂场景短程检查暴露`Scene not prefetched; loading is allowed only at reset`。原`terrain_for`创建SceneBank后未在首次感知前预加载，原队列尚未执行到NPB，因此已有12个平坦结果不受影响。

现已在初始化时按实际scene索引预加载原地图，不修改地形数据或感知权限。真实NPB查询测试和三个实际宏步通过。失败日志保留为`logs/complex_probe.log`，修复后为`logs/npb_probe.log`。该修复另立`fast_02`生产者并继续原状态；`fast_01`只证明加速版本，未作为修复后的NPB证据。

## 验证与证据

文档本地链接、数学分隔符、仓库卫生、全仓Ruff `F821`及`git diff --check`均通过。

本轮针对性测试164项通过、零跳过：执行器54、感知/复核/真实NMPC等98、等价快路径7、队列/144覆盖3、场景恢复与真实NPB预加载2。98项中的`test_upstream_nmpc.py`为24项，包含真实acados求解与解析模型检查；不能把24项全部描述为24次真实求解回归。

```bash
ACADOS_SOURCE_DIR=$PWD/.venv_isaaclab/acados \
LD_LIBRARY_PATH=$PWD/.venv_isaaclab/acados/lib \
.venv_terrain_cuda/bin/python -m pytest -q -o addopts= \
  tests/test_upstream_execution.py tests/test_upstream_nmpc.py \
  tests/test_upstream_sensing.py tests/test_upstream_frame_review.py \
  tests/test_upstream_review_v2.py tests/test_upstream_visibility_flat_review.py \
  tests/test_upstream_equivalent_acceleration.py \
  tests/test_upstream_gathering_continuation.py \
  tests/test_upstream_gathering_resume_scene.py
```

上面为合并复现命令；实际执行分五批，对应JUnit与日志保存在：

```text
outputs/runs/upstream_scale_t11/closure_performance_20261016_01/
  metrics/actuator_checks_01.xml
  metrics/regression_02.xml
  metrics/equivalence.xml
  metrics/scheduler.xml
  metrics/npb_reset.xml
  metrics/hot_state_equivalence.json
  metrics/parallel_throughput_1792135332618567202.json
  metrics/complex_slope.json
  metrics/complex_npb.json
  metrics/deployment_02.json
  logs/probe.py
  logs/compare.py
  logs/deploy.py
  logs/deploy02.py
```

并行吞吐文件实际名称以`metrics/parallel_throughput_*.json`为准。配对源快照、输入状态SHA和输出载荷由各探针JSON记录。两次续跑的`config/equivalent_continuation.json`列出继承文件、SHA、改动源及父run；原采样历史保留在父run，未丢弃。当前进度由独立监控run记录。

## 后续

先收齐余下6个简单集合结果及完整144回合，保持失败分类。性能改动已投入续跑，不再等待完整矩阵才使用。原语47的24条局部闭环另行处理其枚举长尾；其冻结结果和本轮集合结果不混为同一生产者。现阶段不启动100k或正式训练；既定门槛满足后按用户已有授权执行100k信号检查。
