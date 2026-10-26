# 月面训练执行性能修复

## 目标、预算与出口

用户指出硬件利用不足、训练过慢。本轮只优化现有查询、复核及评估调度；最小验收是同状态真实执行的奖励、观测、控制和策略概率保持一致，并使训练采样不再等待初始化评估。月面性能诊断上限200个实际底层步，实际100步、0优化更新；契约测试单独运行，不作为收敛证据。

诊断路径：`outputs/runs/upstream_1km_fixed_far/performance_20261021_01/`。原run及源码归档、checkpoint、日志全部保留。实现和验证完成后立即回到用户已授权的200万步正式训练，不追加调参实验。

## 实测瓶颈

原进程初始化评估时，四个训练owner等待，主进程约137% CPU，GPU采样时多为0%。不能据此认为CUDA未启用：实际MPPI配置仍为CUDA，主要等待的是CPU完整足迹复核。

四个真实月面初态、512样本MPPI、1秒执行段，共20个底层环境步。单线程cProfile总耗时89.38秒，其中`FootprintTerrain.lookup`累计66.54秒，约74%；`FootprintReviewer.trajectory`累计79.48秒，约89%。累计时间相互包含，不能相加。地形查询被拆成143,947个小批，每批重复k=1/k=2近邻查询。分析证据为`metrics/baseline.json`和`baseline.prof`。

## 改动与同状态对照

- `upstream_footprint.py`：批大小256→4096；唯一最近邻只做一次k=2查询，等距情形仍补原k=1来源选择，保留五个通道、危险点合并、支持半径及65,536条邻点分配上限。
- `upstream_flat_review.py`：在不可变局部地形上复用最多4096条精确姿态的静态足迹报告；几何改变自然换缓存。控制、动力学、时变邻车和工作区结果不缓存。静止轨迹可在真实非平地足迹通过后批量检查原有全部未来时刻，不减少扫掠样本。
- `upstream_review_cache.py`：深拷贝时重建临时reviewer，修复月面出生采样器恢复因`ContextVar`不可pickle而失败的问题。配对实验实际执行了环境状态恢复。

未开启cProfile的同状态数据如下。所有组使用相同冻结初态、动作与随机状态，每组实际20步；性能样本较短，不外推整个训练的固定加速比。

| 路径 | CPU工作线程 | 墙钟秒 | 相对参考 |
|---|---:|---:|---:|
| 归档原实现 | 0（串行） | 72.54 | 1.00 |
| 查询＋静态缓存 | 0（串行） | 33.57 | 2.16倍 |
| 查询＋静态缓存 | 6 | 31.72 | 2.29倍 |
| 加非平地静止轨迹批量检查 | 6 | 26.97 | 2.69倍 |

串行对照独立支持约2.16倍计算提速；2.69倍包含不同线程设置。奖励、时长、终端、Actor/Critic观测、位置、log-prob、GRU隐藏状态及实际控制均逐值一致，见`runtime_comparison.json`和`combined.json`。

查询42项契约检查通过；复核、缓存隔离和恢复检查通过（首次新增测试误漏DenseSensor的terrain_cfg参数，修正后重跑通过）；固定评估与旧课程22项检查、TensorBoard6项检查通过。失败日志保留。诊断脚本首次同名遮蔽stdlib profile、并发cProfile工具占用及恢复失败均发生在推进前，未产生学习样本。

## 评估与采样并行

固定远距runner使用一个独立评估进程；初始化及定期评估在独立文件中冻结Actor/Critic和当时学习步数，训练owner可同时采样。评估独立构造环境、RNG与零初始GRU，不引用正在更新的网络；结果横轴仍为冻结时的学习步数。最多一个评估进程，后续评估顺序执行，最终报告等待完成。

评估快照及SHA位于`state/evaluation_<phase>.pt/json`，恢复继续使用原快照。评估异常会终止该训练run，不静默忽略；中断尝试保留整个预算预留，重试另行预留，不把物理步免费重跑。评估子进程只写进度events，主进程将完成的评估JSON投影为正式曲线。新增`sampling/base_steps_per_second`显示实际采样吞吐量，不进入Pinned十项任务指标。

## 替换边界与启动

旧run `a1_seed23_far_lunar_1500s_2m_20261021_01`在0学习步、0更新时冻结。记录到1010评估步，考虑尚未落盘的向量决策，按1110步上界保守记账。初始完整checkpoint SHA核验通过：`f6050f302e6f6553f644ca4c5eded70a159a87aa6c0714db69247a9f58afc3e9`。其SIGTERM只在同步完整评估结束后处理，因此保存替换边界证据后终止被冻结进程；没有丢弃学习更新。

新run `a1_seed23_far_lunar_1500s_2m_perf_20261021_02`于2026-10-21 23:35:52启动，主PID1672256。仍为相同A1／seed23初始化、全新Adam、4 owner×4环境、每owner 2 CPU工作线程；训练配置、1500秒时限、地图、出生、奖励、PPO和MPPI参数不变。旧run不跨源码身份恢复，改用独立run保存执行实现差异。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 \
.venv_terrain_cuda/bin/python -u scripts/run_upstream_t11.py \
  --config configs/experiment/upstream_1km_curriculum_v1.yaml \
  --training-config configs/training/upstream_1km_far_fixed_v1.yaml \
  --run-id a1_seed23_far_lunar_1500s_2m_perf_20261021_02 \
  --representation A1 --mode train --seed 23 --budget 2000000 \
  --num-envs 16 --env-owners 4 --owner-cpu-workers 2 \
  --sampling-protocol owners16_v1 --trainer-backend skrl --policy-device cpu --samples 512 \
  --readiness outputs/runs/upstream_1km_fixed_far/prelaunch_1500s_20261021_02/metrics/summary.json \
  --release-training-pause
```

复用既有有效配置／奖励启动审计，并用本次逐值等价检查覆盖受影响执行路径。学习总预算仍200万；完整评估日程最多36万，加旧评估上界1110仍为361,110／400,000；另记本轮性能诊断100步。没有缩短回合或评估来制造吞吐量改善。

TensorBoard继续使用`http://127.0.0.1:6006`，新events位于新run的`tensorboard/`；原服务已覆盖该run族。启动命令与边界分别见新run的`logs/launch.json`、`metrics/performance_provenance.json`及诊断run的`metrics/replacement_boundary.json`。

当前出口是受影响行为等价与实际并行采样接通；训练成功率与收敛性仍由完整任务评估判断。

## 23:40实际启动复核

新初始Actor/Critic摘要与旧run同为`b5185a57adedce1de80f697a469b49652d18dcaabd26d98933cccd388d4e2e5e`；完整checkpoint、四owner分片、运行源码SHA及冻结评估副本均已核验。首个向量区间已落盘387个实际学习底层步，另有82个独立评估步，PPO尚处首批收集中、0更新。首区间采样约1.85步／秒，包含两个正常地形失败终端，没有failure产物。

10秒硬件采样显示训练与评估合计551.7% CPU（约5.5个逻辑核），GPU该窗口10次读数均为0%，显存1626 MiB。说明四owner已实际工作，但GPU仍受CPU查询／复核供给限制；不能称已充分利用GPU，也不能把短段2.69倍直接当整轮训练加速比。

TensorBoard API已列出新run，事件中的学习387步、采样1.8496步／秒、评估82步与JSON一致。证据：新run的`metrics/launch_verification.json`、`collection_000001_001.json`、`hardware_after.json`和`live_sampling_verification.json`。本轮诊断至此结束，继续原正式预算，不等待首轮PPO完成来追加检查。
