# 月面完整足迹复核的GPU批处理优化

## 目标与边界

用户要求停止训练并优化吞吐。本轮沿用现有局部感知、MPPI、安全复核和执行链，只调整完整足迹／扫掠计算。最多200个真实月面底层环境步，0次优化器更新；不启动或恢复正式训练。验收是同状态行为一致且完整执行耗时改善，不以GPU占用率作为成功标准。

诊断run：`outputs/runs/upstream_1km_fixed_far/gpu_review_20261022_01/`。使用已有四个固定月面初态与冻结A1网络／RNG，MPPI仍512样本、两轮、40步。所有基线与候选依次执行，原训练和评估进程均已停止。

## 正式训练停止现场

`a1_seed23_far_lunar_1500s_2m_perf_20261021_02`已停止，完整初始checkpoint及原始产物保留，PPO更新0。学习采样已落盘2,653步，未提交向量区间最多400步，故物理消耗上界3,053；独立评估已记录871步，上界971。

旧停止处理器只在整批采样与PPO更新完成后检查信号，等待会继续采样一个多小时。因此本次确认0更新与checkpoint校验后冻结并终止训练树，不声称保存了部分rollout的完整恢复点。`metrics/user_stop_for_optimization.json`记录进程、SHA、计数及恢复限制。未来启动需单独处理这部分物理消耗，不能直接从初始checkpoint续跑并把已运行步数清零。

## 实现与取舍

显式候选`configs/experiment/upstream_1km_cuda_review_v1.yaml`选择`review_acceleration_version=cuda_sweep_pose_v1`，继承原1km配置并保持暂停。旧配置及checkpoint身份不变；新版本进入现有解析器、环境和规划器工厂。

- 复用现有`LocalTerrainBatch`及原生CUDA局部空间索引，输入仅为本车合法观测的不可变快照。等距来源与支持边界继续沿用原CPU权威校正。
- 合并同一运动扫掠中的完整足迹，GPU计算接触平面、坡度、残差、台阶及底盘间隙；每批最多32个姿态，不构造查询点×全地图距离矩阵。
- 保留原NumPy足迹坐标变换，避免改变最近邻等距关系。GPU只接受离阈值至少`1e-7`的内部通过点；失败或阈值附近交回原最小二乘计算，保持报告、首次失败原因和计数。这个保护带不修改实际安全阈值。
- 保留精确姿态缓存、去除批内重复姿态；少于8个未缓存姿态使用原CPU缓存路径，避免小批GPU启动开销。只缓存静态地形结果，不缓存邻车、时间、工作区或最终动态验收。
- 单点停车报告、控制合法性、动力学、邻车预测和小型安全QP继续使用现有CPU路径。没有减少扫掠点、MPPI样本或规划时域。

第一版忽略了原CPU路径已有的大量姿态复用，虽然真实回放逐值一致，却从29.15秒变慢至76.53秒，未采用。补回精确缓存并避免小批启动后，同输入耗时降为12.71秒。失败候选源码在`config/cuda_first.py`，结果在`metrics/cuda.json`，不覆盖。

## 已取得证据

| 同状态四初态20底层步 | CPU线程 | 墙钟秒 | 行为一致 |
| --- | ---: | ---: | --- |
| 当前生产基线 | 0 | 29.1547 | 参考 |
| 第一版GPU小批计算 | 0 | 76.5332 | 是，但性能不通过 |
| GPU批处理＋精确复用 | 0 | 12.7093 | 是 |
| 当前生产基线 | 2 | 26.8472 | 参考 |
| GPU批处理＋精确复用 | 2 | 10.5072 | 是 |

串行对照约2.29倍；这是一次短程端到端配对，包含真实感知、规划、安全和执行，不是GPU算子速度，也不直接外推正式16环境的训练工期。奖励、时长、终端、Actor/Critic观测、位置、log-prob、GRU隐藏状态及实际控制均逐值一致。

已有局部查询剖析用于定位取舍：360个未复用姿态CPU参考0.714秒，GPU小批0.063秒、较大批0.033秒；该静态剖析不推进环境，也不能替代上表端到端结果。证据为`metrics/query_profile.json`及对应`.prof`。

与实际owner相同的双线程设置下约2.55倍，耗时减少60.86%。完整回放记录逐字段复核，仅80个`safety_seconds`墙钟字段不同，其余全部一致，见`metrics/record_differences.json`。这不是四owner同时运行的吞吐测量。

最终77项查询、缓存、非平地扫掠、旧平地路径及配置身份检查通过，0失败、0跳过，见`metrics/checks.xml`及`logs/checks.log`。新增检查覆盖精确姿态复用、观测时间更新与地形改变后失效；旧初始checkpoint和四owner分片SHA全部通过。只读记录比较脚本首次对字符串数组错误使用`equal_nan=True`，修正后完成；错误日志保留，不涉及环境或算法失败。

本轮实际100／200诊断底层步（五臂各20步）、0学习更新，已达到出口并结束；静态微基准和契约测试另列，不混入正式学习预算。最终代码／配置SHA及归档在`config/final_sha256.json`、`config/final_source.tar.gz`，汇总在`metrics/summary.json`。正式训练保持停止。采用显式GPU候选，下一次运行前仍需按实际配置检查多owner吞吐与显存；不能把这次短程倍率直接换算为正式完成日期。

测试命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv_terrain_cuda/bin/python -m pytest -q -o addopts= \
tests/test_upstream_cuda_review.py tests/test_upstream_tensor_views.py \
tests/test_upstream_review_cache.py tests/test_upstream_frame_review.py \
tests/test_upstream_visibility_flat_review.py
```

## 复现命令

诊断脚本从同一保存状态恢复；GPU臂仅在该无学习比较进程中替换reviewer构造，不将旧学习状态跨身份恢复到新生产配置。使用新的`--label`，避免覆盖原结果。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 \
.venv_terrain_cuda/bin/python -u \
outputs/runs/upstream_1km_fixed_far/gpu_review_20261022_01/compare.py \
reference --workers 0 --label reference_replay

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 \
.venv_terrain_cuda/bin/python -u \
outputs/runs/upstream_1km_fixed_far/gpu_review_20261022_01/compare.py \
cuda --workers 0 --label cuda_replay
```

基线源码归档`config/reference_source.tar.gz`，初始状态`state/initial.pt`，逐臂记录`state/<label>.pt`、`metrics/<label>.json`、`logs/<label>.log`。本轮不验证收敛或Isaac/PhysX结果。

## 补充：优化版16环境小采样，0 PPO更新

用户最新要求只做优化后小测试，不运行原训练或达到PPO更新批量。因此已准备的`e2e_owner16_20261022_01`整批基线／候选方案在启动前取消，实际0步、0更新；本节是独立追加授权，预算800实际底层步，完成两个决策区间即结束。

run：`outputs/runs/upstream_1km_fixed_far/cuda_owner16_probe_20261022_01/`。使用生产`ProcessUpstreamRollout`和SKRL策略采样器，显式CUDA复核配置；A1／seed23、4 owner×4环境、每owner双CPU线程、MPPI 512样本×2轮、5 s决策、1 s重规划、0.2 s控制。沿用1 km月面训练图及R200–300 m出生分布和1500 s回合上限，无课程、无并发独立评估。配置保留每批32决策／5轮PPO，但此次只调用两次`collect(1)`，不调用任何更新。

| 区间 | 实际底层步 | 团队决策数 | 采样墙钟 | 步／秒 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 387 | 16 | 79.75 s | 4.85 |
| 2 | 377 | 16 | 85.33 s | 4.42 |
| 合计 | 764 | 32 | 165.09 s | 4.63 |

三次回合提前结束使实际步数低于名义800；两区间全部32条转移有效，奖励、观测、隐藏状态和前后价值均有限，实际时长1–25步。更新次数0、Adam状态为空、初始和最终网络SHA一致，且匹配原正式训练初始化`b5185a57…e5e`。独立评估0步。初始化52.96 s单列，采样加TensorBoard及压缩轨迹归档166.83 s（4.58步／秒），含初始化与退出总222.40 s。

采样期间每约2秒记录资源：82个样本，GPU平均41.9%、瞬时最大100%，采样到的整卡显存峰值5140 MiB（约5.02 GiB）；主进程及四owner CPU时间／墙钟约4.68核，最低可用主存4.57 GiB。第二个区间较第一个慢，显存仍在增长；本次未测长期缓存稳定值，不宜据此增加owner数量。该结果是16环境短程采样吞吐，不是完整训练吞吐；未重跑基线，旧长训练还同时执行独立评估，不能把两者相除作为本轮优化加速比。可直接归因的短程配对结果仍为上一节约2.55倍。

验收通过并结束：实际duration累加与计数、TensorBoard横轴387／764及吞吐逐点一致；两份压缩归档SHA、527项源码／配置SHA已核验，所有probe进程退出。没有触发额外环境步或PPO更新。原训练继续停止。

复现入口及命令（须复制至新run并初始化manifest，避免覆盖本次证据）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 \
.venv_terrain_cuda/bin/python -u \
outputs/runs/upstream_1km_fixed_far/cuda_owner16_probe_20261022_01/probe.py
```

本次准确命令存于`config/command.txt`，有效配置与初始化在`config/effective.json`，源码归档为`config/source.tar.gz`；结果`metrics/summary.json`、核验`metrics/verification.json`、资源`metrics/resources.jsonl`、逐区间状态`state/interval_*.pkl.gz`。启动前曾因缺少可选`psutil`和误用禁止覆盖的JSON接口退出，均未创建环境、0物理步；改用`/proc`资源读数与原子manifest状态更新后完成，原错误日志保留于`logs/`。

TensorBoard已在现有`http://127.0.0.1:6006`服务可见，run名`far/cuda_owner16_probe_20261022_01/tensorboard`，性能标签为`Perf/*`，横轴为实际底层步。也可单独回放：

```bash
.venv_terrain_cuda/bin/tensorboard \
  --logdir outputs/runs/upstream_1km_fixed_far/cuda_owner16_probe_20261022_01/tensorboard \
  --host 127.0.0.1 --port 6007
```

## 补充二：顺序优化缓存与轨迹批量查询

用户要求按顺序执行进一步优化，随后沿用前次小批方法验证。run为`outputs/runs/upstream_1km_fixed_far/cuda_owner16_probe_20261022_02/`，预算800实际底层步、0 PPO更新、0独立评估；不运行正式训练。基线复用上一节`cuda_owner16_probe_20261022_01`的相同seed23、4 owner×4环境、每owner双线程和两个5秒决策区间，不增加MPPI样本／迭代或改变地图、出生、奖励、网络、时限与控制频率。

依次实施：

1. **缓存生命周期。** `terrain._review_witnesses`保存`LocalTerrainBatch`，而后者的`_views`反向持有terrain，形成GPU索引循环引用。测试先复现删除最后一个几何所有者后索引仍不释放，日志为`logs/lifecycle_before.log`。修复为索引只持有共享只读数组／空间树且缓存字典为空的查询视图；不复制地图，不改变CPU边界校正。关闭循环GC时仍能及时释放索引，时间戳刷新复用和查询结果一致，见`logs/lifecycle_after.log`。
2. **轨迹静态查询合批。** 非平地移动轨迹先预热最多512个原扫掠姿态的静态结果，GPU按128姿态分块，并合并静态通过掩码回传。原精确姿态缓存与少于8姿态的CPU路径保留；超出预热范围继续逐段检查。起始足迹失败则跳过预热，避免无效整段计算。控制、动力学、动态邻车、工作区、停车阈值和首次失败计数仍由原检查顺序决定。
3. **CPU重复调用。** 将同一轨迹的中心地形查询合批，复用原cKDTree／等距校正规则，各控制步仍读取对应结果并逐项检查。临时结果局部保存，未改变线程数或同步PPO采样协议。本轮没有启用共享GPU服务、增加owner、降低规划质量或更改安全频率。

这是现有实现的性能修复，没有新增行为版本。修改前后源码均归档，旧run／checkpoint不覆盖，后续恢复仍受源码身份校验约束。

受影响检查共81项通过、0跳过：78项查询／缓存／平地和非平地／CUDA检查，以及3项完整40步轨迹后段失败与控制错误优先级检查。见`metrics/checks.xml`、`metrics/horizon_checks.xml`和相应日志。生命周期缺陷的预期失败单独保留，不计为通过；这些检查不推进环境或更新策略。

| 指标 | 优化前probe_01 | 优化后probe_02 |
| --- | ---: | ---: |
| 实际底层环境步 | 764 | 764 |
| 团队决策／PPO更新 | 32／0 | 32／0 |
| 第一区间采样 | 387步／79.75 s | 387步／66.79 s |
| 第二区间采样 | 377步／85.33 s | 377步／77.47 s |
| 总采样耗时 | 165.09 s | 144.26 s |
| 采样吞吐 | 4.63步／s | 5.30步／s |
| 含日志归档吞吐 | 4.58步／s | 5.28步／s |
| 初始化／总墙钟 | 52.96／222.40 s | 54.37／200.09 s |
| 整卡显存采样峰值 | 5140 MiB | 2508 MiB |
| GPU平均利用率 | 41.9% | 37.7% |
| 主进程＋owner平均CPU核数 | 4.68 | 4.85 |
| 最低可用主存 | 4.57 GiB | 7.02 GiB |

本次采样吞吐提高14.4%，采样耗时下降12.6%，显存采样峰值下降51.2%。GPU利用率下降同时吞吐提高，不能用利用率单独判断优化收益。仍是各两个区间的短测，未做重复测量或长期缓存稳定性验证；按5.30步／秒线性外推200万步纯采样约104.9小时，PPO／评估另计，尚未达到一天内完成的水平。

两个区间的全部归档递归比较，只出现3056个`safety_seconds`计时差异，其他字段逐值一致，包括动作／latent／概率、GRU、奖励、价值、实际控制、位置、终止和采样计数。初始及最终网络SHA相同、Adam无状态、所有转移有效；TensorBoard事件横轴387／764及吞吐数值一致。528项源码／配置SHA核验通过，全部probe进程退出，物理预算实际764／800步、0独立评估步、0 PPO更新。本轮结束，不追加采样或恢复正式训练。

启动命令及查看方式：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 \
.venv_terrain_cuda/bin/python -u \
outputs/runs/upstream_1km_fixed_far/cuda_owner16_probe_20261022_02/probe.py

.venv_terrain_cuda/bin/python \
outputs/runs/upstream_1km_fixed_far/cuda_owner16_probe_20261022_02/verify.py
```

准确重定向命令在`config/command.txt`；`metrics/summary.json`是测量结果，`metrics/verification.json`及`metrics/record_differences.json`是对比证据，`state/interval_*.pkl.gz`为非恢复用途的诊断归档，`config/effective.json`为有效配置。现有TensorBoard服务`http://127.0.0.1:6006`已加载`far/cuda_owner16_probe_20261022_02/tensorboard`，查看`Perf/*`；恢复边界仍为0更新的独立测试，不能将此诊断状态直接当正式训练checkpoint。

## 补充三：并行容量与进一步查询优化验证

用户授权在显存释放后尝试扩大并行、检查历史主机OOM风险并继续优化。本轮明确上限为2040实际底层步：20步热点剖析＋20步同状态回放＋16环境最多800步＋24环境最多1200步；0 PPO更新、0独立评估步。沿用A1/seed23、1km月面、1500s回合、5s决策、MPPI512×2、每owner双线程。依次执行，不同时运行两个容量测试；所有步骤保存TensorBoard，正式训练保持停止。

旧OOM不是显存耗尽：历史48环境在集中复制环境并保存时耗尽主机内存；已有逐owner分片修复后，24环境曾完成一批PPO与保存，但保存阶段PSS约10.36GiB、可用主存仅2.59GiB并出现约541MiB进程swap。因此本轮同时监测PSS、USS、SwapPss、MemAvailable和整卡显存，采样约2s一次；主存可用低于3GiB或整卡显存超过5500MiB立即停止。PSS包含共享地图映射，不能用各进程RSS求和判断实际占用。当前任务32决策/批，旧容量记录128决策/批，不能直接搬用历史吞吐或余量。

候选优化在已有CUDA近邻核内证明完整加宽等距集合的五通道属性完全相同，才省去CPU校正；不同属性、支持边界仍走原校正，不更改安全判据。六个新增属性分支检查随原受影响测试共84项通过、0跳过；四环境同状态20+20步动作、控制、奖励、终止逐值一致，只80个计时字段不同。

cProfile中`FootprintTerrain._batch_lookup`仍为最大self项：候选前后均调用4503次，累计约4.87/4.71s；整段15.13/14.66s。该剖析关闭owner线程并打开分析器，仅用于定位，不能作为正式吞吐增益。候选16环境复测相对上一轮5.296→5.159步/s（−2.6%），没有明确收益；全部记录除3056个计时字段外逐值相同。

本轮run根目录为`outputs/runs/upstream_1km_fixed_far/`：`parallel_capacity_20261022_01/`保存剖析／测试／汇总，`cuda_owner16_probe_20261022_03/`和`cuda_owner24_probe_20261022_01/`分别保存容量证据。24环境在两次采样后额外执行一次逐owner状态分片保存，单独计时；这是检查保存容量的诊断，不是含学习器/Adam的完整恢复点。未执行PPO、长期rollout或并发完整评估。

| 指标 | 16环境（4×4） | 24环境（6×4） |
| --- | ---: | ---: |
| 实际底层步／团队决策 | 764／32 | 1147／48 |
| 采样墙钟 | 148.10 s | 207.07 s |
| 采样吞吐 | 5.159步/s | 5.539步/s |
| 含日志归档吞吐 | 5.146步/s | 5.527步/s |
| 采样平均CPU核数（CPU时间/墙钟） | 4.96 | 6.90 |
| GPU平均利用率 | 37.8% | 40.4% |
| 整卡显存采样峰值 | 2508 MiB | 3782 MiB |
| 采样可用主存最低 | 7.94 GiB | 4.83 GiB |
| 全程PSS采样峰值 | 9.23 GiB | 12.43 GiB |
| 全程USS采样峰值 | 5.05 GiB | 8.22 GiB |
| 全程进程SwapPss采样峰值 | 1334 MiB | 1096 MiB |
| 额外owner分片保存 | 未执行 | 6.30 s，6分片 |

24环境的采样阶段进程swap峰值约160MiB，保存阶段上升至1096MiB；保存阶段采样到的PSS峰值12.43GiB、可用主存最低4.92GiB，未触发3GiB停止线，也未发生OOM。16环境本轮初始化已有显著进程换页，两轮后台内存状态不完全相同；不能把两个swap峰值直接解释为24环境更省内存。可用主存高于停止线不等于没有换页压力，约2s监测也不能捕获所有瞬时峰值。

**决定：可以扩大至24环境做短程采样，但当前继续保留16环境作为正式默认，不扩大到32/48。** 环境增加50%，本次吞吐仅增加7.4%，显存增加约50.8%，主机内存和保存换页负担明显。与上一轮16环境5.296步/s相比，24环境仅高4.6%；跨规模增加了不同初态和随机采样序列，且未重复测量，因此不是严格同状态的纯加速倍率。按当前短测线性估计200万步仅采样仍约100.3小时，不包含PPO、评估、后续状态增长。不会因空余显存而静默修改正式PPO批大小或恢复长训。

本轮等属性tie候选没有减少月面CPU查询，端到端也无明确收益，已撤回代码和新增测试，三文件逐字节恢复至本轮前快照；保留上一轮已验证的缓存生命周期和GPU/CPU合批优化。候选源码、84项通过记录及失败/修正日志仍保留。剩余优化空间集中于CPU完整足迹复核/近邻校正中的高频调用与数据往返；下一项应在现有复核实现内消除这些开销，不能只继续增加owner或把所有校正强行搬到GPU。本轮不据热点比例承诺加速倍数，不追加采样预算。

最终验收：两容量run全部80条团队转移有效，网络不变、Adam未更新；duration累计与TensorBoard横轴分别为387/764和582/1147；六分片大小及SHA正确、全部probe进程退出。16环境行为与前轮逐值相同；24环境只验证自身预算/有效性/记录完整性，不宣称跨规模轨迹相同。两run各528项执行源码/配置SHA核验完成。24环境预备源码包先于添加分片保存插桩，首次核验发现仅probe脚本不同；原包和失败日志保留，最终执行包`config/executed_source.tar.gz`按退出manifest中的脚本SHA和其余527个未变SHA补存，见`metrics/source_archive_correction.json`。没有算法源码在测试中途变化。

本轮总消耗**1951／2040实际底层步、0 PPO更新、0独立评估步**；剖析首次因脚本名`profile.py`遮蔽标准库而导入失败，改名后继续，原失败0环境步并单独保留。实际训练仍停止，旧checkpoint和结果不覆盖。

复现命令分别见容量run的`config/command.txt`和套件`config/commands.txt`；使用新run并恢复相应候选源码，不能直接覆盖已完成run或把已撤回候选测试当作当前源码验证。主要证据为各run的`metrics/summary.json`、`metrics/verification.json`、`metrics/capacity_resources.jsonl`，24环境另有`metrics/checkpoint_capacity.json`及`state/capacity_shards/`。套件`metrics/profile_comparison.json`、`checks.xml`、`candidate_decision.json`、`final_source_restoration.json`记录正确性、候选取舍和最终源码。

TensorBoard现有服务`http://127.0.0.1:6006`可回放两个容量run的`Perf/*`及套件的`tensorboard_before/after`；横轴为各自实际底层步，0更新、无学习恢复边界。独立查看：

```bash
.venv_terrain_cuda/bin/tensorboard --logdir outputs/runs/upstream_1km_fixed_far --host 127.0.0.1 --port 6007
```
