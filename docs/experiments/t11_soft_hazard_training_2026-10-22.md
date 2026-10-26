# 暂停预测足迹拦截，使用地形cost和奖励的正式训练

2026-10-22停止边界：用户已要求切换新感知／地形。本文原run在502,369学习底层步／第52批／260轮PPO处正常保存退出，`update_000052.pt`及四owner分片已校验，原PID1762613及owner全部退出。全新策略训练使用剩余1,497,631步，见[新感知正式训练](t11_obstacle_input_training_2026-10-22.md)；下方启动和“继续运行”叙述为该run历史阶段。

## 授权、目标与边界

用户明确要求暂时直接去掉主动足迹复核，调整奖励和MPPI cost鼓励远离坑洼/岩石，优化计算后启动训练。本轮在现有规划、执行、奖励和训练入口内调整，不新增外部行为模块。目标是检验较少主动干预时能否学会完整月面集合；启动只要求真实样本有效、计费正确、日志/恢复可用，不要求未训练策略成功。

采用A1/seed23，全新同初始化Actor/Critic和Adam，16环境（4 owner×4）、每owner双线程、MPPI512×2、32决策/批和5轮PPO；5s策略决策、1s重规划、0.2s底层控制。1km真实月面、R200–300m出生、1500s完整回合、无课程，学习上限200万实际底层步。独立完整评估保留原初始/50万间隔/最终日程，单独预算40万步（计划最坏36万）。旧run的已消耗步数保留在旧停止记录，不导入其checkpoint。

## 代码与配置

- 实验配置：`configs/experiment/upstream_1km_soft_hazard_v1.yaml`。
- 训练配置：`configs/training/upstream_1km_far_soft_hazard_v1.yaml`。
- 显式`hazard_response_version=cost_reward_only_v1`，进入运行配置、缓存/恢复身份；旧默认仍`strict_footprint_v1`。
- 删除此候选路径中的候选完整轨迹足迹复核、信息时刻停车储备复核、每步足迹/CBF投影与停车准入、预测安全轨迹广播计算。`projection_disabled`在训练入口的runtime override后仍保持关闭。
- MPPI既有车中心地形/未知区、邻车、工作区、速度约束继续筛选候选。加权均值必须通过已计算的prescreen；无有效候选时制动，不沿用没有当前prescreen的旧计划/截短前缀。没有增加额外查询或重新启用完整足迹门控。
- 保留当前实际姿态的足迹测量，以产生停车条件、奖励和地形失败终端；碰撞/边界/车辆运动限幅、合法感知通信保持。实际失败是学习结果，不再全部提前拦住。这里仍为Proxy实验，不是PhysX验收。
- `execution_verified`、`execution_review_passed`、`full_review_passed`保持false，显式标记预测复核未执行；planner valid只表示候选可输出，不能解释为完整几何证明。
- 计算精简：删除预测轨迹/停车储备的高频CPU检查；实际单姿态报告和感知更新复用现有owner线程与缓存；MPPI新增风险直接复用已有地形查询结果，不引入接触网格或扫掠查询。

## 奖励与cost

集合势差、持续距离惩罚、驻留势、成功奖励130、超时惩罚80、通信/观测与折扣时间单位不变。

| 参数 | 原值 | 本轮 |
| --- | ---: | ---: |
| MPPI `terrain_scale` | 1 | 3 |
| reward `terrain_weight` | 0.1 | 0.5 |
| reward `hazard_cost` | 0（关闭） | 0.05 |
| reward `failure_penalty` | 55 | 100 |

MPPI原坡度/可通行性风险增加粗糙度与沿轨迹相邻高度差平方项，分别除以现有粗糙度/台阶阈值，幅值截到10后平方；再按地形权重计费。高层奖励增加实际执行位置的归一化坡度、粗糙度、可通行性损失和逐步高度变化平方风险，四车平均，乘0.05扣除。两者均来自已有局部感知，不给Actor全局地形真值。它们提供倾向，不等价于完整车身避障保证，尚无收敛证据。

奖励配置由`configured_reward_settings`供采样、审计和训练入口共用。旧配置无override且hazard_cost为0时保持旧计算/元数据；新reward fingerprint阻止跨语义恢复。

## 最小验证与证据

启动诊断总上限1200实际月面步，0 PPO：四角色检查最多400步，16环境两区间最多800步。无需继续重复旧完整安全验收。

`outputs/runs/upstream_1km_fixed_far/soft_hazard_prelaunch_20261022_01/`：首轮73项检查、输出选择补测81项，去重108项全部通过、0跳过。覆盖禁止预测复核调用、真实失败仍可学习、粗糙cost/奖励方向、CPU/CUDA现有查询复用、被prescreen拒绝的均值/旧轨迹不能执行、新旧恢复身份隔离及原协议回归。见`metrics/checks.xml`、`selection_checks.xml`及`logs/`。

真实四角色初步检查消耗86步（25/25/11/25），0更新。固定评估初态在第11步出现terrain_failure，回合回报−12.57，作为正常有限失败接入学习；未用成功率作准入门槛。见`metrics/summary.json`与`static_audit.json`。输出选择补齐之后的最终实际路径另由16环境probe验证，最终readiness组合这些行为相关证据，不把早期检查冒充最终同状态轨迹验收。

16环境吞吐run：`outputs/runs/upstream_1km_fixed_far/soft_hazard_owner16_probe_20261022_01/`。完成662／800实际底层步、32有效团队转移、0 PPO更新、0独立评估步。采样48.93s（13.53步/s），含日志归档49.17s（13.46步/s），初始化52.36s单列；整卡显存采样峰值1354MiB，可用主存最低7.36GiB，进程SwapPss为0。GPU平均23.9%，CPU平均4.48核。两区间353/309步出现6/7个结束回合，全部有效；退出、531项源码/配置SHA、初末网络一致与TensorBoard横轴353/662已核验。对照完整拦截版本5.16步/s，本轮约2.62倍吞吐，但轨迹、失败/重置和cost均改变，不是等价计算优化的严格加速比。按短测纯采样线性外推200万步约41.1小时，PPO/评估/长期状态变化另计。

整个启动诊断实际748／1200月面底层步、0优化更新。`metrics/readiness.json`组合初步地图/奖励审计、108项不同检查和输出选择补齐后的最终16环境采样，作为正式入口准入证据；旧summary、日志和源码包不覆盖。

## 正式启动

正式run：`outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_soft_hazard_2m_20261022_01/`；主进程PID1762613，独立session后台启动，不依赖终端存活。命令与环境变量保存在`config/command.txt`、`config/launch.json`，配置/源码归档`config/source_and_configs.tar.gz`，准入证据`metrics/prelaunch_evidence.json`，控制台`logs/console.log`链接到启动诊断run中的原始日志。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 \
.venv_terrain_cuda/bin/python -u scripts/run_upstream_t11.py \
  --config configs/experiment/upstream_1km_soft_hazard_v1.yaml \
  --training-config configs/training/upstream_1km_far_soft_hazard_v1.yaml \
  --run-id a1_seed23_far_lunar_soft_hazard_2m_20261022_01 \
  --representation A1 --mode train --seed 23 --budget 2000000 \
  --num-envs 16 --env-owners 4 --owner-cpu-workers 2 --sampling-protocol owners16_v1 \
  --trainer-backend skrl --policy-device cpu --samples 512 \
  --readiness outputs/runs/upstream_1km_fixed_far/soft_hazard_prelaunch_20261022_01/metrics/readiness.json \
  --release-training-pause
```

TensorBoard实时目录为正式run的`tensorboard/`，横轴为实际学习底层步，独立评估另计。现有`http://127.0.0.1:6006`加载far族；`tensorboard/pinned_url.txt`为成功率优先的精选视图。单独回放可用：

```bash
.venv_terrain_cuda/bin/tensorboard --logdir outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_soft_hazard_2m_20261022_01/tensorboard --host 127.0.0.1 --port 6007
```

训练先保存初始完整checkpoint，随后并发独立冻结策略评估与学习采样；首两批保存、之后每5批及评估/结束边界保存。数值或样本有效性错误按现有入口保存失败证据并退出，不自动重试；成功率低作为实验结果，不追加预算。启动复核通过，见`metrics/launch_verification.json`：已落盘662学习底层步、首批仍在采样、0次PPO更新；独立初始化随机评估far面板已85步（单独计账）。完整初始checkpoint与四分片SHA核验通过，248项运行源码身份一致、初始网络匹配既有同种子初始化，全新Adam。TensorBoard吞吐横轴353/662与采样JSON一致，6006服务已显示新run，未发现failure产物。复核后训练继续后台运行；这些是启动证据，不是收敛结论。

## 采样流程与耗时诊断（2026-10-22）

用户询问当前吞吐瓶颈与完整采样步的各阶段耗时。本轮只诊断，不修改运行源码、算法或配置，不停止／重启正式训练。最小出口为正式吞吐、完整调用链和互不重复的阶段计时；诊断预算最多100实际底层环境步、0优化更新，取得结果后结束。

### 正式运行的长期口径

第29–33批合计46,547学习底层步，采样及归档5,944.22 s，两个训练报告之间总墙钟5,961.85 s：采样吞吐 **7.83步/s**，含更新等全部开销 **7.81步/s**。平均每批32次向量策略决策耗时1,188.84 s（19.81分钟），每次16环境向量决策37.15 s；采样及归档占99.70%，PPO／检查点／回收等其余开销合计17.63 s。该统计使用剖析启动前已完成的批次，不受本轮额外诊断进程影响。启动短测的13.53步/s不代表长期吞吐；若按本窗口速率线性计算，200万步纯采样约70.95小时，第33批326,094步后的剩余纯采样约59.38小时。

这里一个底层环境步是**一个四车团队执行0.2 s控制**，不按车辆重复计数。一个策略决策保持动作5 s，最多25个底层步／环境；16环境最多400步，提前终止只计实际执行的步数。每个owner拥有4环境，四个owner同步完成后中央进程继续，不能把四进程墙钟相加。

正式日志没有单独的PPO计时器。以归档sidecar与更新JSON的mtime估计，第29–33批PPO加报告约2.66 s／批，最后一个采样chunk到归档完成约1.75 s／批；二者是文件时间边界近似，不冒充直接函数计时。

### 完整调用链

中央`ProcessUpstreamRollout.collect`先向四owner发送`prepare`：异步重置已结束环境、续签合法运动界、构造Actor／Critic观测；汇总后中央Actor／GRU采样一次原始动作，再向owner发送`step`。owner调用`step_decision`，5 s内续签固定目标并执行最多五个1 s规划区间；每区间通过`step_intents`运行最多五个`tick`。

每个`tick`依次激活到时请求／调用MPPI，读取当前足迹及地形特征，生成跟踪控制并限幅，执行NumPy运动代理，重新密集感知，在线程池中更新私有地形记忆和测量真实姿态足迹，再广播通信、判断失败／停车／驻留并计算奖励。每个1 s区间末尾还构造一次观测，所以本次5 s决策共6次观测构造。soft-hazard模式关闭预测足迹拦截，**当前实际姿态足迹测量仍然执行**。该路径是Proxy运动模型，不是Isaac／PhysX步进。

owner返回终止前观测、奖励、实际duration与执行记录；中央计算value／next_value、更新GRU状态并汇总转移。积累32次向量决策后归档交互，再做5轮PPO和按日程保存checkpoint。TensorBoard／JSON在每次向量决策后更新；独立评估单独计费。

### 隔离owner的直接阶段计时

证据run为`outputs/runs/upstream_1km_fixed_far/sampling_profile_20261022_01/`，脚本`profile_sampling.py`，命令在`config/command.txt`。从正式`update_000030.pt`及已校验SHA的`owner_000.pt`导入策略、GRU切片和环境；保持4环境、双线程、512样本×2轮、月面与有效训练配置。它是checkpoint派生的单owner诊断，未重放原16环境完整动作采样顺序；并发正式训练仍占用CPU／GPU，不能作为新的16环境吞吐基准。

初始化和状态导入51.67 s单列；实际采样 **34.6925 s，84／100底层步，4条有效转移，0优化更新**，四环境duration为14／25／25／20。开始时四个环境均已终止，因此包含一次四环境重置。以下为互不重叠的主线程墙钟分组，合计100%；两工作线程的内部计时不再重复加到表内。

| 流程 | 本次耗时 | 占采样墙钟 |
| --- | ---: | ---: |
| 重置、合法出生筛选与初始感知 | 16.625 s | 47.92% |
| Actor／Critic观测构造，共6次 | 0.304 s | 0.88% |
| Actor采样与两次价值前向 | 0.0178 s | 0.05% |
| 请求激活、规划输入与5批MPPI | 5.188 s | 14.95% |
| 25次底层密集感知 | 5.313 s | 15.32% |
| 并行记忆／当前足迹／中心特征更新及等待 | 6.284 s | 18.11% |
| 跟踪、限幅、通信快照及tick调度 | 0.401 s | 1.16% |
| 实际运动积分及执行记录 | 0.106 s | 0.30% |
| 当前足迹报告读取／缓存缺失补测 | 0.170 s | 0.49% |
| 每步广播 | 0.123 s | 0.35% |
| 任务终止、停车驻留与奖励 | 0.0214 s | 0.06% |
| 请求／租约／转移汇总等其余部分 | 0.140 s | 0.40% |

重置内部193次候选团队感知耗时10.261 s；295次私有记忆构建3.592 s；295次候选车身足迹测量2.362 s，已包含在首行16.625 s内。193组候选只接受4组，189组拒绝，平均48.25次候选／接受团队。`SpawnCurriculum._legal_birth`先扫描候选四车全部密集视场，再逐车建记忆和检查足迹；首车失败也已经支付四车感知成本。每车一次151×81＝12,231个采样点，仅候选出生就扫描约944万点。

在线更新的两个工作线程合计：记忆更新10.605 s，其中native合并7.193 s、构建新只读视图／空间树约3.002 s；当前足迹测量1.196 s。它们在两线程间重叠，包含在主表6.284 s等待内；reset初始刷新另有线程工作，不能把全部worker总时间误当该行的墙钟。

5次MPPI kernel调用总墙钟4.992 s，外层输入校验／激活等后为主表5.188 s。既有诊断记录的索引打包约0.941 s、轨迹预测CUDA events约2.364 s、代价约0.130 s、重要性更新约0.0218 s、候选输出／轨迹评分约0.943 s。CPU墙钟与CUDA事件有重叠，且CUDA事件可能包含CPU供给间隙，不能求和当作纯GPU执行时间；字段`full_review_and_scoring_s`沿用历史名称，本模式没有重新开启完整预测足迹拦截。

### 判断与下一项

当前优先级是**合法出生筛选 → 每步密集感知／地形记忆 → MPPI组织与预测**。前两项在本面板合计81.35%；网络前向约0.05%，真实运动积分约0.30%，先优化PPO或更换Actor设备不能解决主要吞吐问题。正式第29–33批2,560条团队转移中，1,459条结束回合（56.99%），全部为terrain_failure，结束回合平均31.99底层步；频繁失败使昂贵重置持续进入热路径。四环境全部重置面板的47.92%不能直接当作16环境长期精确占比。

下一项可在现有出生检查中减少被拒候选的全视场计算／重复建图，保持候选次序、接受判据和合法出生分布；其次优化现有感知、记忆合并和索引构建。当前没有实施这些候选，也没有减少感知频率、MPPI样本／迭代或改变任务语义。

进程资源短窗口记录约2.46 GiB训练进程SwapPss，但主缺页计数未增长，不能把驻留swap直接认作当前主瓶颈。细节在`metrics/live_resources.json`。

原始分层计时：`metrics/profile.json`；可相加分组：`metrics/stage_summary.json`；正式批次统计：`metrics/live_summary.json`；重置频率：`metrics/reset_frequency.json`；源码与events核验：`metrics/verification.json`。诊断脚本首次漏接策略的value-normalization模块，恢复拒绝发生在推进前；修正前的初始化中止同样0步，日志保留在`logs/`。最终未修改任何训练源码，诊断进程已退出，正式训练继续原预算。

诊断events已从0增长至84实际步，并保存在该run的`tensorboard/`；现有服务`http://127.0.0.1:6006`加载far族。独立查看命令：

```bash
.venv_terrain_cuda/bin/tensorboard --logdir outputs/runs/upstream_1km_fixed_far/sampling_profile_20261022_01/tensorboard --host 127.0.0.1 --port 6007
```

## 出生筛选计算优化（2026-10-22）

用户授权开始减少被拒出生候选的无效计算。目标是保持原候选次序、合法性判据和出生分布，减少拒绝前支付的感知与建图开销。最小验收为同候选完整报告、同checkpoint出生／RNG／计数一致，受影响契约通过并记录耗时；本轮限定两图各256组固定候选，另重放第30批owner0的四次出生请求，不推进环境或优化器。证据run为`outputs/runs/upstream_1km_fixed_far/spawn_optimization_20261022_01/`。

### 实现

- `upstream_curriculum.py`逐车检查，首个不合格车辆立即拒绝本候选，后续车辆不再感知。
- `upstream_sensing.py`在绑定时从原151×81密集网格预计算足迹支持点及其原3×3粗糙度邻域。每辆被检查的车只查询713点，输出609个原足迹支持点，替代完整12,231点扫描；不另造粗网格，也不放宽支持半径。支持点裁取的1e-7余量只多查点，实际支持判定仍用原半径。
- 原直接测量、缺测掩码和去平面粗糙度提取成共享函数；临时出生地形直接构造只读`FootprintTerrain`，省去空历史的完整`DenseTerrainMemory`合并。接受出生后的正常reset仍做完整密集感知及私有记忆初始化。

候选几何、车序、随机抽样、阈值、最大尝试次数及checkpoint状态字段保持。此次计算优化不新建行为版本；旧实现及正式run启动源码仍可追溯，既有源码身份guard继续生效。

### 同输入性能与等价性

使用实际训练地图0／1、seed23／27，CPU单线程，与正式训练并发。两实现均预热后各执行同一候选面板，原实现从修改前归档加载；下表是静态出生筛选墙钟，不包含环境初始刷新或策略执行。

| 面板 | 候选组／接受组 | 修改前 | 修改后 | 加速 |
| --- | ---: | ---: | ---: | ---: |
| 训练地图0，固定候选 | 256／3 | 17.786 s | 3.781 s | 4.70倍 |
| 训练地图1，固定候选 | 256／3 | 20.296 s | 4.852 s | 4.18倍 |
| 正式第30批owner0出生RNG重放 | 193／4 | 17.888 s | 3.044 s | 5.88倍 |

512组固定候选内，每个实际检查车辆的完整`FootprintReport`字典逐值一致（包含失败原因和各数值），没有使用数值容差忽略差异；两图合计查询点从25,049,088降至606,050（−97.58%）。逐车查询次数由每候选一次四车合批，变为411／439次单车小查询，但总计算明显降低。

checkpoint重放从`update_000030.owners/owner_000.pt`恢复原出生状态：均193次尝试、189次拒绝，四组接受位置／朝向、结束RNG完全一致。该项是相同出生筛选计算的对照，不是再次执行先前84步策略采样。另用两张图各3组完整四车帧、非零odom旋转核对普通`DenseSensor.sample`：24帧的坐标、特征、valid、roughness_valid和sensor_pose逐值一致。

55项不同受影响检查通过、0跳过：新增23项覆盖坡度阈值、坑石／粗糙度、未知支持、旋转、首车失败短路、出生顺序／RNG／计数／异步恢复；复用同次修改后原感知／固定采样32项通过结果。首两次新增测试因fixture遗漏风险尺度、None报告比较及未初始化全部环境而失败，修正测试后23项通过；原失败日志/XML保留，不将失败项计为通过。最终证据见`metrics/verification.json`、`checks.xml`中原32项与`birth_checks_final.xml`中新增23项。

### 命令、证据和应用边界

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv_terrain_cuda/bin/python outputs/runs/upstream_1km_fixed_far/spawn_optimization_20261022_01/benchmark_birth.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv_terrain_cuda/bin/python outputs/runs/upstream_1km_fixed_far/spawn_optimization_20261022_01/verify_results.py
```

实际执行命令列表另存`config/commands.txt`。修改前两文件为`config/before.tar.gz`及`before_sha256.json`，最终源码／配置为`config/source_and_configs.tar.gz`及对应SHA；固定候选为`state/candidates_map*.npy`。完整报告与计时分别在`metrics/candidate_panel.json`、`summary.json`，核验在`verification.json`；原始控制台在`logs/`。

新增实际环境步 **0**，优化更新 **0**，未重启正式训练，也未增加其预算；PID1762613及四owner仍使用启动时已加载的旧代码。新实现需在后续进程切换时应用并保护checkpoint来源，不通过改写旧身份绕过恢复guard。5.88倍只描述该出生面板，不能直接宣称16环境整体训练提速相同倍数；下一项应用后按实际采样吞吐判断剩余收益。

静态检查的events已保存并增长至512候选组，横轴明确为候选数量，环境／优化步保持0；现有`http://127.0.0.1:6006`加载far族，独立查看：

```bash
.venv_terrain_cuda/bin/tensorboard --logdir outputs/runs/upstream_1km_fixed_far/spawn_optimization_20261022_01/tensorboard --host 127.0.0.1 --port 6007
```
