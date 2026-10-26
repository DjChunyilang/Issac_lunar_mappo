# 2026-10-22：采样计算等价优化实施与配对验证

本轮按[采样瓶颈诊断](t11_obstacle_sampling_profile_2026-10-22.md)后的用户实施计划完成六组优化。**候选位于独立源码副本，正式训练仍运行原源码；没有切换训练、承接新源码状态或扩大训练预算。**

## 1. 结果与交付

| 同状态配对面板 | 每组实际底层步 | 基线采样时间 | 候选采样时间 | 吞吐变化 |
| --- | ---: | ---: | ---: | ---: |
| 单 owner，4 环境 × 2 决策 | 200 | 50.278 s | 24.645 s | 3.978 → 8.115 步/s，**2.040×** |
| 四 owner，16 环境 × 1 决策 | 389 | 43.851 s | 21.857 s | 8.871 → 17.797 步/s，**2.006×** |

四 owner 的平均 owner 阶段耗时：MPPI **下降63.73%**，感知／真实足迹 **下降27.31%**；整体采样吞吐 **提高100.62%**，达到本轮三个目标（−30%、−25%、+20%）。保留全部六组改动及已有双worker调度。

13组学习张量、动作样本、执行轨迹／判据、最终位姿、GRU和torch RNG通过配对检查。最终诊断cost分量存在末位差异，按既有FP64契约通过，详见第6节。

这是正式训练持续运行、带性能插桩的短面板，不能直接当作正式训练切换后的长期吞吐。初始化／恢复时间另列，不计入上述采样时间。

产物根目录（下文记为 `R`）：

```text
outputs/runs/local_obstacle_input/sampling_optimization_20261022_01/
```

- 实现：[候选源码](../../outputs/runs/local_obstacle_input/sampling_optimization_20261022_01/source/candidate/)，对应冻结基线在 `source/baseline/`。
- 交付：[10个生产文件的补丁](../../outputs/runs/local_obstacle_input/sampling_optimization_20261022_01/config/candidate.patch)、[基线＋候选源码包](../../outputs/runs/local_obstacle_input/sampling_optimization_20261022_01/config/isolated_sources.tar.gz)。补丁已对冻结基线执行 `patch --dry-run`；源码包另含4个新增测试。
- 结果：[汇总JSON](../../outputs/runs/local_obstacle_input/sampling_optimization_20261022_01/metrics/summary.json)、`metrics/single_comparison.json`、`metrics/multi_comparison.json`、`run_manifest.json`。
- 源码身份：`config/source_before.json`、`config/candidate_source_sha256.json`，以及每个arm的 `metrics/identity.json`。基线包含当时未提交／未跟踪的现有内容，不是仅从Git HEAD生成。

## 2. 已实施的内部改动

全部生产文件均位于 `source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/local_goal/`。

| 顺序 | 文件 | 最终实现 |
| --- | --- | --- |
| 1. CUDA格查询 | `upstream_grid.py`、`upstream_mppi.py`、`upstream_mppi_calibration.py` | 不可变tile池＋逐车索引，缓存origin／尺寸／覆盖点／距离参数；`select`共享数据；一次读取features／known／ground；障碍查询只取distance通道，按**全批最多4096姿态／块**展开54点 |
| 2. 循环同步 | `upstream_execution.py`、`upstream_native.py`、`upstream_mppi_precision.py` | 公开接口保留校验；grid CUDA预测内部复用原算术及投影核，设备端累计物理非法标记，完整rollout末尾统一抛错；raw NaN／Inf仍按原numeric掩码处理 |
| 3. 最终评分 | `upstream_mppi.py`、`upstream_planning.py` | 先按原顺序选中并构造实际NumPy执行轨迹，再合批调用原V4 cost；所有行有限性验证后提交warm／previous／time及诊断；HOLD、混合、timeout和单体入口保持契约 |
| 4. 物理视图与格记忆 | `upstream_sensing.py`、`upstream_environment.py`、`upstream_grid.py` | 抽取测量数组，最终坐标下只建一次物理视图；0.8 m粗格先筛、展开16子格后原边界过滤和排序；精确静止复用；EDT按占用依赖失效 |
| 5. 双worker刷新 | `upstream_environment.py` | 主线程测量及传感器计数；独立线程reviewer处理逐车格记忆、足迹和中心查询；固定顺序收集并全部完成后广播 |
| 6. 观测边界 | `upstream_environment.py`、`upstream_decision.py` | 分开内部步进与公开转移封装；正常5 s决策只保留开始／结束观测；提前终止仍在原1 s观测边界冻结对应Actor／Critic行 |

保留车辆模型、K512×2、40步预测、0.2 s执行、1 s重规划、每车独立RNG、FP32原始噪声及FP64执行／积分。观测尺寸、动作、奖励、检查阈值和checkpoint行为语义不变；没有新增行为版本、配置或融合CUDA核，也未绕过正式源码恢复检查。

感知仍是713个原始地形位置、最多609个中心点，真实足迹仍查询2310个车体／接触位置，保留完整3×3邻域和原nearest／tie顺序。12,231点旧完整视场采样在两组面板中均为0次。

## 3. 完整采样阶段

单owner的两次决策，下面各行互斥，合计为完整采样墙钟；重置的初始感知包含在重置行。

| 阶段 | 基线 | 候选 | 候选相对下降 |
| --- | ---: | ---: | ---: |
| MPPI请求激活与规划 | 28.846 s | 9.881 s | 65.75% |
| 正常感知／真实足迹刷新 | 12.567 s | 8.627 s | 31.35% |
| Actor／Critic观测 | 4.287 s | 1.164 s | 72.85% |
| 重置／出生／初始感知 | 1.496 s | 1.485 s | 0.73% |
| 其他执行／通信／奖励／传输／组织 | 3.082 s | 3.488 s | 增加13.17% |
| **总计** | **50.278 s** | **24.645 s** | **50.98%** |

四owner各owner并行工作，**不能把其耗时相加当作采样墙钟**：

| owner | 实际步 | 基线prepare服务 | 候选prepare服务 | 基线step服务 | 候选step服务 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 1.700 s | 2.056 s | 36.322 s | 18.654 s |
| 1 | 100 | 0.482 s | 0.691 s | 41.557 s | 17.788 s |
| 2 | 100 | 0.077 s | 0.076 s | **42.098 s** | **19.685 s** |
| 3 | 89 | 0.047 s | 0.085 s | 33.293 s | 18.310 s |

两组step最慢均为owner 2；中央进程还须先等待prepare最慢owner完成，再广播step。候选owner 3含提前终止，保留3次观测；其他候选owner为2次，基线均6次。四owner阶段平均值为MPPI `25.088 → 9.099 s`、刷新 `8.300 → 6.034 s`、观测 `2.760 → 0.954 s`。

## 4. 组件证据与剩余热点

### MPPI及格查询

零环境步组件用固定合成已见空地、B16、K512×2、40步及同噪声；每组一次暖机、两次不带profiler计时，另一次trace。以下GPU标量同步由trace `External id`关联到CUDA runtime实测，区别于此前源码推导：

| 组件／事件 | 基线 | 候选 |
| --- | ---: | ---: |
| 完整plan_batch墙钟中位 | 3.559 s | 0.963 s |
| 完整plan_batch CUDA事件区间 | 3.560 s | 0.963 s |
| 单次rollout墙钟中位 | 0.994 s | 0.104 s |
| 完整规划触发CUDA同步的标量读取 | 1,363 | 7 |
| 单次rollout触发CUDA同步的标量读取 | 440 | 1 |
| 完整规划 `cudaStreamSynchronize` | 4,212 | 178 |
| 完整规划CUDA kernel launch | 68,079 | 17,719 |

候选完整规划的总 `aten::_local_scalar_dense` 是151，其中144为CPU读取，不能全部称为GPU同步。CUDA事件区间包含主机供给与并发调度间隔，不等于纯kernel计算时间，也不与CPU墙钟相加。292组保存数组逐值一致，5组rollout数组hash一致。证据：`metrics/mppi_component_comparison.json`、`component_mppi/{baseline,candidate}/`。

独立格查询重放中，16车中心lookup中位约 `11.545 → 0.860 ms`；完整 `16×512×40` 障碍距离查询约 `126.83 → 116.34 ms`。后者仅约1.09×，54点覆盖读取仍是剩余开销。证据：`metrics/grid_component.json`。

16车格打包冷构造／原tile复用分别测3次：冷构造中位 `149.80 → 98.78 ms`，复用 `6.36 → 6.84 ms`。两组冷上传都为11,427,840 bytes，复用上传都为0。实际单owner10个规划批次合计上传 **43,941,888 bytes，两组相同**；四owner为101,634,048 bytes，两组也相同。不能把通道裁剪解释为H2D总上传量下降。证据：`metrics/grid_pack_component.json`及各arm的profile。

### 感知与真实足迹

单owner正常执行的50次向量刷新中，物理视图800次调用含521次新测量、279次缓存命中；新测量数两组一致。

| 工作 | 基线 | 候选 | 计量说明 |
| --- | ---: | ---: | --- |
| 刷新关键路径墙钟 | 12.567 s | 8.627 s | 主线程总墙钟 |
| 物理视图获取 | 1.903 s | 1.645 s | 主线程，包含缓存命中 |
| GridMemory线程CPU | 6.254 s | 5.750 s | 候选为worker累计CPU |
| 新姿态足迹报告线程CPU | 4.071 s | 5.360 s | 并行缩短墙钟，但此项CPU工作未下降 |
| EDT次数 | 4,218 | 2,368 | 已含于GridMemory |
| EDT线程CPU | 2.156 s | 1.676 s | 不另加到GridMemory |

候选worker的记忆／足迹累计wall分别6.866／6.022 s，彼此重叠，不能再加到8.627 s。按521次新测量及已核验构造路径推导，正常执行KD-tree构造由1,042次降至521次；这是路径推导，不是全进程所有树的直接计数。

环境传感器新增查询两组均为 `537×713 = 382,881` 点（包含prepare初始感知）；出生筛选独立传感器另113次测量。逐车frames、visible/new/atlas cells、eviction等统计一致，耗时字段单独排除。

固定32帧×4车组件交错重放中，基线串行1.669 s、候选串行1.460 s、候选双worker1.105 s；后两者输出SHA一致。双worker独立降低墙钟24.33%，本轮保留，不增加owner或线程数。证据：`metrics/perception_component_benchmark_02.json`。

候选单owner中MPPI约占40.1%、刷新约占35.0%；下一轮若继续优化，重点仍是障碍覆盖点查询、格打包／记忆组织、真实足迹近邻及tie处理。观测编码已降至约4.7%。本轮已有证据满足结束条件，不追加环境诊断或融合核实现。

## 5. 资源与测量边界

四owner采样窗口内约每2 s记录一次，资源表为**诊断进程树**；GPU数值为整卡，包含持续运行的正式训练。

| 项目 | 基线 | 候选 |
| --- | ---: | ---: |
| 采样窗口资源样本数 | 18 | 9 |
| 峰值PSS | 5.935 GiB | 6.177 GiB |
| 平均CPU使用（单核=1） | 3.718 | 4.404 |
| 诊断进程树采样swap峰值 | 0 | 0 |
| 系统可用内存最低采样值 | 4.656 GiB | 4.607 GiB |
| 整卡显存最高采样值 | 2,208 MiB | 2,206 MiB |
| 整卡温度最高采样值 | 86°C | 86°C |

候选增加约0.24 GiB PSS，CPU利用增加；整卡已较热且正式训练并发，因此这些单次短测不构成稳定容量或热稳态结论。

四owner初始化／恢复为 `69.49 → 132.31 s`，单owner为 `46.84 → 46.28 s`。初始化包含场景构建、出生、库加载和checkpoint恢复，发生在计时采样前；记录这次四owner启动较慢的事实，不把短采样收益推广到启动速度。

单owner基线资源线程在开始时因缺少psutil退出；该组采样和TensorBoard完整完成，资源数据标为缺失，未重跑消耗预算。随后监控直接读取 `/proc`，其余三组有完整资源记录。原错误日志保留。profiler对CPU阶段的插桩开销及与正式训练竞争均存在，不能拿本表绝对吞吐直接替换正式日志的批次吞吐。

## 6. 正确性、计费与失败记录

**104项独立检查通过，无跳过／未修复失败**：格查询7项、MPPI46项、感知38项、宏步观测13项。源代码与测试14个改动文件均通过编译检查，补丁可应用到冻结基线。

覆盖私有anchor、负坐标／边界epsilon、缺tile／未知、车辆重排和稀疏选择、旧快照及淘汰；MPPI同噪声／RNG、低速切换、正反限幅、HOLD、无效输入和评分异常状态不发布；测量数组／可见格顺序／距离／完整报告；不同地形交错线程；宏步第1、5、6、24、25 tick终止与无效环境的观测、next_value、GAE、GRU、duration和后继reset。

单／四owner真实面板分别有10／24项最终 `planner_cost_parts` 末位差异，最大绝对误差 `1.388e-17 / 2.776e-17`；仅这些最终cost浮点分量采用原 `abs=1e-10, rel=1e-8` 容差，其他比较保持逐值。出处为 `tests/test_upstream_mppi.py::test_score_uses_exact_reviewed_trajectory_without_float32_reprojection`。原严格比较保存于 `metrics/single_exact_comparison.json`，修订比较也保留全部原始差值。

真实面板保存13学习张量、完整samples／execution、最终hidden／poses／torch RNG，没有直接归档所有owner内部末状态或完整terminal_actor／terminal_critic；终止快照逐值等价由7个配对场景测试提供。不能把真实面板描述为所有内部状态逐位相等。

| 消耗 | 实际步 | 上限／说明 |
| --- | ---: | --- |
| 单owner基线＋候选 | 400 | 400 |
| 四owner基线＋候选 | 778 | 800；两组同一环境在第14 tick终止 |
| **月面性能面板合计** | **1,178** | **1,200** |
| 合成Proxy契约测试 | 838 | 新观测配对776＋既有回归62，单独记账 |
| 静态组件／trace／出生查询 | 0 | 不调用环境tick |
| **PPO更新** | **0** | 本轮没有优化器更新 |

XML历史共124次执行，116次通过、8次早期失败。感知改造前3次失败用于复现旧路径；MPPI新增夹具5次违反输入契约，修正后通过；原日志均保留，最终104个不同测试均有通过证据。详见 `metrics/test_summary.json`、`metrics/observation_summary.json`、`logs/`。未因性能目标达成而删掉失败或额外消耗未用完的22步。

## 7. 复现与运行边界

Checkpoint为正式run的 `update_000010.pt`；单owner恢复owner 0，四owner恢复全部四分片，校验header及分片SHA，并恢复策略、归一化、GRU、规划器和RNG。每组仅执行 `collect`，不调用PPO。四组实际命令和cwd见 `config/executed_commands.json`，执行工具为 `profile_pair.py`；对比工具为 `compare_pair.py`。新增测试依赖本次同级冻结baseline，应连同源码包保留。

TensorBoard复用 **http://127.0.0.1:6007**，正式训练仍为 **http://127.0.0.1:6006**；两个服务均核验HTTP 200。

```text
R/single_baseline/tensorboard/    0 → 100 → 200
R/single_candidate/tensorboard/   0 → 100 → 200
R/multi_baseline/tensorboard/     0 → 389
R/multi_candidate/tensorboard/    0 → 389
R/component_mppi/{baseline,candidate}/tensorboard/  0步静态组件
R/component_grid/tensorboard/                       0步静态组件
```

横轴是本诊断arm实际底层步，events已确认随分块采样更新，保留回放。查看命令：

```bash
.venv_terrain_cuda/bin/tensorboard --logdir outputs/runs/local_obstacle_input --port 6007
```

现有服务存活时直接访问，不重复启动端口。`config/isolated_sources.tar.gz`保存源码而非地形／checkpoint大产物，运行仍依赖当前项目的数据与环境。`outputs/`不提交；生产补丁尚未应用到正式工作树。

本轮核对正式 `source/`、`scripts/`、`configs/`、`tests/` 均未改变。期间外部更新的 `AGENTS.md`保留；只新增本记录并更新相关文档入口。正式训练源码切换及跨源码状态承接是后续独立操作，沿用原恢复检查及已有预算，本轮结束于已验证的优化候选。

2026-10-23后续：用户随后授权停止原训练并采用优化源码。原run在第23批／261,149步完整保存后退出，生产补丁已应用；中央与四owner完整状态逐值验证后，独立run从该边界恢复，原源码检查与总预算保持。实际启动、剩余预算及events证据见[优化正式续训记录](t11_optimized_training_resume_2026-10-23.md)。
