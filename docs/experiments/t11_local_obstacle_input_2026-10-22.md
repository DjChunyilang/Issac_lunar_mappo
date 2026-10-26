# T11 固定栅格、增量感知与障碍输入实施

日期：2026-10-22。任务来自用户确认的[改造规划](../architecture/local_obstacle_input_plan_2026-10-22.md)：保持车辆模型，减少12,231点完整视场重采样，并将坑石标签接入策略、规划和执行反馈。本记录的实施验收阶段只做契约检查和有界采样，未热切换原soft-hazard进程。其后用户已授权停止旧训练、使用剩余预算启动全新策略，见[正式切换记录](t11_obstacle_input_training_2026-10-22.md)；新观测不能直接继承旧checkpoint。

## 目标与最小出口

1. 地图静态属性可复用；各车只获得自己视场和历史中的信息；语义记忆不再合并浮点点集或构建KD-tree。
2. 坑石标签、小障碍的区域观测、MPPI软cost及实际执行奖励贯通；不新增障碍硬拦截。
3. 当前真实足迹、停车和失败判定继续使用原车辆几何及物理阈值；新测量路径和历史点记忆的差异写入环境身份。
4. 完成CPU/CUDA查询、恢复边界及有界采样；前端和整体吞吐分开记录。诊断总预算800底层环境步，0优化器更新。

## 配置及实现

新入口为`configs/experiment/upstream_1km_obstacle_input_v1.yaml`，继承现有soft-hazard实验；训练参数候选为`configs/training/upstream_1km_far_obstacle_input_v1.yaml`，保留far／1500 s／无课程及PPO设置。唯一新输入定义为`configs/terrain/upstream_1km_obstacle_input_v1.yaml`。旧配置、设计profile和checkpoint未覆盖。

| 部分 | 已实现行为 |
| --- | --- |
| 地图派生 | `terrain_data/obstacle_atlas.py`从原scene与accepted目录生成0.2 m标签、0.8 m普通地形代价；不再抽样坑石或修改高度文件 |
| 标签 | 坑口／岩石水平投影圆盘，与格子有任意交叠即标注；重叠类别按位合并，保留坑口以外的真实地面风险 |
| 普通地形 | 将0.1 m原生处理格按0.8 m块聚合；高度均值、最坏坡度向量、最大RMS、最小通行评分，去局部坡面的残差反映台阶；drive／park使用原10°／5°阈值 |
| 感知 | 仍为前后15 m、侧向±8 m，每0.2 s更新；只有完整0.8 m来源格进入视场，才揭示其0.2 m子格，避免粗格统计越过视场边界；边缘会暂留unknown |
| 私有记忆 | 32×32子格／tile、最多64 tile／65536 allocated cells；新揭示／淘汰后重读静态属性，静止内容不变时复用view；当前可见tile不淘汰 |
| 查询／CUDA | 坐标直接转换为格号和tile索引；tile按内容引用缓存，只上传改变的tile，部分车辆规划不会清空其他车辆缓存；旧的共享点服务不接受新栅格 |
| 障碍距离 | 从已观测占据格构建截断局部距离；不用全图距离场。用54个覆盖圆盘近似含0.3 m裕量的有向车身矩形，保留航向；外部距离偏保守，内部深度是软cost近似，不作为碰撞证书 |
| Actor／Critic | 112×3障碍接近／地面cost／known面积比例，Actor523／Critic1439；25目标区域采用同源聚合和停车cost，9个Oracle槽仍为0 |
| 区域聚合 | 对区域内已见风险取max，known按真实格子交叠面积计算；批量SAT筛选，只有混合已知区域的未知边界格需要多边形裁剪；静止同view／姿态复用结果 |
| MPPI | 原中心地形／unknown／邻车／工作区prescreen保留；新增near=3 cost/s、overlap=20 cost/s，乘0.2 s，对样本、均值和最终重评分使用同一函数 |
| 执行反馈 | 实际执行后姿态读取pre-step只读障碍快照；near=0.2 reward/s、overlap=2 reward/s，乘0.2 s、四车均值；新项单列，原物理奖励保留 |
| 真实地形 | 当前车身及3×3粗糙度邻域沿用出生优化的713查询位置／车；保留609支持点和原完整足迹判据，静止可复用；不维护整个视场的真实高度点历史 |
| 恢复 | 保存私有tile、地图身份、固定坐标、时间和内容revision；观测／cost／reward及派生atlas SHA进入身份，新旧checkpoint不能直接互读 |

圆盘覆盖的间距上限0.8 m，实际覆盖半径约0.5173 m，距离另计0.1414 m格子离散裕量。这是明确的保守近似，未改为外接圆车辆模型；其近障处罚范围与理想矩形距离不同。原地图和真实行驶／停车阈值保持，当前测量改为即时传感子集，不能把历史危险点合并规则的取消宣称为所有物理报告逐值等价。

## 数据与命令

实施证据：`outputs/runs/local_obstacle_input/implementation_20261022_01/`。其中`config/pre_integration_worktree.tar.gz`保留捕获时的已有工作区内容；测试XML／失败日志、聚合优化前源码、初始观测恢复及后续核验均保留。最终运行源码及SHA为`paired_probe_20261022_03/config/source.tar.gz`和`source_sha256.json`，`metrics/verification.json`核对其与交付源码一致。

四张atlas位于`outputs/runs/local_obstacle_input/atlas_20261022_01/<原scene_SHA>/obstacle.json`，manifest关联原scene、文件SHA、输入spec与物理阈值；每个派生数组也有SHA。原scene仍为`outputs/runs/lunar_terrain_validation/npb_1km_fixed_20261021_01/scenes/<role>/scene.json`。

| 原图 | 栅格占据比例 | 普通地形有效比例 |
| --- | ---: | ---: |
| train_0 | 5.832848% | 99.680256% |
| train_1 | 5.861952% | 99.680256% |
| fixed_eval | 5.837768% | 99.680256% |
| unseen | 5.845716% | 99.680256% |

以上为标签格覆盖比例，不能推断整个车身的无障碍概率或任务成功率。目录中只有高度尾部影响目标图、但坑口圆盘在图外的对象不写入标签；首次构建暴露的空裁剪缺陷已修复，失败目录及日志保留。

已执行构建：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv_terrain_cuda/bin/python scripts/build_local_obstacle_atlas.py \
  --scene outputs/runs/lunar_terrain_validation/npb_1km_fixed_20261021_01/scenes/train_0/scene.json \
  --scene outputs/runs/lunar_terrain_validation/npb_1km_fixed_20261021_01/scenes/train_1/scene.json \
  --scene outputs/runs/lunar_terrain_validation/npb_1km_fixed_20261021_01/scenes/fixed_eval/scene.json \
  --scene outputs/runs/lunar_terrain_validation/npb_1km_fixed_20261021_01/scenes/unseen/scene.json
```

已生成的immutable输出不能原地重建。构建新scene可用现有`build_lunar_training_maps.py --obstacle-experiment-config <新实验配置>`同时派生标签。

## 验证与性能

真实地图初始化检查为0环境步／0优化更新：Actor `[1,4,523]`、Critic `[1,1439]`，策略概率有限，环境保存／恢复后观测逐值一致。检查证据为`metrics/initial_inspection.json`。

固定姿态前端对照为0环境步：`frontend_benchmark_20261022_01/summary.json`，四车16帧，包含静止重复、平移和转向，old/new执行顺序交替。旧完整感知＋点记忆＋当前足迹3.6111 s，新增量格＋即时足迹1.7303 s，**2.087倍**；原后端查询782,784→37,076点，另读取49,344个语义子格。这个数不包含Actor区域编码或MPPI，不能代替整体吞吐。

实际采样使用`probe_upstream_obstacle_input.py`，1 owner×4环境、每owner双线程、CPU Actor、CUDA MPPI512×2。原16环境长训并发运行，采样结果只代表该条件。两组策略显式固定mean／hold头的权重，保留完整encoder／GRU前向、quarter std和p_hold=0.35；每区间相同Torch RNG产生可比较动作，不执行PPO。

- `paired_probe_20261022_01/baseline`完成175步／2区间，39.9773 s，约4.378步/s。新组在CUDA路径激活时失败，保留完整日志，未提交消耗保守上界100步。
- `paired_probe_20261022_02/grid`完成176步／2区间，94.5707 s，约1.861步/s。感知减少后，逐区域精确裁剪成为新瓶颈；本结果属于性能回退，不能当作优化交付。
- 聚合优化：同四车、五组相对位置／朝向、主区域和目标区域共8,220值，最大差异3.13e-13；7.1739→0.6629 s，**10.82倍**。使用批量几何筛选和unknown面积补集，未改为点取样或均值风险；证据`metrics/aggregation_equivalence_02.json`。
- `paired_probe_20261022_03/grid`最终复测176步／2区间，31.3008 s，**5.623步/s**；相对对照4.377步/s，按实际步数归一吞吐提高**28.45%**。两区间分别16.8461／14.4546 s，初始创建38.7305 s另列，不混入采样。初始出生姿态、两个区间的连续动作及HOLD逐值一致；控制、真实失败和终止步数允许随新cost／输入语义变化。

累计已提交诊断527底层步，加失败区间未提交消耗的保守上界100步，计费上界**627／800步**，0 PPO更新、0独立评估步；失败尝试不当作零消耗。最终probe源码归档与当前`source/scripts/configs/tests`逐文件SHA一致。

213项不同的受影响检查通过，其中15项新检查覆盖固定格可见性、粗格来源支持、未见障碍不泄漏、静止零读取、小障碍区域聚合、CPU/CUDA一致性、淘汰与快照恢复、完整MPPI流水线和执行奖励。较早失败均保留日志；通过集合来自`contracts_02.xml`、`integration_01.xml`、`contracts_04.xml`及最终`contracts_08.xml`，同名检查只计一次。实际地图初始化还验证新维度和保存／恢复。

原图核验检查四张原scene manifest、实际使用的原生features／valid及目录文件SHA，另核验全部派生数组；结果见`metrics/verification.json`。其他原始文件未写入，manifest引用保留。

CUDA集成缺陷包括`cuda`与`cuda:0`设备标识不一致，以及旧平地原生核误接新栅格；已修复并补充完整预测／cost／选取／重评分检查。失败日志保留，只有最终通过的检查计为通过。

已执行的性能命令（输出目录均需不存在，回放时使用新run名）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv_terrain_cuda/bin/python scripts/benchmark_upstream_perception.py \
  --initialization outputs/runs/local_obstacle_input/paired_probe_20261022_01/baseline/metrics/initialization.json \
  --output outputs/runs/local_obstacle_input/frontend_benchmark_20261022_01

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv_terrain_cuda/bin/python scripts/probe_upstream_obstacle_input.py \
  --run-id paired_probe_20261022_03 --owners 1 --envs-per-owner 4 \
  --decisions 2 --budget 200 --arms grid
```

新检查的复现命令为`.venv_terrain_cuda/bin/python -m pytest -q tests/test_upstream_obstacle_input.py`；旧路径检查按上述XML中的文件集合复用已通过结果。原长训并发、两区间和新cost语义限制了性能外推，28.45%是本次有界采样结果。

## TensorBoard与后续边界

服务：`http://127.0.0.1:6007`。日志位于各`paired_probe_*/<arm>/tensorboard/`及固定帧前端对照的`tensorboard/`；采样events横轴为实际底层环境步，静态对照明确使用frame横轴。JSON／PT记录计费、动作、奖励、cost和来源，events保留回放。已核验baseline events横轴0→84→175，最终grid为0→85→176，采样期间逐区间更新，6007服务可访问；原6006服务保留。

```bash
.venv_terrain_cuda/bin/tensorboard --logdir outputs/runs/local_obstacle_input \
  --host 127.0.0.1 --port 6007
```

本轮没有验证学习后避障改善、完整集合成功率或16环境整体吞吐。实施阶段没有静默转换旧checkpoint。后续已按用户指令启动全新策略／Critic／Adam并继续实时TensorBoard，原训练正常保存退出，新run使用剩余1,497,631步，见[切换记录](t11_obstacle_input_training_2026-10-22.md)。
