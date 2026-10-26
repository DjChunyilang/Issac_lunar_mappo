# 25 分钟完整回合：仅远距月面正式训练

2026-10-21 23:35更新：原run在0学习更新时由[性能修复后的独立run](t11_lunar_runtime_performance_2026-10-21.md)接替，任务／学习参数不变，冻结评估与采样并行。本文保留原启动、地图与审计证据；原PID1654210已退出。

## 授权、目标和结束条件

用户明确要求“以25分钟时限来重设参数并启动训练”。本轮采用每回合1500 s的模拟时限；这不是整项训练的墙钟预算。沿用已审计的A1、seed23、200万实际底层环境步，四owner×四环境，全新Actor／Critic和Adam；不继承历史checkpoint。仅远距出生，不启用课程、近距复习、晋级或25万步阶段cap。

目标是从远距出生完成靠拢、制动和四车5 s连续驻留。预算耗尽、用户停止或数值／采样数据失真时结束；低成功率是有效实验结果，不自动追加预算。Proxy执行与Isaac／PhysX验收分别记录。

## 最终参数

| 项目 | 本次设置 |
|---|---|
| 有效训练配置 | `configs/training/upstream_1km_far_fixed_v1.yaml` |
| 物理／规划继承 | `configs/experiment/upstream_1km_curriculum_v1.yaml`；运行时显式覆盖固定出生、月面、1500 s及训练暂停位 |
| 地图 | 真实NPB 5 m DEM，1 km×1 km原始水平尺度；0.1 m处理网格，NASA SFD v2坑石增强，非实测微地形 |
| 地理分离 | 2张训练图、1张固定评估图、1张最终未见图；窗口与种子见地图manifest |
| 出生 | 基础R=200–300 m，每车乘0.9–1；随机整体旋转、平移、车辆排列、±180°朝向；前进初速0.1 m/s |
| 合法出生检查 | 仅四车实际行驶足迹；沿用合法感知与现有完整足迹审查，不筛选最终成功或路线可达性 |
| 完整回合 | 1500 s = 7500底层步 = 300个完整5 s策略决策 |
| 时序 | 策略／GRU每5 s一次，规划每1 s，控制／安全每0.2 s；原0.5 s请求延迟 |
| 折扣 | 每秒gamma=`0.997**0.8`≈0.9975992791，lambda=.99；底层g=`gamma**.2`，l=`.99**.2` |
| 时限联动理由 | `gamma**1500 = .997**1200`≈.02718；保留原20分钟方案的末端折扣权重，不将折扣参考周期误作策略步 |
| 学习 | 32决策／环境／批，5轮PPO，LR=1e-4；clip=.15、entropy=.0009、value=.5；Actor／Critic各自梯度上限1；running价值归一化 |
| 初始化 | HOLD=.35，连续latent std缩放.25；不加停车偏置、方向辅助或离线预训练 |
| 奖励 | 原log1p靠拢／离散度；所有权重保留，以同一派生g计算势差、区间回报与GAE；终端后继势／bootstrap为0 |
| 完整成功 | 原dmax≤12 m、离散度≤30、停车地形／速度／漂移要求以及四车连续驻留5 s，未放宽 |
| 通信／规划 | FULL 300 m、SPARSE 600 m；MPPI K512×2、CUDA、8 s预测；原车辆／安全约束 |
| 学习预算 | 最多2,000,000实际底层步，16环境合计，不乘4辆车；每向量决策预留400步 |
| 独立评估 | 初始、约50万／100万／150万、最终；每次4例×随机／均值，完整1500 s；最终另4例×两模式未见图 |
| 评估预算 | 日程最坏360,000步，上限400,000；最终预留120,000；另启动检查最多400步，全部不进入PPO |
| TensorBoard | 实际学习底层步为横轴；评估进度单独记录；Pinned首列完整任务`SR_S`、`SR_M`，`Dmax_S`仅诊断 |

本次未改动的完整奖励、网络、车辆、通信和优化数值仍见[全参数审计](t11_far_fixed_parameter_audit_2026-10-21.md)，该页先前3600 s候选由本页1500 s最终值覆盖。所有原审计快照保留。

## 必要的既有框架接线

- 现有ScenePack增加条带构建路径：先生成全图统一NASA概率场和坑石目录，再按条带栅格化并带halo计算局部指标。保持0.1 m网格、密度和跨条带坑石连续性。每张构建约111–126 s，峰值RSS约5.2 GiB。
- 现有SceneBank增加显式CPU mmap查询；复用不可变文件与OS页缓存，避免每owner复制整张8通道图。张量缓存旧默认保持，查询的双线性插值及缺测掩码通过对照。
- 现有SpawnCurriculum承载独立版本`rendezvous_fixed_lunar_1km_v1`固定采样模式。内部沿用状态字段以复用reset／checkpoint链路；实际没有课程推进，只有一个far分布。
- 现有T11入口增加`--training-config`同义参数，正式训练runner按版本选择固定远距评估、预算、独立run族及Pinned。新旧配置／学习状态不跨语义恢复。

## 地图证据

`outputs/runs/lunar_terrain_validation/npb_1km_fixed_20261021_01/`保存构建命令、增强参数、四张不可变ScenePack、原始文件SHA、全局目录／概率场和构建耗时／峰值内存。原NPB数据与历史200 m图均保留。训练地图按owner种子分配：23／31用train_0，27／35用train_1；评估使用独立地理窗口。

## 检查与执行状态

目标／最小验收：固定远距、1500 s时钟、合法月面出生与执行可运行，有限学习转移、完整预算和TensorBoard接通；不要求随机初始化策略先取得任务成功。

- 条带与整图构建、NASA目录、跨条带高度／坡度／RMS、缺测及mmap查询：10项通过。
- 固定出生／恢复、无晋级／无阶段cap、折扣与完整评估预算、成功率Pinned及旧课程受影响行为：21项通过。首次出现测试替身不接受新增progress关键字，已限制新进度参数只用于新固定模式，失败日志保留。
- 无学习启动检查：四张地图合计100底层步通过，0优化步；最终检查位于`outputs/runs/upstream_1km_fixed_far/prelaunch_1500s_20261021_02/`，首次失败与复现保留在`prelaunch_1500s_20261021_01/`。
- 正式训练进程已于2026-10-21 22:31:45启动，PID1654210；初始checkpoint与四owner分片已校验，当前运行随机策略初始化完整评估，随后自动进入PPO学习。

## 启动检查发现并修复：已知地形失败被错误排除出学习

第一轮4环境×1决策的检查触发无效样本，未执行优化。独立复现于累计36个底层步、每环境1.8 s时捕获：环境1的车2足迹坡度由0.167492升至0.176603，越过tan(10°)≈0.176327；完整高度／坡度／接触指标有限，失败原因为`driving_slope`。该步安全控制返回verified=true，但推进后新的直接观测已不满足行驶阈值。实际失败记录和完整环境快照保存在首次prelaunch run，不能把安全投影的事前通过当作永久地形合法性证明。

原环境将已知坡度／台阶等超限和缺测共同标为`environment_invalid`，导致奖励清零、学习掩码为false，正式runner随后停止。这使月面中的正常任务失败无法提供学习反馈。

新固定月面配置显式启用`terrain_failure_semantics: known_terrain_terminal_v1`：已知行驶坡度／粗糙度／台阶／车底间隙／不可通过足迹超限记为`terrain_failure`，当步结束，后继势和bootstrap为0，沿用failure_penalty=55并保留有限转移。未知足迹、退化接触几何及非有限状态仍为无效数据，仍停止受影响训练。行驶／停车阈值、成功标准和安全控制未放宽；旧配置默认与旧checkpoint语义保留。新task契约进入快照，禁止与旧状态直接混用。

针对冻结实测失败事实、终端奖励／掩码／恢复、未知数据隔离及旧任务／环境行为的35项检查通过，日志`prelaunch_1500s_20261021_01/logs/terrain_failure_contracts.log`。这只修复失败记账，不说明安全投影已经消除月面失败，也不保证收敛。

第一轮因异常退出未持久化精确总步数，按完整承诺区间保守计100步上界；复现实际36步；最终检查改为每图1环境×1决策，最多100步。整个启动诊断因此最多236／400步，均不进入PPO。

新固定月面训练每完成一个向量策略决策刷新一次collection JSON和TensorBoard，不等待整个32决策批；PPO仍收满原批量后执行5轮更新，学习时间尺度和样本次序保持。完整评估每决策也更新独立进度。

## 正式启动命令与监控

Run：`outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_1500s_2m_20261021_01/`。完整源代码／配置快照为`config/source_and_configs.tar.gz`；身份包含训练源代码、地形源代码、场景manifest SHA、预算和启动审计SHA。`logs/launch.json`保存PID、时间和精确命令，`logs/console.log`为控制台日志。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 \
.venv_terrain_cuda/bin/python -u scripts/run_upstream_t11.py \
  --config configs/experiment/upstream_1km_curriculum_v1.yaml \
  --training-config configs/training/upstream_1km_far_fixed_v1.yaml \
  --run-id a1_seed23_far_lunar_1500s_2m_20261021_01 \
  --representation A1 --mode train --seed 23 --budget 2000000 \
  --num-envs 16 --env-owners 4 --owner-cpu-workers 2 \
  --sampling-protocol owners16_v1 --trainer-backend skrl --policy-device cpu --samples 512 \
  --readiness outputs/runs/upstream_1km_fixed_far/prelaunch_1500s_20261021_02/metrics/summary.json \
  --release-training-pause
```

上例只用于复现命令审计；该run已在运行，不应重复启动。当前服务 http://127.0.0.1:6006 同时展示历史与新run，事件保存于本run的`tensorboard/`，精选链接见`tensorboard/pinned_url.txt`。单独回放：

```bash
.venv_terrain_cuda/bin/tensorboard \
  --logdir outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_1500s_2m_20261021_01/tensorboard \
  --host 127.0.0.1 --port 6007
```

启动时已确认主进程独立于工具shell运行、TensorBoard HTTP200、events文件真实生成；`update_000000.pt`及四owner分片SHA、1500 s deadline、月面类型与新终端契约已核验，源码与地图指纹一致；证据`metrics/launch_verification.json`。初始化评估期间学习步数和PPO更新数保持0，评估不进入训练缓冲。

初始checkpoint SHA256：`f6050f302e6f6553f644ca4c5eded70a159a87aa6c0714db69247a9f58afc3e9`。四owner各4环境的出生尝试数为257／296／135／228，均仅在行驶足迹合法时接受；这些拒绝采样不推进环境、不产生学习样本。训练步／策略决策／优化更新在初始checkpoint均为0。

实时启动复核：初始化随机策略评估已完成首个向量决策，实际评估82步，四环境时长为5／5／5／1.4 s；进程继续运行，无failure产物。TensorBoard HTTP接口已列出新run，events由41,109字节增长至103,696字节，实际评估进度标量已更新。学习步／PPO更新仍为0；完整初始评估结束后自动进入采样更新。证据`metrics/live_start_verification.json`。
