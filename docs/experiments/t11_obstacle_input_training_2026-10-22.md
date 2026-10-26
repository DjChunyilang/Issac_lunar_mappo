# T11 新感知与障碍输入正式训练切换

日期：2026-10-22。用户要求停止原训练，改用已实施的固定格、增量感知和障碍输入进行训练。观测及cost／reward语义发生变化，本次新建run和全新策略／Critic／Adam／价值归一化状态，不导入旧checkpoint。保留原200万学习底层步总上限，扣除旧训练消耗后分配剩余预算。

## 旧训练停止边界

原run：`outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_soft_hazard_2m_20261022_01/`。主PID1762613收到SIGTERM，在第52批正常保存退出：**502,369实际学习底层步、52采样批／260轮PPO**。完整checkpoint为`checkpoints/update_000052.pt`，头文件和四owner分片均校验通过；原结果、源码包、checkpoint与events保留，主进程和四owner均已退出。

停止证据：`outputs/runs/local_obstacle_input/training_switch_20261022_01/metrics/old_stop_verification.json`。原完整总结为旧run的`metrics/summary_000052.json`，未提交步骤没有遗留到新训练。

## 新训练配置与预算

新run：`outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_remaining_20261022_01/`。

| 项目 | 设置 |
| --- | --- |
| 输入／实验配置 | `configs/experiment/upstream_1km_obstacle_input_v1.yaml`，统一输入为`configs/terrain/upstream_1km_obstacle_input_v1.yaml` |
| 训练配置 | `configs/training/upstream_1km_far_obstacle_input_v1.yaml` |
| 感知与地形 | 0.2 m坑石标签、0.8 m保守普通地形、私有固定格、静态tile复用和增量揭示；当前真实足迹713点原网格子集 |
| 观测／反馈 | Actor523、Critic1439、三通道区域聚合；已观测障碍软cost和实际执行pre-step奖励，原物理阈值和车辆模型保留 |
| 学习 | A1／seed23、全新网络／Adam；16环境／4 owner、每owner双线程、CPU策略、CUDA MPPI512×2；原5 s决策、32决策／5轮PPO |
| 任务 | 1 km远距月面、R200–300 m、1500 s回合、无课程／晋级／复习；继续使用原四张地形图及其派生atlas |
| 学习预算 | 新run至多**1,497,631底层步**，加旧502,369步，合计不超过原2,000,000步；向量完整区间预留可能留有少量尾数 |
| 独立评估 | 新run按原面板与自身学习步轴执行，最坏预留300,000步；旧实际744步，合计上界300,744≤原400,000步 |
| 恢复与日志 | 新横轴从0开始，旧曲线不拼接；保持源码／配置／atlas身份校验，TensorBoard实时保存 |

本次启动入口、CLI及训练器共用预算检查支持固定月面run使用剩余预算，旧课程预算及非训练配置CLI枚举继续保持。奖励预启动审计同步核对新障碍奖励身份。未改变PPO算法、网络初始化或MPPI计算预算。

## 最小启动验收

复用[实施验收](t11_local_obstacle_input_2026-10-22.md)的感知、局部信息边界、CPU／CUDA、区域聚合、cost和恢复证据；不重复运行无关全仓检查。本次45项不同检查通过，覆盖新CLI剩余预算转发、入口／训练器预算、奖励身份，以及受影响的固定远距、课程、SKRL和grid路径；XML为切换run的`metrics/checks_02.xml`及`checks_03.xml`，同名检查只计一次。

实际地图准入位于`outputs/runs/local_obstacle_input/training_prelaunch_20261022_01/`，训练两图、固定评估和未见图共72底层步（25／8／25／14），0优化更新；出生合法、观测和学习数据有限有效。地形失败是正常任务结果，不以初始策略成功率作为准入条件。有效证据为`metrics/readiness_02.json`及`static_audit.json`。

初次预启动输出路径冲突被不可覆盖检查拒绝，0环境步；首次正式启动被旧CLI预算choices拒绝，同样在环境创建前退出，0训练步。已补齐CLI与分发检查后重启；失败日志及初次command／readiness保留，不冒充成功启动。

实际启动命令及环境变量保存在切换run的`config/command_02.txt`与`launch_02.json`，正式run另外保存`config/source_and_configs.tar.gz`、完整有效配置和初始checkpoint。使用的核心参数如下：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv_terrain_cuda/bin/python -u scripts/run_upstream_t11.py \
  --config configs/experiment/upstream_1km_obstacle_input_v1.yaml \
  --training-config configs/training/upstream_1km_far_obstacle_input_v1.yaml \
  --run-id a1_seed23_far_lunar_obstacle_remaining_20261022_01 \
  --representation A1 --mode train --seed 23 --budget 1497631 \
  --num-envs 16 --env-owners 4 --owner-cpu-workers 2 \
  --sampling-protocol owners16_v1 --trainer-backend skrl --policy-device cpu \
  --samples 512 \
  --readiness outputs/runs/local_obstacle_input/training_prelaunch_20261022_01/metrics/readiness_02.json \
  --release-training-pause
```

## 实际启动与监控

主PID**2056784**于本地18:35:45启动，独立session后台运行。初始`checkpoints/update_000000.pt`与四owner分片校验通过；Actor523／Critic1439、三通道输入，实际owner标识为`actual_incremental_private_grid`，网络头为正常随机初始化，未沿用性能探针的零权重头。初始计数为0，Adam状态为空、无旧checkpoint继承。

截至本轮启动核验，已连续记录6个区间，实际学习底层步**372→741→1103→1468→1829→2186**；TensorBoard events同步递增，6006服务已加载新run，无failure／invalid batch产物，当前正在收集首批PPO数据，独立初始评估并行执行。核验记录为切换run的`metrics/launch_verification.json`；只记录启动成立，不把0次已完成PPO更新当作学习效果。控制台为正式run的`logs/console.log`，指向切换run的`logs/training_console_02.log`。

交付源码与新run身份快照逐文件一致，四张原scene SHA及地形源码身份一致；配置、command和新旧预算边界均已写入正式run的`config/`。

复用本地TensorBoard服务：`http://127.0.0.1:6006`，新run日志为`outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_remaining_20261022_01/tensorboard/`。当前服务同时加载历史和fixed-far目录，新run为独立曲线；用新目录下的`pinned_url.txt`查看完整far任务成功率等重点图。查看命令：

```bash
.venv_terrain_cuda/bin/tensorboard \
  --logdir outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_remaining_20261022_01/tensorboard \
  --host 127.0.0.1 --port 6006
```

服务已经在6006运行，无需重复启动；远程IDE转发同一端口。6007继续保留本次预启动／实施探针的events。训练按剩余预算运行，不在启动验收后自动追加预算。任务成功率、terrain_failure、障碍near／overlap、HOLD和实际吞吐形成后续结果；本次启动不代表学习收敛或Isaac／PhysX验收。

## 当前采样性能补充

2026-10-22已完成[新路径的全流程性能分解](t11_obstacle_sampling_profile_2026-10-22.md)，配对200诊断步、0更新，不修改本run源码／配置或预算。主要热点为MPPI的逐车CUDA格查询，其次为真实足迹／记忆刷新和重复观测编码；正式训练继续。
