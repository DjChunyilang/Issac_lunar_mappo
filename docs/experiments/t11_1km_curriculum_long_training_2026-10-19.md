# 1 km 场景与课程长训练（2026-10-19）

用户要求扩大地图、同步通信和出生设置，核对奖励后启动一次正式长训练。本轮只运行 A1、seed 23；授权不恢复旧 A0/A1 队列，也不追加多种子或其它动作表示。目标是观察由近至远的完整集合能否学会，而非仅提升训练回报。

## 2026-10-20 11:33：用户要求停止，已完整保存退出

11:29:11 向已核对命令身份的主进程 `633218` 发送 SIGTERM。训练结束当时在途的第 264 批采样／PPO，保存完整 checkpoint 后退出，未开始新批次或追加评估。11:33:35 核验主进程、资源跟踪进程 `633252`、owner `633253–633256` 全部退出；未发现运行中的 T11 训练／队列，GPU compute-apps 列表为空。

| 最终记录 | 数值 |
| --- | ---: |
| 实际底层训练步 | 1,614,642 |
| 策略决策／有效转移 | 67,584 |
| 采样更新／PPO 轮数 | 264／1,320 |
| 未使用训练预算 | 385,358 |
| 独立评估实际步数 | 21,853 |
| 课程级别 | 1（出生半径 5.5–7 m） |
| 当前级最近 40 回合成功数 | 16 |

最新独立评估 `evaluation_step_1602339.json`：停车随机／均值均 4/4；近距随机 2/4、均值 0/4，HOLD 概率约 79.4%。本轮没有进入远距课程，未通过收敛筛选。停止原因是用户指令，未发现数值／采样失败记录。

完整 checkpoint 为 `checkpoints/update_000264.pt`，另有 learner 备份和四个 owner 分片；公开读取校验全部 SHA，恢复计数、课程状态与 summary 一致。所有产物保留，不自动执行剩余预算、最终远距评估或下一轮实验。

同一 run 下的停止证据：

- `metrics/user_stop_request_20261020.json`：用户停止原因、SIGTERM 时间、进程身份和停止策略。
- `metrics/summary_000264.json`：`stop_reason=["SIGTERM"]`、`budget_complete=false`、最终计数与恢复路径。
- `metrics/user_stop_verified_20261020.json`：进程退出、四分片 SHA 和计数核验，无自动恢复授权。

本节覆盖以下历史启动／继续运行安排。只更新停止状态与记录，不修改已运行源码、配置或历史实验产物。

## 启动时预算与结束条件（历史）

- 已于 2026-10-19 17:58:56（北京时间）启动，主进程 PID `633218`，四 owner × 四环境。
- run：`outputs/runs/upstream_1km_curriculum/a1_seed23_2m_20261019_01/`。
- 正式学习预算：最多 **2,000,000 实际底层环境步**；单步为一个环境推进 0.2 s，四辆车不重复计数。每次向量决策先预留 400 步；余额不足即保存退出。
- 独立评估上限 250,000 步，预留 100,000 给最终完整回合面板；前置实际核对另计 1,304 步。测试中的小型控制推进单独记在测试说明中，不计入训练。
- 数值、有效采样或执行错误保存现场后停止，不自动重试。成功率低、课程停留在初级或预算内未收敛均为有效结果；不会强制课程晋级或扩大预算。
- 当前为平地运动学 Proxy 正式长学习。`system_acceptance_complete=false`、`physx_acceptance=false`，不继承旧完整矩阵验收结果。

## 18:10 启动验收

首批采样 346.07 s，实际 **6,262 底层步／256 条有效决策／5 轮 PPO**；HOLD 按持续步数加权为 49.10%。37 个完成回合中 6 次停车成功，未达到课程晋级条件，仍为第 0 级。首轮重放 ratio 最小／最大均精确为 1；6 条未生效终端动作不参与 Actor 更新。梯度与损失有限，归一化 Critic MSE 从 0.9993 降至 0.9876；这里只作运行有效性检查，不据此推断收敛。

`checkpoints/update_000001.pt`、learner 备份及四个 owner 分片均已保存；完整读取校验确认 6,262 步、5 轮 PPO、课程窗口与日志一致，运行源码未变化。主进程与四 owner 继续运行，没有 failure 产物。机器可读验收为 `metrics/startup_verified.json`；物理配置差异与测试日志 SHA 在 `config/prelaunch_verification.json`。

初始完整回合评估用 2,741 步：第 0 级随机／均值策略成功 2/4、4/4，第 1 级为 2/4、0/4。结果见 `metrics/evaluation_initial.json`，作为本轮后续学习比较基线。

## 配置与课程

独立物理配置为 `configs/experiment/upstream_1km_curriculum_v1.yaml`，绑定 `configs/design/upstream_vehicle_proxy_1km_v1.yaml` 的 SHA；学习配置为 `configs/training/upstream_1km_curriculum.yaml`。旧配置和 checkpoint 保留原语义。

| 项目 | 本轮设置 |
| --- | --- |
| 世界边界 | 1000 m × 1000 m，坐标各轴 −500 至 500 m |
| 完整／稀疏通信半径 | 300／600 m；原消息周期、有效期和合法信息约束不变 |
| 局部传感与记忆 | 前后 15 m、侧向 8 m；原采样间隔、未知区与容量限制不变 |
| 车辆与规划 | 原车体、速度／制动／安全约束；MPPI K512、2 轮、40 步／8 s；MOVE 最大 4 m |
| 策略／重规划／控制周期 | 5／1／0.2 s；固定 odometry 目标，不提前重采样 |
| 训练初始化 | A1、seed 23，全新 Actor/Critic 和 Adam；连续初始 std ×0.25 |
| 优化 | 独立 Actor/Critic 梯度裁剪、未生效终端动作信用屏蔽、running value normalization；lr 1e-4、clip .15、entropy .0009、5 轮 PPO |
| 每批 | 每环境 16 次决策，名义 6,400 底层步；按实际持续步数收费 |
| 出生 | 队形整体随机平移、旋转、车辆编号打乱；初始前进速度 0.1 m/s |

出生半径按采样的队形中心定义，不沿用旧配置相对 leader 的径向定义。第 3 级起每车半径乘 0.9–1.0 的独立扰动，打破精确方阵；大尺度下实际质心可略偏离采样中心。整体平移保留最大半径、传感范围、车体半径及 5 m 余量。没有缩放旧 200 m DEM；本轮地形明确为 `flat_proxy`。

| 级别 | 半径或布局 | 回合上限 | 朝向扰动 |
| --- | --- | ---: | ---: |
| 0 停车 | 7.2 m 边长方阵 | 30 s | 向心朝向 ±180° |
| 1 近距靠拢 | 5.5–7 m | 60 s | ±30° |
| 2 短距靠拢 | 7–12 m | 120 s | ±45° |
| 3 局部靠拢 | 12–30 m | 240 s | ±60° |
| 4 中距靠拢 | 30–75 m | 450 s | ±90° |
| 5 远距靠拢 | 75–150 m | 900 s | ±180° |
| 6 大范围靠拢 | 150–300 m | 1200 s | ±180° |

每级至少消耗 10,000 底层步，且最近 40 个**出生于该级**的已完成回合成功率达到 80% 后晋级。当前级以 80% 概率抽样，20% 均匀复习较低级；第 0 级全部抽样自身。晋级只改变下一次 reset，正在执行的回合保留原难度与截止时间。回合记录包含实际出生级别；课程随机数、每环境级别／截止时间和中央晋级窗口均进入完整 checkpoint。

近处初始化与逐步扩大的思路参考 [Reverse Curriculum Generation](https://arxiv.org/abs/1707.05300)；这里采用显式几何阶段和成功率晋级，没有实现论文的反向状态生成算法。价值归一化和集中式 Critic 参考 [MAPPO](https://arxiv.org/abs/2103.01955)。80%、40 回合和各半径是本实验的工程选择，是否有效由本次学习结果判断。

## 奖励与信用核对

原离散度项按距离平方增长：半径 300 m 的方阵，原 gather 势约 −18,459.5。直接放大出生距离会显著放大 Critic 输入和奖励。本轮将两个超限量分别取 `log1p`，相同状态的 gather 势约 −85.2；保持原进展、状态、能量、安全、地形和终端权重。对应 Critic 三个几何比率采用 `log1p(x)/log(2)`，合法邻车相对坐标采用保留符号的对数尺度；Actor 不接收额外真值。

```text
d = log1p(max(dmax − 12, 0) / 12)
s = log1p(max(dispersion − 30, 0) / 30)
Phi_gather = −9.5*d − 6*s
r_gather = g*Phi_next − Phi_start − 0.13*d_next − 0.04*s_next
```

成功奖励 130；失败扣 55；碰撞另外扣 100；超时扣 80。真正终止后继势与 bootstrap 为零。停车成功仍要求最大车距 ≤12 m、离散度 ≤30 m²、共同区域、实际停稳与连续 5 s 驻留，无 MOVE 标签奖励或远处 HOLD 直接惩罚。

为匹配最长 1200 s 回合，本轮把参考 1 s 上的 gamma 设为 `.997`、lambda 保持 `.95`：底层 `g=.997**.2≈.999399279`、`l=.95**.2≈.989793782`。奖励势差、区间累积、GAE 和日志使用同一派生值；时长 D 的 bootstrap 是 `g**D`，GAE 递推是 `(g*l)**D`。价值归一化只改变 Critic 回归单位，GAE 和行为报告保留原奖励单位。

这次把场景、课程、尺度和信用窗口作为一个长训练候选，结果可判断该组合能否学习，不能单独归因于某一参数。

## 启动前证据

检查 run：`outputs/runs/upstream_1km_curriculum/prelaunch_20261019_01/`。

- 13 项针对性测试通过；随后恢复旧奖励线性分支原浮点运算顺序，5 项奖励／出生检查复核通过（不是 18 项不同测试）。日志：`logs/targeted_tests_01.log`、`logs/reward_final_check.log`。
- 896 组静态出生检查通过：工作区内、安全间距、各级截止时间及稀疏通信覆盖。六个半径的等成本奖励方向和终端／非终端势差恒等式通过，见 `metrics/static_audit.json`。
- 四 owner 的真实小型终端、下一课程 reset、PPO、完整 checkpoint 分片保存／恢复通过；恢复保留中央课程窗口和 owner 随机数，预算余量 399 步不再发起向量决策。
- 实际 CUDA K512 核对 1,304 步、0 优化器更新；近距 HOLD 四例均完成停车驻留。局部 12–30 m 四例的配对结果如下，各行使用完全相同初态。

| 干预（20 s） | 四例折扣奖励 | 四例最大车距缩小量（m） |
| --- | --- | --- |
| HOLD | −17.899、−22.390、−23.087、−13.355 | 0.008、0.007、0.007、0.008 |
| 持续向心 | −0.960、−2.307、−3.829、2.451 | 11.592、19.387、19.801、7.915 |
| 向外 | −30.189、−32.385、−32.734、−28.251 | −16.014、−15.611、−15.646、−15.726 |

逐例奖励排序正确；全轨迹与统计在 `metrics/level_*.json`，总账在 `metrics/summary.json`。干预轨迹不进入 PPO。这些检查确认实验能提供有效反馈，不证明学习收敛。

## 评估、恢复与监控

初始评估分别在第 0、1 级，使用固定未参与训练的种子 104729 和采样种子 104759。每级四个初态，随机策略和均值策略各跑完整回合。每累计约 20 万训练步重复当前级与第 0 级，预算不足时记录跳过并保留最终额度；最终补第 6 级。独立环境和 RNG 不改变训练轨迹。

工程收敛筛选要求：三个不同训练检查点上，第 0 和第 6 级的随机、均值策略各至少成功 3/4；同时保留距离、离散度、HOLD、目标跳变及安全记录。仅最终一个面板通过不称稳定收敛，固定四例也不代表统计显著性。

保存初始化、前两批、每五批、晋级／评估边界及结束的完整 checkpoint，逐 owner 分片并先写 learner 备份。每批保留原始 on-policy 动作、概率、GRU、GAE 输入和执行记录。源码及配置冻结为 `config/source_and_configs.tar.gz`；有效配置见 `config/effective_configuration.json`。不允许从旧语义 checkpoint 恢复；存在更新边界之后的未提交采样或失败现场时，先核算已消耗步数，禁止自动重复预算。

实际启动命令：

```bash
.venv_terrain_cuda/bin/python -u scripts/run_upstream_t11.py \
  --config configs/experiment/upstream_1km_curriculum_v1.yaml \
  --curriculum-config configs/training/upstream_1km_curriculum.yaml \
  --run-id a1_seed23_2m_20261019_01 --representation A1 --mode train \
  --seed 23 --budget 2000000 --env-owners 4 --num-envs 16 \
  --owner-cpu-workers 2 --samples 512 --trainer-backend skrl --policy-device cpu \
  --release-training-pause \
  --readiness outputs/runs/upstream_1km_curriculum/prelaunch_20261019_01/metrics/summary.json
```

监控 `logs/train.log`、`metrics/update_*.json`、`metrics/evaluation_*.json`、`metrics/failure_*.json`；`logs/launch.json` 保存 PID、时间和实际 argv。完整结束写 `metrics/summary_*.json`。需要正常暂停时向主进程发送 SIGTERM，等待当前采样与更新边界保存；恢复使用同一命令追加 `--resume-state <该 run 的完整 update_*.pt>`，不使用 `.learner.pt`。

前置实际核对命令：

```bash
.venv_terrain_cuda/bin/python scripts/audit_upstream_curriculum.py \
  --config configs/experiment/upstream_1km_curriculum_v1.yaml \
  --curriculum-config configs/training/upstream_1km_curriculum.yaml \
  --output outputs/runs/upstream_1km_curriculum/prelaunch_20261019_01
```

该命令所指产物已存在，只用于复现记录，不应覆盖重跑。
