# 下一次正式长训练前：有效参数与配置审查

日期：2026-10-20。状态：**审查完成，参数未修改，训练保持停止；本表不是新训练启动配置。**

目标：核实上一轮正式入口实际使用的参数、代码默认值和覆盖关系，列出下一轮需要审定的项目。最小验收为配置、生产代码、运行元数据与 checkpoint 交叉一致；本次新增环境步、评估步和优化器更新均为 0。不重跑已完成的算法验收，也不以本次静态核对替代新方案的闭环验证。

## 1. 审查对象与证据

上一轮 run：`outputs/runs/upstream_1km_curriculum/a1_seed23_2m_20261019_01/`。下文以此为 `RUN/`；源码简称 `LG/` 指 `source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/local_goal/`。

| 证据 | 用途 |
|---|---|
| `RUN/run_manifest.json` | 原始完整启动命令、源码 SHA256、预算及身份 |
| `RUN/config/effective_configuration.json` | 各 owner 实际环境、奖励、观测、时间及学习参数 |
| `RUN/metrics/initialization.json` | seed23、全新 Adam、未继承 checkpoint、探索缩放 |
| `RUN/checkpoints/update_000264.pt` | 最终网络、Adam param_groups、归一化统计、课程与计数 |
| `RUN/metrics/summary_000264.json` | 最终计数、停止原因、课程晋级历史 |
| `RUN/metrics/evaluation_step_1602339.json` | 最近固定初态独立评估 |
| `RUN/metrics/user_stop_verified_20261020.json` | 保存及进程退出记录 |

核对结果：manifest 记录的 **241 个源码文件均存在且 SHA256 与当前文件一致**。checkpoint 中 Adam 为 `lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0, amsgrad=False`。当前未发现正式训练入口或训练队列进程。此次没有恢复模型采样，也没有启动优化器。

| 配置 | 当前 SHA256 |
|---|---|
| `configs/training/upstream_1km_curriculum.yaml` | `fd2b29b1b397871491d1706f70c92221b45986ca0929d598a241b7f4406a26bd` |
| `configs/experiment/upstream_1km_curriculum_v1.yaml` | `3f580374ec864995a6134e6b1276d618b2efc26a8867b2076b4c7249d133b50f` |
| `configs/design/upstream_vehicle_proxy_1km_v1.yaml` | `c84cac9e5a765bbc64fd325695f922225461e7c65478c82411c964ce44087b83` |

入口为 `scripts/run_upstream_t11.py --mode train --curriculum-config ...`，转入 `scripts/upstream_curriculum_experiment.py`。运行期又由 `scripts/upstream_local_closure.py:runtime_config`、`session_configuration` 和 `make_collector` 设置环境数、初速度、MPPI、课程及暂停释放，不能只看一个 YAML。

实验 YAML 的完整继承链如下，名称均位于 `configs/experiment/`：

```text
upstream_1km_curriculum_v1
→ upstream_scale_s04_braking_lease_cuda_v5
→ upstream_scale_s04_frame_review_cuda_v3
→ upstream_scale_s04_clearance_recovery_cuda_v4
→ upstream_scale_s04_priority_recovery_cuda_v3
→ upstream_scale_s04_visibility_recovery_cuda_v2
→ upstream_scale_s04_selected_cuda_v4
→ upstream_scale_s04_calibration_cuda_v4
→ upstream_scale_s04_cost_cuda_v3 → upstream_scale_s04_cost_v3
→ upstream_scale_s03_environment_v2 → upstream_scale_s03_environment_v1
→ upstream_scale_s03_observation_v1 → upstream_scale_s03_precision_v2
→ upstream_scale_s03 → upstream_scale_s02
→ upstream_scale_s01_review_v2 → upstream_scale_s01 → upstream_scale_s00
```

## 2. 应先审定的结论

| 项目 | 核实的事实 | 对下一轮的含义 |
|---|---|---|
| 主训练出生范围 | 第 0 级已在成功几何区域；第 1 级理想方阵只需径向内移约 0.023–1.523 m | 可以用于停车和临界靠拢，但不宜把其表现等同于远距集合能力 |
| 课程卡级 | 需最近 40 个本级回合至少成功 32 个；没有单级最大预算或卡级处理；上一轮只升到第 1 级 | 应先决定主任务覆盖和单级停止条件；不能继续假定“训练更久自然覆盖远距” |
| 执行速度与动作时间 | 近距速度承诺的初始请求值约 0.10–0.28 m/s；MOVE 初始目标约 2 m、决策持续 5 s | 不能用 2.78 m/s 顶速设计近距时限；目标距离、持续时间、执行速度必须一起审核 |
| 回报与 GAE | 回报折扣半衰期约 230.70 s；GAE 中后续 TD 残差直接递推权重半衰期约 12.77 s | 延长 gamma 没有同时把 lambda 信用递推变成几百秒；是否调整 lambda/序列长度应单独决定 |
| HOLD 的状态区分 | 最近评估停车、近距平均 HOLD 概率均约 0.79；近距随机 2/4、均值 0/4 | 要审查是否学到“何时停”，不能以停车成功或总回报代替靠拢学习 |
| 探索 | 初始 latent std≈0.15163；最终约 0.14842、0.14653，未触及 std 下限 | 当前证据不支持把失败简单归因于连续噪声塌缩；门控、状态区分和训练分布更值得核对 |
| 评估覆盖 | 中途只评停车级和当前级；末级只有达到该级或最终评估才出现 | 下一轮应预先决定固定远距检查点；现有“连续 3 个检查点末级通过”要求可能直到预算末才有机会取证 |
| 1 km 场景含义 | 平地运动学 Proxy；出生仍围绕局部中心，最远初始车距≤600 m | 地图变大不等于已训练全图分散、断联、复杂地形或真实 PhysX 动力学 |

以上区分了确定的配置行为与需要实验判断的学习机制。**本次没有确认某一个奖励系数是唯一根因，也没有选定下一轮的新数值。**

## 3. 地图、车辆与物理约束

来源：设计 profile、`LG/upstream_spec.py`、运行 contracts。

| 参数 | 实际值／语义 |
|---|---|
| 地图、地形、边界 | 1000×1000 m；`flat_proxy`；未知地形和世界边界仍为硬约束 |
| 车数与模型 | 每环境 4 车；差速运动学；midpoint 积分；车体平面几何中心为参考点 |
| 车长／宽／高 | 5.015／3.2／1.85 m |
| 轮径／半径 | 1.0／0.5 m |
| 轮距／轴距 | 1.8／2.7 m；有效轮距倍率 1.0 |
| 接触斑 | 四接触斑；长 0.2 m、宽 0.3 m |
| 前进／倒车速度上限 | 2.7777778／0.5 m/s；前进为 10 km/h |
| 常规／低速角速度上限 | 0.35／0.2 rad/s；低速阈值 0.05 m/s |
| 线加速度／减速度 | 0.2／0.4 m/s² |
| 角加速度／减速度 | 0.15／0.2 rad/s² |
| 轮加速度／推导轮速上限 | 1.2 rad/s²；5.5555556 rad/s，非实测电机参数 |
| 初始控制 | **所有课程实际 v=0.1 m/s，ω=0**；不是 profile 的测试参考速度 1 m/s |
| 名义平面速度增益 | 1.0；地形相关增益仍通过执行模型计算，本轮平地 |
| 车体外接圆半径 | 2.9744842 m |
| 每车安全裕量／有效圆半径 | 0.3／3.2744842 m；两车有效圆中心间隔 6.5489684 m |
| 几何误差预算 | 定位 0.1 m、跟踪 0.1 m、航向 0.5°、附加 0.05 m；解析时验证总预算不超 0.3 m，不表示训练注入了这些噪声 |
| 地形足迹／扫掠采样 | 有向车体矩形＋接触斑；最大扫掠边界间距 0.1 m |
| 行驶／停车坡度限制 | 10°／5° |
| 粗糙度／台阶／离地间隙 | RMS 0.05 m／0.1 m／0.3 m |
| 最低 traversability | 0.25 |
| 占位物理量 | 质量 1500 kg、质心高 0.8 m、重力 1.62 m/s²、摩擦系数 0.6；不是质量／轮土动力学训练，不能据此声称真实动力学验证 |

下轮应先保留用户已给出的尺寸及物理约束；出生距离与课程调整不能靠缩小车辆或取消安全裕量实现。

## 4. 出生、课程与任务成功条件

### 4.1 课程的实际采样

半径 R 指相对**采样队形中心**的车中心距离，不是车间距、车体净空或目标容差。前三级四车在旋转方阵角点；从第 3 级起，每车半径额外乘独立 U[0.9,1.0]，因此表内基础半径下界不等于各车实际半径下界，实际质心也可偏离采样中心。

| 级别 | 基础 R（m） | 回合时限 | 朝向相对向心方向的均匀扰动 |
|---|---:|---:|---:|
| 0 parking | 7.2/√2 = 5.0911688 | 30 s | ±180° |
| 1 near_approach | 5.5–7 | 60 s | ±30° |
| 2 short_approach | 7–12 | 120 s | ±45° |
| 3 local_approach | 12–30，逐车再乘 0.9–1 | 240 s | ±60° |
| 4 medium_approach | 30–75，逐车再乘 0.9–1 | 450 s | ±90° |
| 5 long_approach | 75–150，逐车再乘 0.9–1 | 900 s | ±180° |
| 6 wide_approach | 150–300，逐车再乘 0.9–1 | 1200 s | ±180° |

| 其他课程参数 | 实际值／行为 |
|---|---|
| 采样 | 基础半径区间均匀；队形整体旋转均匀；车辆标签随机置换 |
| 队形中心 | 在满足边界余量的矩形内均匀平移；单轴半宽=500−最大车半径−23.2744842 m |
| 易场景复习 | 每次 reset 有 20% 概率均匀选一个较低级别；这是**回合采样比例**，不是环境步占比；第 0 级不复习 |
| 晋级窗口 | 最近 40 个实际出生于当前级别的结束回合 |
| 晋级阈值 | ≥80%，即至少 32/40 成功 |
| 最少本级步数 | 自晋级时刻起累计 ≥10,000 实际底层步；这个步数包含低级复习 |
| 晋级时机 | 更新边界晋级，新级别在回合 reset 时生效；不强行截断旧回合 |
| 当前缺少的机制 | 单级最大预算、停滞处理、降级、越级或保证较难场景曝光的混合采样 |
| 时钟观测 | Actor/Critic 按各环境本回合真实课程时限计算；不是全部按 1200 s |

第 0 级代码使用 `task.spacing/√2`；修改课程 YAML 第 0 级 `radius_m` 并不会改变这一出生方式。第 1 级朝向大多向心，初始 beta≈0 的前向 MOVE 本来就有向心先验；不能把初始化靠拢全部算为学到的合作能力。

### 4.2 成功判据与 5 m 车长的关系

| 成功／目标参数 | 实际值 |
|---|---:|
| 最大两车中心距离 | ≤12 m |
| 四车关于质心的平均平方距离（离散度） | ≤30 m² |
| 共同工作区半径 | 10 m，车辆有效足迹需容纳在工作区 |
| 停车线速度／角速度 | ≤0.01 m/s／≤0.003 rad/s |
| 车体边界速度 | ≤0.02 m/s |
| 连续驻留 | 5 s＝25 底层步 |
| 驻留最大位置／航向漂移 | 0.1 m／1° |
| 参考方阵边长 | 7.2 m；成功不是“必须精确到四个指定槽位” |
| 局部目标捕获／退出半径 | 0.2／0.4 m，不能拿它们替代团队成功判据 |

理想对称方阵的最大成功半径为 `min(12/2, sqrt(30), 10−3.2744842)=5.4772256 m`。因此：

- 第 0 级：相邻车距 7.2 m、对角车距 10.1823 m、离散度 25.92 m²，出生已满足成功几何，只需正确制动并驻留。
- 第 1 级：相邻车中心 7.7782–9.8995 m；扣除两车有效圆后还剩约 1.2292–3.3505 m，几何上并非车体重叠。
- 第 1 级理想径向靠拢只需约 0.0228–1.5228 m；这是几何距离，不包含转向、避碰、安全制动和驻留时间。
- R=12–30 m 的理想方阵需径向内移约 6.52–24.52 m，才更能检验持续靠拢。

## 5. 感知、通信、网络输入

| 参数 | 实际值／语义 |
|---|---|
| 本地传感范围 | 前 15 m、后 15 m、侧向 ±8 m；理想局部 Proxy 传感 |
| 传感采样／每帧点数 | 0.2 m；151×81＝12,231 点 |
| 本地记忆分辨率／容量 | 0.1 m；65,536 样本 |
| 最近邻支持半径 | 0.155 m；保留未知空隙，不填成已知可通行 |
| 完整通信 | 半径 300 m，周期 0.2 s |
| 稀疏通信 | 半径 600 m，周期 1 s |
| 状态／预测轨迹有效期 | 1／0.6 s |
| 未知发送者速度界 | 2.7777778 m/s；较低速度界必须有覆盖查询时域的有效执行承诺 |
| 通信模型 | 距离模型，无物理链路预算；预测轨迹不等于未来控制承诺 |
| Actor 输入 | 每车 411 维：自身 15＋邻车 51＋地形 224＋聚合 5＋反馈 115＋时钟 1 |
| 地形编码输入 | 112 点×2 通道，三尺度 7×9、3×7、4×7 |
| Critic 输入 | 团队 991 维；集中训练真值，9 个保留 oracle 特征为零 |
| 空间尺度 | `log1p`；合法邻车相对坐标用 signed log1p，以 12 m 和 300 m 定标 |
| Critic 几何尺度 | 车距/12、离散度/30 等先无量纲化，再 `log1p(x)/log(2)`；世界坐标按半宽 500 m 缩放 |
| 其他归一化 | 本车目标按 4 m、速度按物理上限、驻留计数按 25；没有额外在线 observation running mean/std |

Actor 取真实环境的合法局部观测，不读取团队真值中心。课程最远初始两车距离不超过 600 m；当前设计没有覆盖任意 1 km 地图上的断联出生。地图、通信半径和出生半径必须分别理解。

## 6. 高层动作、时间与探索

| 参数 | 实际值／语义 |
|---|---|
| 表达 | A1：HOLD Bernoulli 门控＋MOVE 的极坐标目标；不学习速度帽或终端航向 |
| MOVE 距离 | `rho=2*(tanh(z_rho)+1)`，范围 0–4 m |
| MOVE 方向 | `beta=pi*tanh(z_beta)`，范围 −π–π，决策开始时以车身方向转换到固定 odometry 目标 |
| 策略／GRU 周期 | 5 s，一次决策仅推进一次 GRU、记录一次动作概率 |
| 重规划／底层控制周期 | 1／0.2 s；每次完整决策 25 底层步 |
| 持续决策 | 续发同一固定 odometry 目标；局部到达、HOLD、安全制动不触发提前重采样 |
| 请求有效期／模拟延迟 | 1／0.5 s；延迟按底层步向上取整为 3 步＝0.6 s 的激活边界 |
| 终端信用 | 真实终端且持续步数≤3 时，本次动作没有执行影响，屏蔽其 Actor 信用；仍保留 Critic 和向前传播信用 |
| 初始头部 | MOVE mean、HOLD logit 权重正交 gain=.01、bias=0；HOLD 初始概率约 .5 |
| 连续探索初值 | 原 log_std=−.5，再加 log(.25)，得到 −1.8862944、std≈0.151633；这是 latent 单位 |
| std 范围 | log_std clamp[−5,1]，std约[0.006738,2.718282]；两维独立、全局可学习，不依状态变化 |
| 初始平均目标 | rho≈2 m、beta≈0，近似朝车辆前方 |
| 随机／均值执行 | 随机 Bernoulli＋连续采样；均值执行在 logit>0 时 HOLD，否则 MOVE；门控阈值 .5 |
| 概率与熵 | HOLD 原子概率与条件 MOVE 变换密度；含 tanh Jacobian 和物理尺度；连续熵采用采样估计并乘 MOVE 概率 |

最终 checkpoint 的 log_std 为 `[-1.90768075,-1.92051017]`，相对初始化仅小幅变化。均值策略在 p(HOLD)>.5 时可能始终停住，随机策略偶尔 MOVE 成功不能替代均值策略靠拢能力。

## 7. 奖励完整清单与时间单位

来源：`LG/upstream_reward.py` 与运行 reward settings。以下系数按 **0.2 s 底层步**使用；课程改变了几何尺度和折扣，没有重调这些权重。

| 参数 | 值 | 用途 |
|---|---:|---|
| `distance_progress` | 9.5 | 最大车距势差 |
| `dispersion_progress` | 6 | 离散度势差 |
| `distance_level` / `dispersion_level` | .13 / .04 | 每步几何超限成本 |
| `flatness_progress` / `flatness_excess` | 2 / .02 | 近集合区域平坦度势差／超限成本 |
| `hold_potential` | 12 | 成功驻留进度势；不是“选 HOLD 就加 12” |
| `energy_weight` | .003 | 能耗成本总乘数 |
| `distance_cost` / `turn_cost` | .02 / .007 | 平移／转动成本 |
| `terrain_weight` | .1 | 地形成本总乘数 |
| `roughness_cost` / `traversability_cost` | .15 / .15 | 粗糙度／不可通行成本 |
| `speed_loss_cost` / `height_change_cost` | .14 / .09 | 地形速度损失／高度变化成本 |
| `collision_cost` / `clearance_cost` | 100 / 2.4 | 碰撞事件／有效圆净空超限 |
| `success_bonus` | 130 | 团队真实成功终端 |
| `failure_penalty` | 55 | 碰撞或越界终端 |
| `timeout_penalty` | 80 | 课程截止时间终端 |
| `gather_scaling` | log1p | 对几何超限量进行对数压缩 |

设最大中心距为 dmax，离散度为 S，底层折扣为 g：

```text
ed = log1p(max(dmax−12,0)/12)
es = log1p(max(S−30,0)/30)
Phi_gather = −9.5*ed − 6*es
r_gather = g*Phi_next − Phi_start − .13*ed_next − .04*es_next

near = clip((24−dmax)/12,0,1)
flat_excess = max(flatness_cost−1,0)
Phi_flatness = −2*near*flat_excess
r_flatness = g*Phi_next − Phi_start − .02*near_next*flat_excess_next

Phi_hold = −12*(1−clip(hold_count/25,0,1))
r_hold = g*Phi_next − Phi_start

r_energy = −.003*(.02*mean(abs(v))*.2 + .007*mean(abs(omega)))
r_terrain = −.1*mean(.15*roughness + .15*(1−traversability)
                       + .14*(1−terrain_gain) + .09*abs(height_change))
r_clearance = −2.4*max(2*effective_radius−minimum_center_distance,0)
```

平坦度取四车足迹报告最差值，由坡度/停车坡度、RMS/.05、台阶/.1 和车底间隙构成，上限 10；正常平地为 1，故超限项为零。真实终端的所有后继势与 bootstrap 归零；碰撞终端同时承担 −100 和 −55，外加该步其他项，不能只看 `failure_penalty`。

注意三点：

1. 驻留计数由实际几何、速度、地形与漂移决定，并非 HOLD 标签奖励。固定状态下负势因折扣可以产生小正值，但势差会与后续势及终端共同结算，不能单看一步正值就断言 HOLD 被错误奖励。
2. 状态成本每底层步扣除；能耗里的平移项乘 dt，转动项未乘 dt，地形各项也是当前实现的离散成本。若以后改 dt，不能只改 gamma 而保持这些项含义不审查。
3. log1p 使大距离处单位米的奖励梯度减小，同时避免远距数值过大。近距配对排序正确不自动证明 150–300 m 的收益、时限和执行可达性合适。本次没有新跑远距配对。

## 8. 折扣、GAE、PPO 与价值归一化

| 参数 | 实际值／行为 |
|---|---|
| 折扣参考周期 | 1 s |
| 参考 gamma／lambda | .997／.95 |
| 底层 g／l | .9993992787012722／.9897937816869885 |
| 5 s bootstrap 折扣 | g^25 = .9850897304 |
| 5 s GAE 递推系数 | (g·l)^25 = .7622436551 |
| 每环境 rollout | 16 个决策，名义 80 s；终止回合独立结算，不把下一回合拼到旧动作 |
| 环境／名义批次 | 4 owner×4 环境＝16；256 条团队转移，对应 1024 个车动作 |
| PPO epochs／minibatches | 5／1；完整序列批次更新，无额外随机打散小批 |
| Actor/Critic 学习率 | 共享 Adam，lr=1e-4，固定，无学习率退火 |
| Adam | betas=(.9,.999)，eps=1e-8，weight_decay=0，amsgrad=False |
| PPO clip | .15 |
| 熵系数／价值损失系数 | .0009／.5 |
| 梯度裁剪 | Actor、Critic 分开裁剪，各 norm≤1；不共用一个总 norm |
| Actor 优势归一化 | 对可用于 Actor 的团队优势做均值／标准差归一化；std 下限 1e-6，广播给四车 |
| Critic loss | MSE；无 Huber、value clipping、reward/return clipping |
| KL | 记录 approximate KL；没有 target-KL 提前停止更新机制 |
| 价值归一化 | running，累计团队 return moments，非 EMA、非 PopArt；std 下限 .1 |
| moments 更新 | 每条有效团队目标每批进入一次；不是按 4 辆车或 5 轮 PPO 重复统计 |
| GRU 反传 | 从批次初始 detached hidden 重放完整序列；回合终止重置，缓冲切分不重置 |
| 学习数据 | on-policy；没有旧经验池、方向教师或辅助模仿损失 |

```text
区间奖励 R = Σ(k=0…D−1) g^k*r_k
delta = R + g^D*V(next) − V(start)
GAE carry = (g*l)^D
loss = actor_loss + .5*critic_MSE − .0009*entropy
```

分段奖励按已推进时长折扣，真实终端 bootstrap=0；非终端采样切分保留尾值。GAE 使用反归一化的原奖励单位 V，Critic 训练目标再归一化。

时间换算：gamma 回报权重 e 折时间约 **332.83 s**、半衰期 **230.70 s**；GAE 后续 TD 残差直接权重 e 折时间约 **18.42 s**、半衰期 **12.77 s**，60／80 s 后权重约 .03847／.01299。这里不是硬截断：Critic bootstrap 仍能传播更长时间的价值，但需要 Critic 学得准确。增加 gamma、增加 lambda、加长 rollout 分别改变不同环节。

价值归一化没有同步补偿 Critic 最后一层以保持原始 V 不变。moments 更新会改变归一化坐标；这是现有选项的机制，不是本次新发现的数值错误。最终 moments count=67,584，正好等于有效团队转移数；mean≈0.8296064、M2≈252,871,721.75。跨课程分布变化时，应结合原单位价值误差判断，不能只看归一化 MSE。

## 9. 网络结构与运行设备

| 部分 | 实际结构／设置 |
|---|---|
| 共享 Actor | 四车共享参数，各自独立 hidden |
| 地形卷积 | 2→16→32 通道，3×3、padding=1、ELU；三尺度共享卷积，各池化到 3×3 |
| 地形投影 | 864→64，ELU |
| 自身／邻车／聚合／反馈 | 15→32／51→48／5→16／116→64，均 ELU |
| 融合 trunk | 224→128，ELU |
| 循环单元 | GRUCell(128,128) |
| Actor 输出头 | mean 128→2、HOLD logit 128→1、全局 log_std 两维 |
| Critic | 991→256→128→1，隐藏层 ELU，前馈，无 Critic GRU |
| 其他正则化 | 无 dropout、BatchNorm、LayerNorm；除显式输出头初始化外按模块默认初始化 |
| 框架 | skrl MAPPO 容器，实际更新走自定义 `recurrent_mappo_update`，不能用 skrl 默认超参数推断行为 |
| 设备 | 策略／PPO CPU；各 owner MPPI CUDA；环境运动学 CPU |
| 并发 | 4 owner，每 owner 4 环境、2 CPU workers；torch threads=1；无集中 GPU service |
| 随机种子 | 中央策略 23；owner 23、27、31、35；全新网络和 Adam |

## 10. MPPI、速度承诺与安全投影

### 10.1 MPPI 实际参数

来源：`LG/upstream_factory.py` 实际应用 config overrides，及 `upstream_mppi.py`、`upstream_mppi_precision.py`、`upstream_mppi_calibration.py`。

| 参数 | 实际值 |
|---|---:|
| adapter／cost | `mppi_upstream_v4`／`upstream_mppi_target_normalized_cost_v3` |
| horizon／dt／steps | 8 s／.2 s／40 |
| samples／iterations | 512／2；iterations 在运行入口明确设置 |
| 控制采样噪声 v／ω | .375 m/s／.15 rad/s，覆盖 profile 的 .25／.10 |
| temperature | .5 |
| goal／terrain／neighbor／smooth scale | 2／.5／.5／.5 |
| query_chunk／rollout_chunk | 256／4096 |
| query memory | 64 MiB |
| 候选样本完整复核上限 | 16 |
| 规划 deadline | None，离线运行，无已验收实时截止承诺 |
| 数值精度 | 采样 float32，执行器／预测／复核使用 FP64 路径 |

规划代价并非 RL 奖励，两者不可混淆。实际代价系数如下：

- 目标位置：基础沿程 2、末端 12，再乘 goal_scale=2，以 `max(本次剩余目标距离,.2 m)^2` 归一化；MOVE 使用滚动前向进度与横向误差，经过目标不等于强制停车。
- 终端航向：基础系数 2 再乘 goal_scale；A1 没有终端航向目标，该项不激活。
- 控制幅度：按速度／角速度上限归一化后平方，系数 .05／.03。
- 平滑：控制变化按物理速度上限归一化，基础系数 .5／.2，再乘 smooth_scale=.5；不是旧版的加速度×dt 归一化。
- 倒车：按倒车速度上限归一化后平方，系数 .2。
- 地形：基础 2，再乘 terrain_scale=.5；坡度按行驶坡度归一化，不可通行度按 .75 归一化。
- 未知地形软成本 20，同时仍受未知区域硬约束。
- 邻车：预测安全半径外加 .2 m 的软风险带，风险按 .2 m 归一化后平方，基础 8 再乘 neighbor_scale=.5；硬安全半径独立复核。

初始化引导速度使用目标距离/8 s，例如 2 m 目标约 .25 m/s；它是采样初值，不是另一条永久速度上限。安全复核包含延迟及完整刹车，滚动 MOVE 不要求在局部目标位置停稳。

### 10.2 速度承诺与安全参数

| 参数 | 实际值／行为 |
|---|---|
| 承诺版本 | `braking_clearance_lease_v3`；学习路径仍保留完整时域的间距限速 |
| 学习路径请求速度界 | `min(vmax, .8*max(最近合法邻车净空,0)/(8+1+.5), braking_clearance_limit)` |
| 当前控制保护 | 下发 cap 至少覆盖当前绝对速度；降速经受限刹车，不瞬间跳低 |
| 承诺覆盖时间 | 8+.5+2×1＝10.5 s |
| Safety | `footprint_discrete_cbf_upstream_v1`，OSQP 1.0.4，无 safety slack |
| 线性化／控制模式 | 最多 3 轮线性化，每轮 3 个执行器凸模式 |
| QP tolerances／iterations | eps_abs=eps_rel=1e-6，max_iter=4000 |
| 其他 QP 设置 | adaptive_rho=False、polishing=True；控制偏差按 [.04,.03] 归一化 |
| 扫掠／CBF | 控制步内 9 个姿态采样；终点 barrier 至少保留起点 .8 倍，另做完整非线性复核 |
| 无可行安全控制 | 刹车及安全处理保持有效，不因策略选择 MOVE 而绕过 |

第 1 级理想初态最近有效圆净空 1.2292–3.3505 m，代入完整时域项得到 **.1035–.2821 m/s**。这是承诺请求界的几何估算；实际 cap 还取决于历史承诺、当前速度、合法观测与制动约束，实际执行速度可更低。它解释了为什么按顶速或按 `rho/5 s` 估算到达时间会偏差很大，不能由此直接决定放松安全界。

## 11. 预算、评估、保存与恢复

| 参数 | 上一轮实际协议 |
|---|---|
| 训练预算 | 最多 2,000,000 实际底层环境步；4 辆车不重复计数；新一轮预算尚未确定 |
| 物理计费 | 包含终止步和无效学习样本实际消耗的步，排除已结束环境空转 |
| 预留 | 发起向量决策前预留 16×25＝400 步；不足则退出，不截断已承诺动作 |
| 每批名义物理步 | 16 决策×16 环境×25＝6,400；提前终止使实际步数更少 |
| 独立评估频率 | 初始化、每跨越 200,000 训练步、正常预算结束 |
| 评估上限／最终预留 | 250,000／100,000 底层步，独立记账；按回合最大步预留 |
| 每面板 | 4 个固定未见初态，各运行随机／均值策略至该级完整回合结束 |
| 评估种子 | 环境 104729、策略 104759；不扰动训练 RNG，不进入 PPO |
| 初始化面板 | 第 0、1 级 |
| 中间面板 | 第 0 级＋当前级 |
| 正常最终面板 | 第 0 级＋当前级＋第 6 级，去重 |
| 收敛筛选 | 3 个不同训练步检查点，停车级和第 6 级在随机、均值下均至少 3/4 成功 |
| checkpoint | 初始化、前两批、每 5 批、晋级／评估／结束；中央学习备份＋4 owner 分片 |
| 恢复身份 | 完整配置／源码／时间语义／课程／优化器／计数校验；不支持跨语义直接继承完整学习状态 |
| 正式入口限制 | 当前写死 A1、seed23、4×4、K512、CPU/skrl、100万或200万步等；改变这些需明确调整入口，而非仅修改 YAML |

上述旧预算和旧入口不构成下一次启动授权。用户停止覆盖了此前继续运行安排。

上一轮最终 1,614,642 步、264 批／1,320 轮 PPO、67,584 团队决策，评估 21,853 步；第 0→1 级在 113,843 步发生，其后未再晋级。课程内本级窗口 16/40，仍低于 32/40。未使用的 385,358 步已取消执行。

最近独立评估（1,602,339 步）为：

| 面板 | 成功 | 平均 HOLD 概率 | 平均最大车距缩小 | 平均执行速度 |
|---|---:|---:|---:|---:|
| 停车随机 | 4/4 | .7931 | −.0618 m | .00525 m/s |
| 停车均值 | 4/4 | .7915 | −.0028 m | .00077 m/s |
| 近距随机 | 2/4 | .7945 | .6918 m | .01592 m/s |
| 近距均值 | 0/4 | .7947 | .0075 m | .000067 m/s |

这里是完整回合结果，**不是统一 20 s 指标**。停车级已经在成功几何区域，车距略增不直接否定其停车成功。固定四例仅作工程筛选，不能替代多训练种子统计。用户中止后没有补做最终远距评估，不能报告为远距失败或成功。

## 12. 易误读的默认值、覆盖项与下一轮待定清单

| 文件表面值／标签 | 本轮实际解释 |
|---|---|
| profile `runtime_enabled:false / training_paused:true` | 设计 profile 自身不是运行入口；正式授权启动时 session 配置才释放，当前进程已停止 |
| environment `num_envs:4` | 单 owner 环境数；全局 16，不能再除／乘错计步 |
| profile 出生层级 20–60／60–150／150–300 m | 正式课程覆盖了通用出生器，实际使用第 4 节七级课程 |
| profile `initial_test_speed_mps:1` | 正式课程实际 .1 m/s |
| profile `high_level_period_s:1` | 重规划周期；策略决策周期实际 5 s |
| profile MPPI noise .25／.10 | 实际 overrides .375／.15 |
| reward 类默认 gamma=.99、lambda=.95、linear | 正式注入派生底层折扣及 log1p；不是这组默认值 |
| `scope:probability_contract_only / training_eligible:false` | 概率契约局部元数据；真实环境元数据为 training_eligible=true |
| policy `synthetic_t04_layout_not_new_scale_environment` 标签 | 历史诊断命名；真实 collector 使用 actual dense private sensor，不能据标签断言在用合成训练观测 |
| `leader_execution_version` 等继承标签 | 不能推断 A1 有脚本 leader 提供方向教师；学习路径由 Actor 产生目标 |
| 质量、摩擦及 sensitivity 列表 | 不是已执行的域随机化或敏感性扫参；本轮无质量／摩擦随机化训练 |

建议用户按以下顺序审定。此表列决策内容，不预设必须提高某个超参数：

1. **任务与出生分布**：停车级保留多少；主靠拢训练从什么半径开始；是否保持近似方阵和向心初始朝向，是否需要分散／断联初态。
2. **课程曝光和停止条件**：32/40 晋级是否保留；单级最多花多少实际步；卡级后是结束报告还是使用事先批准的难度混合。不要再让整轮预算隐式耗在一个级别。
3. **动作与时间匹配**：结合实际承诺速度，审定 4 m 目标范围、5 s 决策、8 s 规划、各级时限；物理和安全约束暂不随学习失败而放松。
4. **奖励与信用**：先明确本轮是否调整奖励系数，还是调整 gamma/lambda/rollout；区别几何进步、驻留成功、超时成本和价值估计，不把所有参数一起改变后失去比较依据。
5. **优化与探索**：lr、clip、5 epochs、独立梯度裁剪目前没有配置失配证据；门控探索与连续噪声分开考虑。running value normalization 的跨课程表现应看原单位误差。
6. **固定评估与预算**：指定停车、近／中／远距面板及中途检查点，保证能观测退化和卡级；明确新预算、初始化方式和提前结束条件，再形成下一版有效配置。

本次只新增审查文档与入口说明，保留原 YAML、源码、checkpoint 和原始实验数据。相关历史：[1 km 课程长训练及停止记录](t11_1km_curriculum_long_training_2026-10-19.md)。
