# 六项审定v2课程正式训练启动

日期：2026-10-20。用户在[六项审定](t11_six_point_review_2026-10-20.md)后明确要求“启动训练”，据此释放本轮暂停。旧训练、旧队列与原checkpoint保持原状。

## 2026-10-21结束核验：第2级预算用满，未通过收敛筛选

10月20日19:26:39完成最后一次更新，19:26:43保存完整checkpoint并开始最终评估，22:53:21写出最终评估与summary。训练及定期评估至最后更新约6小时13分，含最终评估的全流程约9小时40分。10月21日核验时，主进程1477449、四owner1477474–1477477和资源跟踪1477473均已退出；日志无Traceback，无failure产物。

最终实际250,000底层环境步、10,000条有效团队决策、20批更新／100轮PPO。当前课程始终为第2级，窗口0/40、晋级历史为空；停止原因为 `stage_budget_exhausted`。外层200万预算还剩175万，但本级25万上限已触发，不续训或强制晋级。独立评估实际82,400步，其中最终71,600步，均未进入PPO；前置无学习准入1,304步另计。

训练结束185个回合：第0级停车复习18、第1级7–12 m复习19、第2级12–30 m共148，全部 `task_deadline`，无成功。最后批次HOLD比例35.11%，最终固定评估HOLD概率约35.2%，没有旧训练高比例HOLD退化。

### 固定20秒趋势：初始化与最终策略

每组使用相同4个初态，已逐项核对初始最大车距及离散度一致；下表是最大车距平均缩小量，正值表示靠拢。半径表示出生布局尺度，采样执行对应历史日志 `sample`，确定性执行对应 `mean`。

| 出生半径 | 采样执行：初始化→25万步 | 确定性执行：初始化→25万步 |
|---|---:|---:|
| 7–12 m | 3.474→3.190 m | 5.182→5.539 m |
| 12–30 m | 4.254→4.132 m | 10.771→10.326 m |
| 60–120 m | 4.695→4.548 m | 5.910→7.059 m |
| 200–300 m | −1.674→−0.979 m | −3.833→−3.526 m |

12–30 m两种执行方式均未优于初始化；60–120 m确定性执行改善，200–300 m发散略减但平均仍未净靠拢。这些短程结果不替代完整回合成功评价。

### 最终完整回合评估

| 出生组 | 回合时限 | 采样执行成功 | 确定性执行成功 | 最大车距平均缩小：采样／确定性 |
|---|---:|---:|---:|---:|
| 已满足集合几何的停车组 | 30 s | 0/4 | 0/4 | −0.933／−1.742 m |
| 7–12 m | 180 s | 0/4 | 0/4 | −6.373／3.112 m |
| 12–30 m | 300 s | 0/4 | 0/4 | −7.468／−16.694 m |
| 200–300 m | 1200 s | 0/4 | 0/4 | 5.260／−118.350 m |

32个完整评估回合全部超时，最终评估未记录几何重叠或最小间距违规。12–30 m组20 s内靠拢，但300 s后平均最大车距反而扩大，说明本轮策略尚未形成持续靠拢并驻留的闭环能力。`stable_convergence_screen_passed=false`；这是本配置、单种子与阶段预算下的结果，不据此确定下一项算法修复。

### 最后恢复点与证据

`checkpoints/update_000020.pt` 的中央文件及四owner分片SHA核验通过；内部计数与最终summary一致，Adam参数步数均100。20批第一轮概率重放ratio均精确为1，所有已记录epoch数值有限。最新Critic解释方差约0.107，数值检查通过不等于价值拟合或策略成功。

原始证据均位于本run：

- `metrics/summary_000020.json`：停止原因、训练及独立评估计数。
- `metrics/update_000020.json`及前19批：课程、结束回合、实际PPO更新。
- `metrics/evaluation_initial.json`、`evaluation_step_0102400.json`、`evaluation_step_0204800.json`、`evaluation_final.json`：固定初态及完整回合对比。
- `metrics/completion_verified_20261021.json`：本次只读核验汇总，含进程、checkpoint、Adam、全部回合及面板统计。

核验使用Python读取上述JSON、聚合 `update_*.json` 的结束回合，使用 `.venv_terrain_cuda/bin/python` 调用 `scripts/upstream_checkpoint.py::read_checkpoint` 校验中央文件与四分片，并检查 `/proc/<pid>` 确认进程退出。此次新增环境步和优化器更新均0；没有修改训练配置、启动额外评估或恢复训练。下方进度为历史记录。

## 16:36进度核验：12.8万步，更新链路正常，尚无完整成功

最新已落盘训练计数128,000实际底层步、5,120条有效团队决策、10批更新／50轮PPO；主进程及四owner继续运行。仍在第2级12–30 m主靠拢，当前级预算使用51.2%，晋级窗口0/40，尚未晋级。累计结束92回合：停车复习9、近距复习10、主靠拢73，全部为任务超时，无成功回合。

`checkpoints/update_000010.pt`（16:27:43保存）及四owner分片SHA核验通过。Adam各参数步数均50；Actor/Critic参数相对初始化的L2变化约.7675/.6757，证明发生了实际更新。10批的第一轮重放ratio均精确为1，所有已记录epoch数值有限；无failure产物。最新批次HOLD比例35.69%，独立评估HOLD概率约35.1%，未出现此前约80%的HOLD退化。

约10万步固定评估（实际检查点102,400步）与初始化几何一致。表中为20 s平均最大车距缩小，正值表示靠拢：

| 面板 | 随机：初始化→10万步 | 均值：初始化→10万步 |
|---|---:|---:|
| 级1，7–12 m | 3.474→3.520 m | 5.182→5.457 m |
| 级2，12–30 m | 4.254→3.918 m | 10.771→10.636 m |
| 级4，60–120 m | 4.695→4.393 m | 5.910→6.023 m |
| 级6，200–300 m | −1.674→−.383 m | −3.833→−3.398 m |

主靠拢随机／均值均略低于初始化；最远组发散减轻但平均仍非净靠拢。近／中距离有方向性运动不能单独计为学习改善，初始化已有朝向先验。10万步面板是短程趋势，不包含新的完整停车评估；初始停车0/4不能冒充10万步后的停车测试。

独立评估累计7,600步。采样每批约16.6–18.0分钟；本次检查进程树PSS约8.04 GiB、swap为0。现有证据支持“运行与优化链路正常，尚无明确学习收益或完整成功”，按已授权的阶段25万步边界继续，不改参数、不扩训。

核验产物：`metrics/progress_verified_20261020_1630.json`（实际核验时间16:36），另见 `metrics/update_000010.json`、`metrics/evaluation_step_0102400.json`。下方13:31“首次更新尚未验收”为历史启动状态，已由本节覆盖。

## 实际启动与范围

- 启动时间：2026-10-20 13:13:19 +08:00；主进程PID `1477449`。
- run：`outputs/runs/upstream_1km_curriculum/a1_seed23_v2_stage250k_20261020_01/`。
- A1、seed23、四owner×四环境、全新Actor/Critic与Adam；不继承原高HOLD模型。
- 物理配置：`configs/experiment/upstream_1km_curriculum_v1.yaml`；课程：`configs/training/upstream_1km_curriculum_v2.yaml`。
- 从第2级12–30 m主靠拢开始，20%低级复习；每级最多250,000实际底层步，卡级保存结束，不强制晋级。
- 总预算参数为2,000,000实际底层步；受五个剩余主阶段各25万上限约束，实际训练最多1,250,000步，可能更早结束。四辆车不重复计数；向量决策预留400步。
- 独立评估上限250,000步，最终预留71,600；按200万外层上限的固定面板最坏预留235,200步。前置无学习检查另计1,304步。
- 保留车辆、MPPI和安全约束、奖励权重；5 s策略决策，gamma=.997、lambda=.99（1 s参考），32决策序列、5轮PPO、初始HOLD约.35。
- 数值／采样错误保存现场停止；不自动重试、追加训练或扩充预算。平地Proxy学习不等于PhysX验收。

## 启动前证据

23项不同契约检查及最后3项预算／评估复核沿用[审定记录](t11_six_point_review_2026-10-20.md)，本次核对所有记录的受影响源码SHA均未变化，不重跑无关测试。启动前GPU可用、无旧正式训练／队列进程，磁盘可用约182 GiB、内存可用约10 GiB。

新配置的无学习实际准入位于：`outputs/runs/upstream_1km_curriculum/prelaunch_v2_20261020_01/`。

| 检查 | 结果 |
|---|---|
| 静态几何／奖励／势差 | 通过；`metrics/static_audit.json`绑定完整v2课程 |
| 真实停车HOLD | 4/4成功，104底层步 |
| 第3级30–60 m实际配对 | HOLD／向心／向外各400步，四个相同初态 |
| 四例奖励顺序 | 每例向心 > HOLD > 向外 |
| 向心20 s最大车距缩小 | 22.804、22.613、21.890、22.533 m |
| 向外20 s最大车距缩小 | −16.708、−16.159、−16.945、−16.144 m |
| 总计 | 1,304实际底层步，0优化器更新，准入通过 |

配对是无学习干预诊断，不进入PPO，也不作为策略已学会靠拢的证据。前置检查不是训练预算的一部分。

## 初始化核验

13:14:38核对 `checkpoints/update_000000.pt` 与四owner分片SHA通过；中央课程从第2级开始，训练计数0、网络与Adam全新，HOLD bias与.35目标一致，32决策、5轮PPO和派生折扣与配置一致。241个源码SHA与启动快照一致。

四owner进程为 `1477474–1477477`，资源跟踪进程 `1477473`。初始化证据为本run下 `metrics/initialization_verified.json`；这仅证明初始化和完整保存，不是首批更新验收。启动先执行固定初态评估，再进入正式采样。

## 13:24：初始化评估完成，进入正式采样

`metrics/evaluation_initial.json` 已保存，独立评估实际消耗4,400步，未进入PPO。20 s趋势面板平均最大车距缩小如下（正值为靠拢）：

| 固定面板 | 随机策略 | 均值策略 |
|---|---:|---:|
| 级1，7–12 m | 3.474 m | 5.182 m |
| 级2，12–30 m | 4.254 m | 10.771 m |
| 级4，60–120 m | 4.695 m | 5.910 m |
| 级6，200–300 m | −1.674 m | −3.833 m |

初始HOLD概率约.35；停车完整回合随机／均值均0/4。上方停车干预4/4证明任务可执行，这里的0/4反映初始策略尚不会停车，两者口径不同。该基线说明主靠拢的朝向先验有效，但最远随机朝向和停车仍需学习；不能把初始化靠拢直接计为学习收益。

## 13:31：启动状态交付

主进程及四owner持续运行，未发现 `failure_*.json`；初始化完整checkpoint、源代码身份与4,400步独立基线已核验。当前处于第1批采样，尚无完整 `collection_*.json` 或 `update_*.json`，**首次PPO更新与更新后的checkpoint尚未验收**，不能把启动成功写成首批学习已通过。13:26进程树PSS约6.81 GiB、swap为0；后续检查四owner仍在计算、GPU有负载。

机器可读启动状态为 `metrics/launch_verified.json`，明确 `first_ppo_update_verified:false`；进程在后台继续按既定预算运行。之后检查进度应先读取实际 `collection_*.json`、`update_*.json`、checkpoint及failure产物，不从运行时长估算已训练步数。

## 命令及产物

本次无学习准入命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv_terrain_cuda/bin/python -u scripts/audit_upstream_curriculum.py \
  --config configs/experiment/upstream_1km_curriculum_v1.yaml \
  --curriculum-config configs/training/upstream_1km_curriculum_v2.yaml \
  --output outputs/runs/upstream_1km_curriculum/prelaunch_v2_20261020_01
```

实际训练命令（作为独立session后台进程运行）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv_terrain_cuda/bin/python -u scripts/run_upstream_t11.py \
  --config configs/experiment/upstream_1km_curriculum_v1.yaml \
  --curriculum-config configs/training/upstream_1km_curriculum_v2.yaml \
  --run-id a1_seed23_v2_stage250k_20261020_01 \
  --representation A1 --mode train --seed 23 --budget 2000000 \
  --env-owners 4 --num-envs 16 --owner-cpu-workers 2 \
  --samples 512 --trainer-backend skrl --policy-device cpu \
  --release-training-pause \
  --readiness outputs/runs/upstream_1km_curriculum/prelaunch_v2_20261020_01/metrics/summary.json
```

完整argv、环境变量、PID和时间在 `logs/launch.json`。标准输出为 `logs/train.log`，与前置run的 `logs/train_launch.log` 硬链接到同一文件；没有启动第二个训练进程。

重点产物：`config/effective_configuration.json`、`metrics/evaluation_initial.json`、`metrics/collection_*.json`、`metrics/update_*.json`、`metrics/failure_*.json`、`checkpoints/update_*.pt`。每5批、晋级／评估及退出时保存，初始化和前两批也保存。统计以实际底层步为准，短程趋势面板与完整成功面板分开。

以上命令是实际执行记录，已存在的run不能覆盖重跑。正常暂停通过主进程SIGTERM，等待采样／更新边界完整保存；本次没有自动续跑旧run或启动其它动作方案。
