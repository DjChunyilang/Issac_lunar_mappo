# 百万步续训故障修复与恢复

最新进展：[SAT三轮配置从零百万步训练](t11_mppi_sat_long_training_2026-10-25.md)已于20:18启动，全新网络／Adam。启动审计修复轮数覆盖问题，41项检查通过；本页旧结果保持。

最新后续：[MPPI查询合并与三轮配置](t11_mppi_query_optimization_2026-10-25.md)已完成。整段耗时减少38–45%，短目标10／10、0失败；6,000诊断步、0学习，采用显式SAT配置。

最新测试：[两种碰撞算法对照](t11_collision_algorithms_2026-10-25.md)已完成。栅格SAT与矩形—圆均保持10／10个短目标到达、0失败；SAT沿用当前感知，整段耗时比416圆盘减少18–21%。圆方案本次使用真值，只作对照。共3,600诊断步、0学习更新，生产源码尚未切换。

最新进展：[MPPI执行修复与调参](t11_mppi_execution_repair_2026-10-25.md)已完成。主要原因是规划时车身覆盖过大，实际离坑约1米仍可能被判侵入；在原MPPI内修正覆盖并调参后，同方向0.5米目标从0／10到达提高到10／10，结束误差均小于10厘米，无实际失败。原目标平均进展0.003→0.652米，但仍含无法放下整车的目标。共5,400诊断步、0学习更新，没有启动新训练；下文百万步原结果保持。

后续诊断：[为什么活下来了，却没有集合](t11_crater_reward_detailed_analysis_2026-10-25.md)进一步分析方向选择、MOVE低推进、地形失败和奖励分项；0新增环境／学习步。下文保留本轮结束统计与恢复过程。

随后完成[同保存态目标执行对照](t11_target_execution_probe_2026-10-25.md)：1,200诊断步、0学习更新。转向队友后2／10辆MOVE车明显推进，其余仍停滞，下一步检查局部规划候选及其筛选／选择。

日期：2026-10-24至2026-10-25，Asia/Shanghai。用户要求检查训练，有问题则恢复；本次沿用原新增100万步授权。10月25日补齐一次性结束总结。

**10月25日06:26:44完成最终评估，按预算正常结束；未形成持续靠拢或完整任务成功。** 实际新增学习采样999,812／1,000,000步（包含原故障丢弃的12,794步），独立评估53,860／88,800步。最终60秒sample／mean各4例均存活到观察截止，但平均最远车距仍扩大3.199／8.507 m；1500秒完整面板各0/4成功，合计5例地形失败、3例到期未完成。最终完整checkpoint、4个owner分片和TensorBoard events已保存并核验。

此前入口把一条应由mask排除的环境无效记录当成整批损坏，原批次没有NaN或Inf。10月24日17:26:50修复后从62,628步完整状态恢复，剩余974,449步；17:30:50短暂启动检查通过后停止模型持续监控。本页下半部分保留当时修复、授权与启动证据。

## 结束结果与判断

本轮回答的问题是：保持奖励、环境和学习算法，增加约100万实际学习底层步后，能否形成安全靠拢并完成集合？结果是**短窗末态仍平均扩散，完整任务0/8成功**。mean的60秒扩散幅度比训练前小，sample变化很小；这不等于已学会持续靠拢。中途失败率也有波动，不按回报增加或最终60秒无失败宣布改善／收敛。

`summary_000082.json`记录`stop_reason=budget_complete`，无`failure*.json`。累计82批／410轮PPO；相对原5万步run新增78批／390轮，其中原run第5批继承一次，恢复后执行第6–82批共77批／385轮。恢复后第6、18批各有1条`masked_environment_invalid_transitions`，其余有效记录继续学习，原入口误停未再发生。

本次总结只读取现存JSON、checkpoint文件及events，新增环境步／优化更新均为0；没有运行新测试、恢复训练或安排下一次调用。原5万步run及原百万步故障run未改动。

### 训练前后：相同4起点、60秒观察窗口

训练前是**49,871步策略**的继承面板，训练后是本分支1,036,889步策略；不是在恢复点62,628步重新评估。每种模式都使用同4起点。`dmax_reduction = 初始Dmax − 末态Dmax`，正数才表示最远两车距离缩小；负数表示扩大。存活到60秒是观察截止（删失），不代表完成集合。

| 模式 | 窗口平均观察时间：前→后 | 窗内失败：前→后 | 平均dmax_reduction：前→后 | 正缩距起点数：前→后 | 平均实际速度：前→后 |
| --- | --- | --- | --- | --- | --- |
| sample | 60.00→60.00 s | 0/4→0/4 | −3.734→−3.199 m | 1/4→2/4 | 0.193→0.167 m/s |
| mean | 60.00→60.00 s | 0/4→0/4 | −22.963→−8.507 m | 0/4→2/4 | 0.315→0.326 m/s |

前后各8例均为窗口截止、0成功。mean平均扩散减少14.456 m，sample减少0.535 m，但两者均值仍为负；部分起点短暂靠拢尚未转化为完整任务成功。

### 中途趋势：10个定期60秒面板

下表每格按sample／mean顺序；恢复后各行的学习步是分支计数，实际历史累计另加12,794步；继承初始化仍对应49,871步。各行仍为同4起点，失败时的缩距是提前终止时值，不能把更短观察时间下的扩散较小单独解释为改善。

| 分支学习步 | 平均观察时间（s） | 窗内失败数（各／4） | 平均dmax_reduction（m） |
| --- | --- | --- | --- |
| 49,871（继承初始化） | 60.00／60.00 | 0／0 | −3.734／−22.963 |
| 100,755 | 60.00／44.60 | 0／2 | −8.751／−5.963 |
| 202,645 | 60.00／54.90 | 0／1 | −4.398／−15.111 |
| 304,550 | 60.00／48.75 | 0／2 | −6.500／−18.448 |
| 406,315 | 60.00／50.80 | 0／1 | −2.589／−12.037 |
| 508,344 | 57.10／46.05 | 1／2 | −2.345／−7.124 |
| 610,399 | 59.85／44.05 | 1／2 | −2.764／−8.422 |
| 712,334 | 60.00／44.45 | 0／2 | −7.982／−12.330 |
| 801,332 | 60.00／38.75 | 0／3 | −4.046／−12.247 |
| 903,419 | 60.00／56.90 | 0／1 | −7.660／−18.455 |
| 1,005,457 | 60.00／55.05 | 0／1 | −4.322／−11.409 |
| 1,036,889（最终） | 60.00／60.00 | 0／0 | −3.199／−8.507 |

sample仅两次定期面板出现1/4失败；mean全部定期面板出现1–3/4失败，最终回到0/4。**所有面板的平均缩距均为负，没有持续转正趋势。** 短窗失败均为`terrain_failure`，涉及`driving_slope`或`terrain_step`。曲线见本run `figures/long_training_progress.png`及同名PDF，横轴为本次授权实际新增采样步，已补计丢弃采样。

### 完整任务：1500秒固定面板，单独统计

完整面板与短窗使用同4起点，但完整终止结果独立报告；各模式不是额外4个不同起点。任务成功要求实际队形、停车与连续驻留条件成立；活到1500秒却未完成属于`task_deadline`失败，不是观察删失。完整面板8例均已终止、无删失。

| 模式 | 完整成功 | 地形失败 | 到期未完成 | 平均存活时间 | 平均dmax_reduction |
| --- | --- | --- | --- | --- | --- |
| full_far_sample | 0/4 | 2/4 | 2/4 | 904.65 s | −34.330 m |
| full_far_mean | 0/4 | 3/4 | 1/4 | 467.10 s | −6.232 m |

| 模式 | 起点（env_id） | 存活时间 | 结束原因 | dmax_reduction |
| --- | --- | --- | --- | --- |
| sample | 0 | 1500.0 s | `task_deadline`：到期未完成 | −57.406 m |
| sample | 1 | 513.0 s | `terrain_failure / impassable_footprint`：足迹不可通行 | −34.110 m |
| sample | 2 | 105.6 s | `terrain_failure / driving_slope`：行驶坡度超限 | −18.503 m |
| sample | 3 | 1500.0 s | `task_deadline`：到期未完成 | −27.303 m |
| mean | 0 | 1500.0 s | `task_deadline`：到期未完成 | +15.435 m |
| mean | 1 | 63.4 s | `terrain_failure / driving_slope`：行驶坡度超限 | −12.856 m |
| mean | 2 | 113.8 s | `terrain_failure / driving_slope`：行驶坡度超限 | −49.088 m |
| mean | 3 | 191.2 s | `terrain_failure / impassable_footprint`：足迹不可通行 | +21.580 m |

sample完整末态4例全部扩散；mean有2例缩距，但一例到期未完成、一例地形失败。终止时Dmax仍约403.5–537.1 m，距离任务12 m队形条件很远。存活、局部缩距和完整成功分别判断。本轮没有训练前完整面板，不作完整任务前后提升结论；仅seed23、固定4起点，无未见图、多种子或Isaac／PhysX验收。

### 已结束训练回合：辅助现象

按`update_000005.json`至`update_000082.json`各计一次，共471个已结束训练回合：400次地形失败、69次到期未完成、2次环境无效、0成功，平均时长407.57 s。训练批第5–14批与第73–82批分别结束67／51回合，平均寿命317.98→449.56 s，平均原始回报35.42→54.07，按批平均HOLD比例36.92%→43.89%。

这些回合按结束先后进入统计，长回合可能跨批，预算结束时仍活跃的回合没有计入；存在删失和结束先后偏差，也不是固定起点配对。寿命或回报增加不能代替上面的固定评估，不能据此宣称安全靠拢改善。地形子原因按车记录可能同回合多次出现，不能把子原因计数当作互斥回合数。

### 最终预算与产物核对

| 账目 | 实际消耗／上限 |
| --- | --- |
| 从49,871步起授权新增的实际学习采样 | **999,812／1,000,000步**，剩188步未用 |
| 恢复后实际采样 | 974,261／974,449步 |
| 本分支累计计数 | 1,036,889／1,037,077步 |
| 含原丢弃采样的历史累计物理消耗 | **1,049,683／1,049,871步** |
| 独立评估 | **53,860／88,800步**，剩34,940步未用 |
| 评估组成 | 继承初始化2,400＋10个定期21,625＋最终29,835步 |
| 最终评估组成 | 60秒窗口2,400＋完整任务27,435步 |

实际新增＝1,036,889−49,871＋12,794＝999,812；故障采样照常扣账。原初始化评估及第5批metrics与父run逐字节一致，均只计一次。剩余步数不自动续用。

最终`checkpoints/update_000082.pt`为`t11_owner_shards_v1`，中央7,416,573字节及4个owner分片（41,151,091／31,330,227／40,669,107／45,880,307字节）均与保存的SHA256和大小一致。它是完整保存点，但不是已收敛或任务成功checkpoint。

TensorBoard保留11个events文件：采样曲线2,447点，首次采样点63,028、末点1,036,889；学习曲线78点，包含恢复时投影的第5批62,628步，最后为第82批。初始化评估绑定49,871步；最终短窗、完整任务与53,860评估步统计均已写入events，最终写入时间为10月25日06:26:44。恢复边界62,628、丢弃采样12,794单独记账，不把曲线横轴当作全部历史消耗。本次没有启动或切换TensorBoard服务；日志目录与回放命令见下文。

一次必要文件核对的证据为本run `metrics/completion_artifact_verification.json`；核对脚本和日志位于准备run `config/verify_completed_artifacts.py`、`logs/completion_artifact_verification.log`。当时未发现更晚同族run或运行中的正式入口，当前状态据此同步。纯JSON自动统计已成功（`analysis_returncode=0`），无需修改汇总或生产代码。

### 证据与复现命令

本节所有短路径均相对本训练run，除非明确标注准备run：

- 原始结束／评估：`metrics/summary_000082.json`、`metrics/evaluation_initial.json`、`metrics/evaluation_step_*.json`、`metrics/evaluation_final.json`；完整面板须读取`results.full_far_sample/full_far_mean.full_episode_evaluation=true`，混合文件顶层的`false`不覆盖这两个子面板。
- 汇总与文件核对：`metrics/long_training_review.json`、`metrics/completion_artifact_verification.json`；展示曲线：[PNG](../../outputs/runs/upstream_1km_fixed_far/a1_seed23_crater_reward_1m_recovered_20261024_01/figures/long_training_progress.png)、[PDF](../../outputs/runs/upstream_1km_fixed_far/a1_seed23_crater_reward_1m_recovered_20261024_01/figures/long_training_progress.pdf)。
- 启动配置／实际命令：`config/launch.json`、`config/training_command.txt`、`config/tensorboard_command.txt`；训练日志`logs/console.log`；完整状态`checkpoints/update_000082.pt`及`checkpoints/update_000082.owners/`。
- 准备run：`metrics/automatic_summary_status.json`、`logs/automatic_metrics_summary.log`；本次进入时为`one_time_model_summary_starting`，本地等待脚本在本会话退出后记录最终状态，不在会话内等待或再调用模型。

自动统计与本次文件核对实际使用的命令（均不运行环境；统计／核对产物已存在，无需重复执行）：

```bash
.venv_terrain_cuda/bin/python outputs/runs/upstream_1km_fixed_far/crater_reward_recovery_20261024_01/config/summarize_completed_training.py
.venv_terrain_cuda/bin/python outputs/runs/upstream_1km_fixed_far/crater_reward_recovery_20261024_01/config/verify_completed_artifacts.py
```

本轮授权训练、恢复验证和一次性总结均已完成。后续问题仍是如何在长时存活中形成持续靠拢并减少实际地形失败；本次不追加实验或预算。

## 要解决的问题与结束条件

目标是让合法的 `environment_invalid` 终端只排除自身，其他有效记录正常学习；数值损坏、终端标记矛盾仍需停止。最小验收为重现原故障、验证无效记录不影响有效学习目标、完整恢复并确认采样与events继续增长。完成启动检查后停止模型监控，训练退出后由本地脚本触发一次总结。

## 为什么停了，改了什么

[原百万步续训](t11_crater_reward_long_training_2026-10-24.md)于00:52:35报 `Invalid or nonfinite learning data`。保存的第6批共有512条决策记录：511条有效，1条 `environment_invalid/unknown_footprint`。后者表示环境在车身足迹处缺少有效地形信息，不能用这条记录学习；它不是已确认的碰撞，也不是数值爆炸。

学习器原本已用 `learning_valid` 从Actor、Critic、价值归一化与GAE中排除该记录，并切断跨越它的回报递推。但训练入口额外要求全部记录有效，两者冲突，使整批被拒绝。

本次只修改 `scripts/upstream_curriculum_experiment.py` 的批次检查：允许“原因是环境无效、已终止、禁止bootstrap”的无效记录通过批次入口，继续由既有mask排除；所有输入和动作概率仍检查有限数，矛盾的终端标记仍报错。更新指标新增 `masked_environment_invalid_transitions`。奖励、地形规则、网络、优化器和训练配置保持原值。

## 验证证据

准备目录：`outputs/runs/upstream_1km_fixed_far/crater_reward_recovery_20261024_01/`。

- 19项针对性测试全部通过，覆盖批次检查、无效目标排除及已有评估调度。日志：`logs/recovery_tests.log`、`logs/recovery_tests.xml`。
- 用原始失败批次复现：旧检查拒绝，新检查通过；511条有效、1条排除，优势与目标全部有限。把被排除记录的奖励改为10亿，有效目标逐值不变。证据：`metrics/saved_batch_reproduction.json`。
- 完整状态迁移：4个owner分片字节完全一致；中央训练器1,816,180个张量值及2,828个标量相同。只更新已审查源码身份及预算元数据，原恢复身份检查保留。证据在新run `metrics/recovery_migration.json`。
- 上述复现、测试和迁移均未推进环境或更新优化器。原环境／图集未改变，复用此前实际100步准入证据，见 `metrics/recovery_readiness.json`。

验证命令：

```bash
PYTHONPATH=scripts:source/lunar_rover_tasks .venv_terrain_cuda/bin/python -m pytest -q \
  tests/test_upstream_learning_batch_admission.py \
  tests/test_upstream_value_normalization.py::test_gae_is_raw_actor_unchanged_and_invalid_targets_excluded \
  tests/test_upstream_bounded_evaluation.py \
  --junitxml=outputs/runs/upstream_1km_fixed_far/crater_reward_recovery_20261024_01/logs/recovery_tests.xml
```

## 恢复位置和预算

新run：`outputs/runs/upstream_1km_fixed_far/a1_seed23_crater_reward_1m_recovered_20261024_01/`。原run及故障批次保留，不覆盖。恢复 `checkpoints/update_000005.pt` 与全部4个owner分片，包含网络、Adam、价值归一化、GRU、环境、规划器与随机状态。累计已完成5批／25轮PPO。

| 账目 | 数值 |
| --- | --- |
| 原授权新增学习 | 1,000,000步，从短训49,871步起算 |
| 故障前实际新增消耗 | 25,551步 |
| 最后完整保存点 | 62,628步，新增12,757步已完成PPO |
| 未保存进度中已消耗的采样 | 12,794步，未进入PPO，但仍扣预算 |
| 本次恢复后最多再采样 | **974,449步** |
| 新run累计计数上限 | **1,037,077**＝62,628＋974,449 |
| 原独立评估授权 | 88,800步，已用2,400步，剩余86,400步 |

新run横轴从62,628继续，回退的12,794步另记账：新增授权的实际消耗＝新run `actual_base_steps` −49,871＋12,794。达到新分支上限时恰好消耗100万新增步，不把丢弃的采样当作免费。

初始化60秒评估直接继承原run的2,400步结果，**对应49,871步策略**，并非62,628步的新评估；不重复执行或扣账。此后按每跨10万累计分支步执行评估，最终60秒和1500秒完整任务面板均已完成，结果见上文。

## 启动、监控与自动总结

17:26:50启动：训练PID11463，TensorBoard PID11464。精确命令、环境变量和PID保存在新run及准备目录的 `config/launch.json`、`config/training_command.txt`、`config/tensorboard_command.txt`。配置仍为 `upstream_1km_far_crater_reward_1m_v1.yaml`，A1／seed23，4 owner×4环境，CPU策略／PPO、CUDA MPPI512×2。

TensorBoard：[http://127.0.0.1:6006](http://127.0.0.1:6006)，仅加载新run。日志目录及回放命令：

```bash
.venv_terrain_cuda/bin/tensorboard \
  --logdir outputs/runs/upstream_1km_fixed_far/a1_seed23_crater_reward_1m_recovered_20261024_01/tensorboard \
  --host 127.0.0.1 --port 6006 --reload_interval 5
```

横轴为本分支实际学习底层步，62,628为恢复边界；评估步另计，丢弃采样按上表补记。旧events全部保留。启动核验结果写入准备目录 `metrics/startup_verification.json`。

实际启动观察240秒：4个owner均运行，12段采样累计到67,428步；events最大step同为67,428，文件由176,568增到178,200字节，HTTP200，无failure文件，源码与冻结快照完全一致。此刻尚未完成恢复后的第一批PPO，只证明完整恢复和采样／日志继续推进，不据此判断学习效果。

17:31:13已启用准备目录 `config/finish_and_summarize.py`，PID13361，状态确认为 `waiting_for_training_and_final_evaluation`：普通本地进程每60秒检查训练PID，等待期间不调用模型；训练及最终评估退出后先做本地统计，再启动一次独立 `codex exec --ephemeral` 总结。状态为 `metrics/automatic_summary_status.json`，模型日志在 `logs/automatic_codex_summary.jsonl`，最终文字落到新run `metrics/automatic_final_summary.md` 并回写本页。异常退出也总结，不自动重启或追加预算。本机与网络需保持可用。
