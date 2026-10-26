# 新奖励百万步续训

最新进展（17:26）：用户要求检查并恢复后，已确认并修复批次入口误停，另建run从62,628步完整恢复，续用剩余974,449步。见[故障修复与恢复](t11_crater_reward_recovery_2026-10-24.md)；以下保留原run结束时的记录。

日期：2026-10-24，Asia/Shanghai。承接[5万步短训](t11_crater_reward_short_training_2026-10-23.md)。用户明确授权从现有状态**新增100万实际学习底层步**，启动后只短暂监控，结束后自动触发一次总结。

**结束结论：00:52:35因学习数据有效性检查失败而异常退出，未完成百万步预算。** 新增实际采样25,551步，其中12,757步完成1批／5轮PPO，后续12,794步所在批次被拒绝；独立评估2,400步。仅有训练前60秒面板，未到首次10万步定期评估，最终短窗及1500秒完整任务评估均未执行。不能得出续训后的存活、靠拢或成功率改善结论；未重试训练。

## 本轮要回答的问题

上一轮短窗失败减少，但车辆平均仍在散开。这轮保持奖励、环境和学习算法，增加训练量，检查是否从“活下来”进一步学到“持续靠拢并完成集合”。低成功率、地形失败或未靠拢是实验结果，不自动追加预算或启动对照。

## 启动与预算

00:32:45启动，主进程PID2777507。run：`outputs/runs/upstream_1km_fixed_far/a1_seed23_crater_reward_1m_20261024_01/`。

| 项目 | 本轮安排 |
| --- | --- |
| 继承点 | 父run `a1_seed23_crater_reward_50k_20261023_01/checkpoints/update_000004.pt` 及全部4个owner分片 |
| 已继承学习 | 49,871底层步、4批／20轮PPO |
| 新授权学习 | 最多1,000,000底层步；累计 `--budget 1049871` |
| 并行与算法 | A1／seed23；4 owner × 4环境；CPU策略／PPO，CUDA MPPI512×2；32决策／5轮PPO |
| 训练任务 | 固定远距出生，1500秒回合；奖励持续成本×0.02，中坑轻罚保持 |
| 启动／定期评估 | 固定4起点，sample和mean各60秒；每跨过10万累计学习步，在批次边界冻结策略评估 |
| 最终评估 | 同样60秒面板，再加相同4起点、sample和mean各1500秒完整任务面板 |
| 独立评估预算 | 最多88,800底层步：启动2,400＋10次定期各2,400＋最终62,400 |
| 停止条件 | 用完新增预算或出现使运行／数据无效的错误；不自动重试训练 |

新配置为[upstream_1km_far_crater_reward_1m_v1.yaml](../../configs/training/upstream_1km_far_crater_reward_1m_v1.yaml)，只修改5个评估调度字段。实验入口仍为[upstream_1km_crater_classes_v1.yaml](../../configs/experiment/upstream_1km_crater_classes_v1.yaml)。`periodic_fixed_window_v1`使中途评估启用，最终`far_sample/mean`保存短窗，`full_far_sample/mean`保存完整回合；不新增未见图或多种子验收。19项相关检查通过，旧短窗和固定任务路径保持。

## 完整状态继承的证据

新run保留自己的manifest、源码包、checkpoint和events，原run不覆盖。仅变更评估元数据、预算及调度源码身份；生产恢复仍使用原严格检查，没有放宽校验。

迁移后重新读取落盘文件，逐值核对1,817,204个张量值、37,147,196个数组值及60,901个标量。网络、Adam、价值归一化、GRU、车辆／规划器、奖励累计和随机状态保持；更改位置完整列在本run `metrics/continuation_migration.json`。中央与4个owner的checkpoint校验和通过。迁移和验证新增环境步／优化器更新均为0。

准备run：`outputs/runs/upstream_1km_fixed_far/crater_reward_1m_preparation_20261024_01/`。`metrics/continuation_readiness.json`明确复用上一轮100步实际环境检查，仅对新调度做零步验证；不是重新执行100步。首次离线核对遇到字符串数组不支持`isnan`，已修正比较工具；失败日志和未完成产物保留于准备run，未进入训练。

首次启动00:25在训练器恢复校验处退出，0新增采样／学习步。原因是评估配置的派生`fingerprint`仍保留旧值；已在迁移工具中显式重算，生产校验未放宽。原启动产物归档于准备run `state/launch_failed_01/`。随后用真实四owner执行严格恢复，并导出全部状态逐值核对通过（`metrics/live_restore_verification.json`），0环境步／0优化器步，再于00:32启动。首次失败时等待脚本提前触发的总结已终止；现改为启动验证通过后才启用等待脚本，最终仍只触发一次完成总结。

## 监控和结束后的总结

只观察启动后的约3分钟；结束本次对话后，训练及TensorBoard继续运行，模型不轮询训练状态。启动检查结果记录到准备run `metrics/startup_verification.json`。 00:36核验通过：实际观察218秒，累计学习步49,871→53,071，新增3,200步；events横轴同步到53,071且文件继续增长，6006返回HTTP200，无失败文件。该时刻尚未完成新的整批PPO更新，不能据此声称学习效果。模型监控至此结束。

TensorBoard：[http://127.0.0.1:6006](http://127.0.0.1:6006)，仅加载本run `tensorboard/`。横轴是累计实际学习底层步，从49,871继续；评估步另计。查看命令（服务已运行时无需重复启动）：

```bash
.venv_terrain_cuda/bin/tensorboard --logdir outputs/runs/upstream_1km_fixed_far/a1_seed23_crater_reward_1m_20261024_01/tensorboard --host 127.0.0.1 --port 6006 --reload_interval 5
```

本地等待脚本PID2779395每60秒检查训练进程，等待期间不调用模型。训练及最终评估结束后先用Python生成 `metrics/long_training_review.json` 和 `figures/long_training_progress.png/pdf`，再以独立 `codex exec --ephemeral` 会话调用模型一次，补写本报告，最终回复另存本run `metrics/automatic_final_summary.md`。使用独立会话以避开上次当前会话写入冲突；一次极小实调用已返回`READY`，不是只检查命令语法。模型失败保留日志，不循环重试。

自动触发状态位于准备run `metrics/automatic_summary_status.json`，模型日志为 `logs/automatic_codex_summary.jsonl`。运行依赖本机训练进程和网络；聊天窗口是否刷新不作为完成证据，以落盘报告为准。

## 结束原因与实际预算

训练于2026-10-24 00:52:35（Asia/Shanghai，失败文件时间戳）异常退出。`metrics/failure_1792774355850755919.json`记录 `phase=sampling`、`ValueError('Invalid or nonfinite learning data')`、累计75,422步及 `automatic_retry=false`；`logs/console.log`保留完整堆栈。没有正常结束的 `summary_*.json`，不能记作 `budget_complete`。

第6批采样JSON记录1次 `environment_invalid`，子因为 `unknown_footprint`；无效批次已保存于 `state/invalid_batch_000006.pkl.gz` 及其JSON元数据。通用异常同时覆盖无效样本和非有限数，本次只按已有记录归档，不把报错直接解释为已确认的NaN，也未加载环境复现。该批未进入PPO，采样消耗仍计入授权预算。

| 账目 | 实际值与边界 |
| --- | --- |
| 初始累计学习采样 | 49,871步，继承4批／20轮PPO |
| 本次新增实际训练采样消耗 | **25,551／1,000,000步**；未用974,449步，不自动续用 |
| 已完成更新覆盖的新增采样 | 12,757步；累计62,628步时完成第5批 |
| 被拒绝批次的采样消耗 | 12,794步；累计观测到75,422步，未作PPO更新 |
| PPO更新 | 新增1批／5轮；累计5批／25轮 |
| 累计实际训练采样 | 75,422／1,049,871步，含被拒绝批次；不等于全部用于学习 |
| 独立评估 | **2,400／88,800步**，仅初始化面板；未用86,400步 |
| 本次总结消耗 | 0新增环境步、0优化器更新 |

计数以 `initialization.json`、`update_000005.json`、两批共64个 `collection_*.json`、失败JSON及 `evaluation_initial.json`交叉核对。第5批的回合同时出现在collection和update文件，只计一次；评估progress／reservation不是新增执行量，不重复相加。两项授权上限均未超过。

## 固定60秒窗口：训练前有数据，训练后缺失

sample与mean使用**相同4个固定起点**。以下是49,871步处的初始化评估，不是新增训练后的结果。存活表示截至终止或观察截止的时间；全部样本在60秒时仍存活，属于右删失，实际寿命只知道至少60秒，**不算任务成功**。`dmax_reduction = 初始Dmax − 最终Dmax`，只有正值才表示最远两车距离缩小。

| 阶段／模式 | 有效样本 | 存活观察时间 | 窗内失败 | 平均dmax_reduction | 最远车距缩小的起点 | 实际平均速度 |
| --- | --- | --- | --- | --- | --- | --- |
| 训练前sample，49,871步 | 4 | 各60.0 s，4例删失 | 0/4 | **−3.734 m** | 1/4 | 0.1933 m/s |
| 训练前mean，49,871步 | 4 | 各60.0 s，4例删失 | 0/4 | **−22.963 m** | 0/4 | 0.3154 m/s |
| 中途定期面板 | 未执行 | — | — | — | — | — |
| 训练后最终sample／mean | 未执行 | — | — | — | — | — |

两模式短窗均记录0次任务成功、无终止原因；平均最远车距分别扩大3.734 m和22.963 m。逐起点 `dmax_reduction` 为sample `[-3.181, +2.295, -13.502, -0.549] m`、mean `[-8.229, -22.577, -26.624, -34.420] m`。mean运动更快不等于更靠拢。中途未到100,000步首个评估阈值，训练后没有可配对数字，缺失不能记为0失败或0成功率。

## 完整任务评估：单独记为未执行

完整任务成功要求实际队形、间距、共同区域、地形、速度和姿态漂移条件同时满足，并连续保持5秒；当前最大车间距阈值12 m、离散度30 m²。60秒窗口活到截止不能替代这些条件。

| 最终1500秒面板 | 实际执行起点数 | 成功／失败数 | 失败原因 | 存活时间 |
| --- | --- | --- | --- | --- |
| `full_far_sample` | 0（计划4） | 未执行，无结果 | 无结果 | 无结果 |
| `full_far_mean` | 0（计划4） | 未执行，无结果 | 无结果 | 无结果 |

`evaluation_final.json`不存在，不能写成0/8成功，也不能用训练回合的失败或寿命补齐该表。本run没有未见图或多种子验收；结果属于Proxy，未作Isaac／PhysX验证。

## 已保存的中途过程

| 位置 | 累计实际采样 | 已完成新增PPO批次 | 本批已结束训练回合 |
| --- | --- | --- | --- |
| 完整继承点 | 49,871 | 0 | 不把父run回合重新计入本run |
| 第5批更新及完整保存点 | 62,628 | 1 | 4次 `terrain_failure`，均为 `driving_slope`；寿命87.4、179.8、167.4、86.8 s，平均130.35 s |
| 第6批采样结束、有效性检查失败 | 75,422 | 仍为1 | 1次 `environment_invalid`／`unknown_footprint`，已记录寿命528.6 s；该批被拒绝 |

第5批已结束回合平均原始回报+9.119、HOLD 37.07%，但4个回合都以地形失败结束，不能按正回报声称改善。只有1个新增完成更新点，不能构造长训上升或下降趋势；第6批的无效回合也不与第5批有效地形失败混算平均寿命。已结束回合有未结束样本删失和结束先后偏差，回合还可能始于继承点之前，不能替代同4起点固定评估。

[过程图PNG](../../outputs/runs/upstream_1km_fixed_far/a1_seed23_crater_reward_1m_20261024_01/figures/long_training_progress.png)／[PDF](../../outputs/runs/upstream_1km_fixed_far/a1_seed23_crater_reward_1m_20261024_01/figures/long_training_progress.pdf)只展示累计采样过程和训练前固定面板，明确标注没有训练后评估。

## 保存点、TensorBoard与汇总证据

最后完整保存点为本run `checkpoints/update_000005.pt`，对应62,628步；`update_000005.owners/owner_000.pt`至`owner_003.pt`四个分片均存在，中央文件及分片的大小／SHA256与保存元数据一致。**75,422步异常现场没有完整checkpoint**；`update_000005.learner.pt`是单独learner备份，不能替代owner完整状态。本次仅校验文件，没有恢复训练，也不把这个保存点推荐为已收敛策略。原5万步run及其恢复点未改。

TensorBoard的两个events文件保留：主日志187,859字节，独立评估日志2,080字节。横轴为累计实际训练环境步，manifest恢复边界49,871；初始化评估记在49,871，采样64点覆盖50,271→75,422，完成更新曲线只有62,628处1点。训练已结束，日志保留供上文6006地址及查看命令回放；本次没有启动或切换服务。

准备run的原自动统计脚本要求存在正常结束summary，因本次异常退出返回1，日志为 `logs/automatic_metrics_summary.log`，状态快照记录 `analysis_returncode=1`。本次用独立的纯JSON汇总脚本补齐异常结果，保留原脚本和错误记录；第一次写图因缺少figures目录失败，补建后完成，两个日志均保留。这是统计产物修正，未修改生产代码、未跑新环境或算法测试。

关键证据路径（下列训练产物均相对本run）：

- `metrics/long_training_review.json`：异常结束、预算、初始化面板及分批回合汇总；缺失的最终评估显式标为 `not_run`。
- `metrics/completion_artifact_verification.json`：一次性checkpoint／owner校验、events点数及横轴、故障时间戳；核对时未发现更晚run的manifest。
- `metrics/failure_1792774355850755919.json`、`logs/console.log`、`state/invalid_batch_000006.pkl.gz`：异常退出与无效批次原始证据。
- `metrics/evaluation_initial.json`、`metrics/update_000005.json`、`metrics/collection_000005_*.json`及`collection_000006_*.json`：现存评估与训练过程。
- 准备run `config/summarize_interrupted_training.py`、`logs/interrupted_metrics_summary.log`和`logs/interrupted_metrics_summary_02.log`：补充统计源码、首次写图失败及修正成功记录。
- 准备run `metrics/automatic_summary_status.json`、`logs/automatic_codex_summary.jsonl`：本地脚本的一次性总结状态及模型日志；最终回复由CLI保存到本run `metrics/automatic_final_summary.md`。

本次补充汇总命令（只读取已保存数据）：

```bash
.venv_terrain_cuda/bin/python outputs/runs/upstream_1km_fixed_far/crater_reward_1m_preparation_20261024_01/config/summarize_interrupted_training.py
```

精确训练启动命令和环境变量仍保存在本run `config/training_command.txt`、`config/launch.json`。本轮到此结束，不重试、不追加预算、不安排后续自动任务。后续若处理此次中断，明确问题是 `environment_invalid/unknown_footprint` 与学习有效性门控；当前资料不足以评价增加训练量后的行为收益。
