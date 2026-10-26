# 坑分档后的奖励修复与5万步短训

日期：2026-10-23，Asia/Shanghai。承接[奖励审计](t11_reward_audit_2026-10-23.md)与[坑分档验证](t11_crater_classes_2026-10-23.md)。评分修复、四张图集和100步无学习启动检查已完成；用户授权的5万步短训22:21启动，23:09按预算结束。本次部署交接依据已保存的统计补齐报告，0新增环境步／优化更新。

## 已解决的评分问题

**新评分已经能把“完成任务”排在“安全超时”和“很快失败”之前。** 本次只选择一个候选：把原有持续扣分缩到2%，保留进展势差、成功奖励、失败惩罚和真正失败终止。刚确定的中坑轻罚及每回合上限保持。

此前235个完整保存回合中，有172个活过5秒。保留每条记录的同一起点和前5秒，比较“假设第5秒失败”与“按记录继续、后来失败”：旧评分172对全部偏向提前失败；新评分172对全部偏向后来失败。额外假设四车从开始一直压中坑，按新规则持续扣分并在2分处封顶，172对的排序仍全部成立。

这些是读取已有轨迹、调用奖励函数的算分检查，**没有新增环境步或优化器更新**。它说明奖励排序被修正，不等于策略已经会绕坑或完成会合。

## 现在具体奖励什么

训练配置为 [upstream_1km_far_crater_reward_v1.yaml](../../configs/training/upstream_1km_far_crater_reward_v1.yaml)，新增 `reward_overrides.continuous_cost_scale: 0.02`。下表列出乘上该尺度后的有效系数：

| 原有持续成本 | 之前 | 新候选 |
| --- | ---: | ---: |
| 仍离队友很远 `distance_level` | 0.13 | 0.0026 |
| 队伍仍分散 `dispersion_level` | 0.04 | 0.0008 |
| 近距区域平坦度不足 `flatness_excess` | 0.02 | 0.0004 |
| 地形项总权重 `terrain_weight` | 0.5 | 0.01 |
| 地形危险项 `hazard_cost` | 0.05 | 0.001 |
| 车间安全距离不足 `clearance_cost` | 2.4 | 0.048 |
| 靠近障碍标签 | 0.2/s | 0.004/s |
| 覆盖障碍标签 | 2/s | 0.04/s |

能量项总权重也从0.003降到0.00006。原有各项的时间单位保持：前六项按既有每控制步公式计算，障碍项乘实际步长；控制周期仍为0.2秒。

**中坑单独保留0.05/s的轻罚**，按四车平均穿入程度计算，每个团队回合累计最多扣2分，reset清除该回合额度，恢复继续使用已扣额度。小坑不增加坑标签罚分，大坑仍按实际车身进入坑口触发失败，分档阈值和停车规则沿用已验证候选。

成功一次加130，地形失败／越界一次扣100，车车碰撞固定合计扣200，超时扣80；这些事件奖罚和折扣系数均保持。进展势差也不缩小，真正终端的后继势仍归零。

旧配置不提供新尺度时默认1.0，默认metadata和算术结果保持；240组奖励调用与修改前实现逐值一致，旧奖励状态可按旧语义恢复。新候选通过设置指纹及 `continuous_cost_scope` 隔离，不把旧checkpoint解释成新奖励下的无缝续训。本次学习使用全新网络和优化器。

## 同一起点的分数比较

下面人为给定同一个远距初态、逐渐靠拢及最后5秒停稳的事实序列，再由生产奖励函数计算折扣总分。成功／失败标签由算分脚本指定，未运行完整环境或车辆动力学；分数越高越好。

| 假设过程 | 旧评分 | 新评分 |
| --- | ---: | ---: |
| 300秒完成 | −532.29 | **145.37** |
| 600秒完成 | −954.36 | **105.08** |
| 1500秒完成 | −1403.91 | **69.42** |
| 安全停留1500秒，但未完成 | −1559.58 | **60.71** |
| 30秒失败 | −115.37 | **0.50** |
| 5秒失败 | −23.20 | **−3.30** |

只拖时间并没有超过及时完成：先原地等待300秒、再于600秒完成，得到102.52分，低于直接300秒完成的145.37分，也低于持续靠拢、600秒完成的105.08分。

中坑扣满2分的压力情形下，1500秒完成仍有67.52分，高于没有中坑轻罚的安全超时60.71分。所有势差恒等式、恒定成本公式及分项求和检查通过；32项奖励、分档和旧障碍输入针对性测试全部通过，无跳过。

需要在真实行为中检查的副作用是：**原有软避险信号也减弱了**。大坑、岩石和车车碰撞的既有几何／失败约束仍在，MPPI代价未随奖励尺度改变；但策略是否更频繁接近危险区域，必须看短训轨迹。奖励变高不能单独算改善，要同时看存活、实际靠拢和失败原因。

## 已授权的短训安排

实验入口为 [upstream_1km_crater_classes_v1.yaml](../../configs/experiment/upstream_1km_crater_classes_v1.yaml)，训练参数使用上面的新YAML。四张图集已齐备，剩余三图共新增5.31 GB、构建210.80秒，旧fixed_eval保留。正式入口的四图启动检查各25步、共100步，观测及学习数据有限有效，0优化器更新。

| 项目 | 设定 |
| --- | --- |
| 初始化 | 新Actor、Critic、Adam及价值归一化状态，seed23 |
| 并行／规划 | 4 owner × 4环境；CUDA MPPI，512采样×2轮 |
| 学习节奏 | 5秒策略决策；32决策rollout，5轮PPO |
| 学习预算 | 最多50,000个实际底层环境步 |
| 任务 | 保留固定远距出生和1500秒回合上限 |
| 初始／最终评估 | 相同4个起点，sample和mean各一次，每回合最多60秒；两阶段合计最多4,800评估步，独立计费 |
| 训练run | `outputs/runs/upstream_1km_fixed_far/a1_seed23_crater_reward_50k_20261023_01/` |

60秒评估用于比较短时间内是否少失败、能安全移动并靠拢；没有在窗口内完成任务，不能直接当作完整1500秒任务失败。窗口结束仍存活明确记录为删失样本；实际速度使用各车真实路程和本环境有效观察时间，避开旧速度统计在终止后重复累计的问题。短窗接线的20项不同针对性测试通过，原完整评估路径保留。

主进程PID2723717，22:21:15启动；精确命令和环境变量在准备run `config/training_command.txt`、`config/launch.json`，控制台在 `logs/training_console_01.log`。学习上限50,000步、评估上限4,800步，100步预启动单独计费。

TensorBoard `http://127.0.0.1:6006` 已切换为只显示本run，events已生成。横轴是实际学习底层步，独立评估步单独记账；历史训练events保留。远程IDE转发6006后访问；回放命令如下，当前服务运行时不需要重复启动：

```bash
.venv_terrain_cuda/bin/tensorboard --logdir outputs/runs/upstream_1km_fixed_far/a1_seed23_crater_reward_50k_20261023_01/tensorboard --host 127.0.0.1 --port 6006 --reload_interval 5
```

用户要求停止模型持续轮询。现由准备run的 `config/finish_and_summarize.py` 在本地等待训练进程及最终评估结束，等待期间不调用模型；结束后先生成统计和图表，再使用 `codex exec resume` 续接当前会话一次，补写本报告。触发状态保存在 `metrics/automatic_summary_status.json`，后台模型日志为 `logs/automatic_codex_summary.jsonl`；不自动追加或重启训练。

后续核验：自动统计和图表生成成功，但自动模型回写返回码1，状态为 `model_summary_failed`；不是训练失败。23:53读取原日志确认，续接被 `thread-store conflict: thread ... already has an active writer` 拒绝，即当前会话已有写入者，无法由另一个CLI进程同时接管；日志另记录模型列表请求超时。原自动化状态与日志保留，本次交接直接依据已有JSON补写以下结果，没有重试自动模型或重跑训练。此前“已设置自动续接”只验证了命令及等待脚本，未验证会话能被再次接管，因此没有实际实现自动唤醒总结。

## 已完成结果与判断

最终 `metrics/summary_000004.json` 为 `stop_reason=budget_complete`：学习49,871／50,000实际底层步，4批／20轮PPO、2016策略决策；129步未用。独立评估4,427／4,800步。最终完整checkpoint为 `checkpoints/update_000004.pt`，包含同名JSON及4个owner分片，learner备份单独保存。23:30检查训练主进程及owner已退出；TensorBoard6006仍提供该run回放。

前后各4个固定起点、60秒观察窗口；保存的初始Dmax与离散度逐值相同。下面“Dmax增加”取最终减初始，正数表示队伍最远车距扩大：

| 策略 | 窗口内失败：初始→最终 | 存活至窗口结束：初始→最终 | 平均观察时间：初始→最终 | 平均Dmax增加：初始→最终 | 平均实际速度：初始→最终 |
| --- | --- | --- | --- | --- | --- |
| sample | 1/4→0/4 | 3/4→4/4 | 59.95→60.00 s | 4.30→3.73 m | 0.197→0.193 m/s |
| mean | 3/4→0/4 | 1/4→4/4 | 41.40→60.00 s | 11.14→22.96 m | 0.308→0.315 m/s |

初始和最终均0次任务成功。最终8个样本均为窗口结束时仍存活的删失样本，不是完整1500秒成功。mean最终观察时间更长，Dmax变化不能当作等观察时长的靠拢速度对比；但最终两种策略的平均Dmax和离散度都仍扩大，没有形成持续集合的证据。

学习期间结束的41个回合全部为 `terrain_failure`：35次 `driving_slope`、5次 `impassable_footprint`、1次 `terrain_step`；结束回合平均寿命117.79秒，不包含预算截断时仍活跃回合。HOLD从首批0.340到末批0.359。当前判断是**同配置短窗存活改善，靠拢问题仍在**；只含1个训练seed和每模式4起点，不按回报或存活改善声称任务收敛。

下一步依据实际动作、朝向与队形变化定位未靠拢行为；本轮预算结束，不自动追加训练或启动对照。本次先完成[Windows／WSL2部署交接](../runbooks/windows_wsl2_deployment_handoff_2026-10-23.md)。

结果证据：本run的 `metrics/short_training_review.json`、`evaluation_initial.json`、`evaluation_final.json`、`summary_000004.json` 及 `figures/short_training_comparison.png`。分析命令为 `.venv_terrain_cuda/bin/python outputs/runs/upstream_1km_fixed_far/crater_reward_preparation_20261023_01/config/summarize_short_training.py`，原命令与源码hash保存在review JSON；本轮交接只读取这些结果。

## 证据与复现

准备run为 `outputs/runs/upstream_1km_fixed_far/crater_reward_preparation_20261023_01/`。本页评分证据全部为0新增环境步／0优化器步：

- [候选验证与有效系数](../../outputs/runs/upstream_1km_fixed_far/crater_reward_preparation_20261023_01/metrics/reward_verification.json)：评分门槛、旧默认兼容性及源码指纹。
- [172对保存轨迹](../../outputs/runs/upstream_1km_fixed_far/crater_reward_preparation_20261023_01/metrics/reward_saved_pairs.json)及[附加中坑持续轻罚](../../outputs/runs/upstream_1km_fixed_far/crater_reward_preparation_20261023_01/metrics/reward_saved_medium_stress.json)：逐对分数及新训练配置一致性。
- [同初态假设评分](../../outputs/runs/upstream_1km_fixed_far/crater_reward_preparation_20261023_01/metrics/reward_scenarios.json)：完整场景、假设、原始／折扣分项和校验。
- [执行命令](../../outputs/runs/upstream_1km_fixed_far/crater_reward_preparation_20261023_01/config/reward_commands.txt)与 [测试JUnit](../../outputs/runs/upstream_1km_fixed_far/crater_reward_preparation_20261023_01/logs/reward_tests_01.xml)：同目录保留评分脚本、修改前源码及控制台日志；原审计结果未覆盖。
