# 2026-10-23：障碍输入正式训练结束与问题分析

2026-10-24最新：[新增100万步续训](t11_crater_reward_long_training_2026-10-24.md)已从短训最终完整状态启动，本页仍为旧训练复核。

后续进展：[坑分档实施与验证](t11_crater_classes_2026-10-23.md)已完成，[新奖励5万步短训](t11_crater_reward_short_training_2026-10-23.md)已于23:09结束，实际学习49,871步。短窗失败减少，但平均车间距离仍扩大；最新结果见该短训报告。本页保留上一轮训练的结果与问题分析。

核查时间：2026-10-23 17:36（Asia/Shanghai）。目标是判断本轮是否获得有效集合行为、主要问题出现在哪一阶段。只读已有训练 JSON、独立评估、TensorBoard 和少量已保存轨迹；新增环境步、评估步、优化器步均为 0。没有修改训练源码、配置或启动下一轮训练。

## 结果与结束条件

本轮在 **15:48 正常完成预算并保存**，不再运行。当前障碍输入分支累计 **1,497,394 底层步、138 批、690 轮 PPO**；其中父 run 完成前 23 批／261,149 步，优化续训新增 1,236,245 步。加更早软风险 run 的 502,369 步，总学习消耗 **1,999,763／2,000,000 步**，剩余 237 步没有追加使用。

障碍输入分支累计 **20,986 个结束回合：20,983 次 `terrain_failure`、3 次 `task_deadline`、0 次成功**。最终独立评估固定／未见地图 × sample／mean 合计 **0/16 成功，16 次均为地形失败**。目前没有可推荐为已收敛的 checkpoint。

当前 run（下文 `R`）：

```text
outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_optimized_20261022_01/
```

父 run 为同实验族的 `a1_seed23_far_lunar_obstacle_remaining_20261022_01/`。两者是同一学习状态的完整续接，可按继承后的步数分析；更早软风险 run 重新初始化前后的曲线不能拼为同一策略。

结束依据为 `R/metrics/summary_000138.json`：`stop_reason=budget_complete`、`unused_base_steps=237`。最终恢复点 `R/checkpoints/update_000138.pt`，SHA-256 为 `2bc275912740f25cebc57ca39b3cbc0c9209522736459167b99ab740fb5700b1`。本分支独立评估共 8,183 步，其中最终 962 步；加旧软风险评估 744 步，共 8,927 步，继承的初始化评估只计一次。

## 学习曲线实际说明什么

首窗口是第 1–10 批，末窗口为最后 10 个完整 512 转移批，即第 127–136 批；第 137／138 批只有 384／64 转移，保留在总体统计和图中，不纳入末窗口。回报与寿命按结束回合加权，HOLD、EV、PPO 和障碍比例为批间均值。

| 指标 | 首 10 批 | 末 10 个完整批 | 判断 |
| --- | ---: | ---: | --- |
| 原始回合回报 | −100.10 | −67.88 | 提高，但没有对应任务进步 |
| 折扣回合回报 | −95.46 | −65.70 | 折扣口径也呈相同现象 |
| 平均存活时间 | 19.00 s | 12.42 s | 缩短约 34.6% |
| 5 s 内结束的回合 | 25.57% | 37.19% | 早期失败变多 |
| HOLD 比例 | 34.87% | 0.89% | 后期几乎只选 MOVE |
| 障碍 near 比例 | 91.44% | 91.07% | 未明显下降 |
| 障碍 overlap 比例 | 66.05% | 67.04% | 未改善 |
| 更新前价值解释方差 EV | 0.0630 | 0.5446 | 能更好预测当前回报，不能代替任务成功 |
| PPO 末轮近似 KL | 0.000484 | 0.004375 | 未见明显数值失稳 |
| PPO 末轮 clip fraction | 0.34% | 10.74% | 策略确实在更新 |

![训练结果与风险曲线](../../outputs/runs/upstream_1km_fixed_far/final_review_20261023_01/figures/training_results.png)

全分支回合寿命中位数 **8.8 s**、95 分位 **41.8 s**；**84.64%** 在 25 s 内结束，只有 9 个回合达到 300 s。3 个存活到 1500 s 时限的回合仍然未完成集合。所有结束回合的最终驻留计数为 0，说明当前主要问题仍在早期安全运动和靠拢阶段，尚没有形成可用停车行为；终止计数不用于推断回合中每一时刻都没有短暂驻留。

所有 138 批已保存 learning 数值有限。有效转移 70,080，Actor 排除动作尚未生效的终端 1,884 个，占 **2.69%**，其余 68,196 个保留；Critic 和 GAE 仍使用有效终端信用。因此现有证据不支持把失败归因于 PPO 没运行，或 effective credit 删除了大部分失败信号。

优化后的长期采样平均吞吐为 **22.22 底层步/s**，父 run 为 13.66 步/s；整体提高约 62.7%。这两个训练窗口的行为／重置分布不同，只能作为运行统计。计算等价性能结论继续使用此前同状态 2.006 倍配对。采样工程优化已生效，任务学习结果没有随之变好。

## 完整任务评估

固定面板每种策略 4 个相同初态；每次均全为 `terrain_failure`、成功 0/4。最大车距缩小量为负表示更分散。

| 训练步 | sample 平均寿命 | mean 平均寿命 | sample 最大车距缩小 | mean 最大车距缩小 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13.80 s | 15.95 s | −0.83 m | −3.13 m |
| 503,278 | 25.70 s | 11.35 s | −3.76 m | −1.96 m |
| 1,001,373 | 278.50 s | 15.75 s | −0.27 m | −2.75 m |
| 1,497,394 | 12.50 s | 12.05 s | −1.52 m | −0.52 m |

100 万步 sample 的平均寿命由单局 1081.2 s 拉高，该局最后仍为地形失败，未延续为最终策略的稳定改善。最终 unseen 的 sample／mean 平均车距缩小分别 **+0.15／−0.48 m**，成功也都是 0/4。最终 16 局只存活 **0.8–26.4 s**，固定／未见初始最大车距均值约 512／443 m，距离 12 m 的任务阈值仍很远。每面板仅 4 局，不据这些小幅距离变化排列 checkpoint 优劣；零成功和早期失败已经足以决定下一步。

原始证据：`R/metrics/evaluation_initial.json`、`evaluation_step_0503278.json`、`evaluation_step_1001373.json`、`evaluation_final.json`。初始化文件与父 run 逐字节相同，是网络初始化时的评估，并非优化迁移边界的新评估。评估使用冻结策略、独立环境／RNG，新增优化器步为 0。

## 主要问题与机制

**1. 回报排序与需要的安全探索行为存在冲突迹象，这是下一项优先诊断。** 全部已结束回合的原始回报与寿命相关系数为 **−0.9949**；末 10 个完整批的折扣回报与寿命相关为 **−0.9950**。更早失败通常得到更高回报。原始回报首末改善 32.22，其中 gather 增加 24.37、overlap 惩罚减少 5.96；结合寿命缩短及风险比例不降，不能将这些分量解释为更有效靠拢或避障。

当前奖励每步收取距离／离散度、地形、障碍等负成本，地形失败统一结算 −100（[奖励实现](../../source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/local_goal/upstream_reward.py)，`advance()` 第 118–151 行）。在同样失败的轨迹里，提早结束能减少后续累积成本。最终 fixed mean 的 env 3 仅 0.8 s 失败，raw return 为 −8.65；其中 gather 为 +80.37、success_hold 为 +12.02、terminal 为 −100。3 次 1500 s deadline 的折扣回报约 −2228 至 −2368，也显著劣于大量秒级失败。

这里必须区分：终端将势函数置零本身是现有正确的 shaping 契约，折扣势差总和为 `−Φ(s₀)`；不能将失败时的正势差直接定性为 bug，也不应简单删除终端势归零。问题是**实际非势成本与失败代价的排序**。已有数据支持激励失衡假设，但不能据相关性断言策略主动寻求失败。

**2. 行为越来越偏向 MOVE，没有获得持续安全运动和停车经验。** HOLD 从约 35% 降到不足 1%，回合同时变短，障碍比例没有下降；这比“训练步数不足”更直接地说明当前学习方向不理想。EV 上升只能说明 Critic 学会了预测这些回报。先核对目标排序，再决定是否需要调整现有奖励、动作／时序或规划职责；当前没有证据支持先扩网络或盲调 PPO。

**3. 软障碍指标和实际失败判据并非同一件事，现有日志不足以排定物理子原因。** `terrain_failure` 聚合 `impassable_footprint`、`body_clearance`、`terrain_step`、`driving_slope`、`contact_roughness`（`upstream_environment.py:474–480`）。near／overlap 来自坑石标签距离和覆盖圆盘，普通地形风险另经 ground 通道提供；它们既不是物理碰撞率，也不精确等同于上述五种失败边界。

只读抽查第 24／81／138 批交互记录的 261 个结束回合，全为地形失败；保存记录没有物理失败子类，不能据此指定“主要撞岩石”或“坡度阈值太严”。如后续需要修复具体地形行为，应在既有实际足迹判定点保留失败车辆与子类，选一个现有失败片段核查合法输入、软代价和判据的一致性，不增加旁路避障规则。

**4. 评估速度字段有独立的统计缺陷。** `scripts/upstream_decision_evaluation.py:145` 对已结束环境继续累计旧 `applied_control`，第 170 行分母却只用有效时长，导致 `mean_applied_speed` 偏大。最终 unseen sample 的 env 3 报 3.27，而实际路径长度／时长对应平均平移速度约 0.186 m/s。本报告排除此字段，成功、结束原因、寿命和直接车距不受影响。`geometry_overlap_steps`、`clearance_violation_steps` 仅统计车间关系，为零也不能解释为没有触及危险地形。这项缺陷已定位，本次未扩展为源码修复。

## 下一步与证据

本轮已达到预算结束条件，不自动追加长训、种子或消融。下一项有界问题是：**固定同一初态，对安全延续与提前地形失败核算折扣回报，分离终端势差和实际非势成本，确认安全行为是否被正确排序。** 验收是能明确指出哪个现有奖励项／终端代价决定排序，并据此决定最小修复；无需先追求整套失败根因。若需新环境采样或学习，另按授权预算执行并保存 TensorBoard。

分析产物 `outputs/runs/upstream_1km_fixed_far/final_review_20261023_01/`：

- `metrics/analysis.json`、`updates.csv`：138 批完整统计、窗口定义、预算与原始评估。
- `metrics/source_provenance.json`、`verification.json`：输入文件 SHA、批序列／PPO／评估计数核验、有限值与 TensorBoard 对齐。
- `metrics/trace_sample.json`、`config/inspect_saved_traces.py`：第24／81／138批的只读轨迹抽样、原始文件SHA及字段范围；命令记录在manifest。
- `figures/training_results.png/pdf`：可回放／导出的训练曲线。
- `config/analyze.py`、`logs/analysis.log`、`run_manifest.json`：只读统计实现、命令及零新增训练计费。

实际命令：

```bash
.venv_terrain_cuda/bin/python \
  outputs/runs/upstream_1km_fixed_far/final_review_20261023_01/config/analyze.py
```

TensorBoard [http://127.0.0.1:6006](http://127.0.0.1:6006) HTTP 200，仅加载 `R/tensorboard/`。采样 events 3,644 点、学习曲线 115 点，末步均为 1,497,394；sample／mean 成功率各有初始化、50 万、100 万、最终四个点，均为 0。训练结束后 events 不再增长是正常现象。原 events／checkpoint 保留，复看命令：

```bash
.venv_terrain_cuda/bin/tensorboard \
  --logdir outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_optimized_20261022_01/tensorboard \
  --host 127.0.0.1 --port 6006 --reload_interval 5
```

已有服务无需重复启动。核查仅涉及数据统计、原始来源、图和文档 diff，未运行无关算法测试。以上结论属于 Proxy，未新增 Isaac／PhysX 验证。

2026-10-23 19:32后续：用户授权的[奖励系数／设计审计与算分实验](t11_reward_audit_2026-10-23.md)已完成，确认持续成本会将假设提前失败排在安全延续及理想成功之前。当前未修改生产奖励或启动学习；具体反事实口径、24个奖励层假设及失败后继续的边界见该记录。

## 最终评测动画补录

按用户要求，复用 `scripts/render_skrl_proxy_rollout.py` 的帧绘制／GIF 保存链，在原入口增加 `--upstream-evaluation final`；评测端复用 `upstream_decision_evaluation.panel()`，只增加实际执行位姿捕获。原 `trace_final_*.json` 没有逐步位姿，因此用 `update_000138.pt`、原地图／种子及重置顺序重放四个面板，不用车距摘要推算轨迹。

重放新增 **962 个独立评测底层步、0 个学习／优化器步**，单独记在 `R/metrics/evaluation_final_render_capture.json`；原最终评测及正式训练预算统计保持原口径。四面板共核对 772 项结果／逐决策字段，全部一致，数值最大绝对差为 **0**；16 局的终止原因、步数、车距、奖励及实际捕获长度均匹配。结果仍为 16 次 `terrain_failure`、0 成功。

动画入口：[16 局动画总览](../../outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_optimized_20261022_01/videos/evaluation_final/index.html)。点击预览图播放对应 GIF；例如[固定地图 sample／env 0](../../outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_optimized_20261022_01/videos/evaluation_final/far_sample_env0.gif)和[未见地图 sample／env 2](../../outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_optimized_20261022_01/videos/evaluation_final/unseen_sample_env2.gif)。

每局包含团队全局视图和四车局部放大，显示实际轨迹、车身／朝向、回合时间及终止原因。背景是实际世界地形的展示，不代表 Actor 全图观测，也不标注原日志未保存的物理失败子类。原始位置按 0.2 s 保存，GIF 每两步取帧并保留首末帧，播放时间与仿真时间一致，末态额外停留 1 s。16 个 GIF 共 500 帧，逐帧解码及时长核验见 `R/metrics/evaluation_final_render_verification.json`；仍属于 Proxy 动画。

实际命令（项目根目录）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv_terrain_cuda/bin/python scripts/render_skrl_proxy_rollout.py \
  --upstream-evaluation final \
  --run-dir outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_optimized_20261022_01 \
  --capture-interval 2 --max-frames 80 \
  > outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_optimized_20261022_01/logs/render_final_evaluation.log 2>&1
```

`R/state/evaluation_final_render.npz` 保存位姿和渲染用地形网格，`metrics/evaluation_final_render_capture.json` 保存重放指标、原始来源 SHA 与逐面板核对，`metrics/evaluation_final_render.json` 保存动画清单与命令，`config/evaluation_final_render_source/` 保留实际渲染源码。原 manifest 仅追加产物入口，原配置、checkpoint 和评测 JSON 不覆盖。后续画面调整可先将已有 `videos/evaluation_final/` 另存，再用 `--from-capture` 读取已保存数据，新增环境步为 0；入口拒绝覆盖已有动画。

针对性验证：`tests/test_proxy_rollout_render.py` 与 `tests/test_upstream_evaluation_replay.py` 共 **14 项通过**，覆盖异步终止截断、初末帧、真实 GIF 毫秒时长及 upstream 画面语义；日志为 `R/logs/render_tests.log`。
