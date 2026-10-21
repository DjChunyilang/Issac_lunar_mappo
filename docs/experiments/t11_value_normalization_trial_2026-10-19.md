# T11价值目标归一化：奖励尺度的单变量短学习

## 已完成结果：停车局部改善，远距学习筛选未通过

本轮已正常退出：新候选实际19,863底层训练步、800团队决策／有效转移、13次采样更新／65轮PPO，含初始与最终评估耗时24.53 min。余下137步不足完整400步向量区间；与复用原组合组的训练步数仅差19步。评估2,517步、契约检查17步，无学习合计2,534／4,000步，本轮新增物理步总计22,397。最终checkpoint及4个owner分片已核验，主进程退出码0，全部训练／owner进程退出。

| 固定20 s指标 | 共同初始化 | 原组合最终 | 归一化最终 |
|---|---:|---:|---:|
| 随机远距：平均最大车距缩小（m） | 4.441 | 2.614 | 2.974 |
| 随机远距：净靠拢初态数 | 4/4 | 4/4 | 4/4 |
| 随机远距：平均离散度缩小（m²） | 37.051 | 24.161 | 26.840 |
| 随机远距：HOLD概率 | 50.01% | 50.52% | 51.46% |
| 均值远距：平均最大车距缩小（m） | 0.000 | 0.000 | 0.000 |
| 随机近距：成功驻留数 | 1/4 | 1/4 | 2/4 |
| 均值近距：成功驻留数 | 4/4 | 4/4 | 4/4 |

归一化最终较原组合多靠拢0.360 m，但较自身初始化减少1.468 m；远距HOLD概率比原组合高0.94个百分点。近距随机成功多1例，均值停车保持4/4。六项预设筛选三项通过，未满足“较原组合多1 m”“较自身初始化多1 m”“远距HOLD不升高”。这次固定小面板显示局部停车收益，未解决远距学习退步，不提升为默认，不追加本轮预算。

原组合／候选训练成功分别7／6次，均来自近距；候选屏蔽6条未生效终端Actor信用，Critic仍使用全部800条有效转移，归一化计数恰为800。候选近距几何停车条件满足4,162／9,863步（42.2%），原组合为4,628／9,844步（47.0%）；训练成功数和几何覆盖也不能替代固定评估结论。候选全部13批更新前Actor概率重放比值为1，训练无碰撞／间距违规。

## 数值效果与比较有效性

候选初始四面板逐值复现原组合；首批所有观测、隐藏状态、原始奖励、时长、价值、终止掩码、latent、log-prob和更新RNG逐值相同，首批5轮后的Actor参数逐值相同。首批Critic梯度范数由原组合约15.6变为约0.29，独立Critic裁剪系数全程中位数由0.0605变为1。

针对同一首批原单位固定GAE目标，更新后MSE为原组合1055.26、候选941.04，偏差−8.27／−0.76，EV为0.0009／0.0480。末批各自原单位目标的EV为0.1165／0.8901，但此时两组轨迹和bootstrap目标已经不同，不能当作同目标或真实价值校准的直接对照。数值拟合改善未转化为远距正学习增益；下一次若研究奖励结构，应与本轮数值处理分开归因。

用户在奖励调整讨论后要求“尝试调整”。本轮先执行建议中的价值目标归一化，奖励权重、动作、安全、MPPI及三项组合修复保持原值；没有新增MOVE奖金、HOLD惩罚或成功奖励调权。

## 问题、对照与结束条件

对照复用已结束的[三项组合组](t11_combined_learning_validation_2026-10-19.md)：`learning_repair_closed_loop_seed23_20k_20261019_01_combo`，实际19,844步。新候选使用相同A1、seed23、随机初始化Actor/Critic、全新Adam、低噪声初始化、独立裁剪与有效终端Actor信用。只增加`value_normalization=running`；不继承对照checkpoint的学习状态。

固定四owner×四环境、8近／8远初态、5 s决策、1 s折扣参考周期、每环境每批4个决策、5轮PPO。每次决策预留完整400步向量区间，余额不足即保存退出。训练上限20,000步，无学习上限4,000步：初始／最终各预留1,600，契约检查实际17步。失败保留已消耗步骤，不自动重试或扩大预算。

原组合组随机远距初始平均最大车距缩小4.441 m、最终2.614 m；近距随机初始／最终1/4成功，均值4/4。候选训练前必须逐值复现四面板的几何、HOLD概率、成功数和实际步数，网络初始化SHA必须一致；优化器选项除归一化外逐项相同。源码差异归档，原运行配置与固定远距快照SHA保持一致。

预设筛选：最终随机远距至少3/4净靠拢、平均缩小量超过自身初始化至少1 m并超过原组合最终至少1 m、平均离散度不增加、远距HOLD概率不高于原组合最终；近距随机／均值成功数分别不低于原组合最终。一次固定种子短学习只决定是否获得候选改善信号，不表示统计显著或完整集合／Isaac验收。预算耗尽即结束。

## 实现语义与验收

依据[MAPPO第5.1节](https://arxiv.org/html/2103.01955v4)的价值目标归一化思路，新增`RunningValueNormalization`，用float64累计有效团队回报的count／mean／M2；每批只更新一次，不按四辆车或5轮PPO重复统计，标准差下限0.1，首次采样前为恒等变换。这里使用累计矩，而非指数滑动平均；不做PopArt最后一层重参数化。

采样与bootstrap通过`policy.value()`得到原奖励单位的价值；先用采样时的原单位V计算变时长GAE和固定return target，再更新统计并训练归一化Critic输出。Actor仍使用原单位GAE及原有优势标准化，环境奖励、势差折扣和终端bootstrap不变。原单位的Critic MSE／偏差／EV另外记录，不能与归一化MSE混排。

归一化统计保存在policy buffer中，随完整／learner checkpoint保存；policy与TrainingOptions身份均带新语义，拒绝直接恢复不同价值语义。默认`none`不增加旧元数据键，不修改旧policy权重结构。训练、单进程／多owner采样及独立评估共用同一value接口。

8项不同的针对性检查通过，零跳过，实际17底层检查步：累计矩分段等价、归一化逆变换、原单位bootstrap及真实终端零值、无效样本排除、每批计数一次、相同随机数下首批Actor逐参数更新一致、单进程与4×4 owner的skrl采样／更新／恢复、旧裁剪及入口身份兼容。首轮1项失败源于测试两条分支未复用熵估计RNG；修正fixture后仅重跑该零环境步检查通过，原失败日志保留。运行环境没有ruff模块，静态检查使用已安装ruff的`.venv_isaaclab`并通过F821。

## 命令与证据

suite：`outputs/runs/upstream_scale_t11/value_normalization_seed23_20k_20261019_01/`；候选后缀`_combo_value_norm`，检查后缀`_checks`。原结果不覆盖，生成产物不提交Git。

```bash
.venv_terrain_cuda/bin/python scripts/run_upstream_t11.py \
  --config configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml \
  --run-id value_normalization_seed23_20k_20261019_01 \
  --mode repair-validation --representation A1 --seed 23 \
  --budget 20000 --samples 512 --env-owners 4 --num-envs 16 \
  --owner-cpu-workers 2 --trainer-backend skrl --policy-device cpu \
  --release-training-pause --value-normalization running \
  --comparison-run outputs/runs/upstream_scale_t11/learning_repair_closed_loop_seed23_20k_20261019_01_combo \
  --far-checkpoint outputs/runs/upstream_scale_t11/oom_fix_owner16_20261017_01_a1/checkpoints/update_000061.pt
```

候选`metrics/initialization.json`、`evaluation_{initial,final}.json`、`update_*.json`和`state/interactions_*.pkl.gz`分别保存初始化、独立行为、学习指标及完整更新输入；`checkpoints/update_*.pt`包含原单位采样计数、归一化统计、Adam、RNG与owner分片。suite的`metrics/summary.json`汇总对照及筛选。

检查目录保留`logs/contracts.log`、`contracts_r2.log`、`learning.log`及对应JUnit；`metrics/source_diff.patch`、`implementation_changes.json`记录相对于原组合冻结源码的6个文件差异。suite的`config/source_snapshot.tar.gz`冻结本轮源码，另保存新测试、依赖版本及复用的原始配置输入归档。

最终证据：suite的`metrics/summary.json`、`first_batch_comparison.json`、`training_accounting.json`和`final_verification.json`。最终候选恢复点为`checkpoints/update_000013.pt`及同名`.owners/`四个分片；原奖励单位、归一化统计、Adam、RNG与计数均保存。当前源码SHA与启动时一致，19份配置归档输入逐字节一致。只读重算命令（0新增环境步／0优化器更新）：

```bash
.venv_terrain_cuda/bin/python outputs/runs/upstream_scale_t11/value_normalization_seed23_20k_20261019_01_checks/state/analyze.py
```
