# exp144：B0后段checkpoint多种子配对趋势审计

## 目的

exp143发现exp125 `relative_quintic` 的1024→2048 checkpoint在原评测种子上仍显著改善，但因约束竞争的严格辅助条件失败，没有授权12M训练。exp144只复核这段后期改善能否跨评测种子稳定出现，并判断地形使用是否也随训练加深而改善。

本实验冻结Actor，不训练、不修改配置、不重新选择checkpoint。它是决定是否值得重新规划B0深度训练的最后一项前置审计，而不是新的网络或奖励方向。

## 冻结输入

- run：`outputs/runs/exp125_decentralized_tiered_b0_pure_rl/b0_screen_seed23_4m_relative_quintic/`；
- checkpoint：`ppo_timestep_001024.pt` 与 `ppo_timestep_002048.pt`；
- 配置：该run保存的 `config/experiment.yaml`；
- 普通评测种子：`1023, 2023, 3023, 4023, 5023`；
- 每个种子512环境、480步；
- terrain-contrast种子：`12023, 13023, 14023`；
- 每个contrast种子256环境、120步；
- 两个checkpoint对每个种子使用相同初始状态和地形随机种子。

checkpoint时步继续决定初始状态课程位置：1024与2048都处于近距warmup分布，因此本审计只比较同一分布上的学习趋势，不宣称验证远距泛化。

## 配对指标

对每个普通评测种子 \(z\)，定义：

\[
I_d(z)=
\frac{d_{1024}(z)-d_{2048}(z)}{d_{1024}(z)},
\]

\[
I_c(z)=
\frac{c_{1024}(z)-c_{2048}(z)}
{\max(c_{1024}(z),10^{-8})},
\]

\[
\Delta s(z)=s_{2048}(z)-s_{1024}(z).
\]

对terrain-contrast种子分别计算动作MSE与路径风险改善：

\[
\Delta M(z)=M_{2048}(z)-M_{1024}(z),
\]

\[
\Delta P(z)=P_{2048}(z)-P_{1024}(z).
\]

其中 \(M\) 是正常地形与地形置零的动作MSE，\(P\) 是正常观测相对地形置零的路径风险降低比例。

## 通过门限

只有以下条件全部满足，才允许另行制定一个不超过12M的B0深度训练计划：

- 五个普通评测种子中至少四个满足 \(I_d(z)>0\)；
- 五个种子中至少四个满足 \(I_c(z)>0\)；
- 五个种子中至少四个满足 \(\Delta s(z)>0\)；
- 平均dmax相对改善不低于15%；
- 平均collision相对改善不低于50%；
- 平均success绝对提高不低于2个百分点；
- 三个terrain-contrast种子中至少两个满足 \(\Delta M(z)>0\)；
- 三个种子中至少两个满足 \(\Delta P(z)>0\)；
- 2048 checkpoint的平均terrain动作MSE不低于 `0.01`；
- 2048 checkpoint的平均路径风险改善高于1024 checkpoint至少0.5个百分点。

最后两项用于防止只因集合几何变好就启动长训练，而地形分支仍没有形成可执行路径差异。

任一条件失败，停止B0深度训练假设；不通过删除地形门限、改评测种子或追加checkpoint来补救。全部通过也只授权形成计划，不直接运行训练，更不授权40M。

## 当前状态

审计已完成，状态为 `stop_b0_depth_training_hypothesis`。

## 结果

普通评测的五个配对种子全部表现出相同趋势：

| 指标 | 五种子平均变化 | 改善种子数 | 门限 | 判定 |
| --- | ---: | ---: | ---: | --- |
| dmax ratio相对改善 | 29.76% | 5/5 | ≥15%，至少4/5 | 通过 |
| collision相对改善 | 89.94% | 5/5 | ≥50%，至少4/5 | 通过 |
| success绝对增量 | 4.41个百分点 | 5/5 | ≥2个百分点，至少4/5 | 通过 |

2048 checkpoint确实比1024 checkpoint更接近集合能力，而不是原评测种子的偶然结果。但terrain-contrast给出了相反结论：

| contrast种子 | t1024动作MSE | t2048动作MSE | t1024路径风险改善 | t2048路径风险改善 |
| --- | ---: | ---: | ---: | ---: |
| 12023 | 0.000559 | 0.007339 | +0.0873% | -0.1270% |
| 13023 | 0.000571 | 0.007526 | +0.0642% | -0.1817% |
| 14023 | 0.000590 | 0.007508 | +0.0848% | -0.2074% |

三个种子中，Actor对地形输入的动作响应均显著增大，但正常地形观测下的实际quintic路径风险全部高于地形置零对照。2048 checkpoint平均动作MSE为 `0.007458`，未达到 `0.01`；平均路径风险趋势相对1024下降 `0.2508` 个百分点，而门限要求提高至少0.5个百分点。

## 结论

exp144否决“只要继续加深B0训练，地形规划会随集合能力自然出现”的假设。当前问题不再是terrain encoder完全未被使用：训练确实增强了地形输入对动作的影响，但这种影响没有指向低风险quintic路径。结合exp128–exp140，最一致的解释仍是共享团队advantage不能把逐车地形结果稳定归因给对应动作。

因此：

- 不启动12M或40M B0深度训练；
- 不删除地形门限，也不通过更换评测种子补救；
- 不扫描相对路径风险权重，因为exp125 `relative_only`已表明放大该项会显著增加collision；
- 下一项若继续，只允许先审计“统一的逐车局部任务回报”是否同时具备集合、地形和安全动作辨识度，不能再分别追加单项信用。

## 产物

- `outputs/runs/exp144_b0_checkpoint_trend_multiseed_audit/frozen_exp125_seed23_t1024_t2048/metrics/paired_trend_gate.json`
- `outputs/runs/exp144_b0_checkpoint_trend_multiseed_audit/frozen_exp125_seed23_t1024_t2048/metrics/checkpoint_seed_sweep/summary.json`
- `outputs/runs/exp144_b0_checkpoint_trend_multiseed_audit/frozen_exp125_seed23_t1024_t2048/metrics/terrain_contrast/`
- `outputs/runs/exp144_b0_checkpoint_trend_multiseed_audit/_suite/metrics/audit_summary.json`
