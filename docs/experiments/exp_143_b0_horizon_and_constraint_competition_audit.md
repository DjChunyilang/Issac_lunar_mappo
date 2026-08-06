# exp143：B0训练时域充分性与约束竞争审计

## 目的

exp142说明真实碰撞约束能够降低collision，但策略以放弃集合换取安全。与此同时，exp125 `relative_quintic` 的4M筛选在后半段仍持续改善。exp143不训练、不修改网络或奖励，只使用已经冻结的exp125与exp142机器可读产物，回答两个问题：

1. exp125在2048个训练时步停止时，是否已经进入稳定平台；
2. exp142的失败是否与标准化cost advantage在集合能力形成前取得与reward advantage相当的尺度有关。

只有审计证明B0仍存在一致的后段改善，才允许制定一个保持全部算法与环境配置不变的12M深度探针。该授权不等于恢复40M，也不放宽地形门限。

## 冻结输入

- exp125：`b0_screen_seed23_4m_relative_quintic`；
- exp142：`collision_lagrangian_seed23_4m`；
- 只读取 `train_metrics.jsonl`、`eval_metrics.json`、`summary.json` 与预注册的constraint history；
- 不重新选择checkpoint，不以动画或TensorBoard曲线作为证据。

## 指标

对B0的1024和2048 checkpoint定义：

\[
I_d=\frac{d_{1024}-d_{2048}}{d_{1024}},
\qquad
I_c=\frac{c_{1024}-c_{2048}}{c_{1024}},
\]

其中 \(d\) 为dmax ratio，\(c\) 为collision rate；success绝对增量为：

\[
\Delta s=s_{2048}-s_{1024}.
\]

训练日志按时序划分四等份。末四分之一相对前一四分之一的dmax改善为：

\[
I_{q4}=\frac{\bar d_{q3}-\bar d_{q4}}{\bar d_{q3}}.
\]

exp142的reward advantage与cost advantage分别标准化。因此进入Actor的cost项与reward项标准差之比在数值上为：

\[
R_k=
\frac{\operatorname{std}(\lambda_k\widehat A^c)}
{\operatorname{std}(\widehat A^r)}
=\lambda_k.
\]

同时记录：

- \(\widehat J_c>1\) 的更新比例；该值表示原“episode等效碰撞率”实际是480步碰撞事件期望数，不是有界episode失败概率；
- 首次 \(\lambda_k\ge0.5\) 的更新；
- 该更新之后exp142新增success数量；
- B0在相同训练区间的新增success数量。

## 审计门限

只有以下条件全部成立，才允许制定exp144的12M B0深度探针：

- B0从1024到2048 checkpoint的 \(I_d\ge20\%\)；
- collision相对降低至少50%；
- success绝对提高至少3个百分点；
- \(I_{q4}\ge10\%\)，说明停止前仍有明确训练改善；
- Actor与各encoder均更新且动作标准差大于 \(10^{-4}\)；
- 2048训练时步仍小于4096步课程warmup，4M尚未覆盖课程扩展；
- exp142在 \(\lambda\ge0.5\) 后success不再增长，且训练后半段至少一半更新满足 \(R_k\ge1\)，证明安全目标与集合主目标的尺度竞争真实发生。

任一条件失败则不训练exp144。全部通过也只授权一个随机初始化、单seed、12M的B0原配置探针；不得加入cost critic、Actor信用、GNN、GRU、安全投影或奖励改动。

## exp144预期边界

若获得授权，exp144固定：

- 仍使用exp125 `relative_quintic`配置；
- seed23、2048环境、6144训练时步，即约1258万环境交互；
- 从零初始化，不从exp125 checkpoint续训；
- 只在3072、4096、5120、6144保存checkpoint；
- 12M只验证更深Pure RL训练是否保持后段改善并开始使用地形，不授权40M。

## 当前状态

审计已完成，状态为 `stop_b0_depth_extension`。

## 结果

| 指标 | 结果 | 门限 | 判定 |
| --- | ---: | ---: | --- |
| 1024→2048 dmax相对改善 | 29.08% | ≥20% | 通过 |
| 1024→2048 collision相对改善 | 90.09% | ≥50% | 通过 |
| 1024→2048 success绝对增量 | 3.71个百分点 | ≥3个百分点 | 通过 |
| q3→q4训练dmax改善 | 14.35% | ≥10% | 通过 |
| B0停止位置 | 2048步 | `<4096` warmup | 通过 |
| exp142后半段 \(R_k\ge1\) 比例 | 100% | ≥50% | 通过 |
| exp142尺度竞争后新增success | 2 | 必须为0 | **失败** |

B0两个候选checkpoint的独立评测变化为：

| checkpoint | dmax ratio | success | collision | timeout |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 0.28826 | 0.01074 | 0.84766 | 0.14160 |
| 2048 | 0.20444 | 0.04785 | 0.08398 | 0.86816 |

exp142的 \(\widehat J_c>1\) 更新占12.5%，说明该量确实可表示“480步内期望碰撞事件数”并超过概率上界。首次使用不低于0.5的cost/reward标准差比发生在第6次更新，即训练时步384。此后exp142只新增2个success，而B0在相同时间点之后新增171个success；exp142后16次更新的cost/reward标准差比全部不低于1。该证据支持“约束竞争显著压制集合”的解释，但没有满足预注册的严格零增长条件。

## 决策

exp143总门限失败，不启动预设exp144 12M训练，也不事后把“新增success为0”改为比例门限。B0后段改善仍是有效诊断事实，但目前只来自单一训练seed和两个checkpoint。若继续评估训练深度，只允许先对冻结B0的1024/2048 checkpoint进行相同场景、多评测种子的配对趋势复核；不得直接延长训练或恢复cost约束。

## 产物

- `outputs/runs/exp143_b0_horizon_and_constraint_competition/frozen_exp125_exp142/metrics/horizon_constraint_audit.json`
- `outputs/runs/exp143_b0_horizon_and_constraint_competition/_suite/metrics/audit_summary.json`
