# exp125：严格去中心化 B0 Pure RL 4M 筛选

## 目的

验证不使用 BC、集中式执行目标和 Actor 后处理的 B0 前馈策略，能否在 96 s 时域、近距完整通信和真实地形平整度成功门限下，同时学会集合与地形相关路径。该实验也用于判断是否具备启动 40M 正式训练以及 B1/B2 架构研究的前提。

## 固定条件

- seed：23；
- 并行环境：2048；
- 训练时步：2048，即 4,194,304 次环境交互；
- rollout：64；
- 评测：1024 环境、480 步；
- Actor：101 维 `ego_v8_decentralized_tiered` 观测与 `branched_v5` 四分支前馈网络；
- 通信：12 m 内 12 维完整消息，12 m 外低频稀疏消息；
- `bc_updates=0`，`init_checkpoint=null`；
- 关闭子目标过滤、安全投影、方向性约束、末段覆盖、集合槽位和显式全局目标；
- 成功必须通过实际团队质心平整度门限。

主配置为 `configs/experiment/exp125_decentralized_tiered_b0_pure_rl.yaml`。三个奖励变体仅用于定位同一 B0 的失败原因，不作为独立研究方向。

## 筛选门限

进入 40M 必须同时满足：训练过程有限且各编码器有效更新、动作标准差大于 \(10^{-4}\)、训练 dmax 至少下降 30%、出现成功 episode、碰撞不超过 10%、地形置零动作对比均方差大于 0.02，以及正常策略相对地形置零对照的路径风险至少下降 5%。

地形对比固定其他观测和通信缓存，只将 Actor 的 50 维地形切片置零。路径风险均沿 Actor 动作生成的实际 quintic 轨迹采样，不再用起点到子目标的直线代理。

## 实现中发现并修正的问题

1. 初版配置继承自 exp063，实际质心平整度奖励权重仍为零。`flatness_fix` 起将其显式恢复；初版 `near` 只保留为诊断结果，不用于判断完整方案。
2. 原路径地形风险在 quintic 轨迹生成前按直线采样，与规划文档定义不一致。现改为沿实际 quintic 轨迹采样。
3. 4M checkpoint 的课程评测原先错误采用最终远距分布。现默认读取 checkpoint 的训练时步，4M screen 正确使用 2.4–3.4 m 近距分布。
4. 为判断绝对风险惩罚是否被背景地形成本掩盖，增加了默认关闭的相对风险诊断：

   \[
   \Delta R_{\mathrm{terrain}}
   =R\!\left(\tau(\rho,\beta)\right)
   -R\!\left(\tau(\rho,0)\right),
   \]

   其中参考轨迹保持相同前进距离 \(\rho\)，仅将转向量置零。该量只进入训练奖励和日志，不进入 Actor 观测，也不修改执行动作。

## 结果

| 运行 | 训练 dmax 降幅 | 评测 dmax ratio | success | collision | timeout | 地形动作 MSE | 路径风险改善 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `near`（缺少 flatness reward，诊断无效） | 28.36% | 0.2398 | 2.44% | 9.86% | 87.70% | 0.0084 | -0.59% |
| `flatness_fix` | 28.89% | 0.3111 | 0.98% | 0 | 99.02% | 0.0203 | -0.47% |
| `reward_focus` | 17.47% | 0.4945 | 0.49% | 16.50% | 83.01% | 0.0023 | -0.49% |
| `quintic_risk` | 25.00% | 0.3295 | 1.56% | 9.96% | 88.48% | 0.0075 | -0.42% |
| `relative_quintic` | 26.94% | **0.2047** | **5.18%** | 9.67% | 85.16% | 0.0073 | -0.16% |
| `relative_only` | 17.49% | 0.3954 | 1.66% | 49.80% | 48.54% | 0.0048 | **+0.88%** |

六个运行均未通过 screen。所有有效运行中 Actor、neighbor encoder 和 terrain encoder 均发生更新，动作非退化，且没有 NaN/Inf，因此失败不是训练链路或网络未更新导致。

## 结果解释

- `relative_quintic` 最接近几何收敛门限，但 success 仍只有 5.18%，地形路径风险也未优于地形置零对照。
- `relative_only` 是唯一得到正路径风险改善的版本，但改善仅 0.88%，同时 collision 上升到 49.80%。这说明相对风险信号具有可学习性，但单独强化它会破坏团队集合和安全目标。
- `reward_focus` 同时恶化 dmax、collision 和地形敏感性，继续扫描同类权重缺乏依据。
- 近距筛选中 `full_message_ratio=1`、`sparse_message_ratio=0`、`far_pair_ratio=0`。当前失败不能归因于远距消息年龄或更新周期，因此不满足 B1 启用条件。
- 基线运行中的重复车辆对冲突不足以解释主要 timeout；`relative_only` 的冲突随碰撞共同增加。因此目前也不满足 B2 的启用依据。

### 共享团队 advantage 信用诊断

`shared_joint` 当前对四辆车使用同一个团队奖励序列计算 GAE，并把同一个 advantage 复制给四组 Actor 样本。对 `relative_quintic` 最优 checkpoint 进一步使用冻结 critic 的一步 TD residual 作为 advantage 代理，在 61,440 个团队样本、245,760 个车辆样本上得到：

| 相关关系 | Pearson | 秩相关 |
| --- | ---: | ---: |
| 相对路径风险 vs. TD advantage 代理 | -0.0123 | -0.0072 |
| 相对路径风险 vs. 单车质心距离进度 | -0.0044 | -0.0043 |
| 相对路径风险 vs. 最近邻距离变化 | 0.0037 | 0.0036 |
| 相对路径风险 vs. 团队 dmax 进度 | -0.0118 | -0.0102 |

相关绝对值均不超过 0.0123。该结果不证明某一种新奖励必然有效，但支持一个具体判断：当前共享团队 advantage 几乎不包含可用于区分单车地形选择的信用信息。`relative_only` 通过放大地形项得到轻微风险改善，却同步破坏碰撞安全，与该判断一致。

## 决策

按停止规则，不启动 B0 40M，不启动 B1、B2 或 B3，也不恢复安全投影和后处理。信用诊断已经确认共享 advantage 与单车地形结果近乎不相关。下一项工作应先形成一份窄范围的信用分配对照设计，再决定是否修改 `shared_joint` 的奖励/advantage 语义；在规划更新和测试门限明确前不直接启动新训练，也不继续盲目扫描奖励权重。

## 产物

- 汇总：`outputs/runs/exp125_decentralized_tiered_b0_pure_rl/_suite/metrics/b0_screen_comparison.json`
- 单次运行：`outputs/runs/exp125_decentralized_tiered_b0_pure_rl/<run_id>/`
- 每个运行的门限判定：`metrics/screen_gate.json`
- 地形对比：`metrics/terrain_contrast.json`
- 信用诊断：`metrics/credit_assignment.json`
- 配置快照：`config/experiment.yaml`
