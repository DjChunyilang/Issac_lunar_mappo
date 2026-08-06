# exp130：团队 PPO 与地形信用梯度冲突审计

## 目的

exp126 的零和地形信用增强了地形响应，却使 collision 激增；exp129 又表明地形风险与轨迹安全的局部动作方向整体近似正交。exp130 进一步检验该冲突是否真实进入共享 Actor 的训练梯度，而不是只存在于动作空间。

该实验不更新参数、不修改优化器，也不改变环境奖励。

## 方法

冻结 exp125 `relative_quintic` checkpoint，严格按照 `SharedPolicyMAPPO` 的样本顺序重建：

- 团队 GAE policy gradient ​\(g_{\mathrm{team}}\)；
- exp126 零和相对地形信用 trace gradient ​\(g_{\mathrm{terrain}}\)。

梯度冲突以余弦定义：

\[
c=
\frac{g_{\mathrm{team}}^{\mathsf T}g_{\mathrm{terrain}}}
{\lVert g_{\mathrm{team}}\rVert_2
 \lVert g_{\mathrm{terrain}}\rVert_2}.
\]

当 ​\(c<0\) 时，两者在一阶近似下相互干扰。正式审计使用两个独立地形种子，每个种子 128 个环境、8 段 64 步 rollout、262,144 个 Actor 样本，并随机计算 32 个 4,096 样本梯度批次。

启用后续计划的预设门限为：两个种子的负余弦比例均不低于 20%，且辅助/主梯度范数中位比均位于 ​\([0.05,20]\)。

## 结果

| 范围 | seed18023 负余弦比例 | seed19023 负余弦比例 | 梯度范数中位比 |
| --- | ---: | ---: | ---: |
| 全 Actor | 46.875% | 46.875% | 1.110 / 1.164 |
| terrain encoder | 50.000% | 50.000% | 1.417 / 1.485 |
| trunk | 53.125% | 50.000% | 1.086 / 1.193 |

全 Actor 余弦均值仅为 `0.0238/0.0021`，10% 分位数为 `-0.506/-0.524`。Actor checkpoint 哈希保持完全不变。

## 结论

exp130 状态为 `allow_c2_plan_only`。exp126 的失败与真实优化冲突一致：约一半更新批次中，直接相加的地形信用梯度会反对团队 PPO 梯度。该结果只允许提出一次主任务优先的梯度投影筛选，不自动授权 40M。

理论依据为 [Yu et al., “Gradient Surgery for Multi-Task Learning,” NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)。本项目采用非对称、主任务优先变体，不宣称等同于原始对称 PCGrad。

## 产物

- `outputs/runs/exp130_actor_gradient_conflict_audit/frozen_exp125_seed23/`
- `metrics/actor_gradient_conflicts.json`

