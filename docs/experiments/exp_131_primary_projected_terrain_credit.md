# exp131：主任务优先地形信用梯度投影 4M 筛选

## 目的

exp130 表明，exp126 的地形信用梯度在约 46.9% 的批次中反对团队 PPO 梯度。exp131 检验：只去除地形辅助梯度中反对团队主梯度的分量，能否保留地形响应，同时避免 exp126 的碰撞退化。

配置为 `configs/experiment/exp131_decentralized_b0_primary_projected_terrain_credit.yaml`。该实验不改变环境奖励、Critic、Actor结构、通信或执行链路。

## 方法

团队 PPO 与熵正则形成主 Actor 梯度 ​\(g_p\)，exp126 的零和地形信用 surrogate 形成辅助梯度 ​\(g_a\)。当二者冲突时：

\[
\widehat g_a=
g_a-
\frac{g_p^{\mathsf T}g_a}
{\lVert g_p\rVert_2^2}g_p.
\]

辅助梯度经过范数限制后再以 exp126 原系数 `0.25` 合成：

\[
\widetilde g_a=
\widehat g_a
\min\left(1,
\frac{\lVert g_p\rVert_2}
{\lVert\widehat g_a\rVert_2+\varepsilon}
\right),
\qquad
g=g_p+0.25\widetilde g_a.
\]

实现中对 float32 投影残差增加一次极小的主方向数值修正，以保证投影后内积非负；这不改变投影语义。理论动机来自 [Yu et al., NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)，但本实验是非对称主任务优先变体。

## 工程验证

- CPU smoke：通过；
- CUDA 256 环境、64 步 rollout smoke：通过，训练核心约 2.01 秒；
- Actor 和 Critic 均完成更新；
- smoke 冲突批次比例：40.63%；
- 投影后最小主梯度内积：正值；
- 合成梯度最低主方向余弦：`0.970142`；
- 团队 reward 与 Critic return 保持原语义。

## 4M 结果

正式运行：`primary_projected_seed23_4m`，seed23、2048 环境、2048 训练时步，共 4,194,304 次环境交互。训练核心耗时约 111.7 秒，自动筛选选择 `t=2048`。

| 指标 | exp125 relative_quintic | exp126 C0 | exp131 C2 |
| --- | ---: | ---: | ---: |
| 训练 dmax 首末四分之一降幅 | 26.94% | -1.70% | 13.45% |
| 评测 dmax ratio | 0.2047 | 0.6932 | 0.3525 |
| success | 5.18% | 0.29% | 3.61% |
| collision | 9.67% | 64.84% | 66.80% |
| timeout | 85.16% | 34.86% | 29.59% |
| 地形置零动作 MSE | 0.0073 | 0.0198 | 0.0321 |
| 路径风险改善 | -0.16% | +2.74% | +2.38% |

新增投影不变量全部通过：最后一次更新中冲突比例 `65.625%`，投影后最小主梯度内积为 `5.56e-9`，合成梯度最低主方向余弦为 `0.970142`。因此失败不是投影未执行或数值异常。

为排除checkpoint选择偏差，单独复评 `t=1024`：dmax ratio `0.5662`、success `0.0078`、collision `0.6719`、timeout `0.3203`，同样失败且不优于 `t=2048`。

## 结论

exp131 未通过 4M screen，不启动 40M，也不扫描投影系数、信用系数或范数上限。主任务优先投影能够限制辅助梯度偏转并恢复部分集合趋势，但不能恢复安全性。原因与 exp128 一致：团队主梯度中的当前安全奖励几乎没有车辆级动作辨识度；仅保证地形辅助梯度不反对团队主梯度，并不等价于提供有效的车辆安全信用。

C2 保留为可复现实验机制，不属于当前采用架构。下一步若继续信用研究，只允许先离线验证由现有最近邻安全门限构造的车辆级势函数信用，不直接训练。

## 产物

- `outputs/runs/exp131_decentralized_b0_primary_projected_terrain_credit/primary_projected_seed23_4m/`
- `metrics/screen_gate.json`
- `metrics/terrain_contrast.json`
- `metrics/eval_t1024.json`
- `_suite/metrics/b0_screen_summary.json`

