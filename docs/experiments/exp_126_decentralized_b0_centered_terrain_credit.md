# exp126：B0 零和地形信用残差 4M 对照

## 目的

验证 exp125 暴露的信用稀释假设：在不改变执行链路、网络结构、集中式 Critic 和团队奖励的前提下，只给共享 Actor 增加车辆特定的地形信用，能否同时改善地形路径选择和团队集合。

该实验是单一信用分配对照，不是新的网络候选。配置为 `configs/experiment/exp126_decentralized_b0_centered_terrain_credit.yaml`。

## 方法

车辆 \(i\) 的相对 quintic 路径风险为：

\[
\Delta R_{t,i}
=R\!\left(\tau_{t,i}(\rho,\beta)\right)
-R\!\left(\tau_{t,i}(\rho,0)\right).
\]

即时信用及车辆间零和残差为：

\[
c_{t,i}=-\Delta R_{t,i},
\qquad
\widetilde c_{t,i}=c_{t,i}-\frac{1}{N}\sum_{j=1}^{N}c_{t,j}.
\]

车辆信用经过 \(\gamma\lambda_c\) trace 后，只加入 Actor advantage：

\[
C_{t,i}=\widetilde c_{t,i}
+\gamma\lambda_c(1-d_t)C_{t+1,i},
\qquad \lambda_c=0.95,
\]

\[
A^{\mathrm{actor}}_{t,i}
=A^{\mathrm{team}}_t
+0.25\,\operatorname{Norm}(C_{t,i}).
\]

原团队 reward 不进行数值修改，并直接写入 Critic memory。车辆信用使用独立 memory tensor，只参与共享 Actor 的 PPO surrogate。该设计受 agent-specific counterfactual advantage 思想启发，但不是完整 COMA，不增加动作条件 Q critic或反事实动作采样。

## 工程验证

- CPU 8 环境 smoke：通过；
- CUDA 256 环境 smoke：通过；
- Actor/neighbor/terrain encoder：均有效更新；
- 最后一次 credit trace 标准差：`1.0`；
- 每步车辆信用和最大误差：`2.98e-8`；
- 团队 reward 修改误差：`0`；
- Critic 仍使用 131,072 个团队样本，Actor 使用 524,288 个车辆样本。

第一次 2048 环境启动在训练更新前被奖励均值保护检查中止。原因是早期实现把信用直接加到 float32 reward 后再求均值。随后改为原 reward 与 Actor credit 分离存储；中止运行不计入训练结果。

## 4M 结果

正式运行：`centered_credit_seed23_4m_v2`，seed23、2048 环境、2048 训练时步，共 4,194,304 次环境交互。checkpoint 筛选选择 `t=1024`。

| 指标 | exp125 `relative_quintic` | exp126 C0 | 变化 |
| --- | ---: | ---: | ---: |
| 训练首末四分之一 dmax 降幅 | 26.94% | -1.70% | 明显退化 |
| 评测 dmax ratio | 0.2047 | 0.6932 | 退化 |
| success | 5.18% | 0.29% | 退化 |
| collision | 9.67% | 64.84% | 严重退化 |
| timeout | 85.16% | 34.86% | 下降主要来自碰撞提前终止 |
| 地形置零动作 MSE | 0.0073 | 0.0198 | 地形敏感性增强 |
| 路径风险改善 | -0.16% | +2.74% | 方向改善但未达到 5% |
| 预测冲突/步 | 0 | 0.2006 | 严重增加 |
| 重复车辆对冲突/步 | 0 | 0.1927 | 严重增加 |

## 结论

C0 未通过 4M screen，不启动 40M，也不扫描信用系数或 trace 参数。

实验同时给出两个有区分度的结论：

1. 车辆特定信用能增强 terrain branch 对路径风险的响应，因此 exp125 的信用稀释诊断是有效问题，而不是纯相关性假象。
2. 只分配地形信用、仍让碰撞后果完全共享，会使车辆学会地形差异却不能识别自身对车辆冲突的边际影响。collision 和重复冲突的同步上升说明该信用定义不完整。

下一步不能简单继续增大地形信用，也不能用安全投影掩盖碰撞。若继续信用分配方向，必须先研究能够同时表达地形和车辆交互边际效应的单一反事实 advantage，并明确其高水平文献依据、集中训练信息边界和计算成本；在新计划批准前不实施。

## 产物

- `outputs/runs/exp126_decentralized_b0_centered_terrain_credit/centered_credit_seed23_4m_v2/`
- `metrics/screen_gate.json`
- `metrics/terrain_contrast.json`
- `_suite/metrics/b0_screen_summary.json`

