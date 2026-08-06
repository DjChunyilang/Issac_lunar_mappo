# exp127：冻结联合动作 Critic 可行性诊断

## 目的

exp126 表明，单独给 Actor 分配车辆地形信用会增强地形响应，但会破坏车辆交互安全。exp127 在不更新 Actor、不改变奖励和不替换 MAPPO 的前提下，检验连续联合动作 Critic 是否具有足够的离线可辨识性，从而决定是否允许提出 C1 反事实信用训练方案。

源策略为 exp125 `relative_quintic` 的 seed23 4M checkpoint。诊断模型不进入 Actor、控制器、环境或正式 checkpoint。

## 方法

冻结 Actor 与原集中式 (V(s))，按 checkpoint 的随机策略标准差采样四车联合动作。对长度为 (H) 的片段构造：

\[
Y_t^{(H)}=
\sum_{k=0}^{H-1}\gamma^k r_{t+k}
+\gamma^H
\left(\prod_{k=0}^{H-1}(1-d_{t+k})\right)
V(s_{t+H}).
\]

在相同训练样本和监督目标上分别拟合：

\[
\widehat Q_s=f_\theta(s_t),
\qquad
\widehat Q_{sa}=g_\phi(s_t,a_{t,1},\ldots,a_{t,4}).
\]

两者均为两层、每层 128 单元的 ELU 网络。状态模型有 23,681 个参数，联合动作模型有 24,705 个参数。使用模型种子 7、17、29，并在独立地形种子 15023、16023 上验证。

逐车反事实量定义为：

\[
\Delta Q_i=
\widehat Q_{sa}(s,\mathbf a)
-\widehat Q_{sa}
\left(s,(\mathbf a_{-i},\mu_i(s_i))\right),
\]

其中只把车辆 (i) 的随机动作替换为其策略均值。预先规定的晋级门限为：held-out MSE 至少改善 15%，安全结果秩相关绝对值至少为 0.30，且两个验证地形种子均满足门限。

## 采样覆盖修正

最初工程验证采用 128/96 个环境步，只覆盖约 25.6/19.2 秒，几乎没有后段冲突和碰撞，不能支撑安全归因结论。正式诊断保持约六万训练样本的规模，但改为：

- 训练：128 个环境、480 步，完整覆盖 96 秒；
- 验证：每个地形种子 64 个环境、480 步；
- 16 步目标的训练样本 59,520 个；
- 每个验证种子 29,760 个样本。

完整覆盖后，两个验证集的预测冲突参与率分别为 17.06% 和 16.03%，因此最终否定结论不是由冲突样本缺失造成的。

## 结果

| 目标时域 | 联合动作 MSE 平均改善 | 最差地形种子改善 | 最大安全秩相关 | 结论 |
| ---: | ---: | ---: | ---: | --- |
| 1 步 | -0.019% | -0.030% | 0.0645 | 未通过 |
| 4 步 | -0.114% | -0.266% | 0.0557 | 未通过 |
| 16 步 | 2.157% | 1.892% | 0.0896 | 未通过 |

16 步结果中，单车 ​\(\Delta Q_i\) 与预测冲突、实际碰撞和相对路径风险的秩相关均接近零。Actor 参数哈希前后相同，探针动作最大变化为 0，排除了诊断过程修改策略的可能。

早期短覆盖数据曾在 1 步目标上显示 18.55% 的改善；完整 episode 复核后该改善消失，说明它是只采样初始易阶段造成的分布偏差，不能作为 C1 依据。

## 结论

exp127 状态为 `stop_before_c1`：

- 不实施 C1；
- 不将 COMA、MADDPG、FACMAC 或 attention critic 接入当前主线；
- 不通过增加 Critic 容量或扫描时域门限推翻预先规定的停止规则；
- 保留诊断模型仅用于结果复核。

## 产物

- `outputs/runs/exp127_joint_action_critic_feasibility/frozen_exp125_relative_quintic_seed23/`
- `metrics/counterfactual_critic_feasibility.json`
- `artifacts/diagnostic_critics.pt`
- `full_episode_horizon_1/` 与 `full_episode_horizon_4/`

