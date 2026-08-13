# 多智能体信用分配技术附录

更新时间：2026-08-12。

本附录给出[研究综述](marl_credit_assignment_review.md)所依据的公式、当前实现审计、CTDE边界和后续 DAE-MAPPO 有界对照接口。本文只形成研究结论和后续实验规范，不修改正在运行的 N0/N1/N2。

## 1. 问题定义

将四车任务写成 Dec-POMDP：

$$
\mathcal M=
\left(
\mathcal N,\mathcal S,
\{\mathcal O_i\},
\{\mathcal A_i\},
P,r,\gamma
\right),
\qquad |\mathcal N|=4.
$$

执行期联合策略保持因子分解：

$$
\boldsymbol\pi_\theta(\mathbf a_t\mid\mathbf o_t)
=
\prod_{i=1}^{4}
\pi_\theta(a_{i,t}\mid o_{i,t}),
$$

其中共享参数 $\theta$ 不表示共享实时观测。每个 $o_{i,t}$ 只能包含车辆 $i$ 的291维局部观测和合法通信缓存。

集中训练时允许 Critic 使用全局状态 $s_t$，目标仍为：

$$
J(\theta)
=
\mathbb E_{\boldsymbol\pi_\theta}
\left[
\sum_{t=0}^{T-1}\gamma^t r_t
\right].
$$

## 2. 当前 shared-joint MAPPO 语义

环境只产生一个团队奖励，并复制到四辆车：

$$
r_{i,t}=r_t,\qquad i=1,\ldots,4.
$$

当前实现使用第一份 memory 的团队奖励和集中式 $V_\phi(s)$ 计算一次 GAE：

$$
\delta_t
=
r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t),
$$

$$
A_t^{\mathrm{GAE}}
=
\sum_{l=0}^{L-1}
(\gamma\lambda)^l
\left(
\prod_{k=0}^{l-1}(1-d_{t+k})
\right)
\delta_{t+l}.
$$

随后执行：

$$
A_{i,t}^{\mathrm{actor}}=A_t^{\mathrm{GAE}}.
$$

代码对应关系如下：

- `gathering_env.py` 将 `terms.total` 扩展到四辆车；
- `shared_policy_mappo.py` 计算一份 `advantages`；
- `actor_advantages = advantages.repeat(4, 1)` 形成四车 Actor 样本。

该实现不是错误的 MAPPO 语义，但它没有显式估计单车边际贡献。

## 3. 历史诊断链

### 3.1 团队 advantage 与单车结果弱相关

exp125 在61,440个团队样本、245,760个车辆样本上得到：

| 关系 | Pearson | Spearman |
| --- | ---: | ---: |
| 相对路径风险与TD advantage代理 | -0.0123 | -0.0072 |
| 相对路径风险与单车集合进度 | -0.0044 | -0.0043 |
| 相对路径风险与最近邻变化 | 0.0037 | 0.0036 |
| 相对路径风险与团队dmax进度 | -0.0118 | -0.0102 |

这些结果支持“共享 advantage 没有区分单车地形选择”，但不直接证明某种信用算法一定有效。

### 3.2 即时奖励的动作可辨识性不均衡

定义状态模型和状态—联合动作模型对分量 $m$ 的动作增益：

$$
I_m
=
\frac{\operatorname{MSE}_{s,m}-\operatorname{MSE}_{s,\mathbf a,m}}
{\operatorname{MSE}_{s,m}}.
$$

exp128 的主要结果为：

| 目标 | $I_m$ |
| --- | ---: |
| gather | 50.998% |
| nearest-neighbor change | 56.182% |
| predicted conflict involvement | 17.557% |
| safety reward | 0.099% |
| terrain reward | -0.261% |
| total reward | 0.127% |

动作能够影响集合和连续安全变化，但当前加权总奖励掩盖了这种影响。这正是 DAE 必须先验证 reward model 的原因。

### 3.3 联合动作 Q Critic 的旧接口可辨识性不足

exp127 拟合：

$$
\widehat Q_s=f(s_t),
\qquad
\widehat Q_{s\mathbf a}=g(s_t,\mathbf a_t)
$$

对1、4、16步目标，联合动作模型的最好 MSE 改善只有 `2.157%`，单车反事实 $\Delta Q_i$ 与安全结果的最大秩相关只有 `0.0896`。这直接降低了 COMA、FACMAC 类联合动作 Q Critic 的优先级。

### 3.4 人工局部信用不是反事实优势

历史信用组件采用：

$$
A_{i,t}^{\mathrm{actor}}
=
A_t^{\mathrm{team}}+\alpha\widetilde c_{i,t},
\qquad
\sum_i\widetilde c_{i,t}=0.
$$

逐步零和只保证平均团队信用不变，并不保证 $\widetilde c_{i,t}$ 是动作的因果边际贡献。exp126、exp140 和 exp150 的工程不变量全部通过，但碰撞分别恶化到 `64.84%`、`22.95%` 和 `99.90%`。因此后续不得再把事件参与标签直接当作反事实 advantage。

## 4. 主要信用方法

### 4.1 COMA

$$
A_i^{\mathrm{COMA}}(s,\mathbf a)
=
Q(s,\mathbf a)
-
\sum_{a_i'}\pi_i(a_i'\mid o_i)
Q(s,(\mathbf a_{-i},a_i')).
$$

优点是单车边际定义清晰；缺点是需要可信的联合动作长期 Q。当前 exp127 证据不支持优先实现。

### 4.2 DAE

DAE 学习即时奖励模型：

$$
\widehat r_{\psi,i}(s_t,\mathbf a_{-i,t},a_i').
$$

实际动作的监督损失为：

$$
\mathcal L_r(\psi)
=
\frac{1}{2}
\left(
r_t-
\widehat r_{\psi,i}(s_t,\mathbf a_{-i,t},a_{i,t})
\right)^2.
$$

反事实期望为：

$$
\overline r_{i,t}
=
\sum_{a_i'=1}^{40}
\pi_\theta(a_i'\mid o_{i,t})
\widehat r_{\psi,i}(s_t,\mathbf a_{-i,t},a_i').
$$

DAE advantage 为：

$$
A_{i,t}^{\mathrm{DAE}}
=
\sum_{l=0}^{L-1}
(\gamma\lambda)^l
\left[
r_{t+l}
-\beta^{l+1}\overline r_{i,t+l}
+\gamma V_\phi(s_{t+l+1})
-V_\phi(s_{t+l})
\right].
$$

这里 $\beta$ 越大，单车反事实扣除越强，但 reward model 误差和策略偏差也越大。本项目若获准训练，固定：

$$
\beta=0.3,
\qquad
\gamma=0.99,
\qquad
\lambda=0.95.
$$

这里的 `0.3` 是保守的工程预注册值，不是论文给出的通用最优值。其目的，是在首个反事实奖励模型仍可能存在估计误差时限制多步扣除强度；本轮不扫描 $\beta$，也不使用 $\beta=1$ 的完整差分奖励。

为便于实现，可将上式拆成“原团队 GAE 减去单车反事实轨迹”：

$$
C_{i,t}
=
\overline r_{i,t}
+\gamma\lambda\beta(1-d_t)C_{i,t+1},
$$

$$
A_{i,t}^{\mathrm{DAE}}
=
A_t^{\mathrm{GAE}}
-\beta C_{i,t}.
$$

其中，$d_t=1$ 表示回合在时刻 $t$ 结束。第一式从当前时刻向后累计“如果只重新采样车辆 $i$ 的动作，预期会得到多少团队奖励”，并以 $\gamma\lambda\beta$ 衰减；第二式从团队 GAE 中扣除该车辆的反事实基线。由于每辆车的 $\overline r_{i,t}$ 不同，四辆车最终获得的 Actor advantage 也不同。

注意：当前奖励依赖动作导致的下一状态、quintic路径和联合几何。不能把已知奖励公式误写成“无需模型即可精确计算所有替代动作奖励”。首轮必须学习训练期 reward model；不得为40个动作分别推进真实 Isaac 环境。

### 4.3 PRD

若团队奖励可以拆成多个局部来源 $r_j$，PRD 为车辆 $i$ 选择相关集合 $R_i$：

$$
A_i^{\mathrm{PRD}}
=
\sum_{j\in R_i}A_j.
$$

本项目当前只有一条混合团队奖励。人为定义 $R_i$ 会重新引入 exp126–150 的奖励归因假设，因此 PRD 只保留为机器人领域相关证据，不进入首轮实现。

### 4.4 值分解与 IGM

QMIX 要求：

$$
\arg\max_{\mathbf a}Q_{\mathrm{tot}}(s,\mathbf a)
=
\left(
\arg\max_{a_1}Q_1(\tau_1,a_1),\ldots,
\arg\max_{a_N}Q_N(\tau_N,a_N)
\right).
$$

其单调混合是实现 IGM 的充分结构条件。QTRAN 和 QPLEX 扩展可表达函数类；SHAQ 用 Markov Shapley value 解释局部价值。共同代价是切换为 off-policy TD、replay buffer 和 target network，与当前 MAPPO 形成大范围混杂变量。

### 4.5 Shapley

对于四辆车：

$$
\phi_i
=
\sum_{S\subseteq\mathcal N\setminus\{i\}}
\frac{|S|!(3-|S|)!}{4!}
\left[v(S\cup\{i\})-v(S)\right].
$$

每辆车需要8个联盟边际量。计算规模不大，主要困难是如何定义可信的联盟价值 $v(S)$。若仍依赖未通过辨识门限的 Q Critic，则精确枚举不会消除模型偏差。因此只用于冻结诊断。

## 5. 空间信用、时间信用与观测混叠诊断

### 5.1 时间影响曲线

冻结场景和随机数，只扰动车辆 $i$ 在时刻 $t$ 的动作，记录各奖励分量的影响：

$$
\Delta r_{i,t}(k)
=
r_{t+k}^{\mathrm{intervene}(i)}-r_{t+k}^{\mathrm{base}}.
$$

定义 rollout 64步以外的影响占比：

$$
M_{>64}
=
\frac{
\sum_{k>64}\gamma^k|\Delta r_{i,t}(k)|
}{
\sum_{k\ge0}\gamma^k|\Delta r_{i,t}(k)|+\varepsilon
}.
$$

只有 $M_{>64}\ge0.30$，并且价值 bootstrap 误差同时显著时，才把主要失败归因于长时间信用。此时先比较 rollout 128 或 recurrent Critic；不直接实施 HCA、CCA、COCOA 或 RUDDER。

### 5.2 空间边际贡献

在短窗旁路中只替换一辆车动作：

$$
\Delta_i
=
G(s,\mathbf a)
-G(s,(\mathbf a_{-i},a_i')).
$$

若真实冲突参与车的 $|\Delta_i|$ 中位数至少是非参与车的2倍，且共享 GAE 与 $\Delta_i$ 的 Spearman 相关低于 `0.20`，则空间信用不足得到额外支持。

对重复车辆对冲突还需估计二阶交互：

$$
I_{ij}
=
Q(s,\mathbf a)
-\mathbb E_{a_i}Q(s,\mathbf a)
-\mathbb E_{a_j}Q(s,\mathbf a)
+\mathbb E_{a_i,a_j}Q(s,\mathbf a).
$$

若只有联合改变车辆 $i,j$ 才能解除冲突，则 $I_{ij}$ 可能显著，而单车 DAE 或 COMA 仍接近零。此时不得把 DAE 失败解释为“信用分配整体无效”；只有冻结诊断证明二阶项能够稳定区分冲突延续与解除，才允许在下一轮研究多层级信用，本轮不实施 MACA。

### 5.3 观测可辨识性

对 Actor 观测与通信缓存的近邻样本，比较离线短窗最优动作。若相近输入对应互斥让行/绕行动作的比例超过 `20%`，则存在 observation aliasing。此时 DAE、COMA 或 Shapley 即使在训练时给出正确梯度，也不能使同一无记忆 Actor 输入同时输出两个互斥动作，必须先处理观测、历史或协调对称性。

## 6. DAE-MAPPO 后续接口规范

本节只定义 N0/N1/N2 结束后、离线门限通过时的实现边界。

### 6.1 Reward model

训练期新增：

```text
CounterfactualRewardModel
input:
  centralized_state
  joint_discrete_actions with agent i masked
  queried_agent_index (training-only)
output:
  [batch, 4, 40] counterfactual immediate rewards
```

约束：

- 输入不加入 Actor observation；
- 实际执行动作位置用于监督相应 logit 的奖励预测；
- 网络只在 PPO update 中调用，不在 environment step 和部署时调用；
- 第一版使用单个训练期 reward model，不同时增加 attention、GNN 或 Shapley head；
- 若无 RNN 的模型不能通过离线门限，停止 DAE，不自动增加 recurrent reward model。

### 6.2 Rollout storage

每步额外保存：

```text
joint_actions: [env, 4]
team_reward: [env, 1]
actor_action_probabilities: [env, 4, 40]
```

集中式状态已存在。存储不得加入执行期通信缓存。旧 standard-MAPPO checkpoint 不加载 reward model；DAE run 必须从随机初始化开始并使用独立 `training_semantics`。

### 6.3 Actor update

PPO ratio、clip、熵、Actor结构和 optimizer 设置保持不变，只将复制的团队 GAE 替换为 $A_{i,t}^{\mathrm{DAE}}$。集中式 $V(s)$ 仍学习原团队 return，环境 reward 不改写。

### 6.4 日志

至少记录：

- `reward_model_train_mse`；
- `reward_model_validation_mse`；
- `reward_model_state_only_improvement`；
- `counterfactual_reward_std`；
- `dae_advantage_agent_std`；
- `dae_vs_team_advantage_spearman`；
- `reward_model_gradient_norm`；
- 按 gather、safety、terrain 与 terminal 分量报告动作辨识度。

## 7. 工程测试

若后续实施 DAE，必须验证：

1. `beta=0` 时 advantage 与现有 GAE 在数值容差内一致；
2. 四车动作相同且 reward model 输出相同时，四车 DAE advantage 相同；
3. 只改变车辆 $i$ 的反事实输出，不影响车辆 $j\ne i$ 的动作概率输入；
4. 40维概率求和为1，反事实期望与显式循环一致；
5. episode终止、timeout和reset后 trace 不跨episode传播；
6. reward model 参数不会被保存到部署 Actor 文件或执行；
7. 固定291维本地观测和通信缓存后，修改全局状态、Oracle、reward model或未发送状态，Actor logits和最终控制命令完全不变；
8. 标准 MAPPO 与 DAE 使用相同 Actor初始化、课程、环境数、rollout和评测集合；
9. CPU小环境和CUDA 256环境前向、反向无NaN或Inf；
10. 训练吞吐和峰值显存单独报告，不因计算成本降低正式评测预算。

## 8. 决策状态

当前状态为：

```text
literature_review_complete
dae_mappo_priority_candidate
implementation_not_authorized_before_exp155_completion
offline_reward_identifiability_gate_required
```

这表示 DAE-MAPPO 是唯一优先候选，不表示其已经被项目采用，也不表示信用分配已被证明是 N0/N1/N2 失败的唯一原因。
