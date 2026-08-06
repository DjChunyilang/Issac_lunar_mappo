# exp142：真实碰撞PPO-Lagrangian组件筛选计划

## 目的

exp141证明，现有54维集中式state能够稳定估计未来64步真实碰撞风险。exp142据此只增加一个训练期collision cost critic和一个标量Lagrange乘子，检验显式约束目标能否降低真实碰撞。它不增加执行模块，不使用近距整形、预测冲突或MAPF标签。

本实验仍是组件级4M筛选；即使通过也不直接进入40M。

## 约束定义

每步团队cost只取真实碰撞终止：

\[
c_t=\mathbb I[\text{collision termination at }t].
\]

新增训练期集中式cost value：

\[
V_c(s_t)=f_{\psi}(s_t),
\]

其中 \(f_{\psi}\) 为两层128单元ELU网络，输入与现有集中式Critic完全相同。cost GAE使用：

\[
\delta_t^c
=
c_t+\gamma_c(1-d_t)V_c(s_{t+1})-V_c(s_t),
\]

\[
A_t^c
=
\delta_t^c+\gamma_c\lambda_c(1-d_t)A_{t+1}^c.
\]

固定 \(\gamma_c=0.99\)、\(\lambda_c=0.95\)。reward advantage和cost advantage分别在联合batch上标准化，Actor使用：

\[
A_t^{\mathrm{actor}}
=
\widehat A_t^r
-
\lambda_k\widehat A_t^c.
\]

两项仍复制给四辆共享Actor；本实验不宣称解决逐车反事实责任，只验证团队碰撞约束。

## 乘子更新

严格目标collision rate为2%。每个64步rollout估计episode等效碰撞率：

\[
\widehat J_c
=
480\,
\frac{\sum c_t}{64N_{\mathrm{env}}}.
\]

乘子更新为：

\[
\lambda_{k+1}
=
\operatorname{clip}
\left(
\lambda_k+0.1(\widehat J_c-0.02),
0,2
\right).
\]

固定 \(\lambda_0=0\)，不使用EMA、PID项或参数扫描。cost critic使用独立Adam优化器，学习率 `3e-4`，value loss系数 `0.5`；现有Actor/reward Critic优化器、团队reward和reward return保持不变。

## 保持不变

- exp125 `relative_quintic`的101维Actor、通信、地形奖励、课程和执行链路；
- Pure RL随机初始化，`bc_updates=0`、`init_checkpoint=null`；
- 所有安全投影、末段覆盖、槽位修正和显式目标关闭；
- cost critic、cost advantage和乘子不作为执行期依赖；
- 不使用near potential、repeated conflict、Oracle或CBS生成cost；
- 不同时加入terrain Actor信用、图注意力或梯度投影。

## 工程门限

- cost目标逐元素等于真实collision done；
- cost GAE在collision终止后不bootstrap，且不跨reset；
- reward GAE与B0在相同rollout上完全一致；
- 团队reward和现有reward Critic参数更新语义不变；
- cost critic只读取集中式state，Actor输入仍为101维；
- dual更新公式、上下界和episode等效换算通过单测；
- CPU与CUDA smoke中cost critic、Actor和reward Critic均更新，lambda有限且cost loss无NaN；
- checkpoint必须记录cost critic与lambda，但加载为执行策略时忽略二者不得改变Actor输出。

## 4M组件门限

seed23、2048环境、2048训练时步，共4,194,304次环境交互。只有以下条件全部满足，才允许制定后续统一地形—安全训练计划：

- 训练末四分之一episode等效碰撞率相对首四分之一至少降低30%；
- 独立评测collision不高于 `0.0677`；
- success不低于 `0.0318`；
- dmax ratio不高于 `0.2547`；
- 两个固定诊断种子的重复冲突中位数均相对B0降低至少20%；
- 最终lambda位于 \((0.05,2.0)\)，未长期钳位上界；
- cost value loss有限，cost value参数发生有效更新；
- Actor、neighbor encoder、terrain encoder和reward Critic均更新且无NaN；
- 仍报告terrain-contrast，但本组件通过不能替代后续terrain门限。

## 停止规则

任一门限失败，则停止约束优化方向，不扫描cost预算、lambda学习率、上界、PID项或cost网络容量，不恢复安全投影。通过也只允许先制定一个统一方案，不直接启动40M。

## 已完成结果

工程实现、单元测试、CPU smoke和CUDA smoke均通过。实现保持101维严格去中心化Actor与原执行链路不变；collision cost critic只读取54维集中式state，并使用独立Adam更新。checkpoint同时记录cost critic与Lagrange乘子，但执行加载只读取Actor权重。工程事实源为：

- `outputs/runs/exp142_collision_lagrangian_component/_suite/metrics/engineering_gate.json`；
- `outputs/runs/exp142_collision_lagrangian_component/collision_lagrangian_seed23_4m/metrics/summary.json`；
- `outputs/runs/exp142_collision_lagrangian_component/collision_lagrangian_seed23_4m/metrics/component_gate.json`。

正式4M训练完成32次联合更新。训练期episode等效碰撞率从首四分之一均值 `1.0963` 降至末四分之一 `0.05447`，相对降低 `95.03%`；最终乘子为 `1.0484`，未触及上界。独立评测collision为 `0.02539`，两个固定诊断种子的失败episode重复冲突中位数均降为0。这说明真实碰撞约束具有明确优化作用。

然而，安全改善由回避集合获得：独立评测success为 `0`、dmax ratio为 `0.68146`、timeout为 `0.97461`。terrain-contrast动作MSE仅 `0.000259`，路径风险改善仅 `0.0259%`。因此策略没有形成“同时集合与避碰”的可行折中，组件门限中的success与dmax两项失败。

## 结论与停止决策

exp142状态为 `stopped_at_component_gate`，不授权40M。按照预注册停止规则：

- 不扫描collision预算、乘子学习率、乘子上界、PID项或cost网络容量；
- 不叠加Actor信用、图注意力、安全投影或新的后处理模块；
- cost critic与Lagrange乘子保留为可复现实验实现，不进入当前正式架构；
- 后续问题应重新聚焦“集合主任务与地形路径信号为何不足”，不能把低碰撞单独解释为收敛。
