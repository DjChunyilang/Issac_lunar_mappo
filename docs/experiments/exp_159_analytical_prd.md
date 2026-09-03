# exp159：解析式Leave-One-Out PRD-MAPPO

更新时间：2026-08-20。

## 目的

exp158表明，学习完整团队总奖励的DAE reward model无法准确估计policy-weighted反事实期望。exp159不再预测反事实奖励，而是在当前团队奖励完全不变的前提下，为每辆车构造只依赖其他车辆动作和状态的单步解析基线。

本实验回答：该基线能否降低共享团队策略梯度方差，同时保持全数据梯度方向和范数基本不变。

## 方法边界

ALO-PRD只改变训练期Actor advantage：

$$
A_{i,t}^{\mathrm{PRD,raw}}
=
A_t^{\mathrm{team,raw}}
-b_{i,t}^{\mathrm{LOO}}.
$$

LOO基线不进行时间递推；Critic仍拟合原团队return。Actor观测、47维动作、通信、轨迹和控制器不变。

该实现是受PRD启发的项目特定解析基线，不等同于带attention的PRD-AC或shared-reward PRD-MAPPO。

## LOO基线

基线包含：

- 其他三车的energy、motion和consistency贡献；
- 其他三车的可加地形成本；
- 其他三车的路径最大风险；
- H1中其他三车的Oracle距离进展；
- 在排除本车后重新计算的三车近距惩罚；
- 其他三车内部碰撞惩罚；
- 其他三车内部碰撞或越界导致的failure penalty。

dmax、dispersion、实际质心平整度、success hold、success bonus和timeout始终留在团队GAE。

与exp150不同，exp159不要求修正项在车辆间逐步零和。单一碰撞对发生碰撞时，参与车辆保留原团队惩罚，非参与车辆移除与自身当前动作无关的碰撞与failure噪声；不会再次加重参与车辆惩罚。

## 接口

```yaml
task:
  analytical_prd_enabled: true

algorithm:
  advantage_estimator: analytical_prd_loo
  prd:
    baseline_scale: 1.0
    temporal_trace: false
    preserve_team_reward: true
```

环境 `info` 新增：

```text
analytical_prd.reward_sources.node: [E,4]
analytical_prd.loo_baseline: [E,4]
analytical_prd.source_reconstruction_error: [E]
analytical_prd.own_action_invariance_error: [E,4]
```

rollout只新增 `[T,E,4]` 的 `prd_loo_baseline`，不增加模型参数或部署字段。

## 工程状态

```text
implementation_complete
unit_and_regression_tests_passed
cpu_smoke_passed
cuda_256_rollout64_smoke_passed
formal_h1_offline_gate_failed
strict_audit_not_started
training_stopped_by_gate
```

正式shape smoke结果：

| 指标 | 结果 | 门限 |
| --- | ---: | ---: |
| Actor初始化hash | GAE/PRD一致 | 必须一致 |
| Critic初始化hash | GAE/PRD一致 | 必须一致 |
| PRD/GAE吞吐 | 约99.7% | 至少90% |
| PRD峰值CUDA显存 | 约6.49 GB | 不超过8 GB |
| Actor样本数 | 65,536 | $64\times256\times4$ |

Smoke只证明工程链路，不证明PRD基线有足够覆盖率或降低梯度方差。

## 正式离线审计

分别审计：

- A-H1：407维观测、Oracle权重0.5、exp157 t134400 checkpoint；
- A-strict：295维观测、Oracle权重0、exp156 N1 t153600 checkpoint。

每套使用128环境×480步训练分布、两个各64环境×480步验证seed，并在六分层共384个状态上枚举四车各47个动作。

全部门限包括：

- 团队奖励逐元素不变；
- source重构误差不超过 $10^{-6}$；
- 遍历本车47动作时LOO基线变化不超过 $10^{-6}$；
- collision参与车不被二次惩罚；
- baseline标准差、非零率和相对raw advantage覆盖率达标；
- baseline-only梯度范数不超过team梯度的10%；
- team/PRD全数据梯度余弦至少0.95，范数差不超过10%；
- 两个验证seed的bootstrap梯度方差均至少降低15%；
- 逐车PRD advantage标准差大于 $10^{-4}$。

A-H1失败后不运行A-strict和完整训练。A-H1通过但A-strict失败时，只允许H1机制验证。

## 完整训练预算

审计通过后，H1比较标准GAE与ALO-PRD：

```yaml
parallel_envs: 256
rollout_length: 64
stage_iterations: [800, 800, 800]
total_timesteps: 153600
environment_interactions_per_run: 39321600
bc_updates: 0
init_checkpoint: null
```

seed23完整配对通过后才运行seed31和47。H1三seed及A-strict均通过后，才进入295维strict配对。最终严格门限仍为每seed、每分层192个episode中的collision `0/192`、success至少 `180/192`、timeout最多 `12/192`，以及dmax ratio置信上界不超过0.20。

## 入口与产物

```bash
.venv_isaaclab/bin/python scripts/run_exp159_prd_validation.py \
  --phase offline --device cuda:0
```

产物：

```text
outputs/runs/exp159_analytical_prd/
├── offline_h1_audit/
├── offline_strict_audit/
├── h1_{gae|prd}_seed*/
├── strict_{gae|prd}_seed*/
└── _suite/
```

## A-H1正式结果

正式审计完成128环境×480步训练分布、两个各64环境×480步验证集，以及384个冻结状态中的72,192个本车动作分支。结果为：

| 检查 | seed15023 | seed16023 | 门限 | 通过 |
| --- | ---: | ---: | ---: | --- |
| 团队奖励变化 | 0 | 0 | 必须为0 | 是 |
| source重构误差 | 0 | 0 | 不超过 $10^{-6}$ | 是 |
| 本车47动作基线最大变化 | $2.98\times10^{-8}$ | $2.98\times10^{-8}$ | 不超过 $10^{-6}$ | 是 |
| baseline-only梯度/团队梯度 | 8.38% | 1.30% | 不超过10% | 是 |
| team/PRD梯度余弦 | 0.9996 | 0.9999 | 至少0.95 | 是 |
| 梯度范数差 | 7.92% | 0.13% | 不超过10% | 是 |
| baseline/raw advantage绝对值比 | 3.45% | 2.22% | 至少10% | 否 |
| 梯度方差降低 | 7.06% | 0.42% | 每个seed至少15% | 否 |
| PRD逐车advantage标准差 | 0.00319 | 0.00596 | 大于 $10^{-4}$ | 是 |

表中的本车47动作变化、奖励保持和source重构为六分层整体最大值；两套验证seed共享同一结果。解析基线满足action-independent baseline的核心不变量，也没有像exp150一样改变参与车梯度方向，但其幅度太小，不能显著降低团队advantage噪声。

### 统计解释边界

验证状态为 `ANALYZED`。11项统计谬误检查全部覆盖：全部预注册门限均已报告，没有从总体均值替代双seed门限，也没有从smoke或单一碰撞场景推断训练收益。需保留的限制是：梯度方差基于冻结exp157 t134400策略分布；它能否代表其他策略阶段尚未独立复现。不过两个seed均远低于15%门限，且覆盖率同时失败，因此停止结论不依赖单个临界指标。

## 当前结论

ALO-PRD工程实现正确，并验证了团队奖励不变、本车动作无关性和梯度方向一致性；但可移除噪声只占raw团队advantage的2.2%–3.4%，梯度方差只降低0.42%–7.06%。按预注册规则，A-H1失败后不运行A-strict，也不启动任何完整训练。不得通过扩大基线来源、降低门限、增加多步trace或改用学习相关集合补救。
