# exp158：DAE-MAPPO两级有效性验证

更新时间：2026-08-19。

## 目的

本实验依次判断：47维动作下的反事实即时奖励能否被可靠估计；共同站点已知时DAE能否改善低层信用；若前两项通过，DAE能否改善295维严格去中心化端到端任务。

DAE只修改训练期advantage。Actor、通信、47维差速原语、轨迹和左右轮控制均不改变。标准shared-joint MAPPO是唯一对照，不同时加入GRU、GNN、注意力、可学习通信或新的奖励分量。

## 当前状态

```text
implementation_complete
engineering_smoke_passed
formal_offline_gate_failed
h1_training_stopped_by_gate
strict_training_gated
```

`exp157-H1`旧run只完成134,400/153,600步，没有最终配对评测，标记为 `incomplete_interrupted`。其checkpoint仅作为离线行为策略，不作为exp158基线，也不恢复训练。

## DAE接口

反事实即时奖励期望为：

$$
\overline r_{i,t}
=
\sum_{a_i'=1}^{47}
\pi_{\theta_{\mathrm{old}}}(a_i'\mid o_{i,t})
\widehat r_\psi(s_t,\mathbf a_{-i,t},a_i').
$$

反事实轨迹和逐车advantage为：

$$
C_{i,t}
=
\overline r_{i,t}
+\gamma\lambda\beta
(1-d_t^{\mathrm{episode}})C_{i,t+1},
$$

$$
A_{i,t}^{\mathrm{DAE,raw}}
=
A_t^{\mathrm{GAE,raw}}-\beta C_{i,t}.
$$

`terminated`和`truncated`都会切断 $C_{i,t}$。Critic继续拟合原团队return；四车DAE advantage拼接后统一标准化。$eta=0$时直接使用原GAE路径。

β日程固定为：

- 更新1—128：$eta=0$；
- 更新129—256：线性升至0.3；
- 更新257—2400：$eta=0.3$。

不扫描β。

## 奖励模型

训练期 `CounterfactualRewardModel` 读取950维集中状态、其他三车离散动作、查询车辆索引和候选动作。模型批量输出 `[B,4,47]`，但只用实际执行动作对应的团队总即时奖励监督。模型不读取Actor隐藏状态，不进入环境步或部署文件。

模型采用独立的多尺度状态编码器、16维动作embedding、8维查询车辆embedding和 `216→256→128→1` ELU主干。奖励模型optimizer、随机数和参数均与Actor/Critic隔离。

rollout新增：

```text
joint_actions: [T,E,4]
old_action_probabilities: [T,E,4,47]
team_rewards: [T,E,1]
```

训练checkpoint在 `dae_training` 字段保存奖励模型、optimizer、更新计数和β；字段显式标记 `deployable: false`。部署Actor仍只包含policy state dict。

## 离线前置门限

正式审计使用旧H1的 `ppo_timestep_134400.pt`：

- 128环境×480步事实训练集；
- 两个独立seed，各64环境×480步验证集；
- 70%策略采样与30%五动作族均衡探索；
- 六个固定分层各64个冻结状态；
- 每个状态固定其他车辆动作，枚举四车各47种替代动作，共72,192个真实反事实标签。

实现使用完整环境状态快照与SHA-256恢复检查进行分支，而不是修改真实执行链。反事实标签只用于验证，不能训练奖励模型。

全部门限通过才允许H1训练：

- 长时影响与bootstrap联合门限不指向时间信用；
- 冲突参与车边际贡献中位数至少是非参与车的2倍；
- 共享advantage与真实边际贡献Spearman绝对值小于0.20；
- observation aliasing不超过20%；
- 两个验证seed的总奖励动作条件MSE改善均至少15%；
- gather、near-safety、path-risk至少两项改善15%；
- 反事实预测标准差大于 $10^{-4}$；
- 各地形seed动作排序Spearman至少0.30；
- policy-weighted期望误差不超过真实奖励标准差的0.25。

失败后不启动训练，也不扩大模型、增加RNN、辅助损失或扫描β。

## 配对训练

H1阶段比较407维 `H1-GAE` 与 `H1-DAE`，均保留Oracle进展权重0.5。通过三seed门限后，strict阶段再比较295维 `Strict-GAE` 与 `Strict-DAE`，Oracle奖励权重为0。

每个run固定：

```yaml
parallel_envs: 256
rollout_length: 64
stage_iterations: [800, 800, 800]
total_timesteps: 153600
environment_interactions: 39321600
bc_updates: 0
init_checkpoint: null
```

seed23每组均须完整训练；通过后才运行seed31和47。所有比较使用相同初始化hash、固定课程和1152场景清单，不从中间checkpoint择优。

## 评测门限

H1 seed23除反事实模型门限外，还须满足：success至少0.50、collision不高于0.20、dmax ratio不高于0.45；相对GAE的success提高至少10个百分点或timeout相对下降至少20%；collision和dmax恶化上界分别不超过0.02；路径风险不恶化。

strict最终验收保持每seed、每分层192个episode：

- collision为 `0/192`；
- success至少 `180/192`；
- timeout最多 `12/192`；
- dmax ratio点估计及单侧95%上界均不高于0.20。

## 工程验证

已完成：

- DAE递推、$eta=0$ GAE及Actor梯度等价性；
- 47动作向量化与显式循环一致性；
- query动作mask、trace终止和奖励模型optimizer隔离；
- 环境快照单步精确重放；
- 407维H1 CPU smoke；
- 407维H1 CUDA 256环境、rollout 64的真实DAE更新；
- GAE/DAE Actor及Critic初始化hash完全相同；
- CUDA峰值显存约6.51 GB；
- DAE相对GAE smoke吞吐约90.4%，通过60%门限。

Smoke只证明工程可运行，不代表离线门限或策略有效性通过。

## 入口与产物

```bash
.venv_isaaclab/bin/python scripts/run_exp158_dae_validation.py \
  --phase offline --device cuda:0
```

离线通过后才允许：

```bash
.venv_isaaclab/bin/python scripts/run_exp158_dae_validation.py \
  --phase h1 --seed 23 --device cuda:0
```

产物位于：

```text
outputs/runs/exp158_dae_validation/
├── offline_credit_audit/
├── h1_{gae|dae}_seed*/
├── strict_{gae|dae}_seed*/
└── _suite/
```

## 正式离线结果

正式审计完成61,440个事实训练样本、两个各30,720样本的验证集、384个冻结状态和72,192个真实反事实标签。结果如下：

| 检查 | 结果 | 门限 | 通过 |
| --- | ---: | ---: | --- |
| $M_{>64}$ | 0.0362 | 不与显著bootstrap改善同时达到0.30 | 是 |
| 冲突参与/非参与边际贡献中位数比 | 1.366 | 至少2.0 | 否 |
| 共享advantage—真实边际Spearman | 0.2004 | 绝对值小于0.20 | 否 |
| observation aliasing | 0 | 不超过0.20 | 是 |
| 总奖励动作模型最差MSE改善 | -5.45% | 每个验证seed至少15% | 否 |
| gather动作增益 | 最差51.92% | 至少15% | 是 |
| near-safety动作增益 | 最差42.82% | 至少15% | 是 |
| path-risk动作增益 | 最差6.90% | 至少15% | 否 |
| 最小反事实预测标准差 | 0.3442 | 大于 $10^{-4}$ | 是 |
| 最差地形seed动作排序Spearman | 0.3023 | 至少0.30 | 是 |
| policy-weighted期望误差 | 2.006个真实奖励标准差 | 不超过0.25 | 否 |

失败不是因为模型输出常量，也不是因为64步rollout遗漏了主要影响。模型能够粗略排列候选动作，但无法校准其奖励数值；而DAE实际扣除的是policy-weighted期望值，因此排序通过不能替代期望误差门限。

### 统计解释边界

本结果的验证状态为 `ANALYZED`，不是独立复现实验后的 `VERIFIED`。统计谬误检查覆盖11/11项：未发现总体与分层方向反转、群体到个体错误外推、collider控制、只报告显著结果或训练结果因果倒置；所有预注册门限均已报告。需要保留两项谨慎解释：冻结状态来自一个未完成H1 checkpoint，存在行为分布选择限制；高冲突和低dmax状态为有意分层抽样，不能把参与者比例解释为自然发生率。

精确动作分支支持“单步动作改变导致即时奖励变化”的局部因果解释，但本轮没有执行DAE完整训练，因此不能推断DAE必然降低或提高最终success。结论只适用于当前团队总奖励、47维动作、950维状态和既定前馈reward model。

## 结论

DAE工程实现有效，但当前团队总即时奖励不足以支持可靠的单车反事实基线。按预注册规则，H1-GAE/H1-DAE完整训练不启动，也不扩大reward model、不增加RNN或辅助损失、不扫描β。该结果不能解释为“所有信用分配方法均无效”；它只否定当前总奖励、单车DAE和既定前馈reward model的组合。
