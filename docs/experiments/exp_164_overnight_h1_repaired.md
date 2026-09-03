# exp164：修复Bottleneck上的隔离H1夜间长训

更新时间：2026-08-21。当前状态：已完成，未通过。

## 目的

在Active-DSTC尚未接入Actor前，隔离验证一个问题：四车已经获得同一个可行站点区域条件时，标准MAPPO低层是否能学会集合、避碰和稳定保持。

本实验不是完整Active-DSTC，也不是strict去中心化最终结果。407维Actor中的站点势场由训练环境给定，Oracle进展奖励权重为0.5，只用于低层能力上界。

## 配置

```text
configs/experiment/exp164_overnight_h1_repaired_n1.yaml
scripts/run_exp164_overnight_h1.py
```

关键参数：

```yaml
seed: 23
parallel_envs: 256
rollout_length: 64
policy_iterations: 4800
timesteps: 307200
environment_interactions: 78643200
actor: multiscale_n1_cnn
critic: structured_multiscale_v3
advantage: gae
bc_updates: 0
init_checkpoint: null
```

三个固定阶段各1600 iterations。Stage C使用Mixed/Bottleneck和远距分布，Bottleneck陨石坑数为30，保留瓶颈墙和通道。熵系数从0.0009到0.0001，在完整307,200步内衰减。

## 结果边界

- 只训练标准GAE，不启用DAE或PRD；
- 不使用BC或旧checkpoint；
- 不使用安全投影、方向性mask、集合槽位或Actor后处理；
- 不把该run作为Active-DSTC最终策略；
- 若长训后success仍处于地板，停止Pure RL低层路线，优先实现R4 GATHER。

## 运行状态

CUDA 256环境、rollout 64真实smoke已通过：Actor、Critic、neighbor encoder和terrain encoder均发生非零更新，无NaN，BC更新为0。

正式run已完成307,200步、78,643,200环境交互，墙钟时间24,604.8 s，约6.83小时：

```text
outputs/runs/exp164_overnight_h1_repaired/n1_seed23_full_4800iter/
```

状态入口：

```text
outputs/runs/exp164_overnight_h1_repaired/_suite/suite_status.json
```

训练日志：

```text
outputs/runs/exp164_overnight_h1_repaired/n1_seed23_full_4800iter/metrics/train_metrics.jsonl
```

## 结果

最终独立评测：

| 指标 | 结果 | 门限 | 判定 |
| --- | ---: | ---: | --- |
| dmax ratio | 0.0961 | 不高于0.20 | 通过 |
| success | 86.98% | 不低于90% | 未通过 |
| collision | 12.50% | 不高于2% | 未通过 |
| timeout | 0.52% | 低于10% | 通过 |
| final dmax | 1.092 m | 1.25 m | 通过 |
| final dispersion | 0.2538 | 0.30 | 通过 |
| final mean speed | 0.124 m/s | 0.25 m/s | 通过 |
| final flatness | 91.67% | 结果诊断 | — |

Strict结果为：

```text
passed: false
failed_checks: [success_rate, collision_rate]
```

Stage B近距Open的最佳冻结点出现在timestep 179,200：success 98.44%、collision 1.30%、timeout 0.26%、dmax ratio 0.169。说明共同站点已知时，标准MAPPO在近距Open能够形成有效低层策略。

进入Stage C后，success为58.59%–83.59%，collision为11.98%–26.82%；最终Stage C评测为success 80.99%、collision 18.23%、timeout 0.78%。最终策略动作约88.01%为前进、11.51%为倒车，S形让行仅0.48%，原地转向为0。复杂地形和远距通信下没有形成足够的显式让行协调。

## 失败分析

训练已经完成4800次Actor/Critic更新，所有分支参数均显著改变且无NaN。Stage B已经证明策略并非完全学不会集合；Stage C持续高碰撞说明失败不是简单训练预算不足，而是从近距Open扩展到Mixed/Bottleneck和远距通信时的协调泛化失败。

继续延长同一训练没有预注册依据。`best.pt`按`selection_gate=final`指向307,200步checkpoint，但该checkpoint只是`candidate`，不得作为推荐策略。179,200步checkpoint只在Stage B分布上表现良好，未经Stage C独立验证，也不能作为最终结果。

## 结论

H1低层Pure RL得到部分成功：能够集合、dmax和timeout均明显改善，但collision远超门限。停止继续延长或训练seed31/47。下一步按既定计划使用R4显式读取邻车承诺原语，验证去中心化GATHER；不通过增加安全投影或Actor后处理修补本run。

## 评测动画

同一307,200步checkpoint生成两段诊断动画：

```text
videos/proxy_eval_success_far_mixed_seed16401.gif
videos/proxy_eval_collision_far_bottleneck_seed16400.gif
```

成功样例在147步完成，final dmax为1.192 m且实际质心平整度通过。碰撞样例在141步终止；车辆已经接近平坦目标，但末段发生碰撞。动画只用于观察行为，不改变本实验strict失败结论。

## 明日判读

优先读取最后一次阶段冻结评测和最终`metrics/summary.json`，检查：

- success是否脱离0；
- dmax reduction是否持续改善；
- collision和timeout是否随Stage B/C恶化；
- hold概率及倒车、转向、让行动作族是否塌缩；
- 正常地形下路径风险；
- 三个尺度terrain encoder是否持续更新。

GIF和TensorBoard曲线只能用于诊断，不能替代冻结评测。
