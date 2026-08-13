# exp156：差速轨迹原语与多尺度Actor消融

## 实验目的

本实验修正 `exp155` 暴露的三个核心问题：旧动作缺少倒车、原地转向和明确让行；熵衰减在训练前8%结束；世界原点朝向与绝对姿态形成初始化捷径。实验同时统一Critic与配对评测，避免把地形随机性误计为架构差异。

## 固定接口

- Actor观测：295维 `ego_v10_multiscale_diff_intent`；
- 动作：47维Categorical差速轨迹原语；
- Critic：950维 `structured_multiscale_v3`；
- 控制：Jackal左右轮差速映射；
- Oracle奖励权重：0；
- 初始航向：每车独立均匀随机；
- BC更新：0；
- 初始化checkpoint：空。

三种候选只改变Actor地形编码：N0展平MLP、N1共享多尺度CNN、N2路径条件CNN。

## 训练预算

原计划每种结构固定：

```text
seed23
256并行环境
64步rollout
Stage A/B/C各800 iterations
总计2400 iterations
39,321,600环境交互
```

该预算已由N0/N1完整执行；N2后来因共同地板效应取消完整训练。

## 评测协议

六个分层各192个固定场景，共1152 episode。所有候选读取同一个 `scenario_manifest`，并验证初始位姿、航向、地形参数和内容哈希。

每个分层的正式门限为：

- collision单侧95%上界不高于0.02；
- success单侧95%下界不低于0.90；
- timeout单侧95%上界严格低于0.10；
- dmax ratio点估计和单侧bootstrap 95%上界均不高于0.20。

192个episode下，离散计数要求为collision `0/192`、success至少 `180/192`、timeout最多 `12/192`。

## 工程门限结果

截至2026-08-13：

- 专项与相关回归测试通过；
- CPU smoke通过；
- N0/N1/N2的CUDA 256环境MAPPO smoke均通过；
- 原语覆盖审计12/12场景通过，其中5个为 `exp155` 后期全hold状态；
- 倒车、原地转向和S形让行均参与过有效联合解；
- 固定1152场景清单已生成。

这些结果仅表明工程链路可运行，不构成收敛证据。

## 运行入口

```bash
.venv_isaaclab/bin/python3.12 scripts/run_exp156_architecture_comparison.py --device cuda:0
```

关键产物：

```text
outputs/runs/exp156_differential_multiscale_ablation/_suite/
├── scenario_manifest.json
├── suite_status.json
├── logs/launcher.log
└── metrics/action_coverage_audit.json
```

## 正式训练结果

N0和N1均完成固定预算训练及同一1152场景清单的配对评测：

| 结构 | 环境交互 | success | collision | timeout | dmax ratio | strict分层 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N0 | 39,321,600 | 0 | 0.0729 | 0.9271 | 0.2242 | 0/6 |
| N1 | 39,321,600 | 0.0017 | 0.0130 | 0.9852 | 0.2367 | 0/6 |

N0和N1均处于成功率地板。N1降低了collision，但同时增加了timeout；该差异不能形成有效的架构优越性结论。N1只作为后续接口验证的临时CNN基线，checkpoint保持 `candidate`。

N2完整训练在第一次PPO更新时触发CUDA `grid_sample` 非连续输入错误。修复后，以256环境、16步rollout、64训练时步完成真实CUDA smoke；4次联合Actor/Critic更新完成，`policy_parameter_delta_l2=0.2106`、`terrain_encoder_parameter_delta_l2=0.0812`，参数均为有限值。产物为：

```text
outputs/runs/exp156_smoke/cuda_n2_contiguous_fix_256env_64step/
```

## 停止决策

N2完整39,321,600交互训练已取消，不再进入架构排名。原因不是N2无法工程运行，而是N0/N1共同呈现success近零、timeout超过92%的地板效应；继续比较编码器难以区分共同选址、动作可行性、协调意图和信用问题。

本实验不选择正式获胜架构，不开展Oracle奖励恢复消融，也不启动seed31/47。后续暂用N1的共享多尺度CNN接口开展H0/H1诊断；这不等于N1通过strict验收。
