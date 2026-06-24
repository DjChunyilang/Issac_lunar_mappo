# exp015 SKRL + BC Warm-up + 偏弱中档地形

## 目的

验证 86 维 `ego_v3_local_terrain_grid` 观测在偏弱中档月面上的正式 SKRL MAPPO 训练能力。先运行 seed23 的 2M env-step 趋势筛选；通过后，从随机初始化重新执行 BC20 和独立 8M env-step 正式训练。

## 配置

配置文件：

```text
configs/experiment/exp015_skrl_weak_warmup_medium_soft.yaml
```

关键设置：

- 2048 个 CUDA 环境，rollout 128，episode 220 steps。
- shared Actor `86→128→128→2`，centralized/shared Critic `54→128→128→1`。
- 训练前执行 20 次 scripted-teacher BC；MAPPO 阶段不加入 teacher loss。
- MAPPO 使用 4 epochs、16 mini-batches、`lr=1.2e-4`、entropy `0.006`。
- 地形 amplitude `0.08`，crater radius `0.40–1.20 m`，depth/diameter `0.10`。
- 静态地形 sanity 约为：高度范围 `0.395 m`、最低通行性 `0.353`、平均速度比例 `0.664`。

## 严格标准

2M 筛选门槛：

```text
dmax_reduction_ratio <= 0.30
success_rate >= 0.50
collision_rate <= 0.03
timeout_rate <= 0.50
```

同时要求无非有限值、策略参数更新、terrain 输入权重更新、BC 更新有效且动作非退化。

8M 正式门槛：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

## 结果表

| seed | run_id | budget | 状态 | 说明 |
| ---: | --- | ---: | --- | --- |
| 23 | `screen_seed23_2m` | 2,097,152 env steps | 未通过 | dmax ratio `0.818`，success `0`，collision `0.124`，timeout `0.876`。 |
| 23 | `formal_seed23_8m` | 8,388,608 env steps | 未启动 | screen gate 未通过，runner 按设计停止。 |

seed23 即使通过，也只记录为 single-seed candidate，不替代 seeds 31、47 的正式扩展。

## 失败分析

工程和训练信号检查全部通过：

- BC loss 从 `0.828` 降至 `0.434`。
- policy 参数变化 L2 为 `2.282`。
- terrain 输入列权重变化 L2 为 `0.653`。
- 训练后动作标准差为 `0.415`，没有动作退化。
- observation schema、86/54 维接口和有限值检查均通过。

失败来自任务表现而不是训练链路：四个趋势指标全部未达门槛，尤其 collision `0.124` 和 timeout `0.876` 明显偏高。当前 BC20 + 2M MAPPO 在该地形和安全约束下没有形成可进入 8M 正式阶段的候选策略。

## 产物路径

```text
outputs/runs/exp015_skrl_medium_soft_terrain_grid/screen_seed23_2m/
outputs/runs/exp015_skrl_medium_soft_terrain_grid/formal_seed23_8m/
outputs/runs/exp015_skrl_medium_soft_terrain_grid/_suite/metrics/
```

每 512 vector timesteps 保存一个 `ppo_timestep_*.pt`；训练结束后逐个独立评估并选择 `checkpoints/best.pt`。

## 结论

exp015 的 CUDA 2M screen 已完成且未通过趋势门槛，因此没有启动 8M formal。并行运行期间未发生 OOM；该结果不能报告为收敛或 strict pass。

## 下一步

先分析 BC teacher 与 collision/timeout 的关系，以及 512/1024 timestep 两个候选的独立评估差异。不要直接扩大到 8M 或 seeds 31、47；下一轮应优先调整安全感知 warm-up、动作分布或训练课程，再重新进行 2M screen。
