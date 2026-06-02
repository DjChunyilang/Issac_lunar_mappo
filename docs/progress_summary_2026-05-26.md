# 项目进度说明与下一步规划（2026-05-26）

## 当前阶段结论

项目已经从“环境与接口搭建”推进到“简化 proxy 环境可训练、可评估、可展示”的阶段。

当前主训练路径仍是 PyTorch kinematic proxy env，不使用 Isaac Sim 物理仿真，也不依赖渲染。这个选择是有意保留的：proxy 环境吞吐高、可快速验证观测、奖励、终止、控制链路，并且适合做多车集合策略的第一阶段算法调试。

Isaac Sim / PhysX 当前定位为高保真验证和展示层，已经可以加载官方 Jetbot 资产、运行单车/四车 smoke evaluation，并输出截图或 GIF。它尚未进入正式训练 loop。

截至本记录，项目已经得到一个初步收敛的四车集合 checkpoint：

```text
outputs/checkpoints/exp_004_proxy_converged.pt
```

独立评估结果表明，在 256 个并行 proxy env、100 步 rollout 下：

```text
initial_dmax: 7.2260
final_dmax: 0.8900
dmax_reduction_ratio: 0.1232
success_rate: 0.9453
collision_rate: 0.0703
timeout_rate: 0.0
```

这满足当前“趋势优先”的初步收敛标准：`final_dmax / initial_dmax <= 0.4`。

## 阶段 A/B 严格收敛完成记录（2026-05-26）

本轮已经完成阶段 A/B：固化 proxy 收敛实验、加入安全约束调优、完成 pure RL / BC-only / BC+PPO 三组对照，并用严格安全标准验收 BC+PPO 主结果。

严格验收标准：

```text
3 个 seeds 全部满足：
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

最终主配置：

```text
configs/experiment/exp_005_safety_tuned_bc_ppo.yaml
```

最终 checkpoint：

```text
outputs/checkpoints/exp_005_safety_tuned_best.pt
```

BC+PPO 严格验收结果：

| seed | dmax_reduction_ratio | success_rate | collision_rate | timeout_rate | best phase |
| --- | ---: | ---: | ---: | ---: | --- |
| 23 | 0.1340 | 1.0000 | 0.0000 | 0.0000 | PPO |
| 31 | 0.1474 | 1.0000 | 0.0000 | 0.0000 | BC |
| 47 | 0.1478 | 0.9961 | 0.0059 | 0.0000 | BC |

结论：BC+PPO 三个固定 seed 全部通过严格标准，未触发 retry。pure RL 在同等预算下未通过，BC-only 已能达到严格门槛，BC+PPO 在 seed 23 上进一步把 dmax ratio 从约 0.148 降到 0.134。当前结果应表述为“safety-aware scripted BC warm-start + PPO 微调”的严格 proxy 收敛，不应表述为纯 RL 从零收敛。

阶段 A/B proxy 产物：

```text
outputs/logs/exp_005_safety_tuned/suite_summary.json
outputs/logs/exp_005_safety_tuned/strict_acceptance.json
outputs/logs/exp_005_safety_tuned/comparison_curves.png
outputs/logs/exp_005_safety_tuned/safety_diagnostics.png
outputs/videos/exp_005_safety_tuned/best_proxy_rollout.gif
```

PhysX 展示 sanity check 已完成，使用最终 checkpoint 在粗糙地形上跑四 Jetbot 渲染评估。PhysX 不作为严格收敛 gate，只作为高保真展示和迁移 sanity check。

PhysX 结果摘要：

```text
status: ok
terrain: rough
n_agents: 4
steps: 100
final_dmax: 0.5434
mean_dmax: 1.1362
final_dispersion: 0.0670
max_tilt_deg: 6.37
collision_count: 3
physics_updates_per_s: 71.50
```

PhysX 展示产物：

```text
outputs/logs/physx_four_jetbots/evaluation_exp005.json
outputs/figures/physx_four_jetbots/evaluation_exp005_scene.png
outputs/videos/physx_four_jetbots/evaluation_exp005_rollout.gif
```

本轮回归测试：

```text
.venv_isaaclab/bin/python -m pytest
23 passed
```

## PPO-only Best 与训练曲线管理补充（2026-05-28）

针对“best checkpoint 不应来自 warm-up 阶段”的问题，本轮新增 `exp_006_ppo_selected` 实验族：

- BC warm-start 仍用于初始化 actor，但 BC 评估只记录为 baseline。
- `algorithm.best_source: ppo` 时，只有 PPO 阶段评估可以竞争 best checkpoint。
- `algorithm.required_best_phase: ppo` 时，严格验收额外检查 `best_metrics.phase == "ppo"`。
- 每个 run 写出 TensorBoard scalar 到 `tensorboard/` 子目录，同时保留原有 JSONL、PNG、GIF。
- PPO 更新加入 reference-policy anchor，降低从 BC 初始化后漂移退化的风险。

新增主配置和对照配置：

```text
configs/experiment/exp_006_ppo_selected_bc_ppo.yaml
configs/experiment/exp_006_ppo_selected_weak_warmstart.yaml
configs/experiment/exp_006_ppo_selected_pure_rl.yaml
```

主配置相对 exp_005 的主要变化：

```text
num_envs: 1024
total_env_steps: 2_000_000
eval_interval_updates: 1
learning_rate: 5e-5
clip_epsilon: 0.15
ppo_epochs: 2
entropy_coef: 0.002 -> 0.0002
reference_policy_coef: 1.0 -> 0.25
best_source: ppo
required_best_phase: ppo
```

exp_006 BC+PPO 三 seed PPO-only 严格验收结果：

| seed | BC baseline ratio | PPO best update | PPO best ratio | success_rate | collision_rate | timeout_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | 0.1478 | 15 | 0.1438 | 1.0000 | 0.0000 | 0.0000 |
| 31 | 0.1474 | 10 | 0.1456 | 1.0000 | 0.0000 | 0.0000 |
| 47 | 0.1478 | 8 | 0.1474 | 1.0000 | 0.0000 | 0.0000 |

结论：三 seed 的最终 best checkpoint 均来自 PPO 阶段，并且全部通过严格标准。PPO 的绝对提升幅度不大，但它不再只是沿用 warm-up checkpoint；seed 47 还把 BC baseline 中的少量碰撞消除为 0。

exp_006 最终产物：

```text
outputs/checkpoints/exp_006_ppo_selected_best.pt
outputs/logs/exp_006_ppo_selected/suite_summary.json
outputs/logs/exp_006_ppo_selected/strict_acceptance.json
outputs/logs/exp_006_ppo_selected/final_eval_best.json
outputs/logs/exp_006_ppo_selected/comparison_curves.png
outputs/videos/exp_006_ppo_selected/best_proxy_rollout.gif
```

最终 checkpoint 独立复评：

```text
checkpoint: outputs/checkpoints/exp_006_ppo_selected_best.pt
num_envs: 512
steps: 160
dmax_reduction_ratio: 0.1419
success_rate: 0.9980
collision_rate: 0.0020
timeout_rate: 0.0
```

TensorBoard 查看命令：

```bash
.venv_isaaclab/bin/tensorboard \
  --logdir outputs/logs/exp_006_ppo_selected \
  --port 6006
```

已写出的 TensorBoard run：

```text
outputs/logs/exp_006_ppo_selected/bc_ppo_seed_23/tensorboard/
outputs/logs/exp_006_ppo_selected/bc_ppo_seed_31/tensorboard/
outputs/logs/exp_006_ppo_selected/bc_ppo_seed_47/tensorboard/
outputs/logs/exp_006_ppo_selected/weak_warmstart_seed_23/tensorboard/
```

弱 warm-start 对照：

| run | bc_steps | PPO best ratio | success_rate | collision_rate | timeout_rate | strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| weak_warmstart_seed_23 | 50 | 0.1812 | 0.2383 | 0.0000 | 0.7734 | false |

结论：50 step 弱 warm-start 下，PPO 能把 dmax ratio 压到严格阈值以内，但无法稳定满足 success / timeout 标准。结合 exp_005 pure RL 三 seed 均失败的结果，当前任务仍依赖足够强的 warm-start；后续若要减少 warm-start，需要增加课程学习、成功判据分阶段、或更长训练预算，而不是简单减少 BC step。

## 已完成能力

### 1. Proxy 多车集合任务

已经实现 4 车集合任务的核心闭环：

- rover 代理运动学积分
- actor observation 与 centralized critic state
- `[rho, beta]` 高层动作解释
- 局部子目标到世界子目标转换
- 直线轨迹生成
- 简化速度跟踪控制
- 几何集合奖励、oracle reward、安全惩罚、终端奖惩
- 成功、碰撞、越界、timeout 终止判据
- 地形结构化特征占位与程序化地形特征

目前 actor observation 不包含 oracle 信息；oracle 只进入 critic / reward 相关路径，保持 actor 执行阶段信息约束。

### 2. 训练与评估

已经具备三类训练/评估入口：

```text
scripts/train.py
scripts/train_skrl_mappo.py
scripts/train_proxy_convergence.py
```

其中 `train_proxy_convergence.py` 是当前收敛主入口，采用：

- scripted gathering controller 行为克隆 warm-start
- shared actor + centralized critic
- PPO rollout / update
- deterministic eval 选择 best checkpoint

评估入口：

```text
scripts/evaluate_proxy_policy.py
scripts/play.py
scripts/validate_first_stage.py
```

主要产物：

```text
outputs/logs/exp_004_proxy_convergence/train_metrics.jsonl
outputs/logs/exp_004_proxy_convergence/eval_metrics.json
outputs/logs/exp_004_proxy_convergence/final_eval.json
outputs/logs/exp_004_proxy_convergence/convergence_curves.png
outputs/logs/exp_004_proxy_convergence/eval_rollout.gif
```

### 3. Isaac Sim / PhysX 高保真验证

已新增并验证：

```text
scripts/physx_jetbot_smoke.py
scripts/evaluate_physx_four_jetbots.py
scripts/view_proxy_rovers_isaac.py
```

当前能力：

- Isaac Sim viewport 可启动
- 官方 Jetbot 资产可加载
- 单 Jetbot 平地 / 崎岖地形 PhysX smoke 可运行
- 四 Jetbot closed-loop evaluation 可运行
- 可输出截图和 GIF

PhysX 层当前用于验证和展示，不参与 policy rollout 采样。

### 4. 测试覆盖

当前回归命令：

```bash
.venv_isaaclab/bin/python -m pytest
```

最近验证结果：

```text
23 passed
```

测试覆盖包括：

- action interpreter
- trajectory generator
- simplified controller
- proxy rover model
- four-rover observation space
- terrain feature
- reward / termination
- CUDA short training smoke
- convergence tools

## 当前重要限制

1. 当前收敛 checkpoint 不是纯 RL 从零训练结果。

   当前 best checkpoint 来自 scripted controller warm-start 阶段。PPO 阶段保持了较高 reward，但未进一步超过 BC 阶段的 dmax ratio。因此后续如果需要强调“强化学习自主收敛”，必须补充 pure RL 或弱 warm-start 对照实验。

2. Proxy 动力学仍是简化模型。

   当前训练没有真实轮地接触、悬挂、打滑、翻车、崎岖地形动力学扰动。它适合验证高层集合策略和算法链路，但不能直接说明真实月面 rover 动力学下的性能。

3. Isaac Sim 目前主要是展示 / 高保真评估层。

   如果后续希望 PhysX 进入训练 loop，需要重新设计吞吐、并行环境、动作接口、reset、观测同步和失败恢复机制。

4. Jetbot 不是月球车。

   Jetbot 是当前可快速使用的官方轮式资产，适合验证 Isaac Sim 差速轮控制链路，但尺寸、轮地接触和越障能力都不等价于月球 rover。后续应评估 NovaCarter 或自定义 rover USD/URDF。

5. Python 环境依赖组合需要保持稳定。

   当前 Isaac Sim / Isaac Lab / torch / skrl 已能运行，但 `pip check` 会报告部分依赖元数据冲突。不要随意全量升级 pip 包。

## 后续工作规划

### 阶段 A：固化 proxy 收敛实验（已完成）

目标：让当前初步收敛结果可复现、可解释、可对比。当前已通过 `exp_005_safety_tuned` suite 完成。

计划：

1. 固化 `exp_004_proxy_convergence` 的实际推荐参数。
2. 将 `--bc-steps 200` 写入推荐命令或新增专用配置，避免默认 2000 step 误导后续运行。
3. 增加训练阶段耗时统计、BC loss 曲线、PPO reward 曲线、dmax ratio 曲线。
4. 增加 pure RL、BC-only、BC+PPO 三组对照实验。
5. 明确报告口径：当前最佳结果是 warm-start convergence，不是 pure RL convergence。

建议验收：

```text
固定 3 个 seed，每个 seed final_dmax / initial_dmax <= 0.4
至少 1 个 seed success_rate >= 0.9
所有 seed timeout_rate 接近 0
```

### 阶段 B：改进奖励和安全约束（已完成）

目标：降低碰撞率，提高策略行为质量，而不只追求 dmax 收缩。当前 BC+PPO 三个 seed 已全部满足严格安全标准。

计划：

1. 分析 `final_eval.json` 中 collision_rate 约 7% 的来源。
2. 调整 near-distance / collision penalty，使集合过程保留更合理间距。
3. 增加队形收缩速度约束，避免过快冲向中心导致碰撞。
4. 增加 min inter-agent distance 曲线和碰撞时间点统计。
5. 将成功标准从趋势指标逐步提升到 success_rate + collision_rate 的联合指标。

建议目标：

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

### 阶段 C：PhysX 高保真闭环评估

目标：验证 proxy checkpoint 在 Isaac Sim 轮式资产和崎岖地形中的迁移表现。

计划：

1. 用 `exp_004_proxy_converged.pt` 跑四 Jetbot PhysX 完整 episode。
2. 输出 high-fidelity evaluation JSON，包括 dmax、dispersion、碰撞、翻车、卡住、sim throughput。
3. 保存一段短视频或 GIF，用于直观检查四车集合过程。
4. 对比 proxy rollout 和 PhysX rollout 的轨迹误差。
5. 如果 Jetbot 崎岖地形稳定性不足，再评估 NovaCarter 或自定义 rover 资产。

建议命令：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --terrain rough \
  --steps 100 \
  --checkpoint outputs/checkpoints/exp_004_proxy_converged.pt \
  --output outputs/logs/physx_four_jetbots/evaluation_exp004.json
```

### 阶段 D：真实 rover 资产与控制接口选型

目标：决定是否从 Jetbot 迁移到更接近月球车的模型。

计划：

1. 调研 Isaac Sim 官方轮式资产和可用 rover / UGV 资产。
2. 明确目标控制接口：差速轮速度、Ackermann、转向角 + 轮速、或力矩控制。
3. 如果没有合适官方资产，制定自定义 USD/URDF 的最小规格。
4. 定义 PhysX 评估环境的地形尺度、重力、摩擦和轮地接触参数。
5. 在不改变高层 action `[rho, beta]` 的前提下，替换底层控制映射。

### 阶段 E：文档与实验管理

目标：让后续实验不会混淆“proxy 训练”“PhysX 验证”“渲染展示”三条链路。

计划：

1. 将实验命令、结果、产物路径统一记录到 `docs/`。
2. 为每个实验配置建立清晰命名：`exp_004_proxy_convergence`、`exp_005_safety_tuned`、`exp_006_physx_eval`。
3. 保留每次关键 checkpoint 对应的配置和 final_eval。
4. 在 README 中增加推荐运行顺序。

## 推荐近期执行顺序

1. 复验当前收敛 checkpoint：

```bash
.venv_isaaclab/bin/python scripts/evaluate_proxy_policy.py \
  --config configs/experiment/exp_004_proxy_convergence.yaml \
  --checkpoint outputs/checkpoints/exp_004_proxy_converged.pt \
  --device cuda \
  --num-envs 256 \
  --steps 100
```

2. 跑四车 PhysX 评估：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --terrain rough \
  --steps 100 \
  --checkpoint outputs/checkpoints/exp_004_proxy_converged.pt \
  --output outputs/logs/physx_four_jetbots/evaluation_exp004.json
```

3. 调整安全奖励并形成 `exp_005_safety_tuned.yaml`。

4. 做 pure RL / BC-only / BC+PPO 对照，明确论文或报告中采用的实验口径。

## 当前推荐判断

短期内不要把 Isaac Sim 放进主训练 loop。更稳妥的路线是：

```text
proxy 高吞吐训练
-> proxy 数值验证
-> PhysX 高保真 closed-loop evaluation
-> 渲染视频展示
```

只有当 PhysX 评估中出现 proxy 无法解释的系统性失败时，再考虑把部分高保真动力学引入训练或做 domain randomization。
