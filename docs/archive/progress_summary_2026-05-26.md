# 项目进度说明与下一步规划（2026-05-26）

> 归档进度日志。当前状态请阅读 `docs/current_status.md` 和 `docs/experiments/README.md`。

## 当前阶段结论

项目已经从“环境与接口搭建”推进到“简化 proxy 环境可训练、可评估、可展示”的阶段。

当前主训练路径仍是 PyTorch kinematic proxy env，不使用 Isaac Sim 物理仿真，也不依赖渲染。这个选择是有意保留的：proxy 环境吞吐高、可快速验证观测、奖励、终止、控制链路，并且适合做多车集合策略的第一阶段算法调试。

Isaac Sim / PhysX 当前定位为高保真验证和展示层，已经可以加载官方 Jetbot 资产、运行单车/四车 smoke evaluation，并完成 lunar crater headless 多 episode evaluation。它尚未进入正式训练 loop。

截至 2026-06-02，当前推荐阶段 C checkpoint 是：

```text
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/checkpoints/best.pt
```

当前 canonical run 目录：

```text
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/
```

独立 proxy 复评结果表明，在 1024 个并行 proxy env、220 步 rollout 下：

```text
initial_dmax: 7.2538
final_dmax: 1.0163
dmax_reduction_ratio: 0.1401
success_rate: 1.0000
collision_rate: 0.0000
timeout_rate: 0.0
```

这满足当前严格 proxy 标准：`dmax_reduction_ratio <= 0.2`、`success_rate >= 0.9`、`collision_rate <= 0.02`、`timeout_rate == 0`，并且 checkpoint metrics 中 `phase=ppo`。

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
device: cuda
num_envs: 1024
rollout_steps: 128
total_env_steps: 2_000_000
actual updates: 15
steps per PPO update: 131072
eval_interval_updates: 1
eval_num_envs: 512
eval_steps: 160
gamma: 0.99
gae_lambda: 0.95
learning_rate: 5e-5
clip_epsilon: 0.15
ppo_epochs: 2
mini_batches: 8
max_grad_norm: 0.5
value_loss_coef: 0.5
entropy_coef: 0.002 -> 0.0002
reference_policy_coef: 1.0 -> 0.25
bc_steps: 300
bc_batch_size: 8192
bc_learning_rate: 1e-3
teacher_stop_radius: 0.45
teacher_slow_distance: 0.40
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

TensorBoard 重点曲线整理：

```text
00_overview/eval_reward       评估平均 reward，优先看整体回报趋势
00_overview/success_rate      任务完成率
00_overview/dmax_ratio        四车集合收敛比例，越低越好
00_overview/collision_rate    碰撞率，安全主指标
00_overview/timeout_rate      超时率，检查是否拖延未完成
00_overview/rollout_reward    PPO 采样阶段 reward，观察训练过程
```

PPO 诊断曲线放在 `01_ppo_health/`：

```text
policy_loss
value_loss
entropy
approx_kl
clip_fraction
explained_variance
reference_policy_loss
```

任务细节曲线放在 `02_task_detail/`：

```text
final_dmax
final_dispersion
mean_nearest_distance
min_nearest_distance
near_violation_rate
```

保留旧的 `bc/`、`eval/`、`ppo/`、`best/` 原始 tag，用于细查；优先阅读 `00_overview/`，再看 `01_ppo_health/` 判断 PPO 是否稳定。重点曲线选择参考了 PPO 常见日志实践：回报/episode 表现、policy/value loss、entropy、KL、clip fraction 和 value fit。

TensorBoard tag 检查命令：

```bash
.venv_isaaclab/bin/python scripts/summarize_tensorboard_tags.py \
  --logdir outputs/logs/exp_006_ppo_selected
```

旧 exp_006 训练发生在 `00_overview/01_ppo_health/02_task_detail` 新 tag 加入之前。为了不重训也能查看完整重点曲线，可以从已有 JSON/JSONL 回填到独立目录：

```bash
.venv_isaaclab/bin/python scripts/backfill_tensorboard_curated_tags.py \
  --log-root outputs/logs/exp_006_ppo_selected \
  --output-subdir tensorboard_curated \
  --overwrite
```

回填目录为每个 run 下的 `tensorboard_curated/`，不会覆盖原始 `tensorboard/`。旧日志可回填 `00_overview/`、`02_task_detail/` 以及 `01_ppo_health/policy_loss|value_loss|entropy|reference_policy_loss`；旧训练没有记录 `approx_kl`、`clip_fraction`、`explained_variance`，因此不伪造这些曲线。

默认回填会跳过名字包含 `smoke` 的短测试 run，避免 TensorBoard 中混入只有 1-2 个点的测试曲线；需要检查 smoke run 时可额外加 `--include-smoke`。

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

结论：原始 50 step 弱 warm-start 下，PPO 能把 dmax ratio 压到严格阈值以内，但无法稳定满足 success / timeout 标准。这个结论已经被后续 exp_007 阶段 C 更新：通过提高弱 BC 的学习效率并在 PPO 阶段加入可记录的 teacher regularization，最终 checkpoint 已来自 PPO 阶段并通过 proxy / PhysX 指标。

## 阶段 C 月面崎岖地形与 PPO-only 收敛记录（2026-06-02）

阶段 C 已完成第一版：新增月面 crater proxy terrain、PhysX lunar crater mesh、多 episode 四 Jetbot 高保真评估，并得到一个最终 best 来自 PPO 阶段的弱 warm-start checkpoint。

月面地形尺度依据：

- NASA Moon Craters 页面说明 simple lunar craters 通常较小，约不超过 10-15 km，形态为相对较深的 bowl/cone shape。
- NASA impact crater 资料说明较小撞击坑可呈 bowl-shaped form。
- NASA mare pit crater 资料给出月面 mare pit craters 约 100 m 直径的例子。
- NASA Surveyor/Apollo 资料提到 Surveyor crater 约 200 m 尺度。

阶段 C 采用的是 Jetbot/四车场景可承受的米级缩尺 crater field，不是把百米/公里级月坑直接放入 9 m 场景。当前默认 PhysX profile：

```text
terrain: lunar_crater
size: 9.0 m
resolution: 64
amplitude: 0.025 m
wavelength: 2.8 m
crater_count: 7
crater_min_radius: 0.35 m
crater_max_radius: 1.15 m
crater_depth_to_diameter: 0.06
crater_rim_height_to_diameter: 0.015
```

新增/调整内容：

```text
configs/experiment/exp_007_phase_c_weak_warmstart.yaml
configs/experiment/exp_007_phase_c_pure_rl.yaml
scripts/evaluate_physx_four_jetbots.py
scripts/physx_jetbot_common.py
source/lunar_rover_tasks/.../terrain_features.py
source/lunar_rover_tasks/.../gathering_env_cfg.py
```

关键训练设置：

```text
mode: weak_warmstart
bc_steps: 50
bc_learning_rate: 3e-3
total_env_steps: 2,000,000
actual PPO updates: 7
num_envs: 2048
rollout_steps: 128
checkpoint selection: required_best_phase=ppo
scripted_teacher_coef: 0.35 -> 0.10
```

说明：本轮仍不是 pure RL 从零收敛；它是弱 BC 初始化 + PPO 阶段继续优化，并且最终 checkpoint 必须来自 PPO 阶段。训练脚本已收紧：当 `required_best_phase=ppo` 时，BC/baseline 不会被保存为 best checkpoint。

最终 checkpoint：

```text
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/checkpoints/best.pt
```

训练内 best 指标：

```text
phase: ppo
update: 7
dmax_reduction_ratio: 0.1430
success_rate: 1.0000
collision_rate: 0.0000
timeout_rate: 0.0000
```

独立 proxy 复评：

```bash
.venv_isaaclab/bin/python scripts/evaluate_proxy_policy.py \
  --config configs/experiment/exp_007_phase_c_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/checkpoints/best.pt \
  --device cuda \
  --num-envs 1024 \
  --steps 220 \
  --seed 2026 \
  --output outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/metrics/final_eval_proxy.json
```

复评结果：

```text
dmax_reduction_ratio: 0.1401
success_rate: 1.0000
collision_rate: 0.0000
timeout_rate: 0.0000
mean_done_step: 83.53
```

PhysX lunar crater 三集评估：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --config configs/experiment/exp_007_phase_c_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/checkpoints/best.pt \
  --terrain lunar_crater \
  --episodes 3 \
  --steps 100 \
  --sim-steps-per-control 8 \
  --seed 2026 \
  --run-dir outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m
```

PhysX 结果：

```text
success_rate: 1.0000
collision_rate: 0.0000
mean_final_dmax: 0.7977
mean_final_dispersion: 0.1427
mean_max_tilt_deg: 14.87
mean_physics_updates_per_s: 158.19
phase_c_acceptance: passed
```

当前 shell 没有 `DISPLAY` / `WAYLAND_DISPLAY`，也没有 `xvfb-run`，因此本轮只完成 Isaac Sim headless PhysX 评估，没有生成 viewport 渲染 GIF。headless 日志显示 RTX 5090 被 Isaac Sim Vulkan 后端识别并用于 PhysX/渲染后端初始化。

阶段 C 当前产物：

```text
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/checkpoints/best.pt
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/metrics/eval_metrics.json
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/metrics/final_eval_proxy.json
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/figures/convergence_curves.png
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/figures/safety_diagnostics.png
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/videos/proxy_eval_rollout.gif
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/physx/metrics/lunar_crater_headless.json
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/physx/metrics/lunar_crater_render.json
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/physx/videos/lunar_crater_rollout.gif
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/run_manifest.json
```

长期输出管理规范见：

```text
docs/output_management.md
```

本轮回归：

```text
.venv_isaaclab/bin/python -m pytest
27 passed
```

参考链接：

```text
https://science.nasa.gov/moon/lunar-craters/
https://www.nasa.gov/solar-system/asteroid-day-and-impact-craters/
https://science.nasa.gov/photojournal/how-common-are-mare-pit-craters/
https://www.nasa.gov/missions/lunar-reconnaissance-orbiter-looks-at-apollo-12-surveyor-3-landing-sites/
```

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
- 四 Jetbot lunar crater 多 episode headless evaluation 可运行
- 可输出截图和 GIF

PhysX 层当前用于验证和展示，不参与 policy rollout 采样。

### 4. 测试覆盖

当前回归命令：

```bash
.venv_isaaclab/bin/python -m pytest
```

最近验证结果：

```text
27 passed
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

1. 当前阶段 C 收敛 checkpoint 不是 pure RL 从零训练结果。

   当前 best checkpoint 来自弱 BC 初始化后的 PPO 阶段，不是强 warmup checkpoint。训练脚本已经保证 `required_best_phase=ppo` 时不会保存 BC/baseline 为 best。若后续需要强调“纯强化学习自主收敛”，仍需继续推进 pure RL 配置。

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

### 阶段 C：PhysX 高保真闭环评估（已完成第一版）

目标：验证 proxy checkpoint 在 Isaac Sim 轮式资产和崎岖地形中的迁移表现。

完成情况：

1. 已用 canonical checkpoint `outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/checkpoints/best.pt` 跑四 Jetbot lunar crater PhysX 三集评估。
2. 已输出 high-fidelity evaluation JSON，包括 dmax、dispersion、碰撞、tilt 和 sim throughput。
3. 当前无显示后端，未生成 viewport 渲染 GIF；proxy rollout GIF 已生成。
4. 后续仍可补充 proxy / PhysX 轨迹误差对比。
5. Jetbot 在当前米级缩尺 crater field 中稳定；更大坡度/坑深再考虑 NovaCarter 或自定义 rover 资产。

建议命令：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --config configs/experiment/exp_007_phase_c_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/checkpoints/best.pt \
  --terrain lunar_crater \
  --episodes 3 \
  --steps 100 \
  --sim-steps-per-control 8 \
  --run-dir outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m
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
2. 为每个实验配置建立清晰命名：`exp_004_proxy_convergence`、`exp_005_safety_tuned`、`exp_006_ppo_selected`、`exp_007_phase_c`。
3. 保留每次关键 checkpoint 对应的配置和 final_eval。
4. 在 README 中增加推荐运行顺序。

## 推荐近期执行顺序

1. 复验当前收敛 checkpoint：

```bash
.venv_isaaclab/bin/python scripts/evaluate_proxy_policy.py \
  --config configs/experiment/exp_007_phase_c_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/checkpoints/best.pt \
  --device cuda \
  --num-envs 1024 \
  --steps 220
```

2. 跑四车 lunar crater PhysX 评估：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --config configs/experiment/exp_007_phase_c_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/checkpoints/best.pt \
  --terrain lunar_crater \
  --episodes 3 \
  --steps 100 \
  --sim-steps-per-control 8 \
  --run-dir outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m
```

3. 在有显示后端时补跑 `--render` 版本，生成 viewport 截图/GIF。

4. 继续推进 pure RL 配置，明确论文或报告中“弱 warm-start + PPO”和“pure RL”的实验口径。

## 当前推荐判断

短期内不要把 Isaac Sim 放进主训练 loop。更稳妥的路线是：

```text
proxy 高吞吐训练
-> proxy 数值验证
-> PhysX 高保真 closed-loop evaluation
-> 渲染视频展示
```

只有当 PhysX 评估中出现 proxy 无法解释的系统性失败时，再考虑把部分高保真动力学引入训练或做 domain randomization。

## 2026-06-03 输出结构整理

本次将长期产物管理统一到 run-oriented layout：

```text
outputs/runs/<experiment_id>/<run_id>/
```

关键变更：

1. 新增/更新 `docs/output_management.md`，记录 canonical 目录结构、run 命名规则、迁移命令、TensorBoard / proxy eval / PhysX eval 的推荐输出路径。
2. 扩展 `scripts/organize_outputs.py`：支持 `--all-known`、`--experiment`、`--preset exp007_phase_c`、`--dry-run`、`--mode symlink|copy`，并生成每个 run 的 `run_manifest.json` 与全局 `outputs/runs/_index.json`。
3. `scripts/train_proxy_convergence.py` 在 `experiment.output_layout: run` 时会直接写入 `config/experiment.yaml`、`metrics/summary.json`、`run_manifest.json`、`metrics/`、`figures/`、`videos/`、`checkpoints/` 和 `tensorboard/`。
4. `scripts/evaluate_proxy_policy.py` 增加 `--run-dir`，默认将独立 proxy 评估写入 `metrics/final_eval_proxy.json`。
5. 已将现有 legacy 产物以 symlink 方式整理到 `outputs/runs/`，并刷新了 `exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m` 的 PhysX 展示产物链接。

当前索引：

```text
outputs/runs/_index.json
```

推荐以后查看成果时优先使用：

```text
outputs/runs/exp_006_ppo_selected/bc_ppo_seed_23/
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/
```

后续新训练命名建议：

```text
<mode>_seed<seed>_<budget>_<terrain>_<short_tag>
```

示例：

```text
weak_warmstart_seed23_6m_lunar_crater_bc20
pure_rl_seed31_2m_lunar_crater
smoke_seed23_8k_cpu
```

## 2026-06-03 exp_008 Terrain3D 严格收敛结果

本阶段将 proxy 环境从平面运动学升级为 terrain-aware 3D 简化动力学：

1. `terrain.dynamics_enabled=true` 时，reset 和 step 会根据 `query_height(xy)` 更新 rover 的 `z`。
2. 有坡度或低 traversability 区域会降低有效线速度，并在 `info` 中记录 `terrain_features`、`terrain_speed_scale`、`height_delta`。
3. reward 新增 `terrain` 项：

```text
terrain_reward = -mean(slope_cost * roughness + terrain_cost * (1 - traversability))
```

4. `lunar_crater_proxy` 采用 9 m 缩尺月坑地形，训练前 sanity check 显示不是平地：

```text
height_range ~= 0.241 m
roughness_max ~= 0.360
traversability_min ~= 0.549
```

### 训练口径

先按 pure RL 优先执行：

- `pure_rl_seed23_8m_lunar_crater_cpu`：失败，`dmax_ratio=0.321`、`success_rate=0.032`、`timeout_rate=0.966`。
- `pure_rl_seed23_15m_lunar_crater_cpu_continued`：失败，`dmax_ratio=0.302`、`success_rate=0.0`、`timeout_rate=0.998`。

因此按计划切换到弱 warm-start fallback，仍限制为 `bc_steps <= 20`，最终 selected policy 是：

```text
weak warm-start (20 BC steps max) + PPO
```

训练中发现并修复了两个选择/评估问题：

1. checkpoint 选择不能只按更低 `dmax_ratio` 覆盖，否则会用 timeout 非零的 checkpoint 覆盖严格通过 checkpoint；已改为 strict-pass 优先。
2. `scripts/play.py` 对 actor checkpoint 做了第二次 `tanh`，导致独立评估动作幅度被压小；已修复为直接使用 `actor(...).mean`，与训练内 deterministic eval 保持一致。

### 严格复验结果

最终独立 proxy evaluation 命令统一使用：

```bash
.venv_isaaclab/bin/python scripts/evaluate_proxy_policy.py \
  --device cpu \
  --num-envs 1024 \
  --steps 220 \
  --run-dir <run_dir>
```

严格标准：

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

最终 3 个 seeds 全部通过：

| seed | final run | dmax_ratio | success | collision | timeout |
| --- | --- | ---: | ---: | ---: | ---: |
| 23 | `weak_warmstart_seed23_8m_lunar_crater_cpu` | 0.1539 | 1.0000 | 0.0000 | 0.0000 |
| 31 | `weak_warmstart_completion_seed31_4m_evalseed0_cpu` | 0.1345 | 0.9961 | 0.0049 | 0.0000 |
| 47 | `weak_warmstart_select_seed47_8m_lunar_crater_cpu` | 0.1560 | 1.0000 | 0.0000 | 0.0000 |

Suite 汇总：

```text
outputs/runs/exp_008_terrain3d/_suite/metrics/strict_acceptance.json
outputs/runs/exp_008_terrain3d/_suite/metrics/suite_summary.json
outputs/runs/exp_008_terrain3d/_suite/figures/comparison_curves.png
outputs/runs/exp_008_terrain3d/_suite/checkpoints/seed_23_best.pt
outputs/runs/exp_008_terrain3d/_suite/checkpoints/seed_31_best.pt
outputs/runs/exp_008_terrain3d/_suite/checkpoints/seed_47_best.pt
```

每个 selected run 还包含：

```text
metrics/final_eval_proxy.json
figures/convergence_curves.png
figures/safety_diagnostics.png
figures/terrain_height_map.png
videos/proxy_eval_rollout.gif
tensorboard/
```

其中 `proxy_eval_rollout.gif` 已叠加 terrain height heatmap 背景，`figures/terrain_height_map.png` 单独展示本次 lunar crater heightfield；suite 入口也包含：

```text
outputs/runs/exp_008_terrain3d/_suite/figures/terrain_height_map.png
```

### 当前结论

在 3D terrain-aware proxy 和缩尺 lunar crater 地形下，pure RL 在当前预算内未收敛；弱 warm-start + PPO 达到了严格 proxy gate。这个结果仍应表述为“弱 warm-start 初始化后的 PPO 收敛”，不能表述为纯 RL 从零严格收敛。

## 后续进度文档

从 2026-06-05 起，新的阶段结果不再继续追加到本文档。exp009 强地形训练与复验记录迁移到：

```text
docs/progress_summary_2026-06-05.md
```
