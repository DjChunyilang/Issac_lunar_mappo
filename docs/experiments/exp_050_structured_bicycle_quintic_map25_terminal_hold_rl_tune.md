# exp050 structured bicycle quintic map25 terminal hold RL tune

## 目的

exp048 已经通过 dmax、success 和 collision gate，只剩 timeout `0.0137`。exp049 证明全局增强 terminal spacing / filter / control safety 会把部分 episode 推出成功保持区，导致 success 和 timeout 明显退化。

exp050 回到 exp048 主体，不改变 action 输出，不新增多点采样，不增强低层控制器规划能力。目标是优先通过 RL 配置做末端稳定微调：

- 保持 `branched_v1 / structured_v1`、`bicycle` proxy、`quintic` 轨迹和 `25 m x 25 m` 地图；
- filter/control 介入比 exp048 更克制或基本持平；
- 主要调整 reward 的 terminal hold、dispersion/dmax 收缩、timeout shaping，以及 PPO 的学习率、clip 和探索噪声；
- 检查能否减少 exp048 的少量 timeout，同时保持 success/collision 已达标状态。

## 配置

```text
configs/experiment/exp050_structured_bicycle_quintic_map25_terminal_hold_rl_tune.yaml
```

关键设置：

- Actor/Critic：`branched_v1 / structured_v1`，接口保持 `86 / 54`。
- 动力学/轨迹：`bicycle` proxy + `quintic` trajectory。
- reset 分布：保持 exp048 的目标 `2.4-3.4 m`、课程起点 `1.6-2.4 m`。
- filter：不扩大候选集合；`apply_probability_end` 从 exp048 的 `0.18` 降到 `0.16`，`score_scale_end` 从 `0.30` 降到 `0.26`。
- terminal spacing：`hold_zone_pairwise_distance` 从 exp048 的 `0.46` 收回到 `0.44`，只略高于 success gate `0.42`，避免 exp049 的全局推开。
- control safety：`projection_strength` 从 `0.45` 降到 `0.42`，`projection_min_linear_scale` 从 `0.65` 提到 `0.68`，`success_zone_linear_scale` 从 `0.95` 提到 `0.97`，减少末段低层阻尼。
- reward：`dmax_progress=9.7`、`dispersion_progress=6.4`、`success_hold_step=14`、`success_bonus=135`、`timeout_penalty=90`，但 `near_distance` 保持 exp048 的 `2.4`，不采用 exp049 的全局安全加硬。
- PPO：`learning_rate=1.0e-4`、`clip_epsilon=0.18`、`initial_log_std=-0.95`、`entropy_loss_scale_end=1.0e-4`，目标是降低末段策略抖动。
- 预算：`20480` timesteps，约 `41,943,040` env steps，checkpoint interval `1024`。

## 严格标准

仍使用标准 proxy strict gate：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

## 验证命令

专项测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaaclab/bin/python -m pytest -q \
  tests/test_config_wiring.py \
  tests/test_skrl_mappo_semantics.py
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp050_structured_bicycle_quintic_map25_terminal_hold_rl_tune.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_terminal_hold_rl_tune \
  --rollout-steps 4 \
  --checkpoint-interval 4 \
  --eval-num-envs 8 \
  --eval-steps 16 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

CUDA smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp050_structured_bicycle_quintic_map25_terminal_hold_rl_tune.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_terminal_hold_rl_tune \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

## 长训命令

```bash
mkdir -p outputs/runs/exp050_structured_bicycle_quintic_map25_terminal_hold_rl_tune/_launcher

systemd-run --user --unit exp050-structured-bicycle-quintic-map25-terminal-hold-rl-tune-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp050_structured_bicycle_quintic_map25_terminal_hold_rl_tune/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp050_structured_bicycle_quintic_map25_terminal_hold_rl_tune/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp050_structured_bicycle_quintic_map25_terminal_hold_rl_tune.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_hold_rl_tune \
    --rollout-steps 64 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 320 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate success_progress_long
```

## 结果表

| seed | run_id | checkpoint | final_eval | strict |
| --- | --- | --- | --- | --- |
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_terminal_hold_rl_tune` | smoke only | 工程通过；`8 env / 8 timesteps`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0858`、action std `0.0621` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_terminal_hold_rl_tune` | smoke only | 工程通过；`256 env / 128 timesteps / rollout 64`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0695`、action std `0.0859` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_hold_rl_tune` | `ppo_timestep_009216.pt` / `best.pt` | dmax ratio `0.1847`、success `0.9590`、collision `0.0059`、timeout `0.0352` | 未通过 |

## 失败分析

exp050 没有达成目标。它仍通过 dmax、success 和 collision gate，但 timeout 从 exp048 的 `0.0137` 升到 `0.0352`。

关键诊断：

- dmax ratio `0.1847`、success `0.9590`、collision `0.0059` 均达标，唯一失败项是 timeout。
- timeout episode 的 `final_dmax_mean≈0.915`、`final_dispersion_mean≈0.166` 已满足几何收缩，但 `final_nearest_neighbor_distance_mean≈0.368`，低于 `min_pairwise_distance=0.42`。
- `max_success_hold_count_mean=7.76/8`，整体接近 hold 完成；但 timeout 子集的 `max_success_hold_count_mean≈2.19/8`，说明失败样本并没有稳定进入 safe hold。
- 相比 exp048，exp050 的 overall 最近邻更宽，但 timeout 子集最近邻更低；提高 `success_hold_step`、`timeout_penalty`、降低探索噪声没有解决灰区，反而让更多 episode 卡在最近邻安全不足。
- filter/control 介入没有新增机制，但 final eval 中 `filter_applied_fraction≈0.421`、`filter_collision_override_fraction≈0.280`、`control_safety_applied_fraction≈0.097`，仍然说明当前路线依赖已有 filter/controller。下一轮不应继续提高 hold reward 或全局 spacing。

## 产物路径

训练后应生成：

```text
outputs/runs/exp050_structured_bicycle_quintic_map25_terminal_hold_rl_tune/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_hold_rl_tune/metrics/final_eval_proxy.json
outputs/runs/exp050_structured_bicycle_quintic_map25_terminal_hold_rl_tune/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_hold_rl_tune/metrics/strict_acceptance.json
outputs/runs/exp050_structured_bicycle_quintic_map25_terminal_hold_rl_tune/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_hold_rl_tune/figures/training_curves.png
outputs/runs/exp050_structured_bicycle_quintic_map25_terminal_hold_rl_tune/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_hold_rl_tune/figures/candidate_eval_curves.png
outputs/runs/exp050_structured_bicycle_quintic_map25_terminal_hold_rl_tune/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_hold_rl_tune/figures/exp048_exp049_exp050_candidate_comparison.png
outputs/runs/exp050_structured_bicycle_quintic_map25_terminal_hold_rl_tune/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_hold_rl_tune/videos/proxy_eval_rollout.gif
```

## 结论

exp050 不能作为当前主结果。当前最佳仍是 exp048。

它的价值是排除了一个方向：在不新增 filter/control 的前提下，单纯提高 terminal hold / timeout shaping 并降低 PPO 探索噪声，没有消除剩余 timeout，反而把 timeout 从 exp048 的 `0.0137` 拉高到 `0.0352`。

## 下一步

下一步不继续加 hold reward 或全局 spacing。建议回到 exp048 主体，只隔离 PPO 稳定性调整，或进一步做减少 filter override 依赖的诊断。
