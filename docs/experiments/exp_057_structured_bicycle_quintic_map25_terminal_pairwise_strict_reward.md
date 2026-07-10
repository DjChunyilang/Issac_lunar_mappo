# exp057 structured bicycle quintic map25 terminal pairwise strict reward

## 目的

exp056 的 terminal pairwise reward 有轻微信号：collision 和 filter override 降低，timeout 子集最近邻略升；但 timeout `0.0117` 仍差于 exp051 的 `0.0098`。这说明 reward 方向不能直接增强，应该更克制。

exp057 回到 exp051/exp056 主体，只把 terminal pairwise reward 改成更窄、更弱：

- `terminal_pairwise_gap: 4.0 -> 2.0`
- `terminal_pairwise_dmax_multiplier: 1.10 -> 1.00`
- `terminal_pairwise_dispersion_multiplier: 1.15 -> 1.00`

目标是只在严格成功几何附近惩罚最近邻不足，减少对正常集合收缩过程的扰动。

## 配置

```text
configs/experiment/exp057_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward.yaml
```

相对 exp051 的实质变量：

- `reward.coefficients.terminal_pairwise_gap: 0.0 -> 2.0`
- `reward.coefficients.terminal_pairwise_dmax_multiplier: 1.00`
- `reward.coefficients.terminal_pairwise_dispersion_multiplier: 1.00`

保持不变：

- Actor 输出仍是单点 `[rho, beta]`；
- 不引入多点采样；
- filter、control safety、reset、terrain、trajectory、episode/eval steps 全部保持 exp051；
- `reward.coefficients.near_distance=2.4` 保持 exp051；
- PPO 保持 exp051：`clip_epsilon=0.18`、`learning_rate=1.0e-4`、`initial_log_std=-0.95`、`entropy_schedule_timesteps=12288`。

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
  tests/test_reward.py \
  tests/test_config_wiring.py \
  tests/test_skrl_mappo_semantics.py
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp057_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward \
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
  --config configs/experiment/exp057_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

## 长训命令

```bash
mkdir -p outputs/runs/exp057_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward/_launcher

systemd-run --user --unit exp057-structured-bicycle-quintic-map25-terminal-pairwise-strict-reward-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp057_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp057_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp057_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward \
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
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward` | smoke only | 工程通过；`8 env / 8 timesteps`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0845`、action std `0.0582` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward` | smoke only | 工程通过；`256 env / 128 timesteps / rollout 64`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0715`、action std `0.0862` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward` | `ppo_timestep_011264.pt` | dmax `0.1850`、success `0.9697`、collision `0.0059`、timeout `0.0254` | 未通过 |

## 失败分析

`strict_acceptance.json` 中 dmax、success 和 collision 通过，但 timeout 失败：

```text
dmax_reduction_ratio: 0.18504005670547485
success_rate: 0.9697265625
collision_rate: 0.005859375
timeout_rate: 0.025390625
```

相对 exp056，exp057 的触发更严格、系数更弱，但 timeout 从 `0.0117` 升到 `0.0254`，success 从 `0.9873` 降到 `0.9697`。timeout 子集仍满足集合几何：`final_dmax_mean≈0.956`、`final_dispersion_mean≈0.164`，但最近邻均值 `0.3519` 仍低于 `0.42`。

这说明 terminal pairwise reward 方向没有解决尾部 timeout；即使只在严格成功几何附近触发，也会扰动末段 hold。下一轮不应继续沿 pairwise reward 增强/减弱扫描。

## 判读重点

- 若 timeout 低于 exp051 的 `0.0098` 且 success/collision 保持达标，说明更窄触发比 exp056 更有效。
- 若结果接近 exp051 但无 timeout 改善，说明 terminal pairwise reward 方向收益不足。
- 若 success 下降或 timeout 升高，说明即使严格触发也会扰动末端 hold，应停止该方向。
- 不应把训练 reward 或 GIF 写成 strict 结论。

## 产物路径

已生成：

```text
outputs/runs/exp057_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward/metrics/final_eval_proxy.json
outputs/runs/exp057_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward/metrics/strict_acceptance.json
outputs/runs/exp057_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward/figures/training_curves.png
outputs/runs/exp057_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward/figures/candidate_eval_curves.png
outputs/runs/exp057_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_pairwise_strict_reward/figures/exp051_exp056_exp057_terminal_pairwise_comparison.png
```

## 结论

exp057 不是当前主结果。terminal pairwise reward 方向在 exp056/exp057 都没有改善 timeout，当前最好仍是 exp051。

## 下一步

回到 exp051，不继续 terminal pairwise reward 扫描。下一步应转向 checkpoint selection gate 或 success-hold 评估一致性诊断，而不是继续加安全间距 reward。
