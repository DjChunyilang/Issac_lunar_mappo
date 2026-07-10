# exp056 structured bicycle quintic map25 terminal pairwise reward

## 目的

exp051 是当前新环境栈 local reset 最好候选：dmax `0.1836`、success `0.9883`、collision `0.0020` 均通过，但 timeout `0.0098` 未清零。exp053 说明全局提高 `near_distance` reward 会把队形推散；exp054/exp055 说明继续调 PPO clip 也不能清掉尾部。

exp056 回到 exp051，不改 action 输出、不新增多点采样、不改 filter/control，只新增一个 reward 侧窄触发项：当 dmax/dispersion 已接近成功区时，对 `nearest < success_thresholds.min_pairwise_distance` 的 gap 给惩罚。目标是只处理 exp051/exp055 timeout 子集中“已经靠拢但最近邻不足”的灰区。

## 配置

```text
configs/experiment/exp056_structured_bicycle_quintic_map25_terminal_pairwise_reward.yaml
```

相对 exp051 的实质变量：

- `reward.coefficients.terminal_pairwise_gap: 0.0 -> 4.0`
- `reward.coefficients.terminal_pairwise_dmax_multiplier: 1.10`
- `reward.coefficients.terminal_pairwise_dispersion_multiplier: 1.15`

保持不变：

- Actor 输出仍是单点 `[rho, beta]`；
- 不引入多点采样；
- filter、control safety、reset、terrain、trajectory、episode/eval steps 全部保持 exp051；
- `reward.coefficients.near_distance=2.4` 保持 exp051，不采用 exp053 的全局 near 增强；
- PPO 回到 exp051：`clip_epsilon=0.18`、`learning_rate=1.0e-4`、`initial_log_std=-0.95`、`entropy_schedule_timesteps=12288`。

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
  --config configs/experiment/exp056_structured_bicycle_quintic_map25_terminal_pairwise_reward.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_terminal_pairwise_reward \
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
  --config configs/experiment/exp056_structured_bicycle_quintic_map25_terminal_pairwise_reward.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_terminal_pairwise_reward \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

## 长训命令

```bash
mkdir -p outputs/runs/exp056_structured_bicycle_quintic_map25_terminal_pairwise_reward/_launcher

systemd-run --user --unit exp056-structured-bicycle-quintic-map25-terminal-pairwise-reward-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp056_structured_bicycle_quintic_map25_terminal_pairwise_reward/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp056_structured_bicycle_quintic_map25_terminal_pairwise_reward/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp056_structured_bicycle_quintic_map25_terminal_pairwise_reward.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_pairwise_reward \
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
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_terminal_pairwise_reward` | smoke only | 工程通过；`8 env / 8 timesteps`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0845`、action std `0.0582` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_terminal_pairwise_reward` | smoke only | 工程通过；`256 env / 128 timesteps / rollout 64`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0703`、action std `0.0865` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_pairwise_reward` | `ppo_timestep_012288.pt` | dmax `0.1864`、success `0.9873`、collision `0.0010`、timeout `0.0117` | 未通过 |

## 失败分析

`strict_acceptance.json` 中 dmax、success 和 collision 通过，但 timeout 失败：

```text
dmax_reduction_ratio: 0.18636061251163483
success_rate: 0.9873046875
collision_rate: 0.0009765625
timeout_rate: 0.01171875
```

相对 exp051，exp056 把 collision 从 `0.0020` 降到 `0.0010`，filter collision override 从 `0.3538` 降到 `0.3005`，timeout 子集最近邻均值也从 `0.3527` 提到 `0.3629`。这些说明方向有轻微信号。

但它没有清掉 timeout，反而从 exp051 的 `0.0098` 升到 `0.0117`。timeout 子集仍有 `12 / 1024` 个 episode，`final_min_pairwise_ok_rate=0.0833`，说明最近邻 gate 仍没有稳定站住。当前 `terminal_pairwise_gap=4.0` 且 dmax/dispersion multiplier 为 `1.10/1.15`，可能触发仍偏早、强度偏大，扰动了少量原本能成功 hold 的 episode。

## 判读重点

- 若 timeout 低于 exp051 的 `0.0098` 且 success/collision 保持达标，说明窄触发 pairwise reward 有助于处理最后安全间距灰区。
- 若 success 下降或 timeout 明显升高，说明该 reward 仍然把队形推散，应调低或关闭。
- 若 filter/control 介入比例明显升高，需要判为不理想。
- 不应把训练 reward 或 GIF 写成 strict 结论。

## 产物路径

已生成：

```text
outputs/runs/exp056_structured_bicycle_quintic_map25_terminal_pairwise_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_pairwise_reward/metrics/final_eval_proxy.json
outputs/runs/exp056_structured_bicycle_quintic_map25_terminal_pairwise_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_pairwise_reward/metrics/strict_acceptance.json
outputs/runs/exp056_structured_bicycle_quintic_map25_terminal_pairwise_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_pairwise_reward/figures/training_curves.png
outputs/runs/exp056_structured_bicycle_quintic_map25_terminal_pairwise_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_pairwise_reward/figures/candidate_eval_curves.png
outputs/runs/exp056_structured_bicycle_quintic_map25_terminal_pairwise_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_pairwise_reward/figures/exp051_exp053_exp055_exp056_terminal_pairwise_comparison.png
```

## 结论

exp056 不是当前主结果。它改善了 collision 和 filter override，但 timeout 未优于 exp051；当前最好仍是 exp051。

## 下一步

若继续 terminal pairwise reward 方向，下一步应更克制：把触发收窄到严格成功几何内，并把 `terminal_pairwise_gap` 降低，避免过早推开队形。
