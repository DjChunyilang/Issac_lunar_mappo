# exp058 structured bicycle quintic map25 gamma995

## 目的

exp051 是当前新环境栈 local reset 最好候选，dmax/success/collision 已过 strict，但 timeout `0.0098` 未清零。后续 exp052/exp054/exp055 说明过早收窄探索或调整 PPO clip 都不能解决尾部 timeout；exp056/exp057 说明 terminal pairwise reward 也不是有效方向。

exp058 回到 exp051，只把 PPO 折扣因子从 `gamma=0.99` 提高到 `0.995`。目标是让 terminal success / timeout / hold 信号在 320-step episode 中传得更远，判断更长 horizon 的 value backup 是否能改善跨 eval seed 稳定存在的尾部 timeout。

## 配置

```text
configs/experiment/exp058_structured_bicycle_quintic_map25_gamma995.yaml
```

相对 exp051 的唯一实质变量：

- `algorithm.gamma: 0.99 -> 0.995`

保持不变：

- Actor 输出仍是单点 `[rho, beta]`；
- 不引入多点采样；
- reward、filter、control safety、reset、terrain、trajectory、episode/eval steps 全部保持 exp051；
- `gae_lambda=0.95`、`learning_rate=1.0e-4`、`clip_epsilon=0.18`、`initial_log_std=-0.95`、`entropy_schedule_timesteps=12288` 保持 exp051。

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
  --config configs/experiment/exp058_structured_bicycle_quintic_map25_gamma995.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_gamma995 \
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
  --config configs/experiment/exp058_structured_bicycle_quintic_map25_gamma995.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_gamma995 \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

## 长训命令

```bash
mkdir -p outputs/runs/exp058_structured_bicycle_quintic_map25_gamma995/_launcher

systemd-run --user --unit exp058-structured-bicycle-quintic-map25-gamma995-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp058_structured_bicycle_quintic_map25_gamma995/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp058_structured_bicycle_quintic_map25_gamma995/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp058_structured_bicycle_quintic_map25_gamma995.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_gamma995 \
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
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_gamma995` | smoke only | 工程通过；`8 env / 8 timesteps`，`discount_factor=0.995`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0805`、action std `0.0449` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_gamma995` | smoke only | 工程通过；`256 env / 128 timesteps / rollout 64`，`discount_factor=0.995`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0721`、action std `0.0867` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_gamma995` | `ppo_timestep_012288.pt` / `best.pt` | dmax ratio `0.1991`、success `0.7451`、collision `0.0020`、timeout `0.2529` | 未通过；success/timeout 失败 |

## 失败分析

`strict_acceptance.json` 中 dmax 和 collision 通过，但 success 与 timeout 明显失败：

```text
dmax_reduction_ratio: 0.1990668922662735
success_rate: 0.7451171875
collision_rate: 0.001953125
timeout_rate: 0.2529296875
```

相对 exp051，`gamma=0.995` 没有改善尾部 timeout，反而把 success 从 `0.9883` 降到 `0.7451`，timeout 从 `0.0098` 升到 `0.2529`。candidate eval 最高 success 只到 `0.7451`，best 落在 `012288` 附近；后续 checkpoint 没有恢复。

| 实验 | gamma | checkpoint | dmax ratio | success | collision | timeout | filter applied | filter override | control safety |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp051 | `0.99` | `013312` | `0.1836` | `0.9883` | `0.0020` | `0.0098` | `0.5072` | `0.3538` | `0.0742` |
| exp058 | `0.995` | `012288` | `0.1991` | `0.7451` | `0.0020` | `0.2529` | `0.3652` | `0.3044` | `0.0899` |

timeout 子集有 `259 / 1024` 个 episode，`final_dmax_mean=1.5240`、`final_dispersion_mean=0.4884`、`final_nearest_neighbor_distance_mean=0.5404`、`max_success_hold_count_mean=0.0502`。这说明 exp058 的失败不是 exp051 那种少量成功区附近的 timeout 尾部，而是大量 episode 没有进入稳定成功区。

本轮没有改变 Actor 输出语义，也没有引入多点采样或额外 filter/control 规划能力；负结果可以较干净地归因到更长折扣因子导致当前训练更慢或 value 估计更难。

## 判读重点

- 若 timeout 低于 exp051 的 `0.0098` 且 success/collision 保持达标，说明 long-horizon discount 有助于末端 hold/timeout 信号传播。
- 若 success 下降或 collision/filter override 明显上升，说明更高 `gamma` 带来的 value 方差或策略保守性不适合当前主线。
- 不应把训练 reward 或 GIF 写成 strict 结论。

## 产物路径

训练后应生成：

```text
outputs/runs/exp058_structured_bicycle_quintic_map25_gamma995/pure_rl_seed23_40m_structured_bicycle_quintic_map25_gamma995/metrics/final_eval_proxy.json
outputs/runs/exp058_structured_bicycle_quintic_map25_gamma995/pure_rl_seed23_40m_structured_bicycle_quintic_map25_gamma995/metrics/strict_acceptance.json
outputs/runs/exp058_structured_bicycle_quintic_map25_gamma995/pure_rl_seed23_40m_structured_bicycle_quintic_map25_gamma995/figures/training_curves.png
outputs/runs/exp058_structured_bicycle_quintic_map25_gamma995/pure_rl_seed23_40m_structured_bicycle_quintic_map25_gamma995/figures/candidate_eval_curves.png
outputs/runs/exp058_structured_bicycle_quintic_map25_gamma995/pure_rl_seed23_40m_structured_bicycle_quintic_map25_gamma995/figures/exp051_exp058_gamma_comparison.png
```

## 结论

exp058 不是有效方向。单独把 `gamma` 提高到 `0.995` 会明显拖慢 terminal convergence，导致 success/timeout 大幅差于 exp051；后续不应继续沿更大 discount factor 搜索。

## 下一步

回到 exp051 的 `gamma=0.99`。下一轮仍应维持原动作接口和原 filter/control 兜底语义，转向更局部的 RL 训练信号，例如不改变 strict gate 的 critic/observation 尾部失败辨识，或更克制地调整 value/GAE 而不是继续拉长 discount horizon。
