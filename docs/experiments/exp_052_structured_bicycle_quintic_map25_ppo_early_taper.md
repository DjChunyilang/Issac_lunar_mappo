# exp052 structured bicycle quintic map25 PPO early taper

## 目的

exp051 在不改变 action 输出、不新增多点采样、不增强 filter/control 的情况下，把 timeout 从 exp048 的 `0.0137` 降到 `0.0098`，但仍未满足 `timeout_rate == 0`。

exp052 以 exp051 为基线，只把 entropy schedule 从 `12288` timesteps 提前到 `8192` timesteps。目标是判断“更早收窄后期探索”能否减少 terminal drift，并清掉 exp051 剩余 `10 / 1024` 个 timeout。

## 配置

```text
configs/experiment/exp052_structured_bicycle_quintic_map25_ppo_early_taper.yaml
```

相对 exp051 的唯一实验变量：

- `algorithm.entropy_schedule_timesteps: 12288 -> 8192`

保持不变：

- Actor 输出仍是单点 `[rho, beta]`；
- 不引入多点采样；
- reward、filter、control safety、reset、terrain、trajectory、episode/eval steps 全部保持 exp051；
- PPO 的 `learning_rate=1.0e-4`、`clip_epsilon=0.18`、`initial_log_std=-0.95`、`entropy_loss_scale=0.0009`、`entropy_loss_scale_end=0.00010` 保持 exp051。

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
  --config configs/experiment/exp052_structured_bicycle_quintic_map25_ppo_early_taper.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_ppo_early_taper \
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
  --config configs/experiment/exp052_structured_bicycle_quintic_map25_ppo_early_taper.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_ppo_early_taper \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

## 长训命令

```bash
mkdir -p outputs/runs/exp052_structured_bicycle_quintic_map25_ppo_early_taper/_launcher

systemd-run --user --unit exp052-structured-bicycle-quintic-map25-ppo-early-taper-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp052_structured_bicycle_quintic_map25_ppo_early_taper/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp052_structured_bicycle_quintic_map25_ppo_early_taper/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp052_structured_bicycle_quintic_map25_ppo_early_taper.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_early_taper \
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
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_ppo_early_taper` | smoke only | 工程通过；`8 env / 8 timesteps`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0845`、action std `0.0582` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_ppo_early_taper` | smoke only | 工程通过；`256 env / 128 timesteps / rollout 64`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0715`、action std `0.0875` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_early_taper` | `ppo_timestep_008192.pt` / `best.pt` | dmax ratio `0.1863`、success `0.8955`、collision `0.0059`、timeout `0.0986` | 未通过；success/timeout 失败 |

## 结果分析

exp052 明显弱于 exp051。把 entropy taper 从 `12288` 提前到 `8192` 后，best checkpoint 也提前到 `008192`，但 final eval 只有 success `0.8955`，刚好低于 strict 的 `0.9`，timeout 升到 `0.0986`。

| 实验 | entropy taper | checkpoint | dmax ratio | success | collision | timeout | filter applied | filter override | control safety |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp051 | `12288` | `013312` | `0.1836` | `0.9883` | `0.0020` | `0.0098` | `0.5072` | `0.3538` | `0.0742` |
| exp052 | `8192` | `008192` | `0.1863` | `0.8955` | `0.0059` | `0.0986` | `0.4190` | `0.3162` | `0.0755` |

timeout 子集有 `101 / 1024` 个 episode，`final_dmax_mean=1.2303`、`final_dispersion_mean=0.2993`、`final_nearest_neighbor_distance_mean=0.4193`、`max_success_hold_count_mean=0.9307`。这说明失败并不是少量接近完成的 hold 尾部，而是 early taper 后 terminal recovery 明显不足。

本轮保持了原动作接口，没有引入多点采样或额外 filter/control 规划能力；因此负结果可以较干净地归因到 entropy schedule 过早收窄。

## 产物路径

训练后应生成：

```text
outputs/runs/exp052_structured_bicycle_quintic_map25_ppo_early_taper/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_early_taper/metrics/final_eval_proxy.json
outputs/runs/exp052_structured_bicycle_quintic_map25_ppo_early_taper/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_early_taper/metrics/strict_acceptance.json
outputs/runs/exp052_structured_bicycle_quintic_map25_ppo_early_taper/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_early_taper/figures/training_curves.png
outputs/runs/exp052_structured_bicycle_quintic_map25_ppo_early_taper/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_early_taper/figures/candidate_eval_curves.png
outputs/runs/exp052_structured_bicycle_quintic_map25_ppo_early_taper/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_early_taper/figures/exp048_exp051_exp052_candidate_comparison.png
```

## 结论

exp052 不是有效方向。更早 entropy taper 会过早收窄探索，降低 terminal recovery，使 success 和 timeout 都明显差于 exp051。

## 下一步

回到 exp051 作为当前最好基线。后续仍应维持原动作接口，避免把策略输出改为多点采样；若继续做 PPO 侧实验，不应再把 entropy schedule 提前到 `8192`。
