# exp060 structured bicycle quintic map25 value loss 0.75

## 目的

exp051 是当前新环境栈 local reset 最好候选，dmax/success/collision 已过 strict，但 timeout `0.0098` 未清零。exp058/exp059 说明继续调整价值估计 horizon 不理想：`gamma=0.995` 和 `gae_lambda=0.90` 都明显降低 success 并抬高 timeout。

exp060 回到 exp051，只把 PPO 的 `value_loss_coef` 从 `0.50` 提高到 `0.75`。目标是让 centralized critic 对末端 hold / timeout 失败更敏感，同时不改变 action 输出、reward、filter、control safety 或环境难度。这个实验检查：更强 critic 学习信号是否能清掉 exp051 中少量已经接近成功区但 hold 不足的 timeout。

## 配置

```text
configs/experiment/exp060_structured_bicycle_quintic_map25_value075.yaml
```

相对 exp051 的唯一实质变量：

- `algorithm.value_loss_coef: 0.50 -> 0.75`

保持不变：

- Actor 输出仍是单点 `[rho, beta]`；
- 不引入多点采样；
- reward、filter、control safety、reset、terrain、trajectory、episode/eval steps 全部保持 exp051；
- `gamma=0.99`、`gae_lambda=0.95`、`learning_rate=1.0e-4`、`clip_epsilon=0.18`、`initial_log_std=-0.95`、`entropy_schedule_timesteps=12288` 保持 exp051。

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
  --config configs/experiment/exp060_structured_bicycle_quintic_map25_value075.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_value075 \
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
  --config configs/experiment/exp060_structured_bicycle_quintic_map25_value075.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_value075 \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

## 长训命令

```bash
mkdir -p outputs/runs/exp060_structured_bicycle_quintic_map25_value075/_launcher

systemd-run --user --unit exp060-structured-bicycle-quintic-map25-value075-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp060_structured_bicycle_quintic_map25_value075/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp060_structured_bicycle_quintic_map25_value075/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp060_structured_bicycle_quintic_map25_value075.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_value075 \
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
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_value075` | smoke only | 工程通过；`8 env / 8 timesteps`，`value_loss_scale=0.75`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0849`、action std `0.0545` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_value075` | smoke only | 工程通过；`256 env / 128 timesteps / rollout 64`，`value_loss_scale=0.75`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0718`、action std `0.0866` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_value075` | `ppo_timestep_012288.pt` / `best.pt` | dmax `0.1837`、success `0.9736`、collision `0.0`、timeout `0.0264`；filter applied `0.5130`、filter collision override `0.3389`、control safety `0.0878` | 未通过；timeout 失败 |

## 判读重点

- 若 timeout 低于 exp051 的 `0.0098` 且 success/collision 保持达标，说明更强 critic loss 可能改善尾部 hold。
- 若 success 下降或 filter/control 介入明显升高，说明 critic 权重过强会压制 policy 学习或放大不稳定更新。
- 若结果接近 exp051 但 timeout 仍不清零，应转向 critic/observation 的尾部失败辨识或更细的末端 hold 信号，而不是继续加大 `value_loss_coef`。
- 不应把训练 reward 或 GIF 写成 strict 结论。

## 产物路径

已生成：

```text
outputs/runs/exp060_structured_bicycle_quintic_map25_value075/pure_rl_seed23_40m_structured_bicycle_quintic_map25_value075/metrics/final_eval_proxy.json
outputs/runs/exp060_structured_bicycle_quintic_map25_value075/pure_rl_seed23_40m_structured_bicycle_quintic_map25_value075/metrics/strict_acceptance.json
outputs/runs/exp060_structured_bicycle_quintic_map25_value075/pure_rl_seed23_40m_structured_bicycle_quintic_map25_value075/figures/training_curves.png
outputs/runs/exp060_structured_bicycle_quintic_map25_value075/pure_rl_seed23_40m_structured_bicycle_quintic_map25_value075/figures/candidate_eval_curves.png
outputs/runs/exp060_structured_bicycle_quintic_map25_value075/pure_rl_seed23_40m_structured_bicycle_quintic_map25_value075/figures/exp051_exp058_exp059_exp060_value_learning_comparison.png
```

## 结论

exp060 未改善 exp051 的尾部 timeout。final eval 通过 dmax、success、collision，但 timeout `0.0264` 高于 exp051 的 `0.0098`，strict 仍失败。`value_loss_coef=0.75` 能保留中段可用候选，但没有把末端 hold 变得更稳，且训练末尾在线 success 只有约 `0.0313`。

候选曲线显示最好区间仍在 `009216-012288` 附近：`009216` candidate eval timeout `0.0186`，`012288` candidate eval timeout `0.0186`，final 独立复评 timeout `0.0264`。这说明更强 value loss 没有清掉 exp051 稳定存在的 timeout 尾部，反而比 exp051 的 best `013312` 更差。

filter/control 没有新增能力，但 final eval filter applied `0.5130`、collision override `0.3389`、control safety `0.0878`，整体仍依赖原有兜底。该结果不支持继续加大 `value_loss_coef`，也不支持转向更强 filter/control。

## 下一步

回到 exp051 作为当前最好候选。下一轮不应继续调 `gamma/gae_lambda/value_loss_coef` 这条价值估计权重路线；更值得做的是针对 exp051 timeout 子集增加诊断，或者在不改变 filter/control 权力的前提下，改进 actor/critic 对末端 hold 失败的可观测性和可学习信号。
