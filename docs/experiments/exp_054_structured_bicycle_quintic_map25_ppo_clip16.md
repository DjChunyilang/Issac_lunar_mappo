# exp054 structured bicycle quintic map25 PPO clip16

## 目的

exp051 是当前新环境栈 local reset 最好候选，但仍有 timeout `0.0098`。exp052 说明 entropy taper 不能提前到 `8192`，exp053 说明全局 near reward 小幅增强也会把队形推散。

exp054 回到 exp051，只把 PPO clip 从 `0.18` 收窄到 `0.16`。目标是减少 policy update 的末端扰动，观察是否能在不增加 filter/control 介入、不改 reward 的前提下降低剩余 timeout。

## 配置

```text
configs/experiment/exp054_structured_bicycle_quintic_map25_ppo_clip16.yaml
```

相对 exp051 的唯一实质变量：

- `algorithm.clip_epsilon: 0.18 -> 0.16`

保持不变：

- Actor 输出仍是单点 `[rho, beta]`；
- 不引入多点采样；
- reward、filter、control safety、reset、terrain、trajectory、episode/eval steps 全部保持 exp051；
- `learning_rate=1.0e-4`、`initial_log_std=-0.95`、`entropy_schedule_timesteps=12288` 保持 exp051。

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
  --config configs/experiment/exp054_structured_bicycle_quintic_map25_ppo_clip16.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_ppo_clip16 \
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
  --config configs/experiment/exp054_structured_bicycle_quintic_map25_ppo_clip16.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_ppo_clip16 \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

## 长训命令

```bash
mkdir -p outputs/runs/exp054_structured_bicycle_quintic_map25_ppo_clip16/_launcher

systemd-run --user --unit exp054-structured-bicycle-quintic-map25-ppo-clip16-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp054_structured_bicycle_quintic_map25_ppo_clip16/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp054_structured_bicycle_quintic_map25_ppo_clip16/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp054_structured_bicycle_quintic_map25_ppo_clip16.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_clip16 \
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
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_ppo_clip16` | smoke only | 工程通过；`8 env / 8 timesteps`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0791`、action std `0.0600` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_ppo_clip16` | smoke only | 工程通过；`256 env / 128 timesteps / rollout 64`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0699`、action std `0.0860` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_clip16` | `ppo_timestep_017408.pt` | dmax `0.1972`、success `0.7168`、collision `0.0029`、timeout `0.2803` | 未通过 |

## 失败分析

`strict_acceptance.json` 中 dmax 和 collision 两项通过，但 success 和 timeout 失败：

```text
dmax_reduction_ratio: 0.1972190886735916
success_rate: 0.716796875
collision_rate: 0.0029296875
timeout_rate: 0.2802734375
```

相对 exp051，clip 从 `0.18` 收窄到 `0.16` 后没有改善尾部 timeout，反而显著降低 success。final eval 中 filter applied `0.4172`、filter collision override `0.2990`、control safety `0.1115`，不是 filter/control 介入异常膨胀导致的结果；主要问题更像是 PPO update 过保守，策略进入和保持 success basin 的能力不足。

## 判读重点

- 若 timeout 低于 exp051 的 `0.0098` 且 success/collision 保持达标，说明更窄 PPO clip 可能有助于末端稳定。
- 若 success 下降或 dmax 失败，说明 clip `0.16` 更新过保守。
- 若 filter/control 介入比例明显升高，需要判为不理想。
- 不应把训练 reward 或 GIF 写成 strict 结论。

## 产物路径

已生成：

```text
outputs/runs/exp054_structured_bicycle_quintic_map25_ppo_clip16/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_clip16/metrics/final_eval_proxy.json
outputs/runs/exp054_structured_bicycle_quintic_map25_ppo_clip16/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_clip16/metrics/strict_acceptance.json
outputs/runs/exp054_structured_bicycle_quintic_map25_ppo_clip16/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_clip16/figures/training_curves.png
outputs/runs/exp054_structured_bicycle_quintic_map25_ppo_clip16/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_clip16/figures/candidate_eval_curves.png
outputs/runs/exp054_structured_bicycle_quintic_map25_ppo_clip16/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_clip16/figures/exp051_exp052_exp053_exp054_candidate_comparison.png
```

## 结论

exp054 不是当前主结果。它保住了 dmax/collision，但 success/timeout 大幅差于 exp051，说明 `clip_epsilon=0.16` 过窄，不能作为继续收敛的主线方向。

## 下一步

回到 exp051 的 PPO 更新幅度，不继续收窄 clip；下一步仍应保持单点 `[rho, beta]` 输出和原 filter/control 兜底，优先搜索 RL 配置或成功区保持信号的小步调整。
