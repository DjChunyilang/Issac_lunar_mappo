# exp055 structured bicycle quintic map25 PPO clip20

## 目的

exp051 是当前新环境栈 local reset 最好候选，但 timeout `0.0098` 仍未清零。exp054 把 PPO clip 收窄到 `0.16` 后 success/timeout 明显退化，说明不能继续压缩 policy update。

exp055 回到 exp051，只把 PPO `clip_epsilon` 从 `0.18` 放宽到 `0.20`。目标是验证在保留 exp051 学习率、探索退火和 reward/filter/control 设置的前提下，稍大的 policy update 是否能改善剩余 timeout 尾部。

## 配置

```text
configs/experiment/exp055_structured_bicycle_quintic_map25_ppo_clip20.yaml
```

相对 exp051 的唯一实质变量：

- `algorithm.clip_epsilon: 0.18 -> 0.20`

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
  --config configs/experiment/exp055_structured_bicycle_quintic_map25_ppo_clip20.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_ppo_clip20 \
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
  --config configs/experiment/exp055_structured_bicycle_quintic_map25_ppo_clip20.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_ppo_clip20 \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

## 长训命令

```bash
mkdir -p outputs/runs/exp055_structured_bicycle_quintic_map25_ppo_clip20/_launcher

systemd-run --user --unit exp055-structured-bicycle-quintic-map25-ppo-clip20-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp055_structured_bicycle_quintic_map25_ppo_clip20/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp055_structured_bicycle_quintic_map25_ppo_clip20/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp055_structured_bicycle_quintic_map25_ppo_clip20.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_clip20 \
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
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_ppo_clip20` | smoke only | 工程通过；`8 env / 8 timesteps`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0874`、action std `0.0557` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_ppo_clip20` | smoke only | 工程通过；`256 env / 128 timesteps / rollout 64`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0697`、action std `0.0875` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_clip20` | `ppo_timestep_017408.pt` | dmax `0.1850`、success `0.9824`、collision `0.0029`、timeout `0.0146` | 未通过 |

## 失败分析

`strict_acceptance.json` 中 dmax、success 和 collision 通过，但 timeout 失败：

```text
dmax_reduction_ratio: 0.18498718738555908
success_rate: 0.982421875
collision_rate: 0.0029296875
timeout_rate: 0.0146484375
```

相对 exp051，clip 从 `0.18` 放宽到 `0.20` 后没有清掉 timeout，反而从 `0.0098` 升到 `0.0146`。final eval 中 filter applied `0.5246`、filter collision override `0.3901`、control safety `0.0876`，说明稍大的 update 会让既有 filter override 更频繁，但没有换来更稳定的 strict 通过。

timeout 子集的 `final_dmax_mean≈0.929`、`final_dispersion_mean≈0.155` 已满足集合几何，但 `final_nearest_neighbor_distance_mean≈0.348`，低于 `success_thresholds.min_pairwise_distance=0.42`，且 `final_min_pairwise_ok_rate=0.0`。这说明剩余失败仍集中在成功区附近的最近邻安全间距 gate，而不是整体集合距离不足。

## 判读重点

- 若 timeout 低于 exp051 的 `0.0098` 且 success/collision 保持达标，说明 exp051 稳定性配置下稍大 clip 可能有助于末端稳定。
- 若 collision 反弹或 success 下降，说明 `clip_epsilon=0.20` 更新过大，应回到 exp051。
- 若 filter/control 介入比例明显升高，需要判为不理想。
- 不应把训练 reward 或 GIF 写成 strict 结论。

## 产物路径

已生成：

```text
outputs/runs/exp055_structured_bicycle_quintic_map25_ppo_clip20/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_clip20/metrics/final_eval_proxy.json
outputs/runs/exp055_structured_bicycle_quintic_map25_ppo_clip20/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_clip20/metrics/strict_acceptance.json
outputs/runs/exp055_structured_bicycle_quintic_map25_ppo_clip20/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_clip20/figures/training_curves.png
outputs/runs/exp055_structured_bicycle_quintic_map25_ppo_clip20/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_clip20/figures/candidate_eval_curves.png
outputs/runs/exp055_structured_bicycle_quintic_map25_ppo_clip20/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_clip20/figures/exp051_exp054_exp055_clip_sweep_comparison.png
```

## 结论

exp055 不是当前主结果。clip `0.20` 能恢复到 exp051 附近的高 success，但 timeout 差于 exp051；结合 exp054 的 `0.16` 退化，当前 clip 扫描结论是 `0.18` 仍是最好点。

## 下一步

回到 exp051 的 `clip_epsilon=0.18`，不继续沿 clip 方向搜索。下一步应围绕成功区附近的最近邻安全间距 gate 做更窄触发的 reward 侧信号，而不是扩大 filter/control 决策权。
