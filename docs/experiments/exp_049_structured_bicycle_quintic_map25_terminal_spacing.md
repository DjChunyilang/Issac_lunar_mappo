# exp049 structured bicycle quintic map25 terminal spacing

## 目的

exp048 已经把新环境栈 local reset 推到接近 strict：

```text
dmax_reduction_ratio: 0.1866  # passed
success_rate: 0.9844          # passed
collision_rate: 0.0020        # passed
timeout_rate: 0.0137          # failed
```

剩余 timeout 不是整体未集合：timeout episode 的 `final_dmax_mean≈1.076`、`final_dispersion_mean≈0.220` 已满足 success 几何阈值，但 `final_nearest_neighbor_distance_mean≈0.393`，低于 `success_thresholds.min_pairwise_distance=0.42`。也就是说，最后少量 episode 卡在 `collision_distance=0.28` 与 `min_pairwise_distance=0.42` 之间的最近邻安全灰区。

exp049 的目标是闭合这个 terminal spacing 灰区，而不是继续全局加速。

## 配置

```text
configs/experiment/exp049_structured_bicycle_quintic_map25_terminal_spacing.yaml
```

关键设置：

- Actor/Critic：`branched_v1 / structured_v1`，接口保持 `86 / 54`。
- 动力学/轨迹：`bicycle` proxy + `quintic` trajectory。
- reset 分布：保持 exp048 的目标 `2.4–3.4 m`、课程起点 `1.6–2.4 m`。
- filter spacing：`hold_zone_pairwise_distance=0.52`、`hold_zone_spacing_weight=4.60`、`endpoint_safe_distance=0.44`、`path_safe_distance=0.32`。
- filter 介入：`apply_probability_end=0.20`、`score_scale_end=0.32`，仍保持课程化，不做 hard filter。
- control safety：`projection_activation_distance=0.64`、`projection_strength=0.55`、`projection_min_linear_scale=0.58`、success-zone scale `0.88`。
- reward：`near_distance=3.4`、`dispersion_progress=6.2`、`timeout_penalty=90`、`success_bonus=135`。
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
  --config configs/experiment/exp049_structured_bicycle_quintic_map25_terminal_spacing.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_terminal_spacing \
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
  --config configs/experiment/exp049_structured_bicycle_quintic_map25_terminal_spacing.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_terminal_spacing \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

## 长训命令

```bash
mkdir -p outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/_launcher

systemd-run --user --unit exp049-structured-bicycle-quintic-map25-terminal-spacing-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp049_structured_bicycle_quintic_map25_terminal_spacing.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_spacing \
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
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_terminal_spacing` | smoke only | 工程通过；`8 env / 8 timesteps`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0989`、action std `0.0588` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_terminal_spacing` | smoke only | 工程通过；`256 env / 128 timesteps / rollout 64`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0855`、action std `0.0774` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_spacing` | `ppo_timestep_010240.pt` / `best.pt` | dmax ratio `0.1884`、success `0.8926`、collision `0.0010`、timeout `0.1064` | 未通过 |

## 失败分析

- exp049 没有达成目标：timeout 从 exp048 的 `0.0137` 升到 `0.1064`，success 从 `0.9844` 降到 `0.8926`。
- collision 进一步降低到 `0.0010`，说明 spacing / safety 修正确实更保守；但这种保守性把部分 episode 推出了 success hold 区。
- timeout episode 不再主要是 exp048 那种 `0.28–0.42 m` 最近邻灰区：exp049 timeout episode 的 `final_nearest_neighbor_distance_mean≈0.569`、`final_min_pairwise_ok_rate≈0.917`，但 `final_dmax_mean≈1.345`、`final_dispersion_mean≈0.401`，说明过强 spacing 破坏了末段几何收缩。
- 因此下一轮不应继续加大全局 terminal spacing、endpoint safety 或 control damping，而应回退 exp048 主体，只做窄触发的 terminal spacing 项。

## 产物路径

```text
outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_spacing/metrics/final_eval_proxy.json
outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_spacing/metrics/strict_acceptance.json
outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_spacing/figures/training_curves.png
outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_spacing/figures/candidate_eval_curves.png
outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_spacing/figures/terrain_height_map.png
outputs/runs/exp049_structured_bicycle_quintic_map25_terminal_spacing/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_spacing/videos/proxy_eval_rollout.gif
```

## 当前结论

exp049 是一个有价值的反例：它证明“把最近邻间距整体推大”会降低碰撞，但会牺牲末端集合/hold 稳定性。当前最佳仍是 exp048；后续应采用只在接近成功区且最近邻低于 `0.42 m` 时生效的窄触发修正，而不是全局 spacing filter。
