# exp048 structured bicycle quintic map25 terminal drive

## 目的

exp047 相比 exp046 又前进了一步：success `0.7188`、collision `0.0059`、dmax ratio `0.2132`，但 strict 仍失败于 dmax、success 和 timeout。timeout episode 平均仍停在成功区外，说明瓶颈不是只差 hold-step，而是部分 episode 进入低速/保守队形。

exp048 保持 exp047 的 local reset 分布和基本安全框架，只做小步 terminal drive 调整：

- 更强 dispersion/dmax 收缩；
- 更高末端参考速度和控制下限；
- 更少 success-zone 减速；
- 小幅收紧 hold-zone 距离，但不降低 success 最近邻安全 gate。

## 配置

```text
configs/experiment/exp048_structured_bicycle_quintic_map25_terminal_drive.yaml
```

关键设置：

- Actor/Critic：`branched_v1 / structured_v1`，接口保持 `86 / 54`。
- 动力学/轨迹：`bicycle` proxy + `quintic` trajectory。
- reset 分布：保持 exp047 的目标 `2.4–3.4 m`、课程起点 `1.6–2.4 m`。
- filter：`apply_probability_end=0.18`、`score_scale_end=0.30`，`hold_zone_pairwise_distance=0.46`。
- terminal drive：`reference_speed=1.15`、`max_linear_speed=1.35`、`projection_min_linear_scale=0.65`、`success_zone_linear_scale=0.95`。
- reward：`dmax_progress=9.5`、`dispersion_progress=6.0`、`success_bonus=130`、`timeout_penalty=80`，terrain weight 降到 `0.10`。
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

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaaclab/bin/python -m pytest -q \
  tests/test_config_wiring.py \
  tests/test_skrl_mappo_semantics.py
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp048_structured_bicycle_quintic_map25_terminal_drive.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_terminal_drive \
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
  --config configs/experiment/exp048_structured_bicycle_quintic_map25_terminal_drive.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_terminal_drive \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

## 长训命令

```bash
mkdir -p outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/_launcher

systemd-run --user --unit exp048-structured-bicycle-quintic-map25-terminal-drive-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp048_structured_bicycle_quintic_map25_terminal_drive.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_drive \
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
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_terminal_drive` | smoke only | 工程通过；`8 env / 8 timesteps`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0958`、action std `0.0517` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_terminal_drive` | smoke only | 工程通过；`256 env / 128 timesteps / rollout 64`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0806`、action std `0.0800` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_drive` | `ppo_timestep_008192.pt` / `best.pt` | dmax ratio `0.1866`、success `0.9844`、collision `0.0020`、timeout `0.0137` | 未通过 |

## 判读重点

- exp048 已经通过 dmax、success 和 collision gate：`0.1866 <= 0.20`、`0.9844 >= 0.90`、`0.0020 <= 0.02`。
- 唯一失败项是 timeout：`0.0137 > 0`，约 14/1024 个 episode。
- timeout episode 的 `final_dmax_mean≈1.076`、`final_dispersion_mean≈0.220` 已满足几何阈值，但 `final_nearest_neighbor_distance_mean≈0.393`，低于 `min_pairwise_distance=0.42`；失败集中在最近邻安全间距灰区。
- 下一轮不应继续全局加速，而应轻量增强 terminal spacing，让最后少量样本从 `0.28–0.42m` 灰区退到安全成功区。

## 产物路径

```text
outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_drive/metrics/final_eval_proxy.json
outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_drive/metrics/strict_acceptance.json
outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_drive/figures/training_curves.png
outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_drive/figures/candidate_eval_curves.png
outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_drive/figures/terrain_height_map.png
outputs/runs/exp048_structured_bicycle_quintic_map25_terminal_drive/pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_drive/videos/proxy_eval_rollout.gif
```

## 当前结论

exp048 是新环境栈 local reset 当前最佳结果：dmax/success/collision 已全部过 strict，仅 timeout gate 因少量末端最近邻间距不足失败。exp049 证明全局 terminal spacing 修正过强，因此下一步应回到 exp048 主体，只做窄触发 spacing 修正。
