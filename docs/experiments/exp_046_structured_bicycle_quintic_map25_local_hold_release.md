# exp046 structured bicycle quintic map25 local hold release

## 目的

exp045 已经让新环境栈恢复非零 success，但 final eval 仍只有 `0.1846`。失败模式不是碰撞，而是多数 episode 停在 success 区外侧：

```text
final_dmax: 1.6466  # threshold 1.25
final_dispersion: 0.5278  # threshold 0.30
final_nearest_neighbor_distance: 0.6534
collision_rate: 0.0
timeout_rate: 0.8174
```

exp046 保持 exp045 的 local reset 分布和新环境栈，但做“末端释放 + 收缩”：

- 降低 subgoal filter 的介入概率和 score scale，减少对末端集合意图的替换；
- 调弱 low-level control safety 的触发距离和强度，避免 local task 中过早减速；
- 增强 dmax/dispersion progress、success bonus 和 timeout penalty；
- 继续保持 collision penalty 与 success 安全间距不变，避免把安全问题重新打开。

本轮从随机初始化开始，不续训 exp045 checkpoint。

## 配置

```text
configs/experiment/exp046_structured_bicycle_quintic_map25_local_hold_release.yaml
```

关键设置：

- Actor/Critic：`branched_v1 / structured_v1`，接口保持 `86 / 54`。
- 动力学/轨迹：`low_level_control.kinematic_model=bicycle`，`trajectory_generator.geometry_method=quintic`。
- 地图/通信：`world_xy_limit=12.5`、`crater_field_size=25.0`、`communication_radius=0.0`。
- reset 分布：沿用 exp045，目标 spawn radius `2.4–3.4 m`，课程起点 `1.6–2.4 m`。
- filter：`apply_probability_end=0.22`、`score_scale_end=0.35`，低于 exp045 的 `0.35/0.50`。
- control safety：activation distance 从 `0.82` 降到 `0.68`，projection strength 从 `0.95` 降到 `0.70`，min linear scale 从 `0.25` 提到 `0.40`。
- low-level：`reference_speed=1.0`、`max_linear_speed=1.2`，success-zone damping scale 提到 `0.65`。
- reward：`dmax_progress=7.0`、`dispersion_progress=3.2`、`success_bonus=85`、`timeout_penalty=45`，terrain weight 降到 `0.15`。
- MAPPO：2048 env、rollout 64、LR `1.2e-4`、entropy `0.0010 -> 0.00020` over 8192 timesteps。
- 预算：`20480` timesteps，约 `41,943,040` env steps，checkpoint interval `1024`。

## 严格标准

仍使用标准 proxy strict gate：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

注意：exp046 仍是 local reset 分布，不能直接等价为 exp044 完整难度通过。

## 工程验证

专项测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaaclab/bin/python -m pytest -q \
  tests/test_config_wiring.py \
  tests/test_observation.py \
  tests/test_skrl_mappo_semantics.py
```

完整测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaaclab/bin/python -m pytest -q -ra
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp046_structured_bicycle_quintic_map25_local_hold_release.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_local_hold_release \
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
  --config configs/experiment/exp046_structured_bicycle_quintic_map25_local_hold_release.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_local_hold_release \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

## 长训命令

```bash
mkdir -p outputs/runs/exp046_structured_bicycle_quintic_map25_local_hold_release/_launcher

systemd-run --user --unit exp046-structured-bicycle-quintic-map25-local-hold-release-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp046_structured_bicycle_quintic_map25_local_hold_release/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp046_structured_bicycle_quintic_map25_local_hold_release/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp046_structured_bicycle_quintic_map25_local_hold_release.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_hold_release \
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
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_local_hold_release` | smoke only | 工程通过；`8 env / 8 timesteps`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0910`、action std `0.0524` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_local_hold_release` | smoke only | 工程通过；`256 env / 128 timesteps / rollout 64`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0750`、action std `0.0667` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_hold_release` | `ppo_timestep_015360.pt` / `best.pt` | dmax ratio `0.2424`、success `0.6123`、collision `0.0`、timeout `0.3877` | 未通过 |

## 判读重点

- exp046 达成了第一目标：success 从 exp045 的 `0.1846` 提高到 `0.6123`，collision 保持 `0.0`。
- 未达 strict 的 gate 是 dmax ratio、success 和 timeout：`0.2424 > 0.20`、`0.6123 < 0.90`、`0.3877 > 0`。
- final eval 中 `final_dmax=1.4600`、`final_dispersion=0.5378`，仍高于 `dmax=1.25` / `dispersion=0.30` success gate。
- timeout episode 的平均 `final_dmax` 约 `1.98`、`final_dispersion` 约 `0.95`，说明失败样本不是只差 hold-step，而是仍停在成功区外。
- `filter_applied_fraction=0.2749`、`control_safety_applied_fraction=0.0559`，且 collision 为 0；下一轮可以继续释放 near/filter/control safety 的末端阻尼，同时增强 dmax/dispersion/timeout shaping。

## 产物路径

```text
outputs/runs/exp046_structured_bicycle_quintic_map25_local_hold_release/
  pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_hold_release/
  _launcher/train.log
```

机器可读事实来源：

```text
outputs/runs/exp046_structured_bicycle_quintic_map25_local_hold_release/pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_hold_release/metrics/summary.json
outputs/runs/exp046_structured_bicycle_quintic_map25_local_hold_release/pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_hold_release/metrics/final_eval_proxy.json
outputs/runs/exp046_structured_bicycle_quintic_map25_local_hold_release/pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_hold_release/metrics/strict_acceptance.json
```

## 当前结论

exp046 未通过 strict，但它是新环境栈下第一次把 local reset success 推到 `0.6+` 且保持 collision 为 0 的有效改进。下一轮 exp047 应聚焦 terminal convergence：允许更紧的安全缓冲和更少的控制阻尼，同时提高 dmax/dispersion/timeout/成功保持奖励，目标是把 `dmax ratio` 推入 `<=0.20` 并把 timeout 大幅压低。
