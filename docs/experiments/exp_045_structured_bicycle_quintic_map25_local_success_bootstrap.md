# exp045 structured bicycle quintic map25 local success bootstrap

## 目的

exp044 证明 initial-state curriculum 能让新环境栈从“几乎不靠拢”进步到中距离靠拢，但 final eval 仍停在 dmax ratio `0.4796`、success `0.0`、timeout `0.9980`。exp045 不再直接追求 exp044 的完整 reset 难度，而是先做 local-success bootstrap：

- 保持 `branched_v1 / structured_v1`、`bicycle`、`quintic`、`25 m × 25 m` 地图和 `communication_radius=0.0`；
- 把目标 reset 分布缩小到更容易进入 success basin 的局部范围；
- 稍微放大 `[rho,beta]` 可达范围与 bicycle 转角；
- 增强中距离 gather progress，减轻 terrain/filter 在早期对集合的抑制；
- 目标是先让 success 从 `0` 起跳，再在后续实验逐步扩展回 exp044 难度。

本轮从随机初始化开始，不续训 exp044 checkpoint。

## 配置

```text
configs/experiment/exp045_structured_bicycle_quintic_map25_local_success_bootstrap.yaml
```

关键设置：

- Actor/Critic：`branched_v1 / structured_v1`，接口保持 `86 / 54`。
- 动力学/轨迹：`low_level_control.kinematic_model=bicycle`，`trajectory_generator.geometry_method=quintic`，`n_trajectory_points=12`。
- 地图/通信：`world_xy_limit=12.5`、`crater_field_size=25.0`、`communication_radius=0.0`。
- 目标 reset 分布：spawn radius `2.4–3.4 m`，center range `±1.0 m`，jitter `0.25 m`。
- initial-state curriculum：前 `4096` timesteps 使用 spawn radius `1.6–2.4 m`、center range `±0.5 m`、jitter `0.20 m`；随后 `8192` timesteps 线性 ramp 到目标分布。
- 动作/低层：`rho_max=1.6`、`beta_max=60°`、`max_steer_angle≈45°`、`reference_speed=0.9`、`max_linear_speed=1.1`。
- 地形：`crater_count=30`、`amplitude=0.09`、`randomize_per_reset=true`、`random_translation_m=5.0`。
- reward：`dmax_progress=5.5`、`dispersion_progress=2.4`、`oracle_mean_distance_progress=3.0`，terrain weight 临时降到 `0.20`。
- filter：保留 safety/path 候选过滤，但 `apply_probability_end=0.35`、`score_scale_end=0.50`，减少早期 post-processing 对集合意图的替换。
- MAPPO：2048 env、rollout 64、LR `1.2e-4`、entropy `0.0012 -> 0.00025` over 8192 timesteps。
- 预算：`20480` timesteps，约 `41,943,040` env steps，checkpoint interval `1024`。

## 严格标准

仍使用标准 proxy strict gate：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

注意：exp045 的 reset 分布比 exp044 容易，因此即便 strict 通过，也只能说明“local bootstrap 难度通过”，不能直接等价为 exp044 完整难度通过。

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
  --config configs/experiment/exp045_structured_bicycle_quintic_map25_local_success_bootstrap.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_local_success \
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
  --config configs/experiment/exp045_structured_bicycle_quintic_map25_local_success_bootstrap.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_local_success \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

## 长训命令

```bash
mkdir -p outputs/runs/exp045_structured_bicycle_quintic_map25_local_success_bootstrap/_launcher

systemd-run --user --unit exp045-structured-bicycle-quintic-map25-local-success-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp045_structured_bicycle_quintic_map25_local_success_bootstrap/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp045_structured_bicycle_quintic_map25_local_success_bootstrap/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp045_structured_bicycle_quintic_map25_local_success_bootstrap.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_success \
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
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_local_success` | smoke only | optimizer `1`、joint update `2`、terrain delta `0.1019`、action std `0.0637` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_local_success` | smoke only | optimizer `1`、joint update `2`、terrain delta `0.0643`、action std `0.0641` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_success` | `ppo_timestep_020480.pt` | dmax `0.2734` / success `0.1846` / collision `0.0` / timeout `0.8174` | 未通过 |

smoke telemetry 已确认：

```text
CPU smoke final progress: 8
CUDA smoke final progress: 128
effective spawn radius: 1.6–2.4 m
effective center range: 0.5 m
```

## 判读重点

训练已完成，机器可读结果：

```text
timesteps: 20480
env_steps: 41943040
best_candidate: ppo_timestep_020480.pt
policy_parameter_delta_l2: 8.0509
terrain_input_weight_delta_l2: 1.2003
post_training_action_std: 0.6819
optimizer_count: 1
joint_update_count: 320
```

final eval：

```text
initial_dmax: 6.0234
final_dmax: 1.6466
dmax_reduction_ratio: 0.2734
final_dispersion: 0.5278
final_nearest_neighbor_distance: 0.6534
success_rate: 0.1846
safe_success_rate: 0.1846
collision_rate: 0.0
timeout_rate: 0.8174
max_success_hold_count_mean: 1.5059
```

判读：

- exp045 相比 exp044 的 `success=0` 有明确改善，说明 local-success bootstrap 方向有效。
- collision 为 `0.0`，当前不是安全不足。
- 大部分失败 episode 停在 success 区外侧：final dmax `1.65 m` 高于 `1.25 m`，dispersion `0.53` 高于 `0.30`。
- control safety applied fraction 约 `0.28`，filter applied fraction 约 `0.43`；在 local task 中这可能对末端收缩形成阻尼。
- 下一轮应继续保持 local reset 分布，但降低 filter/control-safety 末端介入强度，并增强 dmax/dispersion 末端收缩和 hold reward。

## 产物路径

```text
outputs/runs/exp045_structured_bicycle_quintic_map25_local_success_bootstrap/
  pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_success/
  _launcher/train.log
```

机器可读事实来源：

```text
outputs/runs/exp045_structured_bicycle_quintic_map25_local_success_bootstrap/pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_success/metrics/summary.json
outputs/runs/exp045_structured_bicycle_quintic_map25_local_success_bootstrap/pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_success/metrics/final_eval_proxy.json
outputs/runs/exp045_structured_bicycle_quintic_map25_local_success_bootstrap/pure_rl_seed23_40m_structured_bicycle_quintic_map25_local_success/metrics/strict_acceptance.json
```

## 结论

exp045 未通过 strict gate，不能作为收敛结果；但它把 local reset 下 success 从 exp044 的 `0` 提升到 `0.1846`，是新环境栈恢复集合学习的有效中间台阶。

## 下一步

新建 exp046：保持 exp045 的 local reset 分布，降低 filter/control-safety 介入强度，增强末端 dmax/dispersion 收缩和 success hold，目标是把 local success 显著推高。
