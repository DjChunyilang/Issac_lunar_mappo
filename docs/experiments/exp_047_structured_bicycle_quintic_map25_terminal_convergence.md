# exp047 structured bicycle quintic map25 terminal convergence

## 目的

exp046 在新环境栈下显著恢复了 local reset 的集合能力：final eval success 从 exp045 的 `0.1846` 提升到 `0.6123`，collision 保持 `0.0`。但 strict 仍失败：

```text
dmax_reduction_ratio: 0.2424
success_rate: 0.6123
collision_rate: 0.0
timeout_rate: 0.3877
final_dmax: 1.4600
final_dispersion: 0.5378
```

exp047 不扩大 reset 分布，而是先把 exp046 的 local task 推向更接近 strict：释放末端安全缓冲和低层控制阻尼，同时加强 dmax、dispersion、timeout 和 success shaping。

## 配置

```text
configs/experiment/exp047_structured_bicycle_quintic_map25_terminal_convergence.yaml
```

关键设置：

- Actor/Critic：`branched_v1 / structured_v1`，接口保持 `86 / 54`。
- 动力学/轨迹：`bicycle` proxy + `quintic` trajectory。
- 地图/通信：`25 m × 25 m`，`communication_radius=0.0`。
- reset 分布：保持 exp046 的目标 `2.4–3.4 m`、课程起点 `1.6–2.4 m`。
- filter：`apply_probability_end=0.16`、`score_scale_end=0.28`。
- 末端安全缓冲：`hold_zone_pairwise_distance=0.48`、`endpoint_safe_distance=0.42`、`path_safe_distance=0.30`。
- control safety：activation distance `0.62`、projection strength `0.50`、min linear scale `0.55`、success-zone scale `0.80`。
- reward：`dmax_progress=9.0`、`dispersion_progress=4.5`、`success_bonus=115`、`timeout_penalty=65`，terrain weight 降到 `0.12`。
- 预算：`20480` timesteps，约 `41,943,040` env steps，checkpoint interval `1024`。

## 严格标准

仍使用标准 proxy strict gate：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

注意：exp047 仍是 local reset 分布，不等价于 exp044 完整难度收敛。

## 验证命令

专项配置/语义测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaaclab/bin/python -m pytest -q \
  tests/test_config_wiring.py \
  tests/test_skrl_mappo_semantics.py
```

CPU smoke：

```bash
.venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
  --config configs/experiment/exp047_structured_bicycle_quintic_map25_terminal_convergence.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_terminal_convergence \
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
  --config configs/experiment/exp047_structured_bicycle_quintic_map25_terminal_convergence.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_terminal_convergence \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

## 长训命令

```bash
mkdir -p outputs/runs/exp047_structured_bicycle_quintic_map25_terminal_convergence/_launcher

systemd-run --user --unit exp047-structured-bicycle-quintic-map25-terminal-convergence-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp047_structured_bicycle_quintic_map25_terminal_convergence/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp047_structured_bicycle_quintic_map25_terminal_convergence/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp047_structured_bicycle_quintic_map25_terminal_convergence.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_convergence \
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
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_terminal_convergence` | smoke only | 工程通过；`8 env / 8 timesteps`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0948`、action std `0.0540` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_terminal_convergence` | smoke only | 工程通过；`256 env / 128 timesteps / rollout 64`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0944`、action std `0.0788` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_convergence` | `ppo_timestep_015360.pt` / `best.pt` | dmax ratio `0.2132`、success `0.7188`、collision `0.0059`、timeout `0.2764` | 未通过 |

## 判读重点

- exp047 相比 exp046 继续改善：success `0.6123 -> 0.7188`，dmax ratio `0.2424 -> 0.2132`，timeout `0.3877 -> 0.2764`。
- collision 从 `0.0` 变为 `0.0059`，仍在 strict `<=0.02` 内，说明 terminal release 尚未破坏安全 gate。
- 仍未 strict 的 gate 是 dmax、success 和 timeout；final eval 的 `final_dmax=1.2840` 接近 `1.25`，但 `final_dispersion=0.4021` 仍高于 `0.30`。
- timeout episode 平均 `final_dmax≈1.75`、`final_dispersion≈0.74`、速度几乎为 0，不是只差 hold-step；下一轮应提高 terminal drive 与 dispersion 收缩，而不是只延长 episode。

## 产物路径

```text
outputs/runs/exp047_structured_bicycle_quintic_map25_terminal_convergence/
  pure_rl_seed23_40m_structured_bicycle_quintic_map25_terminal_convergence/
  _launcher/train.log
```

## 当前结论

exp047 未通过 strict，但它把新环境栈 local reset 推到目前最好：success `0.7188`、collision `0.0059`、dmax ratio `0.2132`。下一轮 exp048 应在 exp047 附近小步增强 terminal drive/dispersion 收缩，同时保持 collision strict。
