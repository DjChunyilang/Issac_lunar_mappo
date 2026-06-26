# exp022 endpoint safety constrained curriculum filter 随机地形实验

## 目的

exp021 证明课程化 filter 可以恢复集合学习，但也把失败模式推回“会集合但碰撞高”：5 seed mean success `0.6361`，collision `0.1746`。exp022 的目标是只针对这一点迭代：在保留 exp021 课程节奏和集合意图的前提下，把 endpoint / path safety constraint 显式加入子目标后处理。

本轮仍是 proxy 训练，不引入 PhysX、真实车体尺寸、轮地接触、硬性陷车终止或 BC。

## 配置

```text
config: configs/experiment/exp022_randomized_terrain_endpoint_safety_filter.yaml
experiment_id: exp022_randomized_terrain_endpoint_safety_filter
run_id: pure_rl_seed23_20m_endpoint_safety
```

关键设置：

- shared-joint MAPPO pure RL，`bc_updates=0`。
- 随机增强 lunar crater proxy，`randomize_per_reset=true`。
- 通信半径 `12 m`，Actor / Critic 接口仍为 `86 / 54`。
- 2048 CUDA envs，rollout 32，连续 `10240` timesteps，即约 20.97M env steps。
- checkpoint interval `1024` timesteps。
- 候选选择使用 `safe_progress_long`，避免优先选择 success 很低但稍安全的 checkpoint。

constrained curriculum filter：

```text
mode: terrain_safe_candidate_constrained_curriculum
rho_scales: [0.45, 0.70, 0.90, 1.0]
beta_offsets_deg: [-45, -30, -15, 0, 15, 30, 45]
candidate_count: 28
warmup_timesteps: 2048
ramp_timesteps: 4096
apply_probability: 0.0 -> 0.75
score_scale: 0.15 -> 0.75
endpoint_safe_distance: 0.50
path_safe_distance: 0.42
hard_endpoint_near_filter: true
hard_path_collision_filter: true
hard_center_progress_filter: true
center_progress_slack: 0.35
safety_override_after_warmup: true
```

score 相比 exp021 增强的项：

```text
endpoint_near_weight: 8.0
endpoint_collision_weight: 2000.0
path_near_weight: 8.0
path_collision_weight: 2000.0
visible_neighbor_center_weight: 0.50
```

reward 安全项也轻微提高：

```text
safety.near_distance: 0.95
near_distance: 8.0
inter_agent_collision: 120.0
failure_penalty: 60.0
```

## 验收标准

proxy strict gate 不变：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

如果 strict 未通过，本轮诊断重点看：

- collision 是否明显低于 exp021 的 `0.1746`。
- success 是否仍显著高于 exp020 的 `0.0`，避免再次退化为安全但不集合。
- timeout 是否不回到 exp020 的 `0.9506`。
- filter telemetry 中 raw endpoint/path violation 是否被 filtered action 降低。

## 工程验证计划

已新增专项覆盖：

- constrained curriculum warmup 期不覆盖 raw action。
- warmup 后 safety override 可在 raw endpoint unsafe 时强制选择可行候选。
- 不可见 rover 不影响单 rover safety constraint / filter 输出。
- exp022 配置 contract、候选数、约束开关、安全 reward 和 telemetry shape。

标准验证命令：

```bash
.venv_isaaclab/bin/python -m pytest -q \
  tests/test_subgoal_filter.py \
  tests/test_exp019_training.py \
  tests/test_exp020_training.py \
  tests/test_exp021_training.py \
  tests/test_exp022_training.py

.venv_isaaclab/bin/python -m pytest -q -ra
```

CPU / CUDA smoke 见 `docs/runbooks/train_skrl_mappo.md` 的 exp022 section。

## 长训练命令

```bash
ROOT=$(pwd)
mkdir -p outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_launcher

systemd-run --user --unit exp022-endpoint-safety-20m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${ROOT}/outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_launcher/train.log \
  --property=StandardError=append:${ROOT}/outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp022_randomized_terrain_endpoint_safety_filter.yaml \
    --device cuda \
    --timesteps 10240 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_20m_endpoint_safety \
    --rollout-steps 32 \
    --checkpoint-interval 1024 \
    --eval-num-envs 1024 \
    --eval-steps 220 \
    --eval-seed-offset 1000 \
    --bc-updates 0 \
    --selection-gate safe_progress_long
```

监控：

```bash
tail -f outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_launcher/train.log
```

## 产物路径

```text
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/pure_rl_seed23_20m_endpoint_safety/
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_suite/metrics/
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/_launcher/train.log
```

## 当前状态

代码、配置、专项测试和文档已准备。长训练完成前，不写 strict 结论，不提交 `outputs/` 下的 checkpoint、JSON、PNG、GIF 或 TensorBoard 产物。
