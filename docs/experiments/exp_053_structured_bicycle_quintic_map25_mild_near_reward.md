# exp053 structured bicycle quintic map25 mild near reward

## 目的

exp051 是当前新环境栈 local reset 最好候选：dmax `0.1836`、success `0.9883`、collision `0.0020` 均通过，但 timeout `0.0098` 未清零。timeout 子集的 dmax/dispersion 已经足够好，主要卡在最近邻安全间距：`final_nearest_neighbor_distance_mean=0.3527`、`final_min_pairwise_ok_rate=0.1000`。

exp052 说明不能把 entropy taper 提前到 `8192`；这会使 success 和 timeout 明显退化。

exp053 回到 exp051，只做一个 reward 侧小步实验：把已有 safety reward 的 `near_distance` 系数从 `2.4` 提到 `2.8`。目标是让 policy 更主动避开 `0.28–0.42 m` 的最近邻灰区，而不是通过新增 filter/control 规划能力修正动作。

## 配置

```text
configs/experiment/exp053_structured_bicycle_quintic_map25_mild_near_reward.yaml
```

相对 exp051 的唯一实质变量：

- `reward.coefficients.near_distance: 2.4 -> 2.8`

保持不变：

- Actor 输出仍是单点 `[rho, beta]`；
- 不引入多点采样；
- `safety.near_distance=0.72` 不变；
- filter、control safety、reset、terrain、trajectory、episode/eval steps 全部保持 exp051；
- PPO schedule 保持 exp051：`learning_rate=1.0e-4`、`clip_epsilon=0.18`、`initial_log_std=-0.95`、`entropy_schedule_timesteps=12288`。

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
  --config configs/experiment/exp053_structured_bicycle_quintic_map25_mild_near_reward.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_mild_near_reward \
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
  --config configs/experiment/exp053_structured_bicycle_quintic_map25_mild_near_reward.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_mild_near_reward \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

## 长训命令

```bash
mkdir -p outputs/runs/exp053_structured_bicycle_quintic_map25_mild_near_reward/_launcher

systemd-run --user --unit exp053-structured-bicycle-quintic-map25-mild-near-reward-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp053_structured_bicycle_quintic_map25_mild_near_reward/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp053_structured_bicycle_quintic_map25_mild_near_reward/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp053_structured_bicycle_quintic_map25_mild_near_reward.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_mild_near_reward \
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
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_mild_near_reward` | smoke only | 工程通过；`8 env / 8 timesteps`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0845`、action std `0.0582` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_mild_near_reward` | smoke only | 工程通过；`256 env / 128 timesteps / rollout 64`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0719`、action std `0.0849` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_mild_near_reward` | `ppo_timestep_020480.pt` / `best.pt` | dmax ratio `0.2049`、success `0.6416`、collision `0.0039`、timeout `0.3545` | 未通过；dmax/success/timeout 失败 |

## 结果分析

exp053 明显弱于 exp051。小幅提高全局 near reward 后，策略确实更倾向保持间距，但整体集合与 terminal hold 大幅退化：success 从 exp051 的 `0.9883` 降到 `0.6416`，timeout 从 `0.0098` 升到 `0.3545`，dmax ratio 也从 `0.1836` 退到 `0.2049`。

| 实验 | near reward | checkpoint | dmax ratio | success | collision | timeout | min pairwise ok | filter applied | filter override | control safety |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp051 | `2.4` | `013312` | `0.1836` | `0.9883` | `0.0020` | `0.0098` | `0.9886` | `0.5072` | `0.3538` | `0.0742` |
| exp053 | `2.8` | `020480` | `0.2049` | `0.6416` | `0.0039` | `0.3545` | `0.9432` | `0.4080` | `0.2765` | `0.1365` |

timeout 子集有 `363 / 1024` 个 episode，`final_dmax_mean=1.5171`、`final_dispersion_mean=0.4750`、`final_nearest_neighbor_distance_mean=0.4836`、`final_min_pairwise_ok_rate=0.6749`、`max_success_hold_count_mean=0.1983`。这说明 exp053 不是卡在 exp051 那种少量最近邻灰区，而是全局 near penalty 把队形推散，导致无法稳定进入成功区。

本轮没有改变 Actor 输出语义，也没有新增 filter/control 规划能力；负结果可以归因为 reward 中全局 near penalty 即使小幅提高也过强。

## 产物路径

训练后应生成：

```text
outputs/runs/exp053_structured_bicycle_quintic_map25_mild_near_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_mild_near_reward/metrics/final_eval_proxy.json
outputs/runs/exp053_structured_bicycle_quintic_map25_mild_near_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_mild_near_reward/metrics/strict_acceptance.json
outputs/runs/exp053_structured_bicycle_quintic_map25_mild_near_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_mild_near_reward/figures/training_curves.png
outputs/runs/exp053_structured_bicycle_quintic_map25_mild_near_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_mild_near_reward/figures/candidate_eval_curves.png
outputs/runs/exp053_structured_bicycle_quintic_map25_mild_near_reward/pure_rl_seed23_40m_structured_bicycle_quintic_map25_mild_near_reward/figures/exp051_exp052_exp053_candidate_comparison.png
```

## 结论

exp053 不是有效方向。全局 near reward 小幅增强也会让策略过度保持间距，牺牲集合进度和 terminal hold。

## 下一步

回到 exp051 作为当前最好基线。后续不应继续提高全局 `reward.coefficients.near_distance`；如果要处理最近邻灰区，应优先考虑更局部、更终端化的策略侧信息或训练设置，而不是全局安全惩罚。
