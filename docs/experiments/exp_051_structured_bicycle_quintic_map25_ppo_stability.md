# exp051 structured bicycle quintic map25 PPO stability

## 目的

exp050 证明“提高 terminal hold / timeout shaping 并降低探索噪声”不能清掉 exp048 的 timeout 尾部，反而使 timeout 从 `0.0137` 升到 `0.0352`。

exp051 回到 exp048 主体，只隔离 PPO 稳定性调整：

- 不改 Actor 输出；
- 不新增多点采样；
- 不增强 filter/control；
- reward、filter、control safety 全部保持 exp048；
- 只调整 PPO 学习率、clip、entropy schedule 和初始 log std。

目标是判断 exp050 的退化是否主要来自 reward/hold shaping，而不是 PPO 稳定性本身。

## 配置

```text
configs/experiment/exp051_structured_bicycle_quintic_map25_ppo_stability.yaml
```

关键设置：

- Actor/Critic：`branched_v1 / structured_v1`，接口保持 `86 / 54`。
- 动力学/轨迹：`bicycle` proxy + `quintic` trajectory。
- reset、terrain、filter、control safety、reward：保持 exp048。
- PPO 相对 exp048 的变化：`learning_rate=1.0e-4`、`clip_epsilon=0.18`、`initial_log_std=-0.95`、`entropy_loss_scale=0.0009`、`entropy_loss_scale_end=0.00010`、`entropy_schedule_timesteps=12288`。
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
  --config configs/experiment/exp051_structured_bicycle_quintic_map25_ppo_stability.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_ppo_stability \
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
  --config configs/experiment/exp051_structured_bicycle_quintic_map25_ppo_stability.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_ppo_stability \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

checkpoint multi-seed 复验：

```bash
.venv_isaaclab/bin/python scripts/evaluate_proxy_checkpoint_seed_sweep.py \
  --config outputs/runs/exp051_structured_bicycle_quintic_map25_ppo_stability/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_stability/config/experiment.yaml \
  --run-dir outputs/runs/exp051_structured_bicycle_quintic_map25_ppo_stability/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_stability \
  --checkpoint ppo_timestep_012288.pt \
  --checkpoint ppo_timestep_013312.pt \
  --checkpoint ppo_timestep_014336.pt \
  --seeds 1023,2023,3023,4023 \
  --device cuda \
  --num-envs 1024 \
  --steps 320
```

## 长训命令

```bash
mkdir -p outputs/runs/exp051_structured_bicycle_quintic_map25_ppo_stability/_launcher

systemd-run --user --unit exp051-structured-bicycle-quintic-map25-ppo-stability-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp051_structured_bicycle_quintic_map25_ppo_stability/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp051_structured_bicycle_quintic_map25_ppo_stability/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp051_structured_bicycle_quintic_map25_ppo_stability.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_stability \
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
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_ppo_stability` | smoke only | 工程通过；`8 env / 8 timesteps`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0845`、action std `0.0582` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_ppo_stability` | smoke only | 工程通过；`256 env / 128 timesteps / rollout 64`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0715`、action std `0.0875` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_stability` | `ppo_timestep_013312.pt` / `best.pt` | dmax ratio `0.1836`、success `0.9883`、collision `0.0020`、timeout `0.0098` | 未通过；仅 timeout 失败 |

## 结果分析

exp051 相对 exp048 有小幅但真实的改善：timeout 从 `0.0137` 降到 `0.0098`，dmax ratio、success、hold count 也略好；相对 exp050 则明显恢复了 success/timeout。说明 exp050 的退化主要来自 terminal hold / timeout shaping，而不是 PPO 稳定性调整本身。

| 实验 | dmax ratio | success | collision | timeout | filter applied | filter override | control safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp048 | `0.1866` | `0.9844` | `0.0020` | `0.0137` | `0.5717` | `0.3674` | `0.0710` |
| exp050 | `0.1847` | `0.9590` | `0.0059` | `0.0352` | `0.4205` | `0.2798` | `0.0970` |
| exp051 | `0.1836` | `0.9883` | `0.0020` | `0.0098` | `0.5072` | `0.3538` | `0.0742` |

剩余失败来自 `10 / 1024` 个 timeout episode。timeout 子集的 `final_dmax_mean=0.9291`、`final_dispersion_mean=0.1632`、`final_nearest_neighbor_distance_mean=0.3527`、`max_success_hold_count_mean=0.3000`，说明尾部不再是“几乎完成 hold 但差一点”，而是少量 episode 在末端队形/间距附近没有进入稳定成功区。

本轮没有改变 Actor 输出语义，也没有引入多点采样让 filter 选择；filter/control 仍是 exp048 的原有机制。因此 exp051 支持继续沿单点 `[rho, beta]` 子目标输出主线迭代，而不是把策略改成候选点生成器。

### Checkpoint multi-seed 复验

为区分“checkpoint 选点噪声”和“真实尾部 timeout”，对 exp051 附近三个 checkpoint 做 `4` 个 eval seed 复验。结果显示 `ppo_timestep_013312.pt` 仍是附近最稳的点，但没有任何 seed 达到 `timeout_rate == 0`：

| checkpoint | eval seeds | timeout mean | timeout min/max | success mean | collision max | dmax max | strict pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ppo_timestep_012288.pt` | 4 | `0.0205` | `0.0186 / 0.0215` | `0.9722` | `0.0088` | `0.1831` | `0 / 4` |
| `ppo_timestep_013312.pt` | 4 | `0.0134` | `0.0098 / 0.0166` | `0.9841` | `0.0039` | `0.1852` | `0 / 4` |
| `ppo_timestep_014336.pt` | 4 | `0.0295` | `0.0215 / 0.0381` | `0.9651` | `0.0078` | `0.1834` | `0 / 4` |

解释：当前 `best.pt` 的选择并没有明显选错，013312 在 success、collision 和 timeout 平衡上仍最好；但 timeout 尾部跨 seed 稳定存在，后续不能只依赖 checkpoint reselection 或单次 eval 运气，需要回到策略训练信号本身。

## 产物路径

训练后应生成：

```text
outputs/runs/exp051_structured_bicycle_quintic_map25_ppo_stability/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_stability/metrics/final_eval_proxy.json
outputs/runs/exp051_structured_bicycle_quintic_map25_ppo_stability/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_stability/metrics/strict_acceptance.json
outputs/runs/exp051_structured_bicycle_quintic_map25_ppo_stability/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_stability/figures/training_curves.png
outputs/runs/exp051_structured_bicycle_quintic_map25_ppo_stability/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_stability/figures/candidate_eval_curves.png
outputs/runs/exp051_structured_bicycle_quintic_map25_ppo_stability/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_stability/figures/exp048_exp050_exp051_candidate_comparison.png
outputs/runs/exp051_structured_bicycle_quintic_map25_ppo_stability/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_stability/metrics/checkpoint_seed_sweep/summary.json
```

## 结论

exp051 是当前新环境栈 local reset 下最好的 proxy 候选，但不是 strict pass：dmax/success/collision 已过，timeout `0.0098` 仍未满足 `timeout_rate == 0`。multi-seed checkpoint 复验证明 013312 附近选点相对稳定，剩余问题不是简单换 checkpoint 可以解决的 eval 偶然性。

## 下一步

暂时维持原动作接口，不改为多点采样输出。下一轮应以 exp051 为新基线，在 reward/观测/网络或 PPO schedule 上做小步隔离实验，目标是清掉跨 eval seed 稳定存在的尾部 timeout，而不是继续增强 filter/control 的规划能力。
