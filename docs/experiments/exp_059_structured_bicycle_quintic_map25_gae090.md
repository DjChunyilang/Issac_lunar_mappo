# exp059 structured bicycle quintic map25 GAE 0.90

## 目的

exp051 是当前新环境栈 local reset 最好候选，dmax/success/collision 已过 strict，但 timeout `0.0098` 未清零。exp058 说明单纯把 `gamma` 提高到 `0.995` 会拖慢 terminal convergence，不能靠更长折扣 horizon 修复尾部 timeout。

exp059 回到 exp051，只把 GAE 参数从 `gae_lambda=0.95` 降到 `0.90`。目标是降低 advantage 估计方差，判断更短的 GAE trace 是否能在保持 `gamma=0.99` 的同时让 PPO 更新更稳定，从而改善 exp051 跨 eval seed 稳定存在的少量 timeout。

## 配置

```text
configs/experiment/exp059_structured_bicycle_quintic_map25_gae090.yaml
```

相对 exp051 的唯一实质变量：

- `algorithm.gae_lambda: 0.95 -> 0.90`

保持不变：

- Actor 输出仍是单点 `[rho, beta]`；
- 不引入多点采样；
- reward、filter、control safety、reset、terrain、trajectory、episode/eval steps 全部保持 exp051；
- `gamma=0.99`、`learning_rate=1.0e-4`、`clip_epsilon=0.18`、`initial_log_std=-0.95`、`entropy_schedule_timesteps=12288` 保持 exp051。

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
  --config configs/experiment/exp059_structured_bicycle_quintic_map25_gae090.yaml \
  --device cpu \
  --timesteps 8 \
  --seed 23 \
  --num-envs 8 \
  --output-layout run \
  --run-name smoke_cpu_seed23_structured_bicycle_quintic_map25_gae090 \
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
  --config configs/experiment/exp059_structured_bicycle_quintic_map25_gae090.yaml \
  --device cuda \
  --timesteps 128 \
  --seed 23 \
  --num-envs 256 \
  --output-layout run \
  --run-name smoke_cuda_seed23_structured_bicycle_quintic_map25_gae090 \
  --rollout-steps 64 \
  --checkpoint-interval 64 \
  --eval-num-envs 256 \
  --eval-steps 64 \
  --bc-updates 0 \
  --selection-gate success_progress_long
```

## 长训命令

```bash
mkdir -p outputs/runs/exp059_structured_bicycle_quintic_map25_gae090/_launcher

systemd-run --user --unit exp059-structured-bicycle-quintic-map25-gae090-40m \
  --same-dir \
  --collect \
  --property=StandardOutput=append:${PWD}/outputs/runs/exp059_structured_bicycle_quintic_map25_gae090/_launcher/train.log \
  --property=StandardError=append:${PWD}/outputs/runs/exp059_structured_bicycle_quintic_map25_gae090/_launcher/train.log \
  .venv_isaaclab/bin/python scripts/train_skrl_mappo.py \
    --config configs/experiment/exp059_structured_bicycle_quintic_map25_gae090.yaml \
    --device cuda \
    --timesteps 20480 \
    --seed 23 \
    --num-envs 2048 \
    --output-layout run \
    --run-name pure_rl_seed23_40m_structured_bicycle_quintic_map25_gae090 \
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
| 23 | `smoke_cpu_seed23_structured_bicycle_quintic_map25_gae090` | smoke only | 工程通过；`8 env / 8 timesteps`，`gae_lambda=0.90`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0840`、action std `0.0660` | 非收敛验证 |
| 23 | `smoke_cuda_seed23_structured_bicycle_quintic_map25_gae090` | smoke only | 工程通过；`256 env / 128 timesteps / rollout 64`，`gae_lambda=0.90`，一个 optimizer、两次 joint update、terrain 权重更新 `0.0647`、action std `0.0810` | 非收敛验证 |
| 23 | `pure_rl_seed23_40m_structured_bicycle_quintic_map25_gae090` | `ppo_timestep_012288.pt` / `best.pt` | dmax `0.1927`、success `0.6904`、collision `0.0127`、timeout `0.2988`；filter applied `0.4463`、filter collision override `0.3117`、control safety `0.1456` | 未通过；success 与 timeout 失败 |

## 判读重点

- 若 timeout 低于 exp051 的 `0.0098` 且 success/collision 保持达标，说明降低 GAE trace 方差有助于末端稳定。
- 若 success 下降或 timeout 明显升高，说明当前任务仍需要 `gae_lambda=0.95` 的较长 advantage trace。
- 若 filter/control 介入比例明显升高，需要判为不理想。
- 不应把训练 reward 或 GIF 写成 strict 结论。

## 产物路径

已生成：

```text
outputs/runs/exp059_structured_bicycle_quintic_map25_gae090/pure_rl_seed23_40m_structured_bicycle_quintic_map25_gae090/metrics/final_eval_proxy.json
outputs/runs/exp059_structured_bicycle_quintic_map25_gae090/pure_rl_seed23_40m_structured_bicycle_quintic_map25_gae090/metrics/strict_acceptance.json
outputs/runs/exp059_structured_bicycle_quintic_map25_gae090/pure_rl_seed23_40m_structured_bicycle_quintic_map25_gae090/figures/training_curves.png
outputs/runs/exp059_structured_bicycle_quintic_map25_gae090/pure_rl_seed23_40m_structured_bicycle_quintic_map25_gae090/figures/candidate_eval_curves.png
outputs/runs/exp059_structured_bicycle_quintic_map25_gae090/pure_rl_seed23_40m_structured_bicycle_quintic_map25_gae090/figures/exp051_exp058_exp059_value_horizon_comparison.png
```

## 结论

exp059 未改善 exp051 的尾部 timeout，且明显差于 exp051。最终 strict 只通过 dmax 与 collision，success `0.6904 < 0.90`、timeout `0.2988 > 0` 失败。对比 exp051 的 success `0.9883`、timeout `0.0098`，降低 GAE trace 没有带来末端稳定性收益，反而破坏了 terminal convergence。

训练过程也显示中后期策略质量坍缩：最终训练窗口 success 约 `0.0122`，20 个候选 checkpoint 中最好仍回落到 `012288` 附近。说明问题不是单纯评估噪声，也不是更短 advantage trace 能修复的方差问题；当前任务仍更依赖 exp051 的 `gae_lambda=0.95`。

filter/control 介入没有扩大，但 exp059 的 final eval filter applied `0.4463`、collision override `0.3117`、control safety `0.1456` 偏高，说明策略自身动作分布更频繁落到需要兜底的区域。该结果支持“暂时维持原单点动作输出和原 filter/control 语义，不转向多点采样/规划化 filter”的判断。

## 下一步

回到 exp051 作为当前最好候选；不要继续沿着更长 `gamma` 或更短 `gae_lambda` 调整价值估计 horizon。下一轮应优先看 critic/observation 对尾部失败的辨识、末端 hold 可学习信号，或更保守的 checkpoint 诊断，而不是增强 filter/control 或改成多点采样。
