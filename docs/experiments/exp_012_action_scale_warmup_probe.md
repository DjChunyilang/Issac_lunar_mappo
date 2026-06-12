# exp012 action scale warmup probe

## 目的

诊断 SKRL-MAPPO CUDA 训练信号是否受动作尺度、动作饱和、reward component 比例或 success gate 可达性影响。该实验用于把 `scripts/train_skrl_mappo.py` 从 smoke 入口推进到可观测的 CUDA 诊断入口，不作为当前主 strict 结果。

## 配置

- 配置文件：`configs/experiment/exp012_action_scale_warmup_probe.yaml`
- 训练入口：`scripts/train_skrl_mappo.py`
- suite 入口：`scripts/run_exp012_action_scale_suite.sh`
- seed：`7`
- num_envs：`32`
- device：`cuda`
- total_steps：`500000`
- rollout_steps：`32`
- action scale：`rho_max=0.8`、`beta_max=1.0471975512`
- algorithm：`pure_rl`、shared actor、centralized critic、shared value、`learning_rate=0.0005`

## 严格标准

默认 strict gate 仍为：

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

当前 exp012 的机器可读产物是 `metrics.jsonl` 和 `diagnosis_*.json`，不是 `_suite/metrics/strict_acceptance.json`。因此不能按 strict pass 报告成功。

## 结果表

| seed | run_id | timesteps | checkpoint | 诊断文件 | 是否通过 |
| --- | --- | --- | --- | --- | --- |
| 7 | `exp012_action_scale_warmup_probe_1781236923` | 32 | `outputs/checkpoints/exp012_action_scale_warmup_probe_smoke_32.pt` | `outputs/runs/exp012_action_scale_warmup_probe/diagnosis_smoke_32.json` | 未通过；smoke 只验证链路 |
| 7 | `exp012_action_scale_warmup_probe_1781236929` | 20000 | `outputs/checkpoints/exp012_action_scale_warmup_probe_probe_20000.pt` | `outputs/runs/exp012_action_scale_warmup_probe/diagnosis_probe_20000.json` | 未通过；distance 有改善但 success 为 0，动作饱和明显 |

`run_exp012_action_scale_suite.sh` 还包含 `500000` timesteps 长预算。只有生成对应完整 `diagnosis_long_5h_500000.json` 后，才更新本表。

## 失败分析

32-step smoke：

```text
judgement: no_clear_improvement
success_rate.max: 0.0
action_scale_summary.flags: no_obvious_action_scale_issue
next_experiment_focus: success_gate_reachability_diagnostic
```

20k probe：

```text
judgement: clear_improvement
mean_pairwise_distance: 5.7457 -> 5.5086
mean_oracle_distance: 3.5662 -> 3.4172
success_rate.max: 0.0
timeout_done: 6400
action_saturation_fraction: 0.5142
physical_beta_abs_high_fraction: 0.5139
next_experiment_focus: action_scale_ablation, success_gate_reachability_diagnostic
```

结论是训练信号有弱距离改善，但没有触达 success gate；20k 阶段同时出现 normalized action、forward、turn、physical rho 和 physical beta 饱和。下一轮不应只增加 PPO 步数。

## 产物路径

```text
outputs/runs/exp012_action_scale_warmup_probe/metrics.jsonl
outputs/runs/exp012_action_scale_warmup_probe/diagnosis_smoke_32.json
outputs/runs/exp012_action_scale_warmup_probe/diagnosis_probe_20000.json
outputs/runs/exp012_action_scale_warmup_probe/suite_logs/
outputs/checkpoints/exp012_action_scale_warmup_probe_smoke_32.pt
outputs/checkpoints/exp012_action_scale_warmup_probe_probe_20000.pt
```

这些产物位于 ignored 的 `outputs/`，不要提交原始 JSON、日志或 checkpoint。

## 结论

exp012 目前不能作为当前主结果。它的价值是确认 CUDA SKRL-MAPPO 训练链路、checkpoint metadata、NaN 检查和 action/reward telemetry 可用，并暴露动作尺度与 success gate 可达性的下一步诊断方向。

## 下一步

1. 做 action scale 消融，优先比较 `rho_max`、`beta_max`、policy output std / log_std 和 action clipping 对 saturation 的影响。
2. 做 success gate reachability 诊断，确认当前 deterministic eval 是否能接近 `dmax`、`dispersion`、`speed` 和 `hold_steps` 条件。
3. 将 SKRL-MAPPO 诊断输出继续迁移到标准 `outputs/runs/<experiment_id>/<run_id>/metrics/` 和 `_suite/metrics/` layout。
