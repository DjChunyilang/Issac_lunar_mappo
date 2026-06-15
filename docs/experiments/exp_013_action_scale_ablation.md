# exp013 action scale ablation

## 目的

在 exp012 发现 SKRL-MAPPO CUDA 训练存在动作饱和后，进一步做更小动作尺度消融，判断单纯降低 `rho_max` / `beta_max` 是否能降低 saturation、提升 success gate 触达率，并把 SKRL 诊断产物迁移到标准 `outputs/runs/<experiment_id>/<run_id>/` layout。

该实验不修改 actor/critic 结构、不修改 reward 公式、不修改 oracle，也不作为 strict pass 结果。

## 配置

- suite 入口：`scripts/run_exp013_action_scale_ablation_suite.sh`
- checkpoint 统一评估入口：`scripts/run_checkpoint_evaluation.py`
- proxy GIF 渲染入口：`scripts/render_skrl_proxy_rollout.py`
- suite 输出：`outputs/runs/exp013_action_scale_ablation/_suite/`
- seed：`7`
- num_envs：`32`
- device：`cuda`
- rollout_steps：`32`
- algorithm：`pure_rl`、shared actor、centralized critic、shared value、`learning_rate=0.0005`

配置文件：

```text
configs/experiment/exp013_action_scale_rho06_beta45.yaml
configs/experiment/exp013_action_scale_rho05_beta30.yaml
```

动作尺度：

```text
rho06_beta45: rho_max=0.6, beta_max=0.78539816339
rho05_beta30: rho_max=0.5, beta_max=0.5235987756
```

## 严格标准

默认 strict gate 仍为：

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

exp013 没有生成 `_suite/metrics/strict_acceptance.json`，因此本实验只能作为诊断结果。严格结论以 `suite_summary.json`、各 run 的 `final_eval_proxy.json`、`summary.json` 和 `checkpoint_status.json` 为准。

## 结果表

| run_id | timesteps | checkpoint | final_eval | 诊断结论 | 是否通过 |
| --- | ---: | --- | --- | --- | --- |
| `rho06_beta45_seed7_smoke_32` | 32 | `outputs/runs/exp013_action_scale_ablation/rho06_beta45_seed7_smoke_32/checkpoints/best.pt` | `outputs/runs/exp013_action_scale_ablation/rho06_beta45_seed7_smoke_32/metrics/final_eval_proxy.json` | smoke 只验证链路；success 为 0 | 未通过 |
| `rho05_beta30_seed7_smoke_32` | 32 | `outputs/runs/exp013_action_scale_ablation/rho05_beta30_seed7_smoke_32/checkpoints/best.pt` | `outputs/runs/exp013_action_scale_ablation/rho05_beta30_seed7_smoke_32/metrics/final_eval_proxy.json` | smoke 只验证链路；success 为 0 | 未通过 |
| `rho06_beta45_seed7_probe_20000` | 20000 | `outputs/runs/exp013_action_scale_ablation/rho06_beta45_seed7_probe_20000/checkpoints/best.pt` | `outputs/runs/exp013_action_scale_ablation/rho06_beta45_seed7_probe_20000/metrics/final_eval_proxy.json` | 本轮最佳短探针；distance 和 eval reward 最好，但 success 为 0 | 未通过 |
| `rho05_beta30_seed7_probe_20000` | 20000 | `outputs/runs/exp013_action_scale_ablation/rho05_beta30_seed7_probe_20000/checkpoints/best.pt` | `outputs/runs/exp013_action_scale_ablation/rho05_beta30_seed7_probe_20000/metrics/final_eval_proxy.json` | 更保守动作尺度收拢较慢，success 为 0 | 未通过 |
| `rho06_beta45_seed7_long_120000` | 120000 | `outputs/runs/exp013_action_scale_ablation/rho06_beta45_seed7_long_120000/checkpoints/best.pt` | `outputs/runs/exp013_action_scale_ablation/rho06_beta45_seed7_long_120000/metrics/final_eval_proxy.json` | 训练内出现少量 success_done，但 final eval success 仍为 0，动作饱和加重 | 未通过 |

核心测试：

```text
65 passed
```

suite 汇总：

```text
outputs/runs/exp013_action_scale_ablation/_suite/metrics/suite_summary.json
```

每个 run 的统一 checkpoint 状态：

```text
outputs/runs/exp013_action_scale_ablation/<run_id>/metrics/checkpoint_status.json
```

teacher reachability sanity：

```text
outputs/runs/exp013_action_scale_ablation/_suite/metrics/teacher_reachability_summary.json
```

## 失败分析

20k 主候选 `rho06_beta45_seed7_probe_20000`：

```text
judgement: clear_improvement
mean_pairwise_distance: 5.7457 -> 5.5086
mean_oracle_distance: 3.5662 -> 3.4172
success_rate.max: 0.0
timeout_done: 6400
collision_done: 1
safety_done: 1
action_saturation_fraction: 0.4186
final_eval_success_rate: 0.0
final_eval_mean_reward: 0.1141
final_eval_dmax: 5.2041
proxy_gif_pairwise: 5.8122 -> 3.7992
proxy_gif_oracle: 3.6066 -> 2.3733
proxy_gif_done_reason: timeout
```

该 run 是 exp013 中最有学习信号的短探针：proxy GIF 中队形明显收拢，final eval reward 最高，`final_eval_dmax` 也最低。但它仍然 timeout，success gate 没有被触达。

120k 主候选 `rho06_beta45_seed7_long_120000`：

```text
judgement: clear_improvement
mean_pairwise_distance: 5.7457 -> 5.6394, min=5.2840
mean_oracle_distance: 3.5662 -> 3.4963, min=3.2790
success_rate.max: 0.0
success_done: 13
timeout_done: 38359
collision_done: 39
safety_done: 39
action_saturation_fraction: 0.5082
final_eval_success_rate: 0.0
final_eval_mean_reward: 0.0457
final_eval_dmax: 7.0882
```

长预算没有稳定提高成功率，反而让动作饱和从 20k 的约 `0.42` 上升到约 `0.51`，final eval reward 和 dmax 均差于 20k probe。

保守候选 `rho05_beta30_seed7_probe_20000`：

```text
judgement: clear_improvement
mean_pairwise_distance: 5.7457 -> 5.5086
mean_oracle_distance: 3.5662 -> 3.4172
success_rate.max: 0.0
timeout_done: 6400
action_saturation_fraction: 0.4317
final_eval_success_rate: 0.0
final_eval_mean_reward: 0.0295
final_eval_dmax: 6.7573
```

更小动作尺度没有解决 success，也没有优于 `rho06_beta45` 的短探针。

## Teacher Reachability

训练失败后补跑 scripted teacher sanity check，用同一个 teacher policy 检查当前 episode 长度、动作尺度和 success gate 是否可达。该检查不训练网络，只跑确定性 teacher rollout。

| case | max steps | action scale | success_rate | timeout_rate | final_dmax | final_dispersion | 结论 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| exp013 `rho06_beta45` stop 0.45 | 100 | `rho=0.6, beta=pi/4` | 0.0000 | 1.0000 | 1.9998 | 0.9081 | teacher 失败 |
| exp013 `rho06_beta45` stop 0.35 | 100 | `rho=0.6, beta=pi/4` | 0.0059 | 0.9941 | 1.9585 | 0.8692 | teacher 近乎失败 |
| exp013 `rho05_beta30` stop 0.45 | 100 | `rho=0.5, beta=pi/6` | 0.0000 | 1.0000 | 2.7911 | 1.7565 | teacher 失败 |
| exp013 `rho06_beta45` stop 0.35 | 220 | `rho=0.6, beta=pi/4` | 1.0000 | 0.0000 | 1.0416 | 0.2428 | teacher 成功 |
| exp013 full scale stop 0.45 | 100 | `rho=1.2, beta=pi/2` | 0.7188 | 0.3262 | 1.0633 | 0.2741 | 大幅改善但未满 |
| exp008 weak warm-start teacher | 220 | `rho=1.2, beta=pi/2` | 1.0000 | 0.0000 | 1.0874 | 0.2772 | teacher 成功 |

该结果说明当前 exp013 的主要阻塞不是地形，也不是 success gate 本身写错，而是 `rho_max=0.6 / beta_max=pi/4 / 100 steps` 组合对 scripted teacher 都几乎不可达。小动作尺度下恢复到 `220` steps 后 teacher 可以 `100%` 成功；保持 `100` steps 但恢复全动作尺度也能把 teacher success 提升到 `0.7188`。

因此，exp013 的 SKRL policy 失败不能只归因于 pure RL 学习能力。当前配置把任务做得过紧：动作太小、episode 太短，policy 在可达性边界附近学习，难以稳定触发 `hold_steps=8` 的 success。

## 产物路径

```text
outputs/runs/exp013_action_scale_ablation/_suite/metrics/suite_summary.json
outputs/runs/exp013_action_scale_ablation/_suite/metrics/teacher_reachability_summary.json
outputs/runs/exp013_action_scale_ablation/_suite/run_manifest.json
outputs/runs/exp013_action_scale_ablation/_suite/logs/
outputs/runs/exp013_action_scale_ablation/_suite/checkpoints/
outputs/runs/exp013_action_scale_ablation/rho06_beta45_seed7_probe_20000/
outputs/runs/exp013_action_scale_ablation/rho06_beta45_seed7_long_120000/
outputs/runs/exp013_action_scale_ablation/rho05_beta30_seed7_probe_20000/
```

每个 run 目录包含：

```text
config/experiment.yaml
checkpoints/best.pt
metrics/train_metrics.jsonl
metrics/summary.json
metrics/diagnosis.json
metrics/final_eval_proxy.json
metrics/proxy_rollout_render.json
videos/proxy_eval_rollout.gif
run_manifest.json
```

这些产物位于 ignored 的 `outputs/`，不要提交原始 JSON、日志、GIF 或 checkpoint。

## 结论

exp013 不能作为当前主结果，也不支持继续直接长训。它确认了：

- CUDA SKRL-MAPPO 链路和标准 run layout 可用。
- 更小动作尺度能在短探针中产生距离收拢信号，但不能稳定触达 success gate。
- 120k 长预算没有比 20k probe 更好，动作饱和重新加重。
- 单纯调小 `rho_max` / `beta_max` 不足以解决 success/timeout 问题。
- teacher reachability 显示当前小动作尺度 + 100-step episode 本身几乎不可达；下一轮需要先恢复可达性，再讨论 SKRL 学习稳定性。

当前最有诊断价值的 checkpoint 是：

```text
outputs/runs/exp013_action_scale_ablation/rho06_beta45_seed7_probe_20000/checkpoints/best.pt
```

它适合用于 success gate reachability、动作分布、episode 末端状态分析，不适合作为 long training 起点。

## 下一步

1. 做 success gate reachability 诊断：比较 timeout episode 末端的 `dmax`、`dispersion`、`speed`、`success_hold_count` 与成功阈值距离。
2. 做动作饱和机制诊断：区分 normalized policy 输出饱和、物理 `rho/beta` 饱和和 controller 后速度/角速度饱和。
3. 设计 exp014 teacher-reachable 配置，优先保留 `rho=0.6, beta=pi/4` 但把 `episode_length_s` 恢复到 `44.0`，因为 teacher 在该设置下已经 `100%` 成功。
4. 若必须保持 `100` steps，则至少恢复更大的动作尺度，并重新评估 teacher success；当前 full-scale 100-step teacher success 为 `0.7188`，仍未达到严格门槛。
5. 在 teacher 可达配置上，再评估 PPO 超参或 action distribution 约束，例如 entropy、log_std clamp、action penalty/regularization 的诊断版本。
6. 不要继续把当前 `rho06_beta45` 或 `rho05_beta30` 的 100-step 配置直接加长训练步数。
