# 进度摘要 - 2026-06-05

> 归档进度日志。当前状态请阅读 `docs/current_status.md` 和 `docs/experiments/README.md`。

## exp_009 强地形复验结果

本阶段按“强地形版本”重新训练：地形高度范围目标提升到 `0.6-1.0 m`，并提高 crater 深度、正弦起伏、坡度减速和 terrain/slope reward 成本。

配置入口：

```text
configs/experiment/exp_009_terrain3d_strong_weak_warmstart.yaml
```

地形 sanity check 通过：

```text
height_range ~= 0.740 m
roughness_max ~= 1.057
traversability_min ~= 0.096
mean_terrain_speed_scale ~= 0.393
```

严格 gate 保持为：

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

## 训练与复验

seed 23 先运行 12M 基础训练，内部评估通过但 1024 env 独立复验仍有 1 个 timeout；随后执行 6M timeout-only continuation 后严格通过。

seed 31 的原始 12M run 中途未正常写出 summary；随后按固定 retry 规则执行：

1. 20M safety retry：`near_distance=0.90`、`near_penalty=5.0`，不降低 terrain 强度。
2. 6M completion retry：从 20M best 继续，`bc_steps=0`、`learning_rate=4e-5`。
3. 6M conservative retry：从 20M best 继续，`learning_rate=2e-5`、关闭 scripted teacher PPO loss。

最终最稳的 seed 31 checkpoint 仍来自 20M safety retry；它满足 `dmax_ratio` 和 collision gate，但未满足 `success_rate` 和 `timeout_rate` gate。因此 exp009 不报告严格收敛，seed 47 不再继续运行，因为 3-seed strict acceptance 已经不可能成立。

| seed | selected run | dmax_ratio | success | collision | timeout | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 23 | `weak_warmstart_seed23_timeout_retry6m_strong_lunar_crater_cpu_nenv1024_eval1024` | 0.1473 | 1.0000 | 0.0000 | 0.0000 | 通过 |
| 31 | `weak_warmstart_seed31_retry20m_safe090_strong_lunar_crater_cuda_eval1024` | 0.1819 | 0.8740 | 0.0049 | 0.1250 | 未通过 |

## 输出产物

Suite 汇总：

```text
outputs/runs/exp_009_terrain3d_strong/_suite/metrics/strict_acceptance.json
outputs/runs/exp_009_terrain3d_strong/_suite/metrics/suite_summary.json
outputs/runs/exp_009_terrain3d_strong/_suite/figures/comparison_curves.png
outputs/runs/exp_009_terrain3d_strong/_suite/figures/terrain_height_map.png
```

代表性 run：

```text
outputs/runs/exp_009_terrain3d_strong/weak_warmstart_seed23_timeout_retry6m_strong_lunar_crater_cpu_nenv1024_eval1024/
outputs/runs/exp_009_terrain3d_strong/weak_warmstart_seed31_retry20m_safe090_strong_lunar_crater_cuda_eval1024/
```

每个 run 均包含：

```text
metrics/summary.json
metrics/final_eval_proxy.json
figures/convergence_curves.png
figures/safety_diagnostics.png
figures/terrain_height_map.png
videos/proxy_eval_rollout.gif
tensorboard/
```

其中 `terrain_height_map.png` 和 `proxy_eval_rollout.gif` 均带高度热力图和 `height (m)` 图例。

## 当前结论

exp009 证明强地形 proxy 确实生效，且 seed 23 可以在高度范围约 `0.74 m` 的 lunar crater 地形下达到严格 gate。但在相同 terrain 强度和弱 warm-start 限制下，seed 31 经过固定 retry 后仍未达到 3-seed 严格收敛。

下一步应优先改进任务/控制设计，而不是继续无界加训练步数：

1. 重新设计成功 hold 条件附近的减速控制，避免接近集合区后速度条件迟迟不满足。
2. 增加“及时完成但保持安全距离”的 reward 项，减少 `success_rate` 和 collision 之间的拉扯。
3. 考虑把高层动作从单步 `[rho, beta]` 扩展为更稳定的短时轨迹参数。
4. 对 seed31 的失败 episode 做定点回放，检查是地形低 traversability 导致超时，还是队形压缩时 dispersion/speed 条件不稳定。

## 验证

已完成回归：

```text
.venv_isaaclab/bin/python -m pytest
34 passed
```
