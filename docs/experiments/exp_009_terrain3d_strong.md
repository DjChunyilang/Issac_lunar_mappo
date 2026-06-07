# exp_009 强三维地形

## 目的

将地形高度范围提高到 `0.6-1.0 m`，测试弱 warm-start + PPO 流程是否仍能通过严格验收。

## 配置

```text
configs/experiment/exp_009_terrain3d_strong_weak_warmstart.yaml
```

地形 profile：

```text
height_range ~= 0.740 m
roughness_max ~= 1.057
traversability_min ~= 0.096
mean_terrain_speed_scale ~= 0.393
```

## 结果

exp009 未通过严格验收。

| seed | selected run | dmax_ratio | success | collision | timeout | 结果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 23 | `weak_warmstart_seed23_timeout_retry6m_strong_lunar_crater_cpu_nenv1024_eval1024` | 0.1473 | 1.0000 | 0.0000 | 0.0000 | 通过 |
| 31 | `weak_warmstart_seed31_retry20m_safe090_strong_lunar_crater_cuda_eval1024` | 0.1819 | 0.8740 | 0.0049 | 0.1250 | 未通过 |

seed31 失败后没有继续运行 seed47，因为 3-seed strict acceptance 已经不可能成立。

Suite 输出：

```text
outputs/runs/exp_009_terrain3d_strong/_suite/metrics/strict_acceptance.json
outputs/runs/exp_009_terrain3d_strong/_suite/metrics/suite_summary.json
outputs/runs/exp_009_terrain3d_strong/_suite/figures/comparison_curves.png
outputs/runs/exp_009_terrain3d_strong/_suite/figures/terrain_height_map.png
```

## 失败分析

seed31 满足 dmax 和 collision gate，但未满足 success 和 timeout：

```text
dmax_reduction_ratio: 0.1819
success_rate: 0.8740
collision_rate: 0.0049
timeout_rate: 0.1250
```

固定 retry 未解决问题：

1. 20M safety retry 改善了 dmax 和 collision，但没有解决 success/timeout。
2. 6M completion retry 降低了 dmax，但提高了 collision。
3. 6M conservative retry 没有通过 strict，也没有改善最终选中结果。

## 下一步

不要在同一设置上继续无界 PPO。该诊断线近期暂缓，当前项目重心转为 Isaac Sim / Isaac Lab / SKRL / 本地任务包的环境搭建与工程闭环验收。

后续恢复 strong terrain 研究时，再基于 seed31 失败结果重新设计动作表示、控制接口或 terrain curriculum。
