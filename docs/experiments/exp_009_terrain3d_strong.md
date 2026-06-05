# exp_009 Strong Terrain3D

## Purpose

Increase terrain height range to `0.6-1.0 m` and test whether the weak warm-start + PPO workflow still passes strict acceptance.

## Configuration

```text
configs/experiment/exp_009_terrain3d_strong_weak_warmstart.yaml
```

Terrain profile:

```text
height_range ~= 0.740 m
roughness_max ~= 1.057
traversability_min ~= 0.096
mean_terrain_speed_scale ~= 0.393
```

## Result

exp009 did not pass strict acceptance.

| seed | selected run | dmax_ratio | success | collision | timeout | result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 23 | `weak_warmstart_seed23_timeout_retry6m_strong_lunar_crater_cpu_nenv1024_eval1024` | 0.1473 | 1.0000 | 0.0000 | 0.0000 | pass |
| 31 | `weak_warmstart_seed31_retry20m_safe090_strong_lunar_crater_cuda_eval1024` | 0.1819 | 0.8740 | 0.0049 | 0.1250 | fail |

seed 47 was not run after seed31 failed, because 3-seed strict acceptance was already impossible.

Suite outputs:

```text
outputs/runs/exp_009_terrain3d_strong/_suite/metrics/strict_acceptance.json
outputs/runs/exp_009_terrain3d_strong/_suite/metrics/suite_summary.json
outputs/runs/exp_009_terrain3d_strong/_suite/figures/comparison_curves.png
outputs/runs/exp_009_terrain3d_strong/_suite/figures/terrain_height_map.png
```

## Failure Analysis

seed31 meets the dmax and collision gates, but fails success and timeout:

```text
dmax_reduction_ratio: 0.1819
success_rate: 0.8740
collision_rate: 0.0049
timeout_rate: 0.1250
```

Fixed retries did not solve it:

1. 20M safety retry improved dmax and collision but not success/timeout.
2. 6M completion retry reduced dmax but increased collision.
3. 6M conservative retry did not pass strict and did not improve the selected result.

## Next Step

Do not continue unbounded PPO on the same setup. Diagnose failed seed31 episodes and change task/control design before another long run.

