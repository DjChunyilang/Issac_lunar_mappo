# exp_008 Terrain3D

## Purpose

Upgrade the proxy environment from planar motion to terrain-aware 3D simplified dynamics and verify strict convergence on weak lunar crater terrain.

## Configuration

```text
configs/experiment/exp_008_terrain3d_pure_rl.yaml
configs/experiment/exp_008_terrain3d_weak_warmstart.yaml
configs/experiment/exp_008_terrain3d_weak_warmstart_retry.yaml
configs/experiment/exp_008_terrain3d_weak_warmstart_select.yaml
```

Terrain sanity:

```text
height_range ~= 0.241 m
roughness_max ~= 0.360
traversability_min ~= 0.549
```

## Result

Pure RL did not converge in the tested budgets. Weak warm-start + PPO passed strict acceptance for seeds `23, 31, 47`.

| seed | final run | dmax_ratio | success | collision | timeout |
| --- | --- | ---: | ---: | ---: | ---: |
| 23 | `weak_warmstart_seed23_8m_lunar_crater_cpu` | 0.1539 | 1.0000 | 0.0000 | 0.0000 |
| 31 | `weak_warmstart_completion_seed31_4m_evalseed0_cpu` | 0.1345 | 0.9961 | 0.0049 | 0.0000 |
| 47 | `weak_warmstart_select_seed47_8m_lunar_crater_cpu` | 0.1560 | 1.0000 | 0.0000 | 0.0000 |

Suite outputs:

```text
outputs/runs/exp_008_terrain3d/_suite/metrics/strict_acceptance.json
outputs/runs/exp_008_terrain3d/_suite/metrics/suite_summary.json
outputs/runs/exp_008_terrain3d/_suite/figures/comparison_curves.png
outputs/runs/exp_008_terrain3d/_suite/checkpoints/
```

## Notes

This is the current best full 3-seed terrain-aware proxy result.

