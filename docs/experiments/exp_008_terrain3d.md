# exp_008 三维地形

## 目的

将 proxy 环境从平面运动升级为 terrain-aware 3D 简化动力学，并在弱 lunar crater 地形上验证严格收敛。

## 配置

```text
configs/experiment/exp_008_terrain3d_pure_rl.yaml
configs/experiment/exp_008_terrain3d_weak_warmstart.yaml
configs/experiment/exp_008_terrain3d_weak_warmstart_retry.yaml
configs/experiment/exp_008_terrain3d_weak_warmstart_select.yaml
```

地形 sanity：

```text
height_range ~= 0.241 m
roughness_max ~= 0.360
traversability_min ~= 0.549
```

## 结果

Pure RL 在测试预算内没有收敛。弱 warm-start + PPO 在 seeds `23, 31, 47` 上通过严格验收。

| seed | final run | dmax_ratio | success | collision | timeout |
| --- | --- | ---: | ---: | ---: | ---: |
| 23 | `weak_warmstart_seed23_8m_lunar_crater_cpu` | 0.1539 | 1.0000 | 0.0000 | 0.0000 |
| 31 | `weak_warmstart_completion_seed31_4m_evalseed0_cpu` | 0.1345 | 0.9961 | 0.0049 | 0.0000 |
| 47 | `weak_warmstart_select_seed47_8m_lunar_crater_cpu` | 0.1560 | 1.0000 | 0.0000 | 0.0000 |

Suite 输出：

```text
outputs/runs/exp_008_terrain3d/_suite/metrics/strict_acceptance.json
outputs/runs/exp_008_terrain3d/_suite/metrics/suite_summary.json
outputs/runs/exp_008_terrain3d/_suite/figures/comparison_curves.png
outputs/runs/exp_008_terrain3d/_suite/checkpoints/
```

## 说明

这是当前最佳的完整 3-seed terrain-aware proxy 结果，不是 Isaac / PhysX 物理训练结果。候选 checkpoint 后续应通过 `scripts/run_checkpoint_evaluation.py` 补齐 `metrics/checkpoint_status.json`，再报告是否通过高保真闭环评估。
