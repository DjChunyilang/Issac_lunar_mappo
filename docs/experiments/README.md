# Experiment Index

Strict gate unless stated otherwise:

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

| Experiment | Terrain | Method | Seeds | Strict | Canonical doc |
| --- | --- | --- | --- | --- | --- |
| exp006 | flat proxy | BC + PPO, PPO-selected checkpoint | 23, 31, 47 | pass | [exp_006_ppo_selected.md](exp_006_ppo_selected.md) |
| exp007 | lunar crater showcase / Phase C | weak warm-start + PPO | selected run | pass for selected checkpoint | [exp_007_phase_c.md](exp_007_phase_c.md) |
| exp008 | weak lunar crater 3D proxy | weak warm-start + PPO | 23, 31, 47 | pass | [exp_008_terrain3d.md](exp_008_terrain3d.md) |
| exp009 | strong lunar crater 3D proxy | weak warm-start + PPO | 23, 31; 47 not run | fail | [exp_009_terrain3d_strong.md](exp_009_terrain3d_strong.md) |

For any new experiment, add one row here and create a dedicated `exp_###_*.md` file. Keep date-based progress logs in `docs/archive/`.

