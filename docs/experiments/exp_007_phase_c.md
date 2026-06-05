# exp_007 Phase C

## Purpose

Add lunar crater proxy terrain, PhysX lunar crater mesh, four Jetbot evaluation, and a weak warm-start checkpoint whose best checkpoint comes from PPO.

## Configuration

```text
configs/experiment/exp_007_phase_c_weak_warmstart.yaml
configs/experiment/exp_007_phase_c_pure_rl.yaml
```

Representative final checkpoint:

```text
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/checkpoints/best.pt
```

## Training Result

```text
phase: ppo
update: 7
dmax_reduction_ratio: 0.1430
success_rate: 1.0000
collision_rate: 0.0000
timeout_rate: 0.0000
```

Independent proxy evaluation:

```text
dmax_reduction_ratio: 0.1401
success_rate: 1.0000
collision_rate: 0.0000
timeout_rate: 0.0000
mean_done_step: 83.53
```

## PhysX Result

Four Jetbot lunar crater evaluation passed as a high-fidelity sanity check:

```text
success_rate: 1.0000
collision_rate: 0.0000
mean_final_dmax: 0.7977
```

## Notes

PhysX is used for validation and showcase only. It is not in the main training loop.

