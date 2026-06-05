# Current Status

## Current Mainline

- Training loop: PyTorch terrain-aware proxy environment.
- Rendering / high-fidelity sanity check: Isaac Sim / PhysX Jetbot evaluation.
- Visual observations are not part of policy input. Terrain enters the policy as low-dimensional structured features.
- Generated results live under `outputs/runs/` and are ignored by git.

## Best Validated Results

| Experiment | Terrain | Method | Strict status | Notes |
| --- | --- | --- | --- | --- |
| exp006 | flat proxy | BC + PPO | pass | PPO-selected flat baseline. |
| exp008 | weak lunar crater 3D proxy | weak warm-start + PPO | pass, 3 seeds | Current best full 3-seed terrain-aware proxy result. |
| exp009 | strong lunar crater 3D proxy | weak warm-start + PPO | fail | seed23 passed; seed31 failed success/timeout; seed47 not run. |

Current best full-suite checkpoint family:

```text
outputs/runs/exp_008_terrain3d/_suite/checkpoints/
```

Current strong-terrain diagnostic checkpoints:

```text
outputs/runs/exp_009_terrain3d_strong/_suite/checkpoints/seed_23_best.pt
outputs/runs/exp_009_terrain3d_strong/_suite/checkpoints/seed_31_best.pt
```

## Active Blocker

exp009 strong terrain proves the 3D terrain dynamics are active, with height range about `0.74 m`, but the current high-level action and reward/control design do not robustly clear strict gates across seeds.

seed31 failure mode:

```text
dmax_reduction_ratio: 0.1819  # pass
success_rate: 0.8740          # fail
collision_rate: 0.0049        # pass
timeout_rate: 0.1250          # fail
```

## Next Work

Do not keep adding unbounded PPO steps as the first response. Prioritize:

1. Improve behavior near the success region so speed and hold conditions become stable.
2. Add or tune reward terms for timely completion while preserving inter-rover separation.
3. Review whether `[rho, beta]` one-step subgoal actions are too weak for strong terrain.
4. Use failed seed31 rollouts to inspect whether timeout comes from terrain speed scaling, dispersion instability, or speed hold failure.

