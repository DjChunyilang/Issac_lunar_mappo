# exp_006 PPO-Selected Baseline

## Purpose

Establish a flat/proxy strict baseline where the final best checkpoint must come from PPO, not from warm-up.

## Configuration

```text
configs/experiment/exp_006_ppo_selected_bc_ppo.yaml
```

Key settings:

```text
num_envs: 1024
rollout_steps: 128
total_env_steps: 2,000,000
eval_num_envs: 512
eval_steps: 160
bc_steps: 300
learning_rate: 5e-5
clip_epsilon: 0.15
ppo_epochs: 2
best_source: ppo
required_best_phase: ppo
```

## Result

The BC+PPO group passed strict acceptance for seeds `23, 31, 47`. The best seed was seed 23 update 15:

```text
dmax_reduction_ratio: 0.1438
success_rate: 1.0
collision_rate: 0.0
timeout_rate: 0.0
```

## Notes

This result is a PPO-selected result after warm-start. It is not pure RL from scratch.

