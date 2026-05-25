# Experiment Plan

1. Run unit tests.
2. Run `scripts/debug_env.py --steps 200`.
3. Run `scripts/debug_observation.py`.
4. Run `scripts/debug_reward.py`.
5. Run `scripts/train.py --config configs/experiment/exp_001_minimal.yaml --device cpu --timesteps 128`.
6. Keep later ablations disabled until the proxy task is stable.

`scripts/train.py` uses SKRL `MAPPO` by default. The compact local trainer remains available as
`scripts/train.py --backend smoke` for debugging.
