# Multi-Rover Gathering Isaac First-Stage Project

This repository implements the first-stage scaffold from the two design documents in this
directory. The current implementation uses a clearly marked proxy rover model and a torch
vectorized planar dynamics core so the observation, action, reward, termination, and short
training loop can be tested before a real rover USD/URDF articulation is available.

## Documentation

Start with:

```text
docs/README.md
docs/current_status.md
docs/experiments/README.md
```

Long historical progress logs are archived under `docs/archive/`. Generated training artifacts
live under `outputs/` and are ignored by git; use the Markdown experiment notes and suite JSON
summaries for long-term status.

## Environment

The planned local environment is `.venv_isaaclab` with Python 3.12. The Isaac stack target is:

- Isaac Sim 6.0.0
- Isaac Lab v3.0.0-beta
- PyTorch 2.10.0+cu128
- SKRL via Isaac Lab `rl[skrl]`

## First-Stage Commands

```bash
.venv_isaaclab/bin/python -m pip install -e source/lunar_rover_tasks
.venv_isaaclab/bin/python -m pytest
.venv_isaaclab/bin/python scripts/debug_env.py --steps 200
.venv_isaaclab/bin/python scripts/debug_observation.py
.venv_isaaclab/bin/python scripts/debug_reward.py
.venv_isaaclab/bin/python scripts/train.py --config configs/experiment/exp_001_minimal.yaml --device cpu --timesteps 128
```

`scripts/train.py` defaults to the real SKRL `MAPPO` backend. Use
`--backend smoke` only for the compact local trainer used during fast debugging.

## Task ID

The gymnasium task registration ID is:

```text
Isaac-MultiRover-Gathering-Direct-v0
```

Actor observations do not include oracle information. The initial geometric median is used as
the first-stage training oracle point for centralized value/reward shaping only.
