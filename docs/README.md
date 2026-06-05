# Project Documentation

Start here when reading or modifying this project.

## Agent Reading Order

1. Read [current_status.md](current_status.md) for the current result, recommended checkpoints, and active blockers.
2. Read [experiments/README.md](experiments/README.md) before interpreting any training result.
3. Read the specific experiment note under `docs/experiments/` before using a checkpoint.
4. Read [architecture/proxy_training.md](architecture/proxy_training.md) before changing the proxy environment, reward, PPO loop, or terrain dynamics.
5. Read [runbooks/train_proxy.md](runbooks/train_proxy.md), [runbooks/evaluate_proxy.md](runbooks/evaluate_proxy.md), or [runbooks/visualize_results.md](runbooks/visualize_results.md) for commands.

Do not infer success from a GIF or a single training checkpoint. Strict acceptance is defined by `_suite/metrics/strict_acceptance.json` and independent evaluation metrics.

## Current Documents

- [current_status.md](current_status.md): current project state and next work.
- [roadmap.md](roadmap.md): near-term priorities.
- [architecture/proxy_training.md](architecture/proxy_training.md): terrain-aware proxy training architecture.
- [architecture/physx_validation.md](architecture/physx_validation.md): Isaac Sim / PhysX role.
- [experiments/README.md](experiments/README.md): experiment index and pass/fail table.
- [references/output_management.md](references/output_management.md): canonical output layout and naming rules.

## Historical Documents

Long progress logs are archived under [archive/](archive/). They are useful for provenance, but not for current status.

