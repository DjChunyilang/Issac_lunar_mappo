# Proxy Training Runbook

## Standard Command

```bash
.venv_isaaclab/bin/python scripts/train_proxy_convergence.py \
  --config configs/experiment/<config>.yaml \
  --output-layout run \
  --run-name <run_id> \
  --device cuda
```

Use CPU only for smoke tests or when CUDA is unavailable.

## Smoke Test

```bash
.venv_isaaclab/bin/python scripts/train_proxy_convergence.py \
  --config configs/experiment/exp_009_terrain3d_strong_weak_warmstart.yaml \
  --output-layout run \
  --run-name smoke_seed23_16k_strong_lunar_crater_cpu \
  --device cpu \
  --seed 23 \
  --total-env-steps 16384 \
  --eval-num-envs 64 \
  --eval-steps 60
```

## Continuation

Use continuation only with an explicit run name and no warm-up:

```bash
.venv_isaaclab/bin/python scripts/train_proxy_convergence.py \
  --config configs/experiment/exp_009_terrain3d_strong_weak_warmstart.yaml \
  --output-layout run \
  --run-name <retry_run_id> \
  --device cuda \
  --seed <seed> \
  --resume-checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --bc-steps 0 \
  --total-env-steps 6000000 \
  --learning-rate 0.00004 \
  --eval-num-envs 1024 \
  --eval-steps 260
```

## Rules

- Always set `--run-name`.
- Do not overwrite prior failed runs; failures are diagnostics.
- Do not report success unless independent evaluation also passes strict gates.
- For strong terrain, avoid changing terrain strength to make training pass unless the experiment explicitly says so.

