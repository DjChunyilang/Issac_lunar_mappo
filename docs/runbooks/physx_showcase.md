# PhysX Showcase Runbook

PhysX is a validation and showcase layer. It is not in the main training loop.

## Headless Evaluation

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --terrain lunar_crater \
  --episodes 3 \
  --steps 100 \
  --run-dir outputs/runs/<experiment>/<run_id>
```

## Rendered Capture

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --terrain lunar_crater \
  --episodes 1 \
  --steps 100 \
  --render \
  --run-dir outputs/runs/<experiment>/<run_id>
```

Default outputs:

```text
physx/metrics/lunar_crater_headless.json
physx/metrics/lunar_crater_render.json
physx/figures/lunar_crater_render_scene.png
physx/videos/lunar_crater_render_rollout.gif
```

