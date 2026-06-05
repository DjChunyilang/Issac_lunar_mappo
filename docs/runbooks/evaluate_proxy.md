# Proxy Evaluation Runbook

## Standard Independent Evaluation

```bash
.venv_isaaclab/bin/python scripts/evaluate_proxy_policy.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --num-envs 1024 \
  --steps 260 \
  --seed <seed> \
  --run-dir outputs/runs/<experiment>/<run_id> \
  --output outputs/runs/<experiment>/<run_id>/metrics/final_eval_proxy.json
```

## Strict Gates

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

Prefer independent `final_eval_proxy.json` over training-internal best metrics when writing final conclusions.

## Terrain Metrics to Report

```text
terrain_height_range
mean_roughness
max_roughness
min_traversability
mean_terrain_speed_scale
```

