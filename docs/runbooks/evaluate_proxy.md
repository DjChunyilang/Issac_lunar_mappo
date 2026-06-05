# Proxy 评估操作手册

## 标准独立评估

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

## 严格 Gate

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

写最终结论时，优先使用独立评估的 `final_eval_proxy.json`，不要只依赖训练内部 best metrics。

## 必须报告的地形指标

```text
terrain_height_range
mean_roughness
max_roughness
min_traversability
mean_terrain_speed_scale
```
