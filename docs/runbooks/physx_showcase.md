# PhysX 展示操作手册

PhysX 是验证和展示层，不进入主训练 loop。

## 无头评估

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --terrain lunar_crater \
  --episodes 3 \
  --steps 100 \
  --run-dir outputs/runs/<experiment>/<run_id>
```

## 渲染录制

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

默认输出：

```text
physx/metrics/lunar_crater_headless.json
physx/metrics/lunar_crater_render.json
physx/figures/lunar_crater_render_scene.png
physx/videos/lunar_crater_render_rollout.gif
```
