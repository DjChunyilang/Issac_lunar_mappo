# PhysX 高保真闭环评估操作手册

PhysX 是 checkpoint 级 high-fidelity closed-loop evaluation，不进入当前主训练 loop。当前资产为 Jetbot placeholder，不代表最终 lunar rover asset。

## 推荐入口

当 experiment YAML 的 `evaluation.high_fidelity_eval.trigger` 允许时，统一评估脚本会在 proxy gate 通过后触发 PhysX：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/<experiment>/<run_id>
```

渲染版：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/<experiment>/<run_id> \
  --render-physx
```

## 直接运行 PhysX 评估

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --terrain lunar_crater \
  --episodes 3 \
  --steps 100 \
  --run-dir outputs/runs/<experiment>/<run_id>
```

渲染录制：

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
metrics/checkpoint_status.json
```

## 报告口径

- 写作时称为“proxy checkpoint 在 PhysX / Jetbot 场景中的闭环评估结果”。
- 不要称为“Isaac Lab 物理训练结果”。
- 必须同时报告 success、collision、dmax、dispersion、tilt 和 physics throughput。
