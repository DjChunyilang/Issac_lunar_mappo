# Proxy 评估操作手册

Proxy eval 是当前 checkpoint selection 的高频数值评估。它不启动 Isaac Sim / PhysX。

## 推荐入口

优先使用统一 checkpoint 评估脚本：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/<experiment>/<run_id> \
  --skip-physx
```

默认输出：

```text
metrics/final_eval_proxy.json
metrics/checkpoint_status.json
run_manifest.json
```

`--skip-physx` 只跳过高保真评估，不跳过 proxy gate。

## 直接运行 proxy eval

需要单独复评时仍可直接调用底层脚本：

```bash
.venv_isaaclab/bin/python scripts/evaluate_proxy_policy.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --num-envs 1024 \
  --steps 220 \
  --seed <seed> \
  --run-dir outputs/runs/<experiment>/<run_id>
```

带 `--run-dir` 时默认写入：

```text
outputs/runs/<experiment>/<run_id>/metrics/final_eval_proxy.json
```

如需仅用于诊断地复现某一 subgoal-filter 课程阶段，可额外传 `--filter-progress-override <nonnegative-step>`。该参数会被写入结果的 `filter_progress_override` 与 `filter_progress_timestep`，只覆盖本次评测，不能把这种后验结果表述为已经用相同调度训练出的 policy；正式结论仍使用 checkpoint metadata 的训练进度。

## 严格 Gate

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

写最终结论时，优先使用 `_suite/metrics/strict_acceptance.json`、独立 `final_eval_proxy.json` 和 `checkpoint_status.json`，不要只依赖训练内部 best metrics。

## 必须报告的地形指标

```text
terrain_height_range
mean_roughness
max_roughness
min_traversability
mean_terrain_speed_scale
```
