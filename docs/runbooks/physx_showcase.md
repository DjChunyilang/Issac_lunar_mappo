# PhysX Jackal 跟踪测试操作手册

PhysX 是 high-fidelity closed-loop validation 层，不进入当前主训练 loop。当前活跃轮式资产为 Clearpath Jackal，用于验证轮式资产、地形 mesh、控制命令和轨迹跟踪闭环；它仍不代表最终 lunar rover asset。

## 推荐入口

当 experiment YAML 的 `evaluation.high_fidelity_eval.trigger` 允许时，统一评估脚本会在 proxy gate 通过后触发 Jackal tracking summary：

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

## 直接运行 Jackal 跟踪测试

平地调优并运行直线、绕圆、正弦：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_jackal_tracking.py \
  --terrain flat \
  --profile all \
  --tune-flat \
  --steps 660 \
  --sim-steps-per-control 4 \
  --run-dir outputs/runs/physx_jackal_tracking/flat_tuned_final
```

强三维地形使用 exp009 strong lunar crater profile：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_jackal_tracking.py \
  --terrain strong_lunar_crater \
  --profile all \
  --controller-json outputs/runs/physx_jackal_tracking/flat_tuned_final/physx/metrics/flat_tuning_grid.json \
  --steps 660 \
  --sim-steps-per-control 4 \
  --run-dir outputs/runs/physx_jackal_tracking/strong_lunar_crater
```

短烟测可以把 `--steps` 降到 60-140；正式验收使用较长 horizon，因为绕圆和正弦需要足够路径完成率。

默认输出：

```text
physx/metrics/tracking_summary.json
physx/metrics/flat_tuning_grid.json
physx/metrics/<terrain>_<profile>_timeseries.csv
physx/figures/<terrain>_<profile>_tracking.png
run_manifest.json
```

## 报告口径

- 写作时称为“Jackal 在 PhysX 强地形中的轨迹跟踪测试结果”。
- 不要称为“Isaac Lab 物理训练结果”。
- 必须同时报告 `rmse_cross_track_m`、`max_cross_track_m`、`path_completion_ratio`、`max_tilt_deg` 和图/CSV/JSON 产物路径。
- 若强地形未通过阈值，应直接报告失败项；不要用 proxy 的 `dmax/success_rate` gate 替代 tracking gate。
