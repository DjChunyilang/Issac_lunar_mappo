# 实验计划

## 当前主线

当前实验主线是“proxy 训练 + checkpoint 级 high-fidelity closed-loop evaluation”：

- `exp008` 保持为当前已验证的 3-seed terrain-aware proxy baseline。
- `exp009` / `exp010` 作为 strong terrain 诊断记录，近期不继续默认扩展。
- `exp012` / `exp013` 作为 SKRL-MAPPO proxy CUDA 与 action-scale 诊断，不作为 strict success。
- PhysX / Isaac Sim 作为 checkpoint 级高保真闭环评估和展示层，不进入当前主训练 loop。
- 新训练和评估默认写入 `outputs/runs/<experiment>/<run>/`。

严格结论以机器可读结果为准，优先读取：

```text
outputs/runs/<experiment>/_suite/metrics/strict_acceptance.json
outputs/runs/<experiment>/_suite/metrics/suite_summary.json
outputs/runs/<experiment>/<run>/metrics/final_eval_proxy.json
outputs/runs/<experiment>/<run>/metrics/checkpoint_status.json
```

不要从 GIF、截图、单个 checkpoint 文件名或 TensorBoard 曲线推断 strict pass。

## 已确认基线

`exp008` 是当前推荐的完整 terrain-aware proxy 基线：

```text
outputs/runs/exp_008_terrain3d/_suite/
```

结论：

- 弱 lunar crater 3D proxy。
- 弱 warm-start + PPO。
- seeds `23, 31, 47` 均通过 proxy strict gate。
- 后续改动应避免破坏该 baseline，并在需要时用 exp008 suite 做回归对照。

## Checkpoint 复评计划

候选 checkpoint 使用统一入口：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/<experiment>/<run_id>
```

目标产物：

```text
metrics/final_eval_proxy.json
metrics/checkpoint_status.json
physx/metrics/<terrain>_headless.json
run_manifest.json
```

其中 PhysX 评估只在配置允许且 proxy gate 满足时触发。

## 暂缓的诊断线

`exp009` 只作为 strong terrain 诊断实验：

```text
outputs/runs/exp_009_terrain3d_strong/_suite/
```

已知结论：

- seed23 通过 strict gate。
- seed31 未通过 success/timeout gate。
- seed47 未运行，因为 3-seed strict acceptance 已经不可能成立。

seed31 当前失败模式：

```text
dmax_reduction_ratio: 0.1819  # 通过
success_rate: 0.8740          # 未通过
collision_rate: 0.0049        # 通过
timeout_rate: 0.1250          # 未通过
```

目前先不继续 seed31 失败 episode 回放、动作表示原型或 terrain curriculum 短 run。后续恢复训练研究时，再优先区分 timeout 来自地形速度缩放、dispersion 不稳定、speed hold 不稳定或 `[rho, beta]` 表达能力不足。

## 工程验收

近期验收包含：

1. `.venv_isaaclab/bin/python -m pytest -q -ra`。
2. `scripts/validate_first_stage.py --config configs/experiment/exp_001_minimal.yaml --device cpu --steps 32`。
3. `scripts/run_checkpoint_evaluation.py` 在 mock 或真实 run 上生成 `checkpoint_status.json`。
4. `scripts/evaluate_physx_jackal_tracking.py` 的 headless/render sanity 路径可手动运行，并记录产物路径。

如果新增工程 smoke run，应优先写入 `outputs/runs/<experiment>/<run_id>/` 并更新 runbook；不要把 partial train、GIF 或 TensorBoard 曲线写成 strict success。
