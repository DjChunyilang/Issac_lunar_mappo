# 输出管理

生成产物采用 run-oriented 目录结构管理。目标是让长期实验可检索，同时避免提交大型生成文件。

## 标准目录结构

每个具体 run 使用一个独立目录：

```text
outputs/runs/<experiment_id>/<run_id>/
  config/
    experiment.yaml
  checkpoints/
    best.pt
    ppo_update_XXX.pt
  metrics/
    summary.json
    train_metrics.jsonl
    eval_metrics.json
    final_eval_proxy.json
    strict_acceptance.json
    checkpoint_status.json
  figures/
    convergence_curves.png
    safety_diagnostics.png
    comparison_curves.png
  videos/
    proxy_eval_rollout.gif
  tensorboard/
    events.out.tfevents...
  tensorboard_curated/
    events.out.tfevents...
  physx/
    metrics/
      <terrain>_headless.json
      <terrain>_render.json
    figures/
      <terrain>_render_scene.png
    videos/
      <terrain>_render_rollout.gif
  run_manifest.json
```

实验级 suite 输出使用保留 run 名 `_suite`：

```text
outputs/runs/<experiment_id>/_suite/
  metrics/
    suite_summary.json
    strict_acceptance.json
    final_eval_best.json
  figures/
    comparison_curves.png
    safety_diagnostics.png
  run_manifest.json
```

根索引用于汇总所有 manifest：

```text
outputs/runs/_index.json
```

历史路径 `outputs/logs/`、`outputs/checkpoints/`、`outputs/figures/` 和 `outputs/videos/` 保留兼容。新工作优先使用 `outputs/runs/`。

当前 SKRL-MAPPO CUDA 诊断脚本仍处于过渡阶段：

```text
outputs/runs/<experiment_id>/metrics.jsonl
outputs/runs/<experiment_id>/diagnosis_<label>_<timesteps>.json
outputs/runs/<experiment_id>/suite_logs/
outputs/runs/cuda_training_validation_summary.json
```

这些文件是 ignored 的工程诊断产物，不是 strict acceptance。长期结论写入 `docs/experiments/*.md`；后续正式 M3 训练应继续迁移到 `outputs/runs/<experiment_id>/<run_id>/metrics/` 和 `_suite/metrics/`。

当前标准 terrain-aware convergence suite：

```text
outputs/runs/exp_008_terrain3d/_suite/
```

该 suite 包含最终严格 proxy 验收、对比图，以及复制出的各 seed best checkpoint。

## 命名规则

- `experiment_id`：稳定的实验族，使用 lower snake case，通常为 `exp_###_<topic>`。
  - 推荐：`exp_007_phase_c`、`exp_008_terrain3d`。
  - 不要因为 seed 或小 retry 改变 `experiment_id`。
- `run_id`：具体设置，使用 lower snake case，不含空格。
  - 推荐模式：`<mode>_seed<seed>_<budget>_<terrain>_<short_tag>`。
  - 示例：`weak_warmstart_seed23_6m_lunar_crater_bc50`、`pure_rl_seed31_2m_flat`、`smoke_seed23_8k_cpu`。
- checkpoint：
  - `checkpoints/best.pt`：该 run 的标准 checkpoint。
  - `checkpoints/ppo_update_XXX.pt`：可选的 update checkpoint。
- metrics：
  - `metrics/summary.json`：dashboard 和快速检查用的一文件摘要。
  - `metrics/train_metrics.jsonl`：逐 update 训练记录。
  - `metrics/eval_metrics.json`：训练过程中的 deterministic eval 记录。
  - `metrics/final_eval_proxy.json`：训练后的独立 proxy 评估。
  - `metrics/checkpoint_status.json`：统一 checkpoint 评估状态，连接 proxy gate 与 high-fidelity eval。
- visual：
  - Proxy 图和视频放在 `figures/` 和 `videos/`。
  - Isaac Sim / PhysX 产物放在 `physx/metrics|figures|videos`。

## Checkpoint 状态

`metrics/checkpoint_status.json` 的 `state` 只能使用：

```text
candidate
proxy_passed
physx_evaluated
physx_passed
final_selected
```

`candidate` 表示尚未通过 proxy strict gate；`proxy_passed` 表示 proxy gate 通过但 high-fidelity eval 尚未通过或未触发；`physx_evaluated` 表示已跑高保真评估但未通过 gate；`physx_passed` 表示 high-fidelity gate 通过；`final_selected` 只能用于明确选定的最终 checkpoint。

## 整理已有 outputs

只查看迁移计划，不写入文件：

```bash
.venv_isaaclab/bin/python scripts/organize_outputs.py \
  --all-known \
  --mode symlink \
  --dry-run
```

为所有已知历史实验创建标准 symlink：

```bash
.venv_isaaclab/bin/python scripts/organize_outputs.py \
  --all-known \
  --mode symlink
```

刷新单个实验：

```bash
.venv_isaaclab/bin/python scripts/organize_outputs.py \
  --experiment exp_006_ppo_selected \
  --mode symlink
```

刷新 Phase C run，并包含 PhysX 展示产物：

```bash
.venv_isaaclab/bin/python scripts/organize_outputs.py \
  --preset exp007_phase_c \
  --mode symlink \
  --overwrite
```

只有当 run 目录必须完全自包含并脱离历史路径时，才使用 `--mode copy`。除非要刷新生成的标准链接，否则避免使用 `--overwrite`。

## 后续训练

新训练配置中设置：

```yaml
experiment:
  name: exp_008_terrain3d
  output_layout: run
```

运行时显式指定 `run_id`：

```bash
.venv_isaaclab/bin/python scripts/train_proxy_convergence.py \
  --config configs/experiment/exp_008_terrain3d_weak_warmstart.yaml \
  --output-layout run \
  --run-name weak_warmstart_seed23_6m_lunar_crater_bc20 \
  --device cuda
```

训练脚本会直接在 run 目录下写入 `config/experiment.yaml`、`metrics/summary.json`、`metrics/eval_metrics.json`、`run_manifest.json`、TensorBoard events、图、GIF 和 `checkpoints/best.pt`。

TensorBoard：

```bash
.venv_isaaclab/bin/tensorboard \
  --logdir outputs/runs/exp_008_terrain3d \
  --port 6007
```

## 后续 Proxy 评估

优先使用统一 checkpoint 评估入口：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/exp_008_terrain3d_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20 \
  --skip-physx
```

该命令会写入：

```text
metrics/final_eval_proxy.json
metrics/checkpoint_status.json
run_manifest.json
```

如需只运行底层 proxy eval，可使用 `--run-dir` 将独立评估结果写回同一 run：

```bash
.venv_isaaclab/bin/python scripts/evaluate_proxy_policy.py \
  --config configs/experiment/exp_008_terrain3d_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20/checkpoints/best.pt \
  --device cuda \
  --num-envs 512 \
  --steps 160 \
  --run-dir outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20
```

带 `--run-dir` 时默认输出：

```text
outputs/runs/<experiment_id>/<run_id>/metrics/final_eval_proxy.json
```

## 后续 PhysX 评估

优先让 `scripts/run_checkpoint_evaluation.py` 根据 `evaluation.high_fidelity_eval.trigger` 自动触发。需要手动运行底层 PhysX 评估时，使用 `--run-dir` 让高保真评估产物留在同一个 run：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --config configs/experiment/exp_008_terrain3d_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20/checkpoints/best.pt \
  --terrain lunar_crater \
  --episodes 3 \
  --steps 100 \
  --run-dir outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20
```

渲染录制：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --config configs/experiment/exp_008_terrain3d_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20/checkpoints/best.pt \
  --terrain lunar_crater \
  --episodes 1 \
  --steps 100 \
  --render \
  --run-dir outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20
```

带 `--run-dir` 时默认 PhysX 路径：

```text
physx/metrics/lunar_crater_headless.json
physx/metrics/lunar_crater_render.json
physx/figures/lunar_crater_render_scene.png
physx/videos/lunar_crater_render_rollout.gif
```

## Git 策略

`outputs/**` 已被忽略。为了保证结果可复现，需要跟踪：

- `configs/experiment/*.yaml`
- 训练、评估、整理脚本
- 实验 Markdown 文档
- `docs/references/output_management.md`

不要提交生成的 checkpoint、TensorBoard events、PNG、GIF 或 JSON metrics；除非某个报告明确需要少量 curated artifact。
