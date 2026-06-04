# Output Management

Generated artifacts are managed with a run-oriented layout. The goal is to keep long experiments searchable without committing large generated files.

## Canonical Layout

Use one directory per concrete run:

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

Experiment-level suite outputs use a reserved run:

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

A root index summarizes all manifests:

```text
outputs/runs/_index.json
```

Legacy paths such as `outputs/logs/`, `outputs/checkpoints/`, `outputs/figures/`, and `outputs/videos/` are kept for historical compatibility. New work should prefer `outputs/runs/`.

Current canonical terrain-aware convergence suite:

```text
outputs/runs/exp_008_terrain3d/_suite/
```

The suite contains final strict proxy acceptance, comparison figures, and copied seed-specific best checkpoints.

## Naming Rules

- `experiment_id`: stable experiment family, lower snake case, normally `exp_###_<topic>`.
  - Good: `exp_007_phase_c`, `exp_008_terrain3d`.
  - Avoid changing this for every seed or minor retry.
- `run_id`: concrete settings, lower snake case, no spaces.
  - Recommended pattern: `<mode>_seed<seed>_<budget>_<terrain>_<short_tag>`.
  - Examples: `weak_warmstart_seed23_6m_lunar_crater_bc50`, `pure_rl_seed31_2m_flat`, `smoke_seed23_8k_cpu`.
- Checkpoints inside a run:
  - `checkpoints/best.pt`: canonical checkpoint for the run.
  - `checkpoints/ppo_update_XXX.pt`: optional update-specific checkpoint.
- Metrics:
  - `metrics/summary.json`: one-file summary for dashboards and quick inspection.
  - `metrics/train_metrics.jsonl`: update-by-update training records.
  - `metrics/eval_metrics.json`: deterministic evaluation records from training.
  - `metrics/final_eval_proxy.json`: independent post-training proxy evaluation.
- Visuals:
  - Proxy figures/videos stay under `figures/` and `videos/`.
  - Isaac Sim / PhysX artifacts stay under `physx/metrics|figures|videos`.

## Organizing Existing Outputs

Inspect the migration plan without writing files:

```bash
.venv_isaaclab/bin/python scripts/organize_outputs.py \
  --all-known \
  --mode symlink \
  --dry-run
```

Create canonical symlinks for all known legacy experiments:

```bash
.venv_isaaclab/bin/python scripts/organize_outputs.py \
  --all-known \
  --mode symlink
```

Refresh a single experiment:

```bash
.venv_isaaclab/bin/python scripts/organize_outputs.py \
  --experiment exp_006_ppo_selected \
  --mode symlink
```

Refresh the curated Phase C run, including PhysX showcase artifacts:

```bash
.venv_isaaclab/bin/python scripts/organize_outputs.py \
  --preset exp007_phase_c \
  --mode symlink \
  --overwrite
```

Use `--mode copy` only when the run directory must be self-contained and independent of legacy paths. Avoid `--overwrite` unless you are refreshing generated canonical links.

## Future Training

For new training configs, set:

```yaml
experiment:
  name: exp_008_terrain3d
  output_layout: run
```

Run with an explicit `run_id`:

```bash
.venv_isaaclab/bin/python scripts/train_proxy_convergence.py \
  --config configs/experiment/exp_008_terrain3d_weak_warmstart.yaml \
  --output-layout run \
  --run-name weak_warmstart_seed23_6m_lunar_crater_bc20 \
  --device cuda
```

The training script writes `config/experiment.yaml`, `metrics/summary.json`, `metrics/eval_metrics.json`, `run_manifest.json`, TensorBoard events, plots, GIFs, and `checkpoints/best.pt` directly under the run directory.

TensorBoard:

```bash
.venv_isaaclab/bin/tensorboard \
  --logdir outputs/runs/exp_008_terrain3d \
  --port 6007
```

## Future Proxy Evaluation

Use `--run-dir` so independent evaluation is stored with the run:

```bash
.venv_isaaclab/bin/python scripts/evaluate_proxy_policy.py \
  --config configs/experiment/exp_008_terrain3d_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20/checkpoints/best.pt \
  --device cuda \
  --num-envs 512 \
  --steps 160 \
  --run-dir outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20
```

Default output with `--run-dir`:

```text
outputs/runs/<experiment_id>/<run_id>/metrics/final_eval_proxy.json
```

## Future PhysX Evaluation

Use `--run-dir` so high-fidelity evaluation artifacts stay with the same run:

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --config configs/experiment/exp_008_terrain3d_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20/checkpoints/best.pt \
  --terrain lunar_crater \
  --episodes 3 \
  --steps 100 \
  --run-dir outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20
```

Rendered capture:

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

Default PhysX paths with `--run-dir`:

```text
physx/metrics/lunar_crater_headless.json
physx/metrics/lunar_crater_render.json
physx/figures/lunar_crater_render_scene.png
physx/videos/lunar_crater_render_rollout.gif
```

## Git Policy

`outputs/**` is ignored. Keep results reproducible by tracking:

- `configs/experiment/*.yaml`
- training/evaluation/organization scripts
- `docs/progress_summary_*.md`
- `docs/output_management.md`

Do not commit generated checkpoints, TensorBoard events, PNGs, GIFs, or JSON metrics unless a specific report explicitly needs a curated artifact.
