# Visualization Runbook

## TensorBoard

```bash
.venv_isaaclab/bin/tensorboard \
  --logdir outputs/runs/<experiment> \
  --port 6007
```

Priority scalar groups:

```text
00_overview/
01_ppo_health/
02_task_detail/
03_terrain/
```

## Tags

```bash
.venv_isaaclab/bin/python scripts/summarize_tensorboard_tags.py \
  --logdir outputs/runs/<experiment>
```

## Height Maps and GIFs

Expected per-run artifacts:

```text
figures/terrain_height_map.png
videos/proxy_eval_rollout.gif
```

Height plots and GIFs should include a `height (m)` colorbar. If a GIF is hard to read, inspect `figures/terrain_height_map.png` first.

