# 可视化操作手册

## TensorBoard

```bash
.venv_isaaclab/bin/tensorboard \
  --logdir outputs/runs/<experiment> \
  --port 6007
```

优先查看的 scalar 分组：

```text
00_overview/
01_ppo_health/
02_task_detail/
03_terrain/
```

## 查看 Tags

```bash
.venv_isaaclab/bin/python scripts/summarize_tensorboard_tags.py \
  --logdir outputs/runs/<experiment>
```

## 高度图和 GIF

每个 run 期望包含：

```text
figures/terrain_height_map.png
videos/proxy_eval_rollout.gif
```

高度图和 GIF 必须包含 `height (m)` colorbar。GIF 中地形不清楚时，先检查 `figures/terrain_height_map.png`。
