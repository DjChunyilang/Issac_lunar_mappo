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

## exp018 随机地形 rollout GIF

exp018 长训练默认只生成训练 metrics 和 checkpoint；如需查看 best checkpoint 在随机地形上的 proxy rollout，可手动渲染：

```bash
.venv_isaaclab/bin/python scripts/render_skrl_proxy_rollout.py \
  --config configs/experiment/exp018_randomized_terrain_pure_rl.yaml \
  --checkpoint outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/checkpoints/best.pt \
  --device cuda \
  --steps 220 \
  --seed 11023 \
  --run-dir outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain \
  --capture-interval 3 \
  --max-frames 90
```

输出：

```text
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/videos/proxy_eval_rollout.gif
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/figures/terrain_height_map.png
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/metrics/proxy_rollout_render.json
```

GIF 只用于观察轨迹和地形交互，strict 结论仍以 `metrics/final_eval_proxy.json`、`metrics/strict_acceptance.json` 和 `metrics/checkpoint_status.json` 为准。
