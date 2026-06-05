---
name: lunar-rover-output-management
description: 用于本仓库管理 outputs 生成产物、设计 run-oriented 输出目录、命名 experiment_id/run_id、整理历史 logs/checkpoints/figures/videos、判断哪些产物可提交、维护 run_manifest 和 suite 汇总路径时触发。适用于用户要求整理 outputs、规范训练输出、查找 checkpoint/metrics/GIF/TensorBoard、迁移旧产物、清理生成文件或长期管理实验结果；不用于修改训练算法，除非任务是让算法按标准路径写产物。
---

# 月球车 outputs 管理

## 使用范围

这个 skill 只处理生成产物管理：

- 规范 `outputs/runs/<experiment_id>/<run_id>/` 目录结构。
- 判断 checkpoint、metrics、figures、videos、TensorBoard events 应放在哪里。
- 整理历史 `outputs/logs/`、`outputs/checkpoints/`、`outputs/figures/`、`outputs/videos/`。
- 决定哪些生成产物不应提交，哪些结论应写入 Markdown。
- 检查训练、评估、PhysX 展示是否写入标准路径。

详细命令和当前项目规范以 `docs/references/output_management.md` 为准；处理 outputs 前优先读取该文件。

## 核心原则

- 新产物默认写入 `outputs/runs/`，不要继续扩散到旧的 `outputs/logs/` 等平铺目录。
- 每个 run 一个独立目录；跨 seed 或跨方法汇总放到 `_suite/`。
- JSON/JSONL 是结果事实来源；PNG、GIF、TensorBoard 用于展示和诊断。
- `outputs/**` 默认不进 git。需要长期保留的结果，写 Markdown 摘要和路径，不提交大文件。
- 不要删除用户产物。清理或迁移时优先 dry-run；需要破坏性删除时必须有明确用户请求。

## 标准目录

具体 run：

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
    terrain_height_map.png
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

实验级 suite：

```text
outputs/runs/<experiment_id>/_suite/
  metrics/
    suite_summary.json
    strict_acceptance.json
    final_eval_best.json
  figures/
    comparison_curves.png
    safety_diagnostics.png
    terrain_height_map.png
  checkpoints/
    seed_<seed>_best.pt
  run_manifest.json
```

全局索引：

```text
outputs/runs/_index.json
```

## 命名规则

`experiment_id` 是稳定实验族，使用 lower snake case：

```text
exp_006_ppo_selected
exp_008_terrain3d
exp_009_terrain3d_strong
```

不要因为 seed、retry、设备或小参数变化新建 experiment_id；这些写进 run_id。

`run_id` 是具体运行设置，使用 lower snake case，不含空格：

```text
<mode>_seed<seed>_<budget>_<terrain>_<short_tag>
```

示例：

```text
weak_warmstart_seed23_12m_strong_lunar_crater_nenv1024
pure_rl_seed31_2m_flat
smoke_seed23_8k_cpu
```

## 文件职责

- `config/experiment.yaml`：该 run 使用的配置快照。
- `checkpoints/best.pt`：该 run 标准 best checkpoint。
- `metrics/summary.json`：快速检查用摘要。
- `metrics/train_metrics.jsonl`：逐 update 训练记录。
- `metrics/eval_metrics.json`：训练过程 deterministic eval。
- `metrics/final_eval_proxy.json`：训练后独立 proxy 复验。
- `metrics/strict_acceptance.json`：该 run 或 suite 的 strict gate 判定。
- `figures/`：曲线、诊断图、高度热力图。
- `videos/`：proxy GIF。
- `tensorboard/`：原始 TensorBoard events。
- `tensorboard_curated/`：回填或筛选后的重点曲线 events。
- `physx/`：Isaac Sim / PhysX headless 或 render 产物。
- `run_manifest.json`：运行元信息、配置、命令、关键输出索引。

## 结果判读顺序

判断实验是否成功时，按顺序读：

1. `outputs/runs/<experiment_id>/_suite/metrics/strict_acceptance.json`
2. `outputs/runs/<experiment_id>/_suite/metrics/suite_summary.json`
3. `outputs/runs/<experiment_id>/<run_id>/metrics/final_eval_proxy.json`
4. `outputs/runs/<experiment_id>/<run_id>/metrics/summary.json`
5. 图、GIF、TensorBoard 曲线

不要从单个 GIF、截图、训练 reward 曲线或 checkpoint 文件名推断 strict pass。

## 整理历史产物

迁移或整理前先 dry-run：

```bash
.venv_isaaclab/bin/python scripts/organize_outputs.py \
  --all-known \
  --mode symlink \
  --dry-run
```

优先使用 symlink 保留历史路径兼容。只有需要自包含复制时才用 `--mode copy`。除非用户明确要求刷新，否则避免 `--overwrite`。

## Git 规则

通常应提交：

- `configs/experiment/*.yaml`
- 训练、评估、整理脚本
- `docs/experiments/*.md`
- `docs/references/output_management.md`
- `docs/current_status.md`

通常不要提交：

- `outputs/**`
- checkpoint
- TensorBoard events
- PNG、GIF、MP4
- 训练或评估 JSON/JSONL metrics

例外：如果用户明确要求提交少量 curated artifact，先确认文件体积和必要性，并在提交说明中写清楚原因。

## 修改代码时的要求

如果任务要求让训练或评估脚本写入标准 outputs：

- 新训练入口必须支持 `--output-layout run` 和 `--run-name`。
- 独立评估入口应支持 `--run-dir`，并默认写入同一 run 的 `metrics/final_eval_proxy.json`。
- PhysX 展示入口应支持 `--run-dir`，并默认写入同一 run 的 `physx/metrics|figures|videos`。
- 每个 run 应写 `run_manifest.json`，记录命令、配置路径、checkpoint、metrics、figures、videos 和设备信息。
