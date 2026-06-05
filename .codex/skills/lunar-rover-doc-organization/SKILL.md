---
name: lunar-rover-doc-organization
description: 用于本仓库整理 Markdown 文档体系、拆分进度日志、维护实验记录、runbook、current_status、roadmap、outputs 结果索引和 agent-readable 文档入口时触发。适用于用户要求整理文档、总结训练结果、规范输出路径、归档历史日志、建立实验文档模板或让后续 agent 快速理解项目进度；不用于训练算法实现本身，除非任务是记录、解释或整理训练结果。
---

# 月球车文档整理

## 使用范围

这个 skill 只处理文档与结果管理规范：

- 建立或维护 `docs/` 的入口、实验索引、runbook、归档和引用资料。
- 把长篇日期进度日志拆成稳定的实验文档和当前状态文档。
- 根据机器可读 JSON 结果整理实验结论，不凭 GIF 或主观观感判定成功。
- 让人和 agent 都能快速定位当前进度、失败原因、下一步工作和产物路径。

不要把易过期的当前实验结论写进 skill；当前事实应写在 `docs/current_status.md` 和 `docs/experiments/*.md`。

## 文档语言

本项目 Markdown 文档默认使用中文。命令、路径、配置键、指标字段名、库名、算法缩写和产品名可以保留英文原文，例如 `PPO`、`CUDA`、`TensorBoard`、`outputs/runs/...`。新增或修改文档时，优先写中文；不要新增英文主文档。

## 目标结构

整理文档时优先向下面的结构收敛：

```text
docs/
  README.md
  current_status.md
  roadmap.md
  architecture/
    proxy_training.md
    physx_validation.md
  experiments/
    README.md
    exp_006_ppo_selected.md
    exp_007_phase_c.md
    exp_008_terrain3d.md
    exp_009_terrain3d_strong.md
  runbooks/
    train_proxy.md
    evaluate_proxy.md
    visualize_results.md
    physx_showcase.md
  references/
    output_management.md
    tensorboard.md
  archive/
    progress_summary_2026-05-26.md
    build_progress_2026-05-19.md
```

如果当前仓库尚未完全迁移到该结构，整理任务中逐步迁移；不要删除历史信息，先移动到 `docs/archive/` 并加归档提示。

## 阅读顺序

当用户要求了解进度或整理文档时，按顺序读取：

1. `docs/README.md`
2. `docs/current_status.md`
3. `docs/experiments/README.md`
4. 相关的 `docs/experiments/exp_*.md`
5. 涉及命令时读取 `docs/runbooks/*.md`
6. 涉及输出路径时读取 `docs/references/output_management.md`

`docs/archive/` 只用于追溯过程，不作为当前事实来源。

## 核心入口规范

`docs/current_status.md` 控制在 1 到 2 页内，只写当前事实：

- 当前主线：训练环境、展示环境、当前最佳实验。
- 当前推荐 checkpoint 和结果路径。
- 当前阻塞：明确失败 gate，不只写 reward。
- 下一步：具体到要改的模块、reward、action representation 或评估方式。

`docs/README.md` 必须说明 agent 起步阅读顺序，并明确：

- 不要从 GIF、截图或 TensorBoard 曲线推断 strict pass。
- strict 结果以 `_suite/metrics/strict_acceptance.json` 和独立 `final_eval_proxy.json` 为准。
- `outputs/` 是生成目录，默认不要提交。

## 实验文档模板

每个 `docs/experiments/exp_xxx.md` 使用固定结构：

```text
# exp_xxx 名称

## 目的
为什么做这个实验。

## 配置
config 文件、关键参数、terrain 参数、训练预算。

## 严格标准
dmax / success / collision / timeout。

## 结果表
seed、run_id、checkpoint、final_eval、是否通过。

## 失败分析
失败在哪个 gate，不要只写 reward。

## 产物路径
summary.json、final_eval_proxy.json、curves、GIF、TensorBoard。

## 结论
能不能作为当前主结果。

## 下一步
基于该实验应该改什么。
```

`docs/experiments/README.md` 只放实验索引表，便于快速比较：

```text
| exp | terrain | method | seeds | strict | conclusion | doc |
| --- | --- | --- | --- | --- | --- | --- |
```

## 结果可信来源

训练结果以机器可读文件为准，优先级：

1. `outputs/runs/<experiment>/_suite/metrics/strict_acceptance.json`
2. `outputs/runs/<experiment>/_suite/metrics/suite_summary.json`
3. `outputs/runs/<experiment>/<run_id>/metrics/final_eval_proxy.json`
4. `outputs/runs/<experiment>/<run_id>/metrics/summary.json`
5. Markdown 解释和人工总结

GIF、截图和 TensorBoard 曲线只能用于展示和诊断，不能单独作为严格收敛证据。

严格 proxy gate 默认写成：

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

如果某个实验使用不同 gate，必须在该实验文档中显式写出。

## 归档规则

- 日期型流水账、历史构建日志和过长进度摘要放入 `docs/archive/`。
- 归档文件顶部加中文提示：这是归档日志，当前状态请读 `docs/current_status.md`。
- 不要在归档文件中继续追加新主线进度。
- 新训练结果优先写入对应 `docs/experiments/exp_*.md`，再更新 `docs/current_status.md`。

## 输出与 Git 规则

标准输出目录：

```text
outputs/runs/<experiment_id>/<run_id>/
outputs/runs/<experiment_id>/_suite/
```

`outputs/**` 默认被忽略。不要提交 checkpoint、TensorBoard events、PNG、GIF 或 JSON metrics；需要长期保留时，在 Markdown 中记录结论和路径，原始产物留在 `outputs/`。
