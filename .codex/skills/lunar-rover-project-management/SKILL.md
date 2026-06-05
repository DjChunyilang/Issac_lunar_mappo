---
name: lunar-rover-project-management
description: 用于本仓库的文档整理、实验状态总结、proxy/PhysX 训练结果解读、outputs 管理、run 管理和 strict acceptance 判断。用户询问项目进度、实验文档、训练结果、输出目录、agent 阅读上下文或长期管理规则时触发。
---

# 月球车项目管理

## 先读什么

触发本 skill 后，按顺序读取：

1. `docs/current_status.md`
2. `docs/experiments/README.md`
3. 相关的 `docs/experiments/exp_*.md`
4. 如果涉及路径或输出结构，读取 `docs/references/output_management.md`

`docs/archive/` 只用于追溯过程。不要把归档日志当成当前事实来源。

## 文档语言规则

本项目 Markdown 文档默认使用中文。命令、路径、配置键、指标字段名、库名和算法缩写可以保留英文原文。新增或修改文档时，优先写中文；不要再新增英文主文档。

## 严格验收

不要根据 GIF、TensorBoard 曲线或单个看起来不错的 checkpoint 判断成功。严格 proxy 验收为：

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

优先使用独立评估 `metrics/final_eval_proxy.json` 和 suite 级 `_suite/metrics/strict_acceptance.json`，不要只看训练过程中的印象。

## 文档维护规则

- `docs/current_status.md` 保持短小，只写当前事实。
- 每个实验在 `docs/experiments/` 下维护独立文档。
- 长日期流水账移动到 `docs/archive/`。
- 命令写入 `docs/runbooks/`，不要散落在每个实验文档中。
- 代码、配置、脚本、测试和 Markdown 需要跟踪；生成结果不要提交。

## 输出规则

标准生成结果目录：

```text
outputs/runs/<experiment_id>/<run_id>/
```

Suite 级结果目录：

```text
outputs/runs/<experiment_id>/_suite/
```

`outputs/**` 已被忽略。如果某个结果需要纳入 git，应在 Markdown 中总结，原始产物仍留在 `outputs/`。

## 训练结果解读

- exp008 是当前完整 3-seed terrain-aware strict pass。
- exp009 strong terrain 是诊断失败：seed23 通过，seed31 未通过 success/timeout，seed47 未运行。
- 不要因为某个 seed 接近通过就继续无界 PPO。先诊断失败 gate。

