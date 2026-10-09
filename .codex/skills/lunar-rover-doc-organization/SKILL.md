---
name: lunar-rover-doc-organization
description: 整理本仓库 Markdown 文档、实验记录、当前状态、路线图与归档时使用；不用于训练算法实现。
---

# 文档整理

先读 `docs/README.md`，再按任务读取 `docs/current_status.md`、相关实验记录或运行手册。`docs/archive/` 和 `docs/design/` 只在追溯历史或设计依据时读取，不能覆盖当前状态。

- 中文为默认文档语言；命令、路径、配置键和算法缩写保留原文。
- `docs/current_status.md` 只写当前主线、已验证事实、阻塞、推荐 checkpoint 和下一步；历史细节移入实验记录或归档。
- `docs/experiments/README.md` 是实验索引；每项实验记录配置、seed、run、严格门控、结果 JSON、失败原因和结论。
- 不从 GIF、截图、TensorBoard、训练 reward 或 checkpoint 名字推断 strict pass。优先查看 `_suite/metrics/strict_acceptance.json`、`suite_summary.json`、独立 `final_eval_proxy.json`，再看 Markdown 解释。
- `outputs/` 是生成目录，默认不提交。结果路径与目录职责见 `docs/references/output_management.md`。
- 归档页顶部标明历史性质并指向 `docs/current_status.md`。归档内容不再追加当前进度。

文档层级、当前入口和相关链接由 `docs/README.md` 维护；不要在本 skill 复制实验结论、安装命令或版本号。
