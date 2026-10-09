# 项目文档入口

## 阅读顺序

1. [当前状态](current_status.md)：当前主线、验收、checkpoint 和暂停条件。
2. [路线图](roadmap.md)：下一步顺序；运行任务先核对当前状态。
3. [实验索引](experiments/README.md)：找到对应实验后，读取配置、run 和机器可读结果。
4. 按任务查阅[运行手册](runbooks/)、[架构与设计](architecture/)、[输出管理](references/output_management.md)。

长期技术背景见[设计基线](design/README.md)。[历史归档](archive/README.md)、[研究参考](references/)和[状态汇报](status/)只按需查阅，不能覆盖上述当前入口。

从仓库根目录默认运行 `rg` 时，`.rgignore` 会略过 `docs/archive/`、`docs/design/` 和 `docs/status/`；追溯这些目录时明确指定路径或使用 `rg --no-ignore`。

## 事实来源与边界

- 运行参数以 `configs/` 和实际代码为准；文档描述须与其核对。
- 严格 proxy 结果优先看 `_suite/metrics/strict_acceptance.json`、`suite_summary.json`、独立 `metrics/final_eval_proxy.json` 与 `metrics/checkpoint_status.json`。
- GIF、截图、TensorBoard 曲线、训练 reward 和 checkpoint 文件名不能单独证明 strict pass。
- `outputs/` 是生成目录，默认不进入 Git。实验解释写在 `docs/experiments/`，产物目录规范见[输出管理](references/output_management.md)。
- Isaac/PhysX 闭环评估与 proxy 训练分别记录，不把 proxy 结果写成物理训练结果。

Markdown 数学公式使用 `$...$` 和独立行的 `$$`。
