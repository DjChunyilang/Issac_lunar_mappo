---
name: lunar-rover-output-management
description: 管理本仓库 outputs 生成产物、run 路径、manifest、历史产物迁移和 Git 边界时使用；不用于训练算法实现。
---

# 产物管理

操作前读 `docs/references/output_management.md`；当前实验结论读 `docs/current_status.md` 和对应实验记录。不要从 `outputs/` 推断项目当前决策。

- 标准路径是 `outputs/runs/<experiment_id>/<run_id>/`；跨 seed 汇总用 `_suite/`。命名、子目录、manifest 和评估入口以输出管理文档为准。
- JSON/JSONL 是结果证据；图、视频与 TensorBoard 用于展示或诊断。strict gate 优先读 `_suite/metrics/strict_acceptance.json`，再读 suite 与独立 run 评估。
- `outputs/**` 默认不进入 Git；需要长期保留的结论写入 `docs/experiments/` 并记录原始路径。
- 迁移旧产物先用 `scripts/organize_outputs.py --dry-run` 检查计划，再优先用 symlink 保持历史路径兼容。不要覆盖已有 run，除非任务明确要求刷新。
- 不删除用户生成的 checkpoint、metrics 或其他本地结果。清理 Git 中误跟踪的生成物时，先确认可再生成及无运行时消费者，再只移出索引。
- 修改产物写入逻辑时，保持训练、独立评估和 PhysX 结果写回同一 run，并维护 `run_manifest.json`。

具体命令与当前目录示例只在 `docs/references/output_management.md` 维护。
