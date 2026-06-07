# 项目文档入口

阅读或修改本项目时，从这里开始。

## 智能体阅读顺序

1. 先读 [current_status.md](current_status.md)，了解当前结果、推荐 checkpoint 和主要阻塞。
2. 理解当前整体路线先读 [architecture/overall_plan_v3.md](architecture/overall_plan_v3.md)；根目录 V2.0 技术文档是原始路线和历史设计依据。
3. 环境搭建或工程验收先读 [runbooks/setup_environment.md](runbooks/setup_environment.md)。
4. 再读 [experiments/README.md](experiments/README.md)，避免误解训练结果。
5. 使用某个 checkpoint 前，读取 `docs/experiments/` 下对应实验文档。
6. 修改 proxy 环境、reward、PPO loop 或地形动力学前，读取 [technical_design.md](technical_design.md)、[interface_spec.md](interface_spec.md) 和新版整体规划。
7. 训练、评估、可视化命令分别查看 [runbooks/train_proxy.md](runbooks/train_proxy.md)、[runbooks/evaluate_proxy.md](runbooks/evaluate_proxy.md)、[runbooks/visualize_results.md](runbooks/visualize_results.md)。

不要根据 GIF、单个 checkpoint 或 TensorBoard 曲线直接判断成功。严格验收以 `_suite/metrics/strict_acceptance.json` 和独立评估指标为准。

## 当前主文档

- [current_status.md](current_status.md)：当前项目状态和下一步工作。
- [architecture/overall_plan_v3.md](architecture/overall_plan_v3.md)：代码审查后的新版整体规划和偏差 review。
- [roadmap.md](roadmap.md)：近期优先级。
- [experiments/README.md](experiments/README.md)：实验索引和通过/失败表。
- [references/output_management.md](references/output_management.md)：输出目录规范和命名规则。
- [runbooks/setup_environment.md](runbooks/setup_environment.md)：Isaac Sim / Isaac Lab / SKRL / 本地任务包安装和验收。
- [runbooks/](runbooks/)：训练、评估、可视化和 PhysX 展示命令。

## 历史文档

长篇进度日志已经归档到 [archive/](archive/)。归档文档只用于追溯过程，不作为当前 checkpoint、实验结论或下一步计划的唯一来源。
