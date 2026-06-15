# 项目文档入口

阅读或修改本项目时，从这里开始。

## 智能体阅读顺序

1. 先读 [current_status.md](current_status.md)，了解当前主线、推荐 checkpoint、评估状态和结果边界。
2. 理解整体路线读 [architecture/overall_plan_v3.md](architecture/overall_plan_v3.md)；根目录 V1.0 / V2.0 是历史脚手架和原始设计依据。
3. 理解本轮路线修订读根目录 `multi_rover_proxy_train_isaac_eval_supplement_v3_0.md`。
4. 推进高保真评估读 [architecture/env_completion_plan.md](architecture/env_completion_plan.md)。
5. 再读 [experiments/README.md](experiments/README.md)，避免误解训练结果。
6. 使用某个 checkpoint 前，读取 `docs/experiments/` 下对应实验文档，并检查该 run 的 `metrics/checkpoint_status.json`。
7. 训练、评估、可视化命令分别查看 [runbooks/train_proxy.md](runbooks/train_proxy.md)、[runbooks/train_skrl_mappo.md](runbooks/train_skrl_mappo.md)、[runbooks/evaluate_proxy.md](runbooks/evaluate_proxy.md)、[runbooks/physx_showcase.md](runbooks/physx_showcase.md) 和 [runbooks/visualize_results.md](runbooks/visualize_results.md)。
8. 输出路径和 manifest 规范查看 [references/output_management.md](references/output_management.md)。

不要根据 GIF、单个 checkpoint 或 TensorBoard 曲线直接判断成功。严格 proxy 结论以 `_suite/metrics/strict_acceptance.json`、独立 `metrics/final_eval_proxy.json` 和 `metrics/checkpoint_status.json` 为准。PhysX / Jetbot 结果是 high-fidelity closed-loop evaluation，不等于 Isaac 物理训练结果。

## 当前主文档

- [current_status.md](current_status.md)：当前项目状态和下一步工作。
- [architecture/overall_plan_v3.md](architecture/overall_plan_v3.md)：当前“proxy 训练 + Isaac/PhysX 闭环评估”主规划。
- [architecture/env_completion_plan.md](architecture/env_completion_plan.md)：高保真评估层推进清单。
- [roadmap.md](roadmap.md)：近期优先级。
- [experiments/README.md](experiments/README.md)：实验索引和通过/失败表。
- [interface_spec.md](interface_spec.md)：actor observation、critic state、action 和当前 observation schema。
- [references/output_management.md](references/output_management.md)：输出目录规范和命名规则。
- [runbooks/setup_environment.md](runbooks/setup_environment.md)：Isaac Sim / Isaac Lab / SKRL / 本地任务包安装和验收。
- [runbooks/train_skrl_mappo.md](runbooks/train_skrl_mappo.md)：SKRL-MAPPO proxy 训练诊断、exp012 / exp013 和 checkpoint 评估入口。
- [runbooks/](runbooks/)：训练、评估、可视化和 PhysX 展示命令。

## 工程验收入口

当前 CPU unit contract 以 GitHub Actions 和本地同一测试命令为准：

```bash
.venv_isaaclab/bin/python -m pytest -q -ra
```

CI 明确使用 Python 3.12，并固定 CPU 依赖组合。`tests/test_skrl_import.py` 是非 skip 的 SKRL 导入验收，防止 SKRL 相关测试被 skip 后误判为绿灯。

## 历史文档

长篇进度日志已经归档到 [archive/](archive/)。归档文档只用于追溯过程，不作为当前 checkpoint、实验结论或下一步计划的唯一来源。
