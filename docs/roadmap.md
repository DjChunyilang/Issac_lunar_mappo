# 路线图

## 立即处理

1. 将 `multi_rover_design_revision_proxy_train_isaac_eval.md` 固定为当前主路线来源。
2. 保持 exp008 为当前已验证的 3-seed terrain-aware proxy baseline。
3. 暂停默认追加 exp009/exp010 strong terrain retry、exp012/exp013 action-scale long run；新增 proxy run 必须服务 checkpoint 评估、接口回归或明确假设验证。
4. 对候选 checkpoint 统一运行 `scripts/run_checkpoint_evaluation.py`，生成 `metrics/final_eval_proxy.json` 和 `metrics/checkpoint_status.json`。
5. 文档中严格区分三类结果：proxy training、proxy strict evaluation、Isaac/PhysX high-fidelity closed-loop evaluation。

## 近期工作

- D0 回归底线：保持 `.venv_isaaclab/bin/python -m pytest -q -ra` 可通过；SKRL import 测试不能 skip。
- D1 评估编排：让 exp008 / exp013 等配置携带 `evaluation:` block，统一 proxy eval 与 PhysX eval 的触发规则。
- D2 Checkpoint 状态：每个候选 run 都记录 `candidate`、`proxy_passed`、`physx_evaluated`、`physx_passed` 或 `final_selected`。
- D3 PhysX/Jackal 评估：把 Jackal 作为 high-fidelity tracking validation 资产，覆盖平地调优和 strong lunar crater 直线、绕圆、正弦跟踪，记录 tracking error、completion、tilt 和 throughput。
- D4 报告口径：更新根目录 V1.0 / V2.0 的历史提示，并新增 V3.0 补充说明，避免把旧文档误读为当前训练事实。

## 中长期工作

- 用更接近月球车的 USD/URDF 或轮式底盘参数替换 Jackal placeholder。
- 增加低重力、摩擦、轮地接触、坡面稳定性和倾覆风险的高保真评估配置。
- 如果 high-fidelity eval 暴露系统性迁移失败，再引入 Isaac-based fine-tuning、domain randomization 或更高保真的 proxy dynamics。
- 构建报告生成器，从 `strict_acceptance.json`、`final_eval_proxy.json`、`checkpoint_status.json` 和 PhysX metrics 自动更新实验文档。
