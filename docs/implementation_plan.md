# 当前实施计划

本文是当前执行计划，负责描述 V3 路线、实施里程碑、checkpoint 状态机和验收标准。工程骨架见 [scaffold.md](scaffold.md)，技术模型见 [technical_design.md](technical_design.md)，实验结果见 [experiments/README.md](experiments/README.md)。

## 当前路线

当前固定路线为：

```text
高吞吐 proxy 环境训练
-> proxy strict evaluation
-> Isaac Sim / Isaac Lab / PhysX high-fidelity closed-loop evaluation
```

Proxy 环境负责高频采样、reward 调试、observation/action 接口验证、PPO / SKRL-MAPPO 训练诊断和 checkpoint selection。Isaac/PhysX 负责低频 checkpoint 级闭环评估、迁移 sanity check、失效分析和展示。

当前 PhysX 侧使用 Jackal 轮式资产作为 placeholder。Jackal 可以验证控制链路、强三维地形 mesh、姿态稳定性和输出链路，但不能写成最终真实月球车训练结果。

## 分层实施

### D0 文档与接口口径

- 保持 `docs/current_status.md` 为当前事实入口。
- 保持 [technical_design.md](technical_design.md)、[scaffold.md](scaffold.md) 和本文职责分离。
- 所有报告严格区分 proxy training、proxy strict evaluation、Isaac/PhysX high-fidelity closed-loop evaluation。
- GIF、截图、单个 checkpoint 名称和 TensorBoard 曲线不能单独作为 strict pass 证据。

### D1 Proxy 训练与回归

- 维持 exp008 为当前完整 terrain-aware proxy baseline。
- 暂停默认追加 exp009/exp010 strong terrain retry 和 exp012/exp013 long-budget proxy run，除非新 run 服务明确假设验证。
- 新训练产物写入 `outputs/runs/<experiment>/<run_id>/`，并保留 run manifest。
- 基础回归命令：

```bash
.venv_isaaclab/bin/python -m pytest -q -ra
```

### D2 Checkpoint 统一评估

候选 checkpoint 使用统一入口：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/<experiment>/<run_id>
```

必须生成：

```text
metrics/final_eval_proxy.json
metrics/checkpoint_status.json
run_manifest.json
```

`checkpoint_status.json` 是连接 proxy strict 与 high-fidelity eval 的标准产物。

### D3 Isaac/PhysX 高保真评估

- 使用 `scripts/evaluate_physx_jackal_tracking.py` 运行平地和 strong lunar crater tracking。
- 记录 `rmse_cross_track_m`、`max_cross_track_m`、`path_completion_ratio`、`max_tilt_deg` 和 `control_steps_per_s`。
- 每条轨迹保留 `timeseries.csv` 和 `tracking.png`。
- 强地形失败应解释为当前 Jackal 低层跟踪控制或 placeholder 资产限制，不能写成 proxy 集合任务失败或 Isaac 训练失败。

### D4 资产与动力学升级

只有 high-fidelity eval 暴露系统性迁移失败时，再进入以下方向：

- 用更接近 lunar rover 的 USD/URDF 或轮式底盘参数替换 Jackal。
- 增加低重力、摩擦、轮地接触、坡面稳定性和倾覆风险评估配置。
- 考虑 domain randomization、Isaac-based fine-tuning 或更高保真 proxy dynamics。

## Checkpoint 状态机

checkpoint 状态只使用：

```text
candidate
proxy_passed
physx_evaluated
physx_passed
final_selected
```

标准流转：

```text
训练产生 checkpoint
-> proxy deterministic evaluation
-> proxy strict gate
-> high-fidelity PhysX evaluation queue
-> checkpoint_status.json
-> final_selected
```

若 proxy gate 未通过，不默认触发 PhysX 评估；若 PhysX 是手动触发，状态仍应明确记录 skip reason 或 manual trigger。

## 当前结果边界

- exp006 / exp008 是 proxy strict pass，不是 Isaac Lab 物理训练 pass。
- exp008 是当前推荐的完整 terrain-aware proxy baseline。
- exp012 / exp013 是 SKRL-MAPPO proxy CUDA 与 action-scale 诊断，不作为 strict success。
- PhysX / Jackal 结果应写成“高保真闭环评估 / 迁移 sanity check”，不能写成“真实月球车物理训练结果”。

## 验收标准

- 文档入口清晰：`docs/README.md` 能引导到当前状态、技术设计、脚手架、实施计划、实验索引和 runbook。
- 接口契约稳定：`docs/interface_spec.md` 与测试中的 observation schema 一致。
- 结果可信：strict 结论来自 `_suite/metrics/strict_acceptance.json`、`metrics/final_eval_proxy.json` 或 `metrics/checkpoint_status.json`。
- 工程可回归：`.venv_isaaclab/bin/python -m pytest -q -ra` 可作为基础测试入口。
- 输出可追溯：新训练和评估默认写入 `outputs/runs/<experiment>/<run_id>/`，原始产物不提交。
