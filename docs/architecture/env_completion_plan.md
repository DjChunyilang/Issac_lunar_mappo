# 高保真评估推进清单

本文档追踪 Isaac Sim / Isaac Lab / PhysX 评估层的近期工作。当前主线不是把 PhysX 放进每次训练采样，而是把它作为 checkpoint 级高保真闭环评估平台。

## 当前资产

- 已有 proxy 侧 actor observation、critic state、`[rho, beta]` action、reward、termination、terrain feature 和 outputs layout。
- 已有 exp008 terrain-aware proxy strict baseline。
- 已有 SKRL-MAPPO proxy 训练诊断入口。
- 已有 `scripts/evaluate_physx_jackal_tracking.py`，可运行 Jackal 平地调优和 strong lunar crater 直线、绕圆、正弦跟踪测试。
- 已有 `scripts/run_checkpoint_evaluation.py`，统一 proxy final eval、PhysX trigger、checkpoint status 和 manifest 更新。

## 当前缺口

- Jackal 仍是 high-fidelity validation placeholder，不是最终 lunar rover asset。
- 低重力、真实轮壤接触、沉陷、打滑和悬挂响应尚未建模。
- PhysX 样本量仍偏小，不能直接支撑最终论文级统计结论。
- 尚未形成 proxy vs PhysX 的失败案例对齐分析和 trajectory comparison。

## 推进原则

1. Proxy 训练是当前主训练路径；PhysX 评估保持 checkpoint-based 低频触发。
2. `checkpoint_status.json` 是连接 proxy strict 与 high-fidelity eval 的标准产物。
3. Jackal 结果必须写成“高保真评估 / sanity check”，不能写成“真实月球车物理训练结果”。
4. 只有 high-fidelity eval 发现系统性失败时，才考虑 Isaac-based fine-tuning 或更高保真动力学训练。
5. 所有新评估产物写入 `outputs/runs/<experiment>/<run_id>/`。

## M0 Checkpoint 评估闭环

目标：每个候选 checkpoint 都有机器可读状态。

验收命令：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/<experiment>/<run_id>
```

验收输出：

```text
metrics/final_eval_proxy.json
metrics/checkpoint_status.json
run_manifest.json
```

## M1 PhysX / Jackal 跟踪复评

目标：让高保真评估从展示 sanity 升级为可复查的轨迹跟踪测试。

最小要求：

- 平地先运行 `--tune-flat`，保存 `flat_tuning_grid.json`；
- strong lunar crater 使用 exp009 强地形参数；
- 记录 `rmse_cross_track_m`、`max_cross_track_m`、`path_completion_ratio`、`max_tilt_deg` 和 `control_steps_per_s`；
- 每条轨迹保留 `timeseries.csv` 和 `tracking.png`，便于定位偏航、打滑、姿态或控制跟踪失败。

## M2 资产与环境升级

目标：从 Jackal placeholder 向更接近 lunar rover 的评估资产过渡。

候选方向：

- Nova Carter 或其他官方轮式资产；
- 自定义 rover USD/URDF；
- 低重力、摩擦和轮地接触参数化；
- 更丰富的 lunar crater / rough terrain 配置。

## 停止规则

- 如果新工作只是在相同 proxy 配置上继续拉长 PPO 预算，且不改善 checkpoint 评估闭环，暂缓。
- 如果结果只来自 GIF、TensorBoard 或训练 reward，不写成通过。
- 如果 PhysX 仍使用 Jackal placeholder，必须明确它不是最终 lunar rover asset 验收。
