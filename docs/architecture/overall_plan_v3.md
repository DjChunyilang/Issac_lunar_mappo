# 多月球车自组织集合整体规划 V3

本文档是当前工程推进的主规划。V1.0 脚手架文档和 V2.0 技术文档保留为历史设计依据；当前实现口径以根目录 `multi_rover_proxy_train_isaac_eval_supplement_v3_0.md` 和本文档为准。

核心判断：当前项目采用“高吞吐 proxy 训练 + Isaac Sim / Isaac Lab / PhysX 高保真闭环评估”的分层路线。训练样本、奖励调试、checkpoint selection 的主体来自 proxy 环境；Isaac/PhysX 不进入每次 PPO/MAPPO 梯度更新，而用于低频 checkpoint 级闭环评估、失效分析和展示。

## 分层路线

### Proxy 训练层

Proxy 环境是当前主训练环境，不是临时日志或单纯可视化工具。它负责：

- torch-vectorized 多环境采样；
- actor observation、centralized critic state、reward、termination 和 metrics 的接口验证；
- BC warm-start、PPO / SKRL-MAPPO 训练诊断和 ablation；
- 高频 deterministic eval 与 proxy strict checkpoint selection；
- 标准 `outputs/runs/<experiment>/<run_id>/` 产物管理。

当前 proxy 动力学是 2D/2.5D kinematic unicycle 风格模型。地形开启时会查询 procedural heightfield / crater proxy 特征并施加速度缩放，但不包含真实质量、惯量、轮地接触、打滑、沉陷、悬挂或 PhysX contact。

### Isaac / PhysX 高保真评估层

Isaac Sim / Isaac Lab / PhysX 当前定位为 high-fidelity closed-loop policy evaluation：

- 加载 proxy 训练得到的 checkpoint；
- 在 PhysX 轮式资产和地形中执行“观测 -> 动作 -> 物理推进 -> 再观测”的闭环 rollout；
- 报告 success、collision、dmax、dispersion、tilt 和 physics throughput；
- 记录失败案例，用于判断 proxy 策略是否存在系统性迁移问题。

当前使用 Jetbot 作为轮式资产 placeholder。Jetbot 可以验证控制链路和闭环评估流程，但不能代表最终 lunar rover asset。

### Checkpoint 流转

标准流转为：

```text
训练产生 checkpoint
-> proxy deterministic evaluation
-> proxy strict gate
-> high-fidelity PhysX evaluation queue
-> checkpoint_status.json
-> final_selected
```

checkpoint 状态只允许：

```text
candidate
proxy_passed
physx_evaluated
physx_passed
final_selected
```

对应标准入口：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/<experiment>/<run_id>
```

## 与 V2.0 的关系

V2.0 原始文档把 Isaac Sim / Isaac Lab 描述为主要训练与仿真平台。当前实现做了工程修订：

| 维度 | V2.0 原始表述 | 当前 V3 口径 |
| --- | --- | --- |
| 主训练环境 | Isaac Lab / PhysX 多车物理训练 | 高吞吐 proxy 环境 |
| 高保真仿真 | 训练 loop 的物理推进层 | checkpoint 级闭环评估层 |
| 动作接口 | `[rho, beta]` 低维局部子目标 | 保持不变 |
| 轨迹与控制 | 子目标 -> 轨迹 -> 简化控制 | 保持上层接口；PhysX 侧做轮式资产适配 |
| 结果声明 | Isaac 物理训练成功 | proxy strict pass + PhysX closed-loop eval |

该修订不是降低目标，而是让文档与当前代码事实一致：proxy 提供训练吞吐和可控实验，Isaac/PhysX 提供高保真迁移检查。

## 指标体系

Proxy strict gate：

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

High-fidelity gate 当前使用：

```text
success_rate >= 0.9
collision_rate <= 0.02
```

PhysX 评估必须同时保留诊断指标：

```text
mean_final_dmax
mean_final_dispersion
mean_max_tilt_deg
mean_physics_updates_per_s
episode_metrics
```

GIF、截图和 TensorBoard 曲线不能作为 strict pass 证据。

## 当前实现现状

- `MultiRoverGatheringCore` 是当前主训练和评估 proxy core。
- `scripts/train_proxy_convergence.py` 产生 exp006-exp010 的主要 proxy suite 结果。
- `scripts/train_skrl_mappo.py` 已接入 SKRL MAPPO proxy wrapper，用于 CUDA contract、action-scale 诊断和 exp012/exp013。
- `scripts/evaluate_proxy_policy.py` 输出独立 `metrics/final_eval_proxy.json`。
- `scripts/evaluate_physx_four_jetbots.py` 输出 PhysX / Jetbot headless 或 render 评估结果。
- `scripts/run_checkpoint_evaluation.py` 是新的 checkpoint 级统一评估入口。
- `outputs/runs/` 是标准产物目录；`outputs/**` 默认不提交。

## 当前默认下一步

1. 对 exp008 候选 checkpoint 运行统一 checkpoint evaluation，补齐 `checkpoint_status.json`。
2. 保留 exp013 作为 SKRL-MAPPO action-scale 与 reachability 诊断，不把它写成成功结果。
3. 扩大 PhysX / Jetbot 多 episode、多地形复评，记录失败类型与姿态稳定性。
4. 如果 PhysX 评估暴露系统性迁移失败，再考虑 domain randomization、Isaac-based fine-tuning 或真实 rover asset/control adapter。
