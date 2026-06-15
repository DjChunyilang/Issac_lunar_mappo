# 多月球车项目 V3.0 补充说明
## Proxy 训练 + Isaac/PhysX 高保真闭环评估

## 1. 文档定位

本文是对根目录 `multi_rover_isaac_project_scaffold_v1_0.md` 和 `isaac_sim_skrl_mappo_multi_rover_tech_doc_v2_0.md` 的当前实现补充说明。V1.0 仍作为工程脚手架历史依据，V2.0 仍作为原始技术设计依据；当前代码和实验结果的解释口径以本文、`docs/current_status.md` 和 `docs/architecture/overall_plan_v3.md` 为准。

当前固定路线为：

```text
高吞吐 proxy 环境训练
-> proxy strict evaluation
-> Isaac Sim / Isaac Lab / PhysX high-fidelity closed-loop policy evaluation
```

## 2. 当前训练环境

当前主训练环境不是 Isaac Sim / PhysX 物理仿真，而是 PyTorch / torch-vectorized proxy environment。

Proxy 环境承担：

- 多智能体采样和训练吞吐；
- reward、termination、observation、critic state 和 action contract 验证；
- weak warm-start + PPO、SKRL-MAPPO CUDA contract、action-scale ablation；
- 高频 deterministic evaluation 和 checkpoint selection。

当前 proxy 动力学是 2D/2.5D kinematic unicycle 风格模型。地形开启时可使用 procedural heightfield / crater proxy 特征和速度缩放，但不包含真实质量、惯量、轮地接触、打滑、沉陷、悬挂或 PhysX contact。

## 3. Isaac/PhysX 的当前定位

Isaac Sim / Isaac Lab / PhysX 当前作为高保真闭环评估层，而不是主训练采样层。

闭环评估指策略在评估环境中连续执行：

```text
观测 -> actor 动作 -> 控制适配 -> 物理推进 -> 新观测
```

该过程不同于离线轨迹回放，也不同于单步动作打分。它用于检查 proxy checkpoint 在物理仿真、轮式资产和崎岖地形扰动下是否仍能完成集合任务。

当前 PhysX 资产使用 Jetbot placeholder。Jetbot 可验证轮式控制链路和评估流程，但不能代表最终月球车资产，也不能证明真实月面轮壤动力学已经解决。

## 4. Checkpoint 状态机

候选 checkpoint 必须通过统一评估入口生成状态文件：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/<experiment>/<run_id>
```

状态文件路径：

```text
outputs/runs/<experiment>/<run_id>/metrics/checkpoint_status.json
```

允许状态：

```text
candidate
proxy_passed
physx_evaluated
physx_passed
final_selected
```

Proxy strict gate：

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

PhysX gate 当前使用：

```text
success_rate >= 0.9
collision_rate <= 0.02
```

同时保留 `mean_final_dmax`、`mean_final_dispersion`、`mean_max_tilt_deg` 和 `mean_physics_updates_per_s` 作为诊断指标。

## 5. 对 V1.0 / V2.0 的修订关系

V1.0 / V2.0 中关于目录、观测、critic、oracle 边界、`[rho, beta]` 动作、轨迹生成和简化控制的设计仍然有效。

需要修正的是训练平台表述：

```text
旧口径：
Isaac Sim / Isaac Lab 是当前主要训练与仿真平台。

当前口径：
Proxy 是当前主要训练平台；Isaac Sim / Isaac Lab / PhysX 是 checkpoint 级高保真闭环评估平台。
```

因此，现有实验结果应写为：

```text
weak warm-start + PPO 在 proxy 环境中通过 strict gate；
候选 checkpoint 可进一步进入 PhysX / Jetbot 高保真闭环评估。
```

不要写为：

```text
MAPPO 已在 Isaac Sim / Isaac Lab 真实物理环境中完成训练收敛。
```

## 6. 当前结果口径

- `exp006` 是平地 proxy strict baseline。
- `exp008` 是当前最完整的 3-seed terrain-aware proxy baseline。
- `exp007` 证明 checkpoint 可接入 PhysX / Jetbot 闭环评估，但不代表 Isaac 物理训练。
- `exp012` / `exp013` 是 SKRL-MAPPO proxy 诊断，不是 strict pass。
- 当前较好结果来自 weak warm-start + PPO，不能表述为 pure RL 从零严格收敛。

## 7. 后续工作

1. 对当前候选 checkpoint 补齐 `checkpoint_status.json`。
2. 扩大 PhysX / Jetbot 多 episode、多地形评估样本量。
3. 记录 high-fidelity failure cases，包括集合失败、碰撞、越界、tilt、陷入坑洼、超时和控制跟踪失败。
4. 若高保真评估发现系统性迁移失败，再考虑 Isaac-based fine-tuning、domain randomization、真实 rover USD/URDF 或更高保真的动力学模型。
