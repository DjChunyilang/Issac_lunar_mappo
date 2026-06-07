# 多月球车自组织集合整体规划 V3

本文档是在审查当前代码实现后，对 `isaac_sim_skrl_mappo_multi_rover_tech_doc_v2_0.md` 的重新整理。V2.0 仍作为原始技术路线和历史设计依据保留；本文档作为后续工程推进的主规划。

核心判断：当前 `proxy` 实现是偏离原始路线的临时工程绕路。它验证了观测、状态、动作、奖励、训练评估和 outputs 管理等接口，但不能替代最终的 `Isaac Sim / Isaac Lab + SKRL-MAPPO + rover articulation` 训练闭环。后续主线应回到 Isaac Lab 物理环境和 SKRL-MAPPO。

## V2.0 原始目标

原始技术路线定义为：

```text
Isaac Sim / Isaac Lab
+ SKRL-MAPPO
+ 低维局部子目标动作
+ 确定性轨迹生成器
+ 简化速度跟踪控制器
```

原计划的闭环为：

1. Isaac Sim 负责 rover 实体、地形、碰撞、接触和物理推进。
2. Isaac Lab 任务环境负责 reset / step、观测、动作解释、奖励、终止和并行环境。
3. SKRL-MAPPO 负责 centralized training with decentralized execution。
4. Actor 执行期只看局部观测，不看 oracle 集合点。
5. Critic 训练期可看全局状态、团队几何统计、地形摘要和 oracle 辅助信息。
6. Actor 输出 `[rho, beta]` 低维局部子目标动作。
7. 确定性轨迹生成器把子目标变成局部参考轨迹。
8. 简化控制层输出 `[v_cmd, omega_cmd]`，后续替换为 rover articulation 控制接口。

第一阶段原目标不是追求复杂轨迹或精细对接，而是跑通最小 Isaac Lab + SKRL-MAPPO 物理训练闭环。

## 当前代码实现现状

当前仓库已经实现了一套完整的 proxy planning/training 工程链路：

- `MultiRoverGatheringCore` 是 torch-vectorized proxy 环境，负责 reset、step、观测、reward、termination、terrain dynamics 和 metrics。
- Actor observation 包含 ego、neighbor、terrain、aggregation 特征，不包含 oracle。
- Critic state 包含全部 rover 状态、团队几何统计、地形摘要和 oracle 特征。
- 动作接口为归一化 `[rho, beta]`，再映射为车体系局部子目标。
- 轨迹生成器目前只支持 deterministic line trajectory。
- 控制器是简化 velocity tracking，最终由 proxy unicycle integration 推进状态。
- 地形是 procedural heightfield / lunar crater proxy，特征为 `height`、`slope_x`、`slope_y`、`roughness`、`traversability`。
- `scripts/train_proxy_convergence.py` 提供 BC warm-start + 自写 PPO 的主实验流程，产生了 exp006-exp010 的主要结果。
- `scripts/train_skrl_mappo.py` 和 `MultiRoverGatheringSKRLEnv` 提供 SKRL MAPPO smoke wrapper，但不是当前已验证实验结果的主训练来源。
- `scripts/evaluate_physx_four_jetbots.py` 使用 Isaac Sim / PhysX Jetbot 场景做 sanity check 和展示，不是主训练环境。
- `outputs/runs/` 已成为主要结果目录，严格结论以 `_suite/metrics/strict_acceptance.json` 和独立 `final_eval_proxy.json` 为准。

当前推荐 baseline 仍是：

```text
outputs/runs/exp_008_terrain3d/_suite/
```

该结果证明弱 lunar crater 3D proxy 上的 3-seed terrain-aware baseline 可通过 strict gate，但不证明 Isaac Lab 物理训练闭环已经完成。

## 主要偏差 Review

| 维度 | V2.0 规划 | 当前实现 | 风险 / 影响 | 后续归正方向 |
| --- | --- | --- | --- | --- |
| 训练平台 | Isaac Lab 多智能体物理训练环境 | torch-vectorized `MultiRoverGatheringCore` proxy | 训练结果不能直接代表 Isaac Sim 物理真实行为 | 建立 Isaac Lab Direct/manager task，把 proxy core 降级为接口验证层 |
| 算法主线 | SKRL-MAPPO | 可靠结果来自自写 PPO + BC warm-start；SKRL 仅 smoke | 与目标算法库、runner、checkpoint 格式和日志生态不一致 | 以 SKRL-MAPPO 作为正式训练入口，proxy PPO 只保留为 baseline/debug |
| 物理实体 | rover USD/URDF articulation | 无真实 rover asset；PhysX 使用 Jetbot | 控制、碰撞、能耗、地形接触和姿态风险不可等价 | 明确 rover asset/control metadata，接入 articulation |
| 控制接口 | `[v_cmd, omega_cmd]` 经 adapter 转为 articulation 控制 | proxy unicycle integration 直接更新状态 | 无轮速/转向/力矩约束，动力学过简化 | 实现 wheel/articulation adapter，保留同一 planner action contract |
| 轨迹生成 | 带高程、时间戳、参考速度的局部参考轨迹 | line trajectory；高程主要由 proxy terrain dynamics 更新 | 轨迹层对地形约束和曲率选择不足 | 加入 terrain height query、arc/Bezier 选项和可检查的 trajectory validity |
| 网络结构 | 四分支 actor encoder + centralized value | 简单 MLP actor/critic | 无法体现 ego/neighbor/terrain/aggregation 的结构归纳 | SKRL 正式模型采用分支 encoder 或明确说明使用 flat MLP 的理由 |
| 地形来源 | Isaac Sim terrain geometry / height field | procedural heightfield/crater proxy features | proxy 地形难以覆盖真实接触和障碍 | PhysX/Isaac terrain 作为训练环境来源，proxy 只做快速回归 |
| 奖励与成本 | gather/oracle/energy/safety/motion/consistency/terminal | 核心 gather/oracle/safety/motion/terminal 已有；物理能耗、障碍接触成本缺失 | reward 在真实物理下可能重排，强地形失败结论不可直接迁移 | 先固定任务级几何奖励，再接入物理接触、倾覆、能耗和障碍成本 |
| 成功判据 | dmax、dispersion、speed hold | 已实现 dmax/dispersion/speed/hold diagnostics | proxy strict pass 可能高估真实物理成功率 | Isaac Lab 物理评估必须重新定义 strict acceptance |
| 输出管理 | 可追踪实验输出 | `outputs/runs` 基本建立，旧 `outputs/logs` 仍有兼容路径 | 新旧路径并存增加结果误读风险 | 新工作默认 `outputs/runs/<experiment>/<run_id>/` |

## V3 重新规划路线

V3 的核心原则是：保留 proxy 成果作为接口验证和 baseline，不再把 proxy 训练当作最终主线。

后续主线分三层推进：

1. **接口稳定层**
   - 保持 actor observation、critic state、`[rho, beta]` action、reward terms、success gates 的形状和语义稳定。
   - 保持 `outputs/runs` 结果规范。
   - proxy 环境只用于 CPU/GPU 快速回归和结果对照。

2. **Isaac Lab 物理环境层**
   - 新建真实 Isaac Lab 多 rover gathering task。
   - 将 rover USD/URDF asset、terrain、collision、contact、pose/velocity 读取和 reset 逻辑放入 Isaac Lab 环境。
   - 使用 adapter 把 `[v_cmd, omega_cmd]` 转为 rover articulation 控制目标。
   - 让 PhysX 评估从 showcase 变成训练环境同源的 validation。

3. **SKRL-MAPPO 正式训练层**
   - `scripts/train_skrl_mappo.py` 从 smoke 升级为主训练入口。
   - 使用 centralized critic state，actor 执行期不接收 oracle。
   - 输出 checkpoint、metrics、TensorBoard、final eval 和 suite summary 到标准 run layout。
   - proxy PPO/warm-start 只作为 teacher、debug 或 baseline，不再作为正式结论来源。

## 阶段里程碑

### M0 环境与文档闭环

目标：确认依赖、入口文档和 smoke 命令可重复。

验收：

```bash
.venv_isaaclab/bin/python -c "import torch, isaacsim, skrl; import lunar_rover_tasks"
.venv_isaaclab/bin/python scripts/validate_first_stage.py --config configs/experiment/exp_001_minimal.yaml --device cpu --steps 32
.venv_isaaclab/bin/python scripts/train.py --backend skrl --config configs/experiment/exp_001_minimal.yaml --device cpu --timesteps 128
```

不把该阶段产生的 checkpoint 写成训练成功。

### M1 Isaac Lab 环境骨架

目标：建立真实 Isaac Lab task skeleton，不要求策略收敛。

必须具备：

- 4 个 rover articulation 实例可 reset。
- 地形和场景可加载。
- actor observation / critic state shape 与 proxy contract 对齐。
- action 输入仍为 `[rho, beta]`。
- 环境 step 可推进物理并返回 reward/done/info。

验收：

- headless smoke 可运行固定步数。
- observation、state、action、reward、done 无 NaN。
- 产物写入 `outputs/runs/env_smoke/<run_id>/`。

### M2 Rover Asset 与 Control Adapter

目标：替换 proxy unicycle 为 articulation control adapter。

必须具备：

- 明确 rover USD/URDF 路径和关节命名。
- 明确控制接口：轮速、转向角 + 前向速度、关节速度或力矩。
- `simple_controller` 输出仍保持 `[v_cmd, omega_cmd]`，adapter 负责转换。
- 记录轮速/关节命令、倾角、接触/碰撞和越界信息。

验收：

- scripted gather policy 在 flat terrain 可稳定减少 dmax。
- collision/out_of_bounds/tilt failure 可被正确记录。
- PhysX render/headless metrics 路径与 run layout 一致。

### M3 SKRL-MAPPO 主训练

目标：把 SKRL-MAPPO 从 smoke 入口升级为正式训练入口。

必须具备：

- 使用 SKRL MAPPO rollout、memory、model、trainer。
- actor 不接收 oracle。
- critic 使用 centralized state。
- 支持 `--run-dir` / `--run-name`，输出标准 `metrics/`、`checkpoints/`、`tensorboard/`。
- 支持独立 final eval，写入 `metrics/final_eval_proxy.json` 或后续 `metrics/final_eval_physx.json`。

验收：

- flat terrain smoke 可以完成短训练并保存 checkpoint。
- 独立 eval 能读取 checkpoint 并输出机器可读 JSON。
- 不以训练 reward 或单个 GIF 判定通过。

### M4 物理验收与实验恢复

目标：在真实 Isaac/PhysX 环境中恢复实验路线。

顺序：

1. flat terrain scripted baseline。
2. flat terrain SKRL-MAPPO smoke。
3. weak terrain Isaac/PhysX eval。
4. weak terrain SKRL-MAPPO suite。
5. strong terrain 只在前面稳定后恢复。

验收：

- 每个实验有独立 `docs/experiments/exp_*.md`。
- strict acceptance 只来自 suite JSON 和独立 final eval JSON。
- proxy exp008 可作为对照，但不能替代物理环境 strict pass。

## 不应误判的结果

- `exp008` 是当前最好的 proxy baseline，不是 Isaac Lab 物理训练完成证明。
- `exp009/exp010` 是 strong terrain proxy 诊断记录，近期暂缓，不应继续堆 long-budget PPO。
- `scripts/train.py --backend skrl` 当前是 smoke，不是正式训练结果。
- PhysX Jetbot showcase 证明策略可被接入物理展示链路，但 Jetbot 不是最终 lunar rover asset。
- GIF、截图、TensorBoard 曲线和 partial run 不能作为 strict pass 证据。

## 当前默认下一步

1. 完成环境搭建与 smoke 验收，确保 `setup_environment.md` 中命令可重复。
2. 设计 Isaac Lab task skeleton，明确它如何复用当前 observation/state/reward/action 模块。
3. 决定 rover asset 和 control adapter 最小接口。
4. 将 SKRL-MAPPO 输出迁入标准 `outputs/runs` layout。
5. 在物理闭环可运行后，再恢复训练实验和 strong terrain 研究。
