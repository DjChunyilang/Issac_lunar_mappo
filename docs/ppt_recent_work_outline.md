# 近期工作汇报 PPT 提纲

本文档用于制作导师要求的近期工作 PPT。建议控制在 12 页左右，主线是“目标与技术路线 -> 已完成算法与工程实现 -> 实验结果 -> 问题诊断 -> 下一步计划”。

严格结果以 `outputs/runs/**/metrics/*.json` 和实验文档为准。GIF、截图和曲线只作为展示素材，不单独作为通过依据。

## 第 1 页：标题页

**标题**

多月球车自组织集合任务近期进展汇报

**副标题**

Proxy 训练基线、三维地形扩展、强地形诊断与 Isaac Lab 工程回归路线

**图片**

- 首选：`outputs/figures/physx_four_jetbots/evaluation_exp007_lunar_crater_scene.png`
- 备选：`outputs/figures/isaac_render/proxy_rovers_scene.png`

**页面文字**

- 任务：多 rover 在未知初始分布下自主集合。
- 方法：低维局部子目标动作 + 去中心化 actor + centralized critic。
- 当前成果：弱 lunar crater 3D proxy 上完成 3-seed strict baseline。
- 当前方向：从 proxy 结果回归 Isaac Sim / Isaac Lab + SKRL-MAPPO 工程闭环。

## 第 2 页：研究问题与任务定义

**标题**

多 rover 自组织集合任务

**图片**

- 建议画一张任务示意图：4 台 rover 从分散位置移动到集合区域，标注 `dmax`、`dispersion`、`collision`、`timeout`。
- 可用现有 GIF 截帧：`outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_8m_lunar_crater_cpu/videos/proxy_eval_rollout.gif`

**页面文字**

- 输入：每台 rover 的局部状态、邻居相对信息、局部地形特征。
- 输出：每台 rover 的低维局部子目标动作 `[rho, beta]`。
- 目标：缩小队形最大距离 `dmax` 与离散度 `dispersion`，同时避免碰撞和超时。
- 验收指标：
  - `dmax_reduction_ratio <= 0.2`
  - `success_rate >= 0.9`
  - `collision_rate <= 0.02`
  - `timeout_rate == 0`

**讲解重点**

强调不是单车导航，而是多智能体协作集合；成功判定同时看集合质量、安全性和任务完成时间。

## 第 3 页：总体技术路线

**标题**

从快速 proxy 验证到 Isaac Lab 物理闭环

**图片**

- 建议画流程图：
  - `Actor observation / Critic state`
  - `MAPPO / PPO training`
  - `[rho, beta] local goal`
  - `trajectory generator`
  - `velocity tracking controller`
  - `proxy dynamics / Isaac Lab PhysX`
  - `metrics + outputs/runs`

**页面文字**

- Actor：执行期只使用局部观测，不使用 oracle 集合点。
- Critic：训练期使用全局状态、队形几何、地形摘要和 oracle 辅助信息。
- 动作：策略输出归一化 `[rho, beta]`，映射为车体系局部子目标。
- 控制：局部子目标 -> 确定性轨迹 -> 速度跟踪命令。
- 工程路线：proxy 用于快速训练和接口验证，最终回到 Isaac Lab + SKRL-MAPPO。

**参考文档**

- `docs/architecture/overall_plan_v3.md`
- `docs/interface_spec.md`

## 第 4 页：算法与环境实现

**标题**

当前已实现的训练闭环

**图片**

- 建议画模块框图，分成 5 个模块：
  - `MultiRoverGatheringCore`
  - observation/state 构造
  - reward/termination
  - PPO + BC warm-start
  - evaluation + strict gate

**页面文字**

- 实现了 torch-vectorized proxy 环境，支持并行环境训练与评估。
- Actor observation 包含 ego、neighbor、terrain、aggregation 特征。
- Critic state 包含全局 rover 状态、团队几何统计、地形摘要和 oracle 信息。
- Reward 覆盖集合进展、安全距离、运动约束和终端成功。
- 输出统一写入 `outputs/runs/<experiment>/<run_id>/`。

**讲解重点**

说明 proxy 不是最终物理环境，但已经把算法接口、奖励、终止、评估和输出管理串起来了。

## 第 5 页：训练方法演进

**标题**

从平地 baseline 到三维地形 weak warm-start

**图片**

- 主图：`outputs/runs/exp_008_terrain3d/_suite/figures/comparison_curves.png`
- 可辅图：`outputs/logs/exp_006_ppo_selected/comparison_curves.png`

**页面文字**

| 阶段 | 实验 | 地形 | 方法 | 结论 |
| --- | --- | --- | --- | --- |
| 平地基线 | exp006 | flat proxy | BC + PPO | 3 seeds 通过 |
| 弱三维地形 | exp008 | weak lunar crater 3D proxy | weak warm-start + PPO | 3 seeds 通过 |
| 强三维地形 | exp009/010 | strong lunar crater 3D proxy | PPO retry + hold reward 诊断 | 未形成 3-seed strict |

**讲解重点**

纯 RL 在当前预算下不稳定，weak warm-start 明显提升了可训练性；PPO 阶段仍负责最终 checkpoint 选择。

## 第 6 页：三维地形建模

**标题**

Terrain-aware 3D proxy：从平面到 lunar crater

**图片**

- 弱地形图：`outputs/runs/exp_008_terrain3d/_suite/figures/terrain_height_map.png`
- 强地形图：`outputs/runs/exp_009_terrain3d_strong/_suite/figures/terrain_height_map.png`

**页面文字**

弱 lunar crater 3D proxy：

- `height_range ~= 0.241 m`
- `roughness_max ~= 0.360`
- `traversability_min ~= 0.549`

强 lunar crater 3D proxy：

- `height_range ~= 0.740 m`
- `roughness_max ~= 1.057`
- `traversability_min ~= 0.096`
- `mean_terrain_speed_scale ~= 0.393`

**讲解重点**

地形不是只作为背景图，而是进入动力学和观测特征；强地形显著降低可通行性与速度尺度。

## 第 7 页：当前主结果 exp008

**标题**

弱 lunar crater 3D proxy 上的 3-seed strict baseline

**图片**

- 主图：`outputs/runs/exp_008_terrain3d/_suite/figures/comparison_curves.png`
- 可插入右下角动图/截帧：`outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_8m_lunar_crater_cpu/videos/proxy_eval_rollout.gif`

**页面文字**

| seed | dmax ratio | success | collision | timeout | strict |
| --- | ---: | ---: | ---: | ---: | --- |
| 23 | 0.1539 | 1.0000 | 0.0000 | 0.0000 | 通过 |
| 31 | 0.1345 | 0.9961 | 0.0049 | 0.0000 | 通过 |
| 47 | 0.1560 | 1.0000 | 0.0000 | 0.0000 | 通过 |

**结论文字**

exp008 是当前最完整、最可信的 terrain-aware proxy baseline，推荐 checkpoint 位于：

```text
outputs/runs/exp_008_terrain3d/_suite/checkpoints/
```

## 第 8 页：强地形实验 exp009

**标题**

强 lunar crater 下暴露出稳定性问题

**图片**

- 主图：`outputs/runs/exp_009_terrain3d_strong/_suite/figures/comparison_curves.png`
- 辅图：`outputs/runs/exp_009_terrain3d_strong/_suite/figures/terrain_height_map.png`

**页面文字**

| seed | dmax ratio | success | collision | timeout | strict |
| --- | ---: | ---: | ---: | ---: | --- |
| 23 | 0.1473 | 1.0000 | 0.0000 | 0.0000 | 通过 |
| 31 | 0.1819 | 0.8740 | 0.0049 | 0.1250 | 未通过 |
| 47 | - | - | - | - | 未运行 |

**结论文字**

- 强地形不是完全不可解：seed23 可以通过。
- 但 seed31 未通过 `success_rate` 和 `timeout_rate` gate。
- seed31 失败后，3-seed strict 已不可能成立，因此没有继续 seed47。

## 第 9 页：exp010 失败诊断

**标题**

失败主要来自强地形下未及时进入 dmax / dispersion 成功区

**图片**

- 主图：`outputs/runs/exp_010_strong_success_diagnostics/hold_reward_seed31_6m_cont1_from_4m_cuda/figures/convergence_curves.png`
- 辅图：`outputs/runs/exp_010_strong_success_diagnostics/hold_reward_seed31_6m_cont1_from_4m_cuda/figures/safety_diagnostics.png`

**页面文字**

exp009 seed31 failed：

- `success_rate = 0.8740`
- `timeout_rate = 0.1250`
- `speed_ok_rate ~= 0.9998`

hold reward 6M continuation：

- `dmax_reduction_ratio = 0.1720`，通过。
- `success_rate = 0.9014`，通过。
- `collision_rate = 0.0273`，未通过。
- `timeout_rate = 0.0742`，未通过。

**结论文字**

失败不是简单的 speed hold 问题。继续同类 PPO 续训能推高 success，但不能清除 timeout 和 collision，需要改动作表示、控制接口或 curriculum。

## 第 10 页：工程展示与 PhysX sanity

**标题**

策略展示链路已接入 PhysX sanity，但还不是主训练环境

**图片**

- 主图：`outputs/figures/physx_four_jetbots/evaluation_exp007_lunar_crater_scene.png`
- 可用视频：`outputs/videos/physx_four_jetbots/evaluation_exp007_lunar_crater_rollout.gif`

**页面文字**

- 已有 Isaac Sim / PhysX Jetbot 展示路径，用于 sanity check 和可视化。
- 当前 Jetbot 场景不是最终 lunar rover articulation。
- PhysX 展示可以说明策略接口可接入渲染/物理展示链路。
- 不能把 PhysX showcase 等同于 Isaac Lab 物理训练 strict pass。

**讲解重点**

这一页要主动说明边界，避免导师误解“已经完成 Isaac Lab 物理训练”。

## 第 11 页：当前差距与 V3 归正路线

**标题**

从 proxy baseline 回到 Isaac Lab + SKRL-MAPPO

**图片**

- 建议画三层路线图：
  - 接口稳定层：observation/state/action/reward/outputs
  - Isaac Lab 物理环境层：rover articulation、terrain、collision、contact
  - SKRL-MAPPO 正式训练层：runner、memory、centralized critic、final eval

**页面文字**

当前差距：

- 已验证结果来自 proxy PPO + BC warm-start，不是正式 Isaac Lab 物理训练。
- SKRL-MAPPO 当前主要是 smoke 路径，还不是主训练结果来源。
- PhysX 使用 Jetbot 展示，尚未替换为 lunar rover articulation。

下一步归正：

- 固化 `.venv_isaaclab`、Isaac Sim、Isaac Lab、SKRL 和本地任务包安装。
- 跑通 proxy validation、SKRL MAPPO smoke 和 PhysX headless/render sanity。
- 新建 Isaac Lab 多 rover task skeleton。
- 设计 rover asset 与 control adapter。

## 第 12 页：总结与下一步计划

**标题**

阶段结论与后续工作

**图片**

- 左侧：exp008 结果表或 `comparison_curves.png`
- 右侧：V3 路线图或 PhysX 场景图

**页面文字**

已完成：

- 建立多 rover proxy 训练、评估、可视化和输出管理闭环。
- 完成平地 exp006 strict baseline。
- 完成弱 lunar crater 3D proxy exp008 3-seed strict baseline。
- 完成强地形 exp009/exp010 失败诊断，明确当前 reward/control 方向不足。

下一步：

- 不继续无界堆 strong terrain PPO。
- 优先完成 Isaac Sim / Isaac Lab / SKRL 工程闭环验收。
- 将 SKRL-MAPPO 从 smoke 升级为正式训练入口。
- 在真实 Isaac/PhysX 环境稳定后，再恢复 weak/strong terrain 训练实验。

## 备选附录 A：实验结果总表

| 实验 | 地形 | 方法 | seeds | strict | 展示重点 |
| --- | --- | --- | --- | --- | --- |
| exp006 | 平地 proxy | BC + PPO | 23, 31, 47 | 通过 | 平地 baseline |
| exp008 | 弱 lunar crater 3D proxy | weak warm-start + PPO | 23, 31, 47 | 通过 | 当前主结果 |
| exp009 | 强 lunar crater 3D proxy | weak warm-start + PPO retry | 23, 31；47 未运行 | 未通过 | seed31 success/timeout 失败 |
| exp010 | 强地形诊断 | hold reward / safety retry | seed31 continuation | 未通过 | failure mode 定位 |

## 备选附录 B：可直接使用的素材清单

**主结果素材**

- `outputs/runs/exp_008_terrain3d/_suite/figures/comparison_curves.png`
- `outputs/runs/exp_008_terrain3d/_suite/figures/terrain_height_map.png`
- `outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_8m_lunar_crater_cpu/videos/proxy_eval_rollout.gif`
- `outputs/runs/exp_008_terrain3d/weak_warmstart_completion_seed31_4m_evalseed0_cpu/videos/proxy_eval_rollout.gif`
- `outputs/runs/exp_008_terrain3d/weak_warmstart_select_seed47_8m_lunar_crater_cpu/videos/proxy_eval_rollout.gif`

**强地形诊断素材**

- `outputs/runs/exp_009_terrain3d_strong/_suite/figures/comparison_curves.png`
- `outputs/runs/exp_009_terrain3d_strong/_suite/figures/terrain_height_map.png`
- `outputs/runs/exp_009_terrain3d_strong/weak_warmstart_seed31_retry20m_safe090_strong_lunar_crater_cuda_eval1024/figures/safety_diagnostics.png`
- `outputs/runs/exp_010_strong_success_diagnostics/hold_reward_seed31_6m_cont1_from_4m_cuda/figures/convergence_curves.png`
- `outputs/runs/exp_010_strong_success_diagnostics/hold_reward_seed31_6m_cont1_from_4m_cuda/figures/safety_diagnostics.png`

**展示素材**

- `outputs/figures/physx_four_jetbots/evaluation_exp007_lunar_crater_scene.png`
- `outputs/videos/physx_four_jetbots/evaluation_exp007_lunar_crater_rollout.gif`
- `outputs/figures/isaac_render/proxy_rovers_scene.png`
- `outputs/figures/first_stage_validation/trajectory_control_validation.png`
- `outputs/figures/first_stage_validation/proxy_rollout_curves.png`
- `outputs/figures/first_stage_validation/observation_space_heatmap.png`

## 备选附录 C：建议补画的 PPT 图

1. **任务定义图**：4 rover 分散到集合，标注 `dmax`、`dispersion`、collision radius、timeout。
2. **算法闭环图**：observation/state -> policy/value -> `[rho, beta]` -> trajectory -> controller -> environment -> metrics。
3. **V3 路线图**：proxy baseline、Isaac Lab task、rover articulation、SKRL-MAPPO、PhysX validation。
4. **实验演进时间线**：exp006 平地通过 -> exp008 弱三维地形通过 -> exp009/010 强地形诊断 -> 工程回归。

