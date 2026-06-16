# 工程脚手架

本文只定义项目文件结构、模块职责、脚本入口、测试结构和数据流边界。技术模型与接口见 [technical_design.md](technical_design.md)，当前路线与里程碑见 [implementation_plan.md](implementation_plan.md)，实验结论见 [experiments/README.md](experiments/README.md)。

## 顶层结构

```text
configs/                         # env / agent / reward / experiment 配置
docs/                            # 当前文档入口、规划、实验记录、runbook 和归档
scripts/                         # 训练、评估、诊断、可视化和输出整理入口
source/lunar_rover_tasks/         # 本地可安装任务包
tests/                           # 单元测试、接口契约和训练/评估 smoke 测试
outputs/                         # 生成产物目录，默认不提交
```

`docs/archive/` 只保存历史原文，不作为当前事实来源。当前事实以 `docs/current_status.md`、`docs/implementation_plan.md` 和对应实验文档为准。

## 任务包结构

核心任务位于：

```text
source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/
```

| 模块 | 职责 |
| --- | --- |
| `gathering_env_cfg.py` | 定义 simulation、task、planner、trajectory、control、terrain、reward、observation 和 state 配置。 |
| `gathering_env.py` | 组织 `MultiRoverGatheringCore`、Gym wrapper 和 SKRL wrapper；当前主训练路径是 torch proxy core。 |
| `observation.py` | 生成去中心化 actor observation；执行期不暴露 oracle 集合点。 |
| `state.py` | 生成 centralized critic state，可包含训练期 oracle 信息。 |
| `reward.py` | 计算集合、oracle 进展、能耗、安全、运动质量、一致性和终端 reward。 |
| `termination.py` | 计算成功、碰撞、超时等 episode 终止条件。 |
| `action_interpreter.py` | 将归一化 action 映射为局部子目标参数 `[rho, beta]`。 |
| `trajectory_generator.py` | 将局部子目标转换为局部参考轨迹。 |
| `simple_controller.py` | 将参考轨迹转换为 proxy 速度命令。 |
| `terrain_features.py` | 生成 procedural heightfield / crater proxy 特征和速度缩放相关查询。 |
| `communication.py` | 处理邻居筛选和局部共享状态。 |
| `oracle.py` | 计算训练期集合目标、距离和几何辅助量。 |
| `metrics.py` | 汇总 strict gate、成功率、碰撞率、超时率等指标。 |

## 配置结构

- `configs/env/`：基础环境和 terrain profile。
- `configs/agent/`：SKRL-MAPPO agent 配置。
- `configs/reward/`：reward 权重和 ablation 配置。
- `configs/experiment/`：实验级组合配置，是训练与评估脚本的主要入口。
- `configs/task/multi_rover_gathering.yaml`：任务注册和默认任务配置。

实验配置通过 `scripts/_common.py` 加载为 `MultiRoverGatheringEnvCfg`。当前 observation schema 的稳定契约见 [interface_spec.md](interface_spec.md)。

## 脚本入口

| 脚本 | 职责 |
| --- | --- |
| `scripts/train_proxy_convergence.py` | PPO / BC warm-start proxy 训练和 suite 产物生成。 |
| `scripts/train_skrl_mappo.py` | SKRL-MAPPO proxy 训练、CUDA contract、exp012 / exp013 action-scale 诊断。 |
| `scripts/evaluate_proxy_policy.py` | 独立 proxy deterministic evaluation，输出 `metrics/final_eval_proxy.json`。 |
| `scripts/run_checkpoint_evaluation.py` | 统一 checkpoint 评估入口，串联 proxy gate、PhysX 触发和 `checkpoint_status.json`。 |
| `scripts/evaluate_physx_jackal_tracking.py` | Isaac/PhysX Jackal 跟踪验证，输出 tracking metrics、timeseries 和 figures。 |
| `scripts/validate_first_stage.py` | 第一阶段环境接口和 scripted rollout smoke 验收。 |
| `scripts/debug_env.py` / `debug_observation.py` / `debug_reward.py` | 局部诊断入口。 |
| `scripts/render_skrl_proxy_rollout.py` / `view_proxy_rovers_isaac.py` | 展示与可视化入口。 |
| `scripts/organize_outputs.py` | 输出目录整理和 manifest 辅助。 |

## 测试结构

`tests/` 覆盖模块契约、配置 wiring、observation schema、reward、termination、terrain features、trajectory/control、SKRL wrapper 语义、checkpoint evaluation 和 CUDA short-training smoke。基础验收命令：

```bash
.venv_isaaclab/bin/python -m pytest -q -ra
```

## 数据流边界

```text
experiment YAML
-> MultiRoverGatheringEnvCfg
-> MultiRoverGatheringCore / SKRL wrapper
-> observation / state / action_interpreter / trajectory_generator / simple_controller
-> proxy step + reward + termination + metrics
-> checkpoint
-> run_checkpoint_evaluation.py
-> final_eval_proxy.json + checkpoint_status.json + optional PhysX tracking metrics
```

`isaaclab-multi-agent` wrapper 是 SKRL 接口适配层，不代表训练 loop 运行在 Isaac Sim / PhysX。PhysX 当前属于 checkpoint 级高保真闭环评估层，不进入每次 PPO/MAPPO 梯度更新。
