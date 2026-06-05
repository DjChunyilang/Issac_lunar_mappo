# Isaac 多车月面集合第一阶段项目

本仓库实现两个设计文档中的第一阶段脚手架。当前实现使用明确标注的 proxy rover 模型和 torch 向量化动力学核心，用于在真实 rover USD/URDF articulation 可用前，先验证观测、动作、reward、终止条件和训练闭环。

## 文档入口

先阅读：

```text
docs/README.md
docs/current_status.md
docs/experiments/README.md
```

长篇历史进度日志位于 `docs/archive/`。训练生成产物位于 `outputs/`，并由 git 忽略；长期状态以 Markdown 实验文档和 suite JSON 摘要为准。

## 环境

本地目标环境为 `.venv_isaaclab` 和 Python 3.12。Isaac stack 目标为：

- Isaac Sim 6.0.0
- Isaac Lab v3.0.0-beta
- PyTorch 2.10.0+cu128
- 通过 Isaac Lab `rl[skrl]` 安装 SKRL

## 第一阶段命令

```bash
.venv_isaaclab/bin/python -m pip install -e source/lunar_rover_tasks
.venv_isaaclab/bin/python -m pytest
.venv_isaaclab/bin/python scripts/debug_env.py --steps 200
.venv_isaaclab/bin/python scripts/debug_observation.py
.venv_isaaclab/bin/python scripts/debug_reward.py
.venv_isaaclab/bin/python scripts/train.py --config configs/experiment/exp_001_minimal.yaml --device cpu --timesteps 128
```

`scripts/train.py` 默认使用真实 SKRL `MAPPO` backend。紧凑本地 trainer 仅用于快速调试，可通过 `--backend smoke` 启用。

## 任务 ID

Gymnasium task 注册 ID：

```text
Isaac-MultiRover-Gathering-Direct-v0
```

Actor observation 不包含 oracle 信息。初始几何中位点仅作为第一阶段训练中的 centralized value / reward shaping oracle point。
