# Isaac 多车月面集合第一阶段项目

本仓库实现多月球车自组织集合任务的第一阶段工程闭环。当前主路线是：

```text
高吞吐 proxy 环境训练
-> proxy strict evaluation
-> Isaac Sim / Isaac Lab / PhysX high-fidelity closed-loop evaluation
```

训练主环境使用明确标注的 proxy rover 模型和 torch 向量化动力学核心，用于高吞吐采样、奖励调试、观测/动作接口验证和 checkpoint selection。Isaac Sim / PhysX 当前作为 checkpoint 级高保真闭环评估、迁移 sanity check 和展示层，不参与每次 PPO / MAPPO 梯度更新。

## 文档入口

先阅读：

```text
docs/README.md
docs/current_status.md
docs/implementation_plan.md
多月球车自组织集合局部参考轨迹规划技术文档.md
docs/architecture/overall_plan_v3.md
docs/experiments/README.md
```

长期技术路径管理读根目录 `多月球车自组织集合局部参考轨迹规划技术文档.md`，工程脚手架读 `docs/scaffold.md`，短版技术摘要和接口读 `docs/technical_design.md` 与 `docs/interface_spec.md`。旧 V1 / V2 / V3 原文压缩包已移出仓库，存放在仓库父目录 `../original_design_docs_v1_v2_v3_2026-06-16.zip`。训练生成产物位于 `outputs/`，并由 git 忽略；长期状态以 Markdown 实验文档、suite JSON、`final_eval_proxy.json` 和 `checkpoint_status.json` 为准。

## 环境

本地目标环境为 `.venv_isaaclab` 和 Python 3.12。Isaac stack 目标为：

- Isaac Sim 6.0.0
- Isaac Lab v3.0.0-beta
- PyTorch 2.10.0+cu128
- 通过 Isaac Lab `rl[skrl]` 安装 SKRL

## 常用命令

基础验收：

```bash
.venv_isaaclab/bin/python -m pip install -e source/lunar_rover_tasks
.venv_isaaclab/bin/python -m pytest -q -ra
```

Proxy 训练 / 诊断：

```bash
.venv_isaaclab/bin/python scripts/train_proxy_convergence.py \
  --config configs/experiment/exp_008_terrain3d_weak_warmstart.yaml \
  --output-layout run \
  --run-name <run_id> \
  --device cuda
```

Checkpoint 统一评估：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/<experiment>/<run_id>
```

## 任务 ID

Gymnasium task 注册 ID：

```text
Isaac-MultiRover-Gathering-Direct-v0
```

Actor observation 不包含 oracle 信息。oracle 集合点仅用于 centralized critic、reward shaping 和评价指标。PhysX / Jackal 评估结果只能说明 proxy checkpoint 在当前高保真 placeholder 场景中的闭环表现，不能直接写成真实月球车物理训练结果。
