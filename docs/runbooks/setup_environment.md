# 环境搭建与工程闭环验收

本 runbook 用于验证 Isaac Sim / Isaac Lab / SKRL / 本地任务包的最小工程闭环。近期目标是确认安装、导入、proxy validation、SKRL MAPPO smoke 和 PhysX sanity 路径可重复，不以 reward 收敛作为成功标准。

## 前置条件

- Linux + NVIDIA GPU 工作站。
- NVIDIA driver 与 CUDA 12 系列 PyTorch wheel 兼容。
- Python 虚拟环境路径固定为 `.venv_isaaclab/`。
- 预留足够磁盘空间给 Isaac Sim、Isaac Lab、extscache、TensorBoard 和 outputs。
- 网络可访问 PyTorch、NVIDIA PyPI 和 GitHub。

## 安装入口

现有安装脚本：

```bash
scripts/install_stack.sh
```

脚本执行以下工作：

- 安装 `torch==2.10.0` 和 `torchvision==0.25.0` 的 CUDA 12.8 wheel。
- 安装 `isaacsim[all,extscache]==6.0.0`。
- clone `external/IsaacLab` 的 `v3.0.0-beta` 分支。
- editable install `isaaclab`、`isaaclab_assets`、`isaaclab_tasks` 和 `isaaclab_rl[skrl]`。
- editable install `source/lunar_rover_tasks`。

如果 `.venv_isaaclab/` 不存在，先创建虚拟环境并升级基础工具：

```bash
python -m venv .venv_isaaclab
.venv_isaaclab/bin/python -m pip install --upgrade pip wheel
```

然后运行：

```bash
bash scripts/install_stack.sh
```

## 导入检查

安装完成后先做最小导入检查：

```bash
.venv_isaaclab/bin/python -c "import torch, isaacsim, skrl; import lunar_rover_tasks; print(torch.__version__)"
```

再确认本地任务包来自 editable install：

```bash
.venv_isaaclab/bin/python -c "import lunar_rover_tasks, pathlib; print(pathlib.Path(lunar_rover_tasks.__file__).resolve())"
```

期望路径位于：

```text
source/lunar_rover_tasks/lunar_rover_tasks/__init__.py
```

## Proxy Core 验收

第一阶段验证脚本用于检查 proxy core、观测、critic state、reward、终止、轨迹、控制和可视化产物：

```bash
.venv_isaaclab/bin/python scripts/validate_first_stage.py \
  --config configs/experiment/exp_001_minimal.yaml \
  --device cpu \
  --steps 32
```

成功标准：

- 命令退出码为 0。
- 生成 `validation_metrics.json`。
- 生成 rollout 曲线、观测 heatmap、轨迹控制图、高度图和 GIF。
- 不要求 reward 收敛。

## SKRL MAPPO Smoke

使用 SKRL wrapper 做最短 MAPPO smoke：

```bash
.venv_isaaclab/bin/python scripts/train.py \
  --backend skrl \
  --config configs/experiment/exp_001_minimal.yaml \
  --device cpu \
  --timesteps 128
```

成功标准：

- 命令退出码为 0。
- 能创建 SKRL MAPPO trainer 并完成短训练。
- 能保存 smoke checkpoint。
- 不把该 checkpoint 记录为正式实验通过结果。

如果要排查本地 trainer，可运行：

```bash
.venv_isaaclab/bin/python scripts/train.py \
  --backend smoke \
  --config configs/experiment/exp_001_minimal.yaml \
  --device cpu
```

## Debug Smoke

基础回归命令：

```bash
.venv_isaaclab/bin/python scripts/debug_env.py --steps 50
.venv_isaaclab/bin/python scripts/debug_observation.py
.venv_isaaclab/bin/python scripts/debug_reward.py
```

成功标准：

- 命令退出码为 0。
- actor observation 和 critic state 维度符合 `docs/interface_spec.md`。
- reward、done、metrics 中没有 NaN 或 inf。

## PhysX Sanity

PhysX / Isaac Sim 目前作为验证和展示层，不进入主训练 loop。headless sanity 示例：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --config configs/experiment/exp_008_terrain3d_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_008_terrain3d/_suite/checkpoints/seed_23_best.pt \
  --terrain lunar_crater \
  --episodes 1 \
  --steps 60 \
  --run-dir outputs/runs/env_smoke/physx_seed23_headless
```

渲染 sanity 示例：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --config configs/experiment/exp_008_terrain3d_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_008_terrain3d/_suite/checkpoints/seed_23_best.pt \
  --terrain lunar_crater \
  --episodes 1 \
  --steps 60 \
  --render \
  --run-dir outputs/runs/env_smoke/physx_seed23_render
```

成功标准：

- headless/render 命令退出码为 0。
- metrics、figures、videos 写入对应 `run-dir`。
- 结果只作为 sanity check，不作为 strict proxy 验收。

## 结果记录规则

- `exp008` 仍是当前推荐的 3-seed terrain-aware proxy baseline。
- `exp009` 和 `exp010` 作为 strong terrain 诊断记录，近期不继续扩展。
- 新 smoke 优先写入 `outputs/runs/`。
- 不要从 GIF、截图、TensorBoard 曲线或 partial train run 推断 strict pass。
