# 构建进度记录（2026-05-19）

## 当前结论

项目已经完成第一阶段最小闭环构建：基于代理 rover 状态模型，跑通了 4 车自组织集合任务的观测、critic state、`[rho, beta]` 动作解释、确定性轨迹生成、简化速度控制、几何聚集奖励、oracle reward、终止判据、单元测试、环境冒烟测试和 SKRL-MAPPO 短训练。

当前实现是第一阶段可训练代理环境，不是最终真实 Isaac Sim rover articulation 环境。真实 USD/URDF 月球车资产、轮速/转向/力矩接口和复杂轮地接触仍未接入。

## 虚拟环境与安装方式

虚拟环境路径：

```bash
/home/u24/WYR/Issac_sim_lunar_mappo/.venv_isaaclab
```

这是一个 **conda prefix 环境**，不是普通 `python -m venv` 环境。创建方式为在项目目录下用 `conda create -p ... python=3.12` 创建本地隔离环境。这样不会污染 conda `base`，也不是 conda 命名环境。

原因：系统 `/usr/bin/python3.12` 可用，但缺少 `ensurepip`，直接创建标准 venv 时失败；因此改用项目内 conda prefix 环境。

当前 Python：

```text
Python 3.12.13
```

可直接使用：

```bash
.venv_isaaclab/bin/python <command>
```

或激活：

```bash
conda activate /home/u24/WYR/Issac_sim_lunar_mappo/.venv_isaaclab
```

## 已安装栈

已确认 import 的版本：

```text
isaacsim==6.0.0.0
isaaclab==4.5.22
skrl==2.1.0
torch==2.10.0+cu128
gymnasium==1.2.1
lunar-rover-tasks==0.1.0
```

Isaac Lab 源码克隆位置：

```text
external/IsaacLab
```

Isaac Sim EULA 已在首次 import 时接受。CUDA 可用，GPU 为 NVIDIA GeForce RTX 5090。

## 已完成的项目修改

新增并安装了本项目 extension：

```text
source/lunar_rover_tasks
```

任务注册 ID：

```text
Isaac-MultiRover-Gathering-Direct-v0
```

核心模块已实现：

- `gathering_env.py`：torch 向量化代理环境、Gymnasium wrapper、SKRL multi-agent wrapper
- `gathering_env_cfg.py`：任务、仿真、规划、控制、奖励、终止配置
- `observation.py` / `state.py`：actor observation 与 centralized critic state
- `action_interpreter.py`：归一化动作到 `[rho, beta]`、局部/世界子目标
- `trajectory_generator.py`：第一阶段直线轨迹生成
- `simple_controller.py`：简化速度跟踪控制器
- `reward.py` / `termination.py`：奖励分项和成功/失败判据
- `communication.py` / `terrain_features.py` / `oracle.py` / `metrics.py`：邻居共享、平面地形特征、几何中位点 oracle、团队指标

新增配置：

```text
configs/env/
configs/task/
configs/reward/
configs/agent/
configs/experiment/
```

新增脚本：

```text
scripts/debug_env.py
scripts/debug_observation.py
scripts/debug_reward.py
scripts/train.py
scripts/train_skrl_mappo.py
scripts/play.py
scripts/validate_first_stage.py
scripts/benchmark_cuda_short_training.py
scripts/install_stack.sh
```

`scripts/train.py` 当前默认走真实 SKRL `MAPPO` 后端；`--backend smoke` 是保留的轻量调试训练器。

新增测试：

```text
tests/test_action_interpreter.py
tests/test_cuda_short_training.py
tests/test_four_rover_observation_space.py
tests/test_trajectory_generator.py
tests/test_termination.py
tests/test_reward.py
tests/test_observation.py
tests/test_proxy_rover_model.py
tests/test_trajectory_control.py
```

## 当前验证结果

已通过：

```bash
.venv_isaaclab/bin/python -m pytest
```

结果：

```text
12 passed, 1 skipped
```

其中 skipped 项为 `tests/test_cuda_short_training.py` 在 Codex 默认工作区沙箱不可见 CUDA 时的条件跳过。GPU 本机环境可见性已通过下方 CUDA benchmark 和 Isaac Sim 渲染验证单独确认。

已通过 200 步随机动作环境冒烟测试：

```bash
.venv_isaaclab/bin/python scripts/debug_env.py --steps 200 --device cpu
```

已通过 oracle 隔离检查：

```bash
.venv_isaaclab/bin/python scripts/debug_observation.py --device cpu
```

检查结果显示 actor observation 不随 oracle 点改变，critic state 会随 oracle 点改变。

已通过奖励分项检查：

```bash
.venv_isaaclab/bin/python scripts/debug_reward.py --steps 16 --device cpu
```

## 第一阶段模块可视化验证

已新增并通过以下模块级验证：

- rover 代理模型：验证简化速度模型积分、位置/速度/yaw 更新、z 高度约束、控制限幅、观测和奖励有限性。
- 四车观测空间：验证 actor observation shape、critic state shape、每车 3 个邻居可见、terrain 占位特征为 0、oracle 不泄漏到 actor。
- 轨迹生成器与简化速度控制：验证 `[rho, beta] -> world subgoal -> line trajectory -> velocity command` 链路、轨迹端点、时间戳单调、heading 和控制命令限幅。

已执行可视化验证：

```bash
.venv_isaaclab/bin/python scripts/validate_first_stage.py \
  --config configs/experiment/exp_001_minimal.yaml \
  --device cpu \
  --steps 40
```

结果摘要：

```text
status: ok
device: cpu
num_envs: 32
n_agents: 4
actor_obs_shape: [32, 4, 41]
critic_state_shape: [32, 54]
steps_recorded: 40
initial_dmax: 7.9842
final_dmax: 3.7090
initial_dispersion: 13.4741
final_dispersion: 2.4246
mean_speed: 0.2742
mean_reward: 0.5416
```

输出产物：

```text
outputs/logs/first_stage_validation/validation_metrics.json
outputs/figures/first_stage_validation/proxy_rollout_curves.png
outputs/figures/first_stage_validation/observation_space_heatmap.png
outputs/figures/first_stage_validation/trajectory_control_validation.png
outputs/videos/first_stage_validation/proxy_rollout.gif
```

该验证使用 CPU 代理环境，目的是稳定检查第一阶段 torch 代理动力学、观测、轨迹和控制逻辑。它不修改默认训练配置，也不与后续 CUDA 训练冲突。

## CUDA 短训练评估

已新增 CUDA 短训练评估入口：

```bash
.venv_isaaclab/bin/python scripts/benchmark_cuda_short_training.py \
  --config configs/experiment/exp_001_minimal.yaml \
  --device cuda \
  --timesteps 128
```

当前执行结果：

```text
status: ok
cuda_available: true
device_count: 1
gpu_name: NVIDIA GeForce RTX 5090
torch_version: 2.10.0+cu128
num_envs: 32
n_agents: 4
timesteps: 128
env_steps: 4096
agent_steps: 16384
wall_time_s: 4.9433
env_steps_per_s: 828.6031
agent_steps_per_s: 3314.4123
estimated_seconds_per_1m_env_steps: 1206.8504
peak_cuda_memory_mb: 27.1919
artifact: outputs/logs/cuda_short_training/cuda_benchmark.json
```

注意：Codex 默认工作区沙箱隔离了 `/dev/nvidia*`，因此普通沙箱命令里 `nvidia-smi` 和 `torch.cuda.is_available()` 会失败；在非沙箱本机环境中 GPU 正常可见。CUDA benchmark 已在非沙箱本机环境运行通过，说明 PyTorch CUDA 路径可用。

已在非沙箱本机环境单独验证 CUDA pytest：

```bash
.venv_isaaclab/bin/python -m pytest tests/test_cuda_short_training.py -q
```

结果：

```text
1 passed
```

## Isaac Sim 真实三维 GPU 渲染验证

已在本机环境验证 Isaac Sim GUI / Vulkan / RTX 渲染路径：

```bash
.venv_isaaclab/bin/python -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': False, 'width': 960, 'height': 540, 'renderer': 'RealTimePathTracing'}); print('ISAAC_SIM_APP_STARTED', flush=True); [app.update() for _ in range(60)]; print('ISAAC_SIM_RENDER_FRAMES_OK 60', flush=True); app.close(); print('ISAAC_SIM_APP_CLOSED', flush=True)"
```

结果摘要：

```text
NVIDIA-SMI: ok
GPU: NVIDIA GeForce RTX 5090
Driver Version: 595.58.03
CUDA Version reported by nvidia-smi: 13.2
Graphics API: Vulkan
Isaac Sim active GPU: NVIDIA GeForce RTX 5090
Warp device: cuda:0 NVIDIA GeForce RTX 5090
ISAAC_SIM_APP_STARTED
ISAAC_SIM_RENDER_FRAMES_OK 60
Simulation App Shutting Down
```

结论：真实 Isaac Sim 三维 GPU 渲染环境可以启动并完成渲染帧推进。默认 Codex 沙箱下 GPU 设备节点不可见；需要在普通终端或具备完整设备访问的命令环境中运行 Isaac Sim / CUDA 相关验证。

已修复缓存目录权限问题：

```text
/home/u24/.cache/ov -> u24:u24
/home/u24/.cache/ov/texturecache -> u24:u24
/home/u24/.nvidia-omniverse -> u24:u24
/home/u24/.nvidia-omniverse/pycache -> u24:u24
```

其中原 root-owned 的 `/home/u24/.nvidia-omniverse` 已非破坏性备份为：

```text
/home/u24/.nvidia-omniverse.root-owned-backup-20260519_173022
```

修复后重跑 60 帧渲染，最新日志 `kit_20260519_173033.log` 未再出现 `Failed to access OmniCache directory` 或 texture cache 创建失败。

### 四车代理场景真实渲染

新增可视化脚本：

```text
scripts/view_proxy_rovers_isaac.py
```

该脚本在 Isaac Sim GUI 中构建一片空平地、四辆彩色代理 rover、轮子、朝向标记、网格线、DistantLight/DomeLight 和固定相机，用于验证“真实 viewport 中可见内容”而不依赖后续真实 rover USD/URDF 资产。

本次验证命令：

```bash
.venv_isaaclab/bin/python scripts/view_proxy_rovers_isaac.py \
  --duration-s 5 \
  --capture outputs/figures/isaac_render/proxy_rovers_scene.png
```

结果摘要：

```text
GPU: NVIDIA GeForce RTX 5090
Driver Version: 595.58.03
Graphics API: Vulkan
Warp device: cuda:0 NVIDIA GeForce RTX 5090
PROXY_ROVER_STAGE_EXPORTED outputs/isaac_scenes/proxy_rovers_scene.usda
PROXY_ROVER_SCENE_CAPTURED outputs/figures/isaac_render/proxy_rovers_scene.png
PROXY_ROVER_SCENE_READY rovers=4 ground=/World/Ground renderer=RealTimePathTracing headless=False
PROXY_ROVER_SCENE_VISIBLE_SECONDS 5.0
Simulation App Shutting Down
```

产物：

```text
outputs/figures/isaac_render/proxy_rovers_scene.png
outputs/isaac_scenes/proxy_rovers_scene.usda
```

截图验证：

```text
PNG: 1280 x 720, 691K
mean_rgb: [202.38, 202.74, 201.5]
std_rgb: [39.63, 39.79, 40.84]
min_rgb: [3, 3, 2]
max_rgb: [250, 250, 249]
```

结论：Isaac Sim 真实三维 GPU 渲染不仅能启动窗口，也能渲染出包含四辆代理 rover 和平地场景的可见内容。若需要人工观察窗口，使用：

```bash
.venv_isaaclab/bin/python scripts/view_proxy_rovers_isaac.py --keep-open
```

已通过 SKRL-MAPPO 短训练：

```bash
.venv_isaaclab/bin/python scripts/train.py \
  --config configs/experiment/exp_001_minimal.yaml \
  --device cpu \
  --timesteps 128
```

输出 checkpoint：

```text
outputs/checkpoints/exp_001_minimal_skrl_mappo.pt
```

另有轻量 smoke trainer checkpoint：

```text
outputs/checkpoints/exp_001_minimal_proxy.pt
```

Isaac Sim / Isaac Lab 启动验证：

- `isaacsim.SimulationApp(headless=True)` 可以启动并关闭
- `isaaclab.app.AppLauncher + SimulationContext` 可以 reset/step/close
- 官方 `create_empty.py` 教程脚本本身会进入持续仿真循环，使用 `timeout` 会被终止；因此最终使用最小 `SimulationContext` 脚本作为启动验收

## 低精度训练 + PhysX 高保真验证架构

已将“模型精度”和“是否渲染”解耦：

- 主训练仍使用 PyTorch kinematic proxy env，不启动 Isaac Sim，不渲染。
- 快速验证仍使用 pytest、JSON、曲线和 GIF，不依赖 Isaac Sim viewport。
- 高保真验证新增 Isaac Sim PhysX 闭环脚本，使用官方 Jetbot 资产和崎岖地形，仅作为评估/展示层。
- 渲染只在显式 `--render` 时启用，不作为策略观测输入。

### 结构化地形特征

已将原来的 terrain 全 0 占位扩展为可配置 provider，保留 `terrain_dim=5`，不改变 actor observation shape。

5 维含义固定为：

```text
height, slope_x, slope_y, roughness, traversability
```

默认 `flat_proxy` 仍返回全 0，保持现有训练和测试兼容；`lunar_heightfield_proxy` 返回程序化高度场的低维结构化特征。critic 的 global terrain state 使用同类统计特征。

新增/更新文件：

```text
source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/terrain_features.py
source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/gathering_env_cfg.py
tests/test_terrain_features.py
configs/experiment/exp_003_terrain_ablation.yaml
```

地形特征测试结果：

```bash
.venv_isaaclab/bin/python -m pytest tests/test_terrain_features.py tests/test_four_rover_observation_space.py -q
```

```text
4 passed
```

完整快速回归：

```bash
.venv_isaaclab/bin/python -m pytest
```

```text
16 passed in 3.08s
```

无渲染 proxy 验证仍可运行：

```bash
.venv_isaaclab/bin/python scripts/validate_first_stage.py \
  --config configs/experiment/exp_001_minimal.yaml \
  --device cpu \
  --steps 20
```

结果摘要：

```text
status: ok
actor_obs_shape: [32, 4, 41]
critic_state_shape: [32, 54]
initial_dmax: 7.9842
final_dmax: 5.9010
mean_reward: 0.6019
```

### 官方 Jetbot PhysX smoke

新增脚本：

```text
scripts/physx_jetbot_common.py
scripts/physx_jetbot_smoke.py
```

使用官方资产：

```text
/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd
left_wheel_joint, right_wheel_joint
wheel_radius = 0.0335
wheel_base = 0.118
```

平地 smoke：

```bash
.venv_isaaclab/bin/python scripts/physx_jetbot_smoke.py \
  --terrain flat \
  --steps 60 \
  --output outputs/logs/physx_jetbot_smoke/jetbot_flat_smoke.json
```

结果摘要：

```text
status: ok
displacement_xy: 0.1779
min_z: 0.0335
max_tilt_deg: 0.3168
sim_steps_per_s: 157.18
```

崎岖地形 smoke：

```bash
.venv_isaaclab/bin/python scripts/physx_jetbot_smoke.py \
  --terrain rough \
  --steps 60 \
  --output outputs/logs/physx_jetbot_smoke/jetbot_rough_smoke.json
```

结果摘要：

```text
status: ok
displacement_xy: 0.1661
min_z: 0.0279
max_tilt_deg: 7.3585
sim_steps_per_s: 155.05
```

### 四车 PhysX 高保真评估/展示

新增脚本：

```text
scripts/evaluate_physx_four_jetbots.py
```

该脚本加载四个官方 Jetbot，在 Isaac Sim PhysX 中运行闭环评估。策略动作仍为当前 `[rho, beta]`，经现有解码、轨迹生成和简化速度控制链路转为 `linear/angular`，再由 Jetbot differential controller 映射为轮速命令。

scripted 四车 smoke：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --terrain rough \
  --steps 4 \
  --sim-steps-per-control 3 \
  --scripted \
  --output outputs/logs/physx_four_jetbots/evaluation_scripted_smoke.json
```

结果摘要：

```text
status: ok
backend: scripted
n_agents: 4
final_dmax: 3.9679
max_tilt_deg: 6.2675
collision_count: 0
physics_updates_per_s: 105.97
```

checkpoint 四车 smoke：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --terrain rough \
  --steps 4 \
  --sim-steps-per-control 3 \
  --checkpoint outputs/checkpoints/exp_001_minimal_proxy.pt \
  --output outputs/logs/physx_four_jetbots/evaluation_checkpoint_smoke.json
```

结果摘要：

```text
status: ok
backend: smoke
n_agents: 4
final_dmax: 3.9688
max_tilt_deg: 6.1208
collision_count: 0
physics_updates_per_s: 110.61
```

渲染展示 smoke：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py \
  --terrain rough \
  --steps 2 \
  --sim-steps-per-control 2 \
  --scripted \
  --render \
  --capture-interval 1 \
  --output outputs/logs/physx_four_jetbots/evaluation_render_smoke.json \
  --capture outputs/figures/physx_four_jetbots/evaluation_scene.png \
  --gif outputs/videos/physx_four_jetbots/evaluation_rollout.gif
```

产物：

```text
outputs/figures/physx_four_jetbots/evaluation_scene.png
outputs/videos/physx_four_jetbots/evaluation_rollout.gif
outputs/logs/physx_four_jetbots/evaluation_render_smoke.json
```

截图/GIF 检查：

```text
PNG: 1280 x 720, 658K, mean_rgb [220.33, 217.62, 215.53]
GIF: 1280 x 720, 16K, mean_rgb [221.21, 221.22, 218.96]
```

结论：当前架构已支持低精度 proxy 训练、无渲染快速验证、Isaac Sim PhysX 官方轮式资产高保真评估，以及显式渲染展示。PhysX 尚未进入正式训练 loop。

## Proxy 策略初步收敛

目标：在简化 PyTorch proxy 环境中得到可复现的多车集合初步收敛 checkpoint。验收标准采用趋势优先口径：固定评估 256 个 env、100 步 rollout 后，`final_dmax / initial_dmax <= 0.4`，成功率记录但不作为硬门槛。

本轮新增：

```text
configs/experiment/exp_004_proxy_convergence.yaml
scripts/train_proxy_convergence.py
scripts/evaluate_proxy_policy.py
tests/test_convergence_tools.py
```

实现要点：

- 配置解析已支持 `reward.weights`、`reward.coefficients`、`success_thresholds`、`low_level_control` 覆盖。
- 奖励新增默认关闭的绝对紧凑度 shaping：`dmax_level`、`dispersion_level`。
- 终端奖惩改为配置项：`success_bonus`、`failure_penalty`。
- 收敛训练入口使用 shared actor + centralized critic checkpoint 格式，兼容现有 `play.py` 和 PhysX 评估加载。
- warm-start 使用 scripted gathering controller 做行为克隆；随后执行 PPO 微调，并按 deterministic eval 的 dmax ratio 保存 best checkpoint。

训练命令：

```bash
.venv_isaaclab/bin/python scripts/train_proxy_convergence.py \
  --config configs/experiment/exp_004_proxy_convergence.yaml \
  --device cuda \
  --bc-steps 200
```

说明：配置默认保留 `bc_steps: 2000`，但 2000-step warm-start 在当前实现下耗时偏高。本轮为在 2 小时内拿到初步收敛结果，实际采用 `--bc-steps 200`；该 run 已达到验收标准。

训练摘要：

```text
device: cuda
bc_steps: 200
ppo_updates: 15
best_phase: bc
initial_dmax: 7.2455
final_dmax: 0.8875
dmax_reduction_ratio: 0.1225
initial_dispersion: 12.3021
final_dispersion: 0.1485
mean_reward: 0.6207
success_rate: 0.9453
collision_rate: 0.0586
timeout_rate: 0.0
```

独立验收命令：

```bash
.venv_isaaclab/bin/python scripts/evaluate_proxy_policy.py \
  --config configs/experiment/exp_004_proxy_convergence.yaml \
  --checkpoint outputs/checkpoints/exp_004_proxy_converged.pt \
  --device cuda \
  --num-envs 256 \
  --steps 100 \
  --output outputs/logs/exp_004_proxy_convergence/final_eval.json
```

独立验收结果：

```text
initial_dmax: 7.2260
final_dmax: 0.8900
dmax_reduction_ratio: 0.1232
initial_dispersion: 12.2414
final_dispersion: 0.1479
mean_reward: 0.5486
success_rate: 0.9453
collision_rate: 0.0703
timeout_rate: 0.0
mean_done_step: 73.8672
```

产物：

```text
outputs/checkpoints/exp_004_proxy_converged.pt
outputs/logs/exp_004_proxy_convergence/train_metrics.jsonl
outputs/logs/exp_004_proxy_convergence/eval_metrics.json
outputs/logs/exp_004_proxy_convergence/final_eval.json
outputs/logs/exp_004_proxy_convergence/convergence_curves.png
outputs/logs/exp_004_proxy_convergence/eval_rollout.gif
```

结论：在当前简化 proxy 环境中，warm-start + PPO 链路已得到一个初步收敛 checkpoint。该结果不是纯 RL 从零收敛；best checkpoint 来自 BC 阶段，PPO 阶段保持了较高 reward，但没有超过 BC 的 dmax ratio。

## 已知问题与风险

`pip check` 当前会报告 Isaac Sim 6.0.0 与 Isaac Lab v3 beta 的部分依赖元数据冲突，例如 `packaging`、`llvmlite`、`coverage`、`typing_extensions`、`starlette`。这些冲突目前没有阻断实际 import、Isaac Sim/Lab 启动、单元测试或 SKRL-MAPPO 短训练。

不要随意在该环境里执行全量 `pip install -U`，否则可能破坏 Isaac Sim / Isaac Lab 的可运行组合。

当前正式训练仍是第一阶段代理动力学。PhysX Jetbot 层用于高保真评估/展示，不参与主训练梯度或 rollout 采样。Jetbot 体型较小，崎岖地形幅值过大时可能不代表真实月球 rover；如果后续需要更真实尺寸和载荷，应再评估 NovaCarter 或自定义 rover USD/URDF。

## 推荐继续命令

快速回归：

```bash
.venv_isaaclab/bin/python -m pytest
.venv_isaaclab/bin/python scripts/debug_env.py --steps 200 --device cpu
.venv_isaaclab/bin/python scripts/debug_observation.py --device cpu
.venv_isaaclab/bin/python scripts/debug_reward.py --steps 16 --device cpu
.venv_isaaclab/bin/python scripts/validate_first_stage.py --device cpu --steps 40
.venv_isaaclab/bin/python scripts/benchmark_cuda_short_training.py --device cuda --timesteps 128
.venv_isaaclab/bin/python scripts/view_proxy_rovers_isaac.py --keep-open
.venv_isaaclab/bin/python scripts/physx_jetbot_smoke.py --terrain rough --steps 60
.venv_isaaclab/bin/python scripts/evaluate_physx_four_jetbots.py --terrain rough --steps 40 --checkpoint outputs/checkpoints/exp_001_minimal_proxy.pt
.venv_isaaclab/bin/python scripts/evaluate_proxy_policy.py --config configs/experiment/exp_004_proxy_convergence.yaml --checkpoint outputs/checkpoints/exp_004_proxy_converged.pt --device cuda --num-envs 256 --steps 100
```

第一阶段短训练：

```bash
.venv_isaaclab/bin/python scripts/train.py \
  --config configs/experiment/exp_001_minimal.yaml \
  --device cpu \
  --timesteps 128
```

策略回放：

```bash
.venv_isaaclab/bin/python scripts/play.py \
  --config configs/experiment/exp_001_minimal.yaml \
  --checkpoint outputs/checkpoints/exp_001_minimal_skrl_mappo.pt \
  --steps 100
```

## 下一步建议

1. 将 proxy 收敛训练纳入固定 GPU 回归，并优化 BC 进度日志与采样吞吐。
2. 对比 pure RL、BC warm-start、BC+PPO 三条曲线，明确论文/报告中采用哪一种训练设定。
3. 接入真实 rover USD/URDF 前，先确定关节命名、控制接口和期望控制模式。
4. 将四车 PhysX 评估步数扩展到完整 episode，形成稳定高保真评估表。
5. 对比 Jetbot 与 NovaCarter 的崎岖地形稳定性和吞吐，再决定是否更换高保真评估资产。
