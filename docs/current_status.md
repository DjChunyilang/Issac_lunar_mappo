# 当前状态

## 当前主线

- 近期工程主线：按 [architecture/overall_plan_v3.md](architecture/overall_plan_v3.md) 将项目拉回 `Isaac Sim / Isaac Lab + SKRL-MAPPO + rover articulation` 主线。
- 训练结果主线：PyTorch terrain-aware proxy 环境的 `exp008` 仍作为当前已验证 baseline。
- 渲染和高保真 sanity check：Isaac Sim / PhysX Jetbot 评估需要整理成可重复 runbook。
- 视觉观测不进入 policy input；地形以低维结构化特征进入策略。
- CPU unit contract 已接入 CI：Python 3.12、`skrl==2.1.0`、全量 `pytest -q -ra`，并包含非 skip 的 SKRL import 测试。
- 生成结果写入 `outputs/runs/`，并由 git 忽略。

## 当前接口状态

- `reward.coefficients.obstacle_collision` 已从 base reward 配置和 dataclass 移除；当前没有 obstacle collision 输入，因此不保留未消费配置项。
- `observation.communication_radius` 是本轮唯一开放的 observation 配置项；`max_neighbors`、`ego_dim`、`neighbor_dim` 等维度相关字段仍由代码固定。
- actor observation schema 为 `ego_v2_speed_angular`。ego 末尾两个历史零占位通道已替换为 `speed_xy` 和 `abs_angular_velocity`，tensor shape 不变，但 checkpoint 输入语义已经改变。
- SKRL checkpoint metadata 会记录 `training_semantics`、`experiment_name`、`algorithm_mode`、`observation_schema_version`、`shared_actor`、`centralized_critic`、`shared_value`、`device` 和 `checkpoint_path`。
- `scripts/train_skrl_mappo.py` 已加入 CUDA 必需检查、NaN 检查、action scale、reward component、done reason、random baseline 和 post-training eval 遥测；输出用于诊断，不等同于 strict pass。

## 当前 SKRL/CUDA 诊断

- `scripts/run_cuda_training_validation.py` 使用 `configs/experiment/exp_cuda_contract.yaml` 跑 `32 / 512 / 5000` timesteps CUDA contract。最新本地机器可读摘要为 `outputs/runs/cuda_training_validation_summary.json`：三段均 `status: ok`、`nan_detected: false`，但 `success_rate_final: 0.0`，只证明工程链路可运行。
- `exp012_action_scale_warmup_probe` 是 SKRL-MAPPO action-scale 诊断实验，不是当前主结果。20k 探针的 `diagnosis_probe_20000.json` 显示 pairwise/oracle distance 有改善，但 success 仍为 `0.0`，且动作饱和明显。
- exp012 下一步焦点是 `action_scale_ablation` 和 `success_gate_reachability_diagnostic`；500k 长预算只有在生成完整诊断 JSON 后再更新实验结论。

## 已验证结果

| 实验 | 地形 | 方法 | 严格状态 | 说明 |
| --- | --- | --- | --- | --- |
| exp006 | 平地 proxy | BC + PPO | 通过 | PPO 阶段选出的平地 baseline。 |
| exp008 | 弱 lunar crater 3D proxy | 弱 warm-start + PPO | 3 seeds 通过 | 当前最完整的 3-seed terrain-aware proxy 结果。 |
| exp009 | 强 lunar crater 3D proxy | 弱 warm-start + PPO | 未通过 | seed23 通过；seed31 未通过 success/timeout；seed47 未运行。 |
| exp010 | 强 lunar crater 3D proxy | 成功 gate 诊断 + hold reward 短程修复 | 未通过 | seed31 success 可到 0.90，但 collision/timeout 仍失败；strong terrain 诊断线暂缓。 |
| exp012 | proxy SKRL-MAPPO CUDA 诊断 | action scale warmup probe | 未通过 | 20k 探针 distance 有改善但 success 为 0，动作饱和；不作为 strict 结果。 |

当前推荐的完整 suite checkpoint：

```text
outputs/runs/exp_008_terrain3d/_suite/checkpoints/
```

当前强地形诊断 checkpoint 仅作为对照，不作为近期继续训练入口：

```text
outputs/runs/exp_009_terrain3d_strong/_suite/checkpoints/seed_23_best.pt
outputs/runs/exp_009_terrain3d_strong/_suite/checkpoints/seed_31_best.pt
```

## 与总体规划的差距

V2.0 总体规划目标是 `Isaac Sim / Isaac Lab + SKRL-MAPPO + 低维局部子目标动作 + 确定性轨迹生成器 + 简化速度跟踪控制器`。当前已经验证的是 proxy 训练和部分 PhysX 展示链路，还不是完整 Isaac Lab 物理训练闭环。

详细偏差 review 和 V3 归正路线见 [architecture/overall_plan_v3.md](architecture/overall_plan_v3.md)。该文档明确：proxy 是临时工程绕路，后续目标应回到 Isaac Lab 物理环境和 SKRL-MAPPO 主训练。

## 暂缓的训练诊断

exp009 strong terrain 已证明 3D 地形动力学生效，高度范围约 `0.74 m`。但当前高层动作和 reward/control 设计不能在所有 seeds 上稳定清除严格 gate。

seed31 失败模式：

```text
dmax_reduction_ratio: 0.1819  # 通过
success_rate: 0.8740          # 未通过
collision_rate: 0.0049        # 通过
timeout_rate: 0.1250          # 未通过
```

exp010 诊断与短程修复补充：

```text
seed31 hold reward 6M continuation:
  dmax_reduction_ratio: 0.1720  # 通过
  success_rate: 0.9014          # 通过
  collision_rate: 0.0273        # 未通过
  timeout_rate: 0.0742          # 未通过
  speed_ok_rate: 0.9997
  timeout_final_dmax_mean: 3.6919
  timeout_final_dispersion_mean: 4.3987
```

结论：失败不是 speed hold，且不是简单继续增加 PPO 步数能解决的趋势。timeout episode 末端仍远离 `dmax/dispersion` 成功区。

当前先不继续 strong terrain 失败 episode 诊断，也不新增 long-budget PPO。后续恢复训练研究时，再基于这些结论设计动作表示或 curriculum 实验。

## 下一步

近期优先从已补齐的工程闭环继续推进到物理/训练主线，不以单次 reward 曲线作为成功证据：

1. 维持 `.venv_isaaclab/bin/python -m pytest -q -ra` 和 GitHub Actions unit contract 为每次代码修改的最低门槛。
2. 按 [runbooks/setup_environment.md](runbooks/setup_environment.md) 跑通 `scripts/validate_first_stage.py` 的 CPU 短验证，确认 proxy core、观测、奖励、轨迹和图像产物链路可用。
3. 按 [runbooks/train_skrl_mappo.md](runbooks/train_skrl_mappo.md) 复跑 CUDA contract 和 exp012 action-scale 探针，优先做动作尺度消融而不是继续把单个 PPO 预算拉长。
4. 跑通 `scripts/debug_env.py`、`scripts/debug_observation.py` 和 `scripts/debug_reward.py`，作为基础回归检查。
5. 跑通 `scripts/evaluate_physx_four_jetbots.py` 的 headless/render sanity 路径，并把结果写入标准 run 目录。
