# 当前状态

## 当前主线

- 近期工程主线：按 [architecture/overall_plan_v3.md](architecture/overall_plan_v3.md) 将项目拉回 `Isaac Sim / Isaac Lab + SKRL-MAPPO + rover articulation` 主线。
- 训练结果主线：PyTorch terrain-aware proxy 环境的 `exp008` 仍作为当前已验证 baseline。
- 渲染和高保真 sanity check：Isaac Sim / PhysX Jetbot 评估需要整理成可重复 runbook。
- 视觉观测不进入 policy input；地形以低维结构化特征进入策略。
- 生成结果写入 `outputs/runs/`，并由 git 忽略。

## 已验证结果

| 实验 | 地形 | 方法 | 严格状态 | 说明 |
| --- | --- | --- | --- | --- |
| exp006 | 平地 proxy | BC + PPO | 通过 | PPO 阶段选出的平地 baseline。 |
| exp008 | 弱 lunar crater 3D proxy | 弱 warm-start + PPO | 3 seeds 通过 | 当前最完整的 3-seed terrain-aware proxy 结果。 |
| exp009 | 强 lunar crater 3D proxy | 弱 warm-start + PPO | 未通过 | seed23 通过；seed31 未通过 success/timeout；seed47 未运行。 |
| exp010 | 强 lunar crater 3D proxy | 成功 gate 诊断 + hold reward 短程修复 | 未通过 | seed31 success 可到 0.90，但 collision/timeout 仍失败；strong terrain 诊断线暂缓。 |

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

近期优先处理环境搭建与工程闭环，不以 reward 收敛为验收目标：

1. 按 [architecture/overall_plan_v3.md](architecture/overall_plan_v3.md) 和 [runbooks/setup_environment.md](runbooks/setup_environment.md) 固化 `.venv_isaaclab`、Isaac Sim、Isaac Lab、SKRL 和本地任务包安装检查。
2. 跑通 `scripts/validate_first_stage.py` 的 CPU 短验证，确认 proxy core、观测、奖励、轨迹和图像产物链路可用。
3. 跑通 `scripts/train.py --backend skrl` 的短 MAPPO smoke，确认 SKRL wrapper 和 centralized critic state 接口可用。
4. 跑通 `scripts/debug_env.py`、`scripts/debug_observation.py` 和 `scripts/debug_reward.py`，作为基础回归检查。
5. 跑通 `scripts/evaluate_physx_four_jetbots.py` 的 headless/render sanity 路径，并把结果写入标准 run 目录。
