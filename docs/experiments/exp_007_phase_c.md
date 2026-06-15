# exp_007 阶段 C

## 目的

加入 lunar crater proxy 地形、PhysX lunar crater mesh、历史高保真 sanity 评估，并得到一个 best checkpoint 来自 PPO 阶段的弱 warm-start 结果。

## 配置

```text
configs/experiment/exp_007_phase_c_weak_warmstart.yaml
configs/experiment/exp_007_phase_c_pure_rl.yaml
```

代表性最终 checkpoint：

```text
outputs/runs/exp_007_phase_c/phase_c_weak50_lr3e3_teacher_2m/checkpoints/best.pt
```

## 训练结果

```text
phase: ppo
update: 7
dmax_reduction_ratio: 0.1430
success_rate: 1.0000
collision_rate: 0.0000
timeout_rate: 0.0000
```

独立 proxy 评估：

```text
dmax_reduction_ratio: 0.1401
success_rate: 1.0000
collision_rate: 0.0000
timeout_rate: 0.0000
mean_done_step: 83.53
```

## PhysX 结果

历史 lunar crater 高保真 sanity check 通过：

```text
success_rate: 1.0000
collision_rate: 0.0000
mean_final_dmax: 0.7977
```

## 说明

PhysX 只用于 checkpoint 级高保真评估和展示，不进入主训练 loop。该历史结果不能表述为 Isaac Lab 物理训练结果；当前活跃高保真验证入口已切换为 `scripts/evaluate_physx_jackal_tracking.py`。
