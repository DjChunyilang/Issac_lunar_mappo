# exp_007 阶段 C

## 目的

加入 lunar crater proxy 地形、PhysX lunar crater mesh、四 Jetbot 评估，并得到一个 best checkpoint 来自 PPO 阶段的弱 warm-start 结果。

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

四 Jetbot lunar crater 评估作为高保真闭环 sanity check 通过：

```text
success_rate: 1.0000
collision_rate: 0.0000
mean_final_dmax: 0.7977
```

## 说明

PhysX 只用于 checkpoint 级高保真评估和展示，不进入主训练 loop。该结果应表述为“proxy checkpoint 在 PhysX / Jetbot 场景中的闭环评估结果”，不能表述为 Isaac Lab 物理训练结果。
