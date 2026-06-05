# 当前状态

## 当前主线

- 训练主线：PyTorch terrain-aware proxy 环境。
- 渲染和高保真 sanity check：Isaac Sim / PhysX Jetbot 评估。
- 视觉观测不进入 policy input；地形以低维结构化特征进入策略。
- 生成结果写入 `outputs/runs/`，并由 git 忽略。

## 已验证结果

| 实验 | 地形 | 方法 | 严格状态 | 说明 |
| --- | --- | --- | --- | --- |
| exp006 | 平地 proxy | BC + PPO | 通过 | PPO 阶段选出的平地 baseline。 |
| exp008 | 弱 lunar crater 3D proxy | 弱 warm-start + PPO | 3 seeds 通过 | 当前最完整的 3-seed terrain-aware proxy 结果。 |
| exp009 | 强 lunar crater 3D proxy | 弱 warm-start + PPO | 未通过 | seed23 通过；seed31 未通过 success/timeout；seed47 未运行。 |

当前推荐的完整 suite checkpoint：

```text
outputs/runs/exp_008_terrain3d/_suite/checkpoints/
```

当前强地形诊断 checkpoint：

```text
outputs/runs/exp_009_terrain3d_strong/_suite/checkpoints/seed_23_best.pt
outputs/runs/exp_009_terrain3d_strong/_suite/checkpoints/seed_31_best.pt
```

## 当前阻塞

exp009 strong terrain 已证明 3D 地形动力学生效，高度范围约 `0.74 m`。但当前高层动作和 reward/control 设计不能在所有 seeds 上稳定清除严格 gate。

seed31 失败模式：

```text
dmax_reduction_ratio: 0.1819  # 通过
success_rate: 0.8740          # 未通过
collision_rate: 0.0049        # 通过
timeout_rate: 0.1250          # 未通过
```

## 下一步

不要把“继续无限加 PPO 步数”作为第一选择。优先处理：

1. 改进成功区附近的行为，让速度条件和 hold 条件稳定满足。
2. 增加或调整“及时完成且保持安全距离”的 reward 项。
3. 评估 `[rho, beta]` 单步子目标动作在强地形下是否表达能力不足。
4. 回放 seed31 失败 episode，区分 timeout 来自地形速度缩放、dispersion 不稳定，还是 speed hold 条件失败。

