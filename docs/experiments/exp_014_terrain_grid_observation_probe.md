# exp014 局部地形网格观测探针

## 目的

验证 Actor 从脚下单点地形特征升级为车体系 `5×5×2` 局部网格后，观测构造、SKRL-MAPPO CUDA 训练、checkpoint schema 和地形输入梯度链路均可用。

## 配置

```text
configs/experiment/exp014_terrain_grid_observation_probe.yaml
```

关键参数：

```text
seed: 23
num_envs: 256
timesteps: 5000
terrain: weak lunar_crater_proxy
schema: ego_v3_local_terrain_grid
actor_obs_dim: 86
critic_state_dim: 54
```

网格坐标为前后 `[-0.4, 0.0, 0.4, 0.8, 1.2] m`、横向 `[-0.8, -0.4, 0.0, 0.4, 0.8] m`，通道为相对高度和风险。

## 严格标准

本实验不是 strict convergence 实验，不使用 `dmax/success/collision/timeout` gate。工程验收要求：

- 无 NaN/Inf；
- policy 参数发生更新；
- Actor 第一层的 terrain 输入列权重发生更新；
- post-training 动作标准差大于 `1e-4`；
- 非平地地形观测不是全零；
- checkpoint schema 和 86/54 维接口匹配。

## 结果表

| seed | timesteps | policy delta L2 | terrain column delta L2 | action std | terrain max abs | 工程验收 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 23 | 5000 | 3.5165 | 2.0113 | 0.6220 | 0.4346 | 通过 |

## 失败分析

首次运行在 256 环境下暴露了邻居特征 Python 双重循环的性能瓶颈。将“通信半径内按距离排序并选取 3 个邻居”改为等价张量化实现后，256 环境弱月面观测构造约为 `1.97 ms/call`，正式探针完成并通过。

本实验没有执行 strict final eval，因此不能根据参数更新或动作标准差推断集合策略已收敛。

## 产物路径

```text
outputs/runs/exp014_terrain_grid_observation_probe/metrics.jsonl
outputs/runs/exp014_terrain_grid_observation_probe/terrain_observation_validation_summary.json
outputs/checkpoints/exp014_terrain_grid_observation_probe.pt
```

## 结论

新局部地形网格的观测、梯度和 CUDA 训练链路有效。该 checkpoint 仅为工程探针，不是当前主结果，也不替代 exp008 的历史 strict baseline。

## 下一步

以当前 schema 创建正式多 seed terrain-grid 对照实验，运行独立 proxy final eval 和 strict suite；随后在 PhysX / Jackal 层实现同布局 raycast / height scanner。
