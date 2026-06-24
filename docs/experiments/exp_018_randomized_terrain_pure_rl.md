# exp018 随机增强地形 Pure RL

## 目的

在 exp017 单张固定地图 strict 通过后，验证 shared-joint MAPPO 是否能在“每个并行环境、每个 episode 独立重采样”的更强月面上获得可训练信号。该实验同时增强 terrain reward，使策略不仅感知脚下地形，还能对候选落点风险、地形减速和实际高度变化作出反应。

## 配置

```text
configs/experiment/exp018_randomized_terrain_pure_rl.yaml
```

关键设置：

- observation schema 和模型接口保持 `ego_v3_local_terrain_grid / 86 / 54`。
- pure RL、`shared_joint`、communication radius `12 m`，不使用 BC。
- 每个环境在 reset 时独立重采样地形平移、朝向、相位和幅值尺度；episode 内地图固定。
- 地形由 exp017 的 `amplitude=0.08`、7 个 crater 提升为 `amplitude=0.10`、9 个 crater。
- crater radius 为 `0.45–1.30 m`，depth/diameter 为 `0.12`，minimum speed scale 降为 `0.22`。
- terrain reward weight 从 `0.20` 提高到 `0.30`。
- 新增候选子目标风险、实际地形减速和高度变化代价；这些项在旧配置中的默认值均为 0，不改变历史实验语义。

## 地形统计

| profile | height range | minimum traversability | mean speed scale |
| --- | ---: | ---: | ---: |
| exp017 固定偏弱中档地图 | 0.395 m | 0.347 | 0.629 |
| exp018 基准地图 | 0.618 m | 0.151 | 0.506 |
| exp018 64 张随机地图均值 | 0.528 m | 0.211 | 0.569 |

64 张随机地图的高度范围为 `0.369–0.722 m`，并具有 64 个不同 phase。该统计说明并行环境不再共享完全相同的固定地形。

## 工程验证

- 地形旋转、世界坐标转换、局部网格有限值和 reset 子集随机化单元测试通过。
- 完整 `pytest -q -ra` 通过。
- CPU smoke 通过。
- CUDA smoke 使用 256 environments、64 timesteps、rollout 32，通过：
  - 一个 optimizer；
  - 两次 shared-joint update 和两次 critic update；
  - policy parameter delta L2 `0.2064`；
  - terrain input column delta L2 `0.1392`；
  - post-training action std `0.3158`；
  - 无 NaN/Inf。
- 随机地图 proxy GIF 已生成，渲染使用对应 episode 的 terrain runtime，而不是退回固定基准地图。

工程产物：

```text
outputs/runs/exp018_randomized_terrain_pure_rl/smoke_cpu_exp018/
outputs/runs/exp018_randomized_terrain_pure_rl/smoke_cuda_exp018/
outputs/runs/exp018_randomized_terrain_pure_rl/smoke_cuda_exp018/videos/proxy_eval_rollout.gif
```

## 严格标准

本实验仍使用标准 proxy strict gate：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

`pure_rl_long` 候选排序可忽略 timeout 作为训练过程趋势诊断，但最终 `strict_acceptance.json` 仍要求 timeout 为 0。

## Reward 诊断

exp017 中 terrain reward 的绝对贡献约占总绝对 reward 的 `7.8%`。旧 terrain penalty 主要依据 rover 脚下 roughness 和 traversability，且 terrain dynamics 只降低平移速度，不会产生不可通行区域、接触失败或主动绕障约束。因此策略沿近似直线穿过地形在旧模型下可能仍是最优行为。

exp018 的短 smoke 中 terrain reward 绝对贡献约为 `26%–29%`。这证明地形项已进入有效优化信号，但不能据此宣称策略已经学会绕障；是否形成地形相关轨迹仍需正式长训练和跨随机地图独立评估。

## 长训练结果

seed23 连续 20M 长训练已完成：

| 项 | 值 |
| --- | --- |
| run_id | `pure_rl_seed23_20m_randomized_terrain` |
| budget | `10240 timesteps / 20,971,520 env steps` |
| rollout | `32` |
| joint updates | `320` |
| optimizer count | `1` |
| best candidate | `ppo_timestep_010240.pt` |
| policy parameter delta L2 | `4.5074` |
| terrain input weight delta L2 | `1.2220` |
| post-training action std | `0.4886` |

候选评估里程碑：

| checkpoint timestep | env steps | dmax ratio | success | collision | timeout | mean done step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 2,097,152 | 0.2967 | 0.0674 | 0.1709 | 0.7715 | 214.1 |
| 4096 | 8,388,608 | 0.1484 | 0.8125 | 0.1797 | 0.0264 | 158.9 |
| 10240 | 20,971,520 | 0.1421 | 0.9678 | 0.0303 | 0.0039 | 150.2 |

独立 final eval：

| checkpoint | dmax ratio | success | collision | timeout | strict |
| --- | ---: | ---: | ---: | ---: | --- |
| `best.pt` / `ppo_timestep_010240.pt` | 0.1417 | 0.9609 | 0.0352 | 0.0088 | 未通过 |

通过项：

- dmax reduction 已达标，`0.1417 <= 0.20`。
- success 已达标，`0.9609 >= 0.90`。
- 20M 候选的 timeout 已降到 `0.0039`，独立 final eval 为 `0.0088`，明显优于早期但仍非 0。

失败项：

- collision final eval 为 `0.0352`，高于 `0.02` strict gate。
- timeout final eval 为 `0.0088`，未满足 `timeout_rate == 0`。

本轮地形确实比 exp017 更强：final eval 的 terrain height range 为 `0.768 m`，minimum traversability 为 `0.144`，mean terrain speed scale 为 `0.474`。reward 诊断中 terrain 项约占绝对 reward 的 `52%`，说明当前失败不是“地形奖励完全没进训练信号”，而是策略在随机地形上仍倾向高速聚集，末端安全余量不够。

## 失败分析

exp018 相比 exp017 最大变化是随机地图与更强地形。训练趋势是正向的：2M 几乎不会成功，8M 已显著降低 dmax，20M 达到 96% 以上 success。但 collision 从 8M 的高位逐步下降后仍停在 3% 左右，说明当前 reward / action 表达已足够学会集合，却不足以稳定学会“最后阶段安全地集合”。

主要问题：

- 策略在 final eval 中 physical rho 平均约 `1.108 m`，高 rho 占比约 `73%`，末端仍偏高速推进。
- near violation rate 为 `0.151`，说明许多 episode 虽然最终成功，但过程中或末端贴得过近。
- terrain penalty 已很重，继续单纯放大 terrain weight 可能会让 reward 更难平衡，而不一定修复队友间安全。
- timeout episode 的 mean terrain speed scale 为 `0.403`，剩余 timeout 多发生在更慢、更差的随机地形样本上。

## 产物路径

```text
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/checkpoints/best.pt
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/metrics/summary.json
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/metrics/eval_metrics.json
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/metrics/final_eval_proxy.json
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/metrics/strict_acceptance.json
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/metrics/checkpoint_status.json
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/metrics/proxy_rollout_render.json
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/figures/terrain_height_map.png
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/videos/proxy_eval_rollout.gif
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/tensorboard/
```

## 当前结论

exp018 不是 strict pass。它是随机增强地形下的单 seed candidate：已经证明 shared-joint pure RL 能在随机地图上获得集合能力，但安全收敛仍未达到正式 gate。

## 下一步

下一轮优先处理 collision 和末端安全，而不是只继续加长训练或放大 terrain penalty：

1. 增加末端阶段的速度/距离耦合约束，例如接近队友或进入 success neighborhood 后降低 rho 或增加近距惩罚斜率。
2. 把路径级风险或候选轨迹风险纳入 action evaluation，而不是只惩罚落点和脚下地形。
3. 对最慢随机地形样本增加 curriculum 或 trapped / impassable 机制，区分“该绕行”和“该减速等待”。
4. exp018 若再次训练，应记录 collision episode 的地形与相对队形分布，先确认碰撞来自队友聚集几何还是地形绕行造成的局部拥挤。
