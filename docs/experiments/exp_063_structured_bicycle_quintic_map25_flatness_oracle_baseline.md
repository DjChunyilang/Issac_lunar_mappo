# exp063 structured bicycle quintic map25 flatness oracle baseline

## 目的

exp051–exp062 的 checkpoint 均在旧集合语义下训练和选择：oracle 使用旧目标，集合成功也没有要求实际集合区域足够平整。虽然 exp051 在旧语义独立评估中达到 dmax ratio `0.1836`、success `0.9883`、collision `0.0020`、timeout `0.0098`，但这些 success/timeout 数字不能直接沿用到新的 terrain-aware oracle 与 flatness gate。

使用当前代码对 exp051 `best.pt` 做 `1024 env / 320 steps` 新语义复评后得到：

- dmax ratio `0.1465`，通过 dmax gate；
- success `0.1104`，明显低于 `0.90`；
- collision `0.0068`，通过 collision gate；
- timeout `0.8828`，明显高于 `0`；
- oracle 搜索可行率 `1.0`，最终集合点平整率仅 `0.1172`；
- `904` 个 timeout episode 的最终平整率仅 `0.0066`。

这说明主要语义缺口不是“找不到平整 oracle”，而是旧策略没有学会在 terrain-aware 最优集合点附近形成满足平整度约束的稳定队形。

exp063 因此作为新语义的第一条训练基线：保持 exp051 的 Actor/Critic、reward、filter、control、PPO 和 initial-state curriculum 不变，只显式冻结 terrain-aware 最优集合点搜索、实际质心平整度 success gate 和独立评估配置。该实验用于建立可比较的新基线，不预设会通过 strict gate。

## 配置

```text
configs/experiment/exp063_structured_bicycle_quintic_map25_flatness_oracle_baseline.yaml
```

相对 exp051 的关键变化：

- `algorithm.training_semantics` 改为 `exp063_structured_bicycle_quintic_map25_flatness_oracle_baseline`；
- 显式固定 `gather_point.search_method=terrain_aware_multiresolution`；
- 局部搜索使用 `9×9` 粗网格、两层 `5×5` refinement，并启用 `33×33` 全局 fallback 与 beam refinement；
- 平整度圆盘半径为 `0.75 m`，使用 3 圈、每圈 12 点，加中心点共 37 点；
- 平整度要求为 `height_range <= 0.18 m` 且 `max_slope <= 0.25`；
- `gather_point.require_flat_for_success=true`；
- 独立 proxy 评估配置为 `1024 env / 320 steps`；high-fidelity evaluation 仅在 proxy passed 后触发。

保持不变：

- Actor/Critic 为 `branched_v1 / structured_v1`；
- observation schema 为 `ego_v3_local_terrain_grid`，Actor/Critic 接口仍为 `86 / 54`；
- bicycle proxy、quintic trajectory、`25 m × 25 m` 随机 lunar crater 地图；
- exp051 的 reward、subgoal filter、control safety、PPO 超参和 local initial-state curriculum；
- seed23 pure RL，40M 预算为 `20480` timesteps，即 `41,943,040` env steps。

## 严格标准

aggregate proxy strict gate 保持为：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

其中新语义下的 episode success 不再只是几何集合。instant success 同时要求：

```text
dmax <= 1.25 m
dispersion <= 0.30
all rover speed <= 0.25
min_pairwise_distance >= 0.42 m
centroid footprint height_range <= 0.18 m
centroid footprint max_slope <= 0.25
```

全部 instant gates 还必须连续保持 `8` 个 control steps。任一步平整度不满足都会清空 success hold count。

## 结果表

| seed | run_id | 预算 / 评估 | checkpoint | final eval | strict |
| --- | --- | --- | --- | --- | --- |
| 1023 eval | exp051 旧 checkpoint 新语义复评 | `1024 env / 320 steps` | exp051 `best.pt` / `ppo_timestep_013312.pt` | dmax `0.1465`、success `0.1104`、collision `0.0068`、timeout `0.8828`、final flatness `0.1172`、oracle feasible `1.0` | 未通过；success、timeout 失败 |
| 23 | `smoke_seed23_128_cuda_flatness_oracle_baseline` | `128` timesteps / `32,768` env steps；`256 env / 64 steps` eval | `ppo_timestep_000128.pt` / `best.pt` | dmax `0.8102`、success `0`、collision `0`、timeout `0`、finished `0`、final flatness `0.1055`、oracle feasible `1.0` | 工程 smoke；未通过，不能作为收敛结果 |
| 23 | `screen_seed23_4m_flatness_oracle_baseline` | `2048` timesteps / `4,194,304` env steps；`512 env / 320 steps` eval | `ppo_timestep_001536.pt` / `best.pt` | dmax `0.3136`、success `0.0176`、collision `0.0391`、timeout `0.9434`、final flatness `0.0879`、oracle feasible `1.0` | 未通过；四项 aggregate gates 全部失败 |
| 23 | `pure_rl_seed23_40m_flatness_oracle_baseline` | `20480` timesteps / `41,943,040` env steps；1024 env / 320 steps final eval | `ppo_timestep_012288.pt` / `best.pt` | dmax ratio `0.1539`、success `0.0820`、collision `0.0088`、timeout `0.9092`、final flatness `0.0957`、oracle feasible `1.0` | 未通过；success、timeout 失败 |

smoke 的 timeout 为 `0` 是因为 `64` 步评估内没有 episode 完成，`finished_rate=0`，不能解释为通过 timeout gate。4M screen 已产生集合学习信号，但仍同时存在收缩不足、success 极低、collision 超标和大量 timeout，不能从短预算结果推断 40M 会通过。

## 失败分析

### exp051 新语义复评

exp051 旧策略的 dmax 和 collision 仍可通过，但 success 从旧语义的 `0.9883` 降为 `0.1104`，timeout 从 `0.0098` 升为 `0.8828`。同时 oracle 搜索可行率为 `1.0`，timeout episode 的质心圆盘平均高度范围为 `0.2571 m`、平均最大坡度为 `0.4674`，均超过新阈值。

因此旧 exp051 的“仅差约 1% timeout”结论只适用于旧 gate；在新语义下，平整位置到达与稳定保持是主要未解决问题。

### exp063 4M screen

4M screen 的 dmax ratio 从随机/短 smoke 水平收缩到 `0.3136`，说明策略已经出现靠拢信号，但仍高于 `0.20`。最终平整率只有 `0.0879`，success `0.0176`；collision `0.0391` 也超过 `0.02`，timeout `0.9434`。

这意味着早期训练阶段不仅受 flatness gate 限制，几何收缩、末端安全和稳定保持也尚未成熟。应等待完整 40M 独立评估，不应根据训练 reward、单个 checkpoint 或短预算 screen 宣称成功。

## 产物路径

旧 exp051 新语义复评：

```text
outputs/runs/exp051_structured_bicycle_quintic_map25_ppo_stability/pure_rl_seed23_40m_structured_bicycle_quintic_map25_ppo_stability/metrics/final_eval_proxy_flatness_oracle_recheck_seed1023.json
```

exp063 CUDA smoke：

```text
outputs/runs/exp063_structured_bicycle_quintic_map25_flatness_oracle_baseline/smoke_seed23_128_cuda_flatness_oracle_baseline/
```

exp063 4M screen：

```text
outputs/runs/exp063_structured_bicycle_quintic_map25_flatness_oracle_baseline/screen_seed23_4m_flatness_oracle_baseline/
```

exp063 40M 正式 run：

```text
outputs/runs/exp063_structured_bicycle_quintic_map25_flatness_oracle_baseline/pure_rl_seed23_40m_flatness_oracle_baseline/
  config/experiment.yaml
  checkpoints/best.pt
  metrics/summary.json
  metrics/train_metrics.jsonl
  metrics/eval_metrics.json
  metrics/final_eval_proxy.json
  metrics/strict_acceptance.json
  metrics/checkpoint_status.json
  figures/training_curves.png
  figures/candidate_eval_curves.png
  figures/terrain_height_map.png
  videos/proxy_eval_rollout.gif
  tensorboard/
  run_manifest.json
```

训练进行期间只有已经落盘的 config、train metrics 和 TensorBoard 等中间产物可用于监控。`final_eval_proxy.json`、`strict_acceptance.json`、最终曲线和 GIF 必须等训练及独立评估完成后再核验。

## 结论

exp063 已完成 CUDA smoke、4M screen 与 40M formal run，均未通过 strict gate。40M 虽将 dmax 与 collision 压到 gate 内，但实际质心平整度/集合成功仍很低，timeout 占 `90.92%`。

exp063 的价值是建立冻结 terrain-aware oracle 与 flatness success 语义的基线，并证明“只让 Critic/reward 看见搜索点”不足以让 Actor 在随机地形上到达可行平地。后续不触发 high-fidelity evaluation。

## 下一步

1. 将真正的搜索点以不含世界坐标的局部执行目标提供给 Actor，并保持 actual-centroid flatness gate 不变。
2. 在后续目标执行实验中按 flatness、dmax、min-pairwise、collision 分解 timeout/终止原因。
3. 只有 proxy strict gate 通过后，才进入 PhysX / Jackal high-fidelity evaluation；单 seed proxy 结果仍不能替代多 seed 稳健性验证。
