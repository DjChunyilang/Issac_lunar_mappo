# exp065 structured bicycle quintic map25 terrain-aware 集合点目标广播

## 目的

exp063 的 terrain-aware 搜索在全部评测场景中都能找到满足平整度约束的集合点，但 exp063/exp064 的 Actor 被设计为完全看不到该点。exp064 的 4M screen 验证了这种信息结构的限制：最优点可行率为 `1.0`，但 502 个 timeout 中有 477 个实际质心圆盘不平整，同时 470 个仍不满足 dispersion。仅增加 actual-centroid flatness potential 没有把多车引导到同一块可行平地。

这不是放宽 success gate，而是补齐可执行的规划接口：terrain-aware 搜索先生成真正的最优集合点，系统将其以共享执行目标广播给每辆 rover；策略只接收自身机体坐标系中的相对向量和归一化距离，不接收全局地图坐标、oracle 搜索目标值、平整度诊断或 Critic state。这样保留每车本地连续控制，同时让随机地图上的“去同一个平地集合”成为对 Actor 可观测的目标条件决策问题。

## 配置

```text
configs/experiment/exp065_structured_bicycle_quintic_map25_oracle_goal_broadcast.yaml
```

相对 exp064 的唯一行为变化为：

```yaml
task:
  explicit_goal_in_execution: true
observation:
  schema_version: ego_v5_gather_site_goal
algorithm:
  actor_architecture: branched_v3
```

`ego_v5_gather_site_goal` 在原 `ego_v3_local_terrain_grid` 的 86 维输入后追加 3 维：

\[
g_i=\left[
  \operatorname{clip}\left(\frac{R(-\psi_i)(p_\star-p_i)}{s},-2,2\right),
  \operatorname{clip}\left(\frac{\lVert p_\star-p_i\rVert}{s},0,2\right)
\right],
\]

其中 (p_\star) 是 terrain-aware multiresolution search 返回的集合点，(p_i,\psi_i) 是第 (i) 辆车的位置和朝向，(R(-\psi_i)) 将向量旋转到机体系。尺度

\[
s=\max(d_{\mathrm{gate}},r_{\mathrm{spawn,max}}+m_{\mathrm{search}})
\]

取自正常集合搜索包络，避免全局世界边界把目标信号压得过小。该特征不包含 `p_star` 的 world XY，也不包含 search objective / feasibility / height / slope。Actor 维度由 `86` 变为 `89`，使用专用 3→16 goal encoder 的 `branched_v3`；集中 Critic 保持 `54` 维 `structured_v1`。

为防止配置表面开启、实际没有目标输入，loader 强制如下双向契约：`task.explicit_goal_in_execution=true` 当且仅当 schema 为 `ego_v5_gather_site_goal`。exp063/exp064 仍保持 `explicit_goal_in_execution=false`，其 Actor 对 oracle 变化仍严格不敏感。

其余 terrain-aware 搜索、37 点实际质心平整度 gate、exp064 的 gated flatness potential、reward、filter、control safety、PPO、initial-state curriculum、seed 与预算均保持不变。

## 严格标准

aggregate proxy strict gate 不变：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

episode success 仍要求 dmax、dispersion、速度、最小两两距离和实际质心 `0.75 m` 圆盘的 `height_range <= 0.18 m`、`max_slope <= 0.25` 同时成立，并连续保持 8 步。目标广播不替代、不放宽实际集合位置的平整度判定。

## 结果表

| seed | run_id | 预算 / 评估 | checkpoint | final eval | strict |
| --- | --- | --- | --- | --- | --- |
| 23 | `smoke_seed23_128_cuda_oracle_goal_broadcast_local_scale` | `128` timesteps / `32,768` env steps；`256 env / 64 steps` eval | `ppo_timestep_000128.pt` / `best.pt` | 数值有限，Actor/Critic 为 `89 / 54`，oracle feasible `1.0`；评估窗口未完成 episode，不能判定收敛 | 工程 smoke；未通过 |
| 23 | `screen_seed23_4m_oracle_goal_broadcast` | `2048` timesteps / `4,194,304` env steps；`512 env / 320 steps` eval | `best.pt` | dmax ratio `0.3795`、success `0.0020`、collision `0.2148`、timeout `0.7832`、final flatness `0.0645`、oracle feasible `1.0` | 未通过 |

## 产物路径

```text
outputs/runs/exp065_structured_bicycle_quintic_map25_oracle_goal_broadcast/
  smoke_seed23_128_cuda_oracle_goal_broadcast_local_scale/
  screen_seed23_4m_oracle_goal_broadcast/
```

每个完成 run 以各自 `metrics/final_eval_proxy.json`、`metrics/strict_acceptance.json` 和 `run_manifest.json` 为结果事实来源；曲线、terrain map 与 GIF 只用于诊断。

## 结论

exp065 已通过 CPU contract tests 与 CUDA smoke，并完成 4M screen；共享中心目标没有形成安全分散的多车终态，collision 与 timeout 均明显失败。它证明 Actor 需要的不只是公共可行中心，还需要与最优点一致的专属编队结构。

## 下一步

1. 用以 oracle 为中心、均值严格回到 oracle 的对称专属槽位替代共享中心直接收缩。
2. 保持 strict gate，优先按 actual-centroid flatness 与 min-pairwise 分解后续失败。
3. 只有 proxy strict gate 通过后，才触发 PhysX / Jackal high-fidelity evaluation。
