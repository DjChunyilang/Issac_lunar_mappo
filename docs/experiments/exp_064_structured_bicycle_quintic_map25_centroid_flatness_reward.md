# exp064 structured bicycle quintic map25 actual-centroid flatness reward

## 目的

exp063 已把 terrain-aware 最优集合点搜索与实际团队质心平整度纳入成功判据，但其 reward 仍沿用 exp051：`reward.weights.flatness=0.0`，没有直接使用实际质心圆盘平整度变化作为训练信号。平整度主要通过 success hold 与 terminal 结果间接反馈，无法单独判断“末段缺少平整度 shaping”是否是训练瓶颈。

exp064 在 exp063 基础上只增加 actual-centroid flatness shaping，并保持 oracle 搜索、success gate、Actor/Critic、reward 其余分量、filter、control、PPO 和 initial-state curriculum 不变。该实验用于隔离检验实际集合位置的平整度进展信号；当前只完成实现与测试，训练尚未启动，不预设它会改善 success、timeout 或 strict 结果。

## 配置

```text
configs/experiment/exp064_structured_bicycle_quintic_map25_centroid_flatness_reward.yaml
```

相对 exp063 的唯一行为变化为：

```yaml
reward:
  weights:
    flatness: 1.0
  coefficients:
    centroid_flatness_progress: 2.0
    centroid_flatness_excess: 0.02
    centroid_flatness_dmax_multiplier: 2.0
```

设状态推进后实际团队质心圆盘的高度范围为 \(\Delta h_t\)，最大坡度为 \(s_t\)，则归一化平整度代价为：

\[
C_t=\operatorname{clip}\left(
\max\left(\frac{\Delta h_t}{0.18},\frac{s_t}{0.25}\right),
0,3
\right).
\]

\(C_t\le 1\) 与 success gate 的两项平整度硬约束完全等价。令几何成功阈值 \(d_{\mathrm{gate}}=1.25\,\mathrm{m}\)，激活倍数 \(m=2.0\)，则：

\[
a_t=\operatorname{clip}\left(
\frac{m d_{\mathrm{gate}}-d_{\max,t}}
{(m-1)d_{\mathrm{gate}}},
0,1
\right),
\qquad
P_t=a_tC_t,
\]

\[
r_t^{\mathrm{flat}}=
2.0(P_{t-1}-P_t)
-0.02a_t\operatorname{ReLU}(C_t-1).
\]

因此 `dmax >= 2.50 m` 时 \(a_t=0\)，`1.25 m < dmax < 2.50 m` 时线性增强，`dmax <= 1.25 m` 时 \(a_t=1\)。进展项是 gated potential 差 \(P_{t-1}-P_t\)：在激活度不变时对应实际质心平整度代价下降，在跨越激活边界时则同时结算前后两端的 activation。excess 项只在当前已激活且 \(C_t>1\) 时给出非正惩罚。

训练前代码审查将旧的 \(a_t(C_{t-1}-C_t)\) 改为上述 gated potential 差，避免策略利用 activation 边界循环取利。对任意往返轨迹，进展项满足：

\[
\sum_{t=1}^{T}(P_{t-1}-P_t)=P_0-P_T.
\]

若轨迹回到相同的 cost/activation 状态，则 \(P_T=P_0\)，进展 reward 精确抵消；额外 excess 项始终非正，因此这种循环不能获得正的累计 flatness reward。

该 shaping 使用每一步状态推进后的 `metrics.centroid`、`dmax` 与同一 terrain runtime 的圆盘查询，不读取 `oracle_point` 或 `oracle_search` 结果。它只进入 reward 与 telemetry，不增加 policy input 或 centralized state 字段；observation schema 仍为 `ego_v3_local_terrain_grid`，Actor/Critic 维度保持 `86 / 54`。

其余关键配置保持 exp063 不变：

- `terrain_aware_multiresolution` 搜索与全局 fallback；
- 半径 `0.75 m`、37 点实际质心圆盘；
- `height_range <= 0.18 m`、`max_slope <= 0.25` 的 flatness success gate；
- `branched_v1 / structured_v1`、bicycle proxy、quintic trajectory；
- `25 m × 25 m` 随机 lunar crater 地图；
- exp051 reward 的其余分量、subgoal filter、control safety 和 PPO 超参；
- seed23 pure RL；正式预算拟沿用 `20480` timesteps，即 `41,943,040` env steps。

项目默认配置仍将 `reward.weights.flatness`、`centroid_flatness_progress` 和 `centroid_flatness_excess` 设为 `0.0`。因此只有 exp064 这类显式启用配置才改变行为，旧实验不会被追溯性改写。

## 严格标准

aggregate proxy strict gate 保持为：

```text
dmax_reduction_ratio <= 0.20
success_rate >= 0.90
collision_rate <= 0.02
timeout_rate == 0
```

episode instant success 仍同时要求：

```text
dmax <= 1.25 m
dispersion <= 0.30
all rover speed <= 0.25
min_pairwise_distance >= 0.42 m
centroid footprint height_range <= 0.18 m
centroid footprint max_slope <= 0.25
```

全部 instant gates 必须连续保持 `8` 个 control steps。exp064 只改变训练 reward，不放宽任何 strict 或 episode success gate。

## 结果表

| seed | run_id | 预算 / 评估 | checkpoint | final eval | strict |
| --- | --- | --- | --- | --- | --- |
| 不适用 | implementation / tests | actual-centroid flatness reward、配置隔离与 telemetry contract | 不适用 | 实现与相关测试已通过；这不是训练结果 | 不适用 |
| 23 | `screen_seed23_4m_centroid_flatness_reward` | `2048` timesteps / `4,194,304` env steps；512 env / 320 steps final eval | `best.pt` | dmax ratio `0.2684`、success `0.0195`、collision `0`、timeout `0.9805`、final flatness `0.0684`、oracle feasible `1.0` | 未通过 |

## 失败分析

screen 未通过：成功仅 `10/512`，timeout `502/512`。标准 gate 诊断显示 timeout 的 flatness 失败 `477/502`、dispersion 失败 `470/502`、dmax 失败 `434/502`，而 oracle feasible 始终为 `1.0`。这说明 gated potential 本身不能解决 Actor 不知道该向哪块可行平地移动的问题；不继续该 reward-only 方向的 40M long run。

## 产物路径

配置与实验文档已存在：

```text
configs/experiment/exp064_structured_bicycle_quintic_map25_centroid_flatness_reward.yaml
docs/experiments/exp_064_structured_bicycle_quintic_map25_centroid_flatness_reward.md
```

已完成 screen 路径为：

```text
outputs/runs/exp064_structured_bicycle_quintic_map25_centroid_flatness_reward/
  screen_seed23_4m_centroid_flatness_reward/
```

该 run 的 `final_eval_proxy.json`、`success_gate_diagnostics.json`、曲线、terrain map、GIF 与 `run_manifest.json` 已生成；没有启动 40M formal run。

## 结论

exp064 的实现、配置隔离和 screen 已完成，但 shaping 没有改善 actual-centroid flatness 或 success，不能作为主线。下一步应改变 Actor 的可执行目标信息，而非继续放大 flatness reward。

## 下一步

1. 保持 hard gate，转向局部目标广播与队形槽位执行。
2. 仅在 screen 显示 flatness、collision、timeout 同步改善后，再启动 formal long run。
