# exp067：terrain-aware 对称槽位目标 + BC warm start

## 目的

exp065 的共享目标广播让 Actor 知道地形搜索点，但所有 rover 被引向同一点，末段容易互相穿越；exp066 的临时径向环形 teacher 又无法保证环形槽位的平均位置恰好回到搜索点。exp067 将真正的 terrain-aware `oracle_point` 转换为四个等角对称槽位，并在 reset 时枚举最小总初始行驶距离的分配。槽位固定到 episode 结束，四点的算术平均值严格等于 `oracle_point`。

这不是放宽集合定义：episode 成功仍以实际团队质心为中心做 `0.75 m`、37 点地形圆盘评估，并同时要求 dmax、dispersion、速度、最小两两间距与平整度 hold 8 步。

## 接口与配置

```text
configs/experiment/exp067_structured_bicycle_quintic_map25_oracle_slots_bc.yaml
```

- `ego_v6_gather_slot_goal`：Actor 维度 `89`，在基础 86 维后追加本 rover 到专属槽位的车体系 `[local_dx, local_dy, normalized_distance]`；Critic 保持 `54` 维。
- `gather_point.execution_slot_radius=0.45 m`：相邻槽位间距约 `0.636 m`，高于 `0.42 m` 成功安全下限。
- `oracle_slots` teacher：按已分配的槽位产生 BC 标签；128 次 BC 后再做 shared-joint MAPPO。
- Actor 不接收世界 XY、搜索 score、可行性或平整度诊断；`oracle_search` 仍只作为训练特权计算与评估事实。

## 结果

| run | 预算 / 评测 | 关键结果 | strict |
| --- | --- | --- | --- |
| `bc_smoke_seed23_oracle_slots_radius45` | 128 次 BC；512 env、320 steps | BC loss `0.8983 → 0.0603`；success `0.3809`、collision `0.3770`、timeout `0.2422` | 未通过，仅闭环 warm-start 对照 |
| `screen_seed23_4m_oracle_slots_bc_radius45` | 2048 timesteps / 4,194,304 env steps；seed11023、512 env、320 steps | dmax ratio `0.1819`、success `0.4668`、collision `0.1855`、timeout `0.3477`、actual-centroid flatness `0.5566`、oracle feasible `1.0` | 未通过 |

screen 的 success 比 exp066 的 4M oracle-ring BC screen（`0.0352`）明显提高，且 dmax ratio 通过；但 collision 和 timeout 仍远高于 aggregate gate。

## Gate 诊断

标准化诊断（seed11023）记录 `239` success、`95` collision、`178` timeout：

- collision 的 final dmax/dispersion 通过率分别为 `92.6% / 88.4%`，但最小两两间距通过率为 `0%`；主要是末段安全闭环失效。
- timeout 中 flatness 失败 `150/178`，dmax 失败 `68/178`，dispersion 失败 `81/178`，最小间距失败 `63/178`。搜索点可行率为 `1.0`，因此平整度问题来自实际质心没有充分回到搜索点，而非搜索无解。
- 一条 seed11023 成功 rollout 在第 230 步结束，final dmax `1.1051 m`，实际质心平整度 `height_range=0.1571`、`max_slope=0.2493`，均在 hard gate 内。

结论是：槽位分配解决了“真实最优点在哪里、队形中心应落在哪里”的主要信息问题，但尚缺少稳健的共同中心校正与末段相互避碰。因此不启动 40M formal run，也不触发 PhysX high-fidelity evaluation。

## 产物

```text
outputs/runs/exp067_structured_bicycle_quintic_map25_oracle_slots_bc/
  screen_seed23_4m_oracle_slots_bc_radius45/
    metrics/final_eval_proxy.json
    metrics/success_gate_diagnostics.json
    figures/training_curves.png
    figures/candidate_eval_curves.png
    figures/terrain_height_map.png
    videos/proxy_eval_rollout.gif
```

`run_manifest.json` 是完整产物索引；所有结论以 `final_eval_proxy.json` 和 `success_gate_diagnostics.json` 为准。

## 下一步

在不改变 actual-centroid flatness gate 的前提下，增加“共同搜索中心 + 专属槽位”的局部双目标特征，并单独评估其对 centroid-flatness timeout 与最小间距碰撞的影响；只有 screen 证明安全/超时同步改善，才进入更长训练。
