# exp069：对称槽位 + 硬定向安全投影

## 目的

exp067 已证明 terrain-aware 对称槽位能建立真实集合点与队形中心的一致性，但 `95/512` 条轨迹以最小两两距离失败。exp069 只收紧低层 closing-direction safety projection：在预计越过 `0.42 m` 成功安全间距前允许朝向邻居的 rover 完全刹停，同时不缩放向外运动的 rover。真实集合成功定义、平整度 gate、Actor/critic 和 BC teacher 都保持不变。

## 配置与严格标准

- 配置：`configs/experiment/exp069_structured_bicycle_quintic_map25_oracle_slots_hard_safety.yaml`
- 训练：`2048` rollout steps，`2048` env，即 `4,194,304` env steps；seed `23`。
- 终评：seed `11023`，`512` env，`320` steps；Actor/Critic 为 `ego_v6_gather_slot_goal` 的 `89/54`。
- strict：`dmax_reduction_ratio <= 0.2`、`success_rate >= 0.9`、`collision_rate <= 0.02`、`timeout_rate == 0`。

相对 exp067，唯一控制改动是 projection：activation `0.75 m`、stop `0.42 m`、horizon `0.60 s`、strength `1.0`、minimum linear scale `0.0`，并启用 closing-only directional mask。

## 结果

| run | dmax ratio | success | collision | timeout | final flatness | strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `screen_seed23_4m_oracle_slots_hard_safety` | `0.1969` | `0.5664` | `0.0000` | `0.4336` | `0.6504` | 未通过 |

final `dmax=1.1897 m`、dispersion=`0.3083`、nearest=`0.5495 m`。安全投影在 `27.27%` 的控制步介入，平均线速度缩放为 `0.8337`。搜索点可行率为 `1.0`。

## Gate 诊断

固定 seed `11023` 的逐 episode 诊断得到 `290` success、`0` collision、`222` timeout：

- 所有 timeout 的最小两两间距都通过；硬定向刹停确实消除了 exp067 的末段碰撞主因。
- timeout 中 `179/222` 仍未通过实际质心平整度，`94/222` 未过 dmax，`97/222` 未过 dispersion；它们的平均 max slope 为 `0.2890`，高于 `0.25` gate。
- success 轨迹的平均 dmax/dispersion/最小间距分别为 `1.0513/0.2373/0.5501 m`，实际质心平整度均通过。

因此失败不来自“搜索点不平”或“间距不足”，而是安全约束下部分队形不能在 320 步内同时回到紧凑且平整的真实质心区域。

## 产物路径

```text
outputs/runs/exp069_structured_bicycle_quintic_map25_oracle_slots_hard_safety/
  screen_seed23_4m_oracle_slots_hard_safety/
    metrics/final_eval_proxy.json
    metrics/success_gate_diagnostics.json
    figures/training_curves.png
    figures/candidate_eval_curves.png
    figures/terrain_height_map.png
    videos/proxy_eval_rollout.gif
    run_manifest.json
```

GIF 的 seed `11023` 是一个 timeout 示例：最终 dmax 与间距已经通过，但质心圆盘 max slope 为 `0.2505`，刚好高于硬阈值；它只用于失效分析，strict 结论以 JSON 为准。

## 结论与下一步

exp069 是当前 v6 槽位路线中安全性最强的 screen：collision 已为零，但 success/timeout 仍未过 strict，不启动 formal long run 或 PhysX。下一次改动应直接针对“队形中心回到搜索点的最后小位移”，并保留真实质心平整度与 `0.42 m` 间距 gate；不能以放宽平整度或退回共同几何中点代理来取得表面成功。
