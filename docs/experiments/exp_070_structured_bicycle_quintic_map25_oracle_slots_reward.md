# exp070：专属槽位对齐的 oracle 进度奖励（reject）

## 目的

exp069 的 Actor 已读取专属对称槽位，但 dense oracle-progress reward 仍计算所有 rover 到共同 `oracle_point` 的距离，理论上会与 `0.45 m` 安全槽位相冲突。exp070 新增显式、默认关闭的 `task.execution_slot_reward_target=true`：仅在 v6/v7 槽位 schema 下，oracle 进度改为每辆 rover 到其已分配槽位的距离。实际质心平整度、搜索点、Critic 状态和终止 gate 均不变。

## 配置与严格标准

- 配置：`configs/experiment/exp070_structured_bicycle_quintic_map25_oracle_slots_reward.yaml`
- 对照：除实验名、training semantics 和上述 task flag 外，与 exp069 完全相同。
- 训练：seed `23`、`2048` rollout steps、`2048` env，即 `4,194,304` env steps。
- 终评：seed `11023`、`512` env、`320` steps；strict 阈值与 exp069 相同。

16-update BC 闭环烟测的 loss 从 `0.9034` 降至 `0.3846`，确认新 reward target 的配置与训练链路可用；该小预算不用于策略优劣结论。

## 结果

| experiment | dmax ratio | success | collision | timeout | final flatness | safety applied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exp069 shared-site reward | `0.1969` | `0.5664` | `0.0000` | `0.4336` | `0.6504` | `0.2727` |
| exp070 assigned-slot reward | `0.1975` | `0.5645` | `0.0000` | `0.4355` | `0.6563` | `0.2791` |

exp070 的 final dmax/dispersion/nearest 为 `1.1930 m` / `0.3086` / `0.5468 m`。dmax 与 collision 两项 strict check 通过，success 与 timeout 未通过，因此 checkpoint 状态为 `candidate`。

## Gate 诊断

固定 seed `11023` 记录 `289` success、`0` collision、`223` timeout。timeout 中的最终 gate 失败计数为：flatness `176`、dmax `98`、dispersion `100`、min-pairwise `0`。与 exp069 的 `179/94/97/0` 基本相同；两者都表明硬安全投影已守住间距，未通过的主要原因仍是队形中心没有同时满足紧凑和平整条件。

## 产物路径

```text
outputs/runs/exp070_structured_bicycle_quintic_map25_oracle_slots_reward/
  screen_seed23_4m_slots_reward/
    metrics/final_eval_proxy.json
    metrics/success_gate_diagnostics.json
    metrics/strict_acceptance.json
    figures/training_curves.png
    figures/candidate_eval_curves.png
    figures/terrain_height_map.png
    videos/proxy_eval_rollout.gif
    run_manifest.json
```

该 GIF 同样是 seed `11023` timeout：dmax 已为 `0.9111 m`、间距仍安全，但实际质心 max slope=`0.2542` 超过 `0.25`。它支持诊断，不构成成功证据。

## 结论与下一步

该奖励语义改动没有带来实质改善：相对 exp069 的 success 少 `1/512`，timeout 多 `1/512`，安全投影介入略多。故 exp070 为 reject，不启动 formal long run 或 PhysX。后续若继续，应测试能让队形整体在末段朝 terrain-aware 搜索点做受限共同平移的可观测控制/目标设计，而不是只重新定义同一槽位的 dense 距离奖励。
