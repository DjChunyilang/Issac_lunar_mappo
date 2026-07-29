# exp094：96 秒执行时域

## 目的

将 exp092 的 `64 s/320` control-step 执行和评估时域放宽到 `96 s/480`，验证 timeout 是否受 episode 时间预算限制，并将该时域作为后续本配置族的标准时域。

## 配置

配置为 `configs/experiment/exp094_structured_bicycle_quintic_map25_flatness_center_early_radius35_time96.yaml`，继承 exp093/exp092 的实际质心平整度 gate、terrain-aware 最优集合点搜索、`0.35 m` 对称槽位和早触发共同质心校正。仅同步修改：

- `simulation.episode_length_s: 96.0`
- `experiment.eval_steps: 480`
- `evaluation.proxy_eval.steps: 480`
- `evaluation.high_fidelity_eval.steps: 480`

严格 proxy gate 未改变：dmax ratio `<=0.2`、success `>=0.9`、collision `<=0.02`、timeout `=0`。

## 结果

使用 exp092 的 `BC32` checkpoint，在 seed `11023`、1024 环境、480 steps 的独立后验复评：

| checkpoint | dmax ratio | success | collision | timeout | 实际集合点平整率 | strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| exp092 `BC32` | `0.1837` | `0.8594` | `0.0000` | `0.1406` | `0.9092` | 未通过 |

相对原 `64 s/320` 的 `0.1910/0.7002/0/0.2998`，success 增加 `15.92` 个百分点、timeout 降低 `15.92` 个百分点；dmax 和 collision 保持达标。

## 失败分析

96 秒解决了一部分慢收敛 episode，但 `success_rate` 和 `timeout_rate` 仍未通过 strict gate。该选择是执行时间预算调整，不把 timeout 判为成功，也不放宽实际集合点平整度、几何、速度或连续 hold 条件。

## 产物路径

- 配置：`configs/experiment/exp094_structured_bicycle_quintic_map25_flatness_center_early_radius35_time96.yaml`
- 独立复评：`outputs/runs/exp094_structured_bicycle_quintic_map25_flatness_center_early_radius35_time96/counterfactual_exp092_bc32_episode480_eval_1024.json`
- 128 秒上界对照与 PPO 探针：`docs/experiments/exp_097_structured_bicycle_quintic_map25_time_horizon_ppo_probe.md`

## 结论与下一步

后续本配置族固定使用 `96 s/480` 时域。当前策略仍是 exp092 的 BC32 candidate；在该时域内针对“实际点不平整”与“几何未收紧”的 timeout 设计末端干预后，再启动新的训练 run。
