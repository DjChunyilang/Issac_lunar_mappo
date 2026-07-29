# exp076：宽触发共同质心校正（4M reject）

## 目的

实际成功仍以团队质心的平整圆盘为准；专属对称槽位保证平均位置等于 terrain-aware 搜索点，但策略末段可能整体漂移。exp076 在不改变槽位相对几何的前提下，向所有 rover 子目标加入同一个、限幅的“槽位均值减实际质心”偏移。

## 配置

配置为 `configs/experiment/exp076_structured_bicycle_quintic_map25_terminal_center_correction.yaml`。dmax/dispersion 触发倍数均为 `1.75`，最大偏移 `0.35 m`，gain `0.55`；成功 gate、`0.42 m` 槽位与 terrain-aware 搜索不变。

## 结果

固定 exp073 best checkpoint、seed `11023` 的后验对照为 dmax ratio/success/collision/timeout=`0.2054/0.6602/0/0.3398`，共同校正 active fraction=`0.2269`。success/flatness 有提高，但 dmax 已不能通过。

同配置 seed `23`、4,194,304 env steps 的训练终评为：

| dmax ratio | success | collision | timeout | actual flatness | strict |
| ---: | ---: | ---: | ---: | ---: | --- |
| `0.2234` | `0.6250` | `0.0000` | `0.3750` | `0.7480` | 未通过 |

训练没有保持后验的 dmax 表现，故不以该 4M run 作为候选。

## 收紧扫描

exp077 将有效最大平移缩至 `0.10 m`，后验为 `0.2035/0.6367/0/0.3633`；exp078 将触发收紧到真实 dmax/dispersion 门内，后验为 `0.2007/0.6289/0/0.3711`；exp079 再把 gain 降至 `0.45`，未改善。结论是宽触发破坏紧凑度，单纯减小 gain 不能解决问题。

## 产物路径

- 后验评测：`outputs/runs/exp076_structured_bicycle_quintic_map25_terminal_center_correction/counterfactual_exp073_checkpoint_eval.json`
- 4M run：`outputs/runs/exp076_structured_bicycle_quintic_map25_terminal_center_correction/screen_seed23_4m_terminal_center_correction/`
- 曲线：同一 run 的 `figures/training_curves.png`、`figures/candidate_eval_curves.png`，及实验级 `figures/exp073_vs_exp076_training_curves.png`
- 收紧扫描：`outputs/runs/exp077_structured_bicycle_quintic_map25_center_correction_limited/`、`outputs/runs/exp078_structured_bicycle_quintic_map25_strict_terminal_center_correction/`、`outputs/runs/exp079_structured_bicycle_quintic_map25_strict_center_correction_gain45/`

## 结论

共同质心校正保留为可选执行机制，但只保留严格触发版本；宽触发版本 reject，不启动 formal long run 或 PhysX。
