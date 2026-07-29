# exp071：缩小对称槽位半径至 0.40 m（BC reject）

## 目的

exp069 的 `0.45 m` 槽位半径提供了较大间距余量，但可能使队形末段不够紧凑。exp071 仅将 `execution_slot_radius` 降至 `0.40 m`，其正方形相邻槽位距离为 `0.566 m`，仍高于 `0.42 m` success 安全下限。

## 结果

固定 seed `11023`、512 环境、320 步的 128-update BC 快筛结果为：dmax ratio `0.1902`、success `0.5059`、collision `0`、timeout `0.4941`、实际质心平整率 `0.5664`。虽然 dmax 通过，但 success 低于 exp069 的 `0.5508` BC 对照，timeout 更高；不启动 4M。

## 结论

在当前硬定向 safety projection 下，`0.40 m` 不能提升末段收敛，反而降低实际质心落入平地的概率。后续保留 `0.42 m`，它仍提供约 `0.174 m` 的间距余量。

产物：`outputs/runs/exp071_structured_bicycle_quintic_map25_oracle_slots_radius40/bc_smoke_seed23_slots_radius40/`。
