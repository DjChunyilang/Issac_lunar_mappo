# exp066：oracle 径向环形 teacher（reject）

exp066 在共享 terrain-aware 目标广播上加入 `oracle_ring` BC teacher：每辆 rover 按其当前相对 oracle 的径向位置选择一个半径 `0.45 m` 的临时环形目标。

4M screen 的 independent final eval 为：dmax ratio `0.3393`、success `0.0352`、collision `0.3184`、timeout `0.6465`、final flatness `0.1074`，oracle feasible `1.0`，未通过 strict gate。

关键缺陷是径向槽位随当前 rover 位置变化，四个槽位的平均位置不保证等于搜索点；最终队形质心会偏离可行平地。seed11023 的代表 rollout 在 320 步 timeout，最终实际质心 `height_range=0.2027`、`max_slope=0.6146` 均失败，而同 episode oracle 点满足搜索平整度约束。这促成 exp067 的固定对称、最小行驶代价分配槽位。

产物位于：

```text
outputs/runs/exp066_structured_bicycle_quintic_map25_oracle_ring_bc/
  screen_seed23_4m_oracle_ring_bc/
    metrics/final_eval_proxy.json
    figures/training_curves.png
    figures/candidate_eval_curves.png
    figures/terrain_height_map.png
    videos/proxy_eval_rollout.gif
```
