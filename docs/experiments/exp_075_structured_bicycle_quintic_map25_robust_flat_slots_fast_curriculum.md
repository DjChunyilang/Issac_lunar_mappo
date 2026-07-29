# exp075：加速初始状态课程（4M reject）

## 假设

4M screen 只有 `2048` rollout step，而此前初始状态课程的 `warmup=4096`、`ramp=8192` 使训练主要停留在较易出生半径。exp075 只将**初始状态**课程改为 `warmup=256`、`ramp=512`，让训练的大部分阶段暴露于与终评一致的完整 `2.4–3.4 m` 出生分布；搜索、平整 gate、`0.42 m` 槽位、safety projection、奖励与子目标过滤器课程均保持 exp073 不变。

## 结果

seed `23`、4,194,304 env steps 的终评（seed `11023`、512 环境、320 步）为：dmax ratio `0.2050`、success `0.6055`、collision `0`、timeout `0.3945`、实际质心平整率 `0.7070`。严格检查仅 collision 通过。

相对 exp073 的 `0.1997/0.6113/0/0.3887`，四项中没有改善，且 dmax 再次超过 `0.20`。训练期碰撞保持为零，但没有转化为独立 full-distribution 终评的更高集合成功。

训练曲线与 exp073 的对照已写入 `figures/exp073_vs_exp075_training_curves.png`；单 run 的训练/候选评测曲线也在本 run 的 `figures/` 下。

## 结论

“screen 课程过慢造成的训练/评估出生分布断层”不是当前主要原因。exp075 reject，不启动 formal long run 或 PhysX；继续应直接处理安全槽位下的末段共同收敛，而不是单独继续调初始状态课程。

产物：`outputs/runs/exp075_structured_bicycle_quintic_map25_robust_flat_slots_fast_curriculum/screen_seed23_4m_fast_curriculum/`。
