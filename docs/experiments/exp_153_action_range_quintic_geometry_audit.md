# exp153：动作范围与quintic几何分离审计

## 目的

exp152确认：已有避碰quintic轨迹能够稳定传递到低层控制，但t2048在碰撞前8步的无约束局部可行动率只有约`0.665`。本实验进一步区分：

1. `delta=0.15`局部邻域覆盖不足，但现有完整动作范围内存在避碰动作；
2. 现有完整动作范围仍不足；
3. 子目标终端位置可分离，但quintic路径在途中发生交叉；
4. quintic曲线本身限制了可行动性。

本实验只使用冻结旁路候选，不修改策略、动作接口、轨迹生成器或控制器。

## 冻结设置

- 配置、checkpoint、数据种子、环境数和时域与exp152完全相同；
- 局部候选仍为四个`delta=0.15`单维正负扰动；
- 全范围轴向候选将半径或方位角单独设置为`-1`或`+1`；
- 全范围网格为：

\[
\{-1,a_\rho,+1\}\times\{-1,a_\beta,+1\}
\setminus\{(a_\rho,a_\beta)\},
\]

共8个候选，不增加Actor动作维度，也不扩大现有动作边界。

安全距离改善门限保持`0.02 m`。分析时域保持碰撞前`1、2、4、8、16`步。

## 几何分层

对每个候选子目标分别计算：

1. `local_quintic`：局部四候选的物理时间对齐quintic距离，必须精确重建exp152无约束层；
2. `axis_quintic`：全范围轴向候选的quintic距离；
3. `grid_quintic`：全范围8候选的quintic距离；
4. `grid_line`：相同子目标、相同物理时域下的直线路径参考距离；
5. `grid_endpoint`：只比较规划终端位置的车辆对距离。

`line`和`endpoint`只用于定位几何损失，不能替代实际quintic执行或strict evaluation。

定义：

- `range_recovery_rate`：`grid_quintic`可行动而`local_quintic`不可行动；
- `joint_dimension_recovery_rate`：最佳网格候选同时改变两个动作维度的比例；
- `quintic_geometry_loss_rate`：`grid_line`可行动而`grid_quintic`不可行动；
- `path_crossing_loss_rate`：`grid_endpoint`可行动而`grid_line`不可行动；
- 动作边界率：参与车辆采样动作满足 $|a_k|\ge0.85$ 的比例。

## 预注册工程门限

四个组合必须满足：

- 至少100个完整碰撞episode；
- Actor摘要、探针动作和环境执行动作保持不变；
- 非参与车辆对目标碰撞对的影响不超过`1e-6`；
- `local_quintic`对exp152无约束层的全部时域重建误差不超过`1e-6`；
- 候选数量固定为局部4、轴向4、网格8；
- 所有指标有限。

## 预注册决策树

1. 若8步`grid_quintic`跨组合最小值不低于`0.80`，且`range_recovery_rate`最小值不低于`0.15`，判定为`local_coverage_bottleneck`。只允许形成一次训练期动作分布覆盖审计，不直接修改动作范围。
2. 否则，若8步`grid_quintic`最小值低于`0.70`，但`grid_line`最小值不低于`0.70`且`quintic_geometry_loss_rate`最小值不低于`0.15`，判定为`quintic_geometry_bottleneck`。只允许形成单一quintic几何修正计划。
3. 否则，若8步`grid_line`最小值低于`0.70`，但`grid_endpoint`最小值不低于`0.70`且`path_crossing_loss_rate`最小值不低于`0.15`，判定为`path_crossing_or_timing_bottleneck`。
4. 否则，若8步`grid_endpoint`最小值低于`0.70`，判定为`subgoal_reachability_bottleneck`。
5. 其余结果判定为`mixed_action_geometry_bottleneck`，不授权工程或训练改动。

exp153本身不授权4M、12M或40M。只有四组合一致通过某一分支后，才允许预注册对应的单变量工程验证；仍不得同时改Actor、轨迹和reward。

## 明确不做

- 不扫描动作边界、扰动间隔、轨迹切线系数或路径点数；
- 不执行任何旁路候选；
- 不把line或endpoint作为在线规划器；
- 不修改reward或训练信用；
- 不恢复安全投影；
- 不以离线可达率替代strict pass。

## 产物路径

```text
outputs/runs/exp153_action_range_quintic_geometry_audit/
  frozen_exp150_dual_checkpoint_dualseed/
    config/experiment.yaml
    metrics/action_range_quintic_geometry.json
    run_manifest.json
  _suite/
    metrics/suite_summary.json
    run_manifest.json
```

## 当前状态

正式冻结审计已经完成。四个checkpoint—种子组合全部通过工程门限，局部层对exp152的五个时域重建误差均为0，Actor与环境执行保持不变。

| checkpoint—seed | 8步局部quintic | 8步全网格quintic | 8步line | 8步endpoint | 范围恢复 | 途中交叉损失 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t1024—46023 | 0.8436 | 0.9110 | 0.8620 | 1.0000 | 0.0675 | 0.1380 |
| t1024—47023 | 0.8475 | 0.9120 | 0.8563 | 1.0000 | 0.0645 | 0.1437 |
| t2048—46023 | 0.6648 | 0.7841 | 0.6534 | 1.0000 | 0.1193 | 0.3466 |
| t2048—47023 | 0.6676 | 0.7536 | 0.6275 | 1.0000 | 0.0860 | 0.3725 |

正式状态为`mixed_action_geometry_bottleneck`：

- 全动作网格的8步可行动率最小值为`0.7536`，高于局部层但低于`0.80`；
- 范围恢复率最小仅`0.0645`，低于`0.15`，因此不能归因于局部候选覆盖不足；
- line可行动率最小为`0.6275`，没有稳定优于quintic；quintic几何损失仅`0–1.14%`，不能归因于quintic曲线；
- endpoint可行动率四组合均为`1.0`，说明现有子目标范围足够；
- endpoint到完整路径的途中交叉损失在t1024为约`14%`，到t2048升至`34.7%–37.2%`；
- 轴向全范围候选与完整二维网格的可行动率完全相同。虽然最佳网格候选约`75%–83%`同时改变两个维度，但二维联动没有新增可行动episode；
- 动作边界率由t1024约`5.5%`升至t2048约`14%–15%`，但不足以形成单一动作范围结论。

因此不修改动作边界、不替换quintic，也不启动训练。结果更符合“单车路径分别可达，但相互轨迹缺少联合协调”的解释；该解释尚未通过反事实联合动作验证。下一步若继续，只允许预注册冻结的最终碰撞对双车联合动作干预，验证同时改变两辆车能否消除途中交叉；不得直接增加在线协调模块。

正式汇总位于：

```text
outputs/runs/exp153_action_range_quintic_geometry_audit/
  _suite/metrics/suite_summary.json
  frozen_exp150_dual_checkpoint_dualseed/
    metrics/action_range_quintic_geometry.json
    run_manifest.json
```
