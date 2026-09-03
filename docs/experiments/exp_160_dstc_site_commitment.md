# exp160：D-STC共同站点证书H0

更新时间：2026-08-20。

## 目的

本实验推进路线一的第一个可执行阶段：在不训练策略、不修改Actor动作和不使用Oracle候选的前提下，实现并审计以下静态核心：

```text
本车局部地形
→ 有验证余量的平地区域proposal
→ 对候选排列不敏感的物理关联
→ 保守共同站点证书
→ 四车全签commit
```

本实验不把证书接入Actor，也不生成教师动作。它只回答候选证书是否会产生伪平地、坐标捷径或split-brain。

## 配置

配置与入口：

```text
configs/experiment/exp160_dstc_h0_certificate.yaml
scripts/audit_exp160_site_commitment.py
```

固定输入为exp156的六分层1152场景。每车最多保留4个局部候选；成功需要的平整半径为 $0.75\ \mathrm m$，候选先验证 $1.25\ \mathrm m$ 完整圆盘，并显式预留 $0.10\ \mathrm m$ 相对位姿误差。

对proposal中心 $c_i$，允许的共同质心区域半径为：

$$
r_i^{\mathrm{safe}}
=
R_i^{\mathrm{verify}}
-R^{\mathrm{required}}
-\epsilon_{\mathrm{pose}}.
$$

只允许收缩，不允许通过误差膨胀候选区域。证书中心固定为一个已经通过成功gate同构复核的源proposal中心；其他车辆proposal只能增加证据，不能把中心移动到未经任何车辆验证的位置。

commit采用单epoch单票和4-of-4全签。消息丢失或不同站点投票只能导致不提交，不能分别提交两个站点。进入新epoch需要四车共同release。

## 严格标准

H0核心必须同时满足：

- 存在非零证书覆盖；
- 离线真实平整度复核无伪证书；
- 所有支持proposal满足定位误差收缩后的包含关系；
- 候选列表置换不改变证书；
- 场景SE(2)变换不改变site id，证书中心按同一变换变化；
- 完整消息最终只提交一个站点；
- 丢包和冲突投票不产生split-brain；
- 四车终端方形几何满足dmax、dispersion和碰撞间距门限。

候选覆盖率不是通过放松平整门限优化的指标；覆盖不足用于决定是否需要跨时段belief和探索。

## 结果表

| 分层 | 任一候选 | 双车支持证书 | 四车支持证书 | 最终可提交证书 |
| --- | ---: | ---: | ---: | ---: |
| near Open | 99.48% | 13.02% | 0 | 99.48% |
| near Mixed | 7.29% | 0 | 0 | 7.29% |
| near Bottleneck | 0 | 0 | 0 | 0 |
| far Open | 100% | 2.60% | 0 | 100% |
| far Mixed | 36.98% | 0 | 0 | 36.98% |
| far Bottleneck | 1.04% | 0 | 0 | 1.04% |
| 总体 | 40.80% | 2.60% | 0 | 40.80% |

在470个可提交证书中：

- 实际平整度复核为470/470；
- 定位误差保守包含为470/470；
- 候选置换、SE(2)、完整消息提交和对抗消息fail-closed均为470/470；
- SE(2)最大中心误差为0；
- 12 m通信图连通率为100%，但全连接率为82.64%，因此在线实现需要有界多跳转发，不能假定所有车辆直接相邻。

终端中心间距取 $0.42\ \mathrm m$ 时，方形几何为：

$$
R_{\mathrm{formation}}=0.297\ \mathrm m,
\qquad
d_{\max}=0.594\ \mathrm m,
\qquad
\mathrm{dispersion}=0.0882\ \mathrm{m}^2,
$$

均位于当前成功门限内。

## 失败分析

第一版H0曾产生11/478个伪证书。原因不是定位误差模型，而是 $1.25\ \mathrm m$ 大圆盘的离散采样不能严格蕴含平移后 $0.75\ \mathrm m$ 小圆盘的另一组离散成功采样。修正后，每个proposal源车额外执行与成功gate完全相同的复核，并禁止关联器移动证书中心，伪证书降为0/470；总体覆盖率只从41.49%降至40.80%。

当前主要失败是覆盖而不是证书正确性：瞬时4 m局部感知在Mixed和Bottleneck中很少发现带足够验证余量的平地，near Bottleneck甚至为0。静态H0通过不代表完整D-STC已经可执行。

## 产物路径

```text
outputs/runs/exp160_dstc_site_commitment/h0_certificate_audit/
├── run_manifest.json
└── metrics/h0_certificate_audit.json
```

核心代码与测试：

```text
source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/site_commitment.py
tests/test_exp160_site_commitment.py
```

## 结论

`h0_certificate_core_passed=true`，说明静态proposal、保守关联和全签commit核心可以继续使用；但`online_actor_integration_ready=false`。本实验不是strict策略结果，没有checkpoint，也不能生成正式动画。

## 下一步

下一阶段只增加一个能力：有限容量的跨时段候选belief与12 m图上的有界转发。需要在冻结轨迹回放中证明：

1. 车辆运动后Mixed/Bottleneck候选覆盖显著提高；
2. 陈旧、重复、乱序proposal不会覆盖更新版本；
3. 通信恢复后四车在有限轮次获得相同proposal-set digest；
4. commit之前不向Actor提供未经确认的站点势场。

上述动态门限通过前，不接入407维Actor，不启动新的Pure RL长训，也不放松平整度阈值。
