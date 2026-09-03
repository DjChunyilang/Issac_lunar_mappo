# exp162：Active-DSTC H0.5主动证据获取

更新时间：2026-08-20。

## 目的

在exp161确认“共识能传播证据、但不能产生证据”后，实现训练无关的：

```text
DISCOVER → VERIFY → EXCHANGE → COMMIT
```

每车维护每个source最多4项的版本化候选belief；陈旧、重复和乱序消息不能覆盖新版本。车辆使用固定协调令牌和初始相对位姿构造共享局部参考框架，分区搜索远端frontier。候选每2.4 s扫描一次，通过12 m邻接图最多洪泛3轮。

## 配置

```text
configs/experiment/exp162_active_dstc_h05.yaml
scripts/audit_exp162_active_dstc.py
```

候选中心使用与成功gate相同的 $0.75\ \mathrm m$复核，外层验证半径从exp160的 $1.25\ \mathrm m$改为 $0.90\ \mathrm m$。这是因为证书中心已经固定为源proposal中心，不再需要未使用的0.5 m中心移动余量；仍保留0.10 m位姿误差。

## 结果

### 原始100陨石坑Bottleneck，32环境/层

| 分层 | 证书 | collision | timeout |
| --- | ---: | ---: | ---: |
| near Open | 100% | 0 | 0 |
| near Mixed | 100% | 0 | 0 |
| near Bottleneck | 0 | 6.25% | 93.75% |
| far Open | 100% | 0 | 0 |
| far Mixed | 100% | 0 | 0 |
| far Bottleneck | 6.25% | 0 | 93.75% |

所有分层伪证书均为0。

## 失败分析

提高扫描频率、改用分区frontier并让车辆平均行驶24–27 m后，Bottleneck仍几乎没有候选。全地图审计发现：

- 原Bottleneck在1 m网格下通常只有约1个带0.5 m额外余量的候选；
- near/far Bottleneck的Oracle点大量位于 $max(|x|,|y|)\ge9\ \mathrm m$ 的地图边界；
- Mixed的Oracle仍位于中心附近，说明证书和探索机制并未普遍失败；
- 保留瓶颈墙、把陨石坑数从100降到30后，0.5 m网格内部可行中心增加到89个。

因此原Bottleneck同时叠加狭窄通道和100个陨石坑，几乎把内部平地清空，任务只能依赖边界平坦带。继续调整探索策略会鼓励“奔向边界”捷径，而不是研究地形相关共同选址。

## 结论

```text
active_dstc_h05_passed: false
reason: bottleneck benchmark interior feasibility failure
```

exp162没有进入1152场景正式评测，也没有训练或生成教师动作。后续使用独立exp163修复Bottleneck可行域，不覆盖本实验结果。

## 产物

```text
outputs/runs/exp162_active_dstc_h05/diagnostic_32env/metrics/active_dstc_h05.json
```
