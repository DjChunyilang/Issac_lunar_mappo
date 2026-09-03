# exp163：可行Bottleneck上的Active-DSTC正式H0.5

更新时间：2026-08-20。

## 目的

修复exp162暴露的Bottleneck内部可行域缺陷，并正式评测Active-DSTC的`DISCOVER/VERIFY/EXCHANGE/COMMIT`链路。

## 配置

```text
configs/experiment/exp163_feasible_bottleneck_active_dstc.yaml
scripts/audit_exp162_active_dstc.py
```

保留原Bottleneck的平滑墙、中央通道、坡度和所有成功门限，只将Bottleneck的陨石坑数从100调整为30，与Mixed的陨石坑密度一致。该修改不向车辆提供平地坐标，也不改变Open/Mixed场景。

探索避让距离从0.70 m提前到0.90 m。候选belief、全签commit、0.90 m验证半径和2.4 s扫描周期与exp162一致。

## 严格标准

每个六分层均要求：

- 96 s内证书覆盖至少90%；
- 伪证书为0；
- collision不高于2%；
- timeout严格低于10%。

本实验只验收到COMMIT，不等于完成最终集合。

## 正式结果

每层192个episode，共1152个冻结场景：

| 分层 | 证书 | collision | timeout | 平均证书步 |
| --- | ---: | ---: | ---: | ---: |
| near Open | 100% | 0 | 0 | 0.31 |
| near Mixed | 99.48% | 0 | 0.52% | 21.17 |
| near Bottleneck | 97.40% | 0.52% | 2.08% | 18.03 |
| far Open | 100% | 0 | 0 | 0.50 |
| far Mixed | 97.92% | 0 | 2.08% | 16.98 |
| far Bottleneck | 100% | 0 | 0 | 11.31 |

全部1152场景的伪证书率为0，所有分层通过预注册门限：

```text
active_dstc_h05_passed: true
```

## 成功原因

- 删除了与固定证书中心不再相关的过度验证余量；
- Bottleneck保留通道难度，但不再由100个陨石坑清空内部可行平地；
- 跨时段belief保留不同位置发现的候选；
- 版本化有界洪泛使通信恢复后获得相同候选集合；
- 0.90 m提前避让消除了32环境诊断中的单次碰撞。

## 剩余问题

当前每次洪泛发送完整候选缓存。每episode平均发送约1555–4879条proposal记录，而真正改变的记录约132–409条，存在明显冗余。进入完整闭环前应改成delta/event转发。

本实验在证书形成后让车辆hold，没有验证：

- 四车是否能够到达证书区域；
- 进入过程中是否安全；
- 最终dmax、dispersion、速度和hold gate；
- R4在随机地形和消息年龄下的稳定性。

## 产物

```text
outputs/runs/exp163_feasible_bottleneck_active_dstc/formal_1152/
├── run_manifest.json
└── metrics/active_dstc_h05.json
```

## 下一步

1. 将全缓存洪泛改为版本化delta消息并验证digest不变；
2. 把exp161的R4原语best response接到COMMIT之后，作为独立低层控制器；
3. 先做32环境六分层完整闭环；
4. 只有dmax、success、collision和timeout门限通过后再运行1152场景；
5. 本轮仍不训练高层utility、Actor或通信网络。
