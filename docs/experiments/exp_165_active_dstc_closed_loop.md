# exp165 Active-DSTC与R4完整闭环

## Material Passport

- Origin Skill：`academic-research-suite`
- Origin Mode：experiment run
- Origin Date：2026-08-26
- Verification Status：VERIFIED（CUDA 32环境六分层pilot与完整回归测试）

## 目的

将已经分别验证的Active-DSTC共同平地证书与R4去中心化轨迹原语优化接入同一96 s任务回合，直接检查“发现—验证—交换—承诺—集合—稳定保持”的完整闭环。该路线没有可训练参数，因此本实验属于闭环执行验证，而不是神经网络训练。

## 配置

- 配置：`configs/experiment/exp165_active_dstc_closed_loop.yaml`；
- 入口：`scripts/run_exp165_active_dstc_closed_loop.py`；
- 场景：近距/远距与Open/Mixed/Bottleneck六层，每层32个episode；
- 时域：96 s、480个规划步；
- 上层：Active-DSTC本地候选、版本化delta/event转发、四车同digest同site承诺；
- 下层：R4同时枚举47个差速轨迹原语，原语承诺保持2个规划步；
- 执行信息：本车状态、共同站点证书和12 m内邻车上一承诺原语；
- Oracle、BC、Actor、安全投影、集中式动作修正均未使用。

## Pilot门限

每个分层要求：

- certificate不低于90%；
- success不低于26/32，即81.25%；
- collision为0；
- timeout低于10%；
- 平均dmax ratio不高于0.25。

通信还要求最终proposal-set digest和site id与完整缓存洪泛相同，六层累计传输记录数至少减少70%。

## 结果

| 分层 | certificate | success | collision | timeout | dmax ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| near_open | 100.00% | 43.75% | 6.25% | 50.00% | 0.210 |
| near_mixed | 100.00% | 34.38% | 0 | 65.63% | 0.230 |
| near_bottleneck | 100.00% | 37.50% | 0 | 62.50% | 0.222 |
| far_open | 100.00% | 25.00% | 0 | 75.00% | 0.124 |
| far_mixed | 93.75% | 31.25% | 0 | 68.75% | 0.205 |
| far_bottleneck | 93.75% | 21.88% | 0 | 78.13% | 0.228 |

版本化delta/event消息在六层中的最终站点语义均与完整洪泛一致，累计记录量减少84.52%，通过通信门限。完整闭环未通过success、collision和timeout门限，因此不启动每层192个episode的正式评测。

## 失败分析

共同站点发现不是主要失败来源：六层证书覆盖均不低于93.75%，且平均证书形成时间约为0至22.5步。主要失败发生在证书形成后的末段几何收敛：

- near_open最终dmax和dispersion gate分别只有53.13%和46.88%通过；
- far_open对应通过率分别为34.38%和25.00%；
- 平整度通过率为87.5%至100%，最近邻安全距离多数通过，因此timeout主要不是“找不到平地”；
- R4平均动作切换率为57.40%至73.15%，倒车、原地转向和前进之间发生持续切换；
- 一次固定的两轮原语协商对照使near_open collision升至21.88%，已停止并恢复单轮配置，未继续扫描轮数或代价权重。

由此可见，R4的单步best response虽然能解除冻结死锁，但不能直接保证连续闭环中的稳定联合原语承诺。

## 工程验证

- delta消息重复、陈旧和乱序合并测试通过；
- 重连时使用缓存digest触发完整事件重同步，修复有界缓存淘汰造成的ACK漂移；
- R4动作范围、terminal hold和非邻居状态不变性测试通过；
- CUDA 2环境smoke完成；
- 仓库完整pytest回归通过。

## 产物路径

```text
outputs/runs/exp165_active_dstc_closed_loop/pilot_32env_v3/
  config/experiment.yaml
  metrics/summary.json
  metrics/<cell>.json
  run.log
  run_manifest.json
```

## 结论

Active-DSTC已经能够在完整回合中产生正确、低带宽的共同站点证书，但当前R4不满足连续闭环集合要求。`exp165`是失败的闭环pilot，不是训练收敛结果，也不得作为推荐控制器。

## 下一步

停止扩大episode数量和重复运行当前R4。下一项实验应只针对联合原语承诺的跨时稳定性：使用冻结exp165轨迹比较单步best response、带滞回的承诺保持和显式两阶段冲突投票；在32环境near_open中先达到collision 0且动作切换率显著下降后，才恢复六分层闭环。若选择重新采用站点条件Actor，必须另建Active-DSTC证书到Actor观测及PPO控制掩码的训练语义，不能把本次确定性执行run称为训练。
