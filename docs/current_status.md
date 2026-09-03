# 当前状态

更新时间：2026-08-26。

## 当前主线

`exp165`已经完成Active-DSTC与R4的32环境六分层完整闭环pilot。Active-DSTC通信通过，但R4连续闭环失败：

```text
DISCOVER/VERIFY：已通过
→ EXCHANGE/COMMIT：已通过
→ delta/event通信：已通过
→ R4 GATHER闭环：未通过
→ 192场景/层正式验收：未启动
```

六层certificate为93.75%–100%，delta记录量累计减少84.52%，最终digest/site语义与完整洪泛一致。success仅21.88%–43.75%，timeout为50.00%–78.13%，near_open出现2/32 collision。主要失败是证书形成后的dmax/dispersion与动作切换：R4平均动作切换率57.40%–73.15%。因此不启动1152正式评测，也不把该run称为训练或收敛。

机器可读结果：

```text
outputs/runs/exp165_active_dstc_closed_loop/pilot_32env_v3/metrics/summary.json
```

保持执行期严格去中心化。当前不使用BC、奖励模型、集中式目标或Oracle候选。下一步只审计联合原语承诺的跨时稳定性；当前R4不作为推荐控制器。

`exp164`夜间H1长训已经完成并失败。它只隔离验证“共同站点已知时的低层Pure RL”，不代表完整去中心化选址。

exp164使用256环境、4800 iterations、307,200训练步和约78.6M环境交互。最终结果为success 86.98%、collision 12.50%、timeout 0.52%、dmax ratio 0.0961，strict失败。Stage B近距Open曾达到success 98.44%和collision 1.30%，但Stage C远距Mixed/Bottleneck始终保持约12%–27%碰撞，确认协调泛化失败。

运行状态：

```text
outputs/runs/exp164_overnight_h1_repaired/_suite/suite_status.json
```

停止续训、seed31/47和安全投影修补。继续delta通信与R4 GATHER；不得把307,200步`best.pt`或179,200步Stage B checkpoint标记为推荐策略。

## exp155状态

`exp155`已停止。N0在停止命令到达前恰好完成153,600训练时步，但其动作和评测接口已被否决，因此不进入架构排名或strict汇总。N1/N2没有启动。

N0产物保留在：

```text
outputs/runs/exp155_full_rl_ablation/n0_seed23_full_2400iter/
```

其run manifest标记为 `lifecycle_status: stopped_design_revision`，checkpoint状态仍为 `candidate`，且 `eligible_for_suite: false`。

该run的最终诊断为success 0、collision约0.0495、timeout约0.9505、dmax ratio约0.141，确定性hold比例约0.780。该结果只作为旧动作空间hold塌缩证据。

## exp156已完成工程

- 47维动作：hold、39个前进、3个倒车、2个原地转向和2个S形让行；
- 轨迹新增运动方向、计划航向变化和原语类型；
- 差速代理采用Jackal的 $r=0.098\ \mathrm m$、$b=0.376\ \mathrm m$ 和 $18\ \mathrm{rad/s}$ 轮速限制；
- 295维Actor观测和17维分级通信；
- v10 ego移除世界平移/旋转捷径，速度转换到车体坐标系；
- 950维统一多尺度Critic；
- 主线Oracle奖励权重为0；
- 队形整体旋转和每车初始航向独立随机；
- 熵衰减覆盖完整153,600训练时步；
- 记录归一化熵、有效动作数、hold概率和四类动作族概率；
- checkpoint明确区分295/47/Categorical与旧291/40/Gaussian接口；
- 生成六分层、每层192个episode的固定配对场景清单；
- 实现Clopper–Pearson、dmax bootstrap、IQM和配对bootstrap统计；
- 增加活动文档公式分隔符静态检查。

三种Actor参数量为：

| 结构 | 参数量 |
| --- | ---: |
| N0 | 83,343 |
| N1 | 106,591 |
| N2 | 31,752 |

均低于12万参数门限。

## 已完成验证

- exp156专项测试全部通过；
- 历史exp155多尺度和分级通信回归测试通过；
- Oracle、槽位和集中式诊断量不改变轨迹、控制与轮命令；
- 平坦场景整体SE(2)变换后，Actor观测和logits保持不变；
- CPU真实shared-joint MAPPO smoke通过；
- N0、N1、N2的CUDA 256环境真实MAPPO smoke均通过；
- 12个冻结场景的47原语覆盖审计通过，其中5个来自exp155后期全hold状态；
- 覆盖审计的无碰撞解除率为100%，倒车、原地转向和S形让行均参与过有效联合解；
- 1152场景 `scenario_manifest` 已生成。

Smoke只证明工程链路和梯度更新有效，不代表策略收敛。

## exp156训练结论与exp157诊断

`exp156` 当前没有训练进程。N0和N1均完成2400 iterations、39,321,600环境交互及同一1152场景清单的六分层配对评测：

| 结构 | success | collision | timeout | dmax ratio | strict分层 |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0 | 0 | 0.0729 | 0.9271 | 0.2242 | 0/6 |
| N1 | 0.0017 | 0.0130 | 0.9852 | 0.2367 | 0/6 |

两种结构均处于成功率地板，不能据此作有效的网络优劣排名。N1仅因保留多尺度空间结构且collision较低，被指定为后续接口研究的**临时CNN基线**；其checkpoint仍为 `candidate`，不是推荐策略、strict通过结果或最终架构。

N2完整训练已取消。原N2 run在第一次PPO更新时因非连续张量进入CUDA `grid_sample` 而退出；该问题已通过在采样前显式物化连续特征图和网格修复。修复后完成256环境、64训练时步的真实CUDA smoke，4次Actor/Critic更新均完成，参数变化非零且无NaN。smoke产物位于：

```text
outputs/runs/exp156_smoke/cuda_n2_contiguous_fix_256env_64step/
```

N2仍只保留为工程可运行候选，不再投入39,321,600交互，也不进入exp156架构排名。运行状态以以下文件为准：

```text
outputs/runs/exp156_differential_multiscale_ablation/_suite/suite_status.json
```

当前尚不存在exp156推荐checkpoint、六分层strict pass或正式动画。Oracle奖励消融、seed31/47和新的信用算法均暂停；下一项决策先验证共同选址与低层到达能否被分离，而不是继续扩大编码器比较。

“共同平地选择—终端稳定”耦合问题的候选路线为 `D-STC`（Decentralized Site-and-Trajectory Commitment）：有限本地地形 belief、分布式站点承诺、goal-conditioned Actor 与短时轨迹承诺。`D-STC` 是本项目对多篇研究原则的组合命名，不是已有论文中可直接复现的标准算法，也尚未决定采用。

`exp157` H0已完成1152个固定场景审计。12 m通信图连通率为100%，但“至少一辆车初始观测到共同站点且可传播证据”的总体比例仅40.97%，近距Bottleneck为0；这说明通信连接不等于站点信息已经可用。47维动作审计仍为12/12场景存在联合解，站点势场SE(2)误差为 $4.17\times10^{-7}$。H0尚未实现候选关联和commit，因此不能声称去中心化共同选址已就绪。

H1旧run在134,400/153,600训练步后中断，没有最终配对评测，不能作为MAPPO基线或恢复训练。其最后一次课程诊断success为0、collision约0.112、timeout约0.888；`ppo_timestep_134400.pt` 只作为exp158离线审计的行为策略。

## exp158工程状态

已经实现：

- 950维集中状态、四车联合动作和查询车辆条件的训练期反事实奖励模型；
- `[T,E,4,47]` 旧策略概率存储与逐车DAE advantage；
- 更新1—128的β=0拟合期、更新129—256的线性ramp和固定β=0.3阶段；
- 标准GAE与DAE训练语义、checkpoint字段和部署边界隔离；
- 六分层冻结状态快照、状态digest恢复、47动作真实反事实枚举；
- H1/strict配对配置、最终reward-model验证、配对bootstrap和串行门控launcher；
- exp158专项单元测试、CPU smoke和CUDA 256环境真实MAPPO更新。

正式形状的CUDA smoke使用256环境、rollout 64和一次完整更新，GAE/DAE Actor及Critic初始化hash完全一致；经共享状态latent优化后，DAE峰值CUDA显存约6.51 GB，吞吐为GAE的约90.4%，通过8 GB和60%门限。该结果只证明工程链路，不证明奖励模型可辨识或策略有效。

正式离线门限已完成但未通过，因此没有启动H1训练。主要失败为：冲突参与/非参与边际贡献比1.366低于2.0；共享advantage相关0.2004略高于上限0.20；总奖励动作模型最差MSE改善为-5.45%；policy-weighted期望误差达到2.006个真实奖励标准差。虽然六个地形seed的动作排序Spearman均达到0.30，模型仍无法提供DAE所需的校准期望值。

当前状态为 `offline_gate_failed`。按预注册规则，不扩大reward model、不增加RNN或辅助分量监督、不扫描β，也不启动H1/strict配对训练。机器可读结论见：

```text
outputs/runs/exp158_dae_validation/offline_credit_audit/metrics/offline_gate.json
```

详见[exp158实验记录](experiments/exp_158_dae_validation.md)。

## exp159工程状态

已经实现：

- 其他三车动作、地形、near、collision、failure和H1 Oracle进展构成的单步LOO基线；
- 逐车Oracle距离历史，同时保持原mean Oracle reward完全一致；
- `analytical_prd_loo` advantage路径，Critic继续拟合原团队return；
- 训练配置、checkpoint语义、日志、H1/strict双审计和串行launcher；
- 本车47动作基线不变性、单碰撞对非参与者修正和exp150差异测试；
- CPU真实更新与CUDA 256环境、rollout 64真实更新。

CUDA正式shape smoke中，GAE/PRD Actor和Critic初始化hash一致；PRD吞吐约为GAE的99.7%，峰值显存约6.49 GB，通过90%和8 GB门限。

A-H1正式冻结审计已完成但未通过，因此A-strict和完整训练均未启动。LOO基线满足团队奖励不变、source精确重构、本车47动作不变性和全数据梯度一致性；但两个验证seed的基线覆盖率只有3.45%/2.22%，梯度方差仅降低7.06%/0.42%，低于10%和15%门限。

当前状态为 `offline_h1_gate_failed`。这说明严格保持奖励不变且只移除一步、可证明无关的奖励项虽然无偏，但过于保守，不能产生足够的方差降低。按预注册规则，不运行A-strict、不降低门限、不加入多步trace或学习相关集合。机器可读结果：

```text
outputs/runs/exp159_analytical_prd/offline_h1_audit/metrics/offline_gate.json
```

详见[exp159实验记录](experiments/exp_159_analytical_prd.md)。

## exp160工程与H0状态

已实现训练无关的D-STC静态核心：

- 每车最多4个本地平地区域proposal；
- $1.25\ \mathrm m$验证圆盘、$0.75\ \mathrm m$成功平整半径和$0.10\ \mathrm m$位姿误差收缩；
- proposal源车使用与成功gate相同的离散采样二次复核；
- 不使用候选槽位编号的物理关联与稳定site id；
- 候选置换和场景SE(2)不变；
- 单epoch单票、4-of-4全签commit；
- 丢包、冲突投票和陈旧重放fail-closed；
- 新epoch必须取得四车release。

1152个固定场景的最终H0结果为：总体可提交证书覆盖40.80%，双车共同支持2.60%，四车共同支持0；470个证书的实际平整度、定位误差包含、候选置换、SE(2)、完整提交和对抗消息门限均为100%。Open覆盖99.48%/100%，但near/far Bottleneck只有0/1.04%。因此：

```text
h0_certificate_core_passed: true
online_actor_integration_ready: false
```

下一步只实现有限容量的跨时段候选belief和12 m连通图上的有界proposal转发，并使用冻结轨迹回放检查覆盖与版本收敛。动态门限通过前不修改Actor观测、不启动H1或strict长训。机器可读结果：

```text
outputs/runs/exp160_dstc_site_commitment/h0_certificate_audit/metrics/h0_certificate_audit.json
```

详见[exp160实验记录](experiments/exp_160_dstc_site_commitment.md)。

## exp161四路线比较结论

同一冻结套件已经尝试四条路线：

| 路线 | 结果 | 完整任务判定 |
| --- | --- | --- |
| R1站点证书 | 0伪证书，覆盖40.80% | 未通过 |
| R2 HPP式目标belief | 乐观覆盖12.07%，一致率0 | 未通过 |
| R3分布式地图共识 | 有证据时100%一致，平均1.17轮 | 覆盖不足，未通过 |
| R4去中心化原语优化 | 12/12安全解除死锁 | 只通过低层组件 |

R2已经使用全队实时位姿作为超出strict接口的乐观上界，仍无法让私有候选形成共同目标，因此不实现学习式队友预测。R3证明通信一致性不是当前瓶颈；R4证明47维动作具备解除末段死锁的表达能力，但它不能发现平地。

机器可读结论为：

```text
complete_route_passes: []
all_coupled_task_routes_failed: true
successful_components: [R1, R3, R4]
```

下一步只验证Active-DSTC的`DISCOVER → VERIFY`：有限topometric候选belief、任务特定frontier分工和通信重连。Mixed/Bottleneck在96 s内证书覆盖达到分层门限前，不训练高层utility或低层Actor。详见[exp161实验记录](experiments/exp_161_all_routes_feasibility.md)和[下一方向证据简报](references/active_dstc_next_direction.md)。

## exp162/163 Active-DSTC状态

exp162实现了有限候选belief、版本/TTL、候选幂等合并、最多3轮洪泛、2.4 s候选扫描和确定性分区frontier探索。原100坑Bottleneck的32环境诊断结果为near/far证书0/6.25%，而Open/Mixed为100%。全地图审计确认原Bottleneck的可行平地集中在地图边界；问题是基准内部可行域，而不是共识或候选记忆。

exp163保留Bottleneck墙和中央通道，只将陨石坑数从100调整为30。1152场景正式H0.5结果：

| 分层 | 证书 | collision | timeout |
| --- | ---: | ---: | ---: |
| near Open | 100% | 0 | 0 |
| near Mixed | 99.48% | 0 | 0.52% |
| near Bottleneck | 97.40% | 0.52% | 2.08% |
| far Open | 100% | 0 | 0 |
| far Mixed | 97.92% | 0 | 2.08% |
| far Bottleneck | 100% | 0 | 0 |

全部伪证书为0，H0.5所有预注册门限通过。但当前证书形成后车辆hold，尚未完成最终集合。通信还采用完整缓存洪泛，发送记录约为真实变化记录的10倍左右。

下一步固定为：版本化delta/event转发，然后把R4作为独立GATHER控制器接在commit之后。先做32环境六分层完整闭环；未通过时不运行1152正式评测。详见[exp162](experiments/exp_162_active_dstc_h05.md)与[exp163](experiments/exp_163_feasible_bottleneck_active_dstc.md)。

完整接口、预算和验收方法见[实施计划](implementation_plan.md)。
