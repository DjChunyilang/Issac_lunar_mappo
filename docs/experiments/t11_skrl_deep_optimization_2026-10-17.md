# 深度计算优化与skrl迁移：2026-10-17

默认训练入口已迁回 **skrl 2.1.0 MAPPO**。本轮同时实施不可变快照、GPU索引/下载复用、安全批处理、原生停车序列、记忆scratch与阶段融合、常驻环境进程、集中GPU评估服务和扩批入口。保留同一学习/安全协议时，三组运动回放平均7.827→5.360 s，吞吐提高1.460倍；128宏步243.884→166.639 s。更高吞吐需要启用独立的进程采样协议，不能冒充旧两环境checkpoint的等价续训。

正式长训练与四方案队列保持停止。父A1仍为2,048有效团队转移、8批/40轮PPO；本轮新增数据仅用于性能回放和有界更新测试。未修改K512×两轮、40步/8秒复核、任务、奖励、HOLD概率与安全阈值。Proxy测试不构成Isaac/PhysX验证。

## 实施与取舍

| 优化项 | 已实现行为 |
| --- | --- |
| 冻结请求与只读地形 | 内部共享不可变坐标/属性/KD树，时间数组与复核见证独立；公开snapshot仍返回独立副本 |
| GPU地形索引 | 一次全车打包供HOLD、MOVE和最终评分复用；按车缓存设备索引，只重建变化行；行选择保留源顺序、空地形与CPU歧义判定 |
| GPU传输和计时 | 各dtype合并全车候选/诊断下载，复用同一CPU数组；非deadline路径用CUDA events延后读时，保留真实依赖与deadline路径的同步 |
| 安全约束 | 小矩阵交点批量det/solve；9个扫掠位置合并KD查询；邻车初始量外提，保留可行顶点检查算术、OSQP和首失败原因 |
| 停车预测 | 原生CPU一次生成完整制动控制；补齐复核时域后只积分一次，保留完整轨迹与独立非线性验收 |
| 地形记忆 | 缓存旧cell、thread-local原生scratch和连续链索引；独占输出直接构造不可变view，避免重复复制 |
| 每车任务融合 | memory.observe→execution prediction→parking report→中心特征合并为一个worker任务；广播前冻结radio，按原序提交；复用同pose/view中心查询 |
| 常驻owner | spawn进程长期拥有环境组、地形/KD树/规划缓存；中央策略统一采样，宏步仅交换动作、观测与必要记录；checkpoint时才传完整环境 |
| 统一GPU服务 | `--gpu-service`合并同时就绪的评估请求，每车变化地形只上传一次；CPU复核留在owner。已实现、等价回放通过，但本机短批更慢，默认关闭 |
| GPU策略/PPO | `--policy-device cuda`可冷启动GPU策略/学习；完整小批PPO测得更慢，默认CPU。没有为提高占用率扩大网络 |

没有引入PPO期间预采下一批的异步更新，也没有降低采样量/精度/复核覆盖；这些会改变当前on-policy与安全契约。按车索引缓存以整行几何变化失效，未假称已经实现任意地形点级增量索引。

## skrl接入与checkpoint

`upstream_skrl.py`中的`TeamModel`继承skrl Model，`RecurrentTeamMAPPO`继承skrl MAPPO，实际使用框架创建的Adam、RandomMemory、act和agent checkpoint接口。四车共享actor、集中critic与联合优化器，因此在skrl中用一个`team`角色保留四车轴。项目同步采样调度、预算与准入门控继续由原入口管理。

默认MAPPO的固定步长GAE和打乱的前馈mini-batch不符合现有语义。扩展update使用共享的`recurrent_mappo_update`契约，保留变时长GAE、完整GRU序列、learning_valid/terminal masks、HOLD原子、原始latent/log-prob与熵采样随机流；没有调用旧trainer对象来替代skrl agent。

新trainer身份为`upstream_skrl_recurrent_mappo_v1`。`--trainer-backend legacy`仅显式保留旧实现；`--import-checkpoint`在新run执行版本/配置/设备匹配的显式导入，普通`--resume-state`仍要求同会话身份。跨设备热恢复和旧checkpoint转owner协议不静默执行。

已生成`checkpoints/skrl_migrated_update_000008.pt`：仅迁移框架身份，保留2,048转移/40轮PPO及父模型、优化器、环境、GRU和RNG；SHA见同名JSON。父文件SHA为`a57e8bffe72dcf1234e0a1b2e05a93c3c0445a02d01fee75aa9c753a1888d7be`，原文件未覆盖；5,325个非框架状态叶节点与父checkpoint精确相同。此文件是导入产物，未从它启动100k续训。

## 同状态运动配对

i7-10750H（6核/12线程）、RTX2060 6GiB、约15.4GiB RAM；torch2.10+cu126，CPU Torch线程1，原环境6worker。从同一update8恢复，预热后再次恢复，4宏步/8有效团队转移，三次交错基线/当前执行。基线是本轮开始前已优化版本，冻结在`config/before`；当前使用真实skrl act采样。窗口排除初始化、恢复、落盘；CPU/GPU/PSS以200ms外部采样，与窗口交集对齐。

| 指标 | 本轮前 | 本轮后 |
| --- | ---: | ---: |
| 三次平均耗时 | 7.827 s | 5.360 s |
| 吞吐 | 1.022 转移/s | 1.492 转移/s |
| CPU平均占用（100%=一逻辑核） | 189.75% | 198.74% |
| CPU时间/转移 | 1.839 s | 1.326 s |
| GPU平均占用 | 4.66% | 6.45% |
| PSS峰值三次平均 | 1474 MiB | 1595 MiB |
| GPU平均功率 | 27.31 W | 26.69 W |

GPU索引缓存和skrl导入增加约121MiB PSS，换取更少复制/建树与计算。利用率不是优化目标；本轮主要收益是单位转移CPU工作减少。最终审阅复现并修复原生刹车遗漏的非法上一控制检查，67项相关回归通过；有效输入执行值保持不变。修复后再次监控三次为5.361/5.420/5.466s，均值5.416s（相对冻结基线均值约1.445倍）；CPU平均198.7%、GPU平均6.2%，10,048行为字段仍精确一致。原始缺陷复现见`logs/brake_input_defect.log`，最终源码测量见`metrics/final_confirm*.json`。

长窗口从update5回放128宏步，256/256有效转移，最大位移6.162m；166.639s，包含任务截止、异步重置。与历史实际第6批逐字段比较141,051项，除既定墙钟字段外精确一致。前轮同窗口243.884s，原历史训练同批2,135.205s；后者仅作历史量级参考。

## 常驻进程与扩批

先在同一个owner协议/同状态上比较串行与并行调度：2owner×2环境，16转移11.594→6.671s；22,305个字段精确一致（排除墙钟）。这是实际持久进程采样器，中央策略一次采样整个环境批次，不是各进程重复采同一策略动作的旧容量探针。

以下为新协议的有界容量窗口，owner seed为`seed + owner_id * envs_per_owner`。环境数、宏步长度和owner分组变化均写入身份；不同配置的轨迹不相同，表中倍率不能代替同状态配对或学习收敛比较。短批越大，截断位置越不同，不能将其外推为128步GRU长训练的已测吞吐。

| owner×每进程worker | 总环境 | 宏步 | 有效转移 | 秒 | 转移/s | CPU平均 | GPU平均/峰值 | PSS峰值MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2×2 | 2 | 16 | 32 | 17.530 | 1.825 | 241.2% | 14.1% / 22% | 2671 |
| 2×2 | 4 | 8 | 32 | 13.798 | 2.319 | 265.7% | 9.8% / 25% | 2815 |
| 2×2 | 8 | 4 | 32 | 12.389 | 2.583 | 272.0% | 12.0% / 35% | 3074 |
| 2×2 | 16 | 2 | 32 | 10.009 | 3.197 | 266.8% | 9.4% / 46% | 3342 |
| 2×2 | 16 | 8 | 128 | 47.948 | 2.670 | 288.4% | 7.4% / 36% | 3915 |
| 4×2 | 16 | 8 | 128 | 34.759 | 3.683 | 560.3% | 15.4% / 66% | 5702 |
| 4×2 | 32 | 4 | 128 | 31.024 | 4.126 | 557.0% | 13.1% / 83% | 6591 |
| 4×2 | 64 | 2 | 128 | 28.541 | 4.485 | 550.3% | 14.1% / 94% | 7590 |
| 4×2 | 48 | 2 | 96 | 18.685 | 5.138 | 533.4% | 14.4% / 81% | 6702 |
| 6×1 | 48 | 2 | 96 | 19.066 | 5.035 | 547.0% | 17.6% / 94% | 8734 |
| 6×2 | 48 | 2 | 96 | 17.576 | 5.462 | 785.7% | 19.3% / 95% | 8669 |

最终固定48环境、2宏步/96转移、各三次重复：

| 进程数 | 平均秒 | 转移/s | CPU平均 | GPU平均/峰值 | PSS峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 17.208 | 5.579 | 773.5% | 19.0% / 95% | 8.74 GiB |
| 8 | 17.681 | 5.429 | 909.8% | 20.1% / 93% | 10.39 GiB |

本机建议6owner、每owner2worker；8owner吞吐反降约2.7%，更多占用没有换来更多计算效率。这个结论针对所测短采样协议；CPU温度峰值96°C，未测得频率证据，不据此声称已经热降频。保留较少进程以节省约1.65GiB PSS。扩批改变训练更新频率/GRU边界，不能直接拼接原结果。证据见`metrics/final_*_hardware.json`。原signal四方案冻结的2环境×128宏步路径保持原定义；owner入口用于独立协议，不能直接给原100k结果拼接扩批数据。

## 完整更新与入口验证

同一真实128宏步/256转移，5轮PPO，来自冻结旧producer的完整更新与当前skrl CPU输出比较：240个参数/优化器/RNG/指标叶节点全部精确一致。旧实现4.498s，skrl CPU 4.401s，skrl CUDA 7.470s。GPU路径损失/梯度有限，但采用独立CUDA熵随机流；不称跨设备逐位等价。小网络及逐步GRU/标量依赖使此批GPU学习更慢。

GPU服务单独验证：2owner、4环境、16转移，直接GPU6.671s，服务7.402s。归一化显式服务身份字段后22,307项行为字段精确一致；默认关闭该实验选项，保留可复测入口。最终六进程48环境、每次96转移、两次平均20.935s（4.586转移/s），比直接GPU17.208s更慢；CPU平均614.6%、GPU平均14.6%。服务实际合批最大71个车辆请求，各owner最终窗口6次调用中3–6次为跨进程合批，并非未触发服务。修正合批等待后再次与直接执行及串行调度逐字段比较22,307项精确一致。

- 主回归：173 passed，0 failed/skipped，89.39s；6条既有rasterio弃用提示。
- 最后补测：43 passed，0 failed/skipped，15.94s；含进程中央采样/恢复/实际更新、空地形行选择、skrl迁移和优化器重置。与主回归有重复，不相加声称独立用例数。
- 默认CLI：`skrl_entry_20261017_01`完成8转移、1次5轮诊断更新、独立短评估和诊断状态保存。
- owner CLI：`skrl_owners_entry_20261017_01`完成16转移、1次5轮诊断更新、评估和保存。两者均`formal_training_interactions=0`。
- A0/A1/A2/A3四动作头经skrl采样载荷/latent/log-prob逐位一致；最后5项框架测试通过，6.50s。
- skrl原生`agent.save`与项目完整环境checkpoint恢复均通过；正式门控与最多两批诊断限制保留。

初次skrl测试发现skrl 2.1.0对空kwargs字典的配置展开拒绝，已改成显式team键。随后对照测试额外采样消耗熵估计RNG，导致差异；修正测试恢复RNG后精确通过。原失败日志保留，不算作通过。

## 证据与复现

主run：`outputs/runs/upstream_scale_t11/skrl_deep_optimization_20261017_01/`。`run_manifest.json`保存父SHA、源快照、命令和验证索引；`metrics/summary.json`汇总；所有原始监控在`metrics/*_raw.jsonl`。`config/before`与`config/final`保留生产者，生成目录不提交Git。

命令均在仓库根目录，用`.venv_terrain_cuda/bin/python`；复跑须使用新run/label，脚本采用独占写以保护原结果：

```bash
# 三组同状态配对、长窗口、扩批硬件面板
.venv_terrain_cuda/bin/python <新run>/logs/monitor.py --phase paired
.venv_terrain_cuda/bin/python <新run>/logs/monitor.py --phase long
.venv_terrain_cuda/bin/python <新run>/logs/monitor.py --phase owners
.venv_terrain_cuda/bin/python <新run>/logs/monitor.py --phase capacity
.venv_terrain_cuda/bin/python <新run>/logs/monitor.py --phase owner_tune
.venv_terrain_cuda/bin/python <新run>/logs/monitor.py --phase final_capacity
# 有界skrl真实入口；不启动正式长训练
.venv_terrain_cuda/bin/python scripts/run_upstream_t11.py \
  --config configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml \
  --run-id <新run-id> --representation A1 --mode diagnostic \
  --rollout-steps 4 --diagnostic-updates 1
# 独立进程采样协议在上述命令追加
# --env-owners 2 --num-envs 4 --owner-cpu-workers 2
```

逐字段脚本`logs/compare.py`、完整更新脚本`logs/update_bench.py`与实际测试命令均保存在run中。CPU/GPU实际负载取监控窗口，显存为NVML全设备口径；RSS包含共享页，扩容决策优先看PSS/USS。未宣称达到所有场景的全局最优。
