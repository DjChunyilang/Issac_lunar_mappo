# S03密集感知、合法通信与实际停车组件

日期：2026-10-13。状态：**1142项冻结回归、八车MPPI／实际停车组件闭环通过；低速float32边界缺口已复现，完整S03、新尺度Gate B／144回合及T11尚未验收，正式训练暂停。**

本轮继续[原T11](../t11_handoff_plan.md)前置的[S00–S05计划](../implementation_plan.md)。100k、500k×3、2M×3预算保持原计划；没有训练环境交互、优化器更新或训练checkpoint。

## 目标行为与验收方法

将真实局部密集感知、每车私有记忆、合法距离消息、实际控制历史和逐车停车事实接起来，防止未来信息、sender换槽和队列过期造成错误认证。先执行纯契约与旧路径回归，再通过统一工厂运行真实MPPI及实际代理停车。组件fixture与完整出生／集合矩阵分别记录，不以已经接近队形的测试替代S05。

独立配置：[upstream_scale_s03.yaml](../../configs/experiment/upstream_scale_s03.yaml)。继承S02物理规范、K512／两轮及0.5 s离线注入延迟，显式新增`actual_control_history_v1/pose_scope_cache_v1`。旧S02保持`held_previous_control_v1`和无缓存路径；缺省字段不添加到历史元数据。

## 已实现接口

| 模块 | 当前实现与边界 |
| --- | --- |
| `upstream_sensing.py` | 151×81的独立只读`SensorFrame`，固定odometry变换、五通道、有效／直接掩码、观测时刻和地图身份。真实SceneBank的有限占位不视为已知；局部3×3高度去趋势RMS是代理指标 |
| 同模块`DenseTerrainMemory` | 0.1 m索引、65536容量、0.155 m支持；直接点不移到格中心，同格危险坡度／台阶及同位置高度冲突保留；公开恢复与环境隔离 |
| `upstream_communication.py` | FULL／稀疏60／120 m、0.2／1 s发送周期、1／0.6 s有效期，sender绑定目标／场地／轨迹。真值只用于距离路由，接收snapshot不补真值 |
| 同模块`MotionBoundLedger` | 只有已登记且实际执行持续强制的限速承诺可降低未来可达界；新高上限不能撤销旧未到期低上限。单纯HOLD消息不能将未来速度界设零 |
| `upstream_timing.py` | 信息时刻到实际生效时刻的控制、实际速度与odometry只读历史；逐项校验联合限制、midpoint和端点。真实排队运动及生效后保守停车总长度仍必须≤40 |
| `upstream_gathering.py` | leader占方格角点，广播场地中心与朝向；合法消息驱动编号槽位与优先级。`APPROACH → BRAKING → HOLD`依实际停稳及合法停车终点捕获，BRAKING不因误差噪声恢复MOVE |
| `upstream_task.py` | 实际逐车足迹、速度／边缘速度、最大间距／离散度、包络容纳、漂移与25步连续驻留；环境失效、碰撞、越界、成功、截止优先级及异步reset |
| `upstream_review_cache.py` | 只缓存单次轨迹复核内逐值相同位姿的静态足迹结果；动态邻车、工作区、控制／动力学及每个时间点继续检查，退出或异常均销毁缓存 |

邻车未来半径从最后合法位置的可达球生成。使用非承诺名义轨迹作为中心时，额外加入其偏离最后合法位置的距离，年龄、几何和位移界各计一次。消息当前有效不代表8 s控制承诺；证书必须覆盖最后一次未来查询，否则退回25/9 m/s硬上限。

实际排队历史使用本车反馈，不重新用旧地形估计过去增益，也不把实际速度二次缩放。冻结本地快照仍不能含晚于信息时刻的观测；未来预测和停车继续按该快照重新计算增益并做完整足迹检查。历史失败保留`actual_history:*`原因。

缓存对静止40步轨迹将地形查询次数由400降为41，完整360个位姿／时间检查数及报告逐值相同。查询次数不是端到端耗时或实时通过证据。

## 独立证据

当前冻结run：

```text
outputs/runs/upstream_scale_s03/legal_communication_timing_seed23_cpu_cuda_20261013_02/
```

新S03契约由执行前的`metrics/contracts.source.json`绑定；S02、P0、动作、通信及真实acados相关回归单独写入`metrics/related.xml`。源码字节、profile、解析配置、依赖与命令保存在run内。[component_summary.json](../../outputs/runs/upstream_scale_s03/legal_communication_timing_seed23_cpu_cuda_20261013_02/metrics/component_summary.json)保存实际结果。

| 检查 | 冻结结果 |
| --- | --- |
| 新S03与地形契约 | 125通过，0失败／0跳过 |
| 相关旧路径与S02回归 | 1017通过，0失败／0跳过；两批不重合，合计1142 |
| 真实acados | 旧23节点及新增7实际运行用例全部通过；`real_acados_nodes.json`逐项匹配 |
| 真实批量MPPI | B8、K512、两轮；八条MOVE输出完整复核通过，实际执行后可测位移 |
| 实际停车 | 32底层步内，两环境各25步连续驻留，实际结束原因均success |
| 公开恢复与消息 | 23个底层步实际状态／任务逐值一致，sender／预测恢复一致，异步reset隔离 |

这个MOVE只执行三个底层控制步，位移约0.000095–0.028 m，随后请求停车；不是0.3 m局部目标到达证明。初态已是7.2 m方格，使用每车实际持续强制的0.05 m/s测试证书，规范的前进硬上限仍为25/9 m/s。此范围不替代正式出生、完整集合或全速闭环。

早期实际感知／任务run为`outputs/runs/upstream_scale_s03/sensing_task_seed23_cpu_cuda_20261013_02/`：62项冻结回归通过，零失败／跳过；两环境八车实际移动后HOLD与25步停车驻留、逐步恢复和异步reset通过。其初态已经是7.2 m方格，控制移动为脚本fixture；没有通信、完整出生接近或学习链，不能称完整S03验收。

该早期run第一次接线错误地将延迟期间更新的记忆视图交给较早的信息时刻，正确触发“未来地形观测”拒绝。原`logs/parking_probe.py/.log`和失败JSON保留；修正版冻结信息时刻的视图，使用独立v2产物，不覆盖失败。

本轮开发还保留通信15项失败（NumPy布尔值不能通过严格Python布尔消息校验）、一个停车地形测试fixture缺少规范参数的失败及修复重跑。开发通过记录不代替run内冻结JUnit。

前一次冻结run `legal_communication_timing_seed23_cpu_cuda_20261013_01`有125＋1016项测试全部通过，但实际组件循环遇到异步结束错误：两个环境进入HOLD快照时驻留计数分别2／3，诊断在先成功的环境结束时使用`any()`退出，另一个尚差一步。`metrics/component_failure.json`、原日志及公开快照完整保留。修正后仅冻结已结束环境的时间／状态、继续其余环境，并增加每五步轨迹分片；改版重新冻结新run，不覆盖第一次失败。

统一池新增`export_diagnostics()`只读导出末轮原始采样，不通过环境读取具体适配器热状态；公开sample导出与热恢复分别处理。首MOVE采样按八实例全部导出，每调用只覆盖两轮中的末轮一轮，不能写成全部迭代采样覆盖。

## 尚未完成与下一步

当前尚无完整可训练新尺度环境。密集感知与任务组件已接通，但Actor的112点／115项附加观测、新尺度奖励与真实宏步学习转移、安全投影、完整高层队列生命周期及正式出生分布仍待接线。不能把组件task成功写成完整集合矩阵通过。

追加[low_speed_precision_gap.json](../../outputs/runs/upstream_scale_s03/legal_communication_timing_seed23_cpu_cuda_20261013_02/metrics/low_speed_precision_gap.json)复现`previous_control=[0.050000001,0.25]`：NumPy物理检查合法，float32 MPPI将上一控制转换到低速阈值后错误拒绝；float64参考实际完成，首刹车控制`[0.050000001,0.21]`。这不影响上述已测低速fixture通过，但必须先版本化修复，不能宣称完整S03。

继续核查一秒请求有效期与延迟排队／续发的关系、活动限速承诺释放和优先级等待，防止过期运动、错误零速度承诺及永久让行锁。完成S03后推进S04独立校准及S05统一版本完整Gate B和144回合，随后执行T11真实桥接与原定学习筛选。

## 文档与静态收口

Ruff F821、仓库卫生、`git diff --check`、新增文件空白、10份文档335个本地链接及数学分隔符全部通过，命令与SHA记录于run的`metrics/static_checks.json/test_commands.json`。生产源码与归档字节核对一致；用户`writing-block.md`和旧47库SHA未改变。`logs/development/index.json`保留开发失败、重跑和前一异步结束失败run的来源。

`metrics/strict_acceptance.json`只令组件契约、八车组件循环、恢复和异步reset通过；`low_speed_precision_gap_resolved/s03_full_acceptance_pass/s05_gate_b_pass/t11_pass/gate_c_pass`均为false，正式训练继续暂停。
