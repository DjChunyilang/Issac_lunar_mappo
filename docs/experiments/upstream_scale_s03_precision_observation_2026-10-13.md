# S03低速精度修复与真实观测接线

日期：2026-10-13。状态：**低速精度v2、真实观测及八车组件验证通过；1188项联合测试及4项驱动修复通过，完整S03／S05／T11尚未验收。**

本轮继续[原T11](../t11_handoff_plan.md)前置的[S00–S05计划](../implementation_plan.md)。上一轮[组件证据](upstream_scale_s03_components_2026-10-13.md)保留1142项通过、八车实际停车及float32低速误拒原始记录；本轮不覆盖旧run，不启动训练或缩减原预算。

## 目标行为与验收方法

修复合法实际控制在采样精度转换后被误判为非法的边界，保留高斯原始扰动与概率语义；随后将密集私有记忆接入既有N1网络形状。验收覆盖NumPy物理参考、CPU／真实CUDA同噪声、恢复与异步reset、sender／时效、SE2变换及信息不变性，再通过真实八车K512组件循环检查观测与执行。

独立配置为[upstream_scale_s03_precision_v2.yaml](../../configs/experiment/upstream_scale_s03_precision_v2.yaml)和[upstream_scale_s03_observation_v1.yaml](../../configs/experiment/upstream_scale_s03_observation_v1.yaml)。后者启用`dense_actor_clock_upstream_v1`；仍为CPU代理、离线0.5 s延迟、K512两轮、训练暂停。旧S03与S02不自动选择新实现。

## 精度边界

原见证`previous_control=[0.050000001,0.25]`在NumPy检查中合法；float32把线速压到低速阈值的同一表示，错误触发0.2 rad/s角速上限。独立`mppi_upstream_v2`使用`upstream_mppi_kernel_actuator_v2`和状态版本2：原始采样／均值默认float32，敏感代价与重要性累计保持float64，上一实际控制、联合限幅、中点状态及实际速度用float64。

没有通过裁剪实际反馈或放宽低速角速限制修复。旧`upstream_mppi.py`的v1内核保持原样，新实现位于`upstream_mppi_precision.py`；跨适配器、内核、精度和状态版本恢复拒绝。

原始重要性修正仍使用原始扰动，投影后的控制不能替代epsilon。请求局部上限另做硬预筛和最终复核，保留真实制动过渡；高角速多边形所需的阈值epsilon不能用来越过0.05 m/s局部上限。制动上限曲线按B维张量同时计算，只沿时间循环。

## 真实观测与版本

`upstream_observation.py`仅接受本车实际状态、时钟、公开反馈、本地`FootprintTerrain`和接收车绑定snapshot，不接受全局地形、core、Oracle、真实团队计数或未通信车辆状态。

| 部分 | 固定形状／含义 |
| --- | --- |
| Actor | 15本车＋3×17邻车＋112×2地形＋5合法聚合＋115附加＋1剩余时间，共411 |
| 112点 | 7×9车体近场、3×7中场、4×7前后远场；从私有密集记忆独立查询，最大覆盖前后15 m、左右8 m |
| 地形 | 相对观测高度除以车高；风险为通行代理、坡度／10°、RMS／0.05 m归一化平方项的保守最大值，截到[0,1]；未知高度0、风险1 |
| 附加输入 | 保留10意图／反馈、25×3目标周边查询、3×10身份绑定消息；缺失／过期／稀疏目标与轨迹不补真值 |
| 归一化 | 速度用25/9、角速0.35、局部位移4 m、通信位置60 m、消息年龄1 s及任务时限；不残留1 m/s全局尺度 |
| Critic | 保留991槽主干尺寸，实际四车／团队状态、局部地形及40附加项集中读取；9个Oracle槽明确保留为零，任务驻留只进入Critic |

本车世界x/y、世界朝向、二维代理无定义的z槽使用原SE2结构的规范值，不把DEM绝对高程冒充车体位置。Critic的世界位置按200 m窗口归一化；高度／Oracle保留零槽的含义版本化，不加载旧checkpoint。

新版`sender_message_upstream_v2`允许FULL消息携带发送方合法本地五通道，坡度向量随固定odometry变换旋转。稀疏消息撤销该载荷；旧消息v1拒绝新增字段。缓存v2记录`distance_1e-8_sender_v2`：10 nm距离分桶只稳定等距槽位，硬通信距离、几何和支持半径不量化、不放宽。

接收snapshot附环境、receiver和时刻，Actor重新核对身份、episode、sender、掩码及年龄。初次SE2开发检查曾因等距排序的浮点误差产生12个输入差异，原失败保留；修正仅在消息v2启用，旧排序保持原样。

## 冻结证据与范围

当前run：

```text
outputs/runs/upstream_scale_s03/precision_observation_seed23_cpu_cuda_20261013_02/
```

执行前冻结源码字节、配置与测试来源。`precision_observation_seed23_cpu_cuda_20261013_01`实际执行169项新契约＋1019项相关回归全部通过，零失败／跳过；八车诊断在冷恢复时漏传消息v2，正确被版本校验拒绝，原日志、观测帧与公开快照保留。

修正统一首次／恢复构造后，`_02`实际执行4项驱动回归并重跑八车诊断。此前两批1188项JUnit仍引用原run、原路径和SHA，没有复制并声称在新run执行。复用仅允许`diagnose_upstream_s03.py`这一个驱动源文件变化：其他生产文件／归档字节、测试SHA与完整身份、配置、依赖和settings必须相同，新增文件或任何算法变化均拒绝。旧23及新增7真实acados仍绑定实际执行的原JUnit。

最终[component_summary.json](../../outputs/runs/upstream_scale_s03/precision_observation_seed23_cpu_cuda_20261013_02/metrics/component_summary.json)确认：B8、K512、两轮，8条MOVE输出通过完整复核；32步内实际停车，两个环境各25步驻留且结束原因均success；23步公开恢复与任务逐值一致，通信恢复及异步reset通过。9个观测帧包含Actor `[2,4,411]`和Critic `[2,991]`，共享N1／GRU／Critic前向全部有限。首次样本保存末轮8×512个原始样本，2轮中只导出1轮。

单独精度契约覆盖22项：正／倒车阈值、最大速度／轮速角点与NumPy参考、原始噪声、局部cap、状态、换序／稀疏reset、真实CUDA同噪声。观测22项覆盖实际SensorFrame记忆、N1形状、归一化、FULL／稀疏／过期、消息梯度变换、SE2、信息不变性、未知及硬风险。均已包含在169项JUnit内，不重复相加。

组件脚本实际采样B8、K512、两轮，记录末轮全部样本、名义与实际控制及停车。启用观测时每五步／结束事件记录真实Actor／Critic输入，独立冷构造共享N1／GRU／Critic做无学习前向检查；每次使用零初始GRU，只证明形状与数据链可消费，不是循环rollout、PPO更新或完整GRU生命周期验收。

初态仍为7.2 m已成形方格，执行持续强制0.05 m/s测试证书后停车；不是正式出生接近、局部0.3 m目标到达或全速闭环证明。安全QP、奖励／宏步、完整高层排队与T11动作执行桥接仍未连接。

## 下一门槛

在精度与观测证据收口后，继续实现完整环境的排队／有效期、实际执行反馈、安全、奖励、宏步及GRU异步重置。完成S03，再推进独立S04校准、同版本S05完整Gate B与144回合，最终按T11原100k→500k×3→2M×3筛选；本轮不将组件或网络前向通过写成Gate C完成。

## 静态与证据收口

Ruff F821、仓库卫生、`git diff --check`、新增文件空白、9份文档334个本地链接和数学分隔符全部通过。新run当前源码与归档字节一致；`metrics/reuse_audit.json`核对只有诊断驱动变化，旧v1 MPPI源文件、用户`writing-block.md`及旧47库SHA不变。开发失败、重跑和前一消息恢复失败均保留于`logs/development/index.json`及原run。

`metrics/strict_acceptance.json`只认证新版精度／观测、真实组件、恢复和前向接口。完整出生、循环rollout、奖励／安全环境、S03完整验收、S05 Gate B、T11及Gate C全部保持false；这里的组件success不代表T11完成。
