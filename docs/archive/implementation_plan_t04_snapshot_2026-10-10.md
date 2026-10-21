# 历史快照：T04 实施计划

本页保留归档时完整内容。当前事实见[当前状态](../current_status.md)。

# 第三阶段推进计划：T04 低维意图与 HOLD 混合动作

日期：2026-10-10。状态：**T04与Gate A契约验收通过**。依据 `writing-block.md` 的 T04 与 Gate A 要求，衔接已完成的[T03验收](../experiments/t03_planner_protocol_validation_2026-10-10.md)。T03原计划完整保存在[历史快照](../archive/implementation_plan_t03_snapshot_2026-10-10.md)。

已完成453项通过、1项非必要CUDA一致性跳过，12项真实acados均实际通过。详细证据见[T04验收](../experiments/t04_polar_hold_validation_2026-10-10.md)。下一任务为T05，正式训练继续暂停。

## 目标、已确认选择与边界

将新版策略输出从四维请求迁移为 `[rho, beta] + HOLD`，正确贯通采样、冻结意图、rollout、GRU序列重放和PPO概率计算。完成后联合T02验证Gate A；通过Gate A仍不恢复正式训练。

用户已确认两项选择：**条件混合分布，HOLD样本的动作似然只计离散分支概率**；**仅实施T04，运动意图用测试适配器验收，真实NMPC验证HOLD，低维运动的NMPC适配留到T05**。原exp167、P0和T03配置继续保留各自语义；MPPI、CBF/QP、完整基线、地图／通信范围适配不进入本阶段。

当前T03桥接要求运动意图带航向和速度约束，否则返回unsupported_intent。T04不得为了通过运动测试而在环境中补造这两个约束：运动意图使用测试专用适配器验收；真实NMPC验证HOLD及既有回归。unsupported_intent与限幅刹车可以诊断，但不能计为低维运动闭环成功。

前置事实：T03为409项通过、1项非必要CUDA一致性跳过，11项真实acados通过。这些是前置基线，不是T04验收结果。

## 一、动作与概率契约

### 1. 动作定义和坐标转换

新增独立动作版本 `polar_hold_v2`。每车动作由rho、beta两个连续标量及hold布尔值组成；连续张量为[env,agent,2]，hold及log_prob为[env,agent]，不把布尔分支伪装成第三个Gaussian维度。

- 诊断默认 `rho_max=1.6 m`、`beta_max=pi rad`：沿用当前局部请求的单轴距离尺度，允许全方向意图；这不是运动性能最优参数。两者在独立配置中显式声明，不继承旧极坐标路线的范围。
- MOVE分支采样二维独立Normal潜变量z，先令u=tanh(z)，再映射 `rho=rho_max*(u[0]+1)/2`、`beta=beta_max*u[1]`。保留log_std的现有限幅[-5,1]和初值-0.5。
- 目标为请求时本车位置加上旋转到固定odometry frame的 `[rho*cos(beta), rho*sin(beta)]`；生成后冻结，不能随车体姿态重新解码。
- MOVE生成现有LocalIntent，terminal_yaw和speed_cap均为None。HOLD调用LocalIntent.for_hold，冻结请求时位置／航向，速度上限0；实际减速沿用规划与执行限幅。
- 请求有效期仍为1 s，策略每5个0.2 s底层步决策；轨迹时域仍为15步。rho=0不等价于HOLD，所有边界动作均按显式hold分支解释。
- HOLD动作中的rho、beta和latent填0作为非活动占位，motion_active=false；占位不能被解码成MOVE或用于连续分支的动作概率。

### 2. 条件混合分布

共享原N1编码器及128维GRU；新版连续均值头输出2维，另设Bernoulli HOLD logit头，均采用小增益初始化、零bias。初始HOLD概率在0.5附近（小增益权重、零bias）；确定性评测仅在p>0.5时选择HOLD，等于0.5时选择MOVE，MOVE使用tanh(mean)。

令h为HOLD指示，p为其概率，q为MOVE分支在物理极坐标(rho,beta)上的密度：

$$
\log\pi(a\mid s)=h\log p+(1-h)\left[\log(1-p)+\log q(\rho,\beta\mid s,\mathrm{MOVE})\right].
$$

使用logits稳定计算离散概率；连续密度从保存的z重算，包含tanh Jacobian，以及仿射变换的 `log(rho_max/2)+log(beta_max)`。概率定义在极坐标动作空间，不能改用目标位置或规划器控制量计算概率。

- 先采样HOLD分支，只为MOVE行采样连续latent；记录hold、motion_active、原始latent、物理动作、采样时log_prob和hidden。PPO使用保存的原始采样，不重采样、不反推atanh，不用刹车或安全层输出替换动作。
- 熵使用 `H(Bernoulli(p)) + (1-p)*H(q)`。Bernoulli熵解析计算，条件连续熵使用当前分布的独立重参数化样本估计；不能用Normal未变换的熵，也不能将两个分支熵无权相加。
- 熵正则是整个当前分布的期望，即使该次动作是HOLD，仍可通过期望熵更新连续头；关闭熵后，HOLD样本的动作似然不应向连续均值／方差头传梯度。共享编码器／GRU仍可由离散分支更新。
- 保留gamma=0.99、lambda=0.95、现有PPO clip及熵权重；记录连续密度的物理坐标约定，不将熵数值直接与旧四维分布作性能比较。

## 二、环境、观测与学习接入

### 1. 统一意图入口

在现有local_goal包内增加动作编码／采样类型，复用T03 LocalIntent和Planner协议。环境先根据action_version解码，再共用规划、执行、奖励和结束逻辑。新版以intents为唯一真实请求来源，不构造带伪航向／速度的旧LocalGoalRequest供规划器消费。

旧hold_mask保留为旧路线的明确诊断入口；新版拒绝外部hold_mask覆盖，保证执行分支和策略采样一致。批动作在状态修改前完成形状、范围、有限值及版本校验。轨迹反馈、HOLD请求、实际停车、成功驻留分别记录。

### 2. 观测与通信迁移

当前观测和通信直接读取旧goal.target／speed_cap。新版增加观测语义版本 `local_intent_clock_v3`，不因维度相同而视为旧版本兼容。

- 保留基础传感、邻车sender槽位、局部地形和末尾剩余任务时间。基础ego中旧的“请求速度／请求航向变化”两个槽位统一为0并标为保留槽；旧配置语义不变。
- 新版附加观测的前10项固定为：目标相对xy、hold、intent_valid、请求剩余有效期比例、blocked_seconds/10、原规划状态编码、实际前向速度、实际角速度、局部地形known比例。
- 请求剩余有效期比例截断至[0,1]；reset后意图无效且目标／hold／有效期置零。目标xy仍从冻结意图换算到当前车体系，HOLD的冻结目标不能随车重置。
- 后75项地形和30项邻车消息保持原布局；Critic继续追加每车上述前10项。因此预期Actor/Critic仍为合成图411/991、真实三通道图523/1439，构建和校验仍从实际张量推导。
- core中仅同步真实意图的局部／世界目标；旧请求速度和航向字段为保留零值，不作为MOVE约束。基础邻车消息中对应的旧请求字段也保持保留零值，不推断为停车。
- 通信目标从发送方发布的LocalIntent获取。stop_intent必须同时满足有效HOLD、本车实际线速≤0.05 m/s和角速≤0.1 rad/s；由发送方发布，在接收方按已有权限读取，不能读取未通信真值来补足条件。
- 不扩大消息权限或增加网络sender数值特征，沿用缺失／稀疏／过期规则；消息和轨迹结构版本不变，新观测版本承担本阶段语义变化。

### 3. Rollout与PPO

按动作版本选择策略头及概率评估接口；旧四维策略的sample返回与数值行为保留。新版策略采样结果使用明确的结构化载荷，runner和环境共同消费，不用tuple长度或最后一维猜版本。

rollout新增hold和motion_active，latent改为新版二维形状，保存原始动作及旧log_prob。PPO序列重放从已保存hidden开始，按同一动作版本重算分支概率；保留16步序列、异步done时GRU重置、无效地形屏蔽和变时长GAE。

屏蔽无效样本必须发生在网络／分布运算前，包含新增布尔分支及latent；不能用0乘NaN。全无效批次不更新优化器。HOLD是有效策略动作，不能套用learning_valid=false排除其离散损失或价值学习。

## 三、配置、版本与诊断入口

新增 `configs/experiment/t04_polar_hold.yaml`，继承T03诊断配置，显式选择finite_task_v2、local_intent_clock_v3、polar_hold_v2、planner_protocol_v1和nmpc_bridge_v1，保持training_paused=true。动作范围进入完整配置与checkpoint校验。

- contract_metadata从显式动作版本生成合同；省略时仍为旧四维版本。新动作只允许与新版观测、统一规划协议组合，不进行默认升级。
- 统一runner建模、checkpoint、环境快照、run manifest和评测报告的合同来源。修复当前评测报告写死local_goal_request_v1、未包含完整规划协议元数据的路径，补充针对性回归。
- 新版严格拒绝旧动作／观测checkpoint及不匹配的动作范围；即使Actor/Critic输入维度相同也必须拒绝。单独版本化rollout载荷；不自动迁移优化器或热启动训练状态。
- 新增无正式训练的T04审计入口，生成解析概率、坐标转换、HOLD执行及合成PPO检查报告；使用测试专用运动适配器时必须标记test_only。
- T04配置暂不进入旧六单元完整任务评测；入口提前报告低维运动适配依赖T05，不能把unsupported_intent刹车回合伪装成已接通的运动评测。旧配置评测照常保留。
- 不解除正式训练门控。允许固定合成样本上的有限梯度／优化器契约测试，不启动环境交互训练，不生成可推荐的训练checkpoint。

## 四、验收顺序与通过条件

| 顺序 | 验证 | 通过标准 |
| --- | --- | --- |
| 1 | 动作解码 | 距离／角度边界、旋转平移不变性、冻结目标、rho=0与HOLD区分、非法输入在执行前拒绝 |
| 2 | 概率解析 | HOLD概率质量与MOVE积分之和为1；tanh及仿射Jacobian正确，极端logit／latent保持有限；相同参数重算旧动作时PPO ratio=1 |
| 3 | 分支梯度与熵 | HOLD不计占位latent概率；关闭熵后连续头的HOLD似然梯度为0；MOVE正常反传；熵匹配独立数值估计及分支权重 |
| 4 | 循环学习 | 全HOLD、全MOVE、混合、无效／NaN行和异步结束均通过；合成PPO更新损失／梯度有限，GRU不跨回合，普通rollout切分不清零 |
| 5 | 信息与版本 | 新观测字段语义、时钟、sender换槽及权限正确；Oracle／未通信真值不改变本地动作或预测；新旧配置维度与版本拒绝测试通过 |
| 6 | 实际执行 | 真实acados验证策略产生的HOLD从运动到限幅减速、实际停车与驻留中断；原11项必要真实求解器回归全部实际执行，不用skip代替通过 |
| 7 | 联合Gate A | 重跑T02时间／GAE／奖励解析、邻车身份及T03协议回归，并与T04采样概率结果共同生成机器可读Gate A结论 |

概率公式的float64解析核对采用atol=1e-10、rtol=1e-8；Monte Carlo或数值积分采用固定seed和明确统计容差，不使用任意单次采样值作精确真值。真实运动/HOLD仍采用现有限幅与任务阈值。CPU环境的CUDA一致性检查可标记非必要跳过，所有必要真实求解器与概率测试不得跳过。

文档链接、数学分隔符、仓库卫生、Ruff F821和git diff --check必须通过。已有409项基线不机械规定新增总数，验收按覆盖和实际结果记录。

## 五、交付、证据与交接

实施时交付动作解码／条件分布、环境／观测接入、版本化rollout／checkpoint、审计入口及针对性测试；同步接口、架构、运行手册和当前状态。接口与架构已同步T04，完成事实以独立验收记录和机器可读Gate A报告为准。

诊断产物采用独立 `outputs/runs/t04_polar_hold/<run_id>/`，默认seed=23，保存解析配置、代码／配置指纹、命令、JUnit、解析概率误差、分支梯度、真实HOLD轨迹、失败与重跑。新增Gate A机器可读报告须列出每一项门槛和证据路径，不覆盖T02/T03结果。

只有联合契约全部通过，才能将T04及Gate A标记通过。之后交接T05完整基线适配和T06 MPPI基础实现；在T05之前，低维MOVE的NMPC执行仍是明确未完成项。正式训练继续等待后续规划器与无学习闭环门槛，不因Gate A通过而自动恢复。
