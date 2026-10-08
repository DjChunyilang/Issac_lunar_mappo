# 局部信息下的多机器人终端区域选择：文献综述与现有路线重审

日期：2026-09-19。性质：研究综述与设计依据，不是新训练结果，也不是任何方法已在本项目收敛的声明。

配套文件：[完整重设计](../architecture/terminal_feasibility_redesign_v1.md)、[因果诊断与实验设计](../experiments/terminal_feasibility_research_protocol.md)、[27篇文献CSV](terminal_feasibility_literature_2026-09-19.csv)、[检索与证据边界](terminal_feasibility_search_log_2026-09-19.md)。当前运行事实仍以[current_status](../current_status.md)为准。

## 1. 研究判断

项目值得研究的对象是：**在地形信息不完整、消息有范围和时效限制的条件下，多个机器人如何找到并共同进入一个可持续停留的联合终端集合。** 平坦度把原来主要依赖相对几何的集合，变成了依赖环境的选址与运动问题。

目前没有证据把长期失败归结为唯一因素。更符合已有记录的解释是四类困难叠加：终端任务定义与物理意义没有完全对齐；部署时信息不足以支持稳定共同选择；安全运动与任务进展不兼容；奖励与时间尺度没有充分对应最终目标。网络和信用分配只能影响其中部分环节。

新的研究应测量这些机制，而不是先给组合系统命名，再把逐个组件通过当作路线成立。文献支持分层、信息共享、主动感知和约束控制，但没有一篇下列论文直接证明其组合能解决本项目。

## 2. 重新审视项目证据

| 证据 | 可以支持什么 | 不能支持什么 |
| --- | --- | --- |
| exp051旧标准98.83%；新语义复评11.04% | 旧行为与新增终端条件明显不匹配 | 不是只改平坦度的随机受控实验；参考点搜索也变了 |
| exp063约41.94M交互，success 8.20%，final flatness 9.57% | 原策略结构与训练设置没有解决新任务 | 不能证明所有纯RL或所有局部观测策略都不可行 |
| exp064加入实际质心塑形，4.19M交互后success 1.95% | 该项塑形、该预算没有带来有效解决 | 不能证明一切平坦度奖励无效 |
| exp164共同站点条件下success 86.98%、collision 12.50%；近距Open阶段success 98.44% | 有共同目标时部分分布可以学会；复杂场景安全协调仍有问题 | 不是与exp156严格单因素、等预算比较，不能把差值全部归因于站点信息 |
| exp163平地证书覆盖很高；exp165闭环success仅21.88%–43.75% | 证书形成和实际集合之间存在实质缺口 | 不证明共识本身是主要困难，也不证明证书完全无用 |
| exp166后续消融未体现Active-DSTC作用 | 用户提供的历史结论，当前暂停该路线 | 本机缺原始产物，不能补写效应量或显著性 |
| exp167 pilot前后0/96成功；训练规划失败/拒绝54.61% | 学习过程受到明显执行瓶颈影响 | 约0.1M交互与旧几十M预算不可直接比较；不足以否定NMPC或循环策略 |
| exp167修正后固定短目标、长目标改善 | 若干规划—执行缺陷已定位并修正 | 不能替代多车会车、共同选址、完整任务或真实物理验证 |

来源：[exp063](../experiments/exp_063_structured_bicycle_quintic_map25_flatness_oracle_baseline.md)、[exp064](../experiments/exp_064_structured_bicycle_quintic_map25_centroid_flatness_reward.md)、[exp164](../experiments/exp_164_overnight_h1_repaired.md)、[exp165](../experiments/exp_165_active_dstc_closed_loop.md)、[当前状态](../current_status.md)、[exp167诊断](../experiments/exp_167_fixed_goal_diagnosis.md)、[修正](../experiments/exp_167_planner_correction.md)。早期数值本轮只核对文档；exp167另核对本机JSON，不能混称为全部重跑复现。

应修正旧综述的强措辞：“某个长训失败”不是排除预算影响的证明；“若干冻结状态存在联合动作解”不是闭环可达性的证明；“共识快速收敛”不是共同任务可完成的证明。

## 3. 科学问题的结构

### 3.1 收缩、迁移与停留对应不同状态方向

令 $p_i=c+q_i$、$c=N^{-1}\sum_i p_i$、$\sum_iq_i=0$。现有 $d_{\max}$ 和 $D=N^{-1}\sum_i\|q_i\|^2$ 对共同平移不变，而平坦度约束取决于 $c$ 所处的地形。因此几何塑形对共同平移没有方向性进展信号，原Oracle项在历史部分实验中可能提供方向，但当前exp167已关闭它。

要求所有车辆始终朝当前团队中心靠近，还可能妨碍绕障、分工感知和错峰进入。先收紧以后再整体搬迁也不总是好的分解：狭窄通道可能只允许纵队通过。应把队形紧凑主要定义为终端条件，途中只保留由通信和风险真正需要的限制。

### 3.2 四种可行性必须分开

1. **几何可行**：地图上存在满足场地要求且容得下车体的终端构型。
2. **动力学可达**：从当前状态存在符合动力学、避碰与截止时间的联合轨迹。
3. **信息可实现**：各车能在作出不可逆选择前取得足够信息，并协调相容决策。
4. **可学习**：给定策略类、探索、奖励、训练算法与预算，实际能够找到这种行为。

前三项成立仍不保证第四项；有限次规划失败也不能证明第二项不成立。离线全图规划器给出的是成功轨迹见证或未找到解，除非有完备性/不可行性证书，不能把“未找到”标为“任务无解”。

### 3.3 信息不足需要反例，不能凭结果猜测

可构造两张地图：一个机器人直到分叉口前的观测、接收消息和历史完全相同，但在截止时间下两图需要互斥的首次选择；另一机器人能观察差异。对于该信息结构，前者无法保证两图都选对。若允许提前观察、重连或折返，该反例可能失效，所以必须明确决策时限和动作可逆性。

这是本项目可检验的信息必要条件，不是“局部观测必然无法集合”。有限记忆、主动观察与通信都可能打破不可区分性。理论上，共同信息提供了组织去中心化决策的方法，但其模型假设与计算负担必须显式检查。[R01](https://ieeecss.org/paper/decentralized-stochastic-control-partial-history-sharing-common-information-approach)、[R03](https://proceedings.mlr.press/v202/liu23ay.html)

## 4. 文献综述：五条研究线及迁移边界

### 4.1 信息结构、共同知识和通信

Nayyar等的共同信息方法[R01]把私人信息与共享历史分开，为去中心化控制建立动态规划表示。这个“协调者”首先是分析视角，不要求工程上安装中央控制器。MACKRL[R02]在可共同重建的信息上组织协调策略，提示共享网络参数和共享目标知识是不同概念。Liu与Zhang[R03]进一步研究信息共享和可观测性如何改变部分可观测多智能体学习的复杂度。

对项目最有价值的是**先指定决策所需信息，再决定消息格式和网络结构**。需要传递的不一定是整张地图，而可能是候选区域、其未验证部分、各车到达代价与有效期限。可学习性理论的可观测性、共同信息演化和策略无关belief等假设不能直接套到任意丢包、连续地形和GRU上。R03讨论的近似均衡也不等于合作团队的全局最优，不能把其复杂度结果写成“共享消息后即可高效找到最优集合政策”。

SeqComm[R04]指出同时决策可能产生循环依赖，提出分阶段沟通。原文§4.2明确区分全通信推导与局部通信变体；本项目不能把前者的理论性质自动转移给12米内的一跳消息。四车全签能帮助避免冲突选择，但丢包下的完成性、释放和超时是额外问题，不能把有限确认简单称为无限阶共同知识。

### 4.2 主动感知、探索分工和重连

RACER[R05]把探索分工、路径和局部轨迹分开；DSGA[R06]分析并行信息采集造成的重复；Kantaros等[R07]把运动空间和可达信息空间一并规划；iHERO[R08]把通信会合纳入任务计划。这些工作共同提示：不知道平地在哪里时，运动本身也是获取决策信息的手段。

但大面积地图覆盖或熵下降，不等于快速发现全队能进入的停驻区域。本项目的感知价值应该围绕“哪个未知事实会改变终端区域的选择和到达计划”定义。比如验证最后一辆车的入口可能比再扫描一块已确定平整的区域更有价值。这是拟研究的任务特定价值，不是上述论文已验证的结论。

ROAM[R09]展示了分布式地图与规划的一致优化，但不应因此直接引入完整共享三维神经地图。它也不能替代定位与坐标关联的观测来源。共享区域编号前，应先验证不同局部坐标中的两份候选是否确实指向同一物理区域。

### 4.3 地形风险与可持续停驻

STEP[R10]把地形估计不确定性和尾部风险放入运动规划。它支持把未知、已测量、插值支持和高置信可通行分开；不能由某个点坡度合格就声称其周围连续圆盘已经得到传感验证。

Ostafew等的越野路径跟踪[R16]强调学习模型误差与鲁棒约束的结合。月面任务若只使用二维差速加速度缩放，能够研究协调结构，却不能据此保证轮地接触、侧翻、沉陷或低重力下的真实安全。

因此必须区分三个空间对象：共同作业场地区域、各车最终接触足迹、车辆到达时扫过的空间。当前一个质心圆盘无法同时代表三者。

### 4.4 分布式规划：安全不等于到达

MADER[R11]、RMADER[R12]说明已发布轨迹及其有效期比瞬时位置更能表达未来冲突。RMADER的安全论证依赖通信时延上界与delay-check时长关系，不覆盖任意完全失联。其正文还报告过硬约束导致的末端死锁；因此“100%无碰撞轨迹生成”不能解读成所有任务100%完成。

Wang与Ding[R13]对非完整车辆研究终端区域、辅助控制器和预测轨迹相容约束。Mayne等[R15]说明MPC稳定性需要具体条件，有限时域求解本身不是收敛证明。Zhang等[R14]直接研究多安全约束下的非期望平衡点，说明“大家停车”可能是约束控制产生的稳定失败状态。

预测安全过滤[R17]提供学习与约束控制结合的方法，但保证依赖模型、误差集合和可行后备策略。它不能凭空给出探索、选址或通道让行策略，也不能由0碰撞推出高完成率。

### 4.5 学习目标、时间抽象与统计

MAPPO[R18]是有价值的经验基线，不是一般连续Dec-POMDP的收敛定理。COMA[R19]解决特定信用分配问题，不补充部署时缺失的信息。此前DAE/PRD离线门限失败只能否定当时实现满足预注册条件的能力，不能从中推导信用分配对任务永远不重要。

Options[R21]和HIRO[R22]支持把长时决策与短时动作分离，但分层仍需要可达的子目标和稳定低层接口。PRM-RL[R27]进一步说明拓扑规划与局部学习控制可以互补，单独的局部目标控制器不能替代长程拓扑推理。

HER[R23]适用于可重新计算目标奖励的离策略训练；不能把未满足平坦度或碰撞约束的终点重新标为成功，也不能不加修正地接入当前on-policy MAPPO。2026年的多步准度量目标学习[R25]提供学习时间距离的近期方向，但其长时目标到达结果不是去中心化协同选址的现成解。

塑形[R20]需要区分改变目标与帮助优化目标；CPO[R24]提示显式约束是加权惩罚之外的选择，但期望成本约束不等于每条轨迹安全。Agarwal等[R26]支持以多种子、区间与配对分布比较替代“最好checkpoint的一条曲线”。

## 5. 当前技术路线的具体问题

### 5.1 平坦度指标与物理任务存在缺口

当前几何判据允许以下四车位置，单位米：

$$
(0.78,0),\quad(-0.26,-0.42),\quad(-0.26,0),\quad(-0.26,0.42).
$$

它们的质心为0，$d_{\max}\approx1.12161$米，$D=0.291$平方米，最小间距为0.42米，全部满足对应几何门槛；但第一辆车的中心距离质心0.78米，已超出0.75米平坦度查询半径，更不用说车体足迹。这是解析反例，经独立数值计算及项目实际`compute_team_metrics/compute_success_gates`核对，不是模拟训练结果。

对于沿采样轴的理想斜平面 $h(x,y)=ax$，圆盘高度差为 $2ra$。$r=0.75$、高度差上限0.18意味着 $a\le0.12$，对应约6.84°；单独坡度0.25则对应约14.04°。因此当前“平坦”同时限制绝对倾斜与起伏，不能称为去趋势后的粗糙度。有限37点检查也不是对所有连续地形点的证明。

如果任务是共同作业区，这种区域约束可能合理，但需要说明面积与作业依据；如果只是安全停车，更直接的变量可能是足迹内拟合坡度、残差起伏、接触和支撑裕度。应另立新任务版本，不能默默改旧阈值来提高成功率。

### 5.2 平地证书不是联合可达证书

历史Oracle主要检查圆盘地形和直线风险摘要。平地可能位于无法按时到达的连通分量，也可能仅能逐车进入，或一辆先到的车占住其余车入口。旧Active-DSTC侧重发现、关联和达成选择，弱化了最后一辆车的可达性和进入顺序。

这不是要求重新启用Active-DSTC。它已按用户历史反馈暂停。新方案必须先用独立对照证明：把逐车到达代价、入口资源和终端驻留一起考虑，确实优于仅传递平地证据。

### 5.3 地形支持与定位精度是假设

exp167 `LocalTerrainMemory`使用最近邻特征与0.155米支持半径。没有噪声界、地形正则性或连续插值误差界时，“有支持”只能是模型假设，不是该圆盘每一点均被观测且安全的物理证书。

`environment.py`把仿真真值变换成各车起始里程计系，并在模拟消息传输时使用精确坐标变换。这可作为理想定位基线，但尚未建模里程计漂移与跨车配准误差。局部坐标表示并不自动消除这些假设。新架构必须单独报告理想定位与有误差定位两种结果。

### 5.4 多个动作请求可能被执行层压成同一种行为

当大量不同目标都导致旧计划继续执行或减速停车时，高层动作到实际运动的映射接近多对一。Actor即使输出多样动作，也可能收集不到对应的行为差异。应在冻结状态上测量请求变化、实际位移、接受率和回报的联合关系；只看动作熵或参数更新不够。

当前exp167保留原始请求的log-prob，并使用时长感知GAE，不能误写成这两个接口尚未处理。真正未解决的是控制接口的有效作用范围、目标切换代价和可行域外请求比例。

### 5.5 奖励中存在可审计的冲突和时间偏好

当前平坦度分量为 $2(P_{t-1}-P_t)-0.02a_t[C_t-1]_+$，其中 $P=aC$。地形不变时收紧队形会增大激活度 $a$ 并产生负的平坦度进展；散开则相反。该结论只针对这一分量，总奖励是否被它主导需要数据。

未折扣的往返相消不能保证折扣回报相消。例如 $P:1\to0\to1$ 的进展回报为 $2-2\gamma$，在 $\gamma=0.99$ 时为0.02；其他奖励和动作代价可能抵消它，不能称策略已利用此漏洞。标准折扣势函数形式和终止处理见[R20](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)。

exp167在0.2秒步长使用0.99折扣，半衰期约13.79秒，96秒后的权重约0.00803。未来130分的终端奖励若接近480步末端才出现，其起点权重约1.04分。这是目标中的强时间偏好，不是“超过14秒就学不到”的硬界；准确回传还受Critic、GAE与探索影响。新设计应按物理时间解释折扣，并把同一任务目标与不同训练估计器分开比较。

### 5.6 验证链条缺少稳定的因果对照

地图、动力学、动作、网络、奖励和执行机制多次同时改变。大量实验能定位局部失败，但跨版本的成功率差不能自动用于归因。零碰撞可能来自完全不动，低timeout也可能因为更早碰撞终止，首个证书时间也可能与最终集合无关。必须同时报告完成、风险、运动、信息与资源代价。

## 6. 文献清单：27篇及其用途

表中年份按正式出版/会议年份；MADER另标在线年。RA-L属于机器人领域相关同行评审期刊，本文不把所有期刊笼统称为“顶刊”。全文机制核查集中在R01、R03、R04、R10、R12、R20，其余主要依据官方摘要、正式元数据与作者说明；详见检索日志。

| ID | 作者、年份、正式来源 | 论文与真实链接 | 对项目的用处及限制 |
| --- | --- | --- | --- |
| R01 | Nayyar, Mahajan, Teneketzis；2013；IEEE TAC | [Decentralized Stochastic Control with Partial History Sharing: A Common Information Approach](https://ieeecss.org/paper/decentralized-stochastic-control-partial-history-sharing-common-information-approach)；[作者预印本](https://arxiv.org/abs/1209.1695) | 分开共同与私人信息；动态规划不是可直接部署的轻量算法 |
| R02 | Schroeder de Witt等；2019；NeurIPS | [Multi-Agent Common Knowledge Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f968fdc88852a4a3a27a81fe3f57bfc5-Abstract.html) | 协调依赖共同可重建信息；不等于任意有限消息均形成共同知识 |
| R03 | Liu, Zhang；2023；ICML | [Partially Observable Multi-agent RL with (Quasi-)Efficiency: The Blessing of Information Sharing](https://proceedings.mlr.press/v202/liu23ay.html) | 研究信息结构与学习复杂度；定理条件需核查，不给本项目MAPPO保证 |
| R04 | Ding等；2024；NeurIPS | [Multi-Agent Coordination via Multi-Level Communication](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d6be51e667e0b263e89a23294b57f8cf-Abstract-Conference.html) | 分阶段沟通与决策顺序；局部通信版保证弱于完整通信推导 |
| R05 | Zhou, Xu, Shen；2023；IEEE T-RO | [RACER: Rapid Collaborative Exploration With a Decentralized Multi-UAV System](https://doi.org/10.1109/TRO.2023.3236945)；[作者代码](https://github.com/Robotics-STAR-Lab/RACER) | 分工与层级规划；无人机覆盖不等于非完整地面车共同停驻 |
| R06 | Corah, Michael；2017；RSS | [Efficient Online Multi-robot Exploration via Distributed Sequential Greedy Assignment](https://www.roboticsproceedings.org/rss13/p70.html) | 减少重复信息采集；信息目标的结构条件需满足 |
| R07 | Kantaros, Schlotfeldt, Atanasov, Pappas；2019；RSS | [Asymptotically Optimal Planning for Non-Myopic Multi-Robot Information Gathering](https://www.roboticsproceedings.org/rss15/p62.html) | 运动与信息联合规划；不直接给有限通信端到端选址算法 |
| R08 | Tian, Zhang, Wei, Guo；2024；RSS | [iHERO: Interactive Human-oriented Exploration and Supervision Under Scarce Communication](https://www.roboticsproceedings.org/rss20/p115.html) | 主动安排信息交换；人机监督任务与本项目不同 |
| R09 | Asgharivaskasi, Girke, Atanasov；2025；IEEE T-RO | [Riemannian Optimization for Active Mapping With Robot Teams](https://doi.org/10.1109/TRO.2025.3526295)；[作者项目](https://existentialrobotics.org/ROAM/) | 分布式地图/规划一致性；不自动解决终端容量和进入冲突 |
| R10 | Fan等；2021；RSS | [STEP: Stochastic Traversability Evaluation and Planning for Risk-Aware Off-road Navigation](https://www.roboticsproceedings.org/rss17/p021.html) | 地形不确定性、尾风险与动力学规划；不是多车共同终端选择 |
| R11 | Tordesillas, How；2022卷期，2021在线；IEEE T-RO | [MADER: Trajectory Planner in Multiagent and Dynamic Environments](https://doi.org/10.1109/TRO.2021.3080235)；[作者代码](https://github.com/mit-acl/mader) | 发布轨迹与连续避碰；UAV动力学和通信前提不同 |
| R12 | Kondo等；2024；IEEE RA-L | [Robust MADER: Decentralized Multiagent Trajectory Planner Robust to Communication Delay in Dynamic Environments](https://doi.org/10.1109/LRA.2023.3342561)；[全文](https://arxiv.org/html/2303.06222v4) | 时延下的轨迹去冲突；需要时延界、不覆盖完全失联、不保证无死锁 |
| R13 | Wang, Ding；2014；IEEE TAC | [Distributed RHC for Tracking and Formation of Nonholonomic Multi-Vehicle Systems](https://doi.org/10.1109/TAC.2014.2304175) | 终端区域、辅助控制与轨迹相容；同步和模型假设不能忽略 |
| R14 | Zhang等；2025；IEEE T-RO | [Deadlock-Aware Control for Multirobot Coordination With Multiple Safety Constraints](https://doi.org/10.1109/TRO.2025.3600159)；[作者代码](https://github.com/Parker-Zhang/Shaping-CLF-MCBF) | 安全约束引起非期望平衡；不能直接复制为任意地形全局完备规划器 |
| R15 | Mayne, Rawlings, Rao, Scokaert；2000；Automatica | [Constrained Model Predictive Control: Stability and Optimality](https://doi.org/10.1016/S0005-1098(99)00214-9) | 区分可行性、稳定性和有限时域优化；定理均有具体条件 |
| R16 | Ostafew, Schoellig, Barfoot；2016；IJRR | [Robust Constrained Learning-based NMPC Enabling Reliable Mobile Robot Path Tracking](https://doi.org/10.1177/0278364916645661) | 越野模型误差与鲁棒跟踪；不负责共同选址 |
| R17 | Wabersich, Zeilinger；2021；Automatica | [A Predictive Safety Filter for Learning-based Control of Constrained Nonlinear Dynamical Systems](https://doi.org/10.1016/j.automatica.2021.109597) | 约束与学习模块化；保证需误差模型及可行后备策略 |
| R18 | Yu等；2022；NeurIPS Datasets and Benchmarks | [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf) | MAPPO经验基线；不是一般Dec-POMDP收敛定理 |
| R19 | Foerster等；2018；AAAI | [Counterfactual Multi-Agent Policy Gradients](https://ojs.aaai.org/index.php/AAAI/article/view/11794) | 区分个体动作贡献；不能解决缺失信息或不可执行动作 |
| R20 | Ng, Harada, Russell；1999；ICML | [Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf) | 折扣势函数塑形；需处理终止边界，不能承诺有限样本学习成功 |
| R21 | Sutton, Precup, Singh；1999；Artificial Intelligence | [Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning](https://doi.org/10.1016/S0004-3702(99)00052-1) | 持续时间动作与SMDP；分层不是自动可达性证明 |
| R22 | Nachum等；2018；NeurIPS | [Data-Efficient Hierarchical Reinforcement Learning](https://proceedings.neurips.cc/paper/2018/hash/e6384711491713d29bc63fc5eeb5ba4f-Abstract.html) | 目标条件低层与高层；两层同时改变会造成非平稳 |
| R23 | Andrychowicz等；2017；NeurIPS | [Hindsight Experience Replay](https://proceedings.neurips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html) | 稀疏目标奖励的离策略数据利用；不能重标掉安全和平地条件 |
| R24 | Achiam, Held, Tamar, Abbeel；2017；ICML | [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html) | 奖励与约束分开；期望约束不等于逐轨迹安全 |
| R25 | Zheng, Myers, Eysenbach, Levine；2026；ICLR | [Scaling Goal-conditioned Reinforcement Learning with Multistep Quasimetric Distances](https://proceedings.iclr.cc/paper_files/paper/2026/hash/eddc0fab5a42f8d8de6eb5566cd9f1d3-Abstract-Conference.html) | 学习非对称时间距离的近期参考；不证明未知地形多车协调有效 |
| R26 | Agarwal等；2021；NeurIPS | [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) | 多运行不确定性、IQM和性能分布；不把多episode当作多训练seed |
| R27 | Faust等；2018；ICRA | [PRM-RL: Long-range Robotic Navigation Tasks by Combining Reinforcement Learning and Sampling-Based Planning](https://doi.org/10.1109/ICRA.2018.8461096)；[作者预印本](https://arxiv.org/abs/1710.03937) | 长程拓扑与局部控制互补；原地图和多机器人条件需另行处理 |

## 7. 新路线应形成的可证伪结论

研究主张应落在以下三项，而不是宣称“提出一种融合多模块的新网络”：

- **信息机制**：在固定可行地图和执行器下，带历史与任务相关消息的条件，能否减少不可区分场景中的错误共同选择？
- **联合可达机制**：把最迟到达车辆、入口冲突和终端驻留纳入候选评价，能否显著减少“选了平地却进不去”的失败？
- **学习机制**：在信息与执行已验证后，学习的感知/选址决策是否比同信息、同控制器、同预算的确定性策略更好？

任何一项没有体现增益，都应收缩主张。完整方案与反证条件见[重设计](../architecture/terminal_feasibility_redesign_v1.md)和[实验协议](../experiments/terminal_feasibility_research_protocol.md)。
