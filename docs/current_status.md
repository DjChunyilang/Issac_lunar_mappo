# 当前状态

## 当前主线

- 当前执行主线为严格去中心化101维观测、12 m分级通信、Pure RL和MAPF离线诊断。exp125的六组B0 4M screen均未通过；exp137又完成一次单变量B2例外并在基础gate停止，因此不启动40M，也不继续B1/B2/B3结构扩展。B0 `relative_quintic` 的评测dmax ratio为 `0.2047`、success为 `0.0518`；B2虽把dmax改善到 `0.1707`，却把collision推高到 `0.7881`。近距运行始终为完整通信，当前失败不是通信年龄或邻居排列问题，而是地形动作、团队集合和碰撞安全之间的训练信用冲突。B0完整记录见 [exp_125_decentralized_tiered_b0_pure_rl.md](experiments/exp_125_decentralized_tiered_b0_pure_rl.md)，B2结果见 [exp_137_decentralized_b2_graph_attention.md](experiments/exp_137_decentralized_b2_graph_attention.md)。

- 已修正两项会影响结论的语义偏差：路径风险现在沿实际 quintic 轨迹采样，不再以直线代理；4M checkpoint 评测按照 checkpoint 的课程时步使用 2.4–3.4 m 近距分布，不再错误使用最终远距分布。主配置也已显式恢复实际质心平整度奖励。

- exp125 信用诊断已完成：`shared_joint` 对四车复制同一团队 GAE advantage；在 `relative_quintic` checkpoint 的 245,760 个车辆样本上，相对路径风险与一步 TD advantage 代理的 Pearson 相关为 `-0.0123`，与单车质心进度、最近邻距离变化和团队 dmax 进度的相关绝对值均不超过 `0.012`。这支持“共享团队信用无法区分单车地形决策”的假设，但尚不授权直接改用新的多智能体奖励算法。

- exp126 已完成唯一一次零和地形信用残差 4M 对照。团队 reward 和 Critic target 保持原样，车辆信用只进入 Actor advantage；不变量和工程测试全部通过。相对 exp125 `relative_quintic`，路径风险改善从 `-0.16%` 变为 `+2.74%`，但 collision 从 `0.0967` 激增到 `0.6484`，dmax ratio 退化为 `0.6932`，success 仅 `0.0029`，重复车辆对冲突升至 `0.1927/步`。因此 C0 被否决，不启动 40M、不扫描信用系数。结果见 [exp_126_decentralized_b0_centered_terrain_credit.md](experiments/exp_126_decentralized_b0_centered_terrain_credit.md)。

- exp127 已完成冻结联合动作 Critic 的离线可行性诊断。正式数据覆盖完整 96 秒 episode，两个验证种子的预测冲突参与率约为 `16%–17%`；16 步 held-out MSE 仅改善 `2.16%`，最大安全秩相关仅 `0.0896`，1 步和 4 步完整 episode 复核也未改善。C1 被否决，不扩展为 COMA/MADDPG/FACMAC。结果见 [exp_127_joint_action_critic_feasibility.md](experiments/exp_127_joint_action_critic_feasibility.md)。

- exp128 已完成奖励分量动作可辨识性诊断。`gather`、预测冲突和最近邻距离变化的动作增益分别为 `51.00%/17.56%/56.18%`，证明动作与动力学链路有效；但现有 `safety`、`terrain` 和总奖励仅为 `0.10%/-0.26%/0.13%`，关键奖励语义没有提供稳定的当前动作边际信号。按预设门限不改变训练信用、不启动新 4M；下一步只做冻结状态配对动作干预。结果见 [exp_128_reward_component_identifiability.md](experiments/exp_128_reward_component_identifiability.md)。

- exp129 已完成冻结状态的配对单车动作干预。15,360 个车辆—状态样本中，降低相对路径风险与提高轨迹安全裕量的方向余弦中位数仅 `0.0202`；强一致和强冲突分别占 `27.64%/26.94%`。动作对两项目标均有响应，但局部优化方向整体近似正交，因此不启动固定地形—安全权重训练，也不恢复方向性安全投影。结果见 [exp_129_paired_action_interventions.md](experiments/exp_129_paired_action_interventions.md)。

- exp130 已完成团队 PPO 与地形信用 Actor 梯度冲突审计。两个独立地形种子的全 Actor 负余弦批次比例均为 `46.875%`，terrain encoder 均为 `50%`，辅助/主梯度范数中位比约 `1.14`。这证明 exp126 的地形信用直接相加确实经常反对团队目标，达到 C2 离线启用门限。当前只授权一次非对称、主任务优先且带范数上限的梯度投影 4M screen，不授权 40M。结果见 [exp_130_actor_gradient_conflict_audit.md](experiments/exp_130_actor_gradient_conflict_audit.md)。

- exp131 已完成唯一一次 C2 主任务优先梯度投影 4M screen。投影工程门限全部通过，最后一次更新冲突比例 `65.625%`、最低主方向余弦 `0.970142`；但评测 dmax ratio `0.3525`、success `0.0361`、collision `0.6680`、timeout `0.2959`，路径风险改善仅 `2.38%`。`t=1024` collision 同样为 `0.6719`。因此 C2 被否决，不启动 40M、不扫描投影参数。结果见 [exp_131_primary_projected_terrain_credit.md](experiments/exp_131_primary_projected_terrain_credit.md)。

- exp132 已完成现有 \(0.42\,\mathrm m\) 最近邻成功门限的逐车安全势函数审计。两个种子的信用激活率约 `4.1%`，安全/团队梯度范数比为 `0.608/0.544`，地形—安全负余弦批次比例为 `46.875%/59.375%`；但 8 段 rollout 中只有 6 段出现安全事件，总体 trace std 均为 `0.8660`，未达到预注册下界 `0.90`。因此不训练 C3、不事后放宽门限。结果见 [exp_132_agent_safety_potential_credit.md](experiments/exp_132_agent_safety_potential_credit.md)。

- exp133 已完成现有 `safety.near_distance=0.72 m` 团队项的逐车信用审计。信用密度提高到 `17.540%/16.704%`，且梯度非退化；但 seed22023 前两个 rollout 仍无事件，首次激活索引 `2`、激活段比例 `75%`、trace std `0.8660`，未通过预注册时间覆盖门限。因此不训练 C3-near、不扩大距离或扫描权重。结果见 [exp_133_agent_near_distance_credit.md](experiments/exp_133_agent_near_distance_credit.md)。

- exp134 已完成连续近距信用相对冲突和碰撞的提前量诊断。`222/244` 个碰撞车辆事件全部提前进入 `0.72 m`，至少提前 2 步，提前量中位数为 `74/67` 步；但约 1.97 万个预测冲突事件只有 `37.956%/35.646%` 在解除前被近距信用覆盖，未达到 70% 门限。因此 C3-near 仍停止。结果同时表明当前单步 quintic 冲突诊断包含大量会随重规划解除的事件。完整记录见 [exp_134_near_credit_lead_time.md](experiments/exp_134_near_credit_lead_time.md)。

- exp135 已完成单步与 pair-repeated 冲突的事件级结果匹配。两个种子的非重复事件 `14,632/14,352` 个，碰撞结果均为 0；重复事件 `4,556/4,511` 个，碰撞结果率为 `5.224%/4.899%`，并在前 8 步召回全部 `238/221` 个碰撞车辆事件。后续 MAPF/B2 判断只允许使用重复冲突，单步冲突仅保留日志；该结果不自动启用 B2。完整记录见 [exp_135_repeated_conflict_outcomes.md](experiments/exp_135_repeated_conflict_outcomes.md)。

- exp136 已完成失败 episode 的重复冲突触发审计。`141/130` 个失败 episode 全部包含 pair-repeated 冲突，事件数中位数为 `17/18`，collision与timeout失败均为100%命中。B2冲突证据已经满足，但按当时的原始矩阵仍缺基础收敛；随后只通过exp137授权了一次显式例外。完整记录见 [exp_136_failed_episode_repeated_conflicts.md](experiments/exp_136_failed_episode_repeated_conflicts.md)。

- exp137 已完成一次 seed23 的 B2 单跳图注意力例外筛选。工程gate全部通过，但4M评测虽然把dmax ratio改善到 `0.1707`，collision却升至 `0.7881`，success只有 `0.0439`；terrain动作MSE和路径风险gate仍失败。双种子离线复核的重复冲突事件中位数由B0的 `17/18` 变为 `16/22`，没有一致下降。因此B2在基础gate停止，不运行40M或结构调参。完整记录见 [exp_137_decentralized_b2_graph_attention.md](experiments/exp_137_decentralized_b2_graph_attention.md)。

- exp138 已完成现有安全奖励聚合辨识诊断。安全目标激活率为 `28.906%/25.837%`，但worst-pair聚合的动作增益只有 `0.0973%/0.00016%`，相对当前mean聚合仅提高 `0.0120/0.0166` 个百分点。均值稀释不是安全信用瓶颈，因此不修改安全聚合、不启动4M。完整记录见 [exp_138_safety_aggregation_identifiability.md](experiments/exp_138_safety_aggregation_identifiability.md)。

- exp139 已完成逐车局部安全信用动作辨识审计。raw信用动作增益为 `15.791%/14.901%`，第二个验证种子未过15%；满足共享团队语义的逐步零和中心化信用仅为 `8.219%/7.203%`。信用密度、exp134碰撞提前量和冻结Actor不变量均通过，但核心跨种子动作辨识门限失败。因此不重开C3-near、不扫描信用形式或权重，也不把重复冲突写入奖励。完整记录见 [exp_139_local_safety_credit_identifiability.md](experiments/exp_139_local_safety_credit_identifiability.md)。

- exp140已完成非零和逐车近距信用4M组件筛选。工程gate和信用激活均通过，但独立评测collision由B0的 `0.0967` 升至 `0.2295`，双种子重复冲突中位数由 `17/18` 升至 `19/20`；dmax ratio `0.2397`、success `0.0439`。因此该方向停止，不扫scale、trace或距离，不与terrain信用叠加，也不进入40M。协议与结果见 [exp_140_agent_local_near_credit_screen.md](experiments/exp_140_agent_local_near_credit_screen.md)。

- exp141已完成真实collision cost-value冻结审计。双验证种子AUROC为 `0.9392/0.9331`、AUPRC为 `0.6231/0.5792`、Brier改善为 `35.59%/30.00%`，全部通过预注册门限；Actor完全不变。因此只授权形成exp142 PPO-Lagrangian组件计划，尚未授权40M。结果见 [exp_141_collision_cost_value_feasibility.md](experiments/exp_141_collision_cost_value_feasibility.md)。

- exp142已完成训练期真实collision PPO-Lagrangian组件筛选。工程门限和CPU/CUDA smoke全部通过；4M中训练碰撞率首末四分之一降低 `95.03%`，评测collision为 `0.02539`，双种子重复冲突中位数均降为0。但success为 `0`、dmax ratio为 `0.68146`、timeout为 `0.97461`，地形动作MSE仅 `0.000259`。策略以回避集合换取安全，因此状态为 `stopped_at_component_gate`：不启动40M，不扫描cost预算、dual参数、PID项或网络容量。结果见 [exp_142_collision_lagrangian_component_plan.md](experiments/exp_142_collision_lagrangian_component_plan.md)。

- exp143/144已复核“B0只是训练不够深”的解释。t2048相对t1024在五个评测种子上全部改善，平均dmax降低 `29.76%`、collision降低 `89.94%`、success增加 `4.41` 个百分点；但三个terrain-contrast种子中，地形动作MSE虽从约 `0.00057` 增至 `0.00746`，实际quintic路径风险却全部恶化，平均趋势为 `-0.2508` 个百分点。因此不启动12M/40M B0：更深训练会增强地形输入影响，但现有共享团队advantage没有把该影响导向低风险路径。结果见 [exp143](experiments/exp_143_b0_horizon_and_constraint_competition_audit.md) 与 [exp144](experiments/exp_144_b0_checkpoint_trend_multiseed_audit.md)。

- exp145已完成统一逐车局部任务回报审计。局部集合、地形和统一目标动作增益分别为 `77.58%/28.44%/25.69%`，但局部安全平均仅 `9.58%`、最差种子 `8.12%`。安全事件覆盖和方差均充分，Actor摘要与探针动作不变，因此不降低15%门限、不删除安全分量、不训练统一逐车advantage。结果见 [exp145](experiments/exp_145_unified_agent_local_task_reward_identifiability_plan.md)。

- exp146已完成最近邻成对安全动作耦合审计。邻车动作乱序使MSE平均恶化 `91.36%`，但成对总增益最差种子只有 `23.93%`，本车/邻车条件边际增益最差为 `12.17%/14.09%`，未达到25%/15%/15%门限。因此不实现成对安全Critic，不扫描容量或时域。结果见 [exp146](experiments/exp_146_nearest_pair_safety_action_coupling_plan.md)。

- exp147发现当前收敛问题还包含基础轨迹执行契约失配。quintic生成器把最长约1.65 m的完整路径统一压缩为0.2 s，控制器固定跟踪第2个点并立即重规划；双种子 `78.25%–79.45%` 的非零路径要求超过1.35 m/s，实际一步只执行规划弧长约 `1.77%`。这使地形奖励和冲突诊断评价策略未完整执行的路径。全部预注册门限确认失配，只授权 [exp148](experiments/exp_148_trajectory_time_consistency_fix_plan.md) 的单一时间一致性修正；工程门限通过前不训练。

- exp148已完成时间一致性修正和唯一一次B0 4M重新筛选。弧长时域、0.2秒物理前视控制和公共时间冲突对齐全部通过工程门限；时间戳速度违例降为0，冻结双种子单步弧长利用率中位数提高到 `11.82%–12.05%`，Actor不变。但随机初始化4M评测为 dmax ratio `0.3528`、success `0`、collision `0.9990`、timeout `0.0010`；terrain contrast动作MSE仅 `0.00383`，路径风险变化为 `-0.414%`。双种子失败episode复核中 `405/388` 个episode全部碰撞结束且100%出现重复车辆对冲突，中位事件数均为5。故工程修复保留，训练状态为 `stop_before_40m`；近距消息全部完整且年龄为0，不启动B1/B2/B3，不继续调整控制器或奖励。完整记录见 [exp148](experiments/exp_148_trajectory_time_consistency_fix_plan.md)。

- exp149已完成碰撞参与者信用可行性冻结审计。四个checkpoint—种子组合均显示典型碰撞只涉及2辆车，约50%的车辆接收了与自身无直接碰撞关系的团队终止惩罚；碰撞对在终止前8/16步的重复冲突召回约为 `99.7%–100%`，首次命中提前量中位数为 `14–21` 步。该结果只授权按真实终止参与者进行一次信用筛选，预测冲突仍只作诊断。结果见 [exp149](experiments/exp_149_collision_participant_credit_feasibility.md)。

- exp150已完成真实碰撞参与者零和Actor信用工程复核和唯一一次seed23 4M筛选。信用逐步零和、团队reward保持、source reconstruction、CPU/CUDA smoke全部通过；训练dmax降低 `59.22%`，但success episode为0。独立评测dmax ratio `0.2756`、success `0`、collision `0.9990`、timeout `0.0010`；terrain contrast动作MSE仅 `0.000810`，路径风险改善 `0.0466%`。双种子重复冲突中位数为 `9/8`，较exp148的 `5/5` 恶化。因此该方向在4M门限停止，不启动40M，不扫描scale、trace、惩罚或参与距离。结果见 [exp150](experiments/exp_150_collision_participant_actor_credit_plan.md)。

- exp151已完成碰撞参与者信用因果有效性审计。冻结t1024/t2048策略、双种子共四个组合均通过Actor和执行不变性检查；但碰撞前8步存在不增加地形风险和集合退化的局部避碰候选比例最小仅`0.5100`，16步最小仅`0.5616`，低于`0.70/0.60`门限。8步两车均可行动比例只有`0.1977–0.2853`，等量信用支持率只有`0.1777–0.2454`。因此不再设计终止参与者信用，下一步仅做冻结的动作—规划—控制可控性分解。审计过程中还修正了MAPF车辆对时间轴受第三车轨迹时长污染的问题；该修正只影响诊断。结果见 [exp151](experiments/exp_151_collision_credit_causal_validity.md)。

- exp152已完成动作—规划—控制可控性分解。联合层对exp151重建误差为0。t1024的8步无约束避碰可行动率为`0.8436/0.8475`，但t2048降至`0.6648/0.6676`，跨组合最小值未过`0.70`；16步最小值为`0.7020`。已有无约束避碰轨迹的控制传递率仍为`0.8069–0.9091`，线速度没有饱和、角速度饱和最高仅`1.29%`，说明低层控制不是首要瓶颈。约八成最佳候选来自方位角动作。下一步只允许冻结比较动作范围、局部候选覆盖和quintic几何。结果见 [exp152](experiments/exp_152_action_planning_controllability_decomposition.md)。

- exp153已完成动作范围与quintic几何分离。现有全动作网格把t2048的8步可行动率提高到`0.7841/0.7536`，但跨组合最小值未达到`0.80`，范围恢复率最小仅`0.0645`；line参考最小仅`0.6275`且不稳定优于quintic，故动作覆盖和quintic几何都不是单一瓶颈。四组合endpoint可行动率均为1，但t2048途中交叉损失达到`0.3466/0.3725`。这支持“单车终点可达、相互轨迹缺少联合协调”的新假设，但尚未授权训练；下一步只允许冻结双车联合动作干预。结果见 [exp153](experiments/exp_153_action_range_quintic_geometry_audit.md)。

- 当前任务定义保持为“真实地形可行集合点 + 实际质心平整度 gate + 可执行局部目标”。`terrain_aware_multiresolution` 每个 reset 搜索真正的地形约束最优点；成功只认可实际团队质心的 37 点平整圆盘，绝不认可 oracle 代理点。exp092 的 `BC32` 在原 `64 s/320` steps 为 `0.1910/0.7002/0/0.2998`；按当前决策，**后续训练和正式 proxy 评估固定采用 `96 s/480` steps 时域**。这是执行时间预算从 64 秒放宽到 96 秒，严格验收仍为 dmax `<=0.2`、success `>=0.9`、collision `<=0.02`、timeout `=0`，没有放宽。exp094 的基线复评为 `0.1837/0.8594/0/0.1406`；128 秒仅作为“时域仍偏紧”的诊断上界，不能替代 96 秒标准。

- exp098–exp101 已在 96 秒时域完成末段干预和 PPO 诊断。严格逐槽位捕获退化；增大共同中心校正在相同 exp092 BC32 后验对照中暂时最好（exp099：`0.1843/0.8643/0/0.1357`）；真实局部平整候选搜索不再以几何中点代理目标，而对当前质心及附近环形候选运行与 success gate 相同的 37 点平整度检查。它使最终实际平整率达到 `0.9150`，但 success 仅 `0.8604`、timeout `0.1396`，未优于 exp099。以 `1e-5` 从 BC32 warm-start 的 4M environment-step PPO probe 中，`t=512/1024/1536/2048` success 依次降为 `0.8154/0.7539/0.7109/0.6621`，只保留 `t=0`，不触发 PhysX。

- exp102–exp109 已完成条件末段控制、鲁棒集合点和槽位半径的同协议后验筛选，均低于 exp099：宽局部搜索 `0.8359` success，原地几何收紧 `0.8535`，分支组合 `0.8438`；5/7.5 cm 鲁棒搜索将实际平整率提高到 `0.9443/0.9551`，但 success 降至 `0.8535/0.8457`；0.33/0.34 m 槽位也分别为 `0.8379/0.8467`。原地收紧的 gate 复诊显示几何通过率略升却增加平整度失败，动态槽位重匹配无差异。所有变体 collision 都是 0，故全部 reject，不启动 PPO。

- exp110/111 已完成该教师/目标层的短 BC screen。动态真实平整槽位在 `10.4%–15.3%` 的有效步生效，但未更新 BC32/BC8/关闭固定中心校正的 success 分别为 `0.8574/0.8516/0.8027`、timeout 为 `0.1426/0.1484/0.1973`，均低于 exp099；BC8 还令最终实际平整率降至 `0.8945`。这说明随机快照 BC 和单步动态局部平整目标不能形成需要的闭环末段轨迹，且固定 terrain-aware 中心校正并非可简单移除的冲突项。全部 reject，不启动 PPO 或 PhysX。

- exp112--exp117 已完成时序 gate、on-policy 尾部 BC 与执行控制筛选。用约 103 万个 policy-visited 近末段样本做 BC8 仍退化为 `0.1875/0.8457/0/0.1543`，所以同一固定槽位教师的随机/闭环数据比例不是主要瓶颈。仅提高 `k_linear` 则稳定改善：`2.20 -> 3.20` 将同一 BC32 的独立 96 秒评测从 exp099 的 `0.1843/0.8643/0/0.1357` 提升至 exp116 的 `0.1802/0.8916/0/0.1084`；这是当前最佳执行设置，但严格验收仍未通过。对 exp116 的 111 个 timeout，57 个仍缺 dmax、52 个仍缺 dispersion、91 个仍缺实际平整度（可重叠），只有 2 个是纯 hold 未完成；29 个从未进入平整 footprint，62 个进入后又离开。提前共同中心校正使 success 回落到 `0.8838`，故拒绝。

- exp118--exp124 是 2026-07-29 工作树中的未提交探针，尚未进入实验索引。去除末段 damping、anchored/tail-only/disagreement BC、动态平整目标和 teacher-rollout BC 在 seed1023 上最多只带来 `2/1024` 的 success 波动；exp119 BC8 虽在独立 seed11023 达到 `0.1785/0.9043/0/0.0957`，但 seed1023 反而低于 exp116，且 timeout 仍远非 0，不能晋升为正式候选。当前固定槽位教师直接指向目标槽位，相关配置还关闭 `teacher_terrain_scale`，没有提供地形相关 waypoint/绕行知识；动画和跨 seed 结果均不支持继续追加同类 BC。

- 已提交的历史最佳仍是 exp092 的 `BC32` checkpoint 配合 `exp116` 的 `k_linear=3.20` 执行设置；它只是 BC-based 对照 candidate，不能触发 PhysX，也不再作为新策略学习起点。后续主线取消 BC：新训练固定 `bc_updates=0`，不从 BC checkpoint 初始化，在不放宽 96 秒、实际平整 gate 或 strict timeout 的前提下，用 pure RL 学习对局部地形敏感的多步路径；exp063 表明直接复用旧 reward 做朴素 pure RL 不足，下一轮需先验证路径风险进度信号和 terrain-contrast 行为，再决定 40M formal。
- 当前主设计口径已切换为“高吞吐 proxy 训练 + Isaac Sim / Isaac Lab / PhysX 高保真闭环评估”。当前实施路线以 `docs/implementation_plan.md` 和 `docs/architecture/overall_plan_v3.md` 为准。
- 训练主环境仍是 PyTorch / torch-vectorized proxy 环境，用于 MAPPO / PPO 采样、奖励调试、观测接口验证和大规模对照实验。
- Isaac Sim / Isaac Lab / PhysX 不作为当前主训练 loop，而作为 high-fidelity validation、迁移 sanity check、失效分析和可视化展示平台。
- 当前 PhysX 层使用 Clearpath Jackal 作为活跃轮式资产，已替换旧占位资产。Jackal tracking 可验证轮式控制、强三维地形 mesh、姿态稳定性和输出链路，但不能证明真实月球车越障、轮壤接触或低重力动力学已经完成。
- 视觉观测不进入 policy input；地形以车体系 `5×5×2` 局部结构化网格进入策略。
- 当前从“暂停长训完善环境”切回新环境栈长训迭代：结构化 Actor/Critic、bicycle proxy 动力学、quintic 轨迹、`25 m × 25 m` 地图和 `communication_radius=0.0` 无限通信语义已通过 `exp042` smoke。`exp043` 直接 40M env-step 长跑已完成但没有收敛；`exp044` 改为 initial-state curriculum 后能明显缩短 dmax，但 success 仍为 0；`exp045` local-success bootstrap 首次恢复局部 success 信号但未 strict；`exp046`/`exp047` 逐步恢复 terminal convergence；`exp048` 已通过 dmax/success/collision，仅 timeout `0.0137` 未过；`exp049`/`exp050` 分别说明全局增强 spacing/filter 和增强 hold/timeout shaping 都会退化；`exp051` 回到 exp048 reward/filter/control、只隔离 PPO 稳定性后达到 dmax `0.1836`、success `0.9883`、collision `0.0020`、timeout `0.0098`，是当前新环境栈 local reset 最好候选但仍未 strict；`exp052` 早退火到 `8192` 已完成并明显退化，说明 entropy taper 不宜过早；`exp053` 轻微提高全局 near reward 后 success/timeout 大幅退化，说明不能继续提高全局安全间距惩罚；`exp054` 收窄 PPO clip 到 `0.16` 后 dmax/collision 达标，但 success `0.7168`、timeout `0.2803` 明显退化；`exp055` 放宽 PPO clip 到 `0.20` 后 dmax/success/collision 达标，但 timeout `0.0146` 差于 exp051。clip 扫描表明 `0.18` 仍是当前最好点；`exp056`/`exp057` 的 terminal pairwise reward 均未改善 timeout，当前最好仍是 exp051。
- exp051 附近 checkpoint multi-seed 复验已完成：`012288/013312/014336` 在 `1023/2023/3023/4023` 四个 eval seed 上均未 strict，其中 `013312` 的 timeout 均值最低（`0.0134`）但 `timeout_zero_count=0/4`，说明当前 best 选点相对稳定，剩余瓶颈不是简单 checkpoint reselection。
- exp058 已完成：回到 exp051，只把 PPO `gamma` 从 `0.99` 提高到 `0.995`，不改 action/reward/filter/control。final eval dmax `0.1991`、collision `0.0020` 达标，但 success `0.7451`、timeout `0.2529` 明显失败；说明更长折扣 horizon 拖慢 terminal convergence，不能作为下一步主线。
- exp059 已完成：回到 exp051，只把 `gae_lambda` 从 `0.95` 降到 `0.90`，不改 action/reward/filter/control。best 仍回落到 `ppo_timestep_012288.pt`，final eval dmax `0.1927`、collision `0.0127` 达标，但 success `0.6904`、timeout `0.2988` 明显失败；说明更短 GAE trace 也不能修复 exp051 尾部 timeout，价值估计 horizon 方向暂时不作为主线。
- exp060 已完成：回到 exp051，只把 `value_loss_coef` 从 `0.50` 提高到 `0.75`，不改 action/reward/filter/control。best 为 `ppo_timestep_012288.pt`，final eval dmax `0.1837`、success `0.9736`、collision `0.0` 达标，但 timeout `0.0264` 失败且差于 exp051；说明更强 critic loss 权重没有清掉尾部 timeout，不能作为下一步主线。
- exp051 / exp060 success-gate 诊断已完成：exp051 recheck seed1023 的 `15` 个 timeout 中，`min_pairwise` final gate 失败 `15/15`，dmax gate 失败 `2/15`，dispersion/speed gate 均为 `0/15`；exp060 的 `19` 个 timeout 中，`min_pairwise` 失败 `18/19`。这说明尾部主要是“已经接近集合且低速，但最近邻间距没有通过 success gate”，暂时不应把 Actor 输出改成多点采样让 filter 选择。
- exp061/exp062 已完成观测/critic 可观测性诊断，均不改 action 输出、reward、filter 或 control。exp061 给 Actor/Critic 同时加入 terminal gate 特征后明显退化：final eval dmax `0.1890` 达标，但 success `0.8506`、collision `0.0205`、timeout `0.1289` 均失败；说明直接把 gate margin 暴露给 Actor 会诱发激进/饱和动作。exp062 保持 exp051 Actor 观测和 `branched_v1`，只给 critic state 加 `min_pairwise` 并用 `structured_v2`，final eval dmax `0.1832`、success `0.9736`、collision `0.0059` 达标，但 timeout `0.0205` 失败；4-seed checkpoint sweep 的 best `016384` timeout 均值 `0.0161`，不优于 exp051 的 `0.0134`。当前最好仍是 exp051。
- 生成结果写入 `outputs/runs/`，并由 git 忽略。

## 当前接口状态

- exp125 当前执行接口为 `ego_v8_decentralized_tiered`：Actor 输入 101 维，包含 10 维 ego、3×12 维通信缓存邻居、50 维局部地形和 5 维缓存聚合特征；Actor 架构为 `branched_v5`。12 m 内每步更新完整消息，12 m 外只低频保留位置和航向快照。该接口拒绝 86/89/92 维 checkpoint，且不读取 Oracle、集合槽位、全局质心或缓存外邻车状态。
- 历史默认 actor observation schema 为 `ego_v3_local_terrain_grid`，输入维度为 86；exp067–075 使用显式 `ego_v6_gather_slot_goal` 89 维接口。这些历史接口仅用于复现实验，不是 exp125 当前执行接口。`task.execution_slot_reward_target` 默认关闭；开启时仅让 dense oracle-progress reward 对齐已分配槽位，终止条件仍只看真实团队质心。`gather_point.robustness_radius>0` 只在 reset 搜索中要求一圈可能质心偏移均平整，不能替代运行时的实际质心 gate。
- 地形网格通道为相对高度和风险，覆盖前后 `[-0.4, 1.2] m`、横向 `[-0.8, 0.8] m`；critic 仍为 54 维，并使用 5 维网格摘要。地图面积可通过 `world_xy_limit/crater_field_size` 扩大，但本轮不扩大 Actor 局部地形观测窗口。
- checkpoint 加载要求 schema、actor 输入维度和 critic 状态维度完全匹配；新 checkpoint metadata 还记录 Actor/Critic 架构、运动学模型和轨迹生成方法。`--init-checkpoint` 只初始化兼容模型参数、不会恢复 optimizer 或 rollout memory，并保存可筛选的 `ppo_timestep_000000.pt` 基线。旧 `ego_v2_speed_angular` checkpoint 不自动迁移；当前 schema 但缺少架构 metadata 的旧 checkpoint 只按 `mlp_v1` 兼容路径加载。
- centralized critic state 和 reward shaping 可以使用 oracle 信息；默认 Actor 不接收 `p*`、oracle 距离或 oracle 距离下降量。显式 v5/v6 执行契约只传递从其导出的车体系局部目标三元组，Critic 仍为 54 维。
- 动作接口固定为低维 `[rho, beta]`，再经局部子目标、可配置 `line/quintic` 轨迹和简化速度控制器转换为运动命令。
- 当前 proxy 动力学可配置为 `unicycle` 或 `bicycle`；旧配置默认 `unicycle`，`exp042` 显式使用 `bicycle`。二者都没有质量、惯量、轮地接触、打滑、悬挂或 PhysX contact。
- `scripts/train_skrl_mappo.py` 支持历史 `mlp_v1`、`branched_v1` 至 `branched_v4`，以及严格去中心化101维接口的 `branched_v5` 和实验性 `branched_v6_graph_attention`；Critic支持 `mlp_v1|structured_v1|structured_v2`。`branched_v6_graph_attention` 已被exp137否决，只为复现实验保留，不是当前采用结构。
- `scripts/train_skrl_mappo.py` 使用 SKRL MAPPO 训练 proxy wrapper；`isaaclab-multi-agent` wrapper 只是接口层，不代表训练 loop 运行在 Isaac Sim / PhysX。
- exp016 已启用项目侧 `shared_joint` 更新：共享 Actor/Critic 只使用一个 optimizer，每个 rollout 合并四个 rover 的 Actor 样本并只更新一次 Critic。
- 当前 exp016 诊断配置把通信半径临时扩大到 `12 m`；这是训练诊断设置，不是最终通信约束。
- exp017 已完成 pure RL 连续 20M 长跑并通过 seed23 独立 strict eval；这是固定地图、单 seed proxy 结果，不代表随机地图泛化或多 seed 收敛。
- exp018 已加入每环境、每 episode reset 独立地形随机化，并把地形强度提高一档；完整测试、CPU/CUDA smoke 和随机地图渲染已通过。seed23 连续 20M 已完成，dmax 和 success 达标，但 collision / timeout 未通过 strict gate。
- exp019 已在 exp018 基础上完成两个诊断改造：success gate 新增最近邻安全间距 `0.42 m`，terrain reward 扩展到当前点到子目标的路径级风险。seed23 20M 工程链路和 5 轮独立 eval/GIF 已完成，但 strict gate 未通过。
- exp020 已在 exp019 基础上加入 terrain/safety-aware 子目标过滤器；过滤器稳定降低路径风险，但显著抑制集合进度。seed23 20M、5 轮独立 eval/GIF 和训练曲线已完成，strict gate 未通过。
- exp021 已完成 exp020 的课程化/软化 filter 迭代：前期保留 raw action，后期逐步增加 filter 介入概率和 score 权重，并加入 raw-risk / filter-deviation 辅助惩罚。seed23 20M、5 轮独立 eval、GIF、height map 和训练曲线已完成，strict gate 未通过。
- exp022 已完成 endpoint/path safety constrained curriculum filter 迭代：collision 被压到 strict 内，但集合进度塌缩，5 seed mean success `0.0139`、timeout `0.9699`，strict 未通过。
- exp023 已完成 soft progress-preserving filter 迭代：success 从 exp022 的 `0.0139` 回升到 `0.3027`，但 collision `0.2295`、timeout `0.4717`，strict 未通过；失败原因是 static endpoint/path safety 未预测可见邻居同步运动。
- exp024 已完成 mutual path safety filter 迭代：在 exp023 基础上把可见邻居 raw subgoal path 作为动态障碍，按相同时间采样比较候选路径；post-hoc 使用 `success_progress_long` 重选 `ppo_timestep_010240.pt` 为 best，seed1023 final eval 为 dmax ratio `0.1397`、success `0.8398`、collision `0.0674`、timeout `0.0947`，strict 未通过但显著优于 exp023。
- exp025 已完成 dense mutual path safety filter 迭代：基于 exp024 加密 mutual/path safety 采样到 9 点，并适度提高 path/mutual collision 权重；best `ppo_timestep_009216.pt` final eval 为 dmax ratio `0.1434`、success `0.8525`、collision `0.0449`、timeout `0.1035`，strict 未通过。相对 exp024 collision 降低，但 timeout 未改善。
- exp026 已完成 hold-zone filter 诊断：过早/过宽的 `hold_zone_rho/spacing` cost 把 success 从 exp025 的 `0.8525` 拉低到 `0.7529`，collision `0.0615`、timeout `0.1865`，strict 未通过。
- exp027 已完成 strict hold-zone filter 诊断：把 hold-zone activation 收窄到真正 success dmax/dispersion 附近后避免了 exp026 的明显退化，但 final eval success `0.8418`、collision `0.0498`、timeout `0.1123`，未优于 exp025。
- exp028 已完成 hold reward 诊断：回退到 exp025 dense mutual filter，只强化 `success_hold_step=4.0`、`success_bonus=45`、`timeout_penalty=18`；final eval success `0.8691`、collision `0.0469`、timeout `0.0889`，是 exp026–029 中最好但仍未 strict。
- exp029 已完成 hold reward + stronger safety 诊断：在 exp028 基础上加强 path/mutual collision filter 和终端碰撞惩罚，final eval success `0.8262`、collision `0.0557`、timeout `0.1221`，说明继续加安全权重会牺牲成功并未压低真实碰撞。
- exp030 已完成低层 control safety projection 诊断：回到 exp028 主体，只在 `compute_control()` 后、`_integrate()` 前加入相对速度安全投影和 success-zone damping；final eval success `0.8330`、collision `0.0313`、timeout `0.1357`，collision 明显低于 exp028，但投影过强导致 success/timeout 退化。
- exp031–exp034 已完成 control safety 投影条件迭代：简单调弱、closing-only、directional scale 和 directional mask 都未 strict；其中 exp034 的 mask 版本把 success 拉回 `0.8828`、timeout 降到 `0.0840`，但 collision `0.0361` 仍失败。
- exp035–exp036 已完成 directional mask buffer 与 stronger hold/timeout shaping：exp035 首次让 success `0.9072` 和 collision `0.0127` 同时达标；exp036 进一步到 success `0.9336`、collision `0.0088`、timeout `0.0586`，剩余瓶颈转为 timeout/hold。
- exp037 已完成 260-step episode/eval 诊断：timeout 从 exp036 的 `0.0586` 降到 `0.0410`，但 collision 反弹到 `0.0352`，说明单纯延长 episode 会暴露末段碰撞。
- exp038 已完成 success-zone stabilizer + 320-step episode/eval：修正 best 后 final eval success `0.9756`、collision `0.0137`、timeout `0.0107`；旧环境栈随机地形阶段性最佳候选，strict 只剩 timeout gate 失败。
- exp039/exp040 是基于 exp038 best 的诊断复评，不建议长训：hard near stabilizer 和 stronger soft hold stabilizer 都使 timeout 或 collision 差于 exp038。
- exp041 已完成 hold-zone override 诊断与 CPU/CUDA smoke：在 exp038 best 上复评得到 success `0.9795`、collision `0.0107`、timeout `0.0098`，略优于 exp038，但当前已暂停长训，暂不启动 exp041。
- exp042 已完成环境工程探针：`branched_v1` Actor、`structured_v1` Critic、`bicycle` proxy、`quintic` 轨迹生成、`25 m × 25 m` 地图和 `communication_radius=0.0` 无限可见邻居语义在 CPU `8 env / 8 timesteps` 与 CUDA `256 env / 64 timesteps` smoke 中通过；CUDA smoke 显示一个 optimizer、两次 joint update、terrain branch 权重更新 `0.1263`、动作非退化。
- exp043 已完成直接长训：基于 exp042 新环境栈，迁移 exp041 的 hold-zone override，加入可配置 initial-state 分布并扩大初始队形采样，terrain crater density 提高到 `crater_count=48`，seed23 连续 `20480` timesteps / `41,943,040` env steps。训练链路正常、参数和 terrain branch 均更新，但 final eval 为 dmax ratio `0.8596`、success `0.0`、collision `0.0`、timeout `1.0`，strict 未通过。
- exp044 已完成：保留 `branched_v1/structured_v1`、`bicycle`、`quintic`、`25 m × 25 m` 地图和无限通信语义，并加入 `3.0–4.0 m -> 3.8–5.2 m` initial-state curriculum。final eval dmax ratio `0.4796`、success `0.0`、collision `0.00195`、timeout `0.9980`，strict 未通过；相比 exp043 明显靠拢但仍未进入 success basin。
- exp045 已完成：保持新环境栈和 25m 地图，但把目标 reset 分布缩小到 `2.4–3.4 m`，课程起点为 `1.6–2.4 m`，同时放大 `rho/beta` 可达范围、增强 gather progress、临时降低 terrain/filter 干扰。final eval dmax ratio `0.2734`、success `0.1846`、collision `0.0`、timeout `0.8174`，说明 local-success bootstrap 有效但仍未收敛。
- exp046 已完成：沿用 exp045 的 local reset 分布，但降低 filter/control-safety 的末端介入强度，增强 dmax/dispersion progress、success bonus 和 timeout penalty。final eval dmax ratio `0.2424`、success `0.6123`、collision `0.0`、timeout `0.3877`，strict 未通过，但证明新环境栈已进入 local success basin。
- exp047 已完成：保持 exp046 reset 分布，进一步释放 terminal safety/filter/control damping，同时增强 dmax/dispersion/timeout/success shaping。final eval dmax ratio `0.2132`、success `0.7188`、collision `0.0059`、timeout `0.2764`，strict 未通过，但曾是新环境栈 local reset 阶段性最好结果。
- exp048 已完成：在 exp047 附近小步提高 terminal drive、dispersion 收缩和 timeout shaping。final eval dmax ratio `0.1866`、success `0.9844`、collision `0.0020`，均通过 strict；唯一失败是 timeout `0.0137`。
- exp049 已完成：针对 exp048 剩余最近邻安全间距灰区增强 terminal spacing。final eval dmax ratio `0.1884`、success `0.8926`、collision `0.0010`、timeout `0.1064`，strict 未通过且明显差于 exp048；说明 spacing/filter/control safety 介入过强，修复了部分间距但牺牲了成功保持。
- exp050 已完成：回到 exp048 主体，不改 action 输出、不做多点采样、不增强低层控制规划能力；主要调整 terminal hold reward、timeout shaping、PPO 学习率/clip/探索噪声。final eval dmax ratio `0.1847`、success `0.9590`、collision `0.0059` 达标，但 timeout `0.0352` 未过且差于 exp048，说明该 RL 配置微调方向不能作为下一步主线。
- exp051 已完成：reward、filter、control safety 全部回到 exp048，只隔离 PPO 稳定性调整（学习率、clip、entropy schedule、initial log std）。best 为 `ppo_timestep_013312.pt`，final eval dmax ratio `0.1836`、success `0.9883`、collision `0.0020` 均通过，但 timeout `0.0098` 仍失败；相对 exp048 小幅降低 timeout，相对 exp050 明显恢复 success/timeout。
- exp051 checkpoint seed sweep 已完成：对 `012288/013312/014336` 做 `4` 个 eval seed 复验后，013312 仍是附近 timeout 均值最低的 checkpoint，但 `strict_pass_count=0/4`、`timeout_zero_count=0/4`，说明剩余问题不是简单 checkpoint reselection。
- exp052 已完成：以 exp051 为基线，只把 entropy schedule 从 `12288` 提前到 `8192`，不改 action 输出、不新增多点采样、不改 reward/filter/control。best 为 `ppo_timestep_008192.pt`，final eval dmax ratio `0.1863`、collision `0.0059` 达标，但 success `0.8955` 和 timeout `0.0986` 失败，明显差于 exp051。
- exp053 已完成：回到 exp051，只把 reward 中已有的 `near_distance` 安全惩罚系数从 `2.4` 小幅提高到 `2.8`，不改 action 输出、不新增多点采样、不改 filter/control/PPO schedule。best 为 `ppo_timestep_020480.pt`，final eval dmax ratio `0.2049`、success `0.6416`、collision `0.0039`、timeout `0.3545`，明显差于 exp051。
- exp054 已完成：回到 exp051，只把 PPO `clip_epsilon` 从 `0.18` 收窄到 `0.16`，不改 action 输出、不新增多点采样、不改 reward/filter/control。best 为 `ppo_timestep_017408.pt`，final eval dmax ratio `0.1972`、collision `0.0029` 达标，但 success `0.7168` 和 timeout `0.2803` 明显失败；说明更保守 clip 过度抑制 policy update，不能作为下一步主线。
- exp055 已完成：回到 exp051，只把 PPO `clip_epsilon` 从 `0.18` 放宽到 `0.20`，不改 action 输出、不新增多点采样、不改 reward/filter/control。best 为 `ppo_timestep_017408.pt`，final eval dmax ratio `0.1850`、success `0.9824`、collision `0.0029` 达标，但 timeout `0.0146` 仍失败且差于 exp051；说明放宽 clip 没有清掉尾部 timeout。
- exp056 已完成：回到 exp051，不改 action 输出、不新增多点采样、不改 filter/control/PPO，只新增 reward 侧 `terminal_pairwise_gap=4.0`，并在 dmax/dispersion 接近成功区时惩罚 `nearest < min_pairwise_distance` 的 gap。best 为 `ppo_timestep_012288.pt`，final eval dmax ratio `0.1864`、success `0.9873`、collision `0.0010` 达标，但 timeout `0.0117` 仍失败且差于 exp051；说明该项方向有轻微信号但触发/强度仍不理想。
- exp057 已完成：回到 exp051，不改 action 输出、不新增多点采样、不改 filter/control/PPO，只把 terminal pairwise reward 收窄为 `terminal_pairwise_gap=2.0` 且 dmax/dispersion multiplier `1.00/1.00`。best 为 `ppo_timestep_011264.pt`，final eval dmax ratio `0.1850`、success `0.9697`、collision `0.0059` 达标，但 timeout `0.0254` 失败且明显差于 exp051/exp056；说明即使严格触发，terminal pairwise reward 仍会扰动末端 hold。
- exp058 已完成：回到 exp051，不改 action 输出、不新增多点采样、不改 reward/filter/control，只把 PPO `gamma` 从 `0.99` 提高到 `0.995`。best 为 `ppo_timestep_012288.pt`，final eval dmax ratio `0.1991`、collision `0.0020` 达标，但 success `0.7451`、timeout `0.2529` 明显失败；说明更长折扣 horizon 没有改善尾部 timeout，反而拖慢 terminal convergence。
- exp059 已完成：回到 exp051，不改 action 输出、不新增多点采样、不改 reward/filter/control，只把 PPO `gae_lambda` 从 `0.95` 降到 `0.90`。best 为 `ppo_timestep_012288.pt`，final eval dmax ratio `0.1927`、collision `0.0127` 达标，但 success `0.6904`、timeout `0.2988` 明显失败；训练后期 success 还坍缩到约 `0.0122`，说明更短 advantage trace 不适合作为下一步主线。
- exp060 已完成：回到 exp051，不改 action 输出、不新增多点采样、不改 reward/filter/control，只把 PPO `value_loss_coef` 从 `0.50` 提高到 `0.75`。best 为 `ppo_timestep_012288.pt`，final eval dmax ratio `0.1837`、success `0.9736`、collision `0.0` 达标，但 timeout `0.0264` 失败且差于 exp051；说明更强 critic loss 权重没有改善尾部 hold。
- exp061 已完成：回到 exp051，不改 action 输出、不新增多点采样、不改 reward/filter/control，只把 Actor observation 切到 `ego_v4_terminal_gate` 并使用 `branched_v2/structured_v2`，显式加入 dmax/dispersion/speed/hold/pairwise gate 特征。best 为 `ppo_timestep_020480.pt`，final eval dmax ratio `0.1890` 达标，但 success `0.8506`、collision `0.0205`、timeout `0.1289` 失败；gate 诊断中 timeout 仍主要卡 `min_pairwise`，且动作饱和显著上升，不能作为下一步主线。
- exp062 已完成：回到 exp051，Actor observation 和 `branched_v1` 保持不变，只给 centralized critic state 加 terminal `min_pairwise` 并使用 `structured_v2`。best 为 `ppo_timestep_016384.pt`，final eval dmax ratio `0.1832`、success `0.9736`、collision `0.0059` 达标，但 timeout `0.0205` 失败；3-checkpoint/4-seed sweep 中 `016384` timeout mean `0.0161`、`0/4` strict，不优于 exp051。

## Checkpoint 评估工作流

新增标准入口：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/<experiment>/<run_id>
```

该入口会根据 experiment YAML 的 `evaluation:` 配置执行：

1. proxy 独立评估，写入 `metrics/final_eval_proxy.json`；
2. proxy strict gate 判定；
3. 若配置允许且 proxy 通过，再低频触发 PhysX / Jackal tracking validation；
4. 写入 `metrics/checkpoint_status.json` 并更新 `run_manifest.json`。

checkpoint 状态只使用：

```text
candidate
proxy_passed
physx_evaluated
physx_passed
final_selected
```

新增 checkpoint seed sweep 诊断入口，用于在不改变 policy/filter/control 的情况下比较多个 checkpoint 的 eval seed 稳定性：

```bash
.venv_isaaclab/bin/python scripts/evaluate_proxy_checkpoint_seed_sweep.py \
  --config outputs/runs/<experiment>/<run_id>/config/experiment.yaml \
  --run-dir outputs/runs/<experiment>/<run_id> \
  --checkpoint ppo_timestep_012288.pt \
  --checkpoint ppo_timestep_013312.pt \
  --seeds 1023,2023,3023,4023 \
  --device cuda \
  --num-envs 1024 \
  --steps 320
```

该入口写入 `metrics/checkpoint_seed_sweep/summary.json` 和逐 seed eval JSON；它只用于诊断 checkpoint selection / eval 方差，不替代 strict gate。

新增 success-gate 诊断入口，用于逐 episode 记录 timeout 末端到底卡在哪个 success gate：

```bash
.venv_isaaclab/bin/python scripts/diagnose_proxy_success_gates.py \
  --config outputs/runs/<experiment>/<run_id>/config/experiment.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --num-envs 1024 \
  --steps 320 \
  --seed 1023 \
  --run-dir outputs/runs/<experiment>/<run_id>
```

该入口写入 `metrics/success_gate_diagnostics.json`，只用于 failure analysis；若与历史 `final_eval_proxy.json` 数字略有不同，应表述为 recheck/diagnostic，不替换原 strict eval 记录。

## 已验证结果

| 实验 | 地形 | 方法 | 严格状态 | 当前解释 |
| --- | --- | --- | --- | --- |
| exp006 | 平地 proxy | BC + PPO | 通过 | 平地 proxy strict baseline，checkpoint 来自 PPO 阶段。 |
| exp008 | 弱 lunar crater 3D proxy | 弱 warm-start + PPO | 3 seeds 通过 | 当前最完整的 terrain-aware proxy baseline。 |
| exp009 | 强 lunar crater 3D proxy | 弱 warm-start + PPO | 未通过 | seed23 通过；seed31 失败；近期不继续堆 long-budget PPO。 |
| exp010 | 强 lunar crater 3D proxy | hold reward / safety 诊断 | 未通过 | success 可改善，但 collision/timeout gate 仍失败。 |
| exp012 | proxy SKRL-MAPPO CUDA 诊断 | action scale warmup probe | 未通过 | distance 有改善，但 strict gate 未通过。 |
| exp013 | proxy SKRL-MAPPO CUDA 诊断 | action scale ablation + teacher reachability | 未通过 | 当前小动作 100-step 配置对 teacher 也几乎不可达。 |
| exp014 | 弱 lunar crater proxy | 5×5 局部地形网格 CUDA probe | 工程验证通过；未做 strict | 新观测和训练链路有效，不能表述为策略收敛。 |
| exp015 | 偏弱中档 lunar crater proxy | SKRL MAPPO + BC20 | 2M screen 未通过 | 工程信号正常；dmax ratio 0.818、success 0、collision 0.124、timeout 0.876，因此未启动 8M。 |
| exp016 | 偏弱中档 lunar crater proxy | shared-joint MAPPO + local BC100 + comm12 | BC probe 未通过 | shared update 探针通过；BC-only dmax ratio 0.438、collision 0.0088、timeout 0.991，未启动 2M。 |
| exp017 | 固定偏弱中档 lunar crater proxy | shared-joint MAPPO pure RL + comm12 | seed23 strict 通过 | final dmax ratio 0.1318、success 0.9990、collision 0.00098、timeout 0；仍是 single-seed candidate。 |
| exp018 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + comm12 | 未通过 | seed23 20M 完成；final dmax ratio 0.1417、success 0.9609 通过，但 collision 0.0352、timeout 0.0088 未达 strict gate。 |
| exp019 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + safe success gate + path terrain risk | 未通过 | seed23 20M 完成；10240 checkpoint 有集合趋势但 collision 高，当前 best final eval success 0.0195、collision 0.0791、timeout 0.9023；5 seed 复验均值 success 0.0143、collision 0.0801、timeout 0.9082。 |
| exp020 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + terrain/safety subgoal filter | 未通过 | seed23 20M 完成；filter 将 5 seed path risk mean 从 raw 0.3815 降到 0.3187，但 success 0、collision 0.0498、timeout 0.9506，说明过滤器过强地牺牲了集合进度。 |
| exp021 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + curriculum terrain/safety subgoal filter | 未通过 | 课程化 filter 恢复集合进度：5 seed mean success 0.6361、dmax ratio 0.1460、timeout 0.1967，filtered path risk 0.3638；但 collision 0.1746，远高于 strict 0.02。 |
| exp022 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + endpoint/path safety constrained curriculum filter | 未通过 | 5 seed mean：dmax ratio 0.4719、success 0.0139、collision 0.0170、timeout 0.9699；说明 constrained filter 可压住碰撞，但过强地牺牲集合进度。 |
| exp023 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + soft progress-preserving subgoal filter | 未通过 | seed23 final eval：dmax ratio 0.1789、success 0.3027、collision 0.2295、timeout 0.4717；缓解 exp022 standoff，但 static filter 未处理同步运动碰撞。 |
| exp024 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + mutual path safety subgoal filter | 未通过 | post-hoc best `10240`：dmax ratio 0.1397、success 0.8398、collision 0.0674、timeout 0.0947；mutual path filter 明显改善 success/collision 平衡，但 strict 安全和 timeout 仍未达标。 |
| exp025 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + dense mutual path safety filter | 未通过 | best `9216`：dmax ratio 0.1434、success 0.8525、collision 0.0449、timeout 0.1035；dense mutual filter 继续降低碰撞，但仍未达到 strict 安全/timeout gate。 |
| exp026 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + hold-stable subgoal filter | 未通过 | best final eval：success 0.7529、collision 0.0615、timeout 0.1865；hold-zone 介入过早，明显压制集合进度。 |
| exp027 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + strict hold-zone filter | 未通过 | best final eval：success 0.8418、collision 0.0498、timeout 0.1123；严格触发避免 exp026 退化，但仍不优于 exp025。 |
| exp028 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + dense mutual filter + stronger hold reward | 未通过 | best final eval：success 0.8691、collision 0.0469、timeout 0.0889；当前随机地形安全/hold 方向最好结果，但安全 gate 仍失败。 |
| exp029 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + exp028 + stronger safety penalties/filter weights | 未通过 | best final eval：success 0.8262、collision 0.0557、timeout 0.1221；加强安全权重反而退化，不能作为下一步方向。 |
| exp030 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + exp028 + low-level control safety projection | 未通过 | best final eval：success 0.8330、collision 0.0313、timeout 0.1357；动态控制投影能降低碰撞，但当前触发过强，牺牲 success/timeout。 |
| exp031 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + narrow/weak control safety projection | 未通过 | best final eval：success 0.8105、collision 0.0449、timeout 0.1455；简单调弱没有恢复 success，也丢失 exp030 的安全收益。 |
| exp032 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + closing-only control safety projection | 未通过 | best final eval：success 0.8379、collision 0.0361、timeout 0.1279；closing-only 略优于 exp031，但仍未达标。 |
| exp033 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + directional agent-scale projection | 未通过 | best final eval：success 0.8154、collision 0.0488、timeout 0.1387；方向性连续缩放没有带来安全收益。 |
| exp034 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + directional mask projection | 未通过 | best final eval：success 0.8828、collision 0.0361、timeout 0.0840；mask 版本恢复部分 success/timeout，但 collision 仍超 strict。 |
| exp035 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + directional mask buffer | 未通过 | best final eval：success 0.9072、collision 0.0127、timeout 0.0811；success/collision 同时达标，timeout 成主瓶颈。 |
| exp036 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + directional mask + stronger hold/timeout shaping | 未通过 | best final eval：success 0.9336、collision 0.0088、timeout 0.0586；继续改善 timeout，但 strict 仍失败。 |
| exp037 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + directional mask + 260-step episode/eval | 未通过 | best final eval：success 0.9238、collision 0.0352、timeout 0.0410；延长 episode 降 timeout，但 collision 反弹。 |
| exp038 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + success-zone stabilizer + 320-step episode/eval | 未通过 | 修正 best 后 final eval：success 0.9756、collision 0.0137、timeout 0.0107；旧环境栈随机地形阶段性最佳，strict 只剩 timeout 失败。 |
| exp039 | 随机增强 lunar crater proxy | exp038 checkpoint + hard near stabilizer 诊断 | 未长训 | 复评 success 0.9424、collision 0.0254、timeout 0.0322，差于 exp038；不建议按原样长训。 |
| exp040 | 随机增强 lunar crater proxy | exp038 checkpoint + stronger soft hold stabilizer 诊断 | 未长训 | 复评 success 0.9658、collision 0.0186、timeout 0.0166，timeout 差于 exp038；不建议按原样长训。 |
| exp041 | 随机增强 lunar crater proxy | exp038 checkpoint + hold-zone override 诊断 | 暂停长训 | 复评 success 0.9795、collision 0.0107、timeout 0.0098，略优于 exp038；当前不启动长训。 |
| exp042 | 随机增强 lunar crater proxy | 结构化 Actor/Critic + bicycle proxy + quintic trajectory + 25m 地图 + 无限通信工程探针 | 工程 smoke 通过 | CPU `8/8` 与 CUDA `256/64` smoke 通过；只验证环境链路，不代表策略收敛。 |
| exp043 | 随机增强 lunar crater proxy | exp042 新环境栈 + exp041 hold override + 扩大 initial-state 分布 | 未通过 | seed23 40M 完成；工程链路正常但策略几乎不集合，final eval success `0.0`、timeout `1.0`。 |
| exp044 | 随机增强 lunar crater proxy | exp043 新环境栈 + initial-state curriculum | 未通过 | seed23 40M 完成；dmax 从 exp043 明显改善到 `0.4796`，但 success `0.0`、timeout `0.9980`。 |
| exp045 | 随机增强 lunar crater proxy | exp044 新环境栈 + local-success bootstrap | 未通过 | seed23 40M 完成；success `0.1846`、collision `0.0`，证明 local bootstrap 有效但 timeout/dmax/dispersion 仍失败。 |
| exp046 | 随机增强 lunar crater proxy | exp045 local reset + terminal hold release | 未通过 | final eval success `0.6123`、collision `0.0`，但 dmax ratio `0.2424`、timeout `0.3877` 仍失败；local terminal release 有效但不足。 |
| exp047 | 随机增强 lunar crater proxy | exp046 local reset + terminal convergence release | 未通过 | final eval success `0.7188`、collision `0.0059`、dmax ratio `0.2132`，但 timeout `0.2764` 仍失败；曾是新环境栈 local reset 阶段性最好结果。 |
| exp048 | 随机增强 lunar crater proxy | exp047 local reset + terminal drive / dispersion tightening | 未通过 | dmax ratio `0.1866`、success `0.9844`、collision `0.0020` 均通过；唯一失败为 timeout `0.0137`，此前新环境栈 local reset 最佳。 |
| exp049 | 随机增强 lunar crater proxy | exp048 local reset + terminal spacing timeout closure | 未通过 | final eval dmax ratio `0.1884`、success `0.8926`、collision `0.0010`、timeout `0.1064`；过强 spacing 修正降低成功并抬高 timeout，不优于 exp048。 |
| exp050 | 随机增强 lunar crater proxy | exp048 local reset + 克制 filter/control + terminal hold RL tune | 未通过 | final eval dmax ratio `0.1847`、success `0.9590`、collision `0.0059` 达标，但 timeout `0.0352` 差于 exp048；不作为主结果。 |
| exp051 | 随机增强 lunar crater proxy | exp048 local reset + PPO stability only | 未通过 | best `013312`：dmax `0.1836`、success `0.9883`、collision `0.0020` 均通过，timeout `0.0098` 仍失败；4-seed 复验下 013312 仍是附近最好选点但 `0/4` strict。 |
| exp052 | 随机增强 lunar crater proxy | exp051 + earlier entropy taper | 未通过 | best `008192`：dmax `0.1863`、collision `0.0059` 达标，但 success `0.8955`、timeout `0.0986` 失败；过早收窄探索明显差于 exp051。 |
| exp053 | 随机增强 lunar crater proxy | exp051 + mild near reward | 未通过 | best `020480`：dmax `0.2049`、success `0.6416`、collision `0.0039`、timeout `0.3545`；全局 near reward 小幅增强也会推散队形，明显差于 exp051。 |
| exp054 | 随机增强 lunar crater proxy | exp051 + PPO clip 0.16 | 未通过 | best `017408`：dmax `0.1972`、collision `0.0029` 达标，但 success `0.7168`、timeout `0.2803` 明显失败；clip 过窄，不优于 exp051。 |
| exp055 | 随机增强 lunar crater proxy | exp051 + PPO clip 0.20 | 未通过 | best `017408`：dmax `0.1850`、success `0.9824`、collision `0.0029` 达标，但 timeout `0.0146` 失败且差于 exp051；clip `0.20` 不优于当前 best。 |
| exp056 | 随机增强 lunar crater proxy | exp051 + terminal pairwise reward | 未通过 | best `012288`：dmax `0.1864`、success `0.9873`、collision `0.0010` 达标，但 timeout `0.0117` 失败且差于 exp051；pairwise reward 过早/偏强。 |
| exp057 | 随机增强 lunar crater proxy | exp051 + strict terminal pairwise reward | 未通过 | best `011264`：dmax `0.1850`、success `0.9697`、collision `0.0059` 达标，但 timeout `0.0254` 明显差于 exp051；不继续该方向。 |
| exp058 | 随机增强 lunar crater proxy | exp051 + PPO gamma 0.995 | 未通过 | best `012288`：dmax `0.1991`、collision `0.0020` 达标，但 success `0.7451`、timeout `0.2529` 明显失败；更长折扣 horizon 拖慢 terminal convergence。 |
| exp059 | 随机增强 lunar crater proxy | exp051 + PPO GAE 0.90 | 未通过 | best `012288`：dmax `0.1927`、collision `0.0127` 达标，但 success `0.6904`、timeout `0.2988` 明显失败；更短 GAE trace 也会破坏 terminal convergence。 |
| exp060 | 随机增强 lunar crater proxy | exp051 + PPO value loss 0.75 | 未通过 | best `012288`：dmax `0.1837`、success `0.9736`、collision `0.0` 达标，但 timeout `0.0264` 失败且差于 exp051；更强 critic loss 权重没有改善尾部 hold。 |
| exp061 | 随机增强 lunar crater proxy | exp051 + terminal gate Actor/Critic observation | 未通过 | best `020480`：dmax `0.1890` 达标，但 success `0.8506`、collision `0.0205`、timeout `0.1289` 失败；直接暴露 gate margin 给 Actor 导致策略更激进，不作为主线。 |
| exp062 | 随机增强 lunar crater proxy | exp051 + critic-only min_pairwise state | 未通过 | best `016384`：dmax `0.1832`、success `0.9736`、collision `0.0059` 达标，但 timeout `0.0205` 失败；4-seed sweep timeout mean `0.0161`，不优于 exp051。 |

历史完整 suite checkpoint：

```text
outputs/runs/exp_008_terrain3d/_suite/checkpoints/
```

这些 exp008 checkpoint 使用旧 observation schema，历史 strict 结论仍有效，但不能直接加载到当前 86 维 Actor。exp017 已产生当前 schema 的单 seed strict checkpoint；exp014 checkpoint 仍仅用于工程探针。

## 结果解释边界

- exp006 / exp008 是 proxy strict pass，不是 Isaac Lab 物理训练 pass。
- exp006 / exp008 的 checkpoint 属于旧 observation schema；结果可作为历史 baseline，但不能与新 Actor 接口直接混用。
- exp014 只通过有限值、参数更新、地形输入权重更新和动作非退化检查，不是 strict convergence pass。
- PhysX / Jackal 结果应写成“Jackal 在 PhysX 场景中的轨迹跟踪验证结果”或“proxy checkpoint 的高保真迁移 sanity check”，不能写成“物理环境训练结果”。
- exp017 可以表述为“固定地图、seed23、proxy pure RL 从零通过 strict gate”，不能扩展为多 seed、随机地图或 PhysX 收敛。
- exp018 可以表述为“随机增强地形下已获得稳定集合趋势和较高 success，但安全/超时 gate 未完全收敛”，不能写成随机地图 strict pass。
- exp019 可以表述为“成功区安全间距和路径级地形风险链路已接入并可训练/评估，但当前 reward 下策略仍在成功率、碰撞率和超时之间失衡”，不能写成安全地形策略收敛。
- exp020 可以表述为“子目标过滤器确实降低了路径风险，但当前 hard post-processing 过强，导致探索/集合进度塌缩”，不能写成地形规避策略成功。
- exp021 可以表述为“课程化 filter 恢复了集合趋势，但碰撞率显著过高”，不能写成随机地形 strict pass 或安全策略成功。
- exp022 可以表述为“endpoint/path safety constrained filter 把 collision 压到 strict 内，但 success/timeout 严重失败”，不能写成随机地形安全策略收敛。
- exp023 可以表述为“soft progress filter 缓解了 exp022 的 standoff，但 collision/timeout 仍严重失败”，不能写成随机地形安全策略改善或收敛。
- exp024 可以表述为“mutual path safety 明显改善 exp023 的动态路径冲突，但 collision/timeout 仍未过 strict”，不能写成随机地形安全策略收敛。
- exp025 可以表述为“dense mutual path safety 相对 exp024 进一步降低 collision，但仍未解决末段 hold / timeout 稳定性”，不能写成随机地形安全策略收敛。
- exp026/exp027 可以表述为“hold-zone filter 诊断未改善 exp025，过早介入会压制集合”，不能写成 hold 稳定成功。
- exp028 可以表述为“强化 success hold reward 提高了 success/timeout，是 exp026–029 中最好的随机地形结果”，但 collision 仍超 strict，不能写成安全收敛。
- exp029 可以表述为“继续加强安全权重没有降低真实 collision，反而牺牲 success/timeout”，不能写成安全改善。
- exp030 可以表述为“低层动态控制投影降低 collision 但牺牲 success/timeout”，不能写成安全收敛或 strict 改善。
- exp031–exp034 可以表述为“控制层投影条件和方向性 mask 诊断”，不能写成随机地形安全收敛；exp034 是方向性 mask 的有效拐点，但 collision 仍失败。
- exp035/exp036 可以表述为“success/collision 已同时过门槛但 timeout 仍失败”，不能写成 strict pass。
- exp037 可以表述为“延长 episode 降 timeout 但导致 collision 反弹”，不能写成单纯时间预算不足。
- exp038 可以表述为“旧环境栈随机地形阶段性最佳候选，strict 只剩 timeout 尾部未过”，不能写成 strict pass。
- exp039/exp040 只是 exp038 checkpoint 复评诊断，不能写成长训练结果。
- exp041 可以表述为“hold-zone override 在 exp038 checkpoint 上略有改善”，不能写成 exp041 长训练完成。
- exp042 可以表述为“训练环境三项核心改造和 25m/无限通信设置的工程探针通过”，不能写成策略训练收敛或 strict pass。
- exp043 可以表述为“新环境栈直接长训未收敛，主要表现为集合进度不足而非碰撞或数值异常”，不能写成 strict pass。
- exp044 可以表述为“initial-state curriculum 改善了靠拢但仍未产生 success”，不能写成 strict pass。
- exp045 可以表述为“local-success bootstrap 把 success 从 0 提升到 0.1846，但仍未收敛”，不能写成 strict pass。
- exp046/exp047 可以表述为“新环境栈 local reset 下逐步恢复 terminal convergence”，不能写成完整难度 strict 收敛。
- exp048/exp051 可以表述为“dmax/success/collision 已过，剩余 timeout 尾部未清零”，不能写成 strict pass。
- exp049/exp050 可以表述为“增强 spacing/filter 或增强 hold/timeout shaping 的负结果”，不能作为下一步主方向。
- exp052/exp054 可以表述为“PPO 探索或更新过早/过强收窄会明显降低 success 并抬高 timeout”，不能作为下一步主方向。
- exp055 可以表述为“稍微放宽 PPO clip 能恢复 exp051 附近的 success/collision，但没有改善 timeout 尾部”，不能作为当前主结果。
- exp056/exp057 可以表述为“terminal pairwise reward 能轻微影响最近邻/碰撞，但没有改善 timeout，且严格弱化后仍扰动 success/hold”，不能作为下一步主方向。
- exp058 可以表述为“提高 PPO gamma 到 0.995 明显拖慢 terminal convergence”，不能作为下一步主方向。
- exp059 可以表述为“降低 GAE lambda 到 0.90 明显降低 success 并抬高 timeout，且训练后期策略质量坍缩”，不能作为下一步主方向。
- exp060 可以表述为“提高 value loss 权重到 0.75 保留了 dmax/success/collision 达标，但 timeout 明显差于 exp051”，不能作为下一步主方向。
- exp061 可以表述为“terminal gate 特征直接进入 Actor 会导致动作更激进、success/collision/timeout 同时退化”，不能作为下一步主方向。
- exp062 可以表述为“critic-only 显式 min_pairwise state 保留了 dmax/success/collision 达标，但 timeout 仍差于 exp051”，不能作为当前主结果。
- exp051 没有改变 Actor 输出语义，也没有引入多点采样；当前主线仍是单点 `[rho, beta]` 子目标输出加原有 filter/control 兜底。exp051 multi-seed 复验只改变评估采样，不改变 policy 或底层约束。
- GIF、截图和 TensorBoard 曲线只能用于展示和诊断；严格结论以 `_suite/metrics/strict_acceptance.json`、`metrics/final_eval_proxy.json` 和 `metrics/checkpoint_status.json` 为准。

## Jackal 跟踪验证

活跃 PhysX 脚本：

```text
scripts/evaluate_physx_jackal_tracking.py
```

默认测试：

- 平地 `straight/circle/sine` 跟踪，并可通过 `--tune-flat` 保存调参网格。
- 强三维地形使用 exp009 strong lunar crater 参数：`amplitude=0.16`、`crater_min_radius=0.45`、`crater_max_radius=1.25`、`crater_depth_to_diameter=0.18`。
- 结果以 `tracking_summary.json`、`timeseries.csv` 和 `tracking.png` 为准。

本轮 Jackal tracking 输出：

```text
outputs/runs/physx_jackal_tracking/asset_smoke_jackal/
outputs/runs/physx_jackal_tracking/flat_tuned_final_v2/
outputs/runs/physx_jackal_tracking/strong_lunar_crater_final_v2/
```

平地正式结果通过默认阈值：

| profile | rmse_cross_track_m | max_cross_track_m | path_completion_ratio | max_tilt_deg | status |
| --- | ---: | ---: | ---: | ---: | --- |
| straight | 0.078 | 0.193 | 0.971 | 7.9 | pass |
| circle | 0.159 | 0.292 | 0.971 | 19.5 | pass |
| sine | 0.169 | 0.275 | 0.972 | 8.0 | pass |

强三维地形正式结果未通过默认阈值，主要失败项是完成率和横向误差；最大 tilt 仍低于 35 度：

| profile | path_offset_xy | rmse_cross_track_m | max_cross_track_m | path_completion_ratio | max_tilt_deg | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| straight | `[-0.5, -0.5]` | 0.579 | 0.883 | 0.072 | 23.8 | fail |
| circle | `[1.0, -2.0]` | 0.647 | 1.117 | 0.496 | 33.3 | fail |
| sine | `[0.5, 2.0]` | 0.737 | 1.206 | 0.076 | 28.5 | fail |

强地形失败应解释为“当前 Jackal 低层跟踪控制在 exp009 strong mesh 上仍不足”，不是 proxy 集合任务失败，也不是 Isaac 训练失败。

## 下一步

1. 不启动B0或B2的40M，不恢复安全投影、槽位目标或BC；B1缺少消息年龄依据，B2已在exp137失败，B3缺少前置条件。
2. 不继续调节图注意力头数、层数或与GRU组合。exp137已证明邻居聚合结构不是当前最小瓶颈。
3. exp138已否决最危险车辆对聚合；不修改安全聚合、不启动4M，也不扫描softmax、top-k、距离阈值或权重。
4. exp139已否决零和逐车近距信用；不重开C3-near，也不将重复冲突诊断转化为奖励或辅助损失。
5. exp140已失败；不扫描scale、trace或距离，也不与terrain信用、梯度投影或图注意力叠加。
6. exp142已失败；不扫描cost预算、dual学习率/上界、PID项或cost网络容量，也不与terrain信用、安全投影或图注意力叠加。
7. exp143/144已否决“单纯增加B0训练深度会自然形成地形规划”；不启动12M或40M，也不扫描相对路径风险权重。
8. exp145/146均已停止：不训练统一逐车advantage，不实现最近邻成对安全Critic，不降低门限或扫描模型容量。
9. exp148的时间一致性修正继续作为物理执行基线；exp149已完成碰撞前时序审计，exp150又否决真实碰撞参与者Actor信用。该方向不调参、不组合，也不启动40M。
10. exp151已停止终止参与者信用；exp152进一步将首个失败层定位为动作—quintic局部可控性，而非低层控制饱和。
11. exp153没有找到可单独修改的动作范围或quintic参数；endpoint恒可达但完整路径存在明显途中交叉。下一步只允许冻结验证最终碰撞对的双车联合动作是否显著优于单车干预。
12. 在联合动作证据形成前，不增加协调网络、通信模块、预测冲突reward或在线MAPF，也不启动4M。
13. 当前没有已授权的新训练组件；任何新4M必须等待新的单变量假设通过预注册审计。
14. exp051/exp116仅保留为历史BC-based执行对照，不再作为新策略初始化或当前主线。
