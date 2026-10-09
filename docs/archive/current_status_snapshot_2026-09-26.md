# 当前状态

> 历史快照（事实截至 2026-09-26）。当前状态请读 [docs/current_status.md](../current_status.md)；本页不再更新。

更新时间：2026-09-26。

## 地图主线：真实轨道 DEM＋插值表示＋NASA 模型坑石增强

2026-09-25用户确认这条路线为地图部分后续主线。[地图主线说明](architecture/lunar_map_mainline.md)统一记录处理链、数据规模、边界和下一步；[验收记录](experiments/nasa_sfd_v2_2026-09-24.md)提供结果，[运行手册](runbooks/nasa_sfd_v2.md)提供命令。旧纯随机地图保留为历史/机制对照，LUPEX保留为补充数据与跨源评估候选。

现有NASA NPB母图、采样数量和误差图各4000×4000、5米格距，覆盖20×20千米，合计约192.10 MB。当前仅使用一个200×200米窗口，生成7张0.1米处理网格地图；它们不是7个独立地理区域。NASA产品本身包含插值，处理网格加密不提升测量精度；坑石为NASA模型约束的合成假设，未作NPB区域标定，分形残差关闭。

地形相关36项测试通过；基础图及三个增强图各通过32 GPU环境×32步无学习检查，CPU/GPU查询一致。RTX2060驱动已恢复，单图缓存约122 MiB，GPU分配峰值约179 MiB。接口`lunar_training_terrain_v2`、观测`ego_v12_lunar_multiscale`已接入；此前CPU proxy/NMPC检查保留，完整GPU NMPC性能、接触物理和正式任务成功仍未据此验收。

下一步按地理来源扩展窗口和区域，建立隔离的训练/评估地图库，再冻结任务、车辆、信息和执行条件及学习预算。已用于开发的NPB样例不改称未见测试区；当前200米范围和坑石直径上下限是诊断配置。正式训练入口仍保持暂停，不因为地图主线确认自动恢复学习。

## 新地图任务适配：已接受方案，待实现验证

2026-09-26用户暂定通信半径为**24 m**，接受其余任务适配建议，详见[新地图任务适配方案](architecture/lunar_task_adaptation_v1.md)。24 m承接完整消息范围的讨论；稀疏消息最大范围、周期和有效期单独配置，并解除与地图尺寸的绑定。

接受的实验起点：出生半径4–6/8–12/15–25 m分档，地理位置与队形分别采样，增加车体足迹和初始间距检查；新版集合候选为最大车间距离2.0 m、离散度0.75 m²、每车速度≤0.05 m/s、角速度≤0.1 rad/s、连续驻留5 s。共同作业区、逐车停车足迹和净空分别检查；平坦度拆分为倾斜、去趋势粗糙度及台阶。96/180 s时限先做无学习配对比较。

本轮仅记录决定与实施要求，运行代码及配置尚未切换：现有NPB接口诊断仍为12 m完整通信、8–12 m出生半径及20 s回合，成功逻辑仍沿用旧值。具体车体/作业区尺寸、地形安全阈值及稀疏通信参数需在适配时确定；不宣称新版已通过验证。正式训练和Active-DSTC继续暂停。

## 学习系统重审：首轮代码与无学习核验完成

2026-09-26按用户要求审查地图以外的奖励、动作／状态／观测、网络与强化学习算法，详见[学习系统审查与升级方案](architecture/learning_system_review_2026-09-26.md)。已核实：底层γ=0.99对应13.79 s奖励半衰期；任务超时仍bootstrap且无显式剩余时间；原驻留奖励在8步改25步后由54增至156；基础邻车消息按距离排序，附加目标／轨迹按sender编号排列。后两项分别是奖励分量数值和输入合同问题，不代表已证明策略钻奖励漏洞或新结构更优。

新增无学习审计脚本`scripts/audit_learning_contracts.py`，报告位于`outputs/diagnostics/learning_contract_audit_2026-09-26/report_final.json`；相关46项现有合同测试通过，未运行PPO更新或新NMPC性能实验。升级优先级为时间／终止与邻车实体合同→奖励回报和动作执行验证→集合编码及历史条件Critic→同预算算法比较。生产实现未切换，新候选尚未通过性能验证，正式训练与Active-DSTC继续暂停。

## 当前研究：任务与路线重审

按用户要求完成“局部地形观测与有限通信下的终端区域选择”文献调查与项目重设计：[27篇文献综述及现有缺陷](references/terminal_feasibility_review_2026-09-19.md)、[新设计](architecture/terminal_feasibility_redesign_v1.md)、[因果诊断协议](experiments/terminal_feasibility_research_protocol.md)。研究重点为信息条件、联合可达性、进入协调与驻留，以及奖励和可学习性的关系。

本轮是文献、代码和已有结果审查，尚未实现新架构或运行新训练。建议先执行无训练的任务/执行因果审计，再建立完整任务控制参照，之后研究局部信息和学习的独立增益。平坦度的实际作业用途及方法约束在设计中明确标为待确定的任务假设，未修改生产判据。exp167仍为当前实现，Active-DSTC继续暂停。

## 当前实施：连续局部目标＋NMPC

用户批准的新路线已独立实现为 `exp167`：四维连续阶段目标、N1＋GRU、acados NMPC、通用差速执行和序列 MAPPO。旧训练入口保留，Active-DSTC继续搁置。

完整pilot已结束：99,319次正式短训交互、156次更新、113.5分钟；包含开发短训累计99,999次交互。训练前后同一六分层各16场景均为0/96成功、0碰撞、96超时，未收敛。checkpoint保留，不追加训练。

无网络固定目标诊断后，用户已批准并完成规划修正`prefix_consistency_v2`：有限次地形参数刷新、重新生成并复查输出轨迹、可行运动前缀加停车尾段、新鲜完整停车消息语义。原真实安全距离、传感和通信范围未改。配对复测三地形144/144短目标及弧线16/16仍通过；低速/高速1.2米目标均16/16到达，近距停车由960/960规划失败变为0/960。高速仍有63/960次求解失败或超时，内部优化残差并非全为零，不称为完整任务收敛。训练继续暂停，旧pilot checkpoint不能直接混用新版。详见[诊断](experiments/exp_167_fixed_goal_diagnosis.md)和[修正结果](experiments/exp_167_planner_correction.md)。

运行目录：`outputs/runs/exp167_local_goal_nmpc/pilot_seed23_100k_cpu/`。实现边界、预算和安装命令见[exp167记录](experiments/exp_167_local_goal_nmpc.md)及[运行手册](runbooks/local_goal_nmpc.md)。评估耗时独立于训练预算。

## 换机后的当前决定

用户补充：exp166及其后续消融表明Active-DSTC没有发挥作用，暂时搁置该方向。本机尚未恢复这些训练产物，因此该结论记录为用户提供的历史结果，不补写未核验的数值，也不重新启动exp166或相关消融。

Python 3.12代理训练环境和不含Active-DSTC的exp156 smoke验证已完成；这不代表恢复exp156架构比较或策略有效。历史结果与当时计划见[8月进度归档](archive/progress_summary_2026-08-10_to_2026-08-26.md)。

本机使用RTX 2060 6 GB、16 GB内存。沿用`.venv_isaaclab`路径安装proxy-only依赖，Isaac Sim/Isaac Lab高保真栈暂不安装。安装入口为`scripts/install_proxy_stack.sh`，与完整栈的`scripts/install_stack.sh`区分。

当前环境已完成：Python 3.12.3、PyTorch 2.10.0 CPU、Gymnasium 1.2.1、SKRL 2.1.0、项目任务包 editable 安装，`pip check`通过。`exp156_smoke`的2环境/32步真实MAPPO短训已通过，2次joint update、checkpoint和独立评测均成功；短训严格门控未通过属于预期，不能作为策略结果。

## 推荐 checkpoint 与结果边界

当前主线没有通过完整任务验收的推荐 checkpoint。`pilot_seed23_100k_cpu/checkpoints/latest.pt`只保留为旧规划版本的失败诊断记录；新版规划器的固定目标通过不能替代完整任务评估。Isaac/PhysX验证尚未开展。

## 下一步与停止条件

- 地图工作按[主线顺序](architecture/lunar_map_mainline.md)推进：扩展地理覆盖与隔离地图库→任务/执行适配→冻结比较协议与预算。所需GPU NMPC验证单独记录；正式训练仍待明确恢复。
- 保持网络训练暂停，不追加本轮100,000交互预算，不恢复Active-DSTC或旧架构消融。
- 按[已接受的任务适配方案](architecture/lunar_task_adaptation_v1.md)实施24 m通信、出生分层及停车足迹/驻留检查，再以无学习闭环确定任务时限与参数；历史配置和判据保留。
- 学习层按[首轮审查方案L1—L2](architecture/learning_system_review_2026-09-26.md)先处理时间终止、邻车信息对应、回报排序与动作覆盖；网络和算法候选逐项验证，不因本轮审查自动恢复学习。
- [实验协议A0—A4](experiments/terminal_feasibility_research_protocol.md)保留为算法/任务研究参考，不作为真实数据地图接入的前置阶段；新设计目前没有实施结果。
- 下一阶段待讨论的是复杂会车、反复目标切换、稀疏/过期消息和执行误差下的完整闭环验证；依据失败、前缀介入、耗时和模型差异决定后续方案。
- 复现前先读[运行手册](runbooks/local_goal_nmpc.md)，使用独立run目录并核对配置与实现hash；不得用旧pilot直接续训新版。
- 结果以机器可读评估为准；`outputs/`中的checkpoint、日志、指标和图像不提交Git。

## 历史入口

exp155–165的详细阶段记录已移至[8月进度归档](archive/progress_summary_2026-08-10_to_2026-08-26.md)，各实验的权威解释见[实验索引](experiments/README.md)。exp166后续消融结论仍以用户提供的历史说明为界，本机未恢复产物，不能补写数值。
