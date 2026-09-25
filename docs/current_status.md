# 当前状态

更新时间：2026-09-25。

## 地图主线：真实轨道 DEM＋插值表示＋NASA 模型坑石增强

2026-09-25用户确认这条路线为地图部分后续主线。[地图主线说明](architecture/lunar_map_mainline.md)统一记录处理链、数据规模、边界和下一步；[验收记录](experiments/nasa_sfd_v2_2026-09-24.md)提供结果，[运行手册](runbooks/nasa_sfd_v2.md)提供命令。旧纯随机地图保留为历史/机制对照，LUPEX保留为补充数据与跨源评估候选。

现有NASA NPB母图、采样数量和误差图各4000×4000、5米格距，覆盖20×20千米，合计约192.10 MB。当前仅使用一个200×200米窗口，生成7张0.1米处理网格地图；它们不是7个独立地理区域。NASA产品本身包含插值，处理网格加密不提升测量精度；坑石为NASA模型约束的合成假设，未作NPB区域标定，分形残差关闭。

地形相关36项测试通过；基础图及三个增强图各通过32 GPU环境×32步无学习检查，CPU/GPU查询一致。RTX2060驱动已恢复，单图缓存约122 MiB，GPU分配峰值约179 MiB。接口`lunar_training_terrain_v2`、观测`ego_v12_lunar_multiscale`已接入；此前CPU proxy/NMPC检查保留，完整GPU NMPC性能、接触物理和正式任务成功仍未据此验收。

下一步按地理来源扩展窗口和区域，建立隔离的训练/评估地图库，再冻结任务、车辆、信息和执行条件及学习预算。已用于开发的NPB样例不改称未见测试区；当前200米范围和坑石直径上下限是诊断配置。正式训练入口仍保持暂停，不因为地图主线确认自动恢复学习。

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
- 下一阶段待讨论的是复杂会车、反复目标切换、稀疏/过期消息和执行误差下的完整闭环验证；依据失败、前缀介入、耗时和模型差异决定后续方案。
- 复现前先读[运行手册](runbooks/local_goal_nmpc.md)，使用独立run目录并核对配置与实现hash；不得用旧pilot直接续训新版。
- 结果以机器可读评估为准；`outputs/`中的checkpoint、日志、指标和图像不提交Git。

## 历史入口

exp155–165的详细阶段记录已移至[8月进度归档](archive/progress_summary_2026-08-10_to_2026-08-26.md)，各实验的权威解释见[实验索引](experiments/README.md)。exp166后续消融结论仍以用户提供的历史说明为界，本机未恢复产物，不能补写数值。
