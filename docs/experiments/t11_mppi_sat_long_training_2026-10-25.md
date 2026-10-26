# SAT与三轮MPPI：从零训练100万步

日期：2026-10-25，Asia/Shanghai。用户明确要求从零重新训练100万实际底层学习步，并在启动前检查设置。**本轮不继承旧网络、Adam或价值归一化状态。**

**10月25日20:18:04从零启动；10月26日已按用户指令停止，14:49完成停止记录及checkpoint核验。** 正式run为 outputs/runs/upstream_1km_fixed_far/a1_seed23_mppi_sat_fresh_1m_20261025_01/，准备run为同族 mppi_sat_fresh_preparation_20261025_01/。启动健康检查结果见下文。

## 10月26日用户停止

用户先要求暂停，随后明确停止。已关闭自动总结等待进程，再冻结并终止本轮训练进程组（含4个owner和进行中的独立评估）；SIGTERM后仍存活的父进程及resource tracker用SIGKILL退出，核验进程组无活跃进程。没有执行最终评估或触发自动模型总结。

最后已记录学习采样911,630步；完整checkpoint `checkpoints/update_000071.pt`对应902,079步、71批／355轮PPO，中央及全部4个owner分片SHA验证通过。其后9,551个已记录采样步没有形成新完整保存点；停止时未回传的环境步无法精确计数，不把911,630当作精确物理消耗或可恢复计数。

所有checkpoint、评估记录和events保留；TensorBoard服务保留回放，不再有训练写入。run manifest及自动总结状态已标记用户停止，不自动恢复。证据：[user_stop_20261026.json](../../outputs/runs/upstream_1km_fixed_far/a1_seed23_mppi_sat_fresh_1m_20261025_01/metrics/user_stop_20261026.json)。下文为停止前历史检查。

## 10月26日13:47进度检查

训练已经采样862,141／1,000,000实际底层学习步，完成67批／335轮PPO。主进程及4个owner仍在运行，没有failure文件，全部已保存学习指标均为有限值，无需恢复训练。最近完整checkpoint为 `checkpoints/update_000067.pt`，对应850,974步，中央文件及4个owner分片校验和全部通过；之后仍在收集下一批。

TensorBoard仅展示本run，10份events中的采样曲线从400步更新到862,141步，检查时距最后写入约27秒，本机直连HTTP200。近5批实际平均速度14.47步／秒，按该速度剩余学习采样约2.65小时；最终完整评估会额外耗时。独立评估目前消耗21,268步，不计入上述100万学习步。

当前判断是：**短时间内更少失败了，但持续靠拢还没有稳定学会。** 下表的距离变化是最远车距，缩小才符合任务目标；sample指随机采样策略，mean指采用策略均值。

| 学习步 | 模式 | 60秒内失败 | 平均观察时间 | 平均最远车距变化 |
| --- | --- | --- | --- | --- |
| 0 | sample | 0／4 | 60秒 | 扩大1.28米 |
| 0 | mean | 3／4 | 46.75秒 | 扩大10.22米 |
| 710,947 | sample | 0／4 | 60秒 | 缩小0.66米 |
| 710,947 | mean | 0／4 | 60秒 | 缩小0.16米 |
| 800,020 | sample | 0／4 | 60秒 | 扩大2.23米 |
| 800,020 | mean | 0／4 | 60秒 | 扩大0.74米 |

最新两模式平均实际速度分别0.191／0.328米每秒，车辆有运动，但没有稳定缩短相互距离。每种模式只有4个固定起点，活到60秒只是观察到截止，不能当作完整任务成功。学习期间609个已结束回合中，574次地形失败、35次超时，尚无成功；地形失败子原因分别为468次 `driving_slope`、90次 `impassable_footprint`、16次 `terrain_step`，这是当前环境判据，不代表完成了真实车辆倾覆或触底的物理验证。已结束学习回合有结束先后偏差，不能与固定60秒面板直接比较失败率。

本次仅检查现有结果，0新增环境步、0优化更新；继续原有100万步预算，不改动正在运行的参数。训练及最终评估结束后的单次总结脚本仍处于 `waiting_for_training_and_final_evaluation`，等待期间模型调用数为0。

检查快照：[status_check_20261026_134718.json](../../outputs/runs/upstream_1km_fixed_far/a1_seed23_mppi_sat_fresh_1m_20261025_01/metrics/status_check_20261026_134718.json)。其中保存检查时刻、进程身份、checkpoint校验、TensorBoard末点、全部已完成固定评估汇总及剩余采样时间估算。

## 检查结论

发现并修正一项会使实验使用错误参数的问题：实验YAML指定3轮MPPI，但 runtime_config() 固定覆盖成2轮。训练和独立评估都经过这个入口，直接启动会丢失已选定的三轮设置。

修复后，T11入口明确继承实验配置中的轮数，实际kernel参数保持一致；通用历史局部诊断仍默认两轮，旧两轮训练配置保持两轮。新增检查覆盖旧两轮、车身覆盖四轮及SAT三轮配置的训练／评估入口。原缺陷复现为两项失败，修复后41项相关检查全部通过，无跳过。补丁只修改现有运行配置接线，没有修改MPPI评分算法。

复用前轮109项规划检查与6,000步闭环证据；该轮闭环脚本直接设置kernel轮数，三轮行为证据有效，规划算法源码SHA仍一致。旧迁移准备run未启动训练，现已标记弃用，不作为本轮初始化。

本次实际构建了一个不推进环境的训练实例，确认新网络参数与同seed历史随机初始网络逐值摘要一致，Adam槽为空、价值归一化计数与GRU均为零，观测、动作概率和价值输出有限。实际kernel为512采样×3轮SAT、416点软查询。四张scene及全部atlas文件SHA检查通过；没有剩余启动阻塞项。

## 最终生效设置

| 项目 | 本轮设置 |
| --- | --- |
| 初始化 | seed23，新Actor／Critic／Adam，价值统计和GRU清零；无import／resume参数 |
| 实验入口 | [upstream_1km_mppi_grid_sat_v1.yaml](../../configs/experiment/upstream_1km_mppi_grid_sat_v1.yaml) |
| 学习参数 | [upstream_1km_far_crater_reward_1m_v1.yaml](../../configs/training/upstream_1km_far_crater_reward_1m_v1.yaml)；文件名沿用，是否从零由启动方式决定 |
| 并行与设备 | 4 owner×4环境，每环境4车；CPU策略／PPO，CUDA MPPI；每owner 2个CPU线程 |
| MPPI | 512采样×3轮，预测40×0.2秒＝8秒；温度0.5、线速噪声0.2、角速噪声0.1 |
| MPPI代价倍率 | 目标4、地形3、邻车0.5、控制平滑0.5；与奖励系数分开 |
| 规划几何 | 实际5.015×3.2米车身，零额外规划余量；已观测大坑SAT，普通障碍／中坑0.2米间距的416点软查询 |
| 网络／动作 | Actor523、Critic1439，共享CNN＋GRU；A1极坐标MOVE／HOLD，局部目标半径上限4米 |
| 时序 | 5秒策略决策、1秒重规划、0.2秒实际执行；模拟延迟0.5秒 |
| PPO | 每批32次策略决策、5轮更新；学习率1e-4、clip0.15、熵权重0.0009、价值权重0.5 |
| 梯度与价值 | Actor／Critic分别按范数1裁剪；running价值归一化；未生效终端动作不更新Actor，保留合法Critic目标 |
| 初始探索 | latent标准差0.151633，即原值的0.25倍；HOLD偏置概率0.35，可学习；未启用停车探索候选 |
| 折扣 | 1秒gamma0.9975992791、lambda0.99；0.2秒分别0.9995193941、0.9979919517，按实际持续时间计算GAE |
| 地图与出生 | 1km×1km真实月面，2张训练图轮配4个owner；固定评估图1张；出生半径200–300米、随机朝向、回合1500秒；课程／晋级关闭 |
| 运动学上限 | 前进2.778m/s、后退0.5m/s、角速0.35rad/s；线加／减速0.2／0.4m/s²；实际速度受地形与限幅影响 |
| 学习预算 | 本run从0开始，最多1,000,000实际底层环境步；不按车辆数乘4，不继承旧计数 |
| 保存 | 初始及每批完整checkpoint、4个owner分片及TensorBoard events；旧结果保持 |

车辆profile中的0.3米通用间距余量仍供原有车间规则使用，不代表SAT外扩大坑边界。当前未恢复完整预测足迹拦截／每步安全投影；大坑预测拒绝、实际地形／碰撞终止和执行限幅继续有效。

## 奖励与失败

沿用前轮修正后的奖励。原有持续非势成本乘0.02；靠拢势差与终端事件不缩小。地形失败／越界一次扣100，车车碰撞两项合计扣200，超时扣80，成功加130。真正失败结束当前回合，整体训练继续其他回合。

小于0.5米的坑不加坑标签罚；0.5–不足1.8米中坑允许通过，按0.05/s×团队平均穿入程度轻罚，每回合最多2分，这项不再乘0.02；1.8米及以上大坑在实际车身进入坑口后失败。驾驶移除中小坑的旧坑壁几何失败，岩石／底图检查保持，停车仍要求原始地形合格。

本轮重新检查等成本下的评分方向：更靠拢＞原地保持＞远离。前轮完整回放已检查及时成功、安全超时和提前失败的排序，见[奖励与短训记录](t11_crater_reward_short_training_2026-10-23.md)。软避险罚分也随持续成本减弱，是否仍容易驶入危险区域，要由本轮失败率回答；这不是需要再增加启动矩阵的配置错误。

## 验证证据与命令

准备run保留：

- logs/iterations_before.log：3／4轮被覆盖成2轮的缺陷复现。
- logs/launch_checks.log、launch_checks.xml：41项检查通过。其中隔离测试环境执行合计25底层步，0优化更新，单列于正式学习预算外。
- config/iteration_wiring.patch、config/before/：本次接线修改前后，不包含工作区其他已有修改。
- config/audit_configuration.py、logs/configuration_audit.log：有效参数、四图／atlas SHA、随机初始化、实际kernel和数值审计，0环境步／0更新。
- metrics/configuration_audit.json、static_audit.json、readiness.json、config/effective_configuration.json：审计结果和正式入口准入证据。

实际检查命令：

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=source/lunar_rover_tasks .venv_terrain_cuda/bin/python -m pytest -q tests/test_upstream_training_runtime.py tests/test_upstream_soft_hazard.py tests/test_upstream_fixed_far.py tests/test_upstream_bounded_evaluation.py tests/test_upstream_learning_batch_admission.py --junitxml=outputs/runs/upstream_1km_fixed_far/mppi_sat_fresh_preparation_20261025_01/logs/launch_checks.xml

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=source/lunar_rover_tasks .venv_terrain_cuda/bin/python -u outputs/runs/upstream_1km_fixed_far/mppi_sat_fresh_preparation_20261025_01/config/audit_configuration.py

启动脚本：

    PYTHONPATH=source/lunar_rover_tasks .venv_terrain_cuda/bin/python -u outputs/runs/upstream_1km_fixed_far/mppi_sat_fresh_preparation_20261025_01/config/launch_training.py

精确训练命令、环境变量与PID保存于正式run及准备run的 config/launch.json、training_command.txt；训练控制台保存于准备run logs/training_console.log，正式run logs/console.log提供链接。

## 评估、监控与停止条件

初始化、约每10万学习步以及最终，均使用同4个固定起点的sample／mean两种模式，每例观察60秒；最终另做两种模式各4例的1500秒完整面板。理论最多86,400评估步＝2,400初始＋9×2,400中途＋62,400最终，低于配置88,800上限。提前终止按真实消耗记录，评估不混入学习预算。

先比较本轮随机初始和本轮训练后，再对照旧MPPI训练结果。旧策略历史实际学习约104.97万步且权重不同，不把新旧策略的差异简单归因于规划器。60秒活到截止是删失，不是任务成功；dmax_reduction正值才表示最远车距缩小。存活、实际运动、靠拢与失败率共同判断，不只看奖励。

启动最小验收为配置接线、初始零状态、四个owner、正常采样及events增长，达到后停止模型持续监控。预算结束或运行／数据无效错误出现时停止。低成功率、地形失败和超时保留为结果，不自动追加预算或对照。

TensorBoard只加载本run的tensorboard/，地址 http://127.0.0.1:6006。查看／回放命令：

    .venv_terrain_cuda/bin/tensorboard --logdir outputs/runs/upstream_1km_fixed_far/a1_seed23_mppi_sat_fresh_1m_20261025_01/tensorboard --host 127.0.0.1 --port 6006 --reload_interval 5

config/finish_and_summarize.py仅在启动核验通过后启用；本地每60秒检查训练进程，等待期间0模型调用。训练及最终评估退出后先做纯JSON统计，再触发一次独立codex exec总结；状态在准备run metrics/automatic_summary_status.json，最终文字保存于正式run metrics/automatic_final_summary.md并回写本页。异常退出也总结，不自动重训。

## 实际启动核验

训练PID629272、TensorBoard PID629314，20:18:04启动。20:22:05完成约241秒观察：4个owner和冻结评估进程运行，5段采样合计1,961实际学习底层步，0运行错误文件；尚未完成首批PPO，不把正常采样当作行为改善或优化更新成功。

初始完整checkpoint为本run checkpoints/update_000000.pt及4个owner分片。校验和通过，网络摘要与审计随机初态一致；中央采样计数／PPO更新均为0，Adam无历史槽、价值统计／GRU为零；4个owner时钟均为0、规划热状态与执行队列均为空，全部记录512采样×3轮、CUDA、SAT。源码SHA与冻结身份一致，确认没有接入弃用的旧参数承接方案。

TensorBoard本机直连HTTP200，两份events合计从43,021增至142,829字节，学习采样曲线5点由400增长到1,961，与JSON一致。日志目录为本run tensorboard/，横轴本轮新增实际底层环境步，评估另计。证据在准备run及正式run metrics/startup_verification.json。

启动检查后已启用一次性结束脚本，metrics/automatic_summary_status.json确认 waiting_for_training_and_final_evaluation，等待期间模型调用数0。当前不再由模型持续监控；训练或最终评估退出后才做一次独立总结，结果写入本run metrics/automatic_final_summary.md和本页。启动核验本身0额外环境步、0优化更新。
