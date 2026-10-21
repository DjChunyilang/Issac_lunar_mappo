# T11三项组合修复：近／远混合初态闭环学习验证

## 已完成结果：组合未通过预设学习筛选

两组真实闭环学习与初始／最终评估均正常结束。组合最终随机远距表现优于baseline，但比自身未训练状态退步；近距成功驻留数未增加。三项修复已经正确进入采样和PPO路径，本轮没有验证出正的组合学习增益，不将其提升为默认训练方案，也不追加预算。

| 指标 | baseline 初始 | baseline 最终 | combo 初始 | combo 最终 |
|---|---:|---:|---:|---:|
| 远距随机：平均最大车距缩小（m／20 s） | 1.494 | -0.731 | 4.441 | 2.614 |
| 远距随机：净靠拢初态数 | 4/4 | 1/4 | 4/4 | 4/4 |
| 远距均值：平均最大车距缩小（m／20 s） | 0.000 | 0.000 | 0.000 | 0.000 |
| 近距随机：成功驻留数 | 1/4 | 1/4 | 1/4 | 1/4 |
| 近距均值：成功驻留数 | 4/4 | 4/4 | 4/4 | 4/4 |

正的车距缩小表示靠拢，负值表示发散。combo最终比baseline多靠拢3.345 m，但训练前已有2.947 m差距；两组前后变化分别为−2.225 m、−1.827 m。组合的较好终点主要保留了低噪声初始化优势，不能据此称为训练改善。组合满足7项筛选中的5项，未满足“比自身初始化多靠拢至少1 m”和“近距成功数较自身增加”。

远距随机HOLD概率：baseline约50.01%→51.90%，combo约50.01%→50.52%；combo最终近距随机HOLD约50.50%。两组远距均值策略均选择HOLD，20 s没有最大车距缩小；近距均值4/4是初始化已有表现。当前未观察到可靠的“远处推进、近处停车”学习区分。

| 实际计数 | baseline | combo |
|---|---:|---:|
| 训练底层环境步 | 19,910 | 19,844 |
| 未使用训练预算 | 90 | 156 |
| 策略团队决策／有效转移 | 800／800 | 800／800 |
| 采样更新／PPO轮数 | 13／65 | 13／65 |
| 训练成功回合（均来自近距） | 4 | 7 |
| Actor屏蔽的未生效终端动作 | 0／4 | 7／7 |
| 近距几何停车条件满足步数 | 6,049／9,910（61.0%） | 4,628／9,844（47.0%） |
| 独立行为评估步数 | 2,564 | 2,538 |
| 含初始／最终评估耗时 | 25.59 min | 25.34 min |

两组实际预算差66步，均小于一个400步完整向量区间。训练合计39,754步，行为评估5,102步，加新增契约检查16步，无学习合计5,118／8,000步；本轮总计44,872实际底层环境步。全部采样有效，无失败中止或自动重试。

训练成功7次多于4次，但combo近距几何条件覆盖减少，固定评估仍只有随机1/4成功，因此训练成功计数不构成停车学习改善。两组远距训练队列均没有达到完整几何集合条件；本轮是固定20 s行为探针与短学习比较，未执行完整Isaac/PhysX任务验收。

三项组合确实运行：Actor梯度裁剪系数中位数为1（baseline全局裁剪约0.0509），全部7条未生效终端动作只屏蔽Actor，Critic仍使用800个有效转移；连续标准差初始为约0.15163，最终两个分量为0.15181／0.15167，保持可学习。两组Actor/Critic参数均发生实际更新；全部26个采样批次在更新前用于Actor更新的重放概率比值严格为1。训练执行记录重算步数与checkpoint计数一致；训练及全部固定评估均无碰撞和间距违规。

证据：suite的`metrics/summary.json`、`training_accounting.json`、`final_verification.json`；两组完整恢复点均为各自`checkpoints/update_000013.pt`及四个owner分片。主进程退出码0，采样owner和GPU进程已退出。最终检查验证了checkpoint／分片校验值、配置与源码身份及实际计数；保留旧A0/A1/A/B/C产物和原默认配置。

## 问题、范围与结束条件

用户2026-10-19明确要求“验证三项组合的闭环学习效果”。本轮用真实采样→5轮PPO→新策略采样检验独立Actor/Critic裁剪、连续分布初始标准差×0.25、终端未生效动作Actor信用屏蔽的组合。奖励、A1动作、MPPI512×2、安全与成功判据不变。两组均从种子23随机初始化开始，使用全新Adam；不继承旧HOLD退化策略。

| 组别 | 裁剪 | 初始连续标准差倍率 | 终端Actor信用 |
|---|---|---:|---|
| baseline | global | 1 | all |
| combo | actor_critic | 0.25 | effective |

两组均使用C的5 s决策与1 s折扣参考周期，四owner×四环境、每批每环境4个决策（20 s）、5轮PPO。实际推进底层环境步每组上限20,000；每次完整16环境决策预留400步，不足则退出，保留不足400步差额。两组顺序执行，原20万步A/B/C不追加。

为覆盖停车学习，两组都使用`near_far_balanced_v1`：偶数环境保留compact远距初态，奇数环境为四种满足间距／最大车距／离散度／包含约束的近距几何，每种两套车头朝向。近距初始前进速度0.1 m/s，每次重置保持一致。8×7 m矩形取代此前只用于边界诊断、离散度超标的8.5×7 m矩形。该相同初态安排是本轮实验条件，不能直接与旧远距训练C归因比较。

学习前／最终各评估一次：原四个远距固定物理快照、四个近距几何，分别随机与均值策略20 s。近距评估复用训练几何的第一套朝向，是获取停车行为的探针，不作为泛化测试。评估使用独立环境、GRU与RNG，不进入PPO。每阶段预留1,600步；四阶段6,400步，加新增契约检查16步，低于本轮独立评估总上限8,000步。

预先定义组合学习筛选：随机远距至少3/4净靠拢、平均最大车距缩小比baseline最终多至少1 m，并比combo自身初始化多至少1 m；平均离散度不增加、远距HOLD概率不高于baseline；近距两种策略成功数各不低于baseline，至少一种相对自身初始化增加。须同时满足。单种子短预算仅决定是否观察到组合学习信号，不声称统计显著。预算耗尽或影响数据有效性的失败即结束，不因成功率低追加训练。

## 实现与验收

沿用`run_upstream_t11.py`、`upstream_decision_experiment.run_arm`、Process采样、skrl更新和分片checkpoint，新增显式`--mode repair-validation --budget 20000`。混合初态模块为`upstream_learning_cohort.py`，两组编排与筛选为`upstream_learning_validation.py`。旧A/B/C入口保持原预算和默认选项。

环境可选`initial_control`同时设置初始执行控制及实际速度，重置和快照恢复保留；默认不设置时不改变旧元数据。候选状态身份包含该控制和cohort，拒绝直接装入旧环境。2万步研究准入与原signal100k／20万步准入分别检查。

新增两项针对性检查通过（6.07 s，实际16底层步），另两项旧入口／评估预算检查通过（1.84 s，0物理步）：四种近距几何合法、owner划分一致、初速经过重置／恢复保持、旧环境拒绝混合状态；四owner真实终止采样有效、未生效动作标记正确、授权与预算上限拒绝正确。复用上一轮10项修复证据，不重跑无关矩阵。

检查run：`outputs/runs/upstream_scale_t11/learning_repair_closed_loop_seed23_20k_20261019_01_checks/`，日志`logs/contracts.log`，JUnit `metrics/contracts.xml`。

## 运行与证据

suite：`outputs/runs/upstream_scale_t11/learning_repair_closed_loop_seed23_20k_20261019_01/`。两组分别同名前缀加`_baseline`、`_combo`，均不覆盖旧产物。检查run的`logs/learning.log`保留主进程stdout/stderr。

```bash
.venv_terrain_cuda/bin/python scripts/run_upstream_t11.py \
  --config configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml \
  --run-id learning_repair_closed_loop_seed23_20k_20261019_01 \
  --mode repair-validation --representation A1 --seed 23 \
  --budget 20000 --samples 512 --env-owners 4 --num-envs 16 \
  --owner-cpu-workers 2 --trainer-backend skrl --policy-device cpu \
  --release-training-pause \
  --far-checkpoint outputs/runs/upstream_scale_t11/oom_fix_owner16_20261017_01_a1/checkpoints/update_000061.pt
```

各组`metrics/initialization.json`记录缩放前后参数SHA；`metrics/update_*.json`记录实际步、决策、回合、信用屏蔽数及更新指标；`state/interactions_*.pkl.gz`保留完整更新输入和RNG；`checkpoints/update_*.pt`及owner分片保存恢复边界。`metrics/evaluation_{initial,final}.json`为行为证据，suite的`metrics/summary.json`汇总筛选结果。

两组顺序实验已完成。suite的`config/source_snapshot.tar.gz`、`source_archive.json`及`configuration_inputs.tar.gz`保存实际源码、19份继承配置／设计输入与依赖版本；逐文件SHA与运行身份一致。结果见本记录开头。Proxy任务成功与Isaac/PhysX验收分别记录。

只读汇总命令（0新增环境步）：

```bash
.venv_terrain_cuda/bin/python outputs/runs/upstream_scale_t11/learning_repair_closed_loop_seed23_20k_20261019_01_checks/state/analyze.py
```
