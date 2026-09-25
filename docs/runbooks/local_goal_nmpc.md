# 连续局部目标＋NMPC 首轮运行

2026-09-24更新：本文主体描述历史合成CPU路线。真实地图的三通道观测、独立策略/地形设备与无学习检查见[NASA地图运行手册](nasa_training_maps.md)；原六分层评估不能用于新地图。

这是独立的 `exp167` proxy 路线，不要求安装 CUDA、Isaac Sim 或 Isaac Lab，不使用旧 checkpoint。旧 `train_skrl_mappo.py` 和原语环境保持可用。

当前网络训练暂停。2026-09-18规划修正使用 `planner_revision: prefix_consistency_v2`，详见[修正记录](../experiments/exp_167_planner_correction.md)。下列训练命令仅作入口说明，不代表本轮恢复训练；旧pilot的配置/实现hash不匹配新版，不能直接混合续训。

## 安装

先具备现有 `.venv_isaaclab` CPU proxy 环境，再执行：

```bash
bash scripts/install_nmpc.sh
```

该脚本只增加 `requirements-nmpc.txt` 和 acados Python 接口，不重装 PyTorch。acados 固定为 v0.5.5 / `59d93e17d2985fdd73fc58b8a83ed8f83a024171`，CasADi 为 3.7.2。源码、单线程动态库、模板 renderer、生成的求解器均放在已忽略的 `.venv_isaaclab/` 内。acados 的 Python 包元数据仍可能显示 `0.5.1`，复现版本以源码 commit 为准。

安装使用项目虚拟环境内的 CMake，最多 4 个编译进程。已有 acados checkout 不匹配时拒绝覆盖。GitHub TLS 中断可以重试命令，不关闭 TLS 验证。

安装流程依据 [acados 官方 Python 接口文档](https://docs.acados.org/python_interface/index.html#installation)。本机验证范围仅为 CPU proxy，不代表 PhysX 实车动力学。

## 入口与预算

```bash
bash scripts/run_local_goal_nmpc.sh \
  --config configs/experiment/exp167_local_goal_nmpc.yaml \
  --output-layout run --run-name pilot_seed23_100k_cpu
```

默认顺序：

1. 从零初始化 Actor/Critic，保存 `checkpoints/initial.pt`。
2. 六分层各 16 个冻结场景的确定性训练前评估。
3. 4 环境、seed23、每次 rollout 32 个高层步、GRU 序列长度 16 的 MAPPO。
4. 达到 100,000 底层环境交互或 7,200 秒训练墙钟预算后停止采样，不补足预算。
5. 同一批初始状态和地形的六分层训练后评估，校验场景哈希。

**评估不计入训练预算**，其耗时单独写入每层结果，因此整个命令的运行时间可能超过 2 小时。训练在底层周期边界检查墙钟；正在进行的有限次求解、最后一次 PPO 更新及 checkpoint 写出有有限收尾开销。不是实时硬期限进程。

`--max-interactions` 和 `--max-seconds` 只能减少上限。开发 smoke 可使用：

本机首轮正式pilot实际传入 `--max-interactions 99320 --max-seconds 7140`，把先前40＋640次开发短训及其耗时也纳入总上限。

```bash
bash scripts/run_local_goal_nmpc.sh \
  --run-name smoke_seed23_640_cpu \
  --max-interactions 640 --max-seconds 180 --skip-eval
```

`--skip-eval` 或 `--eval-episodes <16` 的产物明确标为 smoke，不能作为完整 pilot 交付。

## 恢复与独立复评

```bash
bash scripts/run_local_goal_nmpc.sh \
  --run-dir outputs/runs/exp167_local_goal_nmpc/pilot_seed23_100k_cpu \
  --resume outputs/runs/exp167_local_goal_nmpc/pilot_seed23_100k_cpu/checkpoints/latest.pt

bash scripts/run_local_goal_nmpc.sh \
  --run-dir outputs/runs/exp167_local_goal_nmpc/pilot_seed23_100k_cpu \
  --evaluate-only
```

checkpoint 包含 Actor、Critic、优化器、学习率调度器、配置/接口版本、消耗预算、GRU 状态、环境状态、局部缓存、消息、上一条规划轨迹及随机数状态。恢复不能扩充 checkpoint 记录的运行预算。只有可信的本地 checkpoint 可以加载，因为完整环境状态采用 Python pickle；不加载外部不可信文件。

训练前后评估的场景定义来自原六分层生成器，Bottleneck 明确修复为 30 坑，不依赖丢失的历史 `outputs/` 配置。各层 16 episode 是诊断规模，不宣称满足正式置信区间门限。已有完整同策略评估结果可以复用；不同策略 hash 重新评估。

## 新接口与信息边界

源码位于 `source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/local_goal/`：

| 模块 | 职责 |
| --- | --- |
| `contracts.py` | 版本化请求、时间轨迹、规划反馈与邻居预测 |
| `memory.py` | 4096 条有界传感样本缓存；区分直接观测、受支撑估计和未知 |
| `planner.py` | 3 秒 / 15 区间 acados NMPC、复核、旧计划回退、受限减速、通用跟踪 |
| `environment.py` | 原传感接口、合法消息传输、真实差速积分/任务判据与高层转移 |
| `learning.py` | N1 多尺度 CNN＋128 GRU、tanh Gaussian 请求动作和序列 MAPPO |

Actor 输入为原 295 维观测加 115 维反馈/记忆/消息，共 410 维；Critic 为原 950 维加四车各 10 维阶段目标/执行反馈，共 990 维。新策略动作是 4 维连续请求，旧配置中的原语字段仅用于创建原环境的既有传感/状态结构，**不进入新执行链或 PPO 概率计算**。

每车 odometry frame 在重置时固定，以自身初始位姿建立。请求只在高层决策时转换，不能随车转动。旧多尺度传感提供的 112 个位置、相对高度和风险样本用于缓存，坡度由这些样本估计；规划器不持有环境或地形后端引用。缓存查询对原观测支持半径之外返回未知，不靠插值增加直接观测标记。

NMPC 使用最近128个可通行观测的支持圆盘之并集构建局部约束，各时刻使用同一可行域，不能把历史轨迹的位置误当成必须到达的时序约束。圆盘遇到已知不可通行样本时保守收缩；再以0.025秒间距复核区间内风险和邻居穿越。这是局部可行域近似，不是全局搜索器。道路通行限制使用traversability≥0.25、坡度范数≤0.8，独立于终端集合平整度阈值。这些离散采样/局部地形估计不构成连续真实地形的无条件安全证明。

差速模型使用原代理的轮半径 0.098 m、轮距 0.376 m、轮速限制 18 rad/s 和 midpoint 积分。控制变化上限每 0.2 秒为 0.3 m/s、0.6 rad/s；无解时按该上限减速。首区间与中间区间均显式施加变化率及安全约束。SQP 最多 12 次；耗时超过 0.15 秒计入 timeout。达到迭代上限但轨迹复核通过的解单独记录，不冒充优化收敛。邻居距离在任务要求 0.42 m 上增加 0.04 m 跟踪余量及随年龄/预测时间增长的余量。

完整轨迹只搭载现有 full-message 链路，稀疏链路不偷取轨迹、目标或实时速度。消息含版本、时间戳、有效期；过期后不再沿旧计划预测，而以陈旧位置和增长的不确定范围处理。

`prefix_consistency_v2`覆盖上述初版规划细节：最多两次地形参数刷新求解，每次6次SQP，总上限仍12次。始终按当前局部地形重算输出控制对应的轨迹。状态0/2候选若无法通过复查，可检查至多14个“原控制前缀＋限速减速停车尾段”，整条3秒轨迹通过原约束才执行，记录为`feasible_prefix`；不修改Actor请求目标。超过0.15秒软时限、异常和非0/2状态仍执行原失败处理。

新鲜完整消息允许表达停车意图；已经停止平移的计划尾段按终端停车延续，不额外添加最大速度未知尾段外推。0.42米名义距离、0.04米余量和年龄/预测时间增长项保留。移动尾段、稀疏或过期消息不获得该停车语义。安全结论受发布意图、模型和跟踪假设限制。

规划遥测新增`prefix_ratio`、`candidate_rejections`、`terrain_refresh_ratio`及`model_error_m_quantiles`。求解器失败改称`solver_failure`，不以状态码证明数学不可行。内部shooting偏差可能仍非零，应同时检查实际发出的轨迹和控制，而不是只看求解状态0。

## 奖励、训练和结果边界

新配置显式映射奖励：保留真实团队几何、平整度、实际安全、成功保持与终止奖励；能耗来自实际位移和转角；关闭原语转角、停滞、切换、Oracle、DSTC 项。未给阶段目标本身奖励，也不提供正确集合点。

请求采用 `tanh(Normal)`，保存采样前 latent，使用含 Jacobian 的一致 log probability。高层奖励按实际底层步数折扣聚合；GAE 使用 `gamma**duration` 和 `(gamma*lambda)**duration`。超时 bootstrap 终止前 Critic 状态，真正终止不 bootstrap，所有终止均截断跨回合 GAE 和 GRU。

本轮采用独立的共享 Actor、集中式 Critic MAPPO 实现，不修改旧 SKRL learner。价值回归为未硬裁剪的 MSE。没有模仿学习、信用重分配或新行为分类。

## 测试

```bash
ACADOS_SOURCE_DIR="$PWD/.venv_isaaclab/acados" \
LD_LIBRARY_PATH="$PWD/.venv_isaaclab/acados/lib" \
OMP_NUM_THREADS=1 .venv_isaaclab/bin/python -m pytest -q \
  tests/test_local_goal_nmpc.py tests/test_local_goal_diagnostics.py \
  tests/test_exp156_differential.py \
  tests/test_tiered_communication.py tests/test_reward.py tests/test_termination.py
```

未设置acados环境时，五项真实求解器测试会skip（四类运动及前进目标切换为倒车目标），其余语义测试用专门的失败注入器；训练入口没有替代或假求解器。

### 2026-09-18 提交前全量验证

使用上述acados环境变量执行 `python -m pytest -q -ra`：exp167测试（含5项真实acados求解器测试）通过；全量存在6项历史配置解析失败，另有1项CUDA测试因CPU环境跳过。失败均来自 `exp156_timeout_diag_{far_bottleneck,near_open}_{96s,144s,192s}.yaml`，依赖本机未恢复的 `outputs/runs/exp156_differential_multiscale_ablation/n0_seed23_full_2400iter/metrics/paired_configs/{far_bottleneck,near_open}.json`。这6项配置已存在于本轮之前，未伪造历史产物或放宽测试，当前不能宣称全量绿灯。

Markdown公式检查、整理文档的相对链接检查、三个新增shell入口语法检查及 `git diff --check`通过。acados产生API弃用警告。

## 产物与判读

标准路径为 `outputs/runs/exp167_local_goal_nmpc/<run_name>/`。重点读取：

- `metrics/baseline_eval.json`、`metrics/final_eval_proxy.json`：六分层完整任务结果和场景 hash。
- `metrics/summary.json`：已消费预算、吞吐、内存、求解分位数、阻塞/失败比例和停止原因。
- `metrics/train_metrics.jsonl`：逐次更新日志，不作为成功证据。
- `metrics/*_trajectories.json.gz`：每层第一批完整轨迹，请求、规划与真实控制分别保存。
- `checkpoints/latest.pt`：可恢复状态，不等于通过验收的最佳策略。
- `run_manifest.json`：命令、配置、设备、运行阶段及产物索引。

生成派生对比表和代表性完整轨迹图：

```bash
.venv_isaaclab/bin/python scripts/summarize_local_goal_nmpc.py \
  --run-dir outputs/runs/exp167_local_goal_nmpc/pilot_seed23_100k_cpu --figures
```

该命令只分析已有记录，不产生新环境交互或策略更新，输出 `metrics/comparison.json` 与 `figures/`。`metrics/progress.json` 可查看当前评估层、批次和底层步数；训练阶段以 `train_metrics.jsonl` 为准。

仅当出现完整任务稳定成功且阻塞改善时，才讨论 CUDA、课程或正式训练预算。若碰撞减少但超时上升、或请求不同而执行趋同，不自动扩大训练，也不继续收紧安全约束。
