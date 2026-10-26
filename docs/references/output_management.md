# 输出管理

2026-10-26部署交接更新：新run为 `outputs/runs/deployment_handoff/windows_wsl2_20261026_01/`，保留旧交接包。新源码／状态包纳入截至本日的工作区、4组完整checkpoint及10月24日至26日训练／诊断证据；当前四图与atlas核验未变，复用旧20.26 GiB数据归档，已传输者无需重复下载。接收方式、局域网地址、依赖和最多500步无学习验收见[交接文档](../runbooks/windows_wsl2_deployment_handoff_2026-10-23.md)。最新SAT训练已于14:49按用户指令停止；完整保存点为第71批／902,079步，日志末点911,630步不等于恢复点，未回传消耗未知。自动总结等待进程已停止，6006只供回放，本次部署不恢复训练。以下为历史启动与运行记录。

2026-10-25当前运行：[SAT三轮MPPI从零百万步训练](../experiments/t11_mppi_sat_long_training_2026-10-25.md)，run为`outputs/runs/upstream_1km_fixed_far/a1_seed23_mppi_sat_fresh_1m_20261025_01/`，20:18启动。全新网络／Adam／价值统计，初始计数0，学习上限1,000,000；独立评估调度最多86,400／上限88,800。启动前修复训练入口轮数覆盖，41项检查及四图／atlas、初始化审计通过。下面旧恢复预算和命令仅为历史。

当前TensorBoard地址 `http://127.0.0.1:6006`，仅加载本run `tensorboard/`，每次学习采样写events。查看命令：

```bash
.venv_terrain_cuda/bin/tensorboard --logdir outputs/runs/upstream_1km_fixed_far/a1_seed23_mppi_sat_fresh_1m_20261025_01/tensorboard --host 127.0.0.1 --port 6006 --reload_interval 5
```

本轮精确命令见run的`config/training_command.txt`、`launch.json`，没有`--resume-state`或任何import参数。准备目录为同族`mppi_sat_fresh_preparation_20261025_01/`；启动检查后本地等待脚本在结束时只触发一次总结，等待期间不调用模型。旧迁移准备`mppi_sat_long_preparation_20261025_01`已弃用，不可用其启动脚本替代本轮。


2026-10-25最新优化：`outputs/runs/upstream_1km_fixed_far/mppi_query_optimization_20261025_01/`，10分支共6,000诊断步、6批冻结规划、0学习更新；`metrics/final_summary.json`、`comparison.json`、`completion_verification.json`保存预算、行为等价和恢复／events证据。源码已接入合并查询、精简回传、SAT和审查优化；通过显式`upstream_1km_mppi_grid_sat_v1.yaml`选择3轮SAT，旧配置／checkpoint保留。TensorBoard6006仅加载本run的`tensorboard/`，events更新至6,000步后回放，诊断进程退出。命令、原始失败准备及计时口径见[优化记录](../experiments/t11_mppi_query_optimization_2026-10-25.md)；下文服务信息均为历史切换过程。

2026-10-25最新对照：`outputs/runs/upstream_1km_fixed_far/mppi_collision_algorithms_20261025_01/`，冻结查询加6分支闭环，共3,600诊断步、0学习更新。`metrics/benchmark.json`、`comparison.json`、`completion_verification.json`保存算法耗时、闭环及恢复／events核验；原型在run的`config/collision_queries.py`，生产源码未改。TensorBoard6006当前只加载本run的`tensorboard/`，2个events更新至3,600步后保留回放。矩形—圆分支明确标注oracle，结论及命令见[算法对照](../experiments/t11_collision_algorithms_2026-10-25.md)；下文前轮服务信息保留历史切换过程。

2026-10-25最新诊断：`outputs/runs/upstream_1km_fixed_far/mppi_execution_repair_20261025_01/`，19次冻结规划、5,400诊断底层步、0学习更新，已完成退出。`metrics/final_comparison.json`记录9个分支，`verification.json`核对验收、预算、恢复一致性及events；`config/original_*`、`phase2_*`、`final_*`隔离修改前、中间及最终实现。TensorBoard6006只加载本run的`tensorboard/`，12个events已随诊断更新至5,400步并保留回放；旧训练及前轮目标诊断产物不覆盖。结论和命令见[MPPI执行修复](../experiments/t11_mppi_execution_repair_2026-10-25.md)。

2026-10-25恢复run已结束：`outputs/runs/upstream_1km_fixed_far/a1_seed23_crater_reward_1m_recovered_20261024_01/`。分支计数1,036,889，加原故障丢弃12,794步后历史累计1,049,683步；相对49,871步新增999,812／100万步，独立评估53,860／88,800步（继承初始化2,400步只计一次）。最终`checkpoints/update_000082.pt`及4个owner分片、11个events文件已核验；`metrics/completion_artifact_verification.json`保存大小／SHA与日志末点，`long_training_review.json`及`figures/long_training_progress.png/pdf`保存汇总。TensorBoard保留恢复边界62,628和初始化评估49,871步，回放只加载本run的`tensorboard/`，本次总结未启动服务。准备产物及一次性结束总结状态在同族 `crater_reward_recovery_20261024_01/`，见[完整记录](../experiments/t11_crater_reward_recovery_2026-10-24.md)。原5万步run及 `a1_seed23_crater_reward_1m_20261024_01/` 的失败批次、checkpoint与events全部保留。

2026-10-23部署交接：最新短训于23:09按预算完成49,871学习步和4,427独立评估步，最终恢复点为同run的 `checkpoints/update_000004.pt` 及全部owner分片，events保留回放。交接产物位于 `outputs/runs/deployment_handoff/windows_wsl2_20261023_01/`，包含源码／最终状态包、当前四图／atlas数据包、版本及SHA清单；不提交outputs。接收与验收见[Windows／WSL2交接](../runbooks/windows_wsl2_deployment_handoff_2026-10-23.md)。下文22:21及更早条目保留各自启动时事实。

2026-10-23 22:21当前短训run：`outputs/runs/upstream_1km_fixed_far/a1_seed23_crater_reward_50k_20261023_01/`。新奖励／坑分档，从0开始，50,000学习步上限；前后60秒评估最多4,800步独立计费。TensorBoard6006只加载该run的`tensorboard/`。`config/training_command.txt`、`launch.json`和`logs/console.log`保留实际启动边界；准备与100步无学习检查分别在同实验族`crater_reward_preparation_20261023_01`及`crater_reward_prelaunch_20261023_01`，见[短训记录](../experiments/t11_crater_reward_short_training_2026-10-23.md)。以下为历史run事实。

2026-10-23当前正式run：`outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_optimized_20261022_01/`。完整继承障碍输入父run第23批／261,149步，累计上限1,497,631步，切换时剩余1,236,482步；更早502,369步仍计入200万总预算。`metrics/migration.json`、`migration_summary.json`、`config/resume_command.txt`及源码包保存谱系与边界，独立`tensorboard/`复用6006。一次性切换工具和验证保存在`outputs/runs/local_obstacle_input/sampling_optimized_training_switch_20261022_01/`，原run产物不覆盖。见[优化续训记录](../experiments/t11_optimized_training_resume_2026-10-23.md)。下列日期条目保留历史运行事实。

2026-10-22当前正式run：`outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_soft_hazard_2m_20261022_01/`，无预测足迹拦截、全新网络/Adam；200万学习／40万独立评估底层步上限。`config/command.txt`、`launch.json`、源码包、`metrics/prelaunch_evidence.json`保留启动边界，`logs/console.log`指向原始启动日志；`tensorboard/`加入现有6006服务，成功率精选链接在`pinned_url.txt`。前置run为`soft_hazard_prelaunch_20261022_01`与`soft_hazard_owner16_probe_20261022_01`，合计748诊断步，不混入正式学习计数。见[本轮记录](../experiments/t11_soft_hazard_training_2026-10-22.md)。

2026-10-22追加：容量套件`outputs/runs/upstream_1km_fixed_far/parallel_capacity_20261022_01/`，16/24环境分别为`cuda_owner16_probe_20261022_03/`、`cuda_owner24_probe_20261022_01/`。总1951／2040步、0 PPO更新；`metrics/verification.json`、`capacity_resources.jsonl`保留计数及PSS/USS/swap，24环境另存六个诊断owner分片，不能作为完整恢复checkpoint。24环境以`config/executed_source.tar.gz`和对应SHA为最终执行源码，原预备包保留。查询候选已撤回，最终源码恢复证据见套件`metrics/final_source_restoration.json`。各TensorBoard events持久保留，正式训练未恢复。

2026-10-22追加：顺序优化复测位于`outputs/runs/upstream_1km_fixed_far/cuda_owner16_probe_20261022_02/`，764／800步、0 PPO更新。基线为`cuda_owner16_probe_20261022_01`；`metrics/verification.json`记录行为逐值核验及吞吐／显存变化，`config/before.tar.gz`和`config/source.tar.gz`分别保留修改前后源码。两个run的`Perf/*`均可在现有TensorBoard服务比较，不覆盖旧events，不恢复正式训练。

2026-10-22：下述正式run已按用户指令停止；`metrics/user_stop_for_optimization.json`记录未提交采样的物理步上界及完整初始checkpoint。`upstream_1km_fixed_far/gpu_review_20261022_01/`为已完成的无学习性能对照，实际100／200环境步。另完成`cuda_owner16_probe_20261022_01/`优化版16环境小采样，764／800步、0 PPO更新；`tensorboard/`的`Perf/Steps_s`、`Perf/Interval_s`、`Perf/Updates`横轴为实际底层环境步，归档在`state/interval_*.pkl.gz`，不是可恢复训练checkpoint。原整批方案`e2e_owner16_20261022_01`已在启动前取消，0步。原TensorBoard events保留回放，不自动恢复正式训练，见[GPU复核记录](../experiments/t11_cuda_footprint_optimization_2026-10-22.md)。

## 25分钟固定远距月面run

2026-10-21 23:35起活跃run为`a1_seed23_far_lunar_1500s_2m_perf_20261021_02`，旧run保留。固定评估单独进程在同一`tensorboard/`写`.evaluation`后缀的进度events（不purge），主进程仍投影完成的JSON及训练曲线；评估快照和SHA在`state/evaluation_<phase>.pt/json`。`sampling/base_steps_per_second`按实际学习底层步显示当前采样批的平均吞吐量，评估步不混入；中断与替换消耗记录于`metrics/performance_provenance.json`和`evaluation_interruption_*.json`。见[性能修复记录](../experiments/t11_lunar_runtime_performance_2026-10-21.md)。

新run族`outputs/runs/upstream_1km_fixed_far/`；训练配置`configs/training/upstream_1km_far_fixed_v1.yaml`。TensorBoard `TrainingTensorBoard(..., pinned=full_far)`将完整任务`Main/SR_S`、`Main/SR_M`、`Main/Train_SR`置于重点曲线，另含Return、Value_EV、KL、Clip、Entropy、HOLD、Dmax_S；仅Dmax是靠拢诊断。完整评估期间`metrics/evaluation_progress.json`为可原子替换的当前状态，最终evaluation JSON仍不可覆盖；事件中的evaluation_progress单独记账，横轴保持实际训练步。服务6006同时加载history课程族和far固定族；每run的`tensorboard/pinned_url.txt`提供精选入口。

生成产物采用 run-oriented 目录结构管理。目标是让长期实验可检索，同时避免提交大型生成文件。

## 标准目录结构

每个具体 run 使用一个独立目录：

```text
outputs/runs/<experiment_id>/<run_id>/
  config/
    experiment.yaml
  checkpoints/
    best.pt
    ppo_update_XXX.pt
  metrics/
    summary.json
    train_metrics.jsonl
    eval_metrics.json
    final_eval_proxy.json
    strict_acceptance.json
    checkpoint_status.json
  figures/
    convergence_curves.png
    safety_diagnostics.png
    comparison_curves.png
  videos/
    proxy_eval_rollout.gif
  tensorboard/
    events.out.tfevents...
  tensorboard_curated/
    events.out.tfevents...
  physx/
    metrics/
      <terrain>_headless.json
      <terrain>_render.json
    figures/
      <terrain>_render_scene.png
    videos/
      <terrain>_render_rollout.gif
  run_manifest.json
```

实验级 suite 输出使用保留 run 名 `_suite`：

```text
outputs/runs/<experiment_id>/_suite/
  metrics/
    suite_summary.json
    strict_acceptance.json
    final_eval_best.json
  figures/
    comparison_curves.png
    safety_diagnostics.png
  run_manifest.json
```

根索引用于汇总所有 manifest：

```text
outputs/runs/_index.json
```

历史路径 `outputs/logs/`、`outputs/checkpoints/`、`outputs/figures/` 和 `outputs/videos/` 保留兼容。新工作优先使用 `outputs/runs/`。

历史 SKRL-MAPPO CUDA 诊断产物仍保留以下过渡路径：

```text
outputs/runs/<experiment_id>/metrics.jsonl
outputs/runs/<experiment_id>/diagnosis_<label>_<timesteps>.json
outputs/runs/<experiment_id>/suite_logs/
outputs/runs/cuda_training_validation_summary.json
```

这些文件是 ignored 的工程诊断产物，不是 strict acceptance。正式 SKRL 训练入口 `scripts/train_skrl_mappo.py` 已支持 `--output-layout run`、`--run-name`、定期兼容 checkpoint、独立候选评估和标准 manifest；长期结论仍写入 `docs/experiments/*.md`。

当前标准 terrain-aware convergence suite：

```text
outputs/runs/exp_008_terrain3d/_suite/
```

该 suite 包含最终严格 proxy 验收、对比图，以及复制出的各 seed best checkpoint。

## 命名规则

- `experiment_id`：稳定的实验族，使用 lower snake case，通常为 `exp_###_<topic>`。
  - 推荐：`exp_007_phase_c`、`exp_008_terrain3d`。
  - 不要因为 seed 或小 retry 改变 `experiment_id`。
- `run_id`：具体设置，使用 lower snake case，不含空格。
  - 推荐模式：`<mode>_seed<seed>_<budget>_<terrain>_<short_tag>`。
  - 示例：`weak_warmstart_seed23_6m_lunar_crater_bc50`、`pure_rl_seed31_2m_flat`、`smoke_seed23_8k_cpu`。
- checkpoint：
  - `checkpoints/best.pt`：该 run 的标准 checkpoint。
  - `checkpoints/ppo_update_XXX.pt`：可选的 update checkpoint。
- metrics：
  - `metrics/summary.json`：dashboard 和快速检查用的一文件摘要。
  - `metrics/train_metrics.jsonl`：逐 update 训练记录。
  - `metrics/eval_metrics.json`：训练过程中的 deterministic eval 记录。
  - `metrics/final_eval_proxy.json`：训练后的独立 proxy 评估。
  - `metrics/checkpoint_status.json`：统一 checkpoint 评估状态，连接 proxy gate 与 high-fidelity eval。
- visual：
  - Proxy 图和视频放在 `figures/` 和 `videos/`。
  - Isaac Sim / PhysX 产物放在 `physx/metrics|figures|videos`。

## Checkpoint 状态

`metrics/checkpoint_status.json` 的 `state` 只能使用：

```text
candidate
proxy_passed
physx_evaluated
physx_passed
final_selected
```

`candidate` 表示尚未通过 proxy strict gate；`proxy_passed` 表示 proxy gate 通过但 high-fidelity eval 尚未通过或未触发；`physx_evaluated` 表示已跑高保真评估但未通过 gate；`physx_passed` 表示 high-fidelity gate 通过；`final_selected` 只能用于明确选定的最终 checkpoint。

## 整理已有 outputs

只查看迁移计划，不写入文件：

```bash
.venv_isaaclab/bin/python scripts/organize_outputs.py \
  --all-known \
  --mode symlink \
  --dry-run
```

为所有已知历史实验创建标准 symlink：

```bash
.venv_isaaclab/bin/python scripts/organize_outputs.py \
  --all-known \
  --mode symlink
```

刷新单个实验：

```bash
.venv_isaaclab/bin/python scripts/organize_outputs.py \
  --experiment exp_006_ppo_selected \
  --mode symlink
```

刷新 Phase C run，并包含 PhysX 展示产物：

```bash
.venv_isaaclab/bin/python scripts/organize_outputs.py \
  --preset exp007_phase_c \
  --mode symlink \
  --overwrite
```

只有当 run 目录必须完全自包含并脱离历史路径时，才使用 `--mode copy`。除非要刷新生成的标准链接，否则避免使用 `--overwrite`。

## 后续训练

2026-10-21 起，用户要求**所有后续训练必须记录 TensorBoard**，包括正式训练、短程对照和学习探针；未来新增／重新启用的学习入口也须遵守。依赖缺失、不可写或无 events 不能当作成功记录。当前 `.venv_terrain_cuda` 已安装 TensorBoard，依赖在 `requirements-proxy.txt` 中。启动时报告 run、日志目录及查看命令；events 随 run 保留，结束后仍可查看。

### 当前 T11 的实时监看与历史回放

`run_upstream_t11.py` 的原始训练、signal、diagnostic、decision-credit、repair-validation、课程训练，以及 `upstream_parking_probe.py`／`upstream_parking_exploration_probe.py` 均在现有采样／更新／评估边界调用 `scripts/training_tensorboard.py`。中央学习进程写入各 run 的 `tensorboard/`，不在 owner 重复计数。新增 manifest 的 `tensorboard` 字段记录路径、单位与必需性，不改变学习身份或原始 checkpoint。

在项目根目录运行：

```bash
.venv_terrain_cuda/bin/tensorboard \
  --logdir outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_optimized_20261022_01/tensorboard \
  --host 127.0.0.1 --port 6006 --reload_interval 5
```

浏览器打开 `http://127.0.0.1:6006`；远程 IDE 在“端口／Ports”面板转发服务器6006，并使用面板给出的本地地址。2026-10-23按用户要求，6006已改为仅加载上述当前run的`tensorboard/`，不再同时扫描history／fixed-far。以后默认只加载当次训练，切换run同步修改`--logdir`；仅在明确需要对照时加载多个run，历史events原位保留。[完整承接与预算](../experiments/t11_optimized_training_resume_2026-10-23.md)保持。6007可按需用于诊断回放，不重复启动占用同一端口的服务。

#### 精选 Pinned 视图

2026-10-21 已按用户要求，为当前停车／靠拢对照增加 10 个短名称 `Main/*` 标签，并生成 TensorBoard **Time Series 原生 Pinned** 链接；不是把普通标签分组命名为 Pinned。打开每个 `tensorboard/` 或 `tensorboard_replay/` 下 `pinned_url.txt` 中的地址，即可固定这 10 张图，并以 `^Main/` 筛选隐藏其他细节图。Text 的 `run/Pinned` 也有入口。Pinned 是浏览器视图状态，需用该链接打开；仅刷新原页面不会替用户修改已保存的置顶选择。链接默认当前服务 `127.0.0.1:6006`，使用其他端口时替换地址即可。

| 短名称（省略 `Main/`） | 观察内容 |
| --- | --- |
| `Return` | 本批已结束训练回合的平均原始回报 |
| `Success` | 本批已结束训练回合的成功比例 |
| `HOLD_park` | 停车训练组的 HOLD 比例 |
| `HOLD_move` | 靠拢训练组的 HOLD 比例，判断是否提前停车 |
| `Value_EV` | 更新前价值解释方差，越接近 1 通常越好，允许负值 |
| `KL` | PPO 末轮近似 KL，观察策略更新幅度 |
| `Park_S`、`Park_M` | 独立停车评估成功比例，S=随机采样，M=均值策略 |
| `Approach_S`、`Approach_M` | 独立 20 s 靠拢评估平均最大车距缩小量（m），正值为靠拢 |

只有原指标存在时才写别名；不把缺失指标补零，不把别的距离组、干预策略或完整回合评估混入这些固定曲线。原始详细曲线保留，清空 tag filter 可查看。当前两组回放各有 10 个重点标签，服务已加载；所有别名的数据点与原标签逐值一致。6 项日志检查通过，本次新增环境步／优化更新均为 0。

| 标签 | 含义 |
| --- | --- |
| `sampling/actual_base_steps`、`train/sampling_updates` | 实际采样进度、已完成学习批数 |
| `ppo/epoch_mean/*`、`ppo/epoch_last/*` | 每批 PPO 各轮均值与末轮：损失、熵、KL、裁剪、梯度 |
| `ppo/diagnostics/*` | 优势、回报目标、价值偏差／解释方差等学习信号 |
| `episode/*` | 本批已结束回合的原始／折扣回报、奖励分项、终止比例 |
| `train/hold_fraction`、`train/behavior/*` | HOLD、驻留及中断，依入口已有 JSON 指标提供 |
| `eval/short/*`、`eval/full/*` | 短程与完整回合独立评估，sample／mean 分开 |
| `evaluation_budget/*`、`summary/evaluation_actual_base_steps` | 单次及最终累计评估消耗，不混入训练横轴 |
| `run/*`（Text 面板） | 配置、命令、源码身份、单位及退出原因 |

T11 横轴为累计**实际训练底层环境步**：一个环境推进 0.2 s 算一步，四车不重复计数；终止时长与无效样本已消耗步数均来自 collector。数组 `/mean` 为该层数值均值，`/case_N` 保留逐初态／槽位数据；`success_rate` 的分母为所有评估初态，不能只除以已结束回合。无已结束回合时不补零回报或成功率；不定义的优势／解释方差不补零。不同折扣口径不能直接按原始回报排名。

每次同步立即 flush，退出时关闭 writer。重新打开 T11 run 时，从不可变 JSON 重建已记录曲线，以 TensorBoard restart 标记隐藏之前的投影，原 events／JSON 不删除；避免重复旧点，并恢复 JSON 已写而 events 未刷出的批次。学习恢复仍遵守原 checkpoint／物理预算校验，日志重建不替代恢复检查。旧转移数协议缺少实际步数的历史 JSON 不猜测横轴；新记录已补物理计数，旧入口的同 run 恢复从对应 update JSON 衔接该计数。

无需重新训练即可把已有 T11 JSON 转成曲线：

```bash
.venv_terrain_cuda/bin/python scripts/training_tensorboard.py \
  --run-dir outputs/runs/upstream_1km_curriculum/parking_lr3e4_seed23_10k_20261021_01
```

导出写入该 run 的 `tensorboard_replay/`，`replay_sources.json` 保存原始指标 SHA 和命令，不覆盖原 manifest、配置、JSON、checkpoint。已补建 `parking_lr3e4_seed23_10k_20261021_01` 和 `parking_exposure_seed23_10k_20261021_01`：各 25 批训练点、初始化与最终评估，可直接用上述监看命令对比。导出不增加环境步或优化更新。

历史入口 `train_skrl_mappo.py` 保留原生 SKRL TensorBoard，配置禁止非正写入间隔；`train_proxy_convergence.py` 保留现有标签但禁止关闭或静默缺失依赖。`train.py --backend smoke` 与 `train_local_goal_nmpc.py` 已补损失／进度 events。历史 SKRL／proxy 标签保留原时步或 update 横轴，不与 T11 物理步混画。离线拟合不是物理步训练，后续执行时须记录独立 optimizer-step 曲线。

针对性检查：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv_terrain_cuda/bin/python -m pytest -q -o addopts= tests/test_training_tensorboard.py`。覆盖实时 flush、PPO 聚合、评估分母、恢复重建、随机数流不变及非有限指标报错；本次未运行新的学习试验。

### 历史训练配置示例

新训练配置中设置：

```yaml
experiment:
  name: exp_008_terrain3d
  output_layout: run
```

运行时显式指定 `run_id`：

```bash
.venv_isaaclab/bin/python scripts/train_proxy_convergence.py \
  --config configs/experiment/exp_008_terrain3d_weak_warmstart.yaml \
  --output-layout run \
  --run-name weak_warmstart_seed23_6m_lunar_crater_bc20 \
  --device cuda
```

训练脚本会直接在 run 目录下写入 `config/experiment.yaml`、`metrics/summary.json`、`metrics/eval_metrics.json`、`run_manifest.json`、TensorBoard events、图、GIF 和 `checkpoints/best.pt`。

TensorBoard：

```bash
.venv_isaaclab/bin/tensorboard \
  --logdir outputs/runs/exp_008_terrain3d \
  --port 6007
```

## 后续 Proxy 评估

优先使用统一 checkpoint 评估入口：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/exp_008_terrain3d_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20 \
  --skip-physx
```

该命令会写入：

```text
metrics/final_eval_proxy.json
metrics/checkpoint_status.json
run_manifest.json
```

如需只运行底层 proxy eval，可使用 `--run-dir` 将独立评估结果写回同一 run：

```bash
.venv_isaaclab/bin/python scripts/evaluate_proxy_policy.py \
  --config configs/experiment/exp_008_terrain3d_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20/checkpoints/best.pt \
  --device cuda \
  --num-envs 512 \
  --steps 160 \
  --run-dir outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20
```

带 `--run-dir` 时默认输出：

```text
outputs/runs/<experiment_id>/<run_id>/metrics/final_eval_proxy.json
```

如需诊断 checkpoint selection 或 eval seed 方差，可对同一 run 的多个 checkpoint 做 multi-seed sweep。该入口只写诊断 JSON，不覆盖标准 `final_eval_proxy.json`：

```bash
.venv_isaaclab/bin/python scripts/evaluate_proxy_checkpoint_seed_sweep.py \
  --config outputs/runs/<experiment_id>/<run_id>/config/experiment.yaml \
  --run-dir outputs/runs/<experiment_id>/<run_id> \
  --checkpoint ppo_timestep_012288.pt \
  --checkpoint ppo_timestep_013312.pt \
  --seeds 1023,2023,3023,4023 \
  --device cuda \
  --num-envs 1024 \
  --steps 320
```

默认输出：

```text
outputs/runs/<experiment_id>/<run_id>/metrics/checkpoint_seed_sweep/summary.json
outputs/runs/<experiment_id>/<run_id>/metrics/checkpoint_seed_sweep/<checkpoint>_seed<seed>_eval.json
```

如需诊断未通过 strict gate 的 episode 具体卡在哪个 success 条件，可运行 success-gate 逐 episode 诊断。该入口只做 failure analysis，不覆盖标准 `final_eval_proxy.json`：

```bash
.venv_isaaclab/bin/python scripts/diagnose_proxy_success_gates.py \
  --config outputs/runs/<experiment_id>/<run_id>/config/experiment.yaml \
  --checkpoint outputs/runs/<experiment_id>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --num-envs 1024 \
  --steps 320 \
  --seed 1023 \
  --run-dir outputs/runs/<experiment_id>/<run_id>
```

默认输出：

```text
outputs/runs/<experiment_id>/<run_id>/metrics/success_gate_diagnostics.json
```

该 JSON 包含整体 `summary`、按 `done_reason` 分组的 gate rate、timeout 的 final gate failure counts，以及每个环境的 episode row。若该诊断与历史 final eval 数字略有出入，应标注为 recheck/diagnostic，不替换原有 strict gate 记录。

## 后续 PhysX / Jackal 跟踪评估

优先让 `scripts/run_checkpoint_evaluation.py` 根据 `evaluation.high_fidelity_eval.trigger` 自动触发。需要手动运行底层 PhysX / Jackal tracking 时，使用 `--run-dir` 让高保真评估产物留在同一个 run：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_jackal_tracking.py \
  --config configs/experiment/exp_008_terrain3d_weak_warmstart.yaml \
  --checkpoint outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20/checkpoints/best.pt \
  --terrain strong_lunar_crater \
  --profile all \
  --controller-json outputs/runs/physx_jackal_tracking/flat_tuned_final/physx/metrics/flat_tuning_grid.json \
  --steps 660 \
  --sim-steps-per-control 4 \
  --run-dir outputs/runs/exp_008_terrain3d/weak_warmstart_seed23_6m_lunar_crater_bc20
```

平地调优：

```bash
.venv_isaaclab/bin/python scripts/evaluate_physx_jackal_tracking.py \
  --terrain flat \
  --profile all \
  --tune-flat \
  --steps 660 \
  --sim-steps-per-control 4 \
  --run-dir outputs/runs/physx_jackal_tracking/flat_tuned_final
```

带 `--run-dir` 时默认 PhysX / Jackal 路径：

```text
physx/metrics/tracking_summary.json
physx/metrics/flat_tuning_grid.json
physx/metrics/<terrain>_<profile>_timeseries.csv
physx/figures/<terrain>_<profile>_tracking.png
```

## Git 策略

`outputs/**` 已被忽略。为了保证结果可复现，需要跟踪：

- `configs/experiment/*.yaml`
- 训练、评估、整理脚本
- 实验 Markdown 文档
- `docs/references/output_management.md`

不要提交生成的 checkpoint、TensorBoard events、PNG、GIF 或 JSON metrics；除非某个报告明确需要少量 curated artifact。
