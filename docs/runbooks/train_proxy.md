# 当前 Proxy 正式训练操作手册

2026-10-25当前运行：[SAT三轮MPPI从零百万步训练](../experiments/t11_mppi_sat_long_training_2026-10-25.md)，run为`outputs/runs/upstream_1km_fixed_far/a1_seed23_mppi_sat_fresh_1m_20261025_01/`，20:18启动。全新网络／Adam／价值统计，初始计数0，学习上限1,000,000；独立评估调度最多86,400／上限88,800。启动前修复训练入口轮数覆盖，41项检查及四图／atlas、初始化审计通过。下面旧恢复预算和命令仅为历史。

当前TensorBoard地址 `http://127.0.0.1:6006`，仅加载本run `tensorboard/`，每次学习采样写events。查看命令：

```bash
.venv_terrain_cuda/bin/tensorboard --logdir outputs/runs/upstream_1km_fixed_far/a1_seed23_mppi_sat_fresh_1m_20261025_01/tensorboard --host 127.0.0.1 --port 6006 --reload_interval 5
```

本轮精确命令见run的`config/training_command.txt`、`launch.json`，没有`--resume-state`或任何import参数。准备目录为同族`mppi_sat_fresh_preparation_20261025_01/`；启动检查后本地等待脚本在结束时只触发一次总结，等待期间不调用模型。旧迁移准备`mppi_sat_long_preparation_20261025_01`已弃用，不可用其启动脚本替代本轮。


更新：2026-10-24。当前使用 `scripts/run_upstream_t11.py` 的固定月面模式；历史训练手册已[完整归档](../archive/formal_training_transition_2026-10-22/docs/runbooks/train_proxy.md)。当前运行以[状态](../current_status.md)和[active 配置](../architecture/planning_pipeline.md)为准，旧 exp／课程命令不作为默认启动命令。

## 当前运行与监控

run：`outputs/runs/upstream_1km_fixed_far/a1_seed23_crater_reward_1m_recovered_20261024_01/`。从父run第5批／62,628步完整恢复；原100万步授权已消耗25,551步，剩余974,449步，新分支累计上限1,037,077。丢弃的12,794步另记账，不免费重放。评估总上限88,800步，已继承2,400步，剩余86,400步。A1／seed23、4 owner×4环境、CPU策略／PPO、CUDA MPPI512×2，训练奖励和环境保持。见[故障修复与恢复](../experiments/t11_crater_reward_recovery_2026-10-24.md)。

本地 TensorBoard：`http://127.0.0.1:6006`。当前服务只加载上述 run 的 `tensorboard/`，默认不展示历史训练。后续切换训练时同步切换服务的 `--logdir`；需要对照时才加载多个 run。远程 IDE 在“端口／Ports”面板转发服务器6006端口，然后使用该面板给出的本地地址；若本地6006被占用，以实际转发端口为准。

```bash
tail -f outputs/runs/upstream_1km_fixed_far/a1_seed23_crater_reward_1m_recovered_20261024_01/logs/console.log
```

日志目录为该 run 的 `tensorboard/`。服务已运行时直接访问；需要重新启动时，仅加载当前目录：

```bash
.venv_terrain_cuda/bin/tensorboard \
  --logdir outputs/runs/upstream_1km_fixed_far/a1_seed23_crater_reward_1m_recovered_20261024_01/tensorboard \
  --host 127.0.0.1 --port 6006 --reload_interval 5
```

学习曲线横轴为本run累计实际底层环境步，从62,628恢复；原49,871步的初始化评估继承、不重跑。独立评估按冻结策略的学习步显示，评估消耗另计；新增预算实际消耗为当前分支步数−49,871＋12,794。实时监控需确认 events 的 step 随实际训练增长，不能只确认服务存活。已结束的事件文件继续保留供回放。

## 启动与恢复边界

本次实际命令与环境变量见[恢复记录](../experiments/t11_crater_reward_recovery_2026-10-24.md)及run的`config/training_command.txt`、`launch.json`、`effective_configuration.json`。四图新版atlas和新奖励已完成100步无学习准入。使用 `.venv_terrain_cuda`；不要用旧 exp 的 `train_proxy_convergence.py` 命令接替当前链。

恢复前核对用户授权的剩余预算、源码／配置／地图／atlas 身份及完整 checkpoint 与 owner 分片；本轮每批保存。保持恢复时环境、RNG、GRU、规划器和优化器状态一致；若行为语义改变，明确新旧 run 边界，不能仅放宽身份检查直接续训。

上一轮计算等价切换证据保留于优化续训记录。本次修复批次有效性入口，训练配置与算法保持；独立run完整继承父run保存点，显式迁移审查后的源码身份和预算元数据，恢复检查未放宽。`--budget` 是累计步上限，恢复已有进度时不可将剩余步数直接作为累计上限。

每次新启动或恢复都保存 TensorBoard events，并记录配置、横轴和恢复位置。已有服务可复用，日志必须使用独立 run 目录。不能静默关闭日志，不能把新观测训练接在旧曲线上。

## 结果与后续

5万步短训及既定评估已结束，短窗存活改善但平均队形仍扩大；用户现已授权本轮新增100万步。启动后仅短暂模型检查，之后由TensorBoard持续记录、本地脚本在结束后触发一次独立模型总结。低成功率、地形失败和性能不足作为结果，不追加旧 T11 矩阵准入。影响运行、数据正确性或比较有效性的具体缺陷应先修复。预算结束后不自动追加长训或启动 A0/A2/A3 队列。

完整任务成功、结束原因及独立评估优先于 reward 和 HOLD 曲线。新修复只验证受影响行为；消融等主方案取得可用结果后安排。产物保存规则见[输出管理](../references/output_management.md)，T15 工作见[治理清单](../t15_governance.md)。
