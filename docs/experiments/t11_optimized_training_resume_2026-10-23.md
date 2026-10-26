# 2026-10-23：优化采样源码正式续训

用户于10月22日要求停止原训练并采用[已验证优化](t11_sampling_optimization_2026-10-22.md)。本次保持学习／行为语义，完整继承策略、Adam、价值归一化、GRU、各环境、规划热启动与RNG；不重新初始化学习，也不增加预算。

## 停止与源码切换

父run：`outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_remaining_20261022_01/`。

10月22日23:55向主PID2056784发送SIGTERM，沿用现有正常退出处理，完成第23批采样及PPO后保存。停止边界为**261,149实际学习底层步、23采样批、115轮PPO**；主进程与四owner均退出，未遗留未提交采样。完整checkpoint为 `checkpoints/update_000023.pt`，头文件与四owner分片SHA验证通过。

停止后才应用优化补丁：10个生产源码文件与已验证candidate的SHA逐项一致，脚本、配置及地形源码保持。复用此前104项专项检查及1,178步同状态性能面板，不重复进行无关准入或额外性能采样。物理模型、观测、动作、奖励、K512×2、40步预测及执行频率不变。

切换证据根目录（下文记为 `S`）：

```text
outputs/runs/local_obstacle_input/sampling_optimized_training_switch_20261022_01/
```

停止与源码验证分别见 `S/metrics/old_stop_verification.json`、`S/metrics/source_activation.json`。原run、checkpoint、原始源码包与events全部保留。

## 显式完整状态承接

新run：

```text
outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_optimized_20261022_01/
```

run ID保留发起切换的10月22日，实际恢复发生于10月23日。

固定远距训练入口要求本run自有checkpoint、源码／配置身份完全一致，不能直接拿旧run的checkpoint跨源码恢复。因此在输出目录内执行一次显式承接：验证父源码等于冻结baseline，当前源码等于经过验证的candidate且差异仅这10个文件；用实际4×4优化环境和训练器按原公开接口加载完整状态，导出新run自有完整checkpoint。

中央policy／Adam／归一化／hidden／RNG／计数及全部owner导出状态逐值比较，共18,915,501个值全部一致，允许差异为0；包含私有格、排队快照、规划warm／previous与独立generator。只重建原本不入checkpoint的派生缓存。新身份仅更新真实 `source_snapshot`，记录父checkpoint SHA和逐文件变更；随后通过原训练入口的标准 `--resume-state` 恢复，**不修改或放宽原源码恢复检查**。

首轮承接已成功加载并导出状态，但辅助配置比较把JSON列表与内存元组当成不同类型而退出。修正仅规范化配置JSON的表示，热状态比较继续严格；原日志和失败JSON保留，利用已导出的checkpoint完成复验及记录，没有重跑环境或覆盖checkpoint。

承接过程新增环境步／PPO均为0。工具及证据在 `S/config/prepare_optimized_resume.py`、`S/config/finish_export.py`、`S/metrics/migration_verification.json`；新run的 `metrics/migration.json` 记录完整谱系，`metrics/migration_summary.json` 记录恢复边界，`config/source_and_configs.tar.gz` 归档实际源码。

## 预算与评估

| 项目 | 数值／口径 |
| --- | --- |
| 原soft-hazard训练已消耗 | 502,369步 |
| 障碍输入分支继承进度 | 261,149步 |
| 本次启动时剩余学习预算 | **1,236,482步** |
| 新run的CLI `--budget` | **1,497,631累计步**，因collector已继承261,149步，不能再传“剩余数”作为累计上限 |
| 合计学习上限 | 502,369＋1,497,631＝2,000,000步 |
| 继承评估 | 595实际步及60,000步初始化面板reservation，复制原结果／预算记录，启动时不重复初始化评估 |
| 独立评估上限 | 仍沿用原400,000步总授权；旧soft-hazard评估744步另保留，重复副本不重复计费 |

以后仍按分支累计学习步触发500k等既定评估。新TensorBoard使用独立run目录，学习横轴从继承的261,149步继续，旧学习曲线不拼接；继承的初始化评估仍绑定原step 0，并在迁移记录中标明来源。

## 启动与监控

保留A1／seed23、16环境／4owner、每owner双worker、CPU策略与PPO、CUDA MPPI、32决策／5轮PPO及1500 s完整回合。原地形／奖励准入证据复用，配置SHA和完整有效配置均核对，新的计算等价证据另外绑定到迁移记录。

实际命令保存在新run的 `config/resume_command.txt`、`config/resume_launch.json`，核心恢复参数为：

```text
--run-id a1_seed23_far_lunar_obstacle_optimized_20261022_01
--budget 1497631
--resume-state outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_optimized_20261022_01/checkpoints/update_000023.pt
```

TensorBoard复用 **http://127.0.0.1:6006**。日志目录：

```text
outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_optimized_20261022_01/tensorboard/
```

查看实时训练输出：

```bash
tail -f outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_optimized_20261022_01/logs/console.log
```

服务已运行时直接访问；需要重新打开服务时：

```bash
.venv_terrain_cuda/bin/tensorboard \
  --logdir outputs/runs/upstream_1km_fixed_far/a1_seed23_far_lunar_obstacle_optimized_20261022_01/tensorboard \
  --host 127.0.0.1 --port 6006 --reload_interval 5
```

实际启动于2026-10-23 00:10:07（Asia/Shanghai），主PID2198844，四owner为2198862–2198865。00:13:45验收时均存活，实际运行源码身份匹配，未生成failure JSON。第24批已完成15个采样段，新增5,182实际学习步，分支累计266,331步；新PPO批尚未完成。最近三个采样段为31.10／30.97／30.59步/s，仅作启动后短时读数，不替代同状态性能对照或长期吞吐。

TensorBoard HTTP 200，新events连续15点从261,472更新至266,331，文件合计321,357字节。进程、源码、实际采样和events检查保存在 `S/metrics/running_verification.json`。本次切换验收完成，正式训练继续按原累计预算运行。

00:18追加完整闭环核验：`metrics/update_000024.json`已保存，首个续训批新增11,247步、5轮PPO，累计272,396步／24批／120轮PPO。该批采样390.97 s，吞吐28.77底层步/s；学习指标均为有限值。126个结束回合均为`terrain_failure`，未出现集合成功。第25批已继续采样，events推进到274,099，未出现failure JSON。该批完成时剩余1,225,235学习步；证据为 `S/metrics/first_update_verification.json`，切换manifest已标记完成，训练继续运行。

11:47监控维护：用户反馈页面打不开、历史run过多。服务器本机原HTTP正常，但原服务加载18组历史／当前记录；将6006的TensorBoard进程1653401替换为2471782，仅指定本run的`tensorboard/`，训练主进程2198844持续运行。新服务HTTP 200，`/data/runs`仅一个run，曲线API实际step从1,187,539推进到1,187,925。启动命令／旧run列表与验证记录保存在本run的`logs/tensorboard_current_20261023_114632.json`及`logs/tensorboard_current_20261023_114632_verification.json`，服务输出为同前缀`.log`。服务器侧可用性已确认，远程浏览器需使用IDE转发6006后的地址。后续默认仅展示当次训练，历史events保留供按需回放。
