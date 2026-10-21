# T11主机OOM修复与完整保存验证

**结论：修复已完成，推荐本机使用四进程16环境。** 连续两批采样、5轮PPO、完整checkpoint保存及新进程重载均通过。含保存平均吞吐2.656团队转移/s；训练树PSS峰值7.43GiB，整机可用内存最低5.21GiB，训练树swap为0。最新完整恢复点为第17批、10,752转移/85轮PPO。验证后已退出，后续方案队列未启动。以下保留逐步容量调整过程。

2026-10-17按用户“按顺序完成修复”执行。目标是消除完整环境checkpoint汇集造成的内存峰值、清理已消费的训练/历史数据，并在环境保存失败时保留独立学习恢复点。先保留六进程、48环境、128宏步及5轮PPO；是否降低并行规模由完整批次内存实测决定。

修复证据run：`outputs/runs/upstream_scale_t11/oom_fix_20261017_01/`。最初48环境容量尝试run：同级`oom_fix_owner48_20261017_01_a1/`。原OOM运行、失败临时文件、原第14批checkpoint全部保留。本次不启动A0/A2/A3队列。

## 已确认缺陷与实现

- 原`ProcessUpstreamRollout.state_dict()`同时向六个owner请求完整状态；环境`state_dict()`整体深复制，再通过进程管道汇集至主进程。新保存入口逐owner直接写盘，只回传文件名、大小与SHA，避免完整环境跨进程复制。加载也逐owner在本地读盘。
- 新格式`t11_owner_shards_v1`的主文件只含中央学习状态及分片清单。所有分片完成后才发布主文件和校验侧录；同目录临时文件、fsync及独占硬链接防止覆盖既有恢复点。旧单文件checkpoint仍可读取。
- 每次完整保存先发布`t11_learner_only_v1`文件，保留策略、Adam、中央RNG及学习计数。该文件只能显式`--import-learner`，环境和GRU重置；完整resume拒绝它，不能将部分保存冒充完整恢复。
- 环境`_records/_episodes`在collector取得独立transition副本后清理，直接tick诊断保留原历史语义；两者原本就不属于环境checkpoint状态。owner发送/恢复后释放临时快照引用。
- 交互文件改为流式pickle/gzip写入和读取，保留原`gzip_pickle5`格式与校验。交互落盘后释放batch、records和skrl RandomMemory；恢复/导入后释放加载对象，避免旧快照驻留整个训练过程。
- 保存/恢复的修改不改变任务、奖励、动作、128步GRU、GAE、5轮PPO或MPPI512×2。环境原有`state_dict()`仍提供独立快照，未将公开接口改成共享可变引用。

完整恢复点迁移时需一起保留`update_NNNNNN.pt`、同名`.json`和`update_NNNNNN.owners/`目录。清单使用相对路径，可整体迁移；读取时先校验所有分片，再让owner逐片恢复。仅学习备份为独立的`update_NNNNNN.learner.pt/.json`，不依赖环境分片。

## 针对性测试

32项测试通过，0失败/跳过：新增/迁移/进程/准入17项（56.92s），环境/采样/会话15项（59.99s）。覆盖分片重载后下一批观测与完整PPO参数精确一致、损坏分片拒绝、写入失败仍可导入学习状态、历史清理不改变转移、流式交互格式兼容及旧环境恢复。证据为`metrics/targeted_tests.xml`、`environment_tests.xml`及同名日志。

```bash
.venv_terrain_cuda/bin/python -m pytest -q -o addopts= \
  tests/test_upstream_checkpoint.py tests/test_upstream_process_rollout.py \
  tests/test_upstream_skrl_migration.py tests/test_upstream_signal_admission.py
.venv_terrain_cuda/bin/python -m pytest -q -o addopts= \
  tests/test_upstream_ablation_learning.py tests/test_upstream_training_session.py \
  tests/test_upstream_environment.py
```

## 完整批次验证

48环境容量尝试已停止：初始分片恢复点保存3.94s，但采样早期整机可用内存已降至约2.5GiB，低于预先设定的3GiB余量目标；在进入PPO前主动终止本次自建进程组，没有等到OOM。未完成完整批次，不能记为48环境容量通过。证据为`metrics/owner48_capacity_decision.json`及`validation_exit.json`（主动SIGTERM）。

按计划第四步，第二次容量验证配置为6owner×4环境=24环境，仍为128宏步、5轮PPO，显式协议`owners24_v1`；每批3,072团队转移。第二证据run为`outputs/runs/upstream_scale_t11/oom_fix_20261017_02/`，实际训练run为同级`oom_fix_owner24_20261017_01_a1/`。从原两环境第14批3,584转移的学习状态初始化新环境，最多两次完整更新后退出，100k总上限不变。并行规模变化单独记录，不拼接为原48环境协议的同状态对照。

24环境协议与队列筛选补测8项通过（含与前述重复6项），合计34项不同测试通过。初次队列测试把既有失败异常误写为SystemExit，改为实际RuntimeError后通过；原失败日志保留。修复了自动队列可能选中新一批仅学习备份的问题：只选择完整数字update文件名。

24环境第15批已完整完成：3,072有效转移、5轮PPO，分片保存5.28s，完整恢复进度为6,656转移/75轮PPO。全部分片SHA已验证。分阶段复核确认保存阶段出现约10.36GiB PSS峰值、整机可用内存最低约2.59GiB（PPO阶段约9.20GiB/3.43GiB），之后训练树swap约541MiB，故未将该配置判为足够余量；第二批采样期间主动停止，保留第15批完整状态。见第二证据run的`owner24_capacity_decision.json`及阶段原始数据。

按计划继续降低进程固定开销，最终采用4owner×4环境=16环境，仍128宏步、5轮PPO，显式`owners16_v1`；每批2,048转移。第三证据run为`outputs/runs/upstream_scale_t11/oom_fix_20261017_03/`，实际run为同级`oom_fix_owner16_20261017_01_a1/`。从刚成功保存的第15批学习状态导入，不丢弃这次有效更新；新环境/GRU重新初始化。已完整执行第16、17批后退出，达到10,752累计转移/85轮PPO，新进程加载验证通过。协议/队列补测9项通过，合计35项不同测试通过。

命令保存在证据run的`metrics/launch.json`；`logs/validation_runner.py`启动真实入口，`logs/instrumented_train.py`只标记阶段。`metrics/hardware_raw.jsonl`每秒记录完整进程树CPU、PSS/RSS、swap、整机MemAvailable、NVML利用率/显存/功率。若MemAvailable连续3次低于1,280MiB，则停止本次验证以避免再次主机OOM，此时记失败而非通过。

完整验收要求两批均完成采样、PPO和完整checkpoint；随后退出并由新进程读取分片，核对策略/Adam/GRU/RNG和后续有效转移。3GiB整机可用内存作为初始容量余量目标，不能只根据短采样或初始保存认定48环境稳定。

## 最终实测与恢复点

最终配置：i7-10750H、RTX2060 6GiB、主机约15.38GiB；4owner×4环境，每owner2线程，skrl CPU学习、CUDA MPPI512×2，128宏步/批、5轮PPO。每批2,048有效团队转移。

| 批次 | 采样 | PPO | checkpoint保存 | 含交互保存的总时间 | 吞吐 | CPU均值 | GPU均值 | PSS峰值 | 整机可用内存最低 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 734.57s | 11.68s | 2.91s | 751.20s | 2.726/s | 560.10% | 10.81% | 7.13GiB | 5.39GiB |
| 17 | 774.65s | 11.77s | 2.72s | 791.21s | 2.588/s | 568.72% | 10.74% | 7.43GiB | 5.21GiB |

两批合计4,096有效转移、10轮PPO，完整窗口1,542.41s，2.656转移/s；CPU加权均值564.52%（100%=一个逻辑核）、GPU10.77%，GPU采样峰值74%，显存峰值3,274MiB（约3.20GiB），平均功率40.57W。训练树swap始终0，整机`pswpout`增量0；整机少量swap-in来自既有系统页，不能混称训练发生换出。第二批PSS峰值比首批高约0.30GiB，仍保留超过5GiB整机可用内存。

原48环境OOM记录的PSS峰值约12.49GiB，新配置为7.43GiB；差异同时包含保存实现修复与环境/进程缩减，**不能归为同配置的纯优化倍率**。原失败运行没有完成checkpoint，因此不能拿其采样吞吐冒充完整训练吞吐。24环境曾完成保存，但未达到本次内存余量目标；其最大峰值经阶段核对位于保存，早期进度中的PPO归因已更正。

完整恢复点：`outputs/runs/upstream_scale_t11/oom_fix_owner16_20261017_01_a1/checkpoints/update_000017.pt`，配套同名JSON和四个owner分片。新进程重载约9.97s，策略/Adam的186个状态叶节点一致，GRU与中央RNG精确一致；再执行1宏步，16/16转移有效、观测与log-prob有限。该恢复探针不更新策略、不写入训练计数。恢复窗口PSS峰值5.34GiB、整机可用内存最低7.42GiB，swap为0。

最终机器可读结果在第三证据run的`metrics/summary.json`、`reload.json`、`validation_exit.json`与`reload_validation_exit.json`，两个进程退出码均为0，未触发低内存保护。原始硬件采样、阶段边界、命令、生产代码与测试源码均已归档。一次汇总脚本在恢复进程结束前读取结果而报FileNotFoundError，恢复完成后重跑汇总成功；该辅助时序错误不计为恢复失败或测试通过证据。

本轮达到结束条件：35项不同回归测试通过、两批完整保存通过、新进程重载通过；不追加训练预算或后续方案。恢复命令如下（本次未执行此长期续跑命令）：

```bash
.venv_terrain_cuda/bin/python scripts/run_upstream_t11.py \
  --config configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml \
  --run-id oom_fix_owner16_20261017_01_a1 --representation A1 \
  --mode signal --authorize-signal100k \
  --readiness outputs/runs/upstream_scale_t11/oom_fix_20261017_03/metrics/readiness.json \
  --seed 23 --budget 100000 --samples 512 \
  --num-envs 16 --env-owners 4 --owner-cpu-workers 2 --rollout-steps 128 \
  --sampling-protocol owners16_v1 --trainer-backend skrl --policy-device cpu \
  --resume-state outputs/runs/upstream_scale_t11/oom_fix_owner16_20261017_01_a1/checkpoints/update_000017.pt
```

这是Proxy训练与生命周期验证，不构成Isaac/PhysX或任务成功率验收。
