# 六进程48环境skrl训练重启

最新状态（2026-10-17 15:00检查）：运行已在14:48:31因主机内存OOM停止，发生于第15批更新后的checkpoint保存阶段。下面启动与首段采样结果仍有效，但不能视为完整训练已稳定运行。失败详情见文末。

2026-10-17用户明确要求采用最高性能面板的六进程、48环境配置。旧两环境训练和监督队列已停止，新训练使用已验收的owner执行框架。此次目标是实际启动扩批学习，并核实有效采样、进程数量和硬件负载；原有大规模评测仍暂停。

## 配置与学习接续

| 项目 | 本次设置 |
| --- | --- |
| 采样 | 6个常驻owner，每个8环境/32车，每owner 2条计算线程；合计48环境/192车 |
| 规划 | MPPI-512、两轮、CUDA，直接由owner调用，集中GPU服务关闭 |
| 学习 | skrl 2.1.0 MAPPO，CPU策略/PPO，128宏步/批、5轮PPO |
| 批量 | 6,144团队转移/批，原两环境为256；更新频率因此改变 |
| 预算 | 累计最多100,000有效团队转移，继承已有计数；尾部不足48不超采 |
| 协议 | `owners48_v1`，独立run，旧配置默认仍为`legacy2` |
| 顺序 | A1继续，之后A0/A2/A3使用相同扩批参数 |

旧run最新完整恢复点为`skrl_signal100k_seed23_20261017_01_a1/checkpoints/update_000014.pt`：3,584有效转移、14批/70轮PPO，SHA256 `a183aaa326e58d28efdcccdd4360c35bc35d551f98130ee240aa7501d4578bc7`。停止凭证保存在旧suite的`metrics/stop_for_owner48.json`，旧结果不覆盖。停止时未提交的采样不计作已完成更新。

通过公开的`SkrlUpstreamTrainer.import_learner_state`继承策略、Adam、更新/转移计数与中央Torch随机数状态。48个环境、各自规划器/通信状态和GRU全新初始化，环境种子按原owner规则从23开始分配。初始化后即保存包含所有owner状态的恢复点；后续用原`--resume-state`完整恢复，不重新进行学习导入。

新协议保留任务、奖励、动作、规划/安全参数。其每次更新的数据量与原两环境不同，A1也带有历史两环境学习阶段；应报告混合来源，不将本次直接解释为同一采样协议下的纯四方案对照。此次选择由用户明确授权，不再以旧两环境准入规则阻止启动。

A1无无效转移时预计停在99,968（剩32额度）；其他从零开始的方案预计99,984（剩16额度）。准入仍要求100k授权和有效配置/证据；不授予500k或2M预算。

## 实现与验证

- 训练入口新增显式`--sampling-protocol owners48_v1`，核对6owner/48环境/2线程/128宏步。
- 新增`--import-learner`，与完整resume/import互斥；新run记录父checkpoint哈希与环境重置。
- 剩余采样长度按可容纳的完整向量步向下取整，避免100k非48整数倍导致超采或拒绝启动。
- 四方案队列支持同一协议参数；失败退出时停止后续方案。
- 13项相关测试通过，0失败/跳过，22.14s；含从真实已更新模型向新owner群组导入Adam/参数、重置GRU、再完成真实skrl更新及恢复校验。

```bash
.venv_terrain_cuda/bin/python -m pytest -q -o addopts= \
  tests/test_upstream_signal_admission.py tests/test_upstream_process_rollout.py \
  tests/test_upstream_skrl_migration.py \
  --junitxml=outputs/runs/upstream_scale_t11/skrl_owner48_seed23_20261017_01/metrics/entry_validation.xml
```

## 运行与监测

Suite：`outputs/runs/upstream_scale_t11/skrl_owner48_seed23_20261017_01/`；A1为同级同名加`_a1`。源码与配置冻结于suite的`config/producer`，manifest保存指纹、参数和辅助脚本哈希。

关键启动参数如下，完整命令与PID在suite的`metrics/launch.json`：

```text
--representation A1 --mode signal --authorize-signal100k
--seed 23 --budget 100000 --samples 512
--num-envs 48 --env-owners 6 --owner-cpu-workers 2 --rollout-steps 128
--sampling-protocol owners48_v1 --trainer-backend skrl --policy-device cpu
--import-learner <原A1 run>/checkpoints/update_000014.pt
```

监控覆盖中央学习进程、六个owner及其辅助子进程，每500ms记录CPU、RSS/PSS/USS、显存和GPU利用率。CPU100%是一个逻辑核；进程树RSS会重复统计共享页，判断总占用优先看PSS。GPU为整卡NVML口径。包装入口支持multiprocessing spawn，子进程不会递归执行训练main。

前3个扩批完整更新（15/16/17）分别写入A1的`metrics/hardware_update_*.json`，原始数据为`hardware_raw.jsonl`；阶段耗时与学习结果仍由正式入口写入。监控完成不停止训练。当前已确认六个GPU计算进程启动，首个128宏步批次正在采样；完整更新监控结果未完成，不能以先前两环境数据填报。

## 首段真实采样结果

首16宏步完成768/768有效团队转移，用时181.18s（含本次学习导入与初始快照保存），约4.24转移/s。观测与log-prob有限，最大实际线速0.480m/s。相较上一两环境运行约1.48转移/s明显提高，但这次是新环境初始阶段，不能当作同状态严格加速比。

采样窗口进程树CPU平均850.3%（约8.5个逻辑核）、GPU平均14.8%，PSS峰值10.30GiB，整卡显存峰值6,076MiB（约5.93GiB，接近本卡6GiB容量）。硬件监控继续运行；这组数据尚不包含首次6,144转移的PPO更新开销。

已实际读取新初始checkpoint：6个owner状态、每owner8环境、GRU形状`[48,4,128]`且为零，策略与Adam相对旧第14批逐张量一致。新checkpoint SHA256为`1927a8645181eb8eb3bdc5ca3c53398846ccf79e7b0fb797ea0b431c36a99c8e`。原始核验与首段资源结果保存在suite的`metrics/first_owner_collection.json`。

本轮重启出口已满足：旧队列停止、六进程48环境实际采样、有效数据和初始恢复状态均已确认。训练继续；完整第15–17批监控仍在后台收集。13项针对性测试、修改脚本Ruff `F821`、文档新增链接与差异检查通过。

## 15:00检查：checkpoint保存阶段主机OOM

第15批采样128宏步/6,144有效转移耗时1,843.84s（约30.73分钟），5轮PPO耗时25.24s；`update_000015.json`和交互快照均已落盘。随后创建的`checkpoints/update_000015.tmp`为0字节，尚无正式checkpoint和SHA侧录。系统日志在14:48:31明确记录`Out of memory: Killed process 3020764 (python)`，主训练退出码-9。属于主机内存耗尽，不是已证实的CUDA显存分配错误。

监测记录到进程树PSS峰值12,790MiB（约12.49GiB），整机内存约15.38GiB；峰值不包含其他系统进程，采样也不能捕获所有瞬时分配。最后一次监测读取间隔因系统压力拉长至约190s。错误发生在完整owner状态汇集/序列化的checkpoint保存阶段，具体各份拷贝的贡献尚未单独剖析。

主进程、六个owner、监督队列和监控均已退出；A0/A2/A3未启动。累计采样9,728不等于可恢复学习进度：最新直接可恢复状态仍是第14批/3,584转移，第15批权重没有成功保存。前3次完整更新硬件汇总未完成，不能记为通过。

系统失败日志保存在suite的`logs/oom_20261017_144831.log`，检查汇总为`metrics/oom_status_20261017_1500.json`。原始采样、更新指标、交互快照和失败临时文件均保留。本次状态检查未启动额外训练。后续应先降低保存阶段内存峰值（释放已保存批次数据、按owner分块保存），验证一次完整保存/恢复后继续六进程训练；此前8–10小时剩余估计失效。
