# 四进程16环境长期训练恢复

最新状态（2026-10-18 13:05）：按用户要求停止全部训练、owner和自动队列，释放资源。A1完成100,000转移，第61批保存成功；A0最新完整恢复点为第39批/79,872转移，主文件及四个分片SHA均已核验；第40批未提交采样不计入可恢复学习进度。A2/A3未启动。停止证据为suite的`metrics/user_pause_20261018.json`、`user_pause_verification_20261018.json`；原始数据与checkpoint不删除，后续需用户明确恢复。

2026-10-17按用户最终选择恢复训练：四进程、16环境、128宏步、5轮PPO。直接复用[OOM修复验证](t11_checkpoint_oom_repair_2026-10-17.md)的完整恢复点和生产实现；启动前核对源码/准入身份一致、主文件及四个owner分片SHA，未重复已经通过的35项测试和两批完整保存验证。

## 训练与预算

- A1 run继续使用`outputs/runs/upstream_scale_t11/oom_fix_owner16_20261017_01_a1/`。
- 监督suite新增为同级`oom_fix_owner16_20261017_01/`，后续A0/A2/A3同名前缀分别建独立run。
- 从`checkpoints/update_000017.pt`及配套分片恢复：10,752有效团队转移、17批/85轮PPO。完整保留策略、Adam、环境、GRU、规划器和RNG，不做冷启动或仅权重导入。
- 采样协议`owners16_v1`：4owner×4环境，每owner2线程；128宏步/批，即2,048团队转移/批，每批5轮PPO。
- 实际后端skrl MAPPO，CPU策略/PPO，CUDA MPPI-512两轮，集中GPU服务关闭。
- A1继续至原累计100k（尚余89,248转移），之后A0→A2→A3分别按相同配置各100k。原完整矩阵评测保持暂停。

按已验收两批的含保存吞吐2.656转移/s估算，A1剩余约9.3小时；只是参考，长时间训练按真实日志更新。首次续训更新为第18批。

## 启动及生命周期

监督suite的`config/producer/`冻结与恢复点一致的scripts/source/configs，manifest记录完整源码指纹、父checkpoint SHA、采样设置和监督脚本SHA。新日志均独占写入，不覆盖原容量验证或学习数据。

使用官方四方案队列启动，精确命令和PID在suite的`metrics/queue_launch.json`及`launch_A1_*.json`：

```text
scripts/run_upstream_t11_signal_suite.py
--config configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml
--suite-id oom_fix_owner16_20261017_01
--readiness <项目>/outputs/runs/upstream_scale_t11/oom_fix_20261017_03/metrics/readiness.json
--authorize-signal100k --sampling-protocol owners16_v1
```

队列对已有A1自动选择第17批完整checkpoint并传入`--resume-state`。未设置`--updates-this-invocation`，因此不再在两次更新后退出。若A1错误退出，后续方案不启动。

沿用通过验证的逐owner分片保存、独立学习备份和批次/历史释放逻辑。后台监控每5秒记录完整进程树CPU、PSS/RSS、swap、整机可用内存和GPU指标；仍使用此前低内存保护：连续3次MemAvailable低于1,280MiB时停止本次训练队列。观测数据写入suite的`metrics/hardware_raw.jsonl`；日志为`logs/suite.log`和`logs/A1_*.log`。

本次无生产算法修改；仅新增运行监督产物和受影响文档。启动后已确认正式队列选择正确恢复点，四个owner实际运行。第18批首16宏步产生256/256有效团队转移，累计采样11,008，观测和log-prob有限、最大实际线速0.489m/s；含恢复的首段墙钟82.95s。完整checkpoint计数仍为10,752，待本批更新后增加。

首段后的实时采样中进程树PSS约5.78GiB、整机可用约6.99GiB、训练树swap为0，CPU约568%；这些是启动窗口观测，不代替完整批次资源峰值。四进程训练持续运行，尚未提交新的第18批更新。运行辅助脚本Ruff `F821`、本次文档链接及差异检查通过。
