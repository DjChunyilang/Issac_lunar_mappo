# T06：Torch 批量 MPPI 基础验收

日期：2026-10-10。状态：CPU 基础契约与简单局部执行已验收；CUDA延期，正式训练暂停。依据[实施计划](../archive/implementation_plan_t05_t07_snapshot_2026-10-10.md)，在[T05](t05_planner_baselines_validation_2026-10-10.md)完成后推进。

共享Torch内核按B、K张量计算，只沿15个时间步和优化轮次循环。状态和随机数按(env_id,agent_id)独立保存；原始高斯扰动用于重要性修正，轮速／幅值／变化率映射后重新预测。最终输出重新复核，全部采样无效及超时进入有原因的旧计划／刹车回退，过期直接刹车。

Run：outputs/runs/t06_mppi_base/acceptance_seed23_cpu_20261010_01。命令：

```bash
OMP_NUM_THREADS=1 .venv_isaaclab/bin/python -m pytest tests/test_mppi.py -ra --disable-warnings
OMP_NUM_THREADS=1 .venv_isaaclab/bin/python scripts/diagnose_planners.py   --config configs/experiment/t06_mppi_base.yaml --planner mppi_base_v1   --samples 128 --steps 100   --output outputs/runs/t06_mppi_base/acceptance_seed23_cpu_20261010_01/metrics/local_loop_seed23.json
```

16项通过、0失败、0跳过（3.27 s），覆盖NumPy float64运动参考、解析代价和原始重要性修正、单条／批量／换序、独立RNG、快照恢复、HOLD／过期、全无效采样、超时，以及2环境×4车真实环境批调用与异步reset。512／1024／2048均实际执行CPU规划，逐项耗时在metrics/tests_retry1.xml，未以替身或skip代替。非NMPC工厂禁止调用acados仍能运行。

K=128、seed=23的给定合法平坦局部观测闭环终点误差0.196731 m，低于0.2 m。实际传感链集成由T07另验；此结果不代表完整集合任务、GPU吞吐或规划器排名。

初轮15项通过、1项测试返回字段读取错误；修正为MacroTransition.records后重跑，日志和JUnit保留。另补强全NaN采样必须回退，不允许将初始化均值当作新解。静态检查在T07联合收尾时记录。

算法依据[Williams等原论文](https://arxiv.org/abs/1707.02342)，本实现的限幅映射、硬复核和回退为独立工程约束。T07现可接入地形与邻车。未生成训练checkpoint。

T07收尾后，硬样本复核统一使用按B/K张量计算的只读查询；最终选中轨迹的物理代价按实际状态／控制批量计算，不重复施加控制映射。最终24项基础MPPI测试全部通过，覆盖极端分数、独立种子、未安装求解器、请求快照、实际选中代价和旧计划复核通过／拒绝；其余地形和诊断测试见[T07最终576项联合回归](t07_mppi_terrain_neighbor_validation_2026-10-10.md)。512/1024/2048在最终版本再次真实调用，profile_*_final.json记录每档配置、CPU资源和耗时；这是CPU正确性记录，不是规划器排名或GPU推断。

最终K=128局部误差仍0.196731 m，末轮采样与物理速度另存到local_loop_final_samples/。HOLD分支不保存旧采样冒充新采样。原始失败和16项阶段出口记录保留。

最终单Torch线程、B=1、给定稠密局部观测的CPU调用记录：K=512/1024/2048规划耗时4.874/9.595/17.429 s，进程峰值RSS约278.7/276.7/282.7 MiB。数值含最终CPU复核与导出，CPU诊断截止关闭；不能视为0.2 s实时周期或公平性能评测。后续CUDA吞吐另验。

整数身份收尾后基础MPPI25项重跑通过；合法Python int与NumPy integer产生同一RNG序列，Python整数输入保持原轨迹逐值一致。证据在metrics/rng_identity_final.xml和共享T07的rng_python_reference_equivalence.json。577项独立相关测试的组成和源码边界见T07记录。
