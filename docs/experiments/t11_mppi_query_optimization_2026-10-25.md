# MPPI查询合并、候选回传与迭代次数对照

后续：[从零百万步启动审计](t11_mppi_sat_long_training_2026-10-25.md)发现T11入口会把配置轮数覆盖为2，已修复。本文闭环实验直接设置kernel轮数，三轮结果仍有效；本轮新训练及评估已按YAML运行三轮。

日期：2026-10-25，Asia/Shanghai。承接[碰撞算法对照](t11_collision_algorithms_2026-10-25.md)。用户要求按顺序实施：先合并软风险查询、减少候选回传，再在SAT上比较2／3／4轮迭代，最后按实际耗时决定感知或审查是否值得继续优化。

**已完成，采用SAT＋合并软查询＋精简回传＋3轮迭代，并省去审查中未使用的已知范围计算。** 同初态两组整段耗时106.3／133.9秒降到58.4／83.2秒，合计少用41.1%时间，约1.70倍吞吐；可达短目标保持10／10到达、0失败。

实际完成6,000／6,000诊断底层步：前8个分支4,800步，审查优化验证2个分支1,200步。6／12批冻结完整规划完成，另有1次输入校验阶段退出、0候选rollout；0学习步／0优化器更新，109项不同针对性检查通过，无失败或跳过。验收达到，诊断进程全部退出，不追加训练。

## 已实施的计算优化

普通障碍和中坑仍采用原距离场、416个覆盖点和原评分公式。`GridTerrainBatch.obstacle_distances`让两个通道共享车身旋转、坐标变换和格索引；每个通道的取值、最小值、半径修正和截断顺序保持。不是把中坑改成碰到就罚固定分，也没有改变实际失败条件。

MPPI先在GPU按原稳定排序选出候选，再把待审候选和必要统计回传CPU。`record_samples=False`时不回传全部512条状态／控制／噪声；`run_upstream_t11.make_collector`使用此模式。独立规划器和`UpstreamPlannerPool`默认继续完整记录，诊断时可设`collector.env.pool.record_samples=True`。开关只影响证据记录，不进入规划热状态身份；省略记录时`last_samples=None`，诊断字段`full_samples_recorded=false`明确标注。

同时将已验证的已观测栅格SAT接入现有`upstream_grid.py`。只合并本车已揭示的大坑格，空间桶筛选后做矩形SAT；CPU最终轨迹检查和CUDA采样使用同一几何。它只替换大坑硬检查的正负号，不提供软风险距离，不读取完整场景圆目录。原始车身、环境失败、观测和奖励保持。

最终选用512采样×3轮迭代，其他目标／地形／噪声参数保持。SAT语义通过显式配置`configs/experiment/upstream_1km_mppi_grid_sat_v1.yaml`隔离，参数为`large_crater_query=observed_grid_sat_v1`，要求零额外规划余量。不设置该参数时继续原圆盘大坑查询，旧metadata不增加空字段；旧规划热状态不能静默载入SAT配置。实验先完整恢复旧状态，再施加记录在案的规划参数干预。

## 同输入验证与组件耗时

复用前轮10辆MOVE车的512条保存候选、41个预测位置，共209,920个位置。查询计时独立执行，CUDA同步、1次预热、5次计时，取中位数；不推进环境。

| 查询内容 | 修改前 | 修改后 | 说明 |
| --- | ---: | ---: | --- |
| 普通障碍＋中坑软查询 | 537.1 ms | 281.6 ms | 两通道共享变换与查格，约1.91倍 |
| 圆盘大坑方案的全部评分 | 816.1 ms | 560.6 ms | 大坑仍用原416点，只衡量软查询合并 |
| SAT方案的全部评分 | 554.3 ms | 298.6 ms | 原软查询→合并软查询，约1.86倍 |

所有软评分分项与大坑通过掩码逐值相同。接入源码后的SAT在209,920个位置与前轮原型逐值一致，选取／重排车辆行也一致。

另做6批同噪声完整规划，比较修改前源码和查询合并／精简回传：实际轨迹、控制、各评分分项、随机数和完整规划热状态逐值一致。这些完整规划调用部分与针对性测试同时进行，**只作为等价性证据，不用于报告加速比**。性能结论使用上表独立查询和后续无并发测试的闭环。

首个冻结规划准备尝试在输入校验阶段发现保存的信息时刻输入没有队列反馈历史，0环境步／0候选rollout。随后沿用前轮已记录的直接规划契约，补齐零长度`ActualControlHistory`并将该静态调用延迟设为0；实际闭环继续原队列和延迟。原错误保留于`logs/component.log`，完成证据在`logs/component_retry.log`，未覆盖。

## 闭环比较设计

从百万步训练`update_000082.pt`恢复owner_000、seed23，4环境／16车，其中10辆MOVE、6辆HOLD。每个分支完整恢复并逐值检查，保持策略、目标方向、地图、控制周期和原队列；不调用Actor重采样或优化器。

比较原SAT原型＋旧软查询／完整回传的4轮基线，以及源码SAT＋合并软查询／精简回传的4／3／2轮。每种运行原目标和同方向0.5米目标两个30秒片段，每分支600底层步。采样仍为512条，40步预测；只改变声明的计算实现或迭代数。

4轮计算优化的验收要求是轨迹、评分和实际控制保持；迭代减少会改变采样和轨迹，单列到达数、首次到达时间、原目标进展和失败数。原目标含无法容纳整车的位置，仍按前轮口径单列，不能以其0到达认定优化计算错误。

## 结果与第三项决策

| 同初态30秒闭环 | 原目标耗时 | 短目标耗时 | 短目标到达／末态到达 | 短目标首次到达（仿真秒） | 实际失败 |
| --- | ---: | ---: | ---: | --- | ---: |
| 原SAT原型、旧软查询、4轮 | 106.3 s | 133.9 s | 10／10、10／10 | 3–16 | 0／8 |
| 计算优化、4轮 | 79.5 s | 105.9 s | 10／10、10／10 | 3–16 | 0／8 |
| 计算优化、3轮 | 69.3 s | 95.5 s | 10／10、10／10 | 3–12 | 0／8 |
| 计算优化、2轮 | 64.8 s | 80.7 s | 9／10、9／10 | 9辆3–13；1辆未到 | 0／8 |
| 最终：3轮＋审查优化 | 58.4 s | 83.2 s | 10／10、10／10 | 3–12 | 0／8 |

每个表格行含原／短目标各4个环境，全部实际观察30秒，没有提前结束。首次到达按1秒快照统计。2轮分支的0/3车30秒后仍离短目标约19.7厘米，因此不选；3轮满足10／10末态误差小于10厘米，末态误差0.6–8.7厘米。原目标平均进展依次为0.716／0.716／0.729／0.760／0.729米，全部仍0／10到达；保持原目标含不可达位置的解释。

4轮原／优化两组的全部30份逐秒位置、路径、奖励、评分、规划有效性和控制状态完全一致，基线还逐值复现上一轮SAT结果。3轮审查优化前后同样逐值一致。2／3／4轮本身会改变采样与行为，不把它们标为计算等价。

![各阶段闭环耗时](../../outputs/runs/upstream_1km_fixed_far/mppi_query_optimization_20261025_01/figures/closed_loop_time.png)

第三项定位到既有`PlanningAdapter._review`：它只使用大坑距离的正负号，却通过`GridTerrain.obstacle_distance`顺带计算41个位姿的完整已知范围并丢弃。新增`include_known=False`只省去这些无用计算，默认调用仍返回原已知范围；预测完整审查若启用，仍由后续reviewer独立检查，实际感知、停车和失败规则保持。它同时覆盖HOLD、刹车和重生成候选的大坑复核，没有降低任何已使用的检查频率。

10车×41姿态的独立CPU查询从151.1 ms降到1.02 ms，碰撞结果逐值相同。这是一个小查询的约149倍，不是整段加速比。闭环最终审查及评分阶段由6.63／8.00秒降到1.47／2.65秒；HOLD准备阶段同样受益，单列阶段不能替代整段计时。

最终原／短目标两段：规划40.63／42.38秒，正常感知刷新4.75／14.82秒，运动积分0.38／0.41秒。感知已有增量更新与双worker，本轮没有足够证据把全部感知搬到GPU；车辆运动积分占比很小，保持CPU。确认并省去无用审查后即结束本轮，不继续添加融合核、GPU环境或行为规则。

精简回传的10车冻结批次：原末轮payload按保存张量尺寸计算为12,190,420 bytes，候选实测151,550 bytes，减少98.76%；仅指该末轮payload，不含地图上传和最终评分结果。前8个闭环的回传计数字段误取首个HOLD行，原始JSON中的0表示该位置未采到计数，汇总已改为null；最后两个分支按批内最大值正确采集，各30批共4,546,500 bytes。此计数缺失不影响执行或墙钟测量。

本轮所有组件／闭环时间均包含相应计算阶段的正常开销。闭环整段从恢复后开始，含实际执行、规划、感知、日志和进度写出，不含恢复初始化；阶段计时使用较小的step范围，不能直接相加替代整段。没有将这些单owner短片段倍率外推为完整训练工期。


## 证据与复现

run：`outputs/runs/upstream_1km_fixed_far/mppi_query_optimization_20261025_01/`。

- `run_manifest.json`：范围、预算、来源、失败准备与状态。
- `config/before/`：修改前当前工作区源码，保护原有未提交内容。
- `metrics/component_cost.json`：软查询合并与原大坑评分的独立计时。
- `metrics/component_planner.json`：6批完整规划等价性；计时受并发测试影响，不作性能结论。
- `metrics/sat_integration.json`：源码SAT与原型全位置一致性，以及SAT评分独立计时。
- `metrics/checks_stage1.xml`、`checks_stage2.xml`：针对性检查及原始日志。
- `metrics/result_*.json`、`trace_*.json`、`state/execution_*.pkl.gz`：闭环记录。
- `metrics/comparison.json`：闭环、首次到达、等价性和109项去重测试汇总。
- `metrics/iteration_selection.json`、`review_component.json`、`host_transfer_audit.json`：3轮选择、CPU审查查询及回传大小。
- `metrics/final_summary.json`、`completion_verification.json`：10分支预算、恢复、源码及events核验；原前8分支`summary.json`与后2分支`refine_summary.json`分别保留。

实际命令：

```bash
RUN=outputs/runs/upstream_1km_fixed_far/mppi_query_optimization_20261025_01
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTHONPATH=scripts:source/lunar_rover_tasks .venv_terrain_cuda/bin/python "$RUN/config/component.py"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTHONPATH=scripts:source/lunar_rover_tasks .venv_terrain_cuda/bin/python "$RUN/config/verify_sat.py"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTHONPATH=scripts:source/lunar_rover_tasks .venv_terrain_cuda/bin/python "$RUN/config/closed_loop.py"
PYTHONPATH=scripts:source/lunar_rover_tasks .venv_terrain_cuda/bin/python "$RUN/config/analyze.py"
# finish_sequence.py在前8分支结束后选择迭代数、执行CPU检查并启动refine_loop.py。
# 它的实际启动命令和PID保存在config/sequence_launch.json。
```

闭环脚本带已运行保护；复现实验须新建run，不能覆盖此处原始结果。TensorBoard当前仅加载本run，横轴为本轮实际诊断底层步，学习步保持0；启动及结束HTTP200，events确认从0增长至6,000诊断步，结束后保留回放。

```bash
.venv_terrain_cuda/bin/tensorboard \
  --logdir outputs/runs/upstream_1km_fixed_far/mppi_query_optimization_20261025_01/tensorboard \
  --host 127.0.0.1 --port 6006 --reload_interval 5
```

访问地址：[http://127.0.0.1:6006](http://127.0.0.1:6006)。本轮为Proxy计算与局部执行诊断，未启动新学习。

针对性测试命令（在闭环开始前完成前两组，最后两项在第三阶段开始前完成）：

```bash
PYTHONPATH=scripts:source/lunar_rover_tasks .venv_terrain_cuda/bin/python -m pytest -q -o addopts= \
  tests/test_upstream_mppi_query_optimization.py tests/test_upstream_crater_grid.py \
  tests/test_upstream_mppi_body_geometry.py tests/test_upstream_factory.py
PYTHONPATH=scripts:source/lunar_rover_tasks .venv_terrain_cuda/bin/python -m pytest -q -o addopts= \
  tests/test_upstream_mppi_query_optimization.py tests/test_upstream_mppi_body_geometry.py \
  tests/test_upstream_mppi.py tests/test_upstream_mppi_precision.py
PYTHONPATH=scripts:source/lunar_rover_tasks .venv_terrain_cuda/bin/python -m pytest -q -o addopts= \
  tests/test_upstream_mppi_query_optimization.py -k final_review_omits
```
