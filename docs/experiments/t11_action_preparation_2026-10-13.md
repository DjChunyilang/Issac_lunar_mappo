# T11动作表示无学习准备（2026-10-13）

依据[原规划T11](../../writing-block.md#t11mappo-动作表示消融)及[交接计划](../t11_handoff_plan.md)，本轮完成A0–A3的独立动作／概率模块、共享网络骨干、循环概率重放和诊断入口。**通过范围为合成输入上的概率准备；T11动作表示筛选、Gate C和新尺度环境接入尚未完成。** 正式训练继续暂停。

## 实现与比较边界

代码为`local_goal/action_ablation.py`，配置为[独立准备配置](../../configs/ablation/t11_action_contracts.yaml)，入口为`scripts/diagnose_t11_actions.py`。该配置不属于experiment schema，旧环境／训练配置加载器拒绝它。旧`ppo_update`也在任何优化器调用前拒绝`diagnostic_only`模型；原有策略路径保留。

| 方案 | 已实现的概率与载荷 | 解码／比较边界 |
| --- | --- | --- |
| A0 | 二维tanh Normal；物理`d rho d beta`密度，包含缩放Jacobian | 始终MOVE，`rho=0`也不变为HOLD |
| A1 | 复用`PolarHoldDistribution`；HOLD离散原子＋条件二维MOVE | HOLD的连续动作与latent占位为零，log-prob不计算连续项；混合分布熵仍取当前策略期望 |
| A2 | 四维tanh Normal；保留exp167归一化动作密度 | 解码完整目标／航向／速度上限，`speed=0`仍是MOVE；不是终端速度命令 |
| A3 | 47维categorical，int64原生编号＋库版本／内容指纹 | 编号0保留原语库的HOLD语义，不转成极坐标；旧原语尺寸／速度尚未迁移新尺度 |

四组共享现有T04的N1 CNN、GRU和集中Critic，只改变必要动作头。工厂从同一模板复制所有共享参数并逐值验证，避免不同头的初始化消耗影响Critic。当前观测是合成T04布局`[E,4,411]`，Critic维度991；元数据明确标记这一准备范围，**不能据此声称新尺度观测／任务已接通**。

动作尺度读取不可变新profile：极坐标半径4 m、速度上限25/9 m/s、有效期1 s。A2保留每轴独立缩放，其方形角点可达`4√2≈5.657 m`，A0/A1为半径4 m圆盘。后续执行桥接前须决定并版本化公平的目标支持范围；不能直接把A2载荷接入要求位移≤4 m的新执行链，也不能裁剪采样动作后沿用原概率。A3执行语义差异同样需要在最终比较中单列。不同测度的原始entropy不直接用于动作方案排名。

`PreparedRequest`仅冻结决策时的目标／航向／时间或原语载荷，不是`PlannerInput`，没有规划成功或实际停车含义。实际执行反馈目前记录为`not_connected`。诊断状态先校验方案、尺度指纹、测度、网络维度、库指纹、状态版本及全部权重，再一次加载；不接收历史checkpoint，不导出训练checkpoint。

## 本轮证据

正式准备run：

```text
outputs/runs/t11_action_contracts/preparation_seed23_31_47_cpu_20261013_02/
```

`metrics/preparation_acceptance.json`记录97项新契约全部通过，零失败／零跳过。`metrics/probability_matrix.json`覆盖四方案×种子23／31／47的12个CPU float32面板：共768个真实采样决策，720个有效重放决策；最终原样本PPO ratio最大误差为`2.384185791015625e-7`，低于`1e-5`门槛，梯度／熵有限，权重未更新，同契约恢复逐值一致。

`samples/actions.json`保存全部768个原始动作、分支／编号、latent、log-prob、学习有效性与结束掩码，采样记录覆盖率100%；每面板首时刻另存解码载荷。诊断只对合成优势做反向传播以检查梯度，没有环境交互、优化器构造或更新。模型权重只在内存中恢复，未生成训练checkpoint。

契约包括独立变换密度参考、A1配对、全部47编号、极端latent／logits、篡改动作拒绝、无效NaN在网络运算前排除、异步结束、普通序列切分保留GRU状态、跨方案恢复拒绝及加载失败不部分修改。沿用现有有限任务bootstrap／变时长GAE，通过相关P0和T04回归核验，未另造学习算法。

最终同源码运行`tests/test_polar_hold.py`、`tests/test_p0_learning_contracts.py`、`tests/test_config_wiring.py`共336项，全部通过，包含2项真实acados HOLD回归，零跳过。与97项新契约文件无重合，本轮共433项。命令、完整日志、JUnit和测试前冻结的源码身份见`logs/regression_command.json`、`logs/regression.log`、`metrics/regression.xml`及`metrics/regression_acceptance.json`。旧P0／T04回归中的合成PPO更新属于既有算法契约测试，未运行环境学习预算。

开发失败run完整保留：

```text
outputs/runs/t11_action_contracts/development_seed23_cpu_20261013_01/
```

该批次83通过、4失败，原因是删除无效环境后float32 GEMM批量尺寸不同，GRU尾状态最大误差`1.4901161193847656e-8`，测试误用了逐位相等。修正为`atol=1e-7/rtol=1e-6`后通过；同批量状态恢复仍要求逐值一致。原失败JUnit、日志及源码／测试指纹未覆盖。

首轮`preparation_seed23_31_47_cpu_20261013_01`的87项通过与12面板保留为开发证据。随后审查补齐两处缺口：重放固定旧采样／初始隐状态和优势的计算图，避免新旧概率梯度抵消；测试源码SHA在子进程前冻结，并在测试后、发布前复核，避免同名测试正文变化错误绑定JUnit。增加八项带图采样梯度回归、正文变更拒绝及CPU设备身份检查，最终run为上述`_02`。相关开发测试中确定性A1全HOLD的测试夹具修正为显式MOVE，以覆盖连续latent计算图；失败与重跑日志一并保留。

## 复跑与后续出口

从仓库根目录执行，使用新的run-id：

```bash
.venv_terrain_cuda/bin/python scripts/diagnose_t11_actions.py \
  --run-id preparation_seed23_31_47_cpu_20261013_03
```

脚本自动执行完整新契约，绑定JUnit测试身份、源码、解析配置、尺度与原语库指纹，再运行12个面板；已有run拒绝覆盖。CPU、float32和单Torch线程单独记录。源文件、配置或库身份变化后不能沿用原验收。

下一出口仍是[S00–S05实施计划](../implementation_plan.md)的新尺度40步规划器、感知通信／任务、安全集成、校准及完整Gate B／144回合。通过统一版本门槛后，接四组运行配置与执行桥接，冻结训练／评估场景、目标支持范围和目标服务器，再按原计划推进100k→500k×3→2M×3。当前机器可读结果保持`environment_connected=false`、`t11_accepted=false`、`gate_c_passed=false`。
