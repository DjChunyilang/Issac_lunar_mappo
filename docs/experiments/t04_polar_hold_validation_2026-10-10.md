# T04：条件极坐标／HOLD 与 Gate A 验收

日期：2026-10-10。基线提交 `4a79b0d5faf6d65037422c352ed91f827c8c40f1`，在已验收的T00–T03未提交工作区上实施。状态：**T04与Gate A契约验收通过，正式训练继续暂停**。实施边界见[阶段计划](../archive/implementation_plan_t04_snapshot_2026-10-10.md)，字段和概率定义见[接口规范](../interface_spec.md)。

## 交付与边界

新增独立t04_polar_hold配置、polar_hold_v2动作、local_intent_clock_v3观测和polar_hold_rollout_v1采样载荷。MOVE为二维条件Normal，经tanh和仿射变换映射至rho/beta；HOLD只计Bernoulli分支动作概率，连续占位不参与似然。完整混合熵仍包含当前MOVE分布的概率加权期望。

动作解码直接产生T03 LocalIntent。环境不为MOVE补造航向或速度约束；旧exp167/P0/T03配置继续保留原语义。新版观测维度仍为411/991或523/1439，但旧观测／动作checkpoint被明确拒绝。runner建模、采样记录、版本校验和评测元数据贯通；修复原评测报告写死四维动作版本的路径。

按用户确认的边界，MOVE使用明确标记test_only的测试适配器验收；真实NMPC验证HOLD。T05低维MOVE完整适配、MPPI及安全投影均未实施。T04进入旧完整任务评测入口时提前拒绝并说明T05依赖。Gate A通过不是正式训练授权，也不代表Gate B/C、完整集合任务或真实月面安全通过。

## 独立证据

Run：`outputs/runs/t04_polar_hold/contract_seed23_cpu_20261010_01/`，seed=23，CPU。原T02/T03产物保留。

| 文件 | 内容 |
| --- | --- |
| config/experiment.yaml | 解析后的诊断配置 |
| run_manifest.json | 版本、基线、工作区、源码／配置指纹、命令与证据索引 |
| metrics/source_snapshot.json | 验收时源码逐文件指纹 |
| metrics/contracts_acceptance.xml | 最终联合回归JUnit |
| logs/contracts_acceptance.log | 测试计数、跳过原因、acados原始输出 |
| metrics/audit_final.json | 独立解析概率、测试专用MOVE、合成PPO及真实HOLD审计 |
| metrics/gate_a.json | 每项Gate A门槛、具体测试与通过状态 |
| metrics/probability_checks.json | 从JUnit提取的概率／熵／重放误差与HOLD轨迹 |
| metrics/summary.json | 最终验收摘要与后续门槛 |
| logs/static_checks.log | 链接、公式、卫生、F821、diff和源码一致性 |

源码AST指纹：`317ffbed219b16df2ddcb223d736544d1842d178491372ea1177db79c870bcbb`。审计脚本自身SHA256另行记录；动作范围也进入合同／完整配置校验。无正式环境交互训练、无训练checkpoint；每次独立审计仅作1次固定合成rollout的优化器契约更新。

## 验证命令

从仓库根目录执行，使用T03相同的本地Python/acados安装。

```bash
ACADOS_SOURCE_DIR="$PWD/.venv_isaaclab/acados" \
LD_LIBRARY_PATH="$PWD/.venv_isaaclab/acados/lib" \
OMP_NUM_THREADS=1 .venv_isaaclab/bin/python -m pytest \
  tests/test_polar_hold.py tests/test_planner_protocol.py \
  tests/test_p0_learning_contracts.py tests/test_local_goal_nmpc.py \
  tests/test_local_goal_diagnostics.py tests/test_reward.py \
  tests/test_observation.py tests/test_termination.py \
  tests/test_tiered_communication.py tests/test_exp156_differential.py \
  tests/test_config_wiring.py tests/test_lunar_training_maps.py \
  --junitxml=outputs/runs/t04_polar_hold/contract_seed23_cpu_20261010_01/metrics/contracts_acceptance.xml \
  -ra --disable-warnings
```

```bash
ACADOS_SOURCE_DIR="$PWD/.venv_isaaclab/acados" \
LD_LIBRARY_PATH="$PWD/.venv_isaaclab/acados/lib" \
OMP_NUM_THREADS=1 .venv_isaaclab/bin/python scripts/audit_polar_hold.py \
  --real-acados \
  --output outputs/runs/t04_polar_hold/contract_seed23_cpu_20261010_01/metrics/audit_final.json
```

审计拒绝覆盖已有输出。复跑必须使用新的输出路径；上述路径为本次证据，不作为可覆盖位置。真实HOLD使用固定诊断logit头、平坦局部传感记忆、10 s求解时限；不能据此宣称生产0.15 s时限或复杂地形可靠性已验收。

## 结果

最终 **453项通过、1项非必要CUDA一致性检查因CPU环境跳过、0失败；12项真实acados全部实际通过**。相对T03增加42项动作契约测试、1项新配置检查和1项新版真实地图观测检查。联合回归115.41 s，日志记录509项警告计数；使用--disable-warnings未展开逐条警告，原始QP输出保留。

| 数值检查 | 结果 |
| --- | --- |
| float64混合log-prob与独立TransformedDistribution | 最大误差7.11e-15，满足atol=1e-10／rtol=1e-8 |
| 物理极坐标独立数值积分的混合总质量 | 与1差9.67e-10，容差1e-5 |
| 混合熵：65536个样本与128点Gauss-Hermite参考 | 误差3.11e-5，容差max(5倍标准误,1e-3)=0.003701 |
| 全HOLD／全MOVE／混合采样重放 | PPO ratio与1的最大误差均为0 |
| 独立审计的解析log-prob | 最大误差0 |

HOLD动作似然不读取未活动的连续latent；关闭熵正则时连续头无HOLD似然梯度。MOVE、混合熵、全无效批次、含NaN的无效行、分支版本／动作篡改拒绝均通过。18步重放验证第16步边界使用保存的hidden；异步结束重置对应GRU，普通序列边界不重置。

真实HOLD从线速命令0.8 m/s、角速命令0.5 rad/s开始，逐步线速命令约为0.5、0.2、0，遵守0.3 m/s和0.6 rad/s的逐步变化率。前两步可因ACADOS_MINSTEP进入显式刹车回退，valid=false且原因保留；后续达到停车。独立任务判据验证角速超限会清零驻留，HOLD本身不会自动成功。

Gate A报告逐项覆盖：任务截止与rollout切分不同；邻车基础／目标／轨迹来自同一sender；HOLD不等于驻留成功；Actor/Critic保留任务时钟；hybrid log-prob与原始采样一致。同时重跑T02回报势函数与终端结算解析测试及T03协议回归。文档链接／数学、仓库卫生、Ruff F821和git diff --check通过。

## 探索失败与修正

- 初始旧契约100项相关回归通过，11项真实求解器当时未运行，不计为最终验收。
- 新增纯动作测试首轮36通过、1失败：stop_intent将float32状态转为Python float后，在0.05 m/s、0.1 rad/s精确阈值处发生比较差异。改为与任务状态相同精度的张量比较，保留边界回归；失败日志为logs/hybrid_initial.log。
- 修正后38项含真实HOLD通过；补充跨16步序列、真实地图和审计入口后55项轻量检查通过，最终完成上述联合验收。所有探索日志保留，不能将早期未执行的必要测试算作通过。

下一阶段是T05完整基线适配；T06可使用既有协议推进MPPI基础。正式训练仍需后续固定场景与无学习闭环门槛，当前没有推荐checkpoint。
