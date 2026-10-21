# T02 P0 学习与任务契约验收

日期：2026-10-09。基线提交：`4a79b0d5faf6d65037422c352ed91f827c8c40f1`；本次为其上未提交工作区修改，实际源码指纹保存在 run 中。状态：T02 契约验收通过；不代表完整任务或 Gate A 全部通过。当前入口见[实施计划](../implementation_plan.md)、[接口规范](../interface_spec.md)。

## 本次任务与交付

T00 仓库审计完成；T01 建立 AGENTS、当前导航、规划架构与活跃配置索引，旧计划／接口／技术说明已完整归档。T02 新增独立 `p0_contract_v2.yaml`，旧 exp167 配置未修改。

新增行为：任务截止真实终止；无效地形独立分类并屏蔽学习；Actor/Critic 剩余时间；邻车附加载荷按 sender_indices 对齐；新版停稳门；折扣一致的进展／驻留势差；按原因结算终端奖励；跨宏步回报累计与恢复；运行、评估及 checkpoint 版本与依赖指纹。

`for_hold` 和环境 `hold_mask` 仅供诊断。runner 不调用该动作覆盖入口，PPO 继续使用真实四维采样动作。新版 runner 在建立输出目录、环境和优化器前拒绝正式训练；旧 core.step 同时拒绝将新版配置误用于原语执行链。

## 环境与产物

Python 3.12.3，torch 2.10.0+cpu，numpy 2.3.1；CPU 求解器使用仓库已安装 acados。seed=23，诊断配置继承96 s任务截止，新增停稳判据为2 m／0.75 m²／0.05 m/s／0.1 rad/s／5 s。

本次 run：

```text
outputs/runs/p0_contract_v2/contract_seed23_cpu_20261009_01/
  config/experiment.yaml
  run_manifest.json
  metrics/source_snapshot.json
  metrics/analytic_audit_final.json
  metrics/contracts_acceptance.xml
  metrics/summary.json
  logs/contracts_acceptance.log
  logs/static_checks.log
```

`source_snapshot.json` 保存最终验收前实现指纹及源码 SHA256；解析后完整配置也随 run 保存。全部生成产物被 Git 忽略，没有生成训练 checkpoint，没有恢复正式训练。测试内的序列 PPO 梯度单测不构成实验训练。

## 验证命令

在仓库根目录运行，先将 `P0_RUN` 设置为新的独立 run 路径；不要复用上方证据目录：

```bash
ACADOS_SOURCE_DIR="$PWD/.venv_isaaclab/acados" \
LD_LIBRARY_PATH="$PWD/.venv_isaaclab/acados/lib" \
OMP_NUM_THREADS=1 .venv_isaaclab/bin/python -m pytest \
  tests/test_p0_learning_contracts.py tests/test_local_goal_nmpc.py \
  tests/test_local_goal_diagnostics.py tests/test_reward.py \
  tests/test_observation.py tests/test_termination.py \
  tests/test_tiered_communication.py tests/test_exp156_differential.py \
  tests/test_config_wiring.py tests/test_lunar_training_maps.py \
  --junitxml="$P0_RUN/metrics/contracts_acceptance.xml" -ra --disable-warnings

.venv_isaaclab/bin/python scripts/audit_p0_contracts.py \
  --output "$P0_RUN/metrics/analytic_audit_final.json"

.venv_isaaclab/bin/python scripts/check_repository_hygiene.py
.venv_isaaclab/bin/python scripts/check_markdown_math_delimiters.py
.venv_isaaclab/bin/python -m ruff check source scripts tools tests --select F821
git diff --check
```

审计脚本拒绝覆盖已有报告；复验使用新的 run。静态检查额外核验本次全部修改／新增 Markdown 的相对链接、归档快照正文完整性及根 AGENTS 行数。

## 解析与行为证据

| 契约 | 核验结果 |
| --- | --- |
| 截止与自举 | 给定R=-80、Vnext=10、k=5，旧示意目标-70.490097；新版任务截止目标精确为-80；有效外部截断保留自举 |
| 时钟 | 合成配置411/991维；初始时钟1、半程0.5、截止0、reset后1；真实地图三通道版本完成实际构建、采样及快照恢复 |
| 结束原因 | invalid > collision > bounds > success > deadline；同一步合法完成驻留优先于截止 |
| 邻车实体 | 距离排序交叉后基础信息与目标／轨迹仍来自同一sender；缺失、过期、稀疏、不支持版本及无效slot不泄漏载荷 |
| 去中心化边界 | 修改Oracle及未发送邻车真值不改变相应执行输入／邻车预测 |
| 驻留 | 角速度越限清零；驻留打断和重新进入满足势差首尾恒等式；8步与25步驻留塑形预算一致 |
| 无效数据 | 无效宏步不进入优势归一化和损失，不调用尾状态Critic；注入NaN的无效行不污染梯度 |
| 宏步与恢复 | 异步终止按实际时长累计，保留终止前观测；跨rollout原始／折扣回报、分量和局部reset／恢复一致 |
| 版本 | 错误组合、缺失新版checkpoint合同、错误规划revision拒绝；旧配置回归保留 |
| HOLD与NMPC | 真实acados覆盖前进、倒车、旋转、静止、前进后切换倒车，以及运动中HOLD限幅减速；fallback单独反馈 |

最终测试结果：**353 项通过，1 项 CUDA 一致性跳过；6 项真实 acados 测试全部实际通过，无必要测试被跳过。**

## 失败、修正与复验

首次新增单测有一处夹具尝试给只读 max_episode_steps 赋值；改为设置 episode_length_s 后通过。记录在 `logs/initial_fixture_failure.txt`。

第一轮宽回归为348通过、2失败、1跳过；两项失败均为旧诊断通过 __new__ 构造对象而未设置契约标记。补齐 legacy 标记后，该模块通过；原始日志／JUnit 保留为 `logs/contracts.log` 和 `metrics/contracts.xml`。后续CPU回归345通过、1跳过，6项真实acados被显式排除以复用前轮证据，结果保存在 `contracts_final.*`。最终验收再次覆盖真实acados和新增配置暂停／版本校验，以上最终结果以 `contracts_acceptance.xml` 为准。

CUDA一致性因当前CPU版PyTorch不可执行，单独记录，不列入P0必要通过项；所有必要真实acados项必须实际通过。acados API弃用警告及rasterio仿射运算弃用警告保留在日志中。

## 边界与下一步

这只是契约和局部执行验收，不证明集合任务收敛、MPPI优势或真实月面安全。旧回报与新版回报定义不同，不可直接比较历史曲线。没有新增checkpoint或完整任务成功结论。

本阶段通过，下一项为T03统一规划器协议。Gate A的混合动作／hybrid log-prob仍依赖T04；24 m通信、分层出生、逐车足迹、地理独立评估及PhysX闭环仍待后续实施。正式训练继续暂停。
