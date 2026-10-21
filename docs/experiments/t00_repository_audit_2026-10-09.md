# T00 仓库现状与技术路线审计

日期：2026-10-09。基线：`main@4a79b0d5faf6d65037422c352ed91f827c8c40f1`。执行前仅 `writing-block.md` 未跟踪，无已有代码修改。此报告记录修改前事实；T02 验收另记。

## 活跃调用链

代码根为 `source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/`。

| 环节 | 当前实际入口及行为 | 证据入口 |
| --- | --- | --- |
| 配置 | `scripts/_common.py::cfg_from_experiment` 解析继承配置；exp167 是实际入口 | `configs/experiment/exp167_local_goal_nmpc.yaml`、`tests/test_config_wiring.py` |
| 观测 | `local_goal/environment.py::observe` 读取 core 观测、局部传感缓存和合法通信；合成地图 410/990 维，真实图接口 522/1438 维 | `tests/test_local_goal_nmpc.py`、`tests/test_lunar_training_maps.py` |
| 采样 | `local_goal/learning.py::RecurrentPolicy.sample`，N1 CNN＋GRU、四维 tanh Normal，存 latent 和含 Jacobian 的 log-prob | log probability、recurrent PPO 单测 |
| 请求 | `local_goal/contracts.py::LocalGoalRequest.from_action` 将 `[dx,dy,yaw,speed]` 转换到重置时固定的 odometry frame | frozen goal 单测 |
| 规划与反馈 | `local_goal/planner.py::NMPCPlanner.plan`，acados、3 s/15 区间、prefix_consistency_v2、旧计划复核与限速刹车 | 真实 acados、失败注入和 swept crossing 单测 |
| 执行 | `track_trajectory` → `LocalGoalEnvironment._physics` → core `_integrate`，实际轮速受差速执行边界限制 | signed motion、差速原语和控制测试 |
| 奖励与结束 | `_physics` 调用共享 `reward.py`、`termination.py`；宏步累计实际底层时长内的折扣奖励 | reward aggregation、reward、termination 单测 |
| 学习 | `scripts/train_local_goal_nmpc.py` 采集宏步和终止前状态；`duration_gae`、`ppo_update` 完成变时长 GAE 与序列更新 | duration GAE、recurrent PPO 单测 |

真实地图训练在 runner 中显式暂停；不能通过仅修改 YAML 绕过。历史 SKRL learner 不属于 exp167 的实际 PPO 更新链。

## 路线状态与引用边界

| 路线 | 状态 | 入口／保留理由 |
| --- | --- | --- |
| exp167 四维请求＋NMPC | ACTIVE | `train_local_goal_nmpc.py` 与 `local_goal/`；存在局部诊断证据，未通过完整任务 |
| 47 差速原语 | BASELINE | exp156 配置、`action_interpreter.py`、`trajectory_generator.py`、差速测试；共享 core 仍复用相关传感／状态字段 |
| `[rho,beta]`＋line/quintic | BASELINE | 共享 action interpreter 和 trajectory generator，多个旧训练／诊断入口仍引用 |
| 40 类原语及旧网络 schema | LEGACY_REPRODUCIBILITY | 配置解析和历史测试仍使用，不删除 |
| Active-DSTC | LEGACY_REPRODUCIBILITY | `active_dstc_runtime.py`、`site_commitment.py` 与 exp160–166 脚本，当前暂停 |
| analytical PRD | LEGACY_REPRODUCIBILITY | `analytical_prd.py`、`scripts/prd_credit.py`、exp159 配置和测试 |
| subgoal filters | LEGACY_REPRODUCIBILITY | `subgoal_filter.py` 仍被共享 core 和旧配置引用；exp167 不走其动作执行链 |
| 学习契约／固定目标诊断 | DIAGNOSTIC | `audit_learning_contracts.py`、`diagnose_local_goal_nmpc.py`、`probe_local_goal_nmpc.py` |
| MPPI | 尚未实现 | 对 source/scripts/tests/configs 搜索 MPPI 无实现命中；仅为研究目标 |

没有模块被确认标为 DEAD。静态未命中不等于无人手动调用；本阶段不移动或删除上述代码。

## 可复用契约与问题

`LocalGoalRequest`、`TimedTrajectory`、`PlannerFeedback`、`NeighborPrediction` 已存在；复用其坐标系、时间戳、轨迹、反馈与信息权限。T03 再统一跨规划器协议，T02 不复制这些类型。

P0：任务 deadline 目前作为可自举 truncation；base Actor/Critic 无显式时钟；基础邻居按距离排序而附加消息按 sender 编号；驻留奖励反复累加，进展奖励未匹配折扣；成功条件缺少角速度门。

P1：四维请求与停车抽象、历史条件 Critic、梯度／学习诊断、完整集合可行性。其研究结论不能由 T02 局部测试推出。

## 修改前验证与证据边界

规划阶段使用 Python 3.12.3、torch 2.10.0+cpu、numpy 2.3.1、pytest 9.0.3，67 项局部目标／诊断／奖励／观测／终止／通信回归通过（排除真实 acados 和 recurrent PPO 更新）。卫生、Markdown 公式和 Ruff F821 检查通过。

对 `audit_learning_contracts.audit` 的无文件输出复核：示意 R=-80、Vnext=10、k=5 的旧目标为 -70.490097；时钟变化的 base 输入差为 0；seed23 的车0基础 sender `[1,3,2]`，附加 `[1,2,3]`；8步驻留奖励分量累计54，25步156。这些是解析／冻结观测探针，不是完整任务性能。

当前状态记录的旧 pilot 前后0/96成功和固定目标诊断只作为历史引用；本轮未重新运行完整任务，未恢复历史缺失产物。

## 交接

T00 完成：调用链、路线、复用类型和 P0 位置已明确。T01 先归档旧实施计划／接口／技术说明，再建立唯一当前入口。T02 修改范围限于版本配置、local_goal 环境和 learner、可选终止／奖励语义、runner 元数据与相关测试；旧默认语义保留。风险集中在跨回合 mask、观测维度、邻车槽位、终端奖励和 checkpoint 兼容性。
