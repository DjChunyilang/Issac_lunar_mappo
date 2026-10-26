# T15：随正式训练持续治理

更新：2026-10-22。依据 [writing-block.md](../writing-block.md) 的 T15，执行顺序服从[当前计划](implementation_plan.md)。目标是让下一次实验找到正确入口、复用有效证据并保留旧结果，不把全仓重构变为训练门槛。

## 本次已完成

| 项目 | 落地结果 |
| --- | --- |
| 完整历史保护 | [快照目录](archive/formal_training_transition_2026-10-22/README.md)保留调整前 12 份文档，包括工作区未提交内容；相对链接重定位，manifest 记录原文及快照 SHA-256 |
| 当前入口 | 状态只保留当前事实；实施计划／路线图改为正式训练优先；实验安排将消融延后 |
| 架构与接口 | 去除逐日追加式旧参数，集中描述当前感知、A1、时序、MPPI、任务和恢复语义；旧全文可追溯 |
| 配置清单 | [架构](architecture/planning_pipeline.md)明确三个 active 配置及设计／地图依赖；旧配置保持加载和复现能力 |
| 代码职责 | 活跃模块、对照分支、历史入口与诊断用途分开标注；没有将未核实模块标为 DEAD |
| 工作流与导航 | 根 README、AGENTS、文档导航和训练手册同步；保留 T11 历史跳转及两个窄职责 Skill |
| 产物 | 复用现有源码／配置快照、owner checkpoint、metrics 和 events；本次不搬移或覆盖 outputs |

本次没有改算法、配置、训练状态或预算，也没有启动新诊断／学习。文档不进入当前训练源码指纹，运行中源码与原身份保持；本次验证限于链接、正文与 diff。

## 引用审查与隔离边界

工厂 `upstream_factory.py` 对 MPPI、NMPC 和几何基线按配置构造；当前 MPPI v4 依赖 `upstream_mppi_calibration.py`，后者继承 `upstream_mppi_precision.py` 及公共 MPPI 实现。历史版本名不能作为移除共享基类的依据。`action_ablation.py` 同时承载当前 A1 与延期的其他表示，消融延期不等于该文件闲置。

本次只核对当前链的直接引用与有效配置，未完成全仓 import graph；因此不删除 Active-DSTC、analytical PRD、旧过滤器或旧环境。此前 [T00 审计](experiments/t00_repository_audit_2026-10-09.md)继续作为历史依据，有依赖变化时再补查。

`gathering_env.py`、`gathering_env_cfg.py`、`action_interpreter.py`、`communication.py` 的整体拆分延后。当前正式链以 `local_goal/upstream_*` 承载相关职责，遇到实际修改困难时再判断必要拆分，不单为目录形式重写环境。

## 持续工作及结束条件

| 触发 | 要做的工作 | 本次工作的结束条件 |
| --- | --- | --- |
| 修改当前组件 | 核对直接消费者及相关测试，必要时内部拆分 | 具体缺陷修复且受影响行为已验证，立即回到实验 |
| 改接口／行为语义 | 同步代码、接口、架构、迁移说明及必要契约测试 | 新旧配置与 checkpoint 边界明确，旧结果可追溯 |
| 确需隔离历史代码 | 引用／import graph → 测试 → 分类 → 隔离 → 受影响回归 → 引用更新 | 无活跃或复现消费者后才考虑删除 |
| 切换主训练配置 | 更新 active 清单、预算／恢复边界和实验记录 | 明确唯一当前入口，启动和 events 可核验 |
| 冻结主方案用于消融 | 固定源码、配置、地图、种子、checkpoint、评估与失败记录 | 对照可复现，再安排后续实验 |

T15 是持续职责：本轮完成上下文和配置入口治理，代码隔离／拆分未宣称全部完成。没有具体维护缺陷时，保持现有目录和模块，不追加无关清理。输出规则见[输出管理](references/output_management.md)，实验事实写对应记录，Skill 仅保留流程。
