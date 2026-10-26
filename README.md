# Isaac 多车月面集合项目

本仓库研究多月球车在局部感知和有限通信下的自主集合。当前使用差速 Proxy、skrl MAPPO 与 CUDA MPPI 进行正式训练，Isaac Sim／PhysX 用于后续物理闭环验证。

当前主线是正式训练及其结果驱动修复；消融在主方案取得可用结果后开展，T15 治理同步执行。运行事实见[当前状态](docs/current_status.md)，工作顺序见[实施计划](docs/implementation_plan.md)，全部文档由[文档入口](docs/README.md)导航。当前尚无通过完整任务收敛评估的推荐 checkpoint。

## 本地使用

当前正式训练使用 `.venv_terrain_cuda`，现有进程已在运行。配置、监控和恢复入口见[训练手册](docs/runbooks/train_proxy.md)；不要重复启动同一 run。新环境和 Isaac 栈安装见[环境手册](docs/runbooks/setup_environment.md)。

Actor 与局部规划器使用合法局部信息；集中式 Critic 使用团队状态，当前 Oracle 预留槽为零。有效契约见[接口规范](docs/interface_spec.md)，源码及活跃配置见[规划架构](docs/architecture/planning_pipeline.md)。

`outputs/` 存放生成结果并由 Git 忽略，长期结论记录在[实验索引](docs/experiments/README.md)及对应文档。每次训练保留实时 TensorBoard 和可追溯恢复状态，规则见[输出管理](docs/references/output_management.md)。
