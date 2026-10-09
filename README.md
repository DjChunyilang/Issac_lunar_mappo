# Isaac 多车月面集合项目

本仓库研究多月球车在局部感知和有限通信下的自组织集合。代理环境使用 Torch 向量化动力学进行训练和诊断；Isaac Sim / PhysX 用于候选策略的高保真闭环评估。

**当前正式训练暂停，没有通过完整任务验收的推荐 checkpoint。** 当前事实、失败门控和下一步见[项目状态](docs/current_status.md)；文档阅读顺序见[文档入口](docs/README.md)。

## 本地起步

本机已有 `.venv_isaaclab` 时，可运行：

```bash
.venv_isaaclab/bin/python -m pip install -e source/lunar_rover_tasks
.venv_isaaclab/bin/python -m pytest -q -ra
```

新环境与 Isaac 栈安装见[环境手册](docs/runbooks/setup_environment.md)。Proxy 训练、独立评估和结果路径分别见[训练手册](docs/runbooks/train_proxy.md)、[评估手册](docs/runbooks/evaluate_proxy.md)和[输出管理](docs/references/output_management.md)。恢复正式训练前先核对[路线图](docs/roadmap.md)及实验配置。

Gymnasium task ID：`Isaac-MultiRover-Gathering-Direct-v0`。Actor 不接收 oracle 集合点；oracle 仅用于 centralized critic、辅助奖励和评估。`outputs/` 存放生成结果并由 Git 忽略，长期结论记录在[实验索引](docs/experiments/README.md)及对应实验文档。
