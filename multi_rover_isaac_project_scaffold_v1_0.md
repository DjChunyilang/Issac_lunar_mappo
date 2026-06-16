# 多月球车自组织集合项目脚手架文档 V1.0

> 这是历史原始脚手架文档的跳转页。当前正式工程脚手架请阅读 `docs/scaffold.md`；当前实施路线请阅读 `docs/implementation_plan.md`；当前事实请阅读 `docs/current_status.md`。

## 当前入口

- 工程脚手架：`docs/scaffold.md`
- 技术设计：`docs/technical_design.md`
- 当前实施计划：`docs/implementation_plan.md`
- 当前状态：`docs/current_status.md`
- 历史原文归档：`docs/archive/multi_rover_isaac_project_scaffold_v1_0.md`

## 历史说明

原 V1.0 长文用于定义早期 Isaac Sim / Isaac Lab 工程脚手架、目录结构和模块边界，已经完整归档。旧文中把 Isaac Sim / Isaac Lab 写作训练主路径的表述，已被当前 V3 口径修订为：

```text
高吞吐 proxy 环境训练
-> proxy strict evaluation
-> Isaac Sim / Isaac Lab / PhysX high-fidelity closed-loop evaluation
```

因此，引用脚手架结构时以 `docs/scaffold.md` 为准；引用当前路线、里程碑和验收时以 `docs/implementation_plan.md` 为准。
