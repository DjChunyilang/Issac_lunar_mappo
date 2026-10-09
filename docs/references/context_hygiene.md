# 仓库上下文边界

更新时间：2026-10-09。此页记录本次审计的处置依据；当前实验事实仍以[当前状态](../current_status.md)为准。外部[静态审计报告](../archive/context_hygiene_audit_2026-10-09.md)保存在归档中，其中的基线提交和若干文件名与本地仓库不同。

| 对象 | 本地核对结果 | 处置 |
| --- | --- | --- |
| `outputs/isaac_scenes/proxy_rovers_scene.usda` | `scripts/view_proxy_rovers_isaac.py` 的默认 `--stage-out` 导出目标；没有测试或运行时读取者 | 从 Git 索引移除，本地文件保留；`outputs/` 整体忽略 |
| `.codex/skills/` | 两个 skill 原含大量目录示例和当前结果规则 | 缩为任务流程，细节指向文档入口与输出管理 |
| 根目录长技术文档 | 设计背景有追溯价值，不能当当前实验状态 | 移入 `docs/design/`，顶部标注设计基线 |
| 根目录 V3 跳转页 | 原文已在 `docs/archive/`，跳转内容已有正式入口 | 移除重复页 |
| 根 `pyproject.toml` / `setup.py` | README 与 CI 均从 `source/lunar_rover_tasks` 安装；根 `setup.py` 仅调用 `setup()`，根 `[project]` 声明了另一个发行名 | 根只保留工具配置，子包 `setup.py` 维护可安装包 metadata |
| `tools/*.py` | 四个独立诊断/辅助 CLI，没有仓库内部引用；静态搜索不足以证明无人手动使用 | 保留，作为手动工具；不作为默认项目入口 |
| 根目录论文 PDF | 没有仓库内部引用 | 保留并移至 `docs/references/papers/` 按需查阅 |
| `exp156_timeout_diag_*` 六个 YAML | `extends` 指向未恢复的旧 run 生成 JSON，干净检出无法解析 | 原文移至 `docs/archive/configs/`，不伪造配对输入；正式配置目录只留自包含配置 |
| 三个 `scripts/debug_*.py` | 各自约 44–55 行，分别检查环境、观测和奖励；共用 `_common.cfg_from_experiment` | 保留三个明确入口；目前没有足够重复代码支持合并 |

从仓库根目录默认运行 `rg` 时，`.rgignore` 会略过历史归档、设计基线和状态汇报；需要追溯时明确指定目录或使用 `rg --no-ignore`。Ruff `F401/F841/F821` 审计初始发现 31 项，其中 `F821` 未定义 `Any` 已修复；余下 28 项未使用 import 和 2 项未使用局部变量留待逐项确认副作用、对外 API 与动态注册。CI 先强制 `F821`，不据此批量删核心代码。

核心环境、注册、训练配置与 debug 脚本未因静态“未引用”而删除。报告示例中的 `b914f0b...` 不是本地清理起点；本次开始时本地 HEAD 为 `79bf71e`。Git 中的旧版本仍可追溯，未创建或改动分支、tag 或提交。
