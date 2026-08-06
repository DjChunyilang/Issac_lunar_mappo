# 路线图

## 2026-08-05 收敛主线更新

- exp125 B0、exp137 B2、exp140逐车信用与exp142 PPO-Lagrangian组件筛选均已按预注册门限停止，不启动40M。
- exp142证明真实collision约束可以显著降低碰撞，但当前策略以放弃集合换取安全；因此不继续扫描约束预算、dual参数或网络容量。
- exp143/144证明B0后段集合能力仍持续改善，但地形影响增强后实际quintic路径风险反而跨种子恶化；因此不通过增加训练预算解决该问题。
- exp145/146已进一步排除统一逐车回报和最近邻成对安全Critic：集合与地形可辨识，但安全的本车及成对条件动作增益均未稳定过门限。
- exp147确认当前quintic规划—执行契约存在系统性失配：约79%路径按时间戳要求速度越界，实际一步仅执行规划弧长约1.77%。
- exp148已完成时间一致性修正：速度违例降为0，实际弧长利用率提高到约12%，全部工程门限通过。随后唯一一次随机初始化B0 4M仍失败，评测success为0、collision为`0.9990`，terrain contrast也未过门限，因此不启动40M。
- exp148双种子失败episode诊断显示全部碰撞episode都包含重复车辆对冲突，中位数均为5；近距通信完整且无消息年龄，故不启动B1，exp137又已否决单独B2。下一步仅允许冻结分析碰撞终止前的奖励—动作—冲突时序，在形成单一且可证伪的训练信用假设前不新增训练模块。
- exp149进一步确认碰撞责任高度局部：典型碰撞仅涉及2辆车，约50%车辆受到无直接责任的团队终止惩罚，碰撞对在终止前8/16步的重复冲突召回约为100%。
- exp150按真实终止参与者进行零和Actor信用分配，工程不变量全部通过，但唯一一次4M仍为success `0`、collision `0.9990`，重复冲突中位数从`5/5`升至`9/8`。因此该信用方向停止，不调参、不组合、不启动40M；下一步仍限于冻结证据分析。
- exp151进一步否决终止参与者信用的因果前提：8/16步局部可行动率跨组合最小仅`0.5100/0.5616`，等量信用支持率仅`0.1777–0.2454`。下一步转为冻结分解动作表示、quintic、控制和地形/集合约束，不启动训练。
- exp152将首个失败层定位为动作—quintic：8步无约束可行动率最小`0.6648`，但已有避碰轨迹的控制传递率最低仍为`0.8069`且控制几乎不饱和。下一步只比较现有动作全范围、局部覆盖和quintic几何，不直接修改训练。
- exp153未找到单一动作范围或quintic瓶颈：全网格quintic最小`0.7536`、line最小`0.6275`，endpoint却恒为1；t2048途中交叉损失约35%–37%。下一步只做碰撞对双车联合反事实干预，不增加在线协调模块。
- MAPF冲突时间对齐已修正为车辆对局部公共时域，消除了第三车轨迹时长对目标车辆对距离的伪影响；该变化只影响诊断日志。
- 严格去中心化101维Actor、12 m分级通信、96 s/480步验收和MAPF离线诊断边界保持不变。

## 立即处理

1. 将 `docs/implementation_plan.md` 和 `docs/architecture/overall_plan_v3.md` 固定为当前主路线来源。
2. 保持 exp008 为当前已验证的 3-seed terrain-aware proxy baseline。
3. 暂停默认追加 exp009/exp010 strong terrain retry、exp012/exp013 action-scale long run；新增 proxy run 必须服务 checkpoint 评估、接口回归或明确假设验证。
4. 对候选 checkpoint 统一运行 `scripts/run_checkpoint_evaluation.py`，生成 `metrics/final_eval_proxy.json` 和 `metrics/checkpoint_status.json`。
5. 文档中严格区分三类结果：proxy training、proxy strict evaluation、Isaac/PhysX high-fidelity closed-loop evaluation。

## 近期工作

- D0 回归底线：保持 `.venv_isaaclab/bin/python -m pytest -q -ra` 可通过；SKRL import 测试不能 skip。
- D1 评估编排：让 exp008 / exp013 等配置携带 `evaluation:` block，统一 proxy eval 与 PhysX eval 的触发规则。
- D2 Checkpoint 状态：每个候选 run 都记录 `candidate`、`proxy_passed`、`physx_evaluated`、`physx_passed` 或 `final_selected`。
- D3 PhysX/Jackal 评估：把 Jackal 作为 high-fidelity tracking validation 资产，覆盖平地调优和 strong lunar crater 直线、绕圆、正弦跟踪，记录 tracking error、completion、tilt 和 throughput。
- D4 报告口径：旧 V1 / V2 / V3 原文集中压缩归档；根目录保留长期技术路径文档，当前实施路线统一维护在 `docs/implementation_plan.md`。

## 中长期工作

- 用更接近月球车的 USD/URDF 或轮式底盘参数替换 Jackal placeholder。
- 增加低重力、摩擦、轮地接触、坡面稳定性和倾覆风险的高保真评估配置。
- 如果 high-fidelity eval 暴露系统性迁移失败，再引入 Isaac-based fine-tuning、domain randomization 或更高保真的 proxy dynamics。
- 构建报告生成器，从 `strict_acceptance.json`、`final_eval_proxy.json`、`checkpoint_status.json` 和 PhysX metrics 自动更新实验文档。
