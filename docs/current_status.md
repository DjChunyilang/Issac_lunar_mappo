# 当前状态

## 当前主线

- 当前主设计口径已切换为“高吞吐 proxy 训练 + Isaac Sim / Isaac Lab / PhysX 高保真闭环评估”。当前实施路线以 `docs/implementation_plan.md` 和 `docs/architecture/overall_plan_v3.md` 为准。
- 训练主环境仍是 PyTorch / torch-vectorized proxy 环境，用于 MAPPO / PPO 采样、奖励调试、观测接口验证和大规模对照实验。
- Isaac Sim / Isaac Lab / PhysX 不作为当前主训练 loop，而作为 high-fidelity validation、迁移 sanity check、失效分析和可视化展示平台。
- 当前 PhysX 层使用 Clearpath Jackal 作为活跃轮式资产，已替换旧占位资产。Jackal tracking 可验证轮式控制、强三维地形 mesh、姿态稳定性和输出链路，但不能证明真实月球车越障、轮壤接触或低重力动力学已经完成。
- 视觉观测不进入 policy input；地形以车体系 `5×5×2` 局部结构化网格进入策略。
- 生成结果写入 `outputs/runs/`，并由 git 忽略。

## 当前接口状态

- actor observation schema 为 `ego_v3_local_terrain_grid`，输入维度为 86，包含 ego、neighbor、50 维局部地形网格和 aggregation 特征，不包含 oracle 集合点。
- 地形网格通道为相对高度和风险，覆盖前后 `[-0.4, 1.2] m`、横向 `[-0.8, 0.8] m`；critic 仍为 54 维，并使用 5 维网格摘要。
- checkpoint 加载要求 schema、actor 输入维度和 critic 状态维度完全匹配；旧 `ego_v2_speed_angular` checkpoint 不自动迁移。
- centralized critic state 和 reward shaping 可以使用 oracle 信息；执行期 actor 不接收 `p*`、oracle 距离或 oracle 距离下降量。
- 动作接口固定为低维 `[rho, beta]`，再经局部子目标、直线轨迹和简化速度控制器转换为运动命令。
- 当前 proxy 动力学是 2D/2.5D kinematic unicycle 风格状态更新；没有质量、惯量、轮地接触、打滑、悬挂或 PhysX contact。
- `scripts/train_skrl_mappo.py` 使用 SKRL MAPPO 训练 proxy wrapper；`isaaclab-multi-agent` wrapper 只是接口层，不代表训练 loop 运行在 Isaac Sim / PhysX。
- exp016 已启用项目侧 `shared_joint` 更新：共享 Actor/Critic 只使用一个 optimizer，每个 rollout 合并四个 rover 的 Actor 样本并只更新一次 Critic。
- 当前 exp016 诊断配置把通信半径临时扩大到 `12 m`；这是训练诊断设置，不是最终通信约束。
- exp017 已完成 pure RL 连续 20M 长跑并通过 seed23 独立 strict eval；这是固定地图、单 seed proxy 结果，不代表随机地图泛化或多 seed 收敛。
- exp018 已加入每环境、每 episode reset 独立地形随机化，并把地形强度提高一档；完整测试、CPU/CUDA smoke 和随机地图渲染已通过。seed23 连续 20M 已完成，dmax 和 success 达标，但 collision / timeout 未通过 strict gate。
- exp019 已在 exp018 基础上完成两个诊断改造：success gate 新增最近邻安全间距 `0.42 m`，terrain reward 扩展到当前点到子目标的路径级风险。seed23 20M 工程链路和 5 轮独立 eval/GIF 已完成，但 strict gate 未通过。
- exp020 已在 exp019 基础上加入 terrain/safety-aware 子目标过滤器；过滤器稳定降低路径风险，但显著抑制集合进度。seed23 20M、5 轮独立 eval/GIF 和训练曲线已完成，strict gate 未通过。
- exp021 已完成 exp020 的课程化/软化 filter 迭代：前期保留 raw action，后期逐步增加 filter 介入概率和 score 权重，并加入 raw-risk / filter-deviation 辅助惩罚。seed23 20M、5 轮独立 eval、GIF、height map 和训练曲线已完成，strict gate 未通过。
- exp022 已完成 endpoint/path safety constrained curriculum filter 迭代：collision 被压到 strict 内，但集合进度塌缩，5 seed mean success `0.0139`、timeout `0.9699`，strict 未通过。
- exp023 已开始 soft progress-preserving filter 迭代：保留 exp021 的集合底座，移除 exp022 hard safety constraint / near-distance override，仅在预测真实碰撞时允许 collision override，并在 filter score 中显式惩罚远离可见邻居中心；完整 pytest、CPU smoke、CUDA smoke 已通过，seed23 20M 长训已作为 `exp023-soft-progress-filter-20m.service` 启动。

## Checkpoint 评估工作流

新增标准入口：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/<experiment>/<run_id>
```

该入口会根据 experiment YAML 的 `evaluation:` 配置执行：

1. proxy 独立评估，写入 `metrics/final_eval_proxy.json`；
2. proxy strict gate 判定；
3. 若配置允许且 proxy 通过，再低频触发 PhysX / Jackal tracking validation；
4. 写入 `metrics/checkpoint_status.json` 并更新 `run_manifest.json`。

checkpoint 状态只使用：

```text
candidate
proxy_passed
physx_evaluated
physx_passed
final_selected
```

## 已验证结果

| 实验 | 地形 | 方法 | 严格状态 | 当前解释 |
| --- | --- | --- | --- | --- |
| exp006 | 平地 proxy | BC + PPO | 通过 | 平地 proxy strict baseline，checkpoint 来自 PPO 阶段。 |
| exp008 | 弱 lunar crater 3D proxy | 弱 warm-start + PPO | 3 seeds 通过 | 当前最完整的 terrain-aware proxy baseline。 |
| exp009 | 强 lunar crater 3D proxy | 弱 warm-start + PPO | 未通过 | seed23 通过；seed31 失败；近期不继续堆 long-budget PPO。 |
| exp010 | 强 lunar crater 3D proxy | hold reward / safety 诊断 | 未通过 | success 可改善，但 collision/timeout gate 仍失败。 |
| exp012 | proxy SKRL-MAPPO CUDA 诊断 | action scale warmup probe | 未通过 | distance 有改善，但 strict gate 未通过。 |
| exp013 | proxy SKRL-MAPPO CUDA 诊断 | action scale ablation + teacher reachability | 未通过 | 当前小动作 100-step 配置对 teacher 也几乎不可达。 |
| exp014 | 弱 lunar crater proxy | 5×5 局部地形网格 CUDA probe | 工程验证通过；未做 strict | 新观测和训练链路有效，不能表述为策略收敛。 |
| exp015 | 偏弱中档 lunar crater proxy | SKRL MAPPO + BC20 | 2M screen 未通过 | 工程信号正常；dmax ratio 0.818、success 0、collision 0.124、timeout 0.876，因此未启动 8M。 |
| exp016 | 偏弱中档 lunar crater proxy | shared-joint MAPPO + local BC100 + comm12 | BC probe 未通过 | shared update 探针通过；BC-only dmax ratio 0.438、collision 0.0088、timeout 0.991，未启动 2M。 |
| exp017 | 固定偏弱中档 lunar crater proxy | shared-joint MAPPO pure RL + comm12 | seed23 strict 通过 | final dmax ratio 0.1318、success 0.9990、collision 0.00098、timeout 0；仍是 single-seed candidate。 |
| exp018 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + comm12 | 未通过 | seed23 20M 完成；final dmax ratio 0.1417、success 0.9609 通过，但 collision 0.0352、timeout 0.0088 未达 strict gate。 |
| exp019 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + safe success gate + path terrain risk | 未通过 | seed23 20M 完成；10240 checkpoint 有集合趋势但 collision 高，当前 best final eval success 0.0195、collision 0.0791、timeout 0.9023；5 seed 复验均值 success 0.0143、collision 0.0801、timeout 0.9082。 |
| exp020 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + terrain/safety subgoal filter | 未通过 | seed23 20M 完成；filter 将 5 seed path risk mean 从 raw 0.3815 降到 0.3187，但 success 0、collision 0.0498、timeout 0.9506，说明过滤器过强地牺牲了集合进度。 |
| exp021 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + curriculum terrain/safety subgoal filter | 未通过 | 课程化 filter 恢复集合进度：5 seed mean success 0.6361、dmax ratio 0.1460、timeout 0.1967，filtered path risk 0.3638；但 collision 0.1746，远高于 strict 0.02。 |
| exp022 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + endpoint/path safety constrained curriculum filter | 未通过 | 5 seed mean：dmax ratio 0.4719、success 0.0139、collision 0.0170、timeout 0.9699；说明 constrained filter 可压住碰撞，但过强地牺牲集合进度。 |
| exp023 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + soft progress-preserving subgoal filter | 运行中 | 针对 exp022 成功率塌缩：降低 filter 介入概率和安全权重，保留 visible-neighbor center / center progress 软约束，只对 raw collision 做 override；使用 `progress_preserving_long` 选 checkpoint。 |

历史完整 suite checkpoint：

```text
outputs/runs/exp_008_terrain3d/_suite/checkpoints/
```

这些 exp008 checkpoint 使用旧 observation schema，历史 strict 结论仍有效，但不能直接加载到当前 86 维 Actor。exp017 已产生当前 schema 的单 seed strict checkpoint；exp014 checkpoint 仍仅用于工程探针。

## 结果解释边界

- exp006 / exp008 是 proxy strict pass，不是 Isaac Lab 物理训练 pass。
- exp006 / exp008 的 checkpoint 属于旧 observation schema；结果可作为历史 baseline，但不能与新 Actor 接口直接混用。
- exp014 只通过有限值、参数更新、地形输入权重更新和动作非退化检查，不是 strict convergence pass。
- PhysX / Jackal 结果应写成“Jackal 在 PhysX 场景中的轨迹跟踪验证结果”或“proxy checkpoint 的高保真迁移 sanity check”，不能写成“物理环境训练结果”。
- exp017 可以表述为“固定地图、seed23、proxy pure RL 从零通过 strict gate”，不能扩展为多 seed、随机地图或 PhysX 收敛。
- exp018 可以表述为“随机增强地形下已获得稳定集合趋势和较高 success，但安全/超时 gate 未完全收敛”，不能写成随机地图 strict pass。
- exp019 可以表述为“成功区安全间距和路径级地形风险链路已接入并可训练/评估，但当前 reward 下策略仍在成功率、碰撞率和超时之间失衡”，不能写成安全地形策略收敛。
- exp020 可以表述为“子目标过滤器确实降低了路径风险，但当前 hard post-processing 过强，导致探索/集合进度塌缩”，不能写成地形规避策略成功。
- exp021 可以表述为“课程化 filter 恢复了集合趋势，但碰撞率显著过高”，不能写成随机地形 strict pass 或安全策略成功。
- exp022 可以表述为“endpoint/path safety constrained filter 把 collision 压到 strict 内，但 success/timeout 严重失败”，不能写成随机地形安全策略收敛。
- exp023 在训练完成前只能表述为“soft progress-preserving filter 修复假设/工程验证”，不能写成随机地形安全策略改善或收敛。
- GIF、截图和 TensorBoard 曲线只能用于展示和诊断；严格结论以 `_suite/metrics/strict_acceptance.json`、`metrics/final_eval_proxy.json` 和 `metrics/checkpoint_status.json` 为准。

## Jackal 跟踪验证

活跃 PhysX 脚本：

```text
scripts/evaluate_physx_jackal_tracking.py
```

默认测试：

- 平地 `straight/circle/sine` 跟踪，并可通过 `--tune-flat` 保存调参网格。
- 强三维地形使用 exp009 strong lunar crater 参数：`amplitude=0.16`、`crater_min_radius=0.45`、`crater_max_radius=1.25`、`crater_depth_to_diameter=0.18`。
- 结果以 `tracking_summary.json`、`timeseries.csv` 和 `tracking.png` 为准。

本轮 Jackal tracking 输出：

```text
outputs/runs/physx_jackal_tracking/asset_smoke_jackal/
outputs/runs/physx_jackal_tracking/flat_tuned_final_v2/
outputs/runs/physx_jackal_tracking/strong_lunar_crater_final_v2/
```

平地正式结果通过默认阈值：

| profile | rmse_cross_track_m | max_cross_track_m | path_completion_ratio | max_tilt_deg | status |
| --- | ---: | ---: | ---: | ---: | --- |
| straight | 0.078 | 0.193 | 0.971 | 7.9 | pass |
| circle | 0.159 | 0.292 | 0.971 | 19.5 | pass |
| sine | 0.169 | 0.275 | 0.972 | 8.0 | pass |

强三维地形正式结果未通过默认阈值，主要失败项是完成率和横向误差；最大 tilt 仍低于 35 度：

| profile | path_offset_xy | rmse_cross_track_m | max_cross_track_m | path_completion_ratio | max_tilt_deg | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| straight | `[-0.5, -0.5]` | 0.579 | 0.883 | 0.072 | 23.8 | fail |
| circle | `[1.0, -2.0]` | 0.647 | 1.117 | 0.496 | 33.3 | fail |
| sine | `[0.5, 2.0]` | 0.737 | 1.206 | 0.076 | 28.5 | fail |

强地形失败应解释为“当前 Jackal 低层跟踪控制在 exp009 strong mesh 上仍不足”，不是 proxy 集合任务失败，也不是 Isaac 训练失败。

## 下一步

1. 当前进行 exp023：用 soft progress-preserving filter 验证是否能在不产生 exp022 安全 standoff 的情况下压低 exp021 的碰撞率。
2. 保留 exp017 作为固定地图 pure RL baseline，保留 exp018/exp019/exp020/exp021/exp022 作为随机地形 failure analysis，不把它们扩写为多 seed 或 PhysX 收敛。
3. 如果 exp023 仍失败，下一步应转向 action representation / planner projection / success geometry，而不是继续加大 post-processing filter 权重。
4. 为 PhysX / Jackal 后续接入同布局 raycast / height scanner，保持 proxy 与高保真观测接口一致。
