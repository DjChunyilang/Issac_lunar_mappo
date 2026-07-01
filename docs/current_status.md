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
- exp023 已完成 soft progress-preserving filter 迭代：success 从 exp022 的 `0.0139` 回升到 `0.3027`，但 collision `0.2295`、timeout `0.4717`，strict 未通过；失败原因是 static endpoint/path safety 未预测可见邻居同步运动。
- exp024 已完成 mutual path safety filter 迭代：在 exp023 基础上把可见邻居 raw subgoal path 作为动态障碍，按相同时间采样比较候选路径；post-hoc 使用 `success_progress_long` 重选 `ppo_timestep_010240.pt` 为 best，seed1023 final eval 为 dmax ratio `0.1397`、success `0.8398`、collision `0.0674`、timeout `0.0947`，strict 未通过但显著优于 exp023。
- exp025 已完成 dense mutual path safety filter 迭代：基于 exp024 加密 mutual/path safety 采样到 9 点，并适度提高 path/mutual collision 权重；best `ppo_timestep_009216.pt` final eval 为 dmax ratio `0.1434`、success `0.8525`、collision `0.0449`、timeout `0.1035`，strict 未通过。相对 exp024 collision 降低，但 timeout 未改善。
- exp026 已完成 hold-zone filter 诊断：过早/过宽的 `hold_zone_rho/spacing` cost 把 success 从 exp025 的 `0.8525` 拉低到 `0.7529`，collision `0.0615`、timeout `0.1865`，strict 未通过。
- exp027 已完成 strict hold-zone filter 诊断：把 hold-zone activation 收窄到真正 success dmax/dispersion 附近后避免了 exp026 的明显退化，但 final eval success `0.8418`、collision `0.0498`、timeout `0.1123`，未优于 exp025。
- exp028 已完成 hold reward 诊断：回退到 exp025 dense mutual filter，只强化 `success_hold_step=4.0`、`success_bonus=45`、`timeout_penalty=18`；final eval success `0.8691`、collision `0.0469`、timeout `0.0889`，是 exp026–029 中最好但仍未 strict。
- exp029 已完成 hold reward + stronger safety 诊断：在 exp028 基础上加强 path/mutual collision filter 和终端碰撞惩罚，final eval success `0.8262`、collision `0.0557`、timeout `0.1221`，说明继续加安全权重会牺牲成功并未压低真实碰撞。
- exp030 已完成低层 control safety projection 诊断：回到 exp028 主体，只在 `compute_control()` 后、`_integrate()` 前加入相对速度安全投影和 success-zone damping；final eval success `0.8330`、collision `0.0313`、timeout `0.1357`，collision 明显低于 exp028，但投影过强导致 success/timeout 退化。
- exp031–exp034 已完成 control safety 投影条件迭代：简单调弱、closing-only、directional scale 和 directional mask 都未 strict；其中 exp034 的 mask 版本把 success 拉回 `0.8828`、timeout 降到 `0.0840`，但 collision `0.0361` 仍失败。
- exp035–exp036 已完成 directional mask buffer 与 stronger hold/timeout shaping：exp035 首次让 success `0.9072` 和 collision `0.0127` 同时达标；exp036 进一步到 success `0.9336`、collision `0.0088`、timeout `0.0586`，剩余瓶颈转为 timeout/hold。
- exp037 已完成 260-step episode/eval 诊断：timeout 从 exp036 的 `0.0586` 降到 `0.0410`，但 collision 反弹到 `0.0352`，说明单纯延长 episode 会暴露末段碰撞。
- exp038 已完成 success-zone stabilizer + 320-step episode/eval：修正 best 后 final eval success `0.9756`、collision `0.0137`、timeout `0.0107`；当前随机地形最佳候选，strict 只剩 timeout gate 失败。
- exp039/exp040 是基于 exp038 best 的诊断复评，不建议长训：hard near stabilizer 和 stronger soft hold stabilizer 都使 timeout 或 collision 差于 exp038。
- exp041 已完成 hold-zone override 诊断与 CPU/CUDA smoke：在 exp038 best 上复评得到 success `0.9795`、collision `0.0107`、timeout `0.0098`，略优于 exp038，是下一轮长训练候选，但尚不是从头训练结果。

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
| exp023 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + soft progress-preserving subgoal filter | 未通过 | seed23 final eval：dmax ratio 0.1789、success 0.3027、collision 0.2295、timeout 0.4717；缓解 exp022 standoff，但 static filter 未处理同步运动碰撞。 |
| exp024 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + mutual path safety subgoal filter | 未通过 | post-hoc best `10240`：dmax ratio 0.1397、success 0.8398、collision 0.0674、timeout 0.0947；mutual path filter 明显改善 success/collision 平衡，但 strict 安全和 timeout 仍未达标。 |
| exp025 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + dense mutual path safety filter | 未通过 | best `9216`：dmax ratio 0.1434、success 0.8525、collision 0.0449、timeout 0.1035；dense mutual filter 继续降低碰撞，但仍未达到 strict 安全/timeout gate。 |
| exp026 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + hold-stable subgoal filter | 未通过 | best final eval：success 0.7529、collision 0.0615、timeout 0.1865；hold-zone 介入过早，明显压制集合进度。 |
| exp027 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + strict hold-zone filter | 未通过 | best final eval：success 0.8418、collision 0.0498、timeout 0.1123；严格触发避免 exp026 退化，但仍不优于 exp025。 |
| exp028 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + dense mutual filter + stronger hold reward | 未通过 | best final eval：success 0.8691、collision 0.0469、timeout 0.0889；当前随机地形安全/hold 方向最好结果，但安全 gate 仍失败。 |
| exp029 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + exp028 + stronger safety penalties/filter weights | 未通过 | best final eval：success 0.8262、collision 0.0557、timeout 0.1221；加强安全权重反而退化，不能作为下一步方向。 |
| exp030 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + exp028 + low-level control safety projection | 未通过 | best final eval：success 0.8330、collision 0.0313、timeout 0.1357；动态控制投影能降低碰撞，但当前触发过强，牺牲 success/timeout。 |
| exp031 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + narrow/weak control safety projection | 未通过 | best final eval：success 0.8105、collision 0.0449、timeout 0.1455；简单调弱没有恢复 success，也丢失 exp030 的安全收益。 |
| exp032 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + closing-only control safety projection | 未通过 | best final eval：success 0.8379、collision 0.0361、timeout 0.1279；closing-only 略优于 exp031，但仍未达标。 |
| exp033 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + directional agent-scale projection | 未通过 | best final eval：success 0.8154、collision 0.0488、timeout 0.1387；方向性连续缩放没有带来安全收益。 |
| exp034 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + directional mask projection | 未通过 | best final eval：success 0.8828、collision 0.0361、timeout 0.0840；mask 版本恢复部分 success/timeout，但 collision 仍超 strict。 |
| exp035 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + directional mask buffer | 未通过 | best final eval：success 0.9072、collision 0.0127、timeout 0.0811；success/collision 同时达标，timeout 成主瓶颈。 |
| exp036 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + directional mask + stronger hold/timeout shaping | 未通过 | best final eval：success 0.9336、collision 0.0088、timeout 0.0586；继续改善 timeout，但 strict 仍失败。 |
| exp037 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + directional mask + 260-step episode/eval | 未通过 | best final eval：success 0.9238、collision 0.0352、timeout 0.0410；延长 episode 降 timeout，但 collision 反弹。 |
| exp038 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + success-zone stabilizer + 320-step episode/eval | 未通过 | 修正 best 后 final eval：success 0.9756、collision 0.0137、timeout 0.0107；当前随机地形最佳，strict 只剩 timeout 失败。 |
| exp039 | 随机增强 lunar crater proxy | exp038 checkpoint + hard near stabilizer 诊断 | 未长训 | 复评 success 0.9424、collision 0.0254、timeout 0.0322，差于 exp038；不建议按原样长训。 |
| exp040 | 随机增强 lunar crater proxy | exp038 checkpoint + stronger soft hold stabilizer 诊断 | 未长训 | 复评 success 0.9658、collision 0.0186、timeout 0.0166，timeout 差于 exp038；不建议按原样长训。 |
| exp041 | 随机增强 lunar crater proxy | exp038 checkpoint + hold-zone override 诊断 | 待长训 | 复评 success 0.9795、collision 0.0107、timeout 0.0098，略优于 exp038；下一轮长训候选。 |

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
- exp023 可以表述为“soft progress filter 缓解了 exp022 的 standoff，但 collision/timeout 仍严重失败”，不能写成随机地形安全策略改善或收敛。
- exp024 可以表述为“mutual path safety 明显改善 exp023 的动态路径冲突，但 collision/timeout 仍未过 strict”，不能写成随机地形安全策略收敛。
- exp025 可以表述为“dense mutual path safety 相对 exp024 进一步降低 collision，但仍未解决末段 hold / timeout 稳定性”，不能写成随机地形安全策略收敛。
- exp026/exp027 可以表述为“hold-zone filter 诊断未改善 exp025，过早介入会压制集合”，不能写成 hold 稳定成功。
- exp028 可以表述为“强化 success hold reward 提高了 success/timeout，是 exp026–029 中最好的随机地形结果”，但 collision 仍超 strict，不能写成安全收敛。
- exp029 可以表述为“继续加强安全权重没有降低真实 collision，反而牺牲 success/timeout”，不能写成安全改善。
- exp030 可以表述为“低层动态控制投影降低 collision 但牺牲 success/timeout”，不能写成安全收敛或 strict 改善。
- exp031–exp034 可以表述为“控制层投影条件和方向性 mask 诊断”，不能写成随机地形安全收敛；exp034 是方向性 mask 的有效拐点，但 collision 仍失败。
- exp035/exp036 可以表述为“success/collision 已同时过门槛但 timeout 仍失败”，不能写成 strict pass。
- exp037 可以表述为“延长 episode 降 timeout 但导致 collision 反弹”，不能写成单纯时间预算不足。
- exp038 可以表述为“当前随机地形最佳候选，strict 只剩 timeout 尾部未过”，不能写成 strict pass。
- exp039/exp040 只是 exp038 checkpoint 复评诊断，不能写成长训练结果。
- exp041 可以表述为“hold-zone override 在 exp038 checkpoint 上略有改善，是下一轮候选”，不能写成 exp041 长训练完成。
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

1. 当前随机地形最佳综合候选是 exp038：success/collision 已过 strict，仅 timeout `0.0107` 未过。
2. exp038 的剩余 timeout 不是整体不可达，而是少量 episode 卡在 `collision_distance=0.28 m` 与 `success_thresholds.min_pairwise_distance=0.42 m` 之间的最近邻灰区，hold count 接近完成但未稳定达标。
3. 不建议继续全局加硬 near/hold filter；exp039/exp040 已说明会扰动 success 或 timeout。
4. 下一轮优先从 exp041 出发，从随机初始化长训 hold-zone override；若仍只剩极少 timeout，再考虑更细的末端 pairwise spacing controller。
5. 保留 exp017 作为固定地图 pure RL baseline，保留 exp018–exp041 作为随机地形 failure analysis，不把它们扩写为多 seed 或 PhysX 收敛。
6. 为 PhysX / Jackal 后续接入同布局 raycast / height scanner，保持 proxy 与高保真观测接口一致。
