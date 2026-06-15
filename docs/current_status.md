# 当前状态

## 当前主线

- 当前主设计口径已切换为“高吞吐 proxy 训练 + Isaac Sim / Isaac Lab / PhysX 高保真闭环评估”。`multi_rover_design_revision_proxy_train_isaac_eval.md` 是本轮路线修订来源。
- 训练主环境仍是 PyTorch / torch-vectorized proxy 环境，用于 MAPPO / PPO 采样、奖励调试、观测接口验证和大规模对照实验。
- Isaac Sim / Isaac Lab / PhysX 不作为当前主训练 loop，而作为 high-fidelity validation、迁移 sanity check、失效分析和可视化展示平台。
- 当前 PhysX 层使用 Clearpath Jackal 作为活跃轮式资产，已替换旧占位资产。Jackal tracking 可验证轮式控制、强三维地形 mesh、姿态稳定性和输出链路，但不能证明真实月球车越障、轮壤接触或低重力动力学已经完成。
- 视觉观测不进入 policy input；地形以低维结构化特征进入策略。
- 生成结果写入 `outputs/runs/`，并由 git 忽略。

## 当前接口状态

- actor observation schema 为 `ego_v2_speed_angular`，包含 ego、neighbor、terrain、aggregation 特征，不包含 oracle 集合点。
- centralized critic state 和 reward shaping 可以使用 oracle 信息；执行期 actor 不接收 `p*`、oracle 距离或 oracle 距离下降量。
- 动作接口固定为低维 `[rho, beta]`，再经局部子目标、直线轨迹和简化速度控制器转换为运动命令。
- 当前 proxy 动力学是 2D/2.5D kinematic unicycle 风格状态更新；没有质量、惯量、轮地接触、打滑、悬挂或 PhysX contact。
- `scripts/train_skrl_mappo.py` 使用 SKRL MAPPO 训练 proxy wrapper；`isaaclab-multi-agent` wrapper 只是接口层，不代表训练 loop 运行在 Isaac Sim / PhysX。

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

当前推荐的完整 suite checkpoint：

```text
outputs/runs/exp_008_terrain3d/_suite/checkpoints/
```

## 结果解释边界

- exp006 / exp008 是 proxy strict pass，不是 Isaac Lab 物理训练 pass。
- PhysX / Jackal 结果应写成“Jackal 在 PhysX 场景中的轨迹跟踪验证结果”或“proxy checkpoint 的高保真迁移 sanity check”，不能写成“物理环境训练结果”。
- 当前较好结果来自 weak warm-start + PPO，不能表述为 pure RL 从零严格收敛。
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

1. 用 `scripts/run_checkpoint_evaluation.py` 复评当前候选 checkpoint，补齐 `checkpoint_status.json`。
2. 维持 exp008 作为当前 proxy baseline，不继续默认追加 exp012/exp013 long-budget proxy PPO。
3. 扩展 PhysX / Jackal 跟踪测试样本量，优先记录 tracking error、path completion、tilt 和控制吞吐。
4. 后续若高保真评估发现系统性迁移失败，再考虑 Isaac-based fine-tuning、domain randomization、真实 rover asset 或更高保真动力学模型。
