# 实验计划

## 当前主线

当前实验主线是“proxy 训练 + checkpoint 级 high-fidelity closed-loop evaluation”：

- `exp008` 保持为当前已验证的 3-seed terrain-aware proxy baseline。
- `exp009` / `exp010` 作为 strong terrain 诊断记录，近期不继续默认扩展。
- `exp012` / `exp013` 作为 SKRL-MAPPO proxy CUDA 与 action-scale 诊断，不作为 strict success。
- `exp017` 是固定地图、seed23、shared-joint MAPPO pure RL 的当前 schema strict pass 单 seed baseline。
- `exp018` 是随机增强地形的 20M 单 seed candidate；dmax/success 达标，但 collision/timeout 未过 strict gate。
- `exp019` 已完成随机增强地形安全/路径风险诊断；工程链路正常，但 strict 未通过，说明当前软路径风险惩罚和安全 gate 组合仍不足。
- `exp020` 已完成 terrain/safety-aware 子目标过滤器诊断；路径风险明显降低，但集合进度被抑制，strict 未通过。
- `exp021` 已完成课程化/软化 filter 诊断；集合进度恢复，但 collision 显著升高，strict 未通过。
- `exp022` 已完成 endpoint/path safety constrained curriculum filter 诊断；collision 被压到 strict 内，但 success/timeout 明显失败，说明 hard constrained post-processing 过强。
- `exp023` 已完成 soft progress-preserving filter 诊断；恢复部分集合进度，但 collision / timeout 仍失败，说明 static endpoint/path safety 不足。
- `exp024` 已完成 mutual path safety filter 诊断；success 提升到 `0.8398`、collision 降到 `0.0674`，但 strict 安全和 timeout 仍未达标。
- `exp025` 已完成 dense mutual path safety filter 诊断；collision 相对 exp024 降低，但 success/timeout/安全 strict 仍未过，下一步应转向末段 hold / success-zone 稳定。
- PhysX / Isaac Sim 作为 checkpoint 级高保真闭环评估和展示层，不进入当前主训练 loop。
- 新训练和评估默认写入 `outputs/runs/<experiment>/<run>/`。

严格结论以机器可读结果为准，优先读取：

```text
outputs/runs/<experiment>/_suite/metrics/strict_acceptance.json
outputs/runs/<experiment>/_suite/metrics/suite_summary.json
outputs/runs/<experiment>/<run>/metrics/final_eval_proxy.json
outputs/runs/<experiment>/<run>/metrics/checkpoint_status.json
```

不要从 GIF、截图、单个 checkpoint 文件名或 TensorBoard 曲线推断 strict pass。

## 已确认基线

`exp008` 是当前推荐的完整 terrain-aware proxy 基线：

```text
outputs/runs/exp_008_terrain3d/_suite/
```

结论：

- 弱 lunar crater 3D proxy。
- 弱 warm-start + PPO。
- seeds `23, 31, 47` 均通过 proxy strict gate。
- 后续改动应避免破坏该 baseline，并在需要时用 exp008 suite 做回归对照。

当前 86 维局部地形网格 schema 下，`exp017` 可作为固定地图 single-seed baseline：

```text
outputs/runs/exp017_shared_mappo_pure_rl_comm12/
```

结论：

- 固定偏弱中档 lunar crater proxy。
- shared-joint MAPPO pure RL，不使用 BC。
- seed23 final eval 通过 strict gate。
- 不能替代 exp008 的 3-seed baseline，也不能证明随机地图泛化。

随机地图主诊断是 `exp018`：

```text
outputs/runs/exp018_randomized_terrain_pure_rl/pure_rl_seed23_20m_randomized_terrain/
```

结论：

- 每环境、每 episode reset 独立随机地形。
- 20M 后 dmax 和 success 达标。
- collision `0.0352`、timeout `0.0088` 未通过 strict gate。
- 下一轮优先改安全和路径级风险，而不是只放大 terrain penalty。

已完成的安全/路径风险诊断是 `exp019`：

```text
outputs/runs/exp019_randomized_terrain_safe_path_risk/
```

结果：

- success gate 已要求最近邻距离 `>= 0.42 m`，显式避开 `collision_distance=0.28 m`。
- terrain reward 已新增当前点到子目标直线路径的 mean/max risk 和高度变化。
- seed23 20M 训练、候选评估、5 轮独立 eval 和 GIF 均已完成。
- 当前 best checkpoint 5 seed 均值：dmax ratio `0.4186`、success `0.0143`、collision `0.0801`、timeout `0.9082`，strict 未通过。
- 10240 checkpoint 的 success 可到 `0.6201`，但 collision `0.1279`，说明后期学习出了集合趋势但安全不达标。

下一轮优先方向：

- 不再只放大 terrain weight；改为把 path risk 前移到子目标候选过滤、局部 planner score 或不可达路径约束。
- 重新成套协调 success 半径、安全半径、episode 长度和 collision penalty，避免“会集合但撞车”和“安全但超时”的两极化。

已完成的子目标过滤诊断是 `exp020`：

```text
outputs/runs/exp020_randomized_terrain_subgoal_filter/
```

结果：

- actor 输出后、轨迹生成前加入 10 候选子目标过滤器，按 path risk、subgoal risk 和 endpoint safety 选择候选。
- seed23 20M 训练、候选评估、5 轮独立 eval、GIF 和训练曲线均已完成。
- 过滤器有效：5 seed 均值 raw path risk `0.3815`，filtered path risk `0.3187`，risk reduction `0.0628`。
- 策略失败：5 seed 均值 dmax ratio `0.3752`、success `0.0`、collision `0.0498`、timeout `0.9506`，strict 未通过。
- 当前结论是 hard filter 过强，牺牲了集合进度；下一轮应做课程化/软化，而不是继续增加过滤强度。

已完成的课程化 filter 诊断是 `exp021`：

```text
outputs/runs/exp021_randomized_terrain_filter_curriculum/
```

结果：

- 保持 exp020 的 pure RL、shared-joint MAPPO、随机增强地形和 `12 m` 通信半径。
- 课程化 filter 前 `2048` timesteps 不替换 action，后续 `4096` timesteps 内把 `apply_probability` 从 `0.0` 线性升到 `0.60`，把 `score_scale` 从 `0.15` 升到 `0.75`。
- seed23 20M 训练、候选评估、5 轮独立 eval、GIF 和训练曲线已完成。
- 5 seed mean：dmax ratio `0.1460`、success `0.6361`、collision `0.1746`、timeout `0.1967`。
- filter 仍降低路径风险：raw path risk `0.3871`，filtered path risk `0.3638`，risk reduction `0.0233`。

判读：

- exp021 相比 exp020 恢复了集合和降低了 timeout，但 collision 明显恶化。
- 下一步不宜继续只调 filter 强度或 terrain reward；应让 endpoint safety 更直接地参与 action/planner 或 policy loss。

已完成的 endpoint/path safety constrained filter 诊断是 `exp022`：

```text
outputs/runs/exp022_randomized_terrain_endpoint_safety_filter/
```

结果：

- 保持 exp021 的 pure RL、shared-joint MAPPO、随机增强地形和 `12 m` 通信半径。
- 子目标 filter 改为 `terrain_safe_candidate_constrained_curriculum`。
- 候选从 10 个扩展到 28 个：`rho_scales=[0.45, 0.70, 0.90, 1.0]`，`beta_offsets_deg=[-45, -30, -15, 0, 15, 30, 45]`。
- warmup 前仍不替换 action；warmup 后对 endpoint/path unsafe raw action 允许 safety override。
- 加入 endpoint safety、path collision 和 visible-neighbor center progress hard constraint，避免只绕远但不集合。
- seed23 20M 训练、候选评估、5 轮独立 eval、GIF 和训练曲线已完成。
- 5 seed mean：dmax ratio `0.4719`、success `0.0139`、collision `0.0170`、timeout `0.9699`。
- filter 很强：raw path risk `0.3379`，filtered path risk `0.2737`，risk reduction `0.0642`，applied fraction `0.6165`。

判读：

- strict gate 未通过。
- collision 从 exp021 的 `0.1746` 降到 `0.0170`，说明 endpoint/path safety constraint 有效。
- success 从 exp021 的 `0.6361` 降到 `0.0139`，timeout 从 `0.1967` 升到 `0.9699`，说明 constrained filter 仍过强。
- 下一轮应转向 policy/action representation、planner projection 或 success geometry，而不是继续放大后处理权重。

`exp023_randomized_terrain_soft_progress_filter` 已完成：

- dmax ratio `0.1789` 通过，但 success `0.3027`、collision `0.2295`、timeout `0.4717` 未通过 strict。
- 和 exp022 相比，soft filter 恢复部分集合进度；和 exp021 相比，collision 更高。
- 失败原因是 static endpoint/path safety 只看邻居当前位置，未预测可见邻居 raw path 的同步运动。

`exp024_randomized_terrain_mutual_path_filter` 已完成：

- post-hoc 使用 `success_progress_long` 重选 10240 checkpoint。
- seed1023 final eval：dmax ratio `0.1397`、success `0.8398`、collision `0.0674`、timeout `0.0947`。
- mutual path filter 明显改善 exp023 的同步路径冲突：`filter_raw_mutual_path_collision_violation_mean=0.0341`，filtered 后为 `0.000879`。
- strict gate 仍未通过，剩余问题主要是 late-stage collision 和少量 timeout。

`exp025_randomized_terrain_dense_mutual_filter` 已完成：

- 仍使用 exp024 的 pure RL、shared-joint MAPPO、随机增强地形、`12 m` 通信半径和 20M budget。
- filter mode 仍为 `terrain_safe_candidate_mutual_progress_curriculum`，不启用 exp022 式 hard endpoint/path safety constraint。
- 把 `path_samples` 从 5 加密到 9，并把候选集合改为 `rho_scales=[0.60, 0.80, 1.0, 1.08]` × `beta_offsets_deg=[-40, -25, -12.5, 0, 12.5, 25, 40]`。
- 适度提高 path/mutual collision 权重：`path_collision_weight=450.0`、`mutual_path_collision_weight=1200.0`。
- checkpoint 选择使用 `success_progress_long`，避免再次选中低 collision 但低 success 的早期 checkpoint。
- best 为 `ppo_timestep_009216.pt`；final eval：dmax ratio `0.1434`、success `0.8525`、collision `0.0449`、timeout `0.1035`。
- 相对 exp024，success 小幅提高、collision 降低，但 timeout 未改善，strict 仍未通过。

当前下一轮方向：

- 不再单纯加大 path/mutual collision 权重。
- 优先在接近 success zone 时加入速度/步长阻尼、hold-aware filter score 或轻量 planner projection。
- 目标是解决 exp025 暴露的末段问题：`max_success_hold_count_mean≈7.22/8` 接近但不稳定，collision 多发生在 `first_collision_step_mean≈176/220` 的后段。

## Checkpoint 复评计划

候选 checkpoint 使用统一入口：

```bash
.venv_isaaclab/bin/python scripts/run_checkpoint_evaluation.py \
  --config configs/experiment/<config>.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --run-dir outputs/runs/<experiment>/<run_id>
```

目标产物：

```text
metrics/final_eval_proxy.json
metrics/checkpoint_status.json
physx/metrics/<terrain>_headless.json
run_manifest.json
```

其中 PhysX 评估只在配置允许且 proxy gate 满足时触发。

## 暂缓的诊断线

`exp009` 只作为 strong terrain 诊断实验：

```text
outputs/runs/exp_009_terrain3d_strong/_suite/
```

已知结论：

- seed23 通过 strict gate。
- seed31 未通过 success/timeout gate。
- seed47 未运行，因为 3-seed strict acceptance 已经不可能成立。

seed31 当前失败模式：

```text
dmax_reduction_ratio: 0.1819  # 通过
success_rate: 0.8740          # 未通过
collision_rate: 0.0049        # 通过
timeout_rate: 0.1250          # 未通过
```

目前先不继续 seed31 失败 episode 回放、动作表示原型或 terrain curriculum 短 run。后续恢复训练研究时，再优先区分 timeout 来自地形速度缩放、dispersion 不稳定、speed hold 不稳定或 `[rho, beta]` 表达能力不足。

## 工程验收

近期验收包含：

1. `.venv_isaaclab/bin/python -m pytest -q -ra`。
2. `scripts/validate_first_stage.py --config configs/experiment/exp_001_minimal.yaml --device cpu --steps 32`。
3. `scripts/run_checkpoint_evaluation.py` 在 mock 或真实 run 上生成 `checkpoint_status.json`。
4. `scripts/evaluate_physx_jackal_tracking.py` 的 headless/render sanity 路径可手动运行，并记录产物路径。

如果新增工程 smoke run，应优先写入 `outputs/runs/<experiment>/<run_id>/` 并更新 runbook；不要把 partial train、GIF 或 TensorBoard 曲线写成 strict success。
