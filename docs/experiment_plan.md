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
