# 实验计划

## 当前主线

当前近期主线从 strong terrain 训练诊断切回环境搭建与工程闭环。目标是让 Isaac Sim / Isaac Lab / SKRL / 本地任务包的安装、导入、最小训练、验证和展示路径可重复。

- `exp008` 保持为当前已验证的 3-seed terrain-aware baseline。
- `exp009` 作为 strong terrain 诊断实验记录，不作为 strict success。
- `exp010` 作为 hold/safety 短程修复诊断记录，不继续扩展。
- PhysX / Isaac Sim 继续作为 sanity check 和展示层，近期目标是跑通 headless/render 工程链路，不进入主训练 loop。
- 新训练和评估默认写入 `outputs/runs/<experiment>/<run>/`，不要继续扩散到旧的 `outputs/logs/`、`outputs/checkpoints/` 等平铺目录。

严格结论以机器可读结果为准，优先读取：

```text
outputs/runs/<experiment>/_suite/metrics/strict_acceptance.json
outputs/runs/<experiment>/_suite/metrics/suite_summary.json
outputs/runs/<experiment>/<run>/metrics/final_eval_proxy.json
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
- seeds `23, 31, 47` 均通过 strict gate。
- 后续改动应避免破坏该 baseline，并在需要时用 exp008 suite 做回归对照。

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

目前先不继续 seed31 失败 episode 回放、动作表示原型或 terrain curriculum 短 run。后续恢复训练研究时，再优先区分 timeout 来自：

1. 地形速度缩放导致到达过慢。
2. 成功区附近 dispersion 不稳定。
3. speed hold 条件未稳定满足。
4. `[rho, beta]` 单步子目标动作表达能力不足。

## 下一步工程验收

近期验收不以 reward 收敛为目标，只确认安装、接口和最小闭环可用：

1. 验证 `.venv_isaaclab` 中 `torch`、`isaacsim`、`skrl` 和 `lunar_rover_tasks` 可导入。
2. 验证 `source/lunar_rover_tasks` 可 editable install。
3. 跑通 `scripts/validate_first_stage.py --config configs/experiment/exp_001_minimal.yaml --device cpu --steps 32`。
4. 跑通 `scripts/train.py --backend skrl --config configs/experiment/exp_001_minimal.yaml --device cpu --timesteps 128`。
5. 跑通 `scripts/debug_env.py --steps 50`、`scripts/debug_observation.py`、`scripts/debug_reward.py`。
6. 跑通 `scripts/evaluate_physx_four_jetbots.py` 的 headless/render sanity 路径，并记录产物路径。

如果新增工程 smoke run，应优先写入 `outputs/runs/<experiment>/<run_id>/` 并更新 runbook；不要把 partial train、GIF 或 TensorBoard 曲线写成 strict success。

## 历史 smoke 检查

以下内容属于基础环境回归检查，近期重新提升为工程验收的一部分：

1. 运行单元测试。
2. 运行 `scripts/debug_env.py --steps 200`。
3. 运行 `scripts/debug_observation.py`。
4. 运行 `scripts/debug_reward.py`。
5. 运行 `scripts/train.py --config configs/experiment/exp_001_minimal.yaml --device cpu --timesteps 128`。

`scripts/train.py` 默认使用 SKRL `MAPPO` backend。紧凑本地 trainer 仍可通过 `scripts/train.py --backend smoke` 用于快速调试。
