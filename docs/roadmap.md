# 路线图

## 立即处理

1. 将 exp008 保持为当前已验证的 3-seed terrain-aware proxy 结果。
2. 将 exp009 视为强地形诊断实验，而不是严格成功结果。
3. 暂停 exp009/exp010 后续强地形失败诊断和 long-budget PPO。
4. 按 [architecture/overall_plan_v3.md](architecture/overall_plan_v3.md) 将 proxy 降级为接口验证层，优先回到 Isaac Lab + SKRL-MAPPO 主线。

## 近期工作

- 固化 `.venv_isaaclab` 安装检查，覆盖 Python、CUDA/NVIDIA driver、Isaac Sim 6.0、Isaac Lab v3.0.0-beta、SKRL 和 `source/lunar_rover_tasks` editable install。
- 跑通 `scripts/validate_first_stage.py` 的短验证，确认 proxy core、观测、奖励、轨迹、图和 GIF 链路可重复。
- 跑通 `scripts/train.py --backend skrl` 的短 MAPPO smoke，验收 SKRL wrapper、multi-agent action/observation 和 centralized critic state。
- 跑通 `scripts/debug_env.py`、`scripts/debug_observation.py`、`scripts/debug_reward.py` 作为基础回归。
- 跑通 `scripts/evaluate_physx_four_jetbots.py` 的 headless/render sanity 路径，PhysX 继续作为验证和展示层，不进入主训练 loop。
- 新 smoke 输出逐步迁到 `outputs/runs/`；旧 `outputs/logs` / `outputs/checkpoints` 仅保留兼容历史脚本。

## 中长期工作

- 在 Isaac Lab 物理闭环稳定后，再恢复强地形动作表示或 terrain curriculum 研究。
- 将真实 rover articulation asset/control 接口接入 Isaac Lab 任务，替换当前 proxy unicycle 执行层。
- 构建可重复的报告生成器，从 `_suite/metrics/*.json` 自动更新实验文档。
