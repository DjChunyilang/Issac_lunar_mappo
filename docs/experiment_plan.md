# 实验计划

1. 运行单元测试。
2. 运行 `scripts/debug_env.py --steps 200`。
3. 运行 `scripts/debug_observation.py`。
4. 运行 `scripts/debug_reward.py`。
5. 运行 `scripts/train.py --config configs/experiment/exp_001_minimal.yaml --device cpu --timesteps 128`。
6. 在 proxy 任务稳定前，后续 ablation 默认保持关闭。

`scripts/train.py` 默认使用 SKRL `MAPPO` backend。紧凑本地 trainer 仍可通过 `scripts/train.py --backend smoke` 用于快速调试。
