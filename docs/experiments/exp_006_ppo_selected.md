# exp_006 PPO 阶段选择基线

## 目的

建立平地 / proxy 严格基线，并要求最终 best checkpoint 必须来自 PPO 阶段，而不是 warm-up 阶段。

## 配置

```text
configs/experiment/exp_006_ppo_selected_bc_ppo.yaml
```

关键参数：

```text
num_envs: 1024
rollout_steps: 128
total_env_steps: 2,000,000
eval_num_envs: 512
eval_steps: 160
bc_steps: 300
learning_rate: 5e-5
clip_epsilon: 0.15
ppo_epochs: 2
best_source: ppo
required_best_phase: ppo
```

## 结果

BC+PPO 组在 seeds `23, 31, 47` 上通过严格验收。最佳结果为 seed23 update 15：

```text
dmax_reduction_ratio: 0.1438
success_rate: 1.0
collision_rate: 0.0
timeout_rate: 0.0
```

## 说明

这是 warm-start 后由 PPO 阶段选出的结果，不是 pure RL 从零训练结果。
该实验是 proxy strict baseline，不是 Isaac / PhysX 物理训练结果；如需进入高保真闭环评估，应先用 `scripts/run_checkpoint_evaluation.py` 为候选 checkpoint 生成 `metrics/checkpoint_status.json`。
