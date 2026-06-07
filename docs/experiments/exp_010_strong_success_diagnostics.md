# exp_010 强地形成功门诊断与短程修复

## 目的

诊断 exp009 seed31 在强 lunar crater 3D proxy 上未通过 `success_rate` 和 `timeout_rate` strict gate 的原因，区分失败来自 `dmax`、`dispersion`、`speed` 还是连续 `hold_steps`。
随后按短程上限验证 hold reward / safety reward 变体是否足以把 seed31 推过 strict gate。

## 配置

诊断评估复用原 checkpoint 与原地形配置：

```text
configs/experiment/exp_009_terrain3d_strong_weak_warmstart.yaml
configs/experiment/exp_008_terrain3d_weak_warmstart_select.yaml
```

新增短程修复配置：

```text
configs/experiment/exp_010_strong_success_hold_reward.yaml
configs/experiment/exp_010_strong_success_hold_reward_safety_retry.yaml
```

该配置保持 exp009 strong terrain 参数不变，启用 `reward.coefficients.success_hold_step: 1.5`，用于 seed31 从 exp009 best checkpoint 续训 `4M` 以内的短 run。
`safety_retry` 只增强 `safety.near_distance`、`near_distance` penalty 和 `inter_agent_collision` penalty，不降低 strong terrain 参数。

## 严格标准

诊断实验和被中止的 partial run 不作为 strict success。所有 selected checkpoint 仍按默认 gate 解读：

```text
dmax_reduction_ratio <= 0.2
success_rate >= 0.9
collision_rate <= 0.02
timeout_rate == 0
```

## 诊断结果表

| 对照 | checkpoint | success | timeout | dmax_ok_rate | dispersion_ok_rate | speed_ok_rate | hold>=8 | timeout 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp009 seed31 failed | `outputs/runs/exp_009_terrain3d_strong/weak_warmstart_seed31_retry20m_safe090_strong_lunar_crater_cuda_eval1024/checkpoints/best.pt` | 0.8740 | 0.1250 | 0.0469 | 0.0405 | 0.9998 | 895/1024 | timeout episode 末端仍未满足 dmax/dispersion。 |
| exp009 seed23 passed | `outputs/runs/exp_009_terrain3d_strong/weak_warmstart_seed23_timeout_retry6m_strong_lunar_crater_cpu_nenv1024_eval1024/checkpoints/best.pt` | 1.0000 | 0.0000 | 0.0520 | 0.0560 | 0.9985 | 1024/1024 | 强地形可通过，说明不是全局地形不可解。 |
| exp008 seed31 passed | `outputs/runs/exp_008_terrain3d/weak_warmstart_completion_seed31_4m_evalseed0_cpu/checkpoints/best.pt` | 0.9961 | 0.0000 | 0.0936 | 0.0995 | 0.8922 | 1020/1024 | 弱地形更早进入成功区。 |

## 训练结果表

| seed | run_id | 预算 | checkpoint / 结果 | strict | 关键结果 |
| --- | --- | ---: | --- | --- | --- |
| 23 | `selected_seed23_exp009_checkpoint_eval_cuda` | eval only | `metrics/final_eval_proxy.json` | 通过 | `dmax=0.1474`、`success=1.0000`、`collision=0.0000`、`timeout=0.0000`。 |
| 31 | `hold_reward_seed31_4m_from_exp009_safe090_cuda` | 4M | `metrics/final_eval_proxy.json` | 未通过 | `dmax=0.1759` 通过，但 `success=0.8818`、`collision=0.0361`、`timeout=0.0830` 失败。 |
| 31 | `hold_reward_seed31_6m_cont1_from_4m_cuda` | 6M continuation | `metrics/final_eval_proxy.json` | 未通过 | `success=0.9014` 达标，但 `collision=0.0273`、`timeout=0.0742` 失败。 |
| 31 | `hold_reward_seed31_6m_cont2_safety_eval4_from_cont1_cuda` | 计划 6M，实际中止于 3.15M | partial `train_metrics.jsonl` | 不判定 | 为避免无界堆 PPO，已停止；没有完整 `summary.json` 或独立 `final_eval_proxy.json`，不能作为通过证据。 |

## 失败分析

exp009 seed31 的 `speed_ok_rate` 接近 1，timeout episode 的 `final_mean_speed_mean` 约 `0.0595`，因此当前失败不应优先归因于 speed hold 条件。

timeout episode 的均值：

```text
final_dmax_mean: 3.0207
final_dispersion_mean: 3.2122
final_mean_speed_mean: 0.0595
mean_terrain_speed_scale: 0.3922
```

这表明 seed31 失败重点是强地形下有一批 episode 没能及时把队形收敛到 `dmax/dispersion` 成功区，而不是已经到达成功区后无法减速保持。

短程 hold reward 续训的独立评估进一步确认：

```text
4M:
  dmax_reduction_ratio: 0.1759
  success_rate: 0.8818
  collision_rate: 0.0361
  timeout_rate: 0.0830
  timeout_final_dmax_mean: 3.4935
  timeout_final_dispersion_mean: 4.0321

6M continuation:
  dmax_reduction_ratio: 0.1720
  success_rate: 0.9014
  collision_rate: 0.0273
  timeout_rate: 0.0742
  timeout_final_dmax_mean: 3.6919
  timeout_final_dispersion_mean: 4.3987
```

也就是说，hold reward 能把整体 `success_rate` 推到 0.90 以上，但没有清除 timeout，且 collision 仍高于 `0.02`。timeout episode 末端的 `final_dmax/final_dispersion` 反而没有改善到接近成功区，因此继续同类 PPO 续训不是优先方向。

6M safety retry 变体已按用户要求停止。停止时只完成到 update 12 / `3,145,728` env steps，未产生完整训练 summary 或独立 final eval；该 partial run 只说明 wall time 受同机 GPU 竞争严重影响，不能说明 strict pass。

## 产物路径

```text
outputs/runs/exp_010_strong_success_diagnostics/diagnostic_exp009_seed31_failed_cpu/metrics/final_eval_proxy.json
outputs/runs/exp_010_strong_success_diagnostics/diagnostic_exp009_seed23_passed_cpu/metrics/final_eval_proxy.json
outputs/runs/exp_010_strong_success_diagnostics/diagnostic_exp008_seed31_passed_cpu/metrics/final_eval_proxy.json
outputs/runs/exp_010_strong_success_diagnostics/selected_seed23_exp009_checkpoint_eval_cuda/metrics/final_eval_proxy.json
outputs/runs/exp_010_strong_success_diagnostics/hold_reward_seed31_4m_from_exp009_safe090_cuda/metrics/final_eval_proxy.json
outputs/runs/exp_010_strong_success_diagnostics/hold_reward_seed31_6m_cont1_from_4m_cuda/metrics/final_eval_proxy.json
```

## 结论

exp010 不通过 3-seed strong terrain strict gate。seed23 可以作为强地形已通过对照，但 seed31 在 hold reward 4M + 6M continuation 后仍未通过 `collision_rate` 和 `timeout_rate`。seed47 未启动，因为 seed31 已经证明当前 reward/control 方向不足以形成 3-seed strict suite。

因此不要继续无界增加 PPO 步数。exp008 仍是当前 3-seed terrain-aware 主结果；exp010 的价值是定位失败模式：强地形下 seed31 仍有一批 episode 无法及时把 `dmax/dispersion` 收敛进成功区。

## 下一步

1. 停止继续堆 exp010 PPO continuation。
2. 近期暂缓 strong terrain 失败诊断和动作/curriculum 原型。
3. 当前项目重心转为环境搭建与工程闭环验收：`.venv_isaaclab`、Isaac Sim、Isaac Lab、SKRL、本地任务包 editable install、proxy validation、SKRL MAPPO smoke 和 PhysX sanity。
4. 后续恢复训练研究时，再基于本实验结论重新设计动作表示、控制接口或 terrain curriculum；不要降低 strong terrain 参数来通过 strict gate。
