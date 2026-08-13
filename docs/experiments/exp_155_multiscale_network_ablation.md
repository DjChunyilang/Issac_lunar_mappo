# exp155 多尺度感知与三结构完整Pure RL消融

> 生命周期：`stopped_design_revision`。N0产物仅保留为旧接口失败证据；N1/N2未启动。本实验不得进入架构排名或strict汇总，当前主线已切换到[exp156](exp_156_differential_multiscale_ablation.md)。

## 目的

验证旧主线未收敛是否主要来自两项接口不足：局部地形空间表达过粗，以及二维连续端点动作不能表达等待和到达时间。实验不增加BC、安全投影、方向性mask或集中式执行修正。

## 配置

主配置：

- `configs/experiment/exp155_multiscale_network_ablation.yaml`
- `configs/experiment/exp155_multiscale_n0_mlp.yaml`
- `configs/experiment/exp155_multiscale_n1_cnn.yaml`
- `configs/experiment/exp155_multiscale_n2_path_conditioned.yaml`
- `configs/experiment/exp155_full_rl_ablation.yaml`

核心接口：

```text
observation: ego_v9_multiscale_intent, 291维
action: 40维Categorical时空原语
communication: 12m内16维，12m外稀疏缓存
episode: 96s / 480步
training: shared-joint MAPPO, Pure RL
```

离线消融固定12万样本、三个初始化种子和30 epoch，但结果仅用于诊断，不再淘汰结构。

N0、N1、N2分别使用seed23、256环境和rollout 64完成Stage A/B/C各800 iterations，即每个结构39,321,600次交互，三者合计117,964,800次交互。不提前停止、不提前晋级，也不允许延长。

三者统一使用 `path_terrain_mean_cost=0.26` 和 `path_terrain_max_cost=0.16`，保持quintic动作路径风险奖励有效。

## 严格标准

离线指标只报告，不作为RL准入门限。完整RL消融的中间Stage门限同样只用于记录学习进度。

正式六分层门限：

```text
dmax ratio <= 0.20
success >= 0.90
collision <= 0.02
timeout < 0.10
```

每层64个episode，最多允许6个timeout。

## 结果表

| 阶段 | Seed/规模 | 产物 | 状态 |
| --- | --- | --- | --- |
| 工程接口 | CPU单元测试与小环境 | 本地测试日志 | 通过 |
| N0/N1/N2构建 | CPU有限前向 | 参数量与logits检查 | 通过 |
| SKRL smoke | seed23，8环境，64时步 | `outputs/runs/exp155_multiscale_n2_path_conditioned/smoke_exp155_n2_cpu/` | 通过工程检查，不是训练结果 |
| 正式离线三选二 | 120,000样本，3 seeds，30 epochs | `_suite/metrics/offline_network_ablation.json` | 未通过，0个候选入选 |
| 固定日程smoke | 3×1 iterations，CPU | 本地summary | 通过调度检查，不是训练结果 |
| 评测去重smoke | 3×2 iterations，CPU | 本地summary | 通过；每个边界仅评测一次 |
| 首次N0启动 | 第100 iteration发现重复冻结评测 | `n0_seed23_full_2400iter_invalid_repeated_eval_20260812/` | 已中止，明确无效 |
| seed23完整RL消融 | 原计划N0/N1/N2各39.3M交互 | `exp155_full_rl_ablation/` | 已停止；仅N0完成，其结果排除在架构排名之外 |
| seed31/47 | 前序种子通过后启动 | 尚无 | 未运行 |

## 失败分析

正式离线消融得到以下三个初始化种子的测试均值：

| 结构 | Spearman | P95归一化误差 | 相对后悔值 | 远场top-1 | 相对吞吐 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| N0 | 0.723 | 0.309 | 0.0030 | 0.309 | 0.905 | 未通过 |
| N1 | 0.742 | 0.291 | 0.0036 | 0.309 | 0.324 | 未通过 |
| N2 | 0.583 | 0.399 | 0.0028 | 0.307 | 0.173 | 未通过 |

三个结构都通过参数量、有限反向和显存门限，并通过相对后悔值门限；但均未通过风险相关性、P95误差和远场方向识别。N1和N2还未通过相对吞吐门限。机器可读字段为 `selected_rl_finalists=[]` 和 `passed=false`，但该字段不再控制后续训练。

这组结果不支持“增加epoch即可解决”的结论。三个结构的远场top-1都集中在约0.31，提示优先检查标签类别分布和任务可辨识性；N2明显低于N0/N1，需核对路径采样坐标与稠密标签是否使用完全一致的几何；N1/N2吞吐需确认是否与具有等价输出任务的参考模型比较。上述检查属于离线任务诊断，不授权扫描网络规模、训练轮数或降低门限。

后续有界诊断给出：

- 测试集远场五类频率为 `0.211/0.196/0.199/0.190/0.205`，多数类准确率仅 `0.211`，不存在足以解释0.31结果的类别不平衡；
- 使用真实13条局部路径风险得到的最优方向，与4 m远场最优方向的一致率仅 `0.309`。当前训练损失没有远场标签，因此 `far_top1>=0.70` 与训练目标不一致；
- 三个速度档的地形风险标签最大差为 `0`，每个样本至少有三个等价最优动作；但交叉熵的单一 `argmin` 只标记其中一个速度档，造成监督冲突；
- 直接对多尺度观测的风险通道沿quintic路径插值，Spearman为 `0.985`、P95误差为 `0.037`，证明局部路径风险可由当前观测重构；
- N2硬编码路径与轨迹生成器的最大坐标差约为 `1.2e-7 m`，路径几何并非本次失败原因。

诊断产物为 `_suite/metrics/offline_task_diagnostic.json`。这些证据说明离线结果不能公平筛选RL结构，因此三个候选直接进入相同预算的完整Pure RL比较；离线权重不会迁移。

## 产物路径

```text
outputs/runs/exp155_multiscale_network_ablation/_suite/metrics/
outputs/runs/exp155_full_rl_ablation/n0_seed23_full_2400iter/
outputs/runs/exp155_full_rl_ablation/n1_seed23_full_2400iter/
outputs/runs/exp155_full_rl_ablation/n2_seed23_full_2400iter/
outputs/runs/exp155_full_rl_ablation/_suite/metrics/
```

正式种子需检查：

```text
metrics/summary.json
metrics/bounded_curriculum_eval.jsonl
metrics/final_eval_proxy.json
metrics/stratified_strict_acceptance.json
metrics/terrain_contrast.json
metrics/strict_acceptance.json
```

## 结论

新接口、固定日程和三结构完整RL编排已经实现。离线失败仅作为监督冲突诊断；exp155尚未完成三结构RL比较，不是已收敛主结果，也没有推荐checkpoint。

## 下一步

按N0、N1、N2顺序运行seed23完整Pure RL消融。三个结构全部训练并完成相同冻结评测后，按strict分层数和预声明综合分数选择相对最优方法；不得根据离线结果或中间曲线提前淘汰。
