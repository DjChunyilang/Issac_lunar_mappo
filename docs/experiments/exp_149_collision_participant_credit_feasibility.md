# exp149：碰撞参与车辆信用可行性审计

## 目的

exp148 修正轨迹执行时间后，B0 能够持续缩小 dmax，但冻结评测的 collision 达到 `0.9990`。双种子失败 episode 中，所有碰撞 episode 都出现重复车辆对冲突。当前环境在碰撞时向四辆车复制相同团队奖励，其中包括 collision penalty 和 failure penalty；该语义无法区分直接碰撞车辆与未参与车辆。

本实验不修改训练，只验证一个单一假设：真实碰撞参与车辆在终止前能否由连续轨迹冲突稳定识别，从而支持后续只对既有碰撞终止回报进行逐车信用分配。

## 冻结配置

- 配置：`configs/experiment/exp148_decentralized_b0_trajectory_time_consistent.yaml`；
- checkpoint：`ppo_timestep_001024.pt` 与 `ppo_timestep_002048.pt`；
- 数据种子：`32023`、`33023`；
- 每个组合：128 个环境、512 个规划步；
- 策略动作：deterministic Actor mean，不加入采样噪声；
- 轨迹冲突：沿 exp148 的公共物理时间网格计算；
- Actor、Critic、奖励、环境状态推进和 checkpoint 均不得改变。

两个 checkpoint 用于区分碰撞归因结构是否只在训练末尾偶然出现。正式判定要求两个 checkpoint、两个数据种子的四个组合方向一致。

## 指标定义

对碰撞终止步 $t_c$，以实际位置定义碰撞车辆对集合：

\[
\mathcal C_{t_c}
=
\left\{
(i,j)\mid i<j,
\left\|\mathbf p_i(t_c)-\mathbf p_j(t_c)\right\|_2
<d_{\mathrm{collision}}
\right\}.
\]

碰撞参与车辆集合为：

\[
\mathcal A_{t_c}
=
\left\{
i\mid \exists j:(i,j)\in\mathcal C_{t_c}
\right\}.
\]

团队碰撞惩罚被复制给未参与车辆的比例定义为：

\[
r_{\mathrm{nonparticipant}}
=
1-\frac{|\mathcal A_{t_c}|}{N}.
\]

令 $R_{ij}(t)$ 表示车辆对 $(i,j)$ 在规划步 $t$ 被连续冲突诊断标记为 repeated。碰撞对在前 $h$ 步的召回率为：

\[
\operatorname{Recall}_h
=
\frac{1}{|\mathcal C_{t_c}|}
\sum_{(i,j)\in\mathcal C_{t_c}}
\mathbb I\left[
\max_{t_c-h\le t<t_c}R_{ij}(t)=1
\right].
\]

同时统计：

- 碰撞车辆数与碰撞车辆对数；
- repeated 冲突首次命中碰撞对的提前步数；
- 前 1、2、4、8、16 步的碰撞对召回率；
- 前 8 步 repeated 车辆对中最终碰撞对的 precision；
- 碰撞前 16 步的 dmax progress、最近邻闭合量、动作半径、转向量；
- 碰撞前与终止步的 gather、safety、terrain、terminal 和 total 加权奖励贡献；
- 碰撞参与车辆与未参与车辆的动作及最近邻闭合量差异。

奖励统计只用于解释现有训练信用，不构造新奖励。

## 晋级门限

只有四个 checkpoint—种子组合全部满足以下条件，才允许形成下一次训练计划：

1. 每个组合至少包含 100 个完整碰撞 episode；
2. 平均未参与车辆比例不低于 25%；
3. 碰撞车辆数中位数不超过 2；
4. 碰撞对在终止前 8 步的 repeated recall 不低于 80%；
5. 碰撞对在终止前 16 步的 repeated recall 不低于 90%；
6. 碰撞对首次 repeated 命中的提前量中位数不少于 4 步；
7. checkpoint Actor 摘要在采集前后完全一致。

门限同时约束“团队惩罚确实存在车辆污染”和“终止车辆身份能稳定追溯到此前动作”。precision 仅作描述性指标，不作为门限，因为下一阶段若获授权，也只能使用真实碰撞终止参与者，不能将预测冲突写入奖励。

## 可能的下一步边界

若全部门限通过，只允许预注册一次逐车碰撞终止信用对照。设碰撞参与车辆数为 $m$，原团队碰撞相关终止项为 $r_t^{\mathrm{collision}}$，候选逐车分配为：

\[
r_{t,i}^{\mathrm{collision}}
=
r_t^{\mathrm{collision}}
\frac{N}{m}\mathbb I[i\in\mathcal A_t].
\]

上式只表示分配系数；正式计划必须根据原奖励符号展开，并验证逐车均值严格等于原团队项：

\[
\frac{1}{N}\sum_i r_{t,i}^{\mathrm{collision}}
=r_t^{\mathrm{collision}}.
\]

集中式 Critic、团队 return、环境 reward、碰撞终止和执行链路保持不变；差异只允许进入 Actor advantage。不得使用 predicted/repeated conflict 作为训练奖励、辅助损失、Actor 输入或在线控制信号。

若任一门限失败，则停止该方向，不调整 lookback、冲突阈值或数据种子，也不启动训练。

## 产物路径

正式产物写入：

```text
outputs/runs/exp149_collision_participant_credit_feasibility/
  frozen_exp148_dual_checkpoint_dualseed/
    config/experiment.yaml
    metrics/collision_participant_credit_feasibility.json
    run_manifest.json
  _suite/
    metrics/suite_summary.json
    run_manifest.json
```

## 当前状态

正式冻结诊断已完成，四个 checkpoint—种子组合全部通过预注册门限。状态为 `participant_credit_feasible_plan_next_screen`，只授权形成一次 exp150 组件筛选计划。

## 结果

| checkpoint | seed | collision episodes | 未参与车辆比例 | 参与车辆中位数 | repeated recall 8/16步 | 首次命中提前量中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t1024 | 32023 | 420 | `0.4970` | 2 | `1.0000/1.0000` | 14 |
| t1024 | 33023 | 423 | `0.4994` | 2 | `0.9976/0.9976` | 15 |
| t2048 | 32023 | 331 | `0.4977` | 2 | `1.0000/1.0000` | 19 |
| t2048 | 33023 | 348 | `0.5000` | 2 | `0.9971/0.9971` | 21 |

四个组合的碰撞车辆数和碰撞车辆对数中位数分别为 `2/1`。这意味着典型碰撞只由一对车辆构成，但另外两辆车仍接收完全相同的团队碰撞终止信用。

碰撞前最后4步，参与车辆的最近邻闭合量均值为 `0.0450–0.0619 m/步`，未参与车辆为 `0.0006–0.0147 m/步`。终止步的平均 safety 与 terminal 加权贡献约为 `−100.63/−55.00`，总回报约为 `−155.47` 至 `−155.69`。因此问题不是碰撞惩罚数值过小，而是强终止信号被复制给未参与车辆，无法在共享团队 advantage 中区分直接责任。

前8步 repeated precision 为 `0.687–0.762`，低于 recall，但只作描述性指标。exp150 不得读取 predicted/repeated conflict；训练信用只能使用真实碰撞终止后的实际参与车辆。

正式结果位于：

```text
outputs/runs/exp149_collision_participant_credit_feasibility/
  _suite/metrics/suite_summary.json
```

## 结论

exp149 证明逐车碰撞终止信用具有稳定的事后可归因性，并且该结构在 t1024 与 t2048、两个独立地形种子上方向一致。因此只授权一次 exp150：保持团队 reward 与集中式 Critic 不变，对 Actor 增加零和的碰撞参与者终止信用残差。该授权不包括权重扫描、预测冲突奖励、图网络、安全投影或在线车辆优先级。
