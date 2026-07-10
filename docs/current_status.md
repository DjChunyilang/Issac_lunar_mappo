# 当前状态

## 当前主线

- 当前主设计口径已切换为“高吞吐 proxy 训练 + Isaac Sim / Isaac Lab / PhysX 高保真闭环评估”。当前实施路线以 `docs/implementation_plan.md` 和 `docs/architecture/overall_plan_v3.md` 为准。
- 训练主环境仍是 PyTorch / torch-vectorized proxy 环境，用于 MAPPO / PPO 采样、奖励调试、观测接口验证和大规模对照实验。
- Isaac Sim / Isaac Lab / PhysX 不作为当前主训练 loop，而作为 high-fidelity validation、迁移 sanity check、失效分析和可视化展示平台。
- 当前 PhysX 层使用 Clearpath Jackal 作为活跃轮式资产，已替换旧占位资产。Jackal tracking 可验证轮式控制、强三维地形 mesh、姿态稳定性和输出链路，但不能证明真实月球车越障、轮壤接触或低重力动力学已经完成。
- 视觉观测不进入 policy input；地形以车体系 `5×5×2` 局部结构化网格进入策略。
- 当前从“暂停长训完善环境”切回新环境栈长训迭代：结构化 Actor/Critic、bicycle proxy 动力学、quintic 轨迹、`25 m × 25 m` 地图和 `communication_radius=0.0` 无限通信语义已通过 `exp042` smoke。`exp043` 直接 40M env-step 长跑已完成但没有收敛；`exp044` 改为 initial-state curriculum 后能明显缩短 dmax，但 success 仍为 0；`exp045` local-success bootstrap 首次恢复局部 success 信号但未 strict；`exp046`/`exp047` 逐步恢复 terminal convergence；`exp048` 已通过 dmax/success/collision，仅 timeout `0.0137` 未过；`exp049`/`exp050` 分别说明全局增强 spacing/filter 和增强 hold/timeout shaping 都会退化；`exp051` 回到 exp048 reward/filter/control、只隔离 PPO 稳定性后达到 dmax `0.1836`、success `0.9883`、collision `0.0020`、timeout `0.0098`，是当前新环境栈 local reset 最好候选但仍未 strict；`exp052` 早退火到 `8192` 已完成并明显退化，说明 entropy taper 不宜过早；`exp053` 轻微提高全局 near reward 后 success/timeout 大幅退化，说明不能继续提高全局安全间距惩罚；`exp054` 收窄 PPO clip 到 `0.16` 后 dmax/collision 达标，但 success `0.7168`、timeout `0.2803` 明显退化；`exp055` 放宽 PPO clip 到 `0.20` 后 dmax/success/collision 达标，但 timeout `0.0146` 差于 exp051。clip 扫描表明 `0.18` 仍是当前最好点；`exp056`/`exp057` 的 terminal pairwise reward 均未改善 timeout，当前最好仍是 exp051。
- exp051 附近 checkpoint multi-seed 复验已完成：`012288/013312/014336` 在 `1023/2023/3023/4023` 四个 eval seed 上均未 strict，其中 `013312` 的 timeout 均值最低（`0.0134`）但 `timeout_zero_count=0/4`，说明当前 best 选点相对稳定，剩余瓶颈不是简单 checkpoint reselection。
- exp058 已完成：回到 exp051，只把 PPO `gamma` 从 `0.99` 提高到 `0.995`，不改 action/reward/filter/control。final eval dmax `0.1991`、collision `0.0020` 达标，但 success `0.7451`、timeout `0.2529` 明显失败；说明更长折扣 horizon 拖慢 terminal convergence，不能作为下一步主线。
- exp059 已完成：回到 exp051，只把 `gae_lambda` 从 `0.95` 降到 `0.90`，不改 action/reward/filter/control。best 仍回落到 `ppo_timestep_012288.pt`，final eval dmax `0.1927`、collision `0.0127` 达标，但 success `0.6904`、timeout `0.2988` 明显失败；说明更短 GAE trace 也不能修复 exp051 尾部 timeout，价值估计 horizon 方向暂时不作为主线。
- exp060 已完成：回到 exp051，只把 `value_loss_coef` 从 `0.50` 提高到 `0.75`，不改 action/reward/filter/control。best 为 `ppo_timestep_012288.pt`，final eval dmax `0.1837`、success `0.9736`、collision `0.0` 达标，但 timeout `0.0264` 失败且差于 exp051；说明更强 critic loss 权重没有清掉尾部 timeout，不能作为下一步主线。
- exp051 / exp060 success-gate 诊断已完成：exp051 recheck seed1023 的 `15` 个 timeout 中，`min_pairwise` final gate 失败 `15/15`，dmax gate 失败 `2/15`，dispersion/speed gate 均为 `0/15`；exp060 的 `19` 个 timeout 中，`min_pairwise` 失败 `18/19`。这说明尾部主要是“已经接近集合且低速，但最近邻间距没有通过 success gate”，暂时不应把 Actor 输出改成多点采样让 filter 选择。
- exp061/exp062 已完成观测/critic 可观测性诊断，均不改 action 输出、reward、filter 或 control。exp061 给 Actor/Critic 同时加入 terminal gate 特征后明显退化：final eval dmax `0.1890` 达标，但 success `0.8506`、collision `0.0205`、timeout `0.1289` 均失败；说明直接把 gate margin 暴露给 Actor 会诱发激进/饱和动作。exp062 保持 exp051 Actor 观测和 `branched_v1`，只给 critic state 加 `min_pairwise` 并用 `structured_v2`，final eval dmax `0.1832`、success `0.9736`、collision `0.0059` 达标，但 timeout `0.0205` 失败；4-seed checkpoint sweep 的 best `016384` timeout 均值 `0.0161`，不优于 exp051 的 `0.0134`。当前最好仍是 exp051。
- 生成结果写入 `outputs/runs/`，并由 git 忽略。

## 当前接口状态

- actor observation schema 为 `ego_v3_local_terrain_grid`，输入维度为 86，包含 ego、neighbor、50 维局部地形网格和 aggregation 特征，不包含 oracle 集合点。
- 地形网格通道为相对高度和风险，覆盖前后 `[-0.4, 1.2] m`、横向 `[-0.8, 0.8] m`；critic 仍为 54 维，并使用 5 维网格摘要。地图面积可通过 `world_xy_limit/crater_field_size` 扩大，但本轮不扩大 Actor 局部地形观测窗口。
- checkpoint 加载要求 schema、actor 输入维度和 critic 状态维度完全匹配；新 checkpoint metadata 还记录 Actor/Critic 架构、运动学模型和轨迹生成方法。旧 `ego_v2_speed_angular` checkpoint 不自动迁移；当前 schema 但缺少架构 metadata 的旧 checkpoint 只按 `mlp_v1` 兼容路径加载。
- centralized critic state 和 reward shaping 可以使用 oracle 信息；执行期 actor 不接收 `p*`、oracle 距离或 oracle 距离下降量。
- 动作接口固定为低维 `[rho, beta]`，再经局部子目标、可配置 `line/quintic` 轨迹和简化速度控制器转换为运动命令。
- 当前 proxy 动力学可配置为 `unicycle` 或 `bicycle`；旧配置默认 `unicycle`，`exp042` 显式使用 `bicycle`。二者都没有质量、惯量、轮地接触、打滑、悬挂或 PhysX contact。
- `scripts/train_skrl_mappo.py` 支持 `actor_architecture=mlp_v1|branched_v1|branched_v2` 和 `critic_architecture=mlp_v1|structured_v1|structured_v2`。默认 exp051 主线仍是 `ego_v3_local_terrain_grid`、Actor/Critic `86/54`；`ego_v4_terminal_gate` 与 critic `55` 维状态仅用于 exp061/exp062 诊断，不作为当前最好主线。
- `scripts/train_skrl_mappo.py` 使用 SKRL MAPPO 训练 proxy wrapper；`isaaclab-multi-agent` wrapper 只是接口层，不代表训练 loop 运行在 Isaac Sim / PhysX。
- exp016 已启用项目侧 `shared_joint` 更新：共享 Actor/Critic 只使用一个 optimizer，每个 rollout 合并四个 rover 的 Actor 样本并只更新一次 Critic。
- 当前 exp016 诊断配置把通信半径临时扩大到 `12 m`；这是训练诊断设置，不是最终通信约束。
- exp017 已完成 pure RL 连续 20M 长跑并通过 seed23 独立 strict eval；这是固定地图、单 seed proxy 结果，不代表随机地图泛化或多 seed 收敛。
- exp018 已加入每环境、每 episode reset 独立地形随机化，并把地形强度提高一档；完整测试、CPU/CUDA smoke 和随机地图渲染已通过。seed23 连续 20M 已完成，dmax 和 success 达标，但 collision / timeout 未通过 strict gate。
- exp019 已在 exp018 基础上完成两个诊断改造：success gate 新增最近邻安全间距 `0.42 m`，terrain reward 扩展到当前点到子目标的路径级风险。seed23 20M 工程链路和 5 轮独立 eval/GIF 已完成，但 strict gate 未通过。
- exp020 已在 exp019 基础上加入 terrain/safety-aware 子目标过滤器；过滤器稳定降低路径风险，但显著抑制集合进度。seed23 20M、5 轮独立 eval/GIF 和训练曲线已完成，strict gate 未通过。
- exp021 已完成 exp020 的课程化/软化 filter 迭代：前期保留 raw action，后期逐步增加 filter 介入概率和 score 权重，并加入 raw-risk / filter-deviation 辅助惩罚。seed23 20M、5 轮独立 eval、GIF、height map 和训练曲线已完成，strict gate 未通过。
- exp022 已完成 endpoint/path safety constrained curriculum filter 迭代：collision 被压到 strict 内，但集合进度塌缩，5 seed mean success `0.0139`、timeout `0.9699`，strict 未通过。
- exp023 已完成 soft progress-preserving filter 迭代：success 从 exp022 的 `0.0139` 回升到 `0.3027`，但 collision `0.2295`、timeout `0.4717`，strict 未通过；失败原因是 static endpoint/path safety 未预测可见邻居同步运动。
- exp024 已完成 mutual path safety filter 迭代：在 exp023 基础上把可见邻居 raw subgoal path 作为动态障碍，按相同时间采样比较候选路径；post-hoc 使用 `success_progress_long` 重选 `ppo_timestep_010240.pt` 为 best，seed1023 final eval 为 dmax ratio `0.1397`、success `0.8398`、collision `0.0674`、timeout `0.0947`，strict 未通过但显著优于 exp023。
- exp025 已完成 dense mutual path safety filter 迭代：基于 exp024 加密 mutual/path safety 采样到 9 点，并适度提高 path/mutual collision 权重；best `ppo_timestep_009216.pt` final eval 为 dmax ratio `0.1434`、success `0.8525`、collision `0.0449`、timeout `0.1035`，strict 未通过。相对 exp024 collision 降低，但 timeout 未改善。
- exp026 已完成 hold-zone filter 诊断：过早/过宽的 `hold_zone_rho/spacing` cost 把 success 从 exp025 的 `0.8525` 拉低到 `0.7529`，collision `0.0615`、timeout `0.1865`，strict 未通过。
- exp027 已完成 strict hold-zone filter 诊断：把 hold-zone activation 收窄到真正 success dmax/dispersion 附近后避免了 exp026 的明显退化，但 final eval success `0.8418`、collision `0.0498`、timeout `0.1123`，未优于 exp025。
- exp028 已完成 hold reward 诊断：回退到 exp025 dense mutual filter，只强化 `success_hold_step=4.0`、`success_bonus=45`、`timeout_penalty=18`；final eval success `0.8691`、collision `0.0469`、timeout `0.0889`，是 exp026–029 中最好但仍未 strict。
- exp029 已完成 hold reward + stronger safety 诊断：在 exp028 基础上加强 path/mutual collision filter 和终端碰撞惩罚，final eval success `0.8262`、collision `0.0557`、timeout `0.1221`，说明继续加安全权重会牺牲成功并未压低真实碰撞。
- exp030 已完成低层 control safety projection 诊断：回到 exp028 主体，只在 `compute_control()` 后、`_integrate()` 前加入相对速度安全投影和 success-zone damping；final eval success `0.8330`、collision `0.0313`、timeout `0.1357`，collision 明显低于 exp028，但投影过强导致 success/timeout 退化。
- exp031–exp034 已完成 control safety 投影条件迭代：简单调弱、closing-only、directional scale 和 directional mask 都未 strict；其中 exp034 的 mask 版本把 success 拉回 `0.8828`、timeout 降到 `0.0840`，但 collision `0.0361` 仍失败。
- exp035–exp036 已完成 directional mask buffer 与 stronger hold/timeout shaping：exp035 首次让 success `0.9072` 和 collision `0.0127` 同时达标；exp036 进一步到 success `0.9336`、collision `0.0088`、timeout `0.0586`，剩余瓶颈转为 timeout/hold。
- exp037 已完成 260-step episode/eval 诊断：timeout 从 exp036 的 `0.0586` 降到 `0.0410`，但 collision 反弹到 `0.0352`，说明单纯延长 episode 会暴露末段碰撞。
- exp038 已完成 success-zone stabilizer + 320-step episode/eval：修正 best 后 final eval success `0.9756`、collision `0.0137`、timeout `0.0107`；旧环境栈随机地形阶段性最佳候选，strict 只剩 timeout gate 失败。
- exp039/exp040 是基于 exp038 best 的诊断复评，不建议长训：hard near stabilizer 和 stronger soft hold stabilizer 都使 timeout 或 collision 差于 exp038。
- exp041 已完成 hold-zone override 诊断与 CPU/CUDA smoke：在 exp038 best 上复评得到 success `0.9795`、collision `0.0107`、timeout `0.0098`，略优于 exp038，但当前已暂停长训，暂不启动 exp041。
- exp042 已完成环境工程探针：`branched_v1` Actor、`structured_v1` Critic、`bicycle` proxy、`quintic` 轨迹生成、`25 m × 25 m` 地图和 `communication_radius=0.0` 无限可见邻居语义在 CPU `8 env / 8 timesteps` 与 CUDA `256 env / 64 timesteps` smoke 中通过；CUDA smoke 显示一个 optimizer、两次 joint update、terrain branch 权重更新 `0.1263`、动作非退化。
- exp043 已完成直接长训：基于 exp042 新环境栈，迁移 exp041 的 hold-zone override，加入可配置 initial-state 分布并扩大初始队形采样，terrain crater density 提高到 `crater_count=48`，seed23 连续 `20480` timesteps / `41,943,040` env steps。训练链路正常、参数和 terrain branch 均更新，但 final eval 为 dmax ratio `0.8596`、success `0.0`、collision `0.0`、timeout `1.0`，strict 未通过。
- exp044 已完成：保留 `branched_v1/structured_v1`、`bicycle`、`quintic`、`25 m × 25 m` 地图和无限通信语义，并加入 `3.0–4.0 m -> 3.8–5.2 m` initial-state curriculum。final eval dmax ratio `0.4796`、success `0.0`、collision `0.00195`、timeout `0.9980`，strict 未通过；相比 exp043 明显靠拢但仍未进入 success basin。
- exp045 已完成：保持新环境栈和 25m 地图，但把目标 reset 分布缩小到 `2.4–3.4 m`，课程起点为 `1.6–2.4 m`，同时放大 `rho/beta` 可达范围、增强 gather progress、临时降低 terrain/filter 干扰。final eval dmax ratio `0.2734`、success `0.1846`、collision `0.0`、timeout `0.8174`，说明 local-success bootstrap 有效但仍未收敛。
- exp046 已完成：沿用 exp045 的 local reset 分布，但降低 filter/control-safety 的末端介入强度，增强 dmax/dispersion progress、success bonus 和 timeout penalty。final eval dmax ratio `0.2424`、success `0.6123`、collision `0.0`、timeout `0.3877`，strict 未通过，但证明新环境栈已进入 local success basin。
- exp047 已完成：保持 exp046 reset 分布，进一步释放 terminal safety/filter/control damping，同时增强 dmax/dispersion/timeout/success shaping。final eval dmax ratio `0.2132`、success `0.7188`、collision `0.0059`、timeout `0.2764`，strict 未通过，但曾是新环境栈 local reset 阶段性最好结果。
- exp048 已完成：在 exp047 附近小步提高 terminal drive、dispersion 收缩和 timeout shaping。final eval dmax ratio `0.1866`、success `0.9844`、collision `0.0020`，均通过 strict；唯一失败是 timeout `0.0137`。
- exp049 已完成：针对 exp048 剩余最近邻安全间距灰区增强 terminal spacing。final eval dmax ratio `0.1884`、success `0.8926`、collision `0.0010`、timeout `0.1064`，strict 未通过且明显差于 exp048；说明 spacing/filter/control safety 介入过强，修复了部分间距但牺牲了成功保持。
- exp050 已完成：回到 exp048 主体，不改 action 输出、不做多点采样、不增强低层控制规划能力；主要调整 terminal hold reward、timeout shaping、PPO 学习率/clip/探索噪声。final eval dmax ratio `0.1847`、success `0.9590`、collision `0.0059` 达标，但 timeout `0.0352` 未过且差于 exp048，说明该 RL 配置微调方向不能作为下一步主线。
- exp051 已完成：reward、filter、control safety 全部回到 exp048，只隔离 PPO 稳定性调整（学习率、clip、entropy schedule、initial log std）。best 为 `ppo_timestep_013312.pt`，final eval dmax ratio `0.1836`、success `0.9883`、collision `0.0020` 均通过，但 timeout `0.0098` 仍失败；相对 exp048 小幅降低 timeout，相对 exp050 明显恢复 success/timeout。
- exp051 checkpoint seed sweep 已完成：对 `012288/013312/014336` 做 `4` 个 eval seed 复验后，013312 仍是附近 timeout 均值最低的 checkpoint，但 `strict_pass_count=0/4`、`timeout_zero_count=0/4`，说明剩余问题不是简单 checkpoint reselection。
- exp052 已完成：以 exp051 为基线，只把 entropy schedule 从 `12288` 提前到 `8192`，不改 action 输出、不新增多点采样、不改 reward/filter/control。best 为 `ppo_timestep_008192.pt`，final eval dmax ratio `0.1863`、collision `0.0059` 达标，但 success `0.8955` 和 timeout `0.0986` 失败，明显差于 exp051。
- exp053 已完成：回到 exp051，只把 reward 中已有的 `near_distance` 安全惩罚系数从 `2.4` 小幅提高到 `2.8`，不改 action 输出、不新增多点采样、不改 filter/control/PPO schedule。best 为 `ppo_timestep_020480.pt`，final eval dmax ratio `0.2049`、success `0.6416`、collision `0.0039`、timeout `0.3545`，明显差于 exp051。
- exp054 已完成：回到 exp051，只把 PPO `clip_epsilon` 从 `0.18` 收窄到 `0.16`，不改 action 输出、不新增多点采样、不改 reward/filter/control。best 为 `ppo_timestep_017408.pt`，final eval dmax ratio `0.1972`、collision `0.0029` 达标，但 success `0.7168` 和 timeout `0.2803` 明显失败；说明更保守 clip 过度抑制 policy update，不能作为下一步主线。
- exp055 已完成：回到 exp051，只把 PPO `clip_epsilon` 从 `0.18` 放宽到 `0.20`，不改 action 输出、不新增多点采样、不改 reward/filter/control。best 为 `ppo_timestep_017408.pt`，final eval dmax ratio `0.1850`、success `0.9824`、collision `0.0029` 达标，但 timeout `0.0146` 仍失败且差于 exp051；说明放宽 clip 没有清掉尾部 timeout。
- exp056 已完成：回到 exp051，不改 action 输出、不新增多点采样、不改 filter/control/PPO，只新增 reward 侧 `terminal_pairwise_gap=4.0`，并在 dmax/dispersion 接近成功区时惩罚 `nearest < min_pairwise_distance` 的 gap。best 为 `ppo_timestep_012288.pt`，final eval dmax ratio `0.1864`、success `0.9873`、collision `0.0010` 达标，但 timeout `0.0117` 仍失败且差于 exp051；说明该项方向有轻微信号但触发/强度仍不理想。
- exp057 已完成：回到 exp051，不改 action 输出、不新增多点采样、不改 filter/control/PPO，只把 terminal pairwise reward 收窄为 `terminal_pairwise_gap=2.0` 且 dmax/dispersion multiplier `1.00/1.00`。best 为 `ppo_timestep_011264.pt`，final eval dmax ratio `0.1850`、success `0.9697`、collision `0.0059` 达标，但 timeout `0.0254` 失败且明显差于 exp051/exp056；说明即使严格触发，terminal pairwise reward 仍会扰动末端 hold。
- exp058 已完成：回到 exp051，不改 action 输出、不新增多点采样、不改 reward/filter/control，只把 PPO `gamma` 从 `0.99` 提高到 `0.995`。best 为 `ppo_timestep_012288.pt`，final eval dmax ratio `0.1991`、collision `0.0020` 达标，但 success `0.7451`、timeout `0.2529` 明显失败；说明更长折扣 horizon 没有改善尾部 timeout，反而拖慢 terminal convergence。
- exp059 已完成：回到 exp051，不改 action 输出、不新增多点采样、不改 reward/filter/control，只把 PPO `gae_lambda` 从 `0.95` 降到 `0.90`。best 为 `ppo_timestep_012288.pt`，final eval dmax ratio `0.1927`、collision `0.0127` 达标，但 success `0.6904`、timeout `0.2988` 明显失败；训练后期 success 还坍缩到约 `0.0122`，说明更短 advantage trace 不适合作为下一步主线。
- exp060 已完成：回到 exp051，不改 action 输出、不新增多点采样、不改 reward/filter/control，只把 PPO `value_loss_coef` 从 `0.50` 提高到 `0.75`。best 为 `ppo_timestep_012288.pt`，final eval dmax ratio `0.1837`、success `0.9736`、collision `0.0` 达标，但 timeout `0.0264` 失败且差于 exp051；说明更强 critic loss 权重没有改善尾部 hold。
- exp061 已完成：回到 exp051，不改 action 输出、不新增多点采样、不改 reward/filter/control，只把 Actor observation 切到 `ego_v4_terminal_gate` 并使用 `branched_v2/structured_v2`，显式加入 dmax/dispersion/speed/hold/pairwise gate 特征。best 为 `ppo_timestep_020480.pt`，final eval dmax ratio `0.1890` 达标，但 success `0.8506`、collision `0.0205`、timeout `0.1289` 失败；gate 诊断中 timeout 仍主要卡 `min_pairwise`，且动作饱和显著上升，不能作为下一步主线。
- exp062 已完成：回到 exp051，Actor observation 和 `branched_v1` 保持不变，只给 centralized critic state 加 terminal `min_pairwise` 并使用 `structured_v2`。best 为 `ppo_timestep_016384.pt`，final eval dmax ratio `0.1832`、success `0.9736`、collision `0.0059` 达标，但 timeout `0.0205` 失败；3-checkpoint/4-seed sweep 中 `016384` timeout mean `0.0161`、`0/4` strict，不优于 exp051。

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

新增 checkpoint seed sweep 诊断入口，用于在不改变 policy/filter/control 的情况下比较多个 checkpoint 的 eval seed 稳定性：

```bash
.venv_isaaclab/bin/python scripts/evaluate_proxy_checkpoint_seed_sweep.py \
  --config outputs/runs/<experiment>/<run_id>/config/experiment.yaml \
  --run-dir outputs/runs/<experiment>/<run_id> \
  --checkpoint ppo_timestep_012288.pt \
  --checkpoint ppo_timestep_013312.pt \
  --seeds 1023,2023,3023,4023 \
  --device cuda \
  --num-envs 1024 \
  --steps 320
```

该入口写入 `metrics/checkpoint_seed_sweep/summary.json` 和逐 seed eval JSON；它只用于诊断 checkpoint selection / eval 方差，不替代 strict gate。

新增 success-gate 诊断入口，用于逐 episode 记录 timeout 末端到底卡在哪个 success gate：

```bash
.venv_isaaclab/bin/python scripts/diagnose_proxy_success_gates.py \
  --config outputs/runs/<experiment>/<run_id>/config/experiment.yaml \
  --checkpoint outputs/runs/<experiment>/<run_id>/checkpoints/best.pt \
  --device cuda \
  --num-envs 1024 \
  --steps 320 \
  --seed 1023 \
  --run-dir outputs/runs/<experiment>/<run_id>
```

该入口写入 `metrics/success_gate_diagnostics.json`，只用于 failure analysis；若与历史 `final_eval_proxy.json` 数字略有不同，应表述为 recheck/diagnostic，不替换原 strict eval 记录。

## 已验证结果

| 实验 | 地形 | 方法 | 严格状态 | 当前解释 |
| --- | --- | --- | --- | --- |
| exp006 | 平地 proxy | BC + PPO | 通过 | 平地 proxy strict baseline，checkpoint 来自 PPO 阶段。 |
| exp008 | 弱 lunar crater 3D proxy | 弱 warm-start + PPO | 3 seeds 通过 | 当前最完整的 terrain-aware proxy baseline。 |
| exp009 | 强 lunar crater 3D proxy | 弱 warm-start + PPO | 未通过 | seed23 通过；seed31 失败；近期不继续堆 long-budget PPO。 |
| exp010 | 强 lunar crater 3D proxy | hold reward / safety 诊断 | 未通过 | success 可改善，但 collision/timeout gate 仍失败。 |
| exp012 | proxy SKRL-MAPPO CUDA 诊断 | action scale warmup probe | 未通过 | distance 有改善，但 strict gate 未通过。 |
| exp013 | proxy SKRL-MAPPO CUDA 诊断 | action scale ablation + teacher reachability | 未通过 | 当前小动作 100-step 配置对 teacher 也几乎不可达。 |
| exp014 | 弱 lunar crater proxy | 5×5 局部地形网格 CUDA probe | 工程验证通过；未做 strict | 新观测和训练链路有效，不能表述为策略收敛。 |
| exp015 | 偏弱中档 lunar crater proxy | SKRL MAPPO + BC20 | 2M screen 未通过 | 工程信号正常；dmax ratio 0.818、success 0、collision 0.124、timeout 0.876，因此未启动 8M。 |
| exp016 | 偏弱中档 lunar crater proxy | shared-joint MAPPO + local BC100 + comm12 | BC probe 未通过 | shared update 探针通过；BC-only dmax ratio 0.438、collision 0.0088、timeout 0.991，未启动 2M。 |
| exp017 | 固定偏弱中档 lunar crater proxy | shared-joint MAPPO pure RL + comm12 | seed23 strict 通过 | final dmax ratio 0.1318、success 0.9990、collision 0.00098、timeout 0；仍是 single-seed candidate。 |
| exp018 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + comm12 | 未通过 | seed23 20M 完成；final dmax ratio 0.1417、success 0.9609 通过，但 collision 0.0352、timeout 0.0088 未达 strict gate。 |
| exp019 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + safe success gate + path terrain risk | 未通过 | seed23 20M 完成；10240 checkpoint 有集合趋势但 collision 高，当前 best final eval success 0.0195、collision 0.0791、timeout 0.9023；5 seed 复验均值 success 0.0143、collision 0.0801、timeout 0.9082。 |
| exp020 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + terrain/safety subgoal filter | 未通过 | seed23 20M 完成；filter 将 5 seed path risk mean 从 raw 0.3815 降到 0.3187，但 success 0、collision 0.0498、timeout 0.9506，说明过滤器过强地牺牲了集合进度。 |
| exp021 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + curriculum terrain/safety subgoal filter | 未通过 | 课程化 filter 恢复集合进度：5 seed mean success 0.6361、dmax ratio 0.1460、timeout 0.1967，filtered path risk 0.3638；但 collision 0.1746，远高于 strict 0.02。 |
| exp022 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + endpoint/path safety constrained curriculum filter | 未通过 | 5 seed mean：dmax ratio 0.4719、success 0.0139、collision 0.0170、timeout 0.9699；说明 constrained filter 可压住碰撞，但过强地牺牲集合进度。 |
| exp023 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + soft progress-preserving subgoal filter | 未通过 | seed23 final eval：dmax ratio 0.1789、success 0.3027、collision 0.2295、timeout 0.4717；缓解 exp022 standoff，但 static filter 未处理同步运动碰撞。 |
| exp024 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + mutual path safety subgoal filter | 未通过 | post-hoc best `10240`：dmax ratio 0.1397、success 0.8398、collision 0.0674、timeout 0.0947；mutual path filter 明显改善 success/collision 平衡，但 strict 安全和 timeout 仍未达标。 |
| exp025 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + dense mutual path safety filter | 未通过 | best `9216`：dmax ratio 0.1434、success 0.8525、collision 0.0449、timeout 0.1035；dense mutual filter 继续降低碰撞，但仍未达到 strict 安全/timeout gate。 |
| exp026 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + hold-stable subgoal filter | 未通过 | best final eval：success 0.7529、collision 0.0615、timeout 0.1865；hold-zone 介入过早，明显压制集合进度。 |
| exp027 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + strict hold-zone filter | 未通过 | best final eval：success 0.8418、collision 0.0498、timeout 0.1123；严格触发避免 exp026 退化，但仍不优于 exp025。 |
| exp028 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + dense mutual filter + stronger hold reward | 未通过 | best final eval：success 0.8691、collision 0.0469、timeout 0.0889；当前随机地形安全/hold 方向最好结果，但安全 gate 仍失败。 |
| exp029 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + exp028 + stronger safety penalties/filter weights | 未通过 | best final eval：success 0.8262、collision 0.0557、timeout 0.1221；加强安全权重反而退化，不能作为下一步方向。 |
| exp030 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + exp028 + low-level control safety projection | 未通过 | best final eval：success 0.8330、collision 0.0313、timeout 0.1357；动态控制投影能降低碰撞，但当前触发过强，牺牲 success/timeout。 |
| exp031 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + narrow/weak control safety projection | 未通过 | best final eval：success 0.8105、collision 0.0449、timeout 0.1455；简单调弱没有恢复 success，也丢失 exp030 的安全收益。 |
| exp032 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + closing-only control safety projection | 未通过 | best final eval：success 0.8379、collision 0.0361、timeout 0.1279；closing-only 略优于 exp031，但仍未达标。 |
| exp033 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + directional agent-scale projection | 未通过 | best final eval：success 0.8154、collision 0.0488、timeout 0.1387；方向性连续缩放没有带来安全收益。 |
| exp034 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + directional mask projection | 未通过 | best final eval：success 0.8828、collision 0.0361、timeout 0.0840；mask 版本恢复部分 success/timeout，但 collision 仍超 strict。 |
| exp035 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + directional mask buffer | 未通过 | best final eval：success 0.9072、collision 0.0127、timeout 0.0811；success/collision 同时达标，timeout 成主瓶颈。 |
| exp036 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + directional mask + stronger hold/timeout shaping | 未通过 | best final eval：success 0.9336、collision 0.0088、timeout 0.0586；继续改善 timeout，但 strict 仍失败。 |
| exp037 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + directional mask + 260-step episode/eval | 未通过 | best final eval：success 0.9238、collision 0.0352、timeout 0.0410；延长 episode 降 timeout，但 collision 反弹。 |
| exp038 | 随机增强 lunar crater proxy | shared-joint MAPPO pure RL + success-zone stabilizer + 320-step episode/eval | 未通过 | 修正 best 后 final eval：success 0.9756、collision 0.0137、timeout 0.0107；旧环境栈随机地形阶段性最佳，strict 只剩 timeout 失败。 |
| exp039 | 随机增强 lunar crater proxy | exp038 checkpoint + hard near stabilizer 诊断 | 未长训 | 复评 success 0.9424、collision 0.0254、timeout 0.0322，差于 exp038；不建议按原样长训。 |
| exp040 | 随机增强 lunar crater proxy | exp038 checkpoint + stronger soft hold stabilizer 诊断 | 未长训 | 复评 success 0.9658、collision 0.0186、timeout 0.0166，timeout 差于 exp038；不建议按原样长训。 |
| exp041 | 随机增强 lunar crater proxy | exp038 checkpoint + hold-zone override 诊断 | 暂停长训 | 复评 success 0.9795、collision 0.0107、timeout 0.0098，略优于 exp038；当前不启动长训。 |
| exp042 | 随机增强 lunar crater proxy | 结构化 Actor/Critic + bicycle proxy + quintic trajectory + 25m 地图 + 无限通信工程探针 | 工程 smoke 通过 | CPU `8/8` 与 CUDA `256/64` smoke 通过；只验证环境链路，不代表策略收敛。 |
| exp043 | 随机增强 lunar crater proxy | exp042 新环境栈 + exp041 hold override + 扩大 initial-state 分布 | 未通过 | seed23 40M 完成；工程链路正常但策略几乎不集合，final eval success `0.0`、timeout `1.0`。 |
| exp044 | 随机增强 lunar crater proxy | exp043 新环境栈 + initial-state curriculum | 未通过 | seed23 40M 完成；dmax 从 exp043 明显改善到 `0.4796`，但 success `0.0`、timeout `0.9980`。 |
| exp045 | 随机增强 lunar crater proxy | exp044 新环境栈 + local-success bootstrap | 未通过 | seed23 40M 完成；success `0.1846`、collision `0.0`，证明 local bootstrap 有效但 timeout/dmax/dispersion 仍失败。 |
| exp046 | 随机增强 lunar crater proxy | exp045 local reset + terminal hold release | 未通过 | final eval success `0.6123`、collision `0.0`，但 dmax ratio `0.2424`、timeout `0.3877` 仍失败；local terminal release 有效但不足。 |
| exp047 | 随机增强 lunar crater proxy | exp046 local reset + terminal convergence release | 未通过 | final eval success `0.7188`、collision `0.0059`、dmax ratio `0.2132`，但 timeout `0.2764` 仍失败；曾是新环境栈 local reset 阶段性最好结果。 |
| exp048 | 随机增强 lunar crater proxy | exp047 local reset + terminal drive / dispersion tightening | 未通过 | dmax ratio `0.1866`、success `0.9844`、collision `0.0020` 均通过；唯一失败为 timeout `0.0137`，此前新环境栈 local reset 最佳。 |
| exp049 | 随机增强 lunar crater proxy | exp048 local reset + terminal spacing timeout closure | 未通过 | final eval dmax ratio `0.1884`、success `0.8926`、collision `0.0010`、timeout `0.1064`；过强 spacing 修正降低成功并抬高 timeout，不优于 exp048。 |
| exp050 | 随机增强 lunar crater proxy | exp048 local reset + 克制 filter/control + terminal hold RL tune | 未通过 | final eval dmax ratio `0.1847`、success `0.9590`、collision `0.0059` 达标，但 timeout `0.0352` 差于 exp048；不作为主结果。 |
| exp051 | 随机增强 lunar crater proxy | exp048 local reset + PPO stability only | 未通过 | best `013312`：dmax `0.1836`、success `0.9883`、collision `0.0020` 均通过，timeout `0.0098` 仍失败；4-seed 复验下 013312 仍是附近最好选点但 `0/4` strict。 |
| exp052 | 随机增强 lunar crater proxy | exp051 + earlier entropy taper | 未通过 | best `008192`：dmax `0.1863`、collision `0.0059` 达标，但 success `0.8955`、timeout `0.0986` 失败；过早收窄探索明显差于 exp051。 |
| exp053 | 随机增强 lunar crater proxy | exp051 + mild near reward | 未通过 | best `020480`：dmax `0.2049`、success `0.6416`、collision `0.0039`、timeout `0.3545`；全局 near reward 小幅增强也会推散队形，明显差于 exp051。 |
| exp054 | 随机增强 lunar crater proxy | exp051 + PPO clip 0.16 | 未通过 | best `017408`：dmax `0.1972`、collision `0.0029` 达标，但 success `0.7168`、timeout `0.2803` 明显失败；clip 过窄，不优于 exp051。 |
| exp055 | 随机增强 lunar crater proxy | exp051 + PPO clip 0.20 | 未通过 | best `017408`：dmax `0.1850`、success `0.9824`、collision `0.0029` 达标，但 timeout `0.0146` 失败且差于 exp051；clip `0.20` 不优于当前 best。 |
| exp056 | 随机增强 lunar crater proxy | exp051 + terminal pairwise reward | 未通过 | best `012288`：dmax `0.1864`、success `0.9873`、collision `0.0010` 达标，但 timeout `0.0117` 失败且差于 exp051；pairwise reward 过早/偏强。 |
| exp057 | 随机增强 lunar crater proxy | exp051 + strict terminal pairwise reward | 未通过 | best `011264`：dmax `0.1850`、success `0.9697`、collision `0.0059` 达标，但 timeout `0.0254` 明显差于 exp051；不继续该方向。 |
| exp058 | 随机增强 lunar crater proxy | exp051 + PPO gamma 0.995 | 未通过 | best `012288`：dmax `0.1991`、collision `0.0020` 达标，但 success `0.7451`、timeout `0.2529` 明显失败；更长折扣 horizon 拖慢 terminal convergence。 |
| exp059 | 随机增强 lunar crater proxy | exp051 + PPO GAE 0.90 | 未通过 | best `012288`：dmax `0.1927`、collision `0.0127` 达标，但 success `0.6904`、timeout `0.2988` 明显失败；更短 GAE trace 也会破坏 terminal convergence。 |
| exp060 | 随机增强 lunar crater proxy | exp051 + PPO value loss 0.75 | 未通过 | best `012288`：dmax `0.1837`、success `0.9736`、collision `0.0` 达标，但 timeout `0.0264` 失败且差于 exp051；更强 critic loss 权重没有改善尾部 hold。 |
| exp061 | 随机增强 lunar crater proxy | exp051 + terminal gate Actor/Critic observation | 未通过 | best `020480`：dmax `0.1890` 达标，但 success `0.8506`、collision `0.0205`、timeout `0.1289` 失败；直接暴露 gate margin 给 Actor 导致策略更激进，不作为主线。 |
| exp062 | 随机增强 lunar crater proxy | exp051 + critic-only min_pairwise state | 未通过 | best `016384`：dmax `0.1832`、success `0.9736`、collision `0.0059` 达标，但 timeout `0.0205` 失败；4-seed sweep timeout mean `0.0161`，不优于 exp051。 |

历史完整 suite checkpoint：

```text
outputs/runs/exp_008_terrain3d/_suite/checkpoints/
```

这些 exp008 checkpoint 使用旧 observation schema，历史 strict 结论仍有效，但不能直接加载到当前 86 维 Actor。exp017 已产生当前 schema 的单 seed strict checkpoint；exp014 checkpoint 仍仅用于工程探针。

## 结果解释边界

- exp006 / exp008 是 proxy strict pass，不是 Isaac Lab 物理训练 pass。
- exp006 / exp008 的 checkpoint 属于旧 observation schema；结果可作为历史 baseline，但不能与新 Actor 接口直接混用。
- exp014 只通过有限值、参数更新、地形输入权重更新和动作非退化检查，不是 strict convergence pass。
- PhysX / Jackal 结果应写成“Jackal 在 PhysX 场景中的轨迹跟踪验证结果”或“proxy checkpoint 的高保真迁移 sanity check”，不能写成“物理环境训练结果”。
- exp017 可以表述为“固定地图、seed23、proxy pure RL 从零通过 strict gate”，不能扩展为多 seed、随机地图或 PhysX 收敛。
- exp018 可以表述为“随机增强地形下已获得稳定集合趋势和较高 success，但安全/超时 gate 未完全收敛”，不能写成随机地图 strict pass。
- exp019 可以表述为“成功区安全间距和路径级地形风险链路已接入并可训练/评估，但当前 reward 下策略仍在成功率、碰撞率和超时之间失衡”，不能写成安全地形策略收敛。
- exp020 可以表述为“子目标过滤器确实降低了路径风险，但当前 hard post-processing 过强，导致探索/集合进度塌缩”，不能写成地形规避策略成功。
- exp021 可以表述为“课程化 filter 恢复了集合趋势，但碰撞率显著过高”，不能写成随机地形 strict pass 或安全策略成功。
- exp022 可以表述为“endpoint/path safety constrained filter 把 collision 压到 strict 内，但 success/timeout 严重失败”，不能写成随机地形安全策略收敛。
- exp023 可以表述为“soft progress filter 缓解了 exp022 的 standoff，但 collision/timeout 仍严重失败”，不能写成随机地形安全策略改善或收敛。
- exp024 可以表述为“mutual path safety 明显改善 exp023 的动态路径冲突，但 collision/timeout 仍未过 strict”，不能写成随机地形安全策略收敛。
- exp025 可以表述为“dense mutual path safety 相对 exp024 进一步降低 collision，但仍未解决末段 hold / timeout 稳定性”，不能写成随机地形安全策略收敛。
- exp026/exp027 可以表述为“hold-zone filter 诊断未改善 exp025，过早介入会压制集合”，不能写成 hold 稳定成功。
- exp028 可以表述为“强化 success hold reward 提高了 success/timeout，是 exp026–029 中最好的随机地形结果”，但 collision 仍超 strict，不能写成安全收敛。
- exp029 可以表述为“继续加强安全权重没有降低真实 collision，反而牺牲 success/timeout”，不能写成安全改善。
- exp030 可以表述为“低层动态控制投影降低 collision 但牺牲 success/timeout”，不能写成安全收敛或 strict 改善。
- exp031–exp034 可以表述为“控制层投影条件和方向性 mask 诊断”，不能写成随机地形安全收敛；exp034 是方向性 mask 的有效拐点，但 collision 仍失败。
- exp035/exp036 可以表述为“success/collision 已同时过门槛但 timeout 仍失败”，不能写成 strict pass。
- exp037 可以表述为“延长 episode 降 timeout 但导致 collision 反弹”，不能写成单纯时间预算不足。
- exp038 可以表述为“旧环境栈随机地形阶段性最佳候选，strict 只剩 timeout 尾部未过”，不能写成 strict pass。
- exp039/exp040 只是 exp038 checkpoint 复评诊断，不能写成长训练结果。
- exp041 可以表述为“hold-zone override 在 exp038 checkpoint 上略有改善”，不能写成 exp041 长训练完成。
- exp042 可以表述为“训练环境三项核心改造和 25m/无限通信设置的工程探针通过”，不能写成策略训练收敛或 strict pass。
- exp043 可以表述为“新环境栈直接长训未收敛，主要表现为集合进度不足而非碰撞或数值异常”，不能写成 strict pass。
- exp044 可以表述为“initial-state curriculum 改善了靠拢但仍未产生 success”，不能写成 strict pass。
- exp045 可以表述为“local-success bootstrap 把 success 从 0 提升到 0.1846，但仍未收敛”，不能写成 strict pass。
- exp046/exp047 可以表述为“新环境栈 local reset 下逐步恢复 terminal convergence”，不能写成完整难度 strict 收敛。
- exp048/exp051 可以表述为“dmax/success/collision 已过，剩余 timeout 尾部未清零”，不能写成 strict pass。
- exp049/exp050 可以表述为“增强 spacing/filter 或增强 hold/timeout shaping 的负结果”，不能作为下一步主方向。
- exp052/exp054 可以表述为“PPO 探索或更新过早/过强收窄会明显降低 success 并抬高 timeout”，不能作为下一步主方向。
- exp055 可以表述为“稍微放宽 PPO clip 能恢复 exp051 附近的 success/collision，但没有改善 timeout 尾部”，不能作为当前主结果。
- exp056/exp057 可以表述为“terminal pairwise reward 能轻微影响最近邻/碰撞，但没有改善 timeout，且严格弱化后仍扰动 success/hold”，不能作为下一步主方向。
- exp058 可以表述为“提高 PPO gamma 到 0.995 明显拖慢 terminal convergence”，不能作为下一步主方向。
- exp059 可以表述为“降低 GAE lambda 到 0.90 明显降低 success 并抬高 timeout，且训练后期策略质量坍缩”，不能作为下一步主方向。
- exp060 可以表述为“提高 value loss 权重到 0.75 保留了 dmax/success/collision 达标，但 timeout 明显差于 exp051”，不能作为下一步主方向。
- exp061 可以表述为“terminal gate 特征直接进入 Actor 会导致动作更激进、success/collision/timeout 同时退化”，不能作为下一步主方向。
- exp062 可以表述为“critic-only 显式 min_pairwise state 保留了 dmax/success/collision 达标，但 timeout 仍差于 exp051”，不能作为当前主结果。
- exp051 没有改变 Actor 输出语义，也没有引入多点采样；当前主线仍是单点 `[rho, beta]` 子目标输出加原有 filter/control 兜底。exp051 multi-seed 复验只改变评估采样，不改变 policy 或底层约束。
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

1. 当前新环境栈 local reset 最好候选是 exp051：dmax/success/collision 已过 strict，仅 timeout `0.0098` 未过。
2. exp051 说明 PPO 稳定性调整有小收益；exp050 说明增强 terminal hold / timeout shaping 会抬高 timeout，不能作为主线。
3. 暂时维持原动作输出，不改为多点采样让 filter 选择；下一轮仍应把 RL policy 作为主体，filter/control 只保留原有兜底语义。exp051/exp060 gate 诊断显示 timeout 主要卡在 terminal `min_pairwise` gate，而不是轨迹速度或 dispersion，因此把 filter 升级成候选选择器会偏离当前“RL 主体、filter 兜底”的口径。
4. 不建议继续全局加硬 near/hold filter；exp039/exp040 和 exp049 已说明会扰动 success 或 timeout。
5. exp052 说明不能把 entropy taper 提前到 `8192`；exp053 说明不能继续提高全局 near reward；exp054/exp055 完成 clip 两侧扫描，当前最好仍是 exp051 的 `clip_epsilon=0.18`；exp056/exp057 说明 terminal pairwise reward 不是有效收敛方向；exp058/exp059 说明拉长 `gamma` 或缩短 `gae_lambda` 都会拖慢或破坏 terminal convergence；exp060 说明单纯提高 `value_loss_coef` 也不能清掉 timeout；exp061/exp062 说明直接加 terminal gate Actor 特征或 critic-only min_pairwise state 都不优于 exp051。下一轮仍应回到 exp051，避免继续扩张安全间距项、filter/control 权限或直接 gate 特征，优先考虑更保守的训练稳定性/seed 稳健性诊断，或只做非常窄的末端 hold 学习信号对照；checkpoint selection 继续只作为诊断，不作为主修复。
6. 保留 exp017 作为固定地图 pure RL baseline，保留 exp018–exp041 作为随机地形 failure analysis，不把它们扩写为多 seed 或 PhysX 收敛。
7. 为 PhysX / Jackal 后续接入同布局 raycast / height scanner，保持 proxy 与高保真观测接口一致。
