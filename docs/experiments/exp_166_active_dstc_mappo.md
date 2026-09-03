# exp166 Active-DSTC条件MAPPO主线

## Material Passport

- Origin Skill：`academic-research-suite`
- Origin Mode：experiment run
- Origin Date：2026-08-26
- Verification Status：RUNNING

## 目的

恢复强化学习主线：Active-DSTC只维护本地候选belief、delta/event通信和四车共同站点证书，共享MAPPO Actor从回合第一步到结束始终直接选择47维差速轨迹原语。R4不进入执行链。

## 执行链

```text
本车状态、多尺度局部地形和通信缓存
→ Active-DSTC本地belief/证书空间势场
→ N1共享Categorical Actor
→ 47维差速轨迹原语
→ quintic或定时转向轨迹
→ 左右轮命令
```

Active-DSTC不生成探索动作，不覆盖Actor动作。Actor在DISCOVER、VERIFY、EXCHANGE、COMMIT和GATHER阶段均保持控制权。

## 配置

- 正式配置：`configs/experiment/exp166_active_dstc_mappo_n1.yaml`；
- smoke配置：`configs/experiment/exp166_active_dstc_mappo_smoke.yaml`；
- launcher：`scripts/run_exp166_active_dstc_mappo.py`；
- Actor：407维站点belief观测、N1多尺度CNN、47维Categorical动作；
- Critic：950维集中式状态，站点分支使用Active-DSTC训练期聚合状态，不使用Oracle点；
- 算法：shared-joint MAPPO、标准GAE；
- BC：关闭；Oracle奖励：0；
- R4、安全投影、方向性mask和Actor后处理：关闭；
- seed23、256并行环境、rollout 64、4800 iterations、307,200训练步，约78.6M环境交互。

## Active-DSTC奖励

训练奖励只加入三项直接由Actor轨迹影响的信号：

$$
r_t^{\mathrm{DSTC}}
=c_b(\Phi_{t+1}-\Phi_t)
+c_c\mathbb I_t^{\mathrm{new\ commit}}
+c_d\mathbb I_t^{\mathrm{committed}}(d_{t-1}^{c}-d_t^{c}).
$$

共同站点来自Active-DSTC证书，不是Oracle集合点或车辆专属槽位。

## 课程

沿用固定三阶段、每阶段1600 iterations：Open近距软碰撞、Open近距硬碰撞、修复后的Mixed/Bottleneck与远距分布。课程按固定训练步切换，不因中间评测提前停止、延长或修改奖励。

## 工程门

已经通过407维Actor、950维Critic、47维动作、Oracle输入不变性、Actor动作不被覆盖、无站点时势场清零、子环境reset隔离、CPU 2环境PPO更新和CUDA 256环境PPO更新。Actor、Critic、terrain encoder和neighbor encoder均发生非零更新；BC为0且无NaN。

## 训练状态

正式seed23训练已经启动。实时状态：

```text
outputs/runs/exp166_active_dstc_mappo/_suite/suite_status.json
```

正式run：

```text
outputs/runs/exp166_active_dstc_mappo/n1_seed23_full_4800iter/
```

## 结论边界

当前只证明新主线工程链路和训练更新有效，尚不能声称收敛、优于R4或通过strict gate。最终只使用307,200步checkpoint和固定六分层评测作出结论。
