# 差速轨迹原语、可观测性修正与可信收敛评测实施计划

本文是当前执行计划。`exp156` 的N0/N1完整训练均出现成功率地板，N2完整训练已取消；后续暂用N1 CNN作为诊断接口，但没有架构通过strict验收。

## 1. 目标与边界

当前执行链路固定为：

```text
295维多尺度局部观测与分级通信
→ N1共享多尺度CNN Actor（临时基线）
→ 47维离散差速轨迹原语
→ 时标化位姿轨迹
→ 左右轮差速控制
```

研究保持Pure RL、shared-joint MAPPO、CTDE和执行期严格去中心化。训练时Critic可读取集中式状态；执行时Actor、轨迹生成器和控制器只能读取本车观测与通信缓存。

本轮继续禁止：

- BC和历史checkpoint初始化；
- 安全投影、方向性mask和子目标过滤；
- 集合槽位修正、集中式动作覆盖和在线MAPF；
- Oracle集合点、全局质心、全局dmax和未通信邻车状态进入执行链；
- 同时增加DAE、GRU、GNN或可学习通信。

标准MAPPO仍是唯一训练算法。只有本轮完整实验失败，且冻结因果审计确认空间信用分配是主要瓶颈时，才另立DAE计划。

## 2. 公共接口

### 2.1 47维差速轨迹原语

动作索引及语义固定如下：

| 索引 | 数量 | 语义 |
| --- | ---: | --- |
| 0 | 1 | 真实hold，线速度和角速度均为零 |
| 1–39 | 39 | 原有13个前进终点与3个速度档的笛卡尔积 |
| 40–42 | 3 | 短距倒车至 $(-0.4,-0.4)$、$(-0.4,0)$、$(-0.4,0.4)$ |
| 43–44 | 2 | 原地转向，计划航向变化为 $\pm\pi/4$ |
| 45–46 | 2 | S形左右让行至 $(0.8,\pm0.8)$，末端航向与起始航向平行 |

前进速度为：

$$
v\in\{0.45,0.80,1.15\}\ \mathrm{m/s}.
$$

倒车参考速度为 $-0.45\ \mathrm{m/s}$，S形让行速度为 $0.45\ \mathrm{m/s}$。平移动作使用quintic轨迹；hold和原地转向使用位置不变的定时位姿轨迹。轨迹显式保存：

- `motion_direction`；
- `planned_yaw_delta`；
- `primitive_type`。

这些字段避免倒车被跟踪器解释为掉头，也使原地转向不依赖伪造的位置目标。

### 2.2 差速执行

Jackal代理采用：

$$
\omega_L=\frac{v-\frac{b}{2}\omega}{r},
\qquad
\omega_R=\frac{v+\frac{b}{2}\omega}{r},
$$

其中 $r=0.098\ \mathrm m$，$b=0.376\ \mathrm m$。左右轮分别裁剪到 $[-18,18]\ \mathrm{rad/s}$，再反算代理环境实际执行的线速度和角速度：

$$
v_{\mathrm{eff}}=\frac{r}{2}(\omega_L+\omega_R),
\qquad
\omega_{\mathrm{eff}}=\frac{r}{b}(\omega_R-\omega_L).
$$

代理与PhysX因此共享同一轮速边界。日志同时记录左右轮命令、有效角速度和转弯半径。

### 2.3 295维Actor观测

观测schema为 `ego_v10_multiscale_diff_intent`：

$$
15+3\times17+224+5=295.
$$

- ego：15维；
- neighbor：最多3辆邻车，每车17维；
- terrain：224维多尺度本车坐标系地形；
- aggregation：仅由通信缓存计算的5维统计量。

v10的基础ego状态不再包含可形成世界坐标捷径的绝对平面位置和绝对航向；速度转换到车体坐标系。基础10维布局保持稳定，其中全局平移和旋转槽位被规范化为常量。其后加入本车上一规划步的局部终点2维、参考速度1维、计划航向变化1维和固定协调令牌1维。

三个地形尺度分别为：

$$
\begin{aligned}
\mathcal G_f:&\ x=-0.4:0.2:0.8,\quad y=-0.8:0.2:0.8,\\
\mathcal G_m:&\ x=(0.8,1.2,1.6),\quad y=-1.2:0.4:1.2,\\
\mathcal G_c:&\ x=(1.6,2.4,3.2,4.0),\quad y=-2.4:0.8:2.4.
\end{aligned}
$$

每个采样点包含相对高度和风险两个通道，因此：

$$
2(7\times9+3\times7+4\times7)=224.
$$

### 2.4 17维分级通信

12 m内，在原12维消息后加入：

$$
[\Delta x_b^{\mathrm{plan}},\Delta y_b^{\mathrm{plan}},
v^{\mathrm{plan}},\Delta\psi^{\mathrm{plan}},z_i].
$$

新增的计划信息只描述发送车辆上一规划步已经承诺的轨迹，不是当前待选动作。12 m外立即清零速度、地形、轨迹终点、计划速度、计划航向变化和协调令牌，只保留低频位置与航向快照。通信缓存仍只在真实环境步和reset更新。

### 2.5 950维统一Critic

Critic状态为原54维集中式状态加四辆车各自的224维地形：

$$
54+4\times224=950.
$$

每车8维全局运动状态编码为32维；每车224维地形由共享多尺度CNN编码为32维，两者融合为32维。四车特征使用mean/max聚合，再与团队8维、地形摘要5维及Oracle 9维分支共同进入价值主干。Critic与Actor参数完全分离，执行时不加载。

### 2.6 Oracle边界

主线固定：

```yaml
reward:
  weights:
    oracle: 0.0
```

Oracle仅允许进入Critic、诊断、离线评测和集中式上界。策略奖励与成功只依据实际团队质心、实际平整度、集合进展、碰撞和路径风险。

当前不存在获胜Actor，因此Oracle奖励恢复消融暂停。N1只用于保持后续H0/H1接口一致，不恢复Oracle奖励。

## 3. 初始化与探索

每次reset均执行：

- 队形整体旋转从 $[-\pi,\pi]$ 均匀采样；
- 每辆车初始航向独立从 $[-\pi,\pi]$ 均匀采样；
- 不再固定朝向世界原点；
- 保留团队中心、地形平移和地形旋转随机化。

熵系数在完整153,600训练时步内线性衰减：

```yaml
entropy_loss_scale: 0.0009
entropy_loss_scale_end: 0.0001
entropy_schedule_timesteps: 153600
```

训练记录归一化熵、有效动作数、hold概率，以及前进、倒车、原地转向和让行四个动作族的采样比例与策略概率。

## 4. 网络结构消融结论

三种Actor共享295维输入、47维输出、统一950维Critic、奖励、课程、seed和场景清单：

- N0：224维地形展平后通过两层MLP；83,343个参数；
- N1：三个尺度使用共享两层CNN，经空间池化后融合；106,591个参数；
- N2：共享多尺度CNN，对13条前进quintic路径采样，所有47个动作同时由三尺度地形上下文条件化；31,752个参数。

三者均低于12万参数。N0/N1各完成39,321,600环境交互和1152场景配对评测，但success分别为0和0.0017，strict分层均为0/6。该地板效应使编码器排名失去解释力。N2完整训练取消；修复非连续路径采样网格后只做CUDA训练smoke。N1因具有明确多尺度空间归纳偏置而暂作后续诊断接口，不构成性能胜出结论。

## 5. 训练前工程门限

正式训练前必须全部通过：

1. 295维观测、47维Categorical和950维Critic单元测试；
2. hold、倒车、原地转向、S形轨迹和轮速裁剪测试；
3. 17维通信与12 m清零测试；
4. 旧291维、40动作和Gaussian checkpoint拒绝测试；
5. 固定缓存下Oracle与未发送全局量的执行不变性测试；
6. 场景整体SE(2)变换下Actor观测和logits不变性测试；
7. CPU小环境和CUDA 256环境的真实PPO smoke；
8. 冻结原语覆盖审计。

覆盖审计不生成教师数据。当前审计包含5个 `exp155` 后期全hold状态和7个人工相向/交叉场景；每个场景枚举4车联合逃逸原语。准入要求为至少90%的场景在16步内存在无碰撞解，并且倒车、原地转向、S形让行均至少一次参与有效解。

## 6. Pure RL训练停止决策

原预注册预算为：

```yaml
seed: 23
parallel_envs: 256
rollout_length: 64
stage_a_iterations: 800
stage_b_iterations: 800
stage_c_iterations: 800
total_policy_iterations: 2400
environment_interactions: 39321600
bc_updates: 0
init_checkpoint: null
```

N0和N1已按该预算完成三阶段训练：

1. Stage A：Open、近距，碰撞惩罚但不终止；
2. Stage B：Open、近距，恢复碰撞终止；
3. Stage C：加入Mixed、Bottleneck和远距分级通信。

N0/N1均未突破成功率地板，N2不再投入完整预算。当前不选择获胜结构、不开展Oracle消融，也不训练seed31和seed47。下一步仅使用N1接口执行H0/H1可辨识性与低层可解性诊断。

## 7. 配对评测与统计验收

固定 `scenario_manifest` 记录每个场景的初始位置、航向、地形运行参数、距离组、拓扑组、seed和内容哈希。所有候选必须复现相同哈希；不再以不同基础seed代替配对场景。

每个checkpoint正式评测：

```text
近距/远距 × Open/Mixed/Bottleneck × 192
= 1152 episodes
```

每个训练seed、每个分层均须满足：

- collision的Clopper–Pearson单侧95%上界不高于0.02；
- success的Clopper–Pearson单侧95%下界不低于0.90；
- timeout的Clopper–Pearson单侧95%上界严格低于0.10；
- dmax ratio的点估计和单侧bootstrap 95%上界均不高于0.20。

192个episode对应：

- collision必须为 `0/192`；
- success至少为 `180/192`；
- timeout最多为 `12/192`。

报告同时给出点估计、置信区间、分层IQM及架构间配对bootstrap差异。64个episode的训练中间评测只能标记为diagnostic，不得标记为strict。

## 8. OOD与最终验收

冻结OOD评测包括：

- 随机初始航向对照世界原点朝向；
- 场景整体平移和旋转；
- 存在多个平坦候选区域的场景；
- 固定通信缓存后修改Oracle、槽位、未发送邻车状态和全局诊断量。

最后一项要求Actor logits、动作、局部轨迹和左右轮命令完全不变。三个训练seed全部通过六个分层后，才生成正式动画并更新 `docs/technical_design.md`。

## 9. 运行与产物

主配置：`configs/experiment/exp156_differential_multiscale_ablation.yaml`。

历史正式入口保留用于复现，但其完整训练候选已收缩为N0/N1，N2仅保留CUDA smoke；不得用该入口重新启动N2完整训练：

```bash
.venv_isaaclab/bin/python3.12 scripts/run_exp156_architecture_comparison.py --device cuda:0
```

主产物位于：

```text
outputs/runs/exp156_differential_multiscale_ablation/
├── _suite/
│   ├── scenario_manifest.json
│   ├── suite_status.json
│   ├── logs/launcher.log
│   └── metrics/
├── n0_seed23_full_2400iter/
├── n1_seed23_full_2400iter/
└── n2_seed23_full_2400iter/
```

`exp155`的N0产物保留为 `stopped_design_revision`，checkpoint仍为 `candidate`，但明确排除suite排名和strict汇总。

## 10. 文档公式约束

活动文档的行内公式统一使用 `$...$`，块级公式统一使用独占行的 `$$...$$`。禁止使用反斜杠圆括号或反斜杠方括号作为数学分隔符。提交或启动实验前运行：

```bash
.venv_isaaclab/bin/python3.12 scripts/check_markdown_math_delimiters.py
```
