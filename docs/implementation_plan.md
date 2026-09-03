# D-STC严格去中心化共同站点证书实施计划

本文是当前唯一执行计划。`exp158`的学习式DAE和`exp159`的解析式ALO-PRD均在训练前离线门限停止；当前不继续信用算法或无限短训，转而推进D-STC路线一：本地平地区域proposal、严格去中心化共同站点commit、站点条件Actor和短时轨迹承诺。

## 0. 当前阶段与停止边界

exp160静态H0、exp161四路线比较、exp162主动探索失败诊断、exp163修复基准后的H0.5以及exp165完整闭环pilot均已完成。exp165证明delta/event通信可保持完整洪泛的digest/site语义并减少84.52%记录，但R4闭环success仅21.88%–43.75%，未通过：

```text
DISCOVER/VERIFY：已通过
→ EXCHANGE/COMMIT：已通过
→ delta/event通信：已通过
→ R4 GATHER闭环：未通过
→ 每层192场景：停止
```

原exp156 Bottleneck同时使用通道墙和100个陨石坑，内部可行平地几乎被清空，Oracle高度集中于地图边界。exp163保留通道墙，仅将Bottleneck陨石坑数调整为30；这是当前正式Active-DSTC评测基准。不得把exp163与原exp156 Bottleneck指标直接作同地图配对比较。

exp165已经完成下列准入条件的实际检查：

1. delta消息不能改变最终proposal-set digest和site id；
2. 陈旧、重复、乱序delta必须幂等；
3. 平均传输记录量相对完整缓存洪泛至少下降70%；
4. R4只能读取本车状态、commit证书和12 m内邻车已承诺原语；
5. R4替换Pure RL Actor，不允许Actor后动作覆盖；
6. 32环境六分层完整闭环同时满足证书、实际质心平整度、dmax、dispersion、低速hold和安全门限；
7. 32环境通过后才运行每层192场景；
8. 本轮不训练高层utility、Actor、GNN、GRU或通信编码器。

其中1–3通过，4–5完成工程接入，条件6失败，因此条件7明确停止。下一阶段不得继续扩大当前R4评测或扫描代价权重；只允许使用冻结exp165轨迹诊断原语承诺振荡，并预注册一个跨时稳定性机制。固定本车观测和消息日志后，修改Oracle及全局地形真值不得改变候选、commit、原语或轮速命令。exp165不得标记为项目最终success或strict pass。

exp164已经额外完成407维H1标准MAPPO的78.6M交互长训。最终success 86.98%、collision 12.50%，未通过；Stage B近距Open曾达到success 98.44%、collision 1.30%，但Stage C碰撞保持约12%–27%。该结果关闭“继续延长Pure RL低层预算”的分支，进一步支持使用显式邻车承诺原语的R4 GATHER。

以下章节保留exp156执行接口和exp157—159诊断边界，作为D-STC低层基线与历史停止依据。

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
- 在DAE之外同时增加GRU、GNN、注意力或可学习通信。

标准shared-joint MAPPO仍是唯一基线。DAE只改变训练期advantage；离线前置门限失败时不得启动完整训练或切换其他信用算法。

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

## 11. exp157 H0/H1因果拆分

停止继续比较编码器后，先执行两个串行诊断：

1. H0不训练，使用固定1152场景审计局部站点证据、12 m通信图、站点势场SE(2)不变性、终端区域容量和47维联合原语可行性；
2. H1只使用N1，将一个共享可行站点区域编码为第三个多尺度空间通道，从零进行39.3M Pure RL训练。

H1的407维观测与0.5 Oracle进展权重只构成低层goal-conditioned能力上界。只有H1显著成功、H0同时表明当前信息边界不足时，才允许制定完整D-STC工程计划；H1失败时优先检查低层动作、奖励和终端稳定，不增加候选共识模块。

## 12. exp158 DAE-MAPPO两级验证

### 12.1 当前阶段

`exp157-H1`旧run标记为 `incomplete_interrupted`，只允许其134,400步checkpoint为离线审计提供行为分布。exp158当前顺序固定为：

```text
工程测试与GAE等价性
→ 冻结因果及反事实奖励可辨识性审计
→ H1-GAE/H1-DAE seed23完整配对
→ H1 seed31/47
→ Strict-GAE/Strict-DAE seed23完整配对
→ Strict seed31/47
```

任一门限失败即停止后续阶段。正式离线审计现已完成并失败，因此当前执行在第一道门限终止，H1和strict训练未启动。

### 12.2 DAE定义

对车辆 $i$：

$$
\overline r_{i,t}
=
\sum_{a_i'=1}^{47}
\pi_{\theta_{\mathrm{old}}}(a_i'\mid o_{i,t})
\widehat r_\psi(s_t,\mathbf a_{-i,t},a_i'),
$$

$$
C_{i,t}
=
\overline r_{i,t}
+\gamma\lambda\beta(1-d_t^{\mathrm{episode}})C_{i,t+1},
$$

$$
A_{i,t}^{\mathrm{DAE,raw}}
=
A_t^{\mathrm{GAE,raw}}-\beta C_{i,t}.
$$

其中 $gamma=0.99$、$lambda=0.95$。`terminated`与`truncated`均切断反事实trace。Critic继续拟合原团队return，环境奖励不改变。

β日程不扫描：前128次更新为0，随后128次线性升至0.3，余下训练固定0.3。奖励模型只预测团队总即时奖励，不增加分量辅助损失。

### 12.3 离线门限

正式审计使用128环境×480步事实训练集、两个独立64环境×480步验证集，以及六分层共384个冻结状态的72,192个真实单步反事实标签。替代动作标签只用于验证，不能训练奖励模型。

必须同时通过时间信用、单车边际贡献、observation aliasing、总奖励MSE改善、关键结果动作辨识、反事实输出非塌缩、动作排序和policy-weighted期望误差门限。

正式结果中，时间信用、aliasing、输出非塌缩和动作排序通过；冲突参与者边际比、共享advantage相关、总奖励MSE改善及policy-weighted期望误差失败。尤其期望误差为2.006个真实奖励标准差，远高于0.25门限。因此 `decision=stop_before_dae_training`。完整数值见[exp158实验记录](experiments/exp_158_dae_validation.md)。

### 12.4 配对预算

H1与strict均使用N1、256环境、rollout 64、三个固定800-iteration阶段。每个run为153,600训练步和39,321,600环境交互。seed23两臂完整结束并通过相对及绝对门限后，才运行seed31和47；不得短训淘汰、延长或选择中间checkpoint。

最大两级预算为471,859,200环境交互。所有run串行执行。

### 12.5 执行边界

奖励模型读取950维集中状态和联合动作，只在PPO更新中使用。训练checkpoint将其保存在 `dae_training` 非部署字段；Actor observation、通信、轨迹和轮速命令均不增加集中信息。固定Actor后修改奖励模型、Oracle或未发送状态，执行输出必须不变。

## 13. exp159解析式ALO-PRD

### 13.1 启用原因

exp158正式审计中，DAE动作排序勉强通过，但policy-weighted期望误差达到2.006个真实奖励标准差。exp159不扩大reward model，而是使用只依赖其他车辆动作的解析基线：

$$
A_{i,t}^{\mathrm{PRD,raw}}
=
A_t^{\mathrm{team,raw}}-b_{i,t}^{\mathrm{LOO}}.
$$

基线只减去当前步噪声，不进行多步trace；Critic return和团队奖励保持不变。

### 13.2 固定语义

LOO基线包含其他三车的动作成本、可加地形成本、路径最大风险、H1 Oracle进展、排除本车后计算的near penalty，以及其他三车内部碰撞、越界和对应failure penalty。

dmax、dispersion、实际质心平整度、success、hold和timeout始终保留在共享团队GAE。baseline scale固定为1，不允许扫描。

与exp150不同，exp159不构造车辆间零和残差。碰撞参与车保留原团队惩罚，非参与车只移除与自身当前动作无关的碰撞噪声，参与车不会被二次惩罚。

### 13.3 串行门控

执行顺序固定为：

```text
专项测试与rollout64 smoke
→ A-H1冻结无偏性/方差审计
→ A-strict冻结审计（仅A-H1通过后）
→ H1 seed23完整GAE/PRD配对
→ H1 seed31/47
→ strict seed23及seed31/47
```

A-H1必须同时通过奖励不变、source重构、本车47动作不变性、全数据梯度一致性和两个验证seed至少15%的梯度方差降低。任一失败即停止后续阶段。

正式A-H1结果中，奖励不变、source重构、本车动作不变性和梯度一致性均通过；但baseline/raw advantage覆盖率为3.45%/2.22%，梯度方差降低为7.06%/0.42%，未达到10%/15%门限。因此exp159在A-H1终止，A-strict和所有完整训练不启动。

### 13.4 训练与验收

每个正式run继续使用256环境、rollout 64、三个固定800-iteration阶段和39,321,600环境交互。seed23不使用短训淘汰；最终只比较153,600步checkpoint和固定1152场景。

完整配置、门限、产物及结果见[exp159实验记录](experiments/exp_159_analytical_prd.md)。
