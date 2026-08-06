# exp138：现有安全奖励聚合语义辨识诊断

## 目的

exp128证明当前团队 `safety` 奖励的联合动作辨识增益只有约0.1%，而最近邻距离变化的增益超过56%；exp137又表明增强邻居关系建模只会加快几何收缩并显著增加碰撞。这说明下一步应先检查现有安全量如何汇总为团队奖励，而不是继续更换网络。

exp138只离线比较两种现有近距惩罚聚合方式，不增加奖励项、动作后处理或网络：

\[
p_{\mathrm{mean}}
=
\frac{1}{N}\sum_i
\max\!\left(d_{\mathrm{near}}-d_i^{\mathrm{nn}},0\right),
\]

\[
p_{\mathrm{worst}}
=
\max_i
\max\!\left(d_{\mathrm{near}}-d_i^{\mathrm{nn}},0\right).
\]

当前实现使用 \(p_{\mathrm{mean}}\)。候选 \(p_{\mathrm{worst}}\) 只改变同一安全项内部的聚合算子，仍为团队标量，不引入pair-repeated冲突奖励、CBS约束或逐车辅助advantage。

## 方法

冻结exp125 `relative_quintic` Actor和环境设置，完整采集96秒episode。对每个状态分别保存实际采样联合动作以及执行后的两种安全目标：

\[
r_{\mathrm{safety}}^{x}
=
-w_s
\left(
c_{\mathrm{near}}p_x
+c_{\mathrm{collision}}\mathbb{I}_{\mathrm{collision}}
\right),
\qquad
x\in\{\mathrm{mean},\mathrm{worst}\}.
\]

为每个目标训练结构、容量、样本和随机种子完全相同的状态模型与状态—联合动作模型。动作辨识增益定义为：

\[
I_x=
\frac{
\operatorname{MSE}_{s,x}-\operatorname{MSE}_{s,a,x}
}{
\operatorname{MSE}_{s,x}
}.
\]

模型对每个目标单独标准化，因此 \(p_{\mathrm{worst}}\) 数值幅度更大不会自动获得优势。

## 数据与门限

- checkpoint：exp125 `b0_screen_seed23_4m_relative_quintic/checkpoints/best.pt`
- 训练数据：128环境×480步，seed30023；
- 验证数据：每个种子64环境×480步，seed31023和32023；
- 回归模型种子：7、17、29；
- 两层128单元ELU，30 epochs；
- Actor参数与探针动作必须完全不变。

只有同时满足以下条件，才允许另行规划一次“只替换安全聚合算子”的4M：

- \(I_{\mathrm{worst}}\) 的双种子均值不低于15%；
- 每个验证种子的 \(I_{\mathrm{worst}}\) 均不低于15%；
- 相对当前聚合的动作增益提升在每个验证种子均不少于10个百分点；
- 每个验证种子的候选安全目标激活率均不低于5%，排除零方差目标造成的误差比假阳性；
- 两种目标的碰撞终止项、系数和激活阈值完全一致；
- Actor参数和探针动作不变。

任一条件失败则停止，不扫描softmax温度、top-k、距离阈值或安全权重。

## 结果

正式数据完整覆盖96秒episode：训练61,440个样本，两个验证种子各30,720个样本。当前mean目标重构误差为0，Actor参数摘要前后完全一致。

| 验证seed | 安全目标激活率 | mean聚合动作增益 | worst-pair动作增益 | 增益差 |
| ---: | ---: | ---: | ---: | ---: |
| 31023 | 28.906% | 0.0853% | 0.0973% | 0.0120个百分点 |
| 32023 | 25.837% | -0.0165% | 0.00016% | 0.0166个百分点 |
| 双种子均值 | — | 0.0344% | 0.0487% | 0.0143个百分点 |

激活率显著高于5%，但worst-pair聚合既未达到15%的绝对动作增益，也未相对mean聚合提高10个百分点。三个模型种子的结果同样接近零，不能归因于单次回归初始化。

## 结论

exp138状态为 `stop_safety_aggregation_change`。将四车平均近距gap改成最危险车辆对gap不能解决安全信用问题，因此不启动4M，也不扫描softmax、top-k、距离阈值或权重。

结合exp128和exp137，当前证据已经排除两种较简单解释：失败既不是邻居排列建模不足，也不是安全gap被四车均值明显稀释。瓶颈位于共享团队即时奖励与单车动作责任之间的对应关系。继续训练需要重新审议“禁止逐车训练信用或新的约束优化目标”的边界；在边界未修订前，不应继续堆叠网络或参数扫描。

## 产物路径

- `outputs/runs/exp138_safety_aggregation_identifiability/frozen_exp125_seed23/`
- `metrics/safety_aggregation_identifiability.json`
- `artifacts/diagnostic_regressors.pt`
