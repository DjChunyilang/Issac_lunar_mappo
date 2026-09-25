# LUPEX六区域真实地形审计与基准接入

文档定位（2026-09-25）：本文保留LUPEX审计批次及其六区四组基准，后续作为补充来源和跨源评估候选。
当前地图路线已确认，见[真实DEM与NASA模型增强主线](../architecture/lunar_map_mainline.md)。


当前优先级更正（2026-09-23）：用户要求调研他人的地形数据来源与处理方法，插值仅为举例。以[方法核查](../references/lunar_terrain_methods_review_2026-09-23.md)为准；本页引用的地图生成草案已撤回实施安排，不能据此预选插值或直接开始生成。既有数据与验证事实保留。

日期：2026-09-22。实验族：`lunar_terrain_validation`。**已完成数据审计、来源关联的区域划分、原生数据查询及接口诊断；没有启动学习训练。** 实施依据为真实数据优先的修订规划，旧25米场地和平坦度阈值不参与本轮分析。

2026-09-23方向更新：本记录保留为已完成的审计事实，后续按[真实数据生成训练地图计划](../architecture/real_terrain_training_maps_v1.md)推进。相关长度未知不妨碍在训练区裁剪，原始分辨率有限不妨碍插值构建仿真地面；需保留地理分组和处理来源。本文的数据/车辆缺口不再作为地图生成的前置门槛。

## 权威产物与复现

最终run：`outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/`。

- [审计汇总JSON](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/metrics/summary.json)：完整产品与ROI统计、配准、尺度、相关性与质量标记。
- [冻结划分](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/data/benchmark.json)、[场景引用](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/data/windows.json)、[接口检查](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/metrics/integration_smoke.json)。
- [运行手册](../runbooks/lunar_terrain.md)、[接口与任务适配/旧约束处置](../architecture/real_lunar_terrain_v1.md)、[合成校准和后续比较协议](lunar_terrain_comparison_protocol.md)。

原始包76,134,049字节，MD5=`4ad67232e49b502897ecff197336da29`，SHA256=`4a2cbb1d9f6ed4a1abd804f9faeee45c6c1f847030b31ba7fb986f280e8ecd77`。原包未改动；工作副本只读，全部文件记录校验和。来源为[Zenodo Data S1](https://zenodo.org/records/17153447)，方法依据为[Hemmi等，2025](https://doi.org/10.3847/PSJ/ae10a4)，全文与坡度脚本从[作者仓库](https://github.com/ryodohemmi/public/tree/main/Hemmi%2B2025)读取并保存在ignored证据目录。

## 数据与质量清单

18份栅格均为32位浮点、1米格距，NoData为float32最小值。参考系为月球南极投影米制坐标，参考球半径1,737,400米；波段未写垂直单位标签，高程/高度敏感性的米单位依据论文确认。ROI shapefile与栅格使用等效但不同写法的投影定义，因此执行了显式坐标转换；六区ROI边界转换前后差异至多约`9.1e-13 m`。

下表统计完整DTM的全部有效像元，不限于原白色ROI。有效面积总计约11.282平方千米；原ROI内有效面积另存JSON。

| 区域 | 原生行×列 | 有效覆盖 | 有效面积km² | 全域高差m | 16米基线坡度中位数 / p95（度） | 同尺度去趋势RMS中位数m |
| --- | --- | --- | --- | --- | --- | --- |
| CR1 | 1706×2006 | 90.21% | 3.087 | 215.0 | 10.77 / 17.07 | 0.080 |
| GR1 | 1405×1805 | 88.01% | 2.232 | 243.5 | 11.98 / 20.05 | 0.086 |
| GR2 | 1608×911 | 77.45% | 1.134 | 74.7 | 8.03 / 13.53 | 0.067 |
| LP1 | 2015×928 | 87.55% | 1.637 | 131.9 | 7.10 / 12.05 | 0.065 |
| MP1 | 1229×2020 | 89.02% | 2.210 | 205.4 | 11.74 / 16.60 | 0.091 |
| MP2 | 1520×933 | 69.18% | 0.981 | 109.5 | 9.69 / 19.11 | 0.085 |

坡度是窗口拟合平面的倾角，RMS是同一窗口去趋势残差。它们描述具备完整数据支持的区域，不是车辆可通行率。数据有效域的最大连通分量占有效像元99.58%以上，但这仅表示数据覆盖连通，不能推得车辆可达。

正射影像均与DTM存在亚像元原点或形状差异，禁止按数组同下标混用。CR1–MP1敏感性图与DTM同网格；MP2敏感性图为1521×934，比DTM多出西侧一列、南侧一行，已按坐标匹配。存在“DTM有效、敏感性缺测”的像元：CR1 15,482、GR1 497、GR2 11,365、LP1 35,385、MP1 1,803、MP2 495，缺失误差信息没有被填为零。

局部经纬往返变换检查的最大误差约`8.2e-9 m`。区域中心投影尺度因子约1.000022–1.001599；当前局部距离和基线均使用投影米，后续任务距离不能误称为精确月面测地长度。

## 不确定性、伪影与可支持尺度

论文第4.1节解释：固定相邻像元，仅扰动目标像元高度，在0–5米范围以0.01米间隔探测，使亮度失配加倍的高度偏移作为敏感性量。这近似反映不确定性的下界，不是后验标准差；遗漏空间相关、反照率、光度模型、配准、初始DEM与相机误差。适合产品内部相对诊断，不能直接作为独立高斯噪声或安全概率。[作者全文](https://github.com/ryodohemmi/public/blob/main/Hemmi%2B2025/Hemmi%2B2025.pdf)

此次原生敏感性产品有效像元计数与论文表5六项完全一致，均值约0.032–0.049米。这个数值不能替代外部绝对高程精度。论文第4.2节显示约10米以下存在平滑，MP2在1–4米尤为明显；第4.4节讨论MP2局部数米失配及控制源等可能原因，不能由较小敏感性值排除这些问题。[论文](https://doi.org/10.3847/PSJ/ae10a4)

审计从原生格距产生2、4、8、16、32、64、128米基线；较宽区域另有256米。上限要求区域最短边至少容纳4个完整分析窗口，是统计探索的工程规则，并非有效分辨率定义。方形窗口全部有效才计算；无效值和边界从指标掩码中移除，不补洞。

| 分析对象 | 尺度与有效支持 | 误差依据 | 当前结论 |
| --- | --- | --- | --- |
| 高程起伏、区域位置、覆盖 | 全原生区域，1米采样 | SfS重建与参考系；绝对误差不等同于敏感性 | 支持区域地貌描述，保留原ROI及缺测分层 |
| 极短基线坡度/粗糙度 | 2/4/8米 | 论文提示平滑；有效分辨率未知 | 作为数据/数值诊断，不认证亚米地形与车轮接触 |
| 区域坡度/去趋势粗糙度候选 | 16/32米 | 约10米细节的定性证据；严格完整支持掩码 | 可继续研究区域形态，需披露缺测条件化偏差 |
| 大窗口粗糙度 | 64–256米 | 孔洞导致完整支持急剧减少 | 部分区域统计过少或为零，不能当作稳定全域分布 |
| 空间相关与双方向坡度曲线 | 1米起的二倍距离，至各方向半幅范围 | 相关性用去全域平面残差；坡度用原高程有效点对 | 只给描述相关，不能据其证明裁剪独立 |
| 缺测邻边、高敏感性、突变 | 原生像元与相邻差分 | 本产品p95敏感性/p99.9相邻高差仅作候选标记 | 没有自动剔除陡峭地形；已验证伪影掩码仍未知 |
| 车辆足迹、岩石、摩擦、停驻安全 | 当前不具备可靠支持 | 缺近地面测量和车辆/作业规范 | 未知，不自动写入生产判据 |

**缺测偏差不可忽略**：16米基线完整窗口中心占有效DTM像元的比例依次为52.28%、52.67%、37.13%、39.45%、54.79%、31.38%；32米下降到约9.01%–31.05%。128米只有CR1 1,521、GR1 5,015、MP1 28,105、MP2 48个中心；GR2/LP1为零。不能把这些条件分布推广成整片地形的无偏统计。所有空统计保留`count=0`，不替换成“粗糙度为零”。

## 地理隔离结果

DTM覆盖范围互不重叠，最小边界距离为7,479米。论文表1的NAC影像ID在六区间不重复；但表2表明GR1/GR2共享TC控制产品，MP1/MP2共享四份TC控制产品。因此按关联分量得到：

```text
g0 = CR1
g1 = GR1 + GR2
g2 = LP1
g3 = MP1 + MP2
```

采用4个外层留组测试，每个外层分别用其他3组之一验证，余下2组开发，共12个组合。产品和处理链共性仍存在，结论只能称为该产品族内部的区域隔离评估，不能宣称跨仪器或月球总体泛化。

六区至少一个方向的去趋势相关性在可分析范围内尚未跌破`1/e`：CR1/GR1/MP1东向至512米，LP1/MP2东向至256米，GR2北向至512米仍未跨越。因相关长度上界未知，本轮**没有制造固定25米窗口或宣称相邻裁剪独立**。每折的窗口设计只查看训练区相关统计；全部触发完整区域保留策略。`windows.json`中的72项是6区×12折的引用，真实区域仍为6、保守关联组仍为4。

冻结划分SHA256：`153095788a7219172c8fb841423e960e0ac209cd16c82c68a6d2e400869cc2e2`。同地点的未来裁剪/重采样/增强必须继承原分组，变化建立新基准版本。当前未按坡度、平地区域数量或算法效果排除任何区域。

## 图集

每区提供全产品高程、覆盖/ROI、原生2米基线指标、敏感性与质量标记，以及独立地理范围的正射影像。色标p01–p99仅用于显示，统计没有裁掉尾部，所有缺测继续留空。

| 区域 | 原始覆盖/地形/敏感性 | 正射影像 |
| --- | --- | --- |
| CR1 | [图集](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/figures/cr1_atlas.png) | [影像](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/figures/cr1_ortho.png) |
| GR1 | [图集](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/figures/gr1_atlas.png) | [影像](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/figures/gr1_ortho.png) |
| GR2 | [图集](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/figures/gr2_atlas.png) | [影像](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/figures/gr2_ortho.png) |
| LP1 | [图集](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/figures/lp1_atlas.png) | [影像](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/figures/lp1_ortho.png) |
| MP1 | [图集](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/figures/mp1_atlas.png) | [影像](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/figures/mp1_ortho.png) |
| MP2 | [图集](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/figures/mp2_atlas.png) | [影像](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/figures/mp2_ortho.png) |

[双方向坡度—基线图](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/figures/slope_baselines.png)、[区域/来源关联图](../../outputs/runs/lunar_terrain_validation/audit_lupex_v1_20260922_final/figures/region_relationships.png)。图不构成安全或任务成功证明。

## 接入、验证与边界

新模块实现原生分块LRU、可逆坐标、源哈希核验、显式尺度的坡度/RMS、严格无效掩码、敏感性语义、原生patch导出和特征版本。旧接口继续服务历史合成任务，明确拒绝把真实栅格当成旧语义。三个数据层已登记；本轮没有生成合成微地形，没有向Actor开放全图。

- 13项新地形合同通过：解析平面/粗糙表面、尺度依赖、无效值/边界、双线性缺洞、错位误差图、坐标往返、有限缓存、分组防泄漏、训练区尺度拟合、权限和旧语义拒绝。
- 加上旧地形回归，共27项通过；依赖`pip check`通过。
- 六区真实场景均完成加载、有限局部查询、坐标往返、4步点运动和默认轨道先验拒绝。初始点按数据支持选取仅用于接口smoke，不参与基准筛选。
- 全仓收集786项：774通过、6跳过、6失败。失败均为旧exp156 timeout诊断配置继承的`far_bottleneck.json`/`near_open.json`在本机缺失；代码路径在本轮未改。跳过为CUDA及可选acados环境。未伪造历史文件或把全仓回归写成通过。
- Isaac/PhysX栈不在当前环境，本轮未开展物理车辆验证；点运动不表示exp167控制器已适配真实地图。

早期`audit_lupex_v1_20260922`因ROI与栅格投影定义字符串不同停止，未形成基准；修正为显式投影转换后完成`r2`，最终run另补充源统计、投影尺度、证据清单和显示处理。早期产物保留，当前事实以`*_final`为准。

## 缺口与补充来源

| 缺口 | 优先调查的真实来源 | 边界 |
| --- | --- | --- |
| 绝对高程与低频形态的独立交叉检查 | [NASA PGDA南极LOLA产品](https://pgda.gsfc.nasa.gov/products/78)：提供点云、点数、5米插值DEM及误差产品 | 可帮助检查控制与缺测，不提供亚米接触地形；部分源已用于SfS控制，需核查独立性 |
| 毫米至分米粗糙度/近地面重建 | [Guo等2021嫦娥四号研究](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL094931)使用PCAM立体影像并给出数据入口；[国家空间科学数据中心嫦娥四号入口](https://www.nssdc.ac.cn/nssdc_en/html/task/change4.html) | 优先核查可下载的具体产品、相机标定和重建误差；不是同一南极地点，不能直接贴到LUPEX图上当作真实细节 |
| 厘米级地形及石块/坑统计 | [嫦娥四号着陆区厘米模型研究](https://www.sciencedirect.com/science/article/pii/S0012821X20306105) | 论文证明存在相关方法；本轮未取得可直接使用的完整DEM，不声称已建接触尺度数据集 |
| 各像元独立覆盖/照明/伪影真值 | 当前Data S1未提供可直接使用的完整源影像重叠数和人工伪影分类 | 需追加原始影像/重建中间产物，当前标为未知 |
| 车辆尺寸、动力学、作业安全与通信 | 车辆技术要求和任务定义 | 不能由轨道地形统计推导 |

因此当前成果足以支撑真实区域地形描述及后续粗尺度规划研究；车辆安全、任务距离/时限和训练恢复条件仍须按[任务适配提案](../architecture/real_lunar_terrain_v1.md)单独论证。合成校准与公平比较已形成[协议](lunar_terrain_comparison_protocol.md)，尚未开始拟合或训练。
