# 月球车研究中的地形来源：真实月面数据、合成地形与多尺度验证

文档定位（2026-09-25）：本文保留9月19日的候选文献调查，末尾“本轮”指当时调研。
后续数据下载、实现和地图路线已推进，当前统一入口为[真实DEM与NASA模型增强主线](../architecture/lunar_map_mainline.md)。


日期：2026-09-19。性质：定向文献调研与现有地形代码审查；尚未导入月面数据、修改环境或启动训练。

后续更新：真实数据已于2026-09-22完成[六区审计和基准接入](../experiments/lunar_terrain_data_audit_2026-09-22.md)。本文保留为历史调研；涉及固定25米场地、原平坦度标准或默认合成细节的建议，均由[真实数据优先的约束处置](../architecture/real_lunar_terrain_v1.md)替代，不能继续作为数据筛选条件。

配套：[逐项证据表](lunar_terrain_evidence_2026-09-19.csv)。本文关注任务所处地形是否具有月面依据，以及数据尺度能否支持终端区域判据。

## 1. 主要判断

**用户提出的问题成立：现有合成地形缺少与真实月面数据的校准，单凭这些地图上的训练和测试，不能充分支持月面适用性。** 不过，文献并没有要求完全取消程序化生成。实际可借鉴的路线是：以真实 DEM/DTM 约束区域地形，用有统计依据的合成模型补充轨道数据无法解析的小坑与岩石，再用未参与训练的真实区域和近地面数据检验迁移。

DEM/DTM 在本文均指提供高程的地形栅格。光学影像、正射影像和着色地形图不能直接当作高程。尤其需要区分原始观测支持的细节尺度、输出格距和最终仿真网格密度。

本次找到的直接证据包括：

- 两篇相关强化学习路径规划论文采用“合成地图训练＋真实月面 DEM 测试”，其中一篇是多月球车任务。[Hu与Zhang，2022](https://doi.org/10.3390/aerospace9020101)、[Lu等，2024](https://doi.org/10.3390/aerospace11040253)
- NASA 月球车仿真和 OmniLRS 均补充轨道 DEM 中缺失的细尺度地形，并对陨石坑或岩石的生成施加形态、分布依据。[Allan等，2019](https://ulandwong.com/wp-content/uploads/publications/RPSim_paper_draft_2018.pdf)、[OmniLRS论文](https://arxiv.org/html/2309.08997v1)
- 嫦娥四号实际运行采用轨道信息支持粗规划、车载立体视觉重建局部 DEM 支持近距离通行判断。[Wang等，2020](https://doi.org/10.3390/rs12040624)

这些证据支持改进地形与评估设计；不证明换成真实 DEM 就能解决当前学习或闭环执行失败。

## 2. 其他研究具体如何处理地形

| 工作 | 地形与实验做法 | 对本项目的价值与边界 |
| --- | --- | --- |
| **R01 Allan等，2019，IEEE Aerospace：Planetary Rover Simulation for Lunar Exploration Missions** | 在月面 DEM 基础上用分形方法增加细节；按月面尺寸—频数模型生成坑和岩石；考虑喷出物导致的岩石聚集，并避免重复计入已有大坑。 | 最直接的“有月面依据的随机化”参考。服务于驾驶、视觉和任务仿真；不等于每块合成岩石都有真实位置。[作者全文](https://ulandwong.com/wp-content/uploads/publications/RPSim_paper_draft_2018.pdf) |
| **R02 Wang等，2020，Remote Sensing：Vision-Based Decision Support for Rover Path Planning in the Chang’e-4 Mission** | 在沿途站点用立体影像生成局部 DEM，再提取障碍图、搜索路径、将车轮轨迹投回影像检查；局部 DEM 通常覆盖距车约16米、格距0.02米。 | 提供真实任务依据：区域先验与局部精细测量承担不同职责。0.02米是格距，文中明确远处有效分辨率变差；这是地面遥操作决策支持，不能称为自主RL验证。[论文](https://doi.org/10.3390/rs12040624) |
| **R03 Hu与Zhang，2022，Aerospace：Fast Path Planning for Long-Range Planetary Roving Based on a Hierarchical Framework and Deep Reinforcement Learning** | 用模拟坑与高地构造训练集合；在嫦娥二号 CE2TMap2015 的两处20米/像素、10×10公里真实区域上测试。DEM 转为包含地形可通行约束的特征图。 | 可直接借鉴训练／真实地形测试的分工。但研究尺度是长程规划，不能用其20米格距验证本项目亚米尺度平地选择。[论文](https://doi.org/10.3390/aerospace9020101) |
| **R04 Lu等，2024，Aerospace：Lunar Rover Collaborated Path Planning with Artificial Potential Field-Based Heuristic on Deep Reinforcement Learning** | 在随机障碍和任务地图上训练；§4.4在南极附近 Leibnitz beta plateau 的真实 DEM 区域测试双车采集与路径规划，在平地放置加工站。 | 是与多月球车最贴近的先例。该节的真实地图演示规模有限，未交代足够的源产品编号、格距与预处理细节；不能当作完善的跨区域泛化验证，也不等同于分布式共同选址。[论文](https://doi.org/10.3390/aerospace11040253) |
| **R05 Richard等，ICRA 2024：OmniLRS: A Photorealistic Simulator for Lunar Robotics** | 同时支持实验场扫描、真实月面 DEM 和程序化地形；论文§III-A用LRO影像经ASP重建、超分辨率增强，再选配细尺度程序化地形；坑形来自提取、归一化后的实际剖面。 | 与本仓库 Isaac 路线接近，适合借鉴地形生成和导入方法。主要论文验证是岩石分割，不能据此推导多车控制或轮土力学已验证。[论文](https://arxiv.org/html/2309.08997v1) |
| **R06 Daftry等，IEEE Aerospace 2023：LunarNav: Crater-based Localization for Long-range Autonomous Lunar Rover Navigation** | 混合使用月面 DEM 渲染、参数化陨石坑的2.5D定位仿真、Nobile区域的RSIM激光雷达仿真及嫦娥真实图像。 | 展示“按验证问题选地形保真度”：算法单元试验可合成，传感与定位验证另接真实来源。主要目标是定位，不能直接迁移其控制性能结论。[论文](https://arxiv.org/abs/2301.01350) |
| **R07 Chen等，POLAR-Sim，2023初稿／2025修订预印本** | 依据 NASA POLAR 的地球模拟月面场景，用激光点云重建13种场景的地面和岩石网格，并进行感知及车辆仿真。 | 为近地面几何、阴影和传感器误差提供验证素材。属于地球月面模拟场的数字化，不是月球原位实测地形。[核查版本](https://arxiv.org/html/2309.12397v2) |
| **R08 Hemmi等，2025，The Planetary Science Journal：LROC NAC-derived Meter-scale Topography of the Moon’s South Polar Landing Sites: Digital Terrain Models and Their Quality Assessments** | JAXA公布六个 LUPEX 候选区的1米/像素多视图明暗恢复DTM；配套数据包含正射影像、区域边界和高程不确定性图。 | 值得优先审查的真实区域来源。本文核查了官方产品说明与数据清单，未全文审读质量评估论文，也未读取栅格验证误差。[JAXA产品](https://jlpeda.jaxa.jp/en/product/archive/detail_14/index.html)、[作者数据](https://zenodo.org/records/17153447) |

另一个工程先例是 NASA **DLES/DUST**：LRO 地形、星历、合成纹理和代表性坑／岩石分布共同构成仿真场景。NASA介绍的重点区域采用20厘米格距合成地形，南极大范围基础数据为5米格距。前者不能标成“20厘米实测月面”。DUST的用途和获取条件以当前官方软件目录为准，不能沿用旧论文对其开放状态的描述。[NASA介绍](https://www.nasa.gov/general/prototype-immersive-technologies-lab/)、[软件目录](https://software.nasa.gov/software/MSC-27522-1)

OmniLRS也要区分版本：原论文描述的程序化增强可到4厘米/像素；当前仓库的 LargeScale 概述为通常5米/像素真实 DEM 加增强到2.5厘米/像素。这是软件发展后的配置描述，不能全部归给2024论文。它仍保留小尺度纯程序化环境。[官方仓库](https://github.com/OmniLRS/OmniLRS)、[环境说明](https://github.com/OmniLRS/OmniLRS/wiki/Environments)

## 3. 可以取得哪些月面数据

| 数据 | 已核实的尺度与内容 | 本项目适用性 |
| --- | --- | --- |
| **LROC NAC 立体 DTM** | PDS ODE说明通常为2–5米/像素，局部覆盖；每个产品应同时查看README、质量与影像资料。 | 用于真实起伏、坑缘及区域拓扑；不直接解析车轮尺度岩石。[PDS ODE](https://ode.rsl.wustl.edu/moon/pagehelp/Content/Missions_Instruments/LRO/LROC/SDR/Intro.htm) |
| **NASA PGDA 南极 LOLA 高分辨率产品** | 5米/像素GeoTIFF，提供坡度、点数、误差图等；该产品说明指出约90%的5米像元需要插值，不能把格距当独立测量间距。 | 适合作为南极宏观地形和不确定性来源；激光测高不受光学阴影同样的限制，但仍有采样空隙。[官方产品](https://pgda.gsfc.nasa.gov/products/78) |
| **JAXA LUPEX 六区域 SfS DTM** | 1米/像素；作者数据包含32位高程、正射影像和高程不确定性GeoTIFF，压缩包约76.1MB。 | 相比公里尺度全局数据，更值得先用于本项目25米局部窗口的尺度审查；仍不足以提供亚米地形真值。[JAXA](https://jlpeda.jaxa.jp/en/product/archive/detail_14/index.html)、[数据与许可说明](https://zenodo.org/records/17153447) |
| **CE2TMap2015** | R03明确使用20米/像素产品进行公里尺度路径规划。 | 可做大区域选址与规划参考，不适合直接作为当前25米场地的精细地面。[R03 §6.3](https://doi.org/10.3390/aerospace9020101) |
| **车载／着陆器近景重建、模拟场点云** | R02提供厘米格距局部重建实例；POLAR-Sim提供模拟场景网格。 | 对局部障碍与停驻足迹更有价值，但需要检查覆盖范围、误差、可获取性和其是否真正在月面采集。 |

优先考虑哪一片区域，应由任务地点决定。若尚未限定月球南极，不能用南极一类地形代表整个月球；月海、高地、坑缘与喷出物区域应分别校准与报告。

## 4. 当前仓库的具体缺口

本轮读取了[地形实现](../../source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/terrain_features.py)、[配置定义](../../source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/gathering_env_cfg.py)，并递归合并了[exp167配置](../../configs/experiment/exp167_local_goal_nmpc.yaml)的继承链。以下是静态代码事实；运行脚本对场景的额外覆盖需另行核查。

1. **目前是解析合成地形族。** `_heightfield_height` 用正弦项构造起伏，按配置叠加碗状坑、坑缘和带通道的山脊；`_crater_layout`用解析规则安排坑位与半径。随机化主要作用于相位、平移、旋转和尺度。检查到的主地形查询链没有加载实测 DEM。
2. **当前exp167基础场景并非每回合随机地图。** 合并配置为 `topology_profile: open`、`crater_count: 0`、`randomize_per_reset: false`，保留振幅0.09米、波长2.8米的解析起伏。实验仍可另外设置复杂测试场景；不能用这条基础配置概括全部历史试验。
3. **地形指标的物理意义需要改进。** `_base_features`里的`roughness`是高程梯度模长乘系数，与坡度没有独立信息；`traversability`又由该量计算。它们尚未区分光滑斜坡、去趋势后的粗糙度、台阶／岩石与松软土。更换地图文件不会自动修复这一点。
4. **当前信息来源过于理想。** 局部地形栅格从环境高程函数直接查询。真实数据作为仿真地面后，还应单独定义车辆知道哪些数据、局部重建误差和遮挡；“仿真器有全图”不代表Actor可以得到全图。
5. **任务与数据存在明显尺度差。** exp167继承的边界为±12.5米，即25×25米；平坦度检查半径0.75米、高差阈值0.18米、坡度阈值0.25。这里坡度来自高程梯度，不应读成0.25度。5米DEM在场地单边只有约5个格距，直径1.5米的检查圆盘甚至小于一个格距；1米DEM也不足以可靠表达该圆盘内的细节。

最后一点不应处理成“插值到5厘米就够了”。加密输出网格不会创造独立观测；超分辨率或程序化补全得到的细节属于推断或合成。也不能直接把DEM绝对高程误差与0.18米局部高差阈值比较后宣布不可用：共同高程偏移可能抵消，真正需要检查的是对应基线上的相对高程与坡度误差及空间相关性。

由上述代码与文献作出的研究判断是：**当前平地密度、可达域、通道宽度和终端区域大小，都可能受生成器形式主导。** 这种偏差的大小尚未测量；不能把它写成已被证明的训练失败原因。

## 5. 建议的地形与验证路线

以下是本项目研究建议，不是已经执行的改造，也不是任何单篇论文给出的统一标准。

### 5.1 保留三类地形，分别回答问题

| 地形层 | 主要用途 | 可报告的结论 |
| --- | --- | --- |
| 可控解析地形 | 隔离平地稀缺、窄通道、观测不足和会车等因素 | 在明确构造条件下的因果诊断 |
| 真实DEM支撑的混合地形 | 保留真实区域低频地形，增加经校准的未解析坑、岩石与微起伏 | 在真实来源宏观地形与指定微地形模型下的性能 |
| 独立区域及近景／模拟场验证 | 未见区域、替代生成器、局部几何与传感器误差测试 | 跨区域、跨生成器或对应模拟场的迁移能力 |

其中第二层最适合作为候选训练分布，但必须同时评估“原DEM／仅插值DEM”和“加细节DEM”，才能看出结论究竟由真实底图还是合成细节主导。岩石应具有与视觉一致的碰撞几何；只加贴图不能验证避障。

### 5.2 先做数据审计，再决定训练分布

建议第一步是对JAXA六个候选区与一组LROC／LOLA产品做小规模离线审计：

1. 记录产品编号、地点、投影、米制坐标、高程基准、原生格距、有效细节尺度、无效像元和误差图；保持真实水平与垂直比例，不把公里区域压缩成25米。
2. 在多个真实区域裁剪25米及更大上下文窗口。比较坡度分布、不同尺度去趋势粗糙度、可通行连通域、瓶颈宽度、候选停驻区域面积和各车可达性。导数与平坦度计算应使用明确的物理基线，不能只沿用解析函数的0.05米差分步长。
3. 检查0.75米检查半径和0.18米阈值需要什么观测支持。低于可解析尺度的判定应标为未知、依赖补全模型或需要局部传感验证，不能由平滑插值自动认证合格。
4. 从真实地貌类别与文献统计校准未解析部分；已有坑不重复添加，小坑与岩石的尺寸分布、密度、聚集程度及与坑缘／喷出物的关系均应记录。宏观底图与微地形分别设种子。

建议为每块地形保存最小来源记录：`source_product_id`、区域边界、原生／仿真格距、坐标变换、质量掩码、误差来源、已实测与合成内容、生成参数、种子、处理版本和数据划分。此记录是后续可复现设计建议，本轮未新增运行产物规范。

### 5.3 保持任务的信息条件

如果仍研究“仅局部地形观测＋有限通信”，真实全图可供仿真真值与离线诊断使用，Actor继续只接收视野内信息。若另研究“低分辨率轨道先验＋局部高分辨率感知”，应作为单独信息条件，并给所有对照方法同样的先验。

来自轨道DEM的先验、地面真实几何、带误差的局部观测应分开保存。月面光照、土壤沉陷与滑移也不是高程文件自带的属性；只有研究目标涉及这些机制时，才进一步引入相应模型和验证数据。

### 5.4 建立不会泄漏区域信息的比较

至少比较：现有合成地形训练、月面统计校准的合成训练、真实底图混合训练。固定算法、信息条件、预算和车辆模型，在相同测试场景上配对评估。

按地理区域／源产品分组划分训练、验证、测试；避免同一地点的重叠裁剪、不同分辨率版本或仅旋转后的副本跨集合。另做留出微地形生成方法的测试，防止只学到某种合成器。按地形类别分别报告完成率、碰撞、超时、候选区域的实际可用性与到达后稳定性。

场景采样时同时报告任务可行性和求解器能力：已找到可行联合轨迹是可行见证；规划失败不自动等于场景无解。不要为了提升通过率持续修改测试地形，也不要把仅保留可行场景后的结果称为无条件月面任务成功率。

## 6. 对近期项目路线的建议

优先级应是：**地形数据与尺度审计 → 冻结真实区域测试集 → 校准合成细节 → 接入现有地形查询接口 → 再比较训练效果。** 先完成小范围数据处理和离线判据检查，比立即迁移整个仿真平台更能判断数据是否适合当前任务。

`query_height`／`query_terrain_features`一类统一查询入口是自然的后续接入点，但需要同时考虑坡度与粗糙度语义、局部观测误差以及PhysX碰撞地面的一致性。本轮不把接入设想记为已实现，也不更改exp167执行状态。

## 7. 检索与证据边界

这是围绕本项目问题的定向调研，不是对所有月球车论文的系统综述。检索覆盖月球车导航、协同规划、强化学习、月面仿真与官方DTM数据；使用的代表性检索词包括：

- `lunar rover navigation simulation real lunar DEM procedural terrain reinforcement learning`
- `OmniLRS lunar robotics simulator DEM terrain generation`
- `Lunar Rover Collaborated Path Planning Leibnitz`
- `Planetary Rover Simulation for Lunar Exploration Missions`
- `Vision-Based Decision Support Chang'e-4 DEM`
- `LROC NAC DTM resolution`、`LUPEX DTM uncertainty`

R01–R07核对了相关方法／实验章节，其中R01、R02、R03、R04、R06通过作者或出版社PDF核查，R05、R07通过arXiv全文核查。R08仅核查JAXA官方产品说明和作者Zenodo数据清单。DUST、OmniLRS现行功能和数据规格依据各自官方页面。部分出版社网页限流，已通过其PDF补读；未从二手综述反推核心论文的方法细节。

另发现两条近期线索，尚不作为核心技术依据：

- **Safety-Aware Reinforcement Learning for Lunar Rover Navigation Using Orbital DEMs**，IEEE Access，2026，DOI `10.1109/ACCESS.2026.3717682`。出版社索引摘要描述了LRO地形训练和留出DEM块评估；本轮未取得全文，具体预处理、区域划分独立性和地形真实性仍待核查。[出版社](https://ieeexplore.ieee.org/abstract/document/11630678/)
- **HIA-APF: A hierarchical based path planning method for lunar rovers on a partially observable surface**，2026。出版社摘要描述低分辨率DEM与有限局部高分辨率观测的结合，并在真实DEM区域测试；本轮只核查摘要。[出版社](https://www.sciencedirect.com/science/article/pii/S0921889026000618)

本轮没有下载并分析实际月面栅格，没有复现上述算法，没有得到真实地形上的本项目成功率；对训练路线的建议均需后续数据审计与配对实验验证。
