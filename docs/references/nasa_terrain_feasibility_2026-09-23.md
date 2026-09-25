# NASA地形路线可行性核查与服务器实测

文档定位（2026-09-25）：本文是2026-09-23选型与可行性证据；下载范围、驱动状态和接入能力按当时记录保留。
当前地图路线已确认，见[真实DEM与NASA模型增强主线](../architecture/lunar_map_mainline.md)。


日期：2026-09-23。结论对应当前工作站与固定上游版本，不代表完整训练或物理仿真已通过。

## 结论

真实月面DEM加有依据的合成细节，可用于本项目训练地图。当前RTX 2060能够执行GPU批量地形查询与已有策略网络；不能因为项目目前安装CPU版PyTorch，就把后续方案限定为CPU。推荐方案见[详细实施规划](../architecture/nasa_terrain_training_implementation_plan.md)，目前待审核、未实施。

LuNaSynth可以提供方法与代码参考，但所核查版本不适合未经修正直接生成训练真值。2019年NASA Ames论文、NASA JPL LuNaSynth和NASA公开LOLA产品是三项不同资源；未找到并验证2019年论文完整场景包，也未证明其具体底图就是本次LOLA样本。

## 已执行与未执行

已完成硬件与依赖检查、上游代码阅读、小型GeoTIFF下载、CPU栅格资源测试、两个上游数值问题复现，以及CUDA地形查询/现有策略网络前向测试。

未安装或升级任何依赖，未修改生产地形或训练代码，未启动训练，未运行完整Blender/LuNaSynth、Isaac Sim或新的NMPC闭环。CPU测试使用重采样仅用于资源测量，并不意味着已确定地图处理算法。

产物目录：`outputs/runs/lunar_terrain_validation/nasa_feasibility_20260923/`。

- `metrics/hardware.json`：硬件快照。
- `metrics/cpu_probe.json`、`metrics/gpu_probe.json`：数值结果与边界。
- `probe_cpu.py`、`probe_gpu.py`：可复现探针；没有优化器更新。
- `evidence/download_manifest.json`：源码/样本来源与SHA256，包含未成功下载项。
- `run_manifest.json`：验证范围及运行环境。

## 服务器与软件

| 项目 | 实测 | 影响 |
| --- | --- | --- |
| 系统 | Ubuntu 24.04.3，Linux 6.17 | 现有GIS与CUDA探针可执行 |
| CPU | i7-10750H，6核12线程 | 支持分块预处理；CPU NMPC并行规模需测量 |
| 内存 | 约16 GB，初查可用约9.8 GiB | 适合按块生成和缓存，避免全区域厘米级实体化 |
| GPU | RTX 2060，6144 MiB显存，驱动580.126.09 | 已通过CUDA数值与网络前向测试 |
| 存储 | 本仓库所在NVMe剩余约360 GiB | 足够样本与初期地图库；不等于适合保存全区域所有增强版本 |
| 项目环境 | Python3.12.3，PyTorch2.10.0+cpu | 当前项目CUDA不可用是安装状态，不是硬件不支持 |
| GIS | rasterio1.4.4、SciPy1.18.1、NumPy2.3.1、pyproj3.7.2 | 已用于CPU探针 |
| LuNaSynth依赖 | 上游要求Python3.11.10、bpy ^4.1.0、NumPy ^1.26.4 | 与项目环境隔离；不能直接混装 |
| 本机缺少 | bpy、Blender可执行文件、git-lfs、gdalinfo；Isaac Sim/Lab未安装 | rasterio可读GeoTIFF，不依赖gdalinfo命令；完整渲染链仍未验证 |

CUDA探针借用已有`/home/yu/anaconda3/envs/vad/bin/python`（Python3.8.20、PyTorch2.4.1+cu121）。只读取库并执行探针，没有修改该环境。该版本包含`sm_75`，与设备匹配；不作为本项目将来依赖版本。

[PyTorch官方安装表](https://pytorch.org/get-started/previous-versions/#v2100)提供2.10.0的cu126/cu128等版本，后续应在独立Python3.12环境核查架构、安装和运算。此次尚未验证目标环境的CUDA wheel。

当前[Isaac Sim官方要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)列出最低32 GB RAM、RTX 4080/16 GB VRAM；本机低于该要求。因此，本机GPU张量训练与完整Isaac/RTX物理渲染应分别判断。不能以Isaac不足推导CUDA训练不可行，也不能以CUDA探针通过承诺Isaac可运行。

## 数据可获得性

固定LuNaSynth提交：`8618e3d15ed769f78817b5527549d563a1c2291a`。

已从[NASA JPL官方仓库](https://github.com/nasa-jpl/lunasynth/tree/8618e3d15ed769f78817b5527549d563a1c2291a)下载`assets/base_meshes/NPB_final_adj_5mpp_surf_piece_1km_2_12.tif`：

- 160,723字节，SHA256 `4c8627c364317439ee5365f1fb0b2b8fe819a9dc47e974febb7e019f7e8fc45f`。
- 200×200格点、5米格距、覆盖1000×1000米、float32。
- 月球半径1,737,400米的南极立体投影，平面单位米；CRS名称为unknown，但投影参数可读。
- 投影范围x=177000至178000米、y=-1500至-500米；高程约-3679.648至-3628.413米。
- 样本无声明nodata且全部数值有限；这不证明全部高程都是直接观测，也不证明质量均匀。

[PGDA官方数据入口](https://pgda.gsfc.nasa.gov/products/78)公开NPB及其他地点的完整DEM、采样数量图、误差图等。本次没有下载完整NPB母图并逐像元核对仓库裁剪，不把文件名当作完整来源证明。正式地图库应从PGDA原产品建立来源，再验证裁剪关系。

LOLA 5米产品已包含官方插值，官方说明插值像元比例约90%。已有LUPEX 1米产品可以继续使用；不同重建方法、控制点和误差语义须分别记录，不能按格距简单排名。

## LuNaSynth代码核查

主要文件：[terrain_enhancement.py](https://github.com/nasa-jpl/lunasynth/blob/8618e3d15ed769f78817b5527549d563a1c2291a/src/lunasynth/terrain_enhancement.py)、[dem_tools.py](https://github.com/nasa-jpl/lunasynth/blob/8618e3d15ed769f78817b5527549d563a1c2291a/src/lunasynth/dem_tools.py)、[blender_helper.py](https://github.com/nasa-jpl/lunasynth/blob/8618e3d15ed769f78817b5527549d563a1c2291a/src/lunasynth/blender_helper.py)。代码采用Apache-2.0；具体外部岩石资产仍需逐项追踪来源。

| 发现 | 证据等级 | 实际影响 |
| --- | --- | --- |
| `Terrain.generate`将输入格距写死为5米 | 源码第1024行 | 直接输入LUPEX 1米数据会产生错误倍率 |
| `DEM.zoom`注释称双线性，实际调用SciPy默认三阶样条 | 原类隔离执行；与order=3完全一致，与order=1最大差0.606934米 | 复现时必须显式记录真正的插值核，不能照搬注释 |
| zoom后`meta`格距更新，但`transform/width`属性仍旧 | 原类隔离执行；5米→2.5米时属性仍5米，width仍200、数组已400 | 元数据与数组不能作为一致接口直接复用 |
| 陨石坑位置和直径为米，写入函数直接用作像元索引且没有格距参数 | `place_crater_dems_numba`静态审查 | 格距不是1米时位置与物理尺寸可能错误；需构造非方形、不同格距测试 |
| `Crater.crater_depth(np.float64(0))`返回NaN | 原类隔离执行，无Blender/Numba | 正好命中坑中心时需验证有限极限与公式含义；不能用随意epsilon掩盖模型问题 |
| `prange`按坑并行、向重叠DEM切片累加 | 静态审查，未运行Numba并行版 | 存在写冲突风险，需与确定性串行参考比较 |
| 网格接口仅一个`mesh_size`同时用于x/y | 静态审查 | 非方形区域、像元中心/边缘关系需要显式验证 |
| 插值器在添加坑之前构造 | 静态审查 | 要验证数组共享/复制语义，不能仅凭调用顺序断言岩石一定悬空 |
| 上层生成器主要写CSV、图像及可选`.blend` | `configuration_manager.py`生成链审查 | 尚非带质量/地理元数据的训练heightfield或碰撞体导出接口 |

本次执行的是从原文件AST提取的原类定义，以隔离Blender依赖；没有执行完整上游模块。这足以复现上表标记“执行”的局部问题，不等于完整上游测试。

2019年NASA论文的分形/频率分离方法，与LuNaSynth的重采样加坑石实现不能画等号。建议保留论文作为方法依据，复用经过验证的局部代码；不整体导入上游默认参数。

## CPU栅格资源测试

从上述NASA仓库样本中央取资源测试窗口。只减去一个高程原点以改善数值精度，不缩放真实地形。下列尺寸仅是工程测试点，不是任务范围或科学分辨率的选择。

| 窗口 | 处理格距 | 高程+两向梯度容量 | 重采样耗时 | 梯度及打包耗时 |
| --- | --- | --- | --- | --- |
| 100×100米 | 0.1米 | 11.44 MiB | 0.059秒 | 0.014秒 |
| 200×200米 | 0.1米 | 45.78 MiB | 0.221秒 | 0.034秒 |
| 200×200米 | 0.05米 | 183.11 MiB | 0.873秒 | 0.310秒 |
| 400×400米 | 0.1米 | 183.11 MiB | 0.870秒 | 0.260秒 |

单进程逐项测试累计峰值RSS约657 MiB。该表不包含坑石生成、Blender网格、误差图、粗糙度、网络训练；插值后梯度是数值表面导数，不是新增的月面观测。时间为单次工程探针，不作长期吞吐保证。

## GPU实测

四份2000×2000、3通道float32地图占约183.11 MiB，实际内容重复，仅验证共享地图缓存机制；不构成四个独立地点。每个等效环境包含4车，每车112个查询点，坐标已在设备上。

网络直接加载项目`local_goal/learning.py`的`RecurrentPolicy`，随机初始化，410维观测、128维隐状态；critic_dim=54仅用于构造，本次未执行critic。CPU/GPU使用同一权重，固定4个CPU数值线程。预热后查询重复20次、Actor前向重复10次，每次GPU显式同步。

| 等效环境数 | 查询点数 | CPU查询中位ms | GPU查询中位ms | CPU Actor前向中位ms | GPU Actor前向中位ms | CUDA峰值allocated/reserved MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 14336 | 0.319 | 0.089 | 2.479 | 1.693 | 196.6 / 208 |
| 128 | 57344 | 1.131 | 0.297 | 7.552 | 1.691 | 207.1 / 232 |
| 512 | 229376 | 10.438 | 1.064 | 25.733 | 4.092 | 249.6 / 272 |

查询结果CPU/GPU最大绝对差为0；Actor动作最大差约1.05e-9。这些是样本上的结果，不作为未来所有输入的误差界。

显存数值仅为PyTorch跟踪分配，不含驱动上下文等全部占用。尚未计入数据上传、无效值分支、轨迹查询、环境状态更新、通信、rollout、反向传播、优化器及NMPC；不能声称“512个完整训练环境已跑通”，更不能把局部加速比当作训练加速比。

## 生产接入边界

- `terrain_data/query.py`已有分块原生栅格读取与逐点尺度指标，但`RasterTerrain`强依赖LUPEX敏感性图、标量查询与其语义；需泛化数据源并另建批量后端。
- `terrain_features.py`、`gathering_env.py`及观测、奖励、终止判据大量使用旧5通道语义与程序化`TerrainRuntime`；真实地图不能只替换一个高度函数。
- `local_goal/environment.py`当前强制CPU、使用NumPy和4线程求解器池。Actor可运行CUDA，不代表整个NMPC执行链自动上GPU。
- `train_local_goal_nmpc.py`也存在CPU张量创建、`.numpy()`、状态hash和RNG处理约定；混合设备需要完整边界设计。
- GPU批量proxy和GPU策略训练可以推进；exp167保留CPU求解器的混合模式应独立验证。不能为了展示GPU加速而同时偷偷更换控制器。

下一步仅按[待审核规划](../architecture/nasa_terrain_training_implementation_plan.md)推进，训练仍暂停。
