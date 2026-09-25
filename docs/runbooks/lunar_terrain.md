# 真实月面地形审计与查询

本页保留LUPEX数据审计、原生查询与区域分组工具。2026-09-25地图主线已确定为
[真实DEM与NASA增强](../architecture/lunar_map_mainline.md)，当前地图生成和验收命令
见[NASA SFD手册](nasa_sfd_v2.md)。本页不再将继续文献调研或旧撤回草案作为地图生成前置。

LUPEX历史事实见[审计记录](../experiments/lunar_terrain_data_audit_2026-09-22.md)，
原生接口见[说明](../architecture/real_lunar_terrain_v1.md)。以下命令不启动学习训练。

## 环境与原始文件

在仓库根目录执行，沿用Python 3.12 proxy环境，额外安装GIS依赖：

```bash
.venv_isaaclab/bin/python -m pip install -r requirements-terrain.txt
.venv_isaaclab/bin/python -m pip check
```

原始包已下载完成，无需重复下载：

```text
outputs/runs/lunar_terrain_validation/_suite/data/lupex_17153447_v1/DataS1.zip
```

MD5：`4ad67232e49b502897ecff197336da29`；字节数76,134,049。入口为[Zenodo记录](https://zenodo.org/records/17153447)。解压工作副本在同级`products/`，只读保存。脚本先验证MD5；存在的解压文件若校验不匹配就停止，不覆盖它。

配套论文与作者坡度分析脚本保存在同级`evidence/`，来源为[作者仓库](https://github.com/ryodohemmi/public/tree/main/Hemmi%2B2025)。重建工作环境时可从该入口恢复证据文件；审计脚本不会隐式联网。`configs/terrain/lupex_sources_v1.json`保存已人工核对的论文来源表转录和误差定义。`evidence_manifest.json`记录本次证据文件校验和。

## 重新生成审计与冻结划分

每次使用新的run名称；拒绝覆盖已有run以保留冻结结果：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv_isaaclab/bin/python scripts/audit_lunar_terrain.py \
  --output-layout run --run-name audit_lupex_reproduction
```

可用`--archive`指定原始包、`--provenance`指定版本化来源表、`--run-dir`指定独立运行目录。基准变化需要新版本，不能覆盖已有`benchmark.json`后继续沿用旧实验解释。

产物结构：

```text
outputs/runs/lunar_terrain_validation/<run_id>/
  config/lupex_sources_v1.json
  data/product_manifest.json       # ZIP及全部成员校验
  data/evidence_manifest.json      # 论文/作者脚本证据
  data/benchmark.json             # 地理/来源关联、12个嵌套划分、冻结hash
  data/windows.json               # 本次72个完整区域引用，不是72个独立样本
  data/layers.json                 # 真实产品/任务表示/空增强层
  data/scenes/{cr1,...,mp2}.json   # 原生场景、来源及坐标
  data/*_quality_flags.tif        # 缺测/邻边/敏感性尾部/突变候选
  metrics/*_audit.json
  metrics/summary.json
  metrics/integration_smoke.json
  figures/*_atlas.png
  figures/*_ortho.png
  figures/slope_baselines.png
  figures/region_relationships.png
  run_manifest.json
```

质量位标记为1=DTM缺测、2=缺测邻边、4=本产品敏感性p95以上、8=相邻高差p99.9以上候选。它们不是安全分类；没有据此排除地形或补洞。作者已经指出的伪影与自动突变候选不等价。

## 场景读取与物理坐标查询

```python
from pathlib import Path
from lunar_rover_tasks.terrain_data.runtime import load_scene

scene = Path("outputs/runs/lunar_terrain_validation/"
             "audit_lupex_v1_20260922_final/data/scenes/cr1.json")
with load_scene(scene) as terrain:
    # 局部投影米坐标；h=8意味着17×17原生样本、16米中心间基线。
    sample = terrain.query([0.0, 0.0], half_width=8)
    print(sample)  # 查询位置可能缺测；必须检查height_valid/metrics_valid/error_valid。
    lonlat = terrain.local_to_geographic([0.0, 0.0])
    patch = terrain.native_patch((100, 100, 32, 32))
    # patch用于接口示例，不表示32米任务场地已经获选。
```

双线性高度需要显式`interpolation="bilinear"`；它不降低声明误差或增加原生分辨率。指标始终在显式原生支持窗口上计算。查询返回NaN不能替换为平地高度或安全值。

点运动诊断使用`TerrainProbeRuntime`，必须显式传入基线与局部访问半径；轨道先验默认关闭。它是CPU坐标/接口测试，不是PhysX或真实月球车控制。不要把`lupex_dtm`塞入旧`TerrainCfg`尝试复用五通道特征，旧入口会明确拒绝。

## 验证

```bash
.venv_isaaclab/bin/python -m pytest tests/test_lunar_terrain_data.py tests/test_terrain_features.py -o addopts='' -q
.venv_isaaclab/bin/python -m pytest -q -ra
```

单元测试只生成解析小栅格，不需要下载真实数据；CI安装`requirements-terrain.txt`保证执行这些合同。实际六区接入结果另见`integration_smoke.json`。本机全仓回归存在旧exp156配对配置未恢复导致的6项失败，不能把它们写成已通过，也不要临时伪造旧输出修饰测试结果。
