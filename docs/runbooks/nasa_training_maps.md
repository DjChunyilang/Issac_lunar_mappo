# NASA真实地图生成与无学习验证

路线入口：[真实DEM与NASA模型增强](../architecture/lunar_map_mainline.md)（2026-09-25用户确认为地图后续主线）。本页职责为复现操作。


当前交付与限制见[实施记录](../experiments/nasa_terrain_implementation_2026-09-24.md)。学习训练暂停；以下入口没有优化器更新。不要使用旧合成训练入口加载新地图。

## 环境与数据

```bash
bash scripts/install_terrain_cuda.sh
```

脚本保留`.venv_isaaclab`，独立安装`.venv_terrain_cuda`，最后检查CUDA设备。2026-09-24驱动已恢复，后续[NASA SFD验收](../experiments/nasa_sfd_v2_2026-09-24.md)完成四组32 GPU环境×32步检查。若换机或内核升级导致CUDA不可用，不要将CPU回退当作GPU验收。

混合NMPC复用既有acados源码/动态库，安装Python接口到新环境：

```bash
.venv_isaaclab/bin/python -m pip --python .venv_terrain_cuda/bin/python \
  install -e .venv_isaaclab/acados/interfaces/acados_template
export ACADOS_SOURCE_DIR="$PWD/.venv_isaaclab/acados"
export LD_LIBRARY_PATH="$PWD/.venv_isaaclab/acados/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
```

下载器核验已存在产品，不覆盖无对应清单的文件。下载完成后原始TIFF设为只读；源产品不做就地修改。

```bash
.venv_terrain_cuda/bin/python scripts/download_lunar_sources.py \
  --catalog configs/terrain/nasa_lola_sources_v1.json \
  --output outputs/runs/lunar_terrain_validation/_suite/data/nasa_lola_v1
```

## 地图包

输出目录必须不存在。以下命令完整指定200米工程样例，不把它设为所有月面任务的默认尺寸。

```bash
.venv_terrain_cuda/bin/python scripts/build_lunar_training_maps.py \
  --dtm outputs/runs/lunar_terrain_validation/_suite/data/nasa_lola_v1/npb/dtm.tif \
  --height-error outputs/runs/lunar_terrain_validation/_suite/data/nasa_lola_v1/npb/height_error.tif \
  --sample-count outputs/runs/lunar_terrain_validation/_suite/data/nasa_lola_v1/npb/sample_count.tif \
  --source-id pgda78_npb --source-url https://pgda.gsfc.nasa.gov/products/78 \
  --group nasa_npb --role development \
  --window 480 2480 40 40 --spacing 0.2 --metric-baseline 0.8 --source-half-width 2 \
  --error-semantics 'LOLA total height RMS; correlated orbit/interpolation errors' \
  --output outputs/runs/lunar_terrain_validation/npb_rebuild/scenes/real_base
```

生成增强版本时换用新的输出目录，增加`--enhancement-config configs/terrain/nasa_enhancement_v1.yaml`。该配置是显式假设。地图元数据中的原生分辨率、处理格距、源尺度指标及仿真指标不可混用。

已有本轮地图位于`outputs/runs/lunar_terrain_validation/nasa_implementation_20260923/scenes/`。校验失败应追查来源，不修改哈希绕过检查。生产查询遇到未预取场景会报错；活动场景总容量超过缓存预算也会报错，需减少同时常驻场景或分块，不缩放真实物理尺寸。

## 无学习执行检查

CPU参考命令：

```bash
.venv_terrain_cuda/bin/python scripts/benchmark_lunar_training_maps.py \
  --scene outputs/runs/lunar_terrain_validation/nasa_implementation_20260923/scenes/npb_200m_real_base/scene.json \
  --mode proxy --device cpu --num-envs 32 --steps 8 \
  --output outputs/runs/lunar_terrain_validation/npb_recheck/metrics/proxy_cpu.json

.venv_terrain_cuda/bin/python scripts/benchmark_lunar_training_maps.py \
  --scene outputs/runs/lunar_terrain_validation/nasa_implementation_20260923/scenes/npb_200m_real_base/scene.json \
  --mode nmpc --device cpu --num-envs 1 4 8 --steps 4 \
  --output outputs/runs/lunar_terrain_validation/npb_recheck/metrics/nmpc_cpu.json
```

扩展GPU资源评估时，proxy可改成`--device cuda --num-envs 32 128 512`；NMPC改成`--device cuda --num-envs 1 4 8`并使用不同输出文件。当前NASA SFD轮次已验收32环境proxy，未宣称其余预算均已完成。CUDA不可见时入口明确失败，不静默回退。`mode=nmpc`的CPU host/求解器保持不变，GPU负责策略和地形查询；含回传开销。两种mode不作算法效果排名。

配置`configs/terrain/npb_interface_diagnostic.yaml`只用于接口检查，保留旧任务阈值时已有明确的诊断标记。脚本运行固定动作/未训练策略和固定张量反向，校验参数哈希不变，不生成学习更新。计时为同步插桩，嵌套阶段不能相加；长时间资源或成功率结论需另行设计。

## 回归与证据

```bash
PYTHONPATH="$PWD/scripts:$PWD/source/lunar_rover_tasks" \
  .venv_terrain_cuda/bin/python -m pytest -ra
```

真实acados测试需先设置上面的环境变量。新地图测试在`tests/test_lunar_training_maps.py`；无CUDA时数值一致性测试会skip。旧exp156的6个历史配置仍引用未恢复的outputs文件，不能伪造配置来获得全绿。

本轮关键证据：

- `metrics/source_verification.json`：JPL裁剪与母图逐像元核对、全图有效性及LDEC计数。
- `metrics/geographic_split_v2.json`：跨源5组及留组折，NPB仅开发。
- `metrics/hardware_current.json`：该早期run的历史驱动阻塞证据，不代表后续GPU恢复后的状态。
- `metrics/proxy_cpu.json`、`metrics/nmpc_cpu.json`：基础地图完整执行检查。
- `metrics/nmpc_hypothesis_cpu.json`：增强地图执行检查。
- `metrics/regression_final.xml`与`logs/regression_final.log`：最终回归事实。
- `config/cuda_freeze.txt`：独立环境精确版本；`run_manifest.json`：运行范围和产物索引。

路径前缀统一为`outputs/runs/lunar_terrain_validation/nasa_implementation_20260923/`。生成地图、依赖、指标和图集都留在ignored outputs/环境目录，不加入Git。
