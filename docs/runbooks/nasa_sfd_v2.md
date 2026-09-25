# NASA SFD 地图生成、可视化与无学习验证

路线入口：[真实DEM与NASA模型增强](../architecture/lunar_map_mainline.md)（2026-09-25用户确认为地图后续主线）。本页职责为复现操作。


使用独立 `.venv_terrain_cuda`；额外依赖位于 `requirements-terrain.txt`，
OpenSimplex 0.4.5.1，Numba 0.65.1。Numba 加速背景概率场，不改变公式。
旧地图及旧生成方法保留；新方法配置为 `configs/terrain/nasa_sfd_v2.yaml`。

## 生成地图

下列目录名是新运行示例，已存在目录不得覆盖：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 \
.venv_terrain_cuda/bin/python scripts/build_nasa_sfd_suite.py \
  --run-dir outputs/runs/lunar_terrain_validation/nasa_sfd_v2_reproduction
```

默认源是先前 NPB 基础场景，入口从其清单取得实际源产品、坐标窗口和质量属性。
`--reference-scene` 可显式指定来源。当前验收参数对应 NPB，不能未经分析将
所选直径范围和 0.1 米网格推广为其他数据源的观测尺度。

可使用 `--resume` 核验并复用同配置下已完成的不可变包。若某个包生成中断而没有
完整清单，应保留该失败目录并换新 run，不默默覆盖。若已有正式验收记录，复核用
独立 run，避免把原生成计时替换为读取计时。

## 可视化

```bash
.venv_terrain_cuda/bin/python scripts/visualize_nasa_sfd_suite.py \
  --run-dir outputs/runs/lunar_terrain_validation/nasa_sfd_v2_reproduction
```

为当前三个种子生成全图、差值、大小—频率、空间概率、局部二维/三维和剖面图。
另有种子汇总与单坑径向诊断。原始 JSON/NPY 是事实来源，图片用于人工验收。

## 测试和 GPU 检查

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 \
.venv_terrain_cuda/bin/python -m pytest \
  tests/test_nasa_sfd.py tests/test_lunar_training_maps.py tests/test_lunar_terrain_data.py \
  -q --junitxml=/tmp/nasa_sfd_tests.xml

.venv_terrain_cuda/bin/python scripts/benchmark_lunar_training_maps.py \
  --config configs/terrain/npb_interface_diagnostic.yaml \
  --scene outputs/runs/lunar_terrain_validation/nasa_sfd_v2_reproduction/scenes/real_base/scene.json \
  --output outputs/runs/lunar_terrain_validation/nasa_sfd_v2_reproduction/metrics/gpu_real_base.json \
  --mode proxy --device cuda --num-envs 32 --steps 32
```

再对 `seed23_craters_rocks`、`seed24_craters_rocks`、`seed25_craters_rocks`
重复 GPU 命令，输出分别命名 `gpu_seed23.json`、`gpu_seed24.json`、`gpu_seed25.json`。
每次使用独立进程，保证显存峰值按地图测量。CUDA 不可用时不回退 CPU 并宣称通过。

```bash
.venv_terrain_cuda/bin/python scripts/validate_nasa_sfd_suite.py \
  --run-dir outputs/runs/lunar_terrain_validation/nasa_sfd_v2_reproduction \
  --junit /tmp/nasa_sfd_tests.xml
```

最后一步核验源哈希、测试报告、地图身份、四组 GPU 报告、实际地图的 CPU/GPU
查询一致性、三组空间概率抽样与图片哈希；保存本 run 的
`metrics/engineering_acceptance.json` 并更新 `_suite/metrics/nasa_sfd_v2_*`。
不会覆盖旧的 LUPEX 审计总结果。`passed` 只表示工程接入，不表示车辆安全、任务
成功或跨区域泛化。所有入口均不启动学习更新。

当前实测结果及局限见 [验收记录](../experiments/nasa_sfd_v2_2026-09-24.md)。
